"""
Carpet Plot / Constraint Diagram Visualization for Aircraft Sizing Optimization

Extracts optimization history data from OpenMDAO SQL case recorder files
and generates constraint diagrams showing feasible design space regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import openmdao.api as om
from scipy.interpolate import griddata
import pandas as pd
from pathlib import Path


def extract_optimization_data(sql_filename):
    """
    Extract design variables and outputs from OpenMDAO SQL case recorder.
    
    Parameters
    ----------
    sql_filename : str
        Path to the SQLite case recorder file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all design variables and outputs for each iteration
    """
    cr = om.CaseReader(sql_filename)
    driver_cases = cr.get_cases('driver')
    
    # Define what to extract
    design_vars = [
        'S_ref_dv',
        'AR_dv', 
        'wing_toverc_dv',
        'n_parallel_per_str_dv',
        'rated_power_em_scalar',
    ]
    
    constraints = [
        'min_climb_grad_watlim_2nd',
        'min_climb_grad_watlim_4th',
        'min_climb_grad_approach',
        'min_climb_grad_landing',
        'min_climb_grad_aeo',
        'tofl_unlim.TOFL_ft',
        'ceiling_altitude',
        'span_dv',
        'max_hy_range.fuel_margin',
        'max_hy_range_tow_margin.tow_margin',
        'max_hy_range.mission_range',
        'max_hy_range.ks_soc.KS',
        'max_hy_range.ks_voltage.KS',
        'max_e_range.fuel_margin',
        'max_e_range_tow_margin.tow_margin',
        'max_e_range.mission_range',
        'max_e_range.ks_soc.KS',
        'max_e_range.ks_voltage.KS',
        'min_fuel_tow_margin.tow_margin',
        'min_fuel.approach.fuel_used_final',
    ]
    
    recorded_outputs = [
        'max_hy_range.ac|weights|TOW',
        'max_hy_range.ac|weights|OEW',
        'max_hy_range.total_fuel_used',
        'max_e_range.ac|weights|TOW',
        'max_e_range.total_fuel_used',
        'min_fuel.ac|weights|TOW',
        'min_fuel.total_fuel_used',
    ]
    
    # Collect all data
    data = {'iteration': list(range(len(driver_cases)))}
    
    # Initialize columns
    for var in design_vars + constraints + recorded_outputs:
        data[var] = []
    
    for case in driver_cases:
        # Get design variables (unscaled)
        desvars = case.get_design_vars(scaled=False)
        for var in design_vars:
            if var in desvars:
                val = desvars[var]
                data[var].append(val[0] if hasattr(val, '__len__') else val)
            else:
                data[var].append(np.nan)
        
        # Get constraints (unscaled)
        cons = case.get_constraints(scaled=False)
        for var in constraints:
            if var in cons:
                val = cons[var]
                data[var].append(val[0] if hasattr(val, '__len__') else val)
            else:
                data[var].append(np.nan)
        
        # Get recorded outputs
        for var in recorded_outputs:
            try:
                val = case.get_val(var)
                data[var].append(val[0] if hasattr(val, '__len__') else val)
            except KeyError:
                data[var].append(np.nan)
    
    return pd.DataFrame(data)


def create_carpet_plot_pair(df, x_var, y_var, z_var, x_label, y_label, z_label,
                            ax=None, cmap='viridis', levels=15, 
                            scatter_size=50, show_trajectory=True):
    """
    Create a single carpet plot panel showing z as function of x and y.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with optimization data
    x_var, y_var, z_var : str
        Column names for x, y, z variables
    x_label, y_label, z_label : str
        Axis labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    cmap : str
        Colormap name
    levels : int
        Number of contour levels
    scatter_size : float
        Size of scatter points
    show_trajectory : bool
        If True, connect points in iteration order
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get data
    x = df[x_var].values
    y = df[y_var].values
    z = df[z_var].values
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    
    if len(x) < 4:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate z values onto grid
    try:
        zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')
        
        # Fill NaN with linear interpolation fallback
        if np.any(np.isnan(zi_grid)):
            zi_linear = griddata((x, y), z, (xi_grid, yi_grid), method='linear')
            zi_grid = np.where(np.isnan(zi_grid), zi_linear, zi_grid)
        
        # Plot filled contours
        cf = ax.contourf(xi_grid, yi_grid, zi_grid, levels=levels, cmap=cmap, alpha=0.8)
        plt.colorbar(cf, ax=ax, label=z_label, shrink=0.8)
        
        # Add contour lines
        cs = ax.contour(xi_grid, yi_grid, zi_grid, levels=levels//2, colors='k', 
                        linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    except Exception as e:
        # Fallback to scatter only
        print(f"Warning: Could not create contours for {z_var}: {e}")
    
    # Scatter points colored by iteration
    iterations = df['iteration'].values[mask]
    norm = Normalize(vmin=iterations.min(), vmax=iterations.max())
    
    scatter = ax.scatter(x, y, c=iterations, cmap='coolwarm', s=scatter_size, 
                        edgecolors='white', linewidths=0.5, zorder=10,
                        norm=norm)
    
    # Show trajectory
    if show_trajectory and len(x) > 1:
        ax.plot(x, y, 'k-', alpha=0.3, linewidth=1, zorder=5)
        # Mark start and end
        ax.scatter(x[0], y[0], marker='o', s=100, c='green', edgecolors='white', 
                  linewidths=2, zorder=15, label='Start')
        ax.scatter(x[-1], y[-1], marker='*', s=200, c='red', edgecolors='white', 
                  linewidths=2, zorder=15, label='End')
    
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def generate_carpet_plots(sql_filename, output_dir=None, save_plots=True):
    """
    Generate carpet plots showing design variable effects on critical outputs.
    
    Parameters
    ----------
    sql_filename : str
        Path to SQL case recorder file
    output_dir : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save plots to files
    """
    print(f"\n{'='*70}")
    print("CARPET PLOT GENERATION")
    print(f"{'='*70}")
    print(f"Reading: {sql_filename}")
    
    # Extract data
    df = extract_optimization_data(sql_filename)
    print(f"Extracted {len(df)} iterations")
    print(f"\nColumns with data (showing final iteration value):")
    for col in df.columns:
        if not df[col].isna().all():
            n_valid = df[col].notna().sum()
            try:
                print(f"  - {col}: {df[col].iloc[-1]:.3f} ({n_valid}/{len(df)} valid)")
            except:
                print(f"  - {col}: ({n_valid}/{len(df)} valid)")
    
    # Check critical outputs specifically
    print(f"\n--- Critical outputs check ---")
    for var in ['max_hy_range.mission_range', 'max_e_range.mission_range', 
                'min_fuel.approach.fuel_used_final', 'max_hy_range.ac|weights|OEW']:
        if var in df.columns:
            vals = df[var].dropna()
            if len(vals) > 0:
                print(f"  {var}: min={vals.min():.1f}, max={vals.max():.1f}, final={vals.iloc[-1]:.1f}")
            else:
                print(f"  {var}: NO DATA")
        else:
            print(f"  {var}: NOT IN DATAFRAME")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(sql_filename).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Design variable mappings (column name -> display label)
    dv_mapping = {
        'n_parallel_per_str_dv': ('Battery nP/str', 'cells'),
        'AR_dv': ('Aspect Ratio', ''),
        'S_ref_dv': ('Wing Area', 'm²'),
        'rated_power_em_scalar': ('EMotor Power', 'kW'),
        'wing_toverc_dv': ('Wing t/c', ''),
    }
    
    # Output variable mappings
    output_mapping = {
        'max_hy_range.mission_range': ('Hybrid Range', 'NM'),
        'max_e_range.mission_range': ('Electric Range', 'NM'),
        'min_fuel.approach.fuel_used_final': ('Block Fuel', 'lbm'),
        'max_hy_range.ac|weights|OEW': ('OEW', 'lbm'),
        'min_climb_grad_watlim_2nd': ('WATLIM 2nd Grad', '%'),
        'min_climb_grad_watlim_4th': ('WATLIM 4th Grad', '%'),
        'min_climb_grad_approach': ('Approach Grad', '%'),
        'min_climb_grad_landing': ('Landing Grad', '%'),
        'min_climb_grad_aeo': ('AEO Grad', '%'),
        'ceiling_altitude': ('Ceiling', 'ft'),
        'tofl_unlim.TOFL_ft': ('TOFL', 'ft'),
    }
    
    # Primary design variable pairs for carpet plots
    dv_pairs = [
        ('n_parallel_per_str_dv', 'AR_dv'),
        ('S_ref_dv', 'AR_dv'),
        ('rated_power_em_scalar', 'S_ref_dv'),
        ('n_parallel_per_str_dv', 'rated_power_em_scalar'),
    ]
    
    # Critical outputs to visualize
    critical_outputs = [
        'max_hy_range.mission_range',
        'max_e_range.mission_range',
        'min_fuel.approach.fuel_used_final',
        'max_hy_range.ac|weights|OEW',
        'min_climb_grad_watlim_2nd',
        'min_climb_grad_watlim_4th',
        'ceiling_altitude',
        'tofl_unlim.TOFL_ft',
    ]
    """
    
    # Generate carpet plots for each DV pair and output combination
    # Split outputs into chunks of max 4 (2 rows x 2 cols) per figure
    max_rows = 2
    n_cols = 2
    max_per_figure = max_rows * n_cols
    
    for x_var, y_var in dv_pairs:
        x_label, x_unit = dv_mapping[x_var]
        y_label, y_unit = dv_mapping[y_var]
        
        # Filter to available outputs
        available_outputs = [z for z in critical_outputs 
                           if z in df.columns and not df[z].isna().all()]
        
        # Split into chunks for multiple figures
        n_figures = (len(available_outputs) + max_per_figure - 1) // max_per_figure
        
        for fig_idx in range(n_figures):
            start_idx = fig_idx * max_per_figure
            end_idx = min(start_idx + max_per_figure, len(available_outputs))
            outputs_chunk = available_outputs[start_idx:end_idx]
            
            # Determine rows needed for this figure
            n_outputs_chunk = len(outputs_chunk)
            n_rows = (n_outputs_chunk + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, z_var in enumerate(outputs_chunk):
                z_label, z_unit = output_mapping.get(z_var, (z_var, ''))
                z_label_full = f"{z_label} [{z_unit}]" if z_unit else z_label
                
                # Choose colormap: green=good for Range/Ceiling/Grad, green=good for low Fuel/TOFL
                higher_is_better = 'Range' in z_label or 'Ceiling' in z_label or 'Grad' in z_label
                cmap_choice = 'RdYlGn' if higher_is_better else 'RdYlGn_r'
                
                create_carpet_plot_pair(
                    df, x_var, y_var, z_var,
                    f"{x_label} [{x_unit}]" if x_unit else x_label,
                    f"{y_label} [{y_unit}]" if y_unit else y_label,
                    z_label_full,
                    ax=axes[i],
                    cmap=cmap_choice,
                    show_trajectory=True
                )
                axes[i].set_title(z_label, fontsize=11, fontweight='bold')
            
            # Hide unused subplots
            for j in range(len(outputs_chunk), len(axes)):
                axes[j].set_visible(False)
            
            fig.suptitle(f'Carpet Plots: {x_label} vs {y_label}\n(Figure {fig_idx+1} of {n_figures})', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            if save_plots:
                filename = f"carpet_{x_var}_vs_{y_var}_part{fig_idx+1}.png"
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"Saved: {filepath}")
            
            plt.show()
    """
    
    # Create summary matrix plot showing correlation between DVs and outputs
    create_correlation_matrix(df, dv_mapping, output_mapping, output_dir, save_plots)
    
    # Note: Sensitivity matrix requires HTML scaling report, not SQL file
    # This would need to be called separately with HTML filename
    
    return df


def create_sensitivity_matrix(html_filename, output_dir=None, save_plots=True):
    """
    Extract Jacobian matrix from OpenMDAO scaling report HTML and create sensitivity heatmap.
    
    This shows ACTUAL SENSITIVITIES (partial derivatives) = TRUE CAUSAL RELATIONSHIPS.
    
    Sensitivities = ∂output/∂design_var = Direct physical cause-and-effect
    
    Unlike statistical correlations (which can be misleading), these are actual 
    partial derivatives computed from the physics/math of your model, showing 
    true cause-and-effect relationships with correct signs (positive/negative).
    
    Parameters
    ----------
    html_filename : str
        Path to HTML scaling report file
    output_dir : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save plots
    """
    import json
    import re
    
    # Read HTML file
    with open(html_filename, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract the JavaScript data object using regex
    # Look for: var data = {...};
    pattern = r'var\s+data\s*=\s*({.*?});'
    match = re.search(pattern, html_content, re.DOTALL)
    
    if not match:
        print(f"Could not find data object in HTML file: {html_filename}")
        return None
    
    # Parse the JavaScript object - it's actually valid JSON (OpenMDAO generates it that way)
    data_str = match.group(1)
    
    try:
        # Try direct JSON parsing first
        data = json.loads(data_str)
    except json.JSONDecodeError:
        # If that fails, try using ast.literal_eval (safer than eval)
        import ast
        try:
            # Replace JavaScript true/false/null with Python equivalents
            data_str = data_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            data = ast.literal_eval(data_str)
            # Convert back None to null-like structure if needed
        except Exception as e:
            print(f"Could not parse data object: {e}")
            print("Trying eval as last resort (trusted source)...")
            try:
                data = eval(data_str)
            except Exception as e2:
                print(f"All parsing methods failed: {e2}")
                return None
    
    # Extract Jacobian data
    if 'var_mat_list' not in data or 'oflabels' not in data or 'wrtlabels' not in data:
        print("Missing required data in HTML file (var_mat_list, oflabels, or wrtlabels)")
        return None
    
    oflabels = data['oflabels']  # Output names
    wrtlabels = data['wrtlabels']  # Design variable names
    var_mat_list = data['var_mat_list']  # List of [output, dv, sensitivity] tuples
    
    # Design variable mappings (for display)
    dv_mapping = {
        'S_ref_dv': 'Wing Area',
        'AR_dv': 'Aspect Ratio',
        'wing_toverc_dv': 'Wing t/c',
        'n_parallel_per_str_dv': 'Battery nP/str',
        'rated_power_em_scalar': 'EMotor Power',
    }
    
    # Output mappings (for display)
    output_mapping = {
        'max_hy_range.mission_range': 'Hybrid Range',
        'max_e_range.mission_range': 'Electric Range',
        'min_fuel.approach.fuel_used_final': 'Block Fuel',
        'min_climb_grad_watlim_2nd': 'WATLIM 2nd',
        'min_climb_grad_watlim_4th': 'WATLIM 4th',
        'min_climb_grad_approach': 'Approach Grad',
        'min_climb_grad_landing': 'Landing Grad',
        'min_climb_grad_aeo': 'AEO Grad',
        'ceiling_altitude': 'Ceiling',
        'tofl_unlim.TOFL_ft': 'TOFL',
    }
    
    # Filter to only primary design variables and key outputs
    primary_dvs = ['S_ref_dv', 'AR_dv', 'wing_toverc_dv', 'n_parallel_per_str_dv', 'rated_power_em_scalar']
    key_outputs = list(output_mapping.keys())
    
    # Build sensitivity matrix
    dv_indices = {dv: i for i, dv in enumerate(wrtlabels) if dv in primary_dvs}
    output_indices = {out: i for i, out in enumerate(oflabels) if out in key_outputs}
    
    if not dv_indices or not output_indices:
        print("Could not find matching design variables or outputs")
        return None
    
    sensitivities = np.zeros((len(dv_indices), len(output_indices)))
    
    # Map from var_mat_list to matrix
    for entry in var_mat_list:
        if len(entry) >= 3:
            output_name, dv_name, sens_value = entry[0], entry[1], entry[2]
            if output_name in output_indices and dv_name in dv_indices:
                i = list(dv_indices.keys()).index(dv_name)
                j = list(output_indices.keys()).index(output_name)
                sensitivities[i, j] = sens_value
    
    # Get ordered lists
    dv_list = list(dv_indices.keys())
    output_list = list(output_indices.keys())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use symmetric colormap centered at zero
    vmax = np.max(np.abs(sensitivities[sensitivities != 0])) if np.any(sensitivities != 0) else 1.0
    vmin = -vmax
    
    im = ax.imshow(sensitivities, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Sensitivity (∂Output/∂DV)', shrink=0.8)
    
    # Labels
    dv_labels = [dv_mapping.get(dv, dv) for dv in dv_list]
    out_labels = [output_mapping.get(out, out) for out in output_list]
    
    ax.set_xticks(range(len(output_list)))
    ax.set_xticklabels(out_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(dv_list)))
    ax.set_yticklabels(dv_labels, fontsize=10)
    
    # Add sensitivity values as text
    for i in range(len(dv_list)):
        for j in range(len(output_list)):
            val = sensitivities[i, j]
            if not np.isnan(val) and abs(val) > 1e-10:
                # Format based on magnitude
                if abs(val) >= 1:
                    fmt_str = f'{val:.2f}'
                elif abs(val) >= 0.01:
                    fmt_str = f'{val:.3f}'
                else:
                    fmt_str = f'{val:.2e}'
                
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, fmt_str, ha='center', va='center', 
                       fontsize=8, color=color, fontweight='bold')
    
    ax.set_title('SENSITIVITIES (Jacobian) = True Causal Relationships\n∂Output/∂DesignVariable (Partial Derivatives, NOT Correlations)', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Outputs', fontsize=11)
    ax.set_ylabel('Design Variables', fontsize=11)
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        filepath = Path(output_dir) / "sensitivity_matrix.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    return sensitivities, dv_list, output_list


def compute_sensitivity_matrix_direct(prob, design_vars, outputs, 
                                      dv_labels=None, output_labels=None, 
                                      title=None, save_path=None):
    """
    Compute and visualize ACTUAL SENSITIVITIES (Jacobian) directly from an OpenMDAO problem.
    
    IMPORTANT: This computes TRUE CAUSAL RELATIONSHIPS (partial derivatives), 
    NOT correlations which can be misleading!
    
    Sensitivities = ∂output/∂design_var = How output ACTUALLY changes when input changes
    This shows direct cause-and-effect, including sign (positive/negative).
    
    This is fundamentally different from correlation, which can show misleading 
    relationships due to indirect effects, feedback loops, or confounding variables.
    
    NOTE: This does NOT require a converged optimization! It only needs the problem
    to be evaluated once (prob.run_model()). Sensitivities are computed at whatever
    point the design variables are currently set to.
    
    Parameters
    ----------
    prob : om.Problem
        OpenMDAO problem (must be set up and run at least once with prob.run_model())
    design_vars : list of str
        Names of design variables (inputs) to include
    outputs : list of str
        Names of outputs to include
    dv_labels : list of str, optional
        Display labels for design variables (defaults to design_vars)
    output_labels : list of str, optional
        Display labels for outputs (defaults to outputs)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    np.ndarray
        Sensitivity matrix (design_vars × outputs)
        
    Examples
    --------
    # Compute sensitivities at initial design point
    prob.setup()
    prob.run_model()
    sens_initial = compute_sensitivity_matrix_direct(prob, dvs, outputs)
    
    # Compute sensitivities at a different point
    prob.set_val('wing_area', 100.0)
    prob.set_val('aspect_ratio', 12.0)
    prob.run_model()
    sens_at_new_point = compute_sensitivity_matrix_direct(prob, dvs, outputs)
    
    # Compute sensitivities at optimized solution
    prob.driver = om.ScipyOptimizeDriver()
    prob.run_driver()  # Solves optimization
    sens_optimized = compute_sensitivity_matrix_direct(prob, dvs, outputs)
    """
    # Default labels
    if dv_labels is None:
        dv_labels = design_vars
    if output_labels is None:
        output_labels = outputs
    
    # Compute total derivatives (Jacobian) at the current design point
    # NOTE: This computes ∂output/∂input at wherever the design variables are currently set
    # You can call prob.set_val() to change DVs, then prob.run_model(), then compute_totals()
    # to get sensitivities at different points in the design space
    print("\nComputing sensitivities (total derivatives) at current design point:")
    print("Current design variable values:")
    for dv in design_vars:
        try:
            val = prob.get_val(dv)
            if hasattr(val, '__len__') and len(val) == 1:
                val = val[0]
            print(f"  {dv} = {val}")
        except:
            print(f"  {dv} = (unavailable)")
    
    sensitivities_dict = prob.compute_totals(of=outputs, wrt=design_vars, return_format='dict')
    
    # Build sensitivity matrix
    n_dvs = len(design_vars)
    n_outputs = len(outputs)
    sensitivities = np.zeros((n_dvs, n_outputs))
    
    for i, dv in enumerate(design_vars):
        for j, output in enumerate(outputs):
            if output in sensitivities_dict and dv in sensitivities_dict[output]:
                # Extract the scalar value (handle array outputs)
                val = sensitivities_dict[output][dv]
                if hasattr(val, '__len__'):
                    val = val.flatten()[0]
                sensitivities[i, j] = val
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(max(10, n_outputs * 0.8), max(6, n_dvs * 0.6)))
    
    # Use symmetric colormap centered at zero for signed sensitivities
    nonzero_sens = sensitivities[np.abs(sensitivities) > 1e-10]
    if len(nonzero_sens) > 0:
        vmax = np.max(np.abs(nonzero_sens))
    else:
        vmax = 1.0
    vmin = -vmax
    
    # Red-Blue diverging colormap: Red = positive, Blue = negative
    im = ax.imshow(sensitivities, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Sensitivity (∂Output/∂DV)', shrink=0.8)
    
    # Set tick labels
    ax.set_xticks(range(n_outputs))
    ax.set_xticklabels(output_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_dvs))
    ax.set_yticklabels(dv_labels, fontsize=10)
    
    # Add sensitivity values as text overlays
    for i in range(n_dvs):
        for j in range(n_outputs):
            val = sensitivities[i, j]
            if abs(val) > 1e-10:
                # Format based on magnitude
                if abs(val) >= 100:
                    fmt_str = f'{val:.1f}'
                elif abs(val) >= 1:
                    fmt_str = f'{val:.2f}'
                elif abs(val) >= 0.01:
                    fmt_str = f'{val:.3f}'
                else:
                    fmt_str = f'{val:.1e}'
                
                # Choose text color for visibility
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, fmt_str, ha='center', va='center', 
                       fontsize=8, color=color, fontweight='bold')
    
    # Title and labels
    if title is None:
        title = 'Design Variable Sensitivities (Jacobian)\n∂Output/∂DesignVariable'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Outputs', fontsize=11)
    ax.set_ylabel('Design Variables', fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(n_outputs) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_dvs) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nSensitivity Matrix Summary:")
    print(f"  Shape: {n_dvs} design vars × {n_outputs} outputs")
    print(f"  Range: [{sensitivities.min():.3e}, {sensitivities.max():.3e}]")
    print(f"  Non-zero entries: {np.sum(np.abs(sensitivities) > 1e-10)} / {n_dvs * n_outputs}")
    print(f"  Positive entries: {np.sum(sensitivities > 1e-10)}")
    print(f"  Negative entries: {np.sum(sensitivities < -1e-10)}")
    
    return sensitivities


def create_correlation_matrix(df, dv_mapping, output_mapping, output_dir, save_plots=True):
    """
    Create correlation matrix heatmap between design variables and outputs.
    
    ⚠️ WARNING: CORRELATION ≠ CAUSATION! ⚠️
    
    This shows statistical correlations from optimization history data.
    Correlations can be MISLEADING because they include:
    - Indirect effects through coupled systems
    - Feedback loops
    - Confounding variables
    - Optimizer-induced patterns
    
    For TRUE CAUSAL RELATIONSHIPS, use compute_sensitivity_matrix_direct() 
    or create_sensitivity_matrix() which compute actual partial derivatives (Jacobian).
    
    Use correlations only for:
    - Exploratory data analysis
    - Understanding optimizer behavior patterns
    - Identifying potential relationships to investigate further
    
    DO NOT use correlations to understand direct physical cause-and-effect!
    """
    dv_cols = list(dv_mapping.keys())
    out_cols = [k for k in output_mapping.keys() if k in df.columns and not df[k].isna().all()]
    
    # Calculate correlations
    correlations = np.zeros((len(dv_cols), len(out_cols)))
    for i, dv in enumerate(dv_cols):
        for j, out in enumerate(out_cols):
            mask = ~(df[dv].isna() | df[out].isna())
            if mask.sum() > 2:
                correlations[i, j] = np.corrcoef(df[dv][mask], df[out][mask])[0, 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
    
    # Labels
    dv_labels = [dv_mapping[k][0] for k in dv_cols]
    out_labels = [output_mapping[k][0] for k in out_cols]
    
    ax.set_xticks(range(len(out_cols)))
    ax.set_xticklabels(out_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(dv_cols)))
    ax.set_yticklabels(dv_labels, fontsize=10)
    
    # Add correlation values as text
    for i in range(len(dv_cols)):
        for j in range(len(out_cols)):
            val = correlations[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=9, color=color, fontweight='bold')
    
    ax.set_title('Design Variable vs Output CORRELATIONS\n⚠️ Statistical Correlation ≠ Physical Causation ⚠️\n(Use Sensitivity Matrix for True Cause-Effect)', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Outputs', fontsize=11)
    ax.set_ylabel('Design Variables', fontsize=11)
    
    plt.tight_layout()
    
    if save_plots:
        filepath = output_dir / "correlation_matrix.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    
    plt.show()


def create_parallel_coordinates(df, output_dir=None, save_plots=True):
    """
    Create parallel coordinates plot showing optimization trajectory.
    """
    # Normalize all columns to [0, 1]
    cols_to_plot = [
        'n_parallel_per_str_dv', 'AR_dv', 'S_ref_dv', 'rated_power_em_scalar',
        'max_hy_range.mission_range', 'max_e_range.mission_range',
        'max_hy_range.ac|weights|OEW', 'tofl_unlim.TOFL_ft', 'ceiling_altitude'
    ]
    
    cols_available = [c for c in cols_to_plot if c in df.columns and not df[c].isna().all()]
    
    # Create normalized dataframe
    df_norm = df[cols_available].copy()
    for col in cols_available:
        min_val, max_val = df_norm[col].min(), df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot each iteration as a line
    n_iter = len(df_norm)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=n_iter-1)
    colors = cmap(norm(np.arange(n_iter)))
    
    for i in range(n_iter):
        ax.plot(range(len(cols_available)), df_norm.iloc[i].values, 
               c=colors[i], alpha=0.8, linewidth=1)
    
    # Highlight start and end using colormap colors
    start_color = colors[0]
    end_color = colors[-1]
    ax.plot(range(len(cols_available)), df_norm.iloc[0].values, 
           c=start_color, linewidth=3, label='Start', zorder=10)
    ax.plot(range(len(cols_available)), df_norm.iloc[-1].values, 
           c=end_color, linewidth=3, label='End', zorder=10)
    
    # Short labels
    short_labels = ['nP/str', 'AR', 'S_ref', 'P_em', 'Hy Range', 'E Range', 'OEW', 'TOFL', 'Ceiling']
    short_labels = short_labels[:len(cols_available)]
    
    ax.set_xticks(range(len(cols_available)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Normalized Value (0-1)', fontsize=11)
    ax.set_title('Parallel Coordinates: Optimization Trajectory', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Iteration', fontsize=11, rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        filepath = Path(output_dir) / "parallel_coordinates.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    
    plt.show()


def create_constraint_diagram(df, x_var='AR_dv', y_var='S_ref_dv', 
                              output_dir=None, save_plots=True,
                              constraint_targets=None):
    """
    Create a constraint diagram showing feasible design space.
    
    This generates a classic aircraft sizing trade study visualization where
    constraint boundaries are shown as contour lines and the feasible region
    is highlighted.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with optimization data
    x_var : str
        Column name for x-axis design variable
    y_var : str
        Column name for y-axis design variable  
    output_dir : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save plots
    constraint_targets : dict, optional
        Dictionary of constraint target values
    """
    if constraint_targets is None:
        constraint_targets = {
            'electric_range_nm': 100.0,
            'hybrid_range_nm': 500.0,
            'block_fuel_lbm': 1600.0,
            'tofl_ft': 4200.0,
            'ceiling_ft': 21000.0,
            'span_m': 35.9,  # 118 ft = 35.97 m
        }
    
    # Define constraint mappings
    constraint_config = {
        'max_e_range.mission_range': {
            'label': 'Electric Range',
            'target': constraint_targets['electric_range_nm'],
            'type': 'lower',  # value >= target is feasible
            'color': '#1E90FF',  # Dodger blue
            'unit': 'NM',
        },
        'max_hy_range.mission_range': {
            'label': 'Hybrid Range', 
            'target': constraint_targets['hybrid_range_nm'],
            'type': 'lower',
            'color': '#FFA500',  # Orange
            'unit': 'NM',
        },
        'min_fuel.approach.fuel_used_final': {
            'label': 'Block Fuel',
            'target': constraint_targets['block_fuel_lbm'],
            'type': 'upper',  # value <= target is feasible
            'color': '#228B22',  # Forest green
            'unit': 'lbm',
        },
        'tofl_unlim.TOFL_ft': {
            'label': 'TOFL',
            'target': constraint_targets['tofl_ft'],
            'type': 'upper',  # value <= target is feasible
            'color': '#8B4513',  # Saddle brown
            'unit': 'ft',
        },
        'ceiling_altitude': {
            'label': 'Ceiling',
            'target': constraint_targets['ceiling_ft'],
            'type': 'lower',
            'color': '#8B008B',  # Dark magenta
            'unit': 'ft',
        },
        'span_dv': {
            'label': 'Wingspan',
            'target': constraint_targets['span_m'],
            'type': 'upper',
            'color': '#DC143C',  # Crimson
            'unit': 'm',
        },
    }
    
    # Get data
    x = df[x_var].values
    y = df[y_var].values
    
    # Create grid for interpolation
    x_margin = (x.max() - x.min()) * 0.1
    y_margin = (y.max() - y.min()) * 0.1
    
    # Expand grid beyond data range for better visualization
    x_min, x_max = x.min() - x_margin, x.max() + x_margin
    y_min, y_max = y.min() - y_margin, y.max() + y_margin
    
    # Override with reasonable bounds for AR and S_ref
    if x_var == 'AR_dv':
        x_min, x_max = 10, 15
    if y_var == 'S_ref_dv':
        y_min, y_max = 80, 100
    
    n_grid = 100
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Track feasibility for each constraint
    feasibility_masks = []
    legend_elements = []
    
    for constraint_name, config in constraint_config.items():
        if constraint_name not in df.columns or df[constraint_name].isna().all():
            print(f"Skipping {constraint_name} - not in data")
            continue
        
        z = df[constraint_name].values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        
        if mask.sum() < 4:
            print(f"Skipping {constraint_name} - insufficient data points")
            continue
        
        x_valid, y_valid, z_valid = x[mask], y[mask], z[mask]
        
        # Use linear interpolation for straight constraint lines
        try:
            zi_grid = griddata(
                (x_valid, y_valid), z_valid, 
                (xi_grid, yi_grid), 
                method='linear'
            )
        except Exception as e:
            print(f"Linear interpolation failed for {constraint_name}: {e}")
            continue
        
        target = config['target']
        color = config['color']
        label = config['label']
        
        # Draw constraint boundary at target value
        try:
            cs = ax.contour(xi_grid, yi_grid, zi_grid, levels=[target], 
                           colors=[color], linewidths=2.5)
            
            # Add label to contour
            fmt = {target: f"{label}={target:.0f} {config['unit']}"}
            ax.clabel(cs, inline=True, fontsize=9, fmt=fmt, inline_spacing=10)
            
            # Draw 99.5% constraint line (dashed)
            margin_target = target * 0.995 if config['type'] == 'lower' else target * 1.005
            cs_margin = ax.contour(xi_grid, yi_grid, zi_grid, levels=[margin_target],
                                  colors=[color], linewidths=1.5, linestyles='dashed')
            
        except Exception as e:
            print(f"Could not draw contour for {constraint_name}: {e}")
            continue
        
        # Create feasibility mask for this constraint
        if config['type'] == 'lower':
            feasible = zi_grid >= target
        else:
            feasible = zi_grid <= target
        feasibility_masks.append(feasible)
        
        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, 
                   label=f"{label} {'≥' if config['type']=='lower' else '≤'} {target:.1f} {config['unit']}")
        )
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=1.5, linestyle='dashed',
                   label=f"99.5% of constraint")
        )
    
    # Compute overall feasible region
    if feasibility_masks:
        overall_feasible = np.all(feasibility_masks, axis=0)
        
        # Shade feasible region with hatching
        ax.contourf(xi_grid, yi_grid, overall_feasible.astype(float), 
                   levels=[0.5, 1.5], colors=['#ADD8E6'], alpha=0.3)
        ax.contour(xi_grid, yi_grid, overall_feasible.astype(float),
                  levels=[0.5], colors=['#4682B4'], linewidths=1, linestyles='-')
        
        # Add hatching to feasible region
        ax.contourf(xi_grid, yi_grid, overall_feasible.astype(float),
                   levels=[0.5, 1.5], colors='none', hatches=['//'])
        
        legend_elements.insert(0, Patch(facecolor='#ADD8E6', alpha=0.5, 
                                        hatch='//', edgecolor='#4682B4',
                                        label='Feasible Region'))
    
    # Plot optimization trajectory
    ax.plot(x, y, 'k-', alpha=0.4, linewidth=1.5, zorder=5)
    ax.scatter(x, y, c=df['iteration'], cmap='viridis', s=30, 
              edgecolors='white', linewidths=0.5, zorder=10, alpha=0.7)
    ax.scatter(x[0], y[0], marker='o', s=150, c='green', edgecolors='white',
              linewidths=2, zorder=15, label='Start')
    ax.scatter(x[-1], y[-1], marker='*', s=250, c='red', edgecolors='white',
              linewidths=2, zorder=15, label='Optimum')
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                                  markersize=10, label='Start'))
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                  markersize=12, label='Optimum'))
    
    # Labels and formatting
    x_label = 'Aspect Ratio (AR)' if x_var == 'AR_dv' else x_var
    y_label = 'Wing Reference Area (m²)' if y_var == 'S_ref_dv' else y_var
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title('Wing Sizing Trade Study - Feasible Design Space', 
                fontsize=14, fontweight='bold')
    
    # Invert y-axis to match reference (larger area at bottom)
    ax.invert_yaxis()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Inverted
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             framealpha=0.95, ncol=1)
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        filepath = Path(output_dir) / "constraint_diagram_AR_Sref.png"
        fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    return fig, ax


def create_multi_panel_constraint_diagram(df, output_dir=None, save_plots=True,
                                          constraint_targets=None):
    """
    Create multiple constraint diagrams for different design variable pairs.
    """
    if constraint_targets is None:
        constraint_targets = {
            'electric_range_nm': 90.0,
            'hybrid_range_nm': 475.0,
            'block_fuel_lbm': 1600.0,
            'ceiling_ft': 16000.0,
            'span_m': 35.9,
            'watlim_2nd': 3.0,
            'watlim_4th': 1.7,
        }
    
    # Design variable pairs to plot
    dv_pairs = [
        ('AR_dv', 'S_ref_dv', 'AR vs Wing Area'),
        ('n_parallel_per_str_dv', 'AR_dv', 'Battery nP/str vs AR'),
        ('rated_power_em_scalar', 'S_ref_dv', 'Motor Power vs Wing Area'),
        ('n_parallel_per_str_dv', 'rated_power_em_scalar', 'Battery vs Motor Power'),
    ]
    
    for x_var, y_var, title_suffix in dv_pairs:
        if x_var not in df.columns or y_var not in df.columns:
            print(f"Skipping {x_var} vs {y_var} - variables not in data")
            continue
        
        print(f"\nGenerating constraint diagram: {x_var} vs {y_var}")
        create_constraint_diagram(
            df, x_var=x_var, y_var=y_var,
            output_dir=output_dir, save_plots=save_plots,
            constraint_targets=constraint_targets
        )


if __name__ == "__main__":
    # Example usage
    sql_file = r"C:\Users\AlexanderAmos\Documents\code\aircraft_results\11_01_2026\aircraft_sizing_20260111_170042.sql"
    
    print("Extracting optimization data...")
    df = extract_optimization_data(sql_file)
    
    print("\nGenerating constraint diagram...")
    output_dir = Path(sql_file).parent
    #create_constraint_diagram(df, output_dir=output_dir, save_plots=False)
    
    print("\nGenerating carpet plots...")
    generate_carpet_plots(sql_file, output_dir=output_dir, save_plots=False)

    
    #print("\nGenerating sensitivity matrix from HTML scaling report...")
    #html_file = r"C:\Users\AlexanderAmos\Documents\code\aircraft_results\11_01_2026\sizing_scaling_report_20260111_170042.html"
    #try:
    #    create_sensitivity_matrix(html_file, output_dir=output_dir, save_plots=True)
    #except Exception as e:
    #    print(f"Could not generate sensitivity matrix: {e}")
    #    import traceback
    #    traceback.print_exc()
    
    print("\nGenerating parallel coordinates...")
    #create_parallel_coordinates(df, output_dir=output_dir, save_plots=True)
    
    print("hello")

