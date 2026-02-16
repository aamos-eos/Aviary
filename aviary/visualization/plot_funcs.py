try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError:
    # don't want a matplotlib dependency on Travis/Appveyor
    pass
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import openmdao.api as om
import os
import re

def _extract_oew_lbm_rounded(prob):
    """Return rounded OEW in lbm from ac|weights|OEW, or None if unavailable."""
    for units in ("lbm", None):
        try:
            val = prob.get_val("ac|weights|OEW", units=units)
            return int(round(float(np.ravel(val)[0]), 0))
        except Exception:
            continue
    return None


def _append_oew_to_output_filename(output_filename, oew_lbm):
    """Append OEW tag to output filename base once."""
    if output_filename is None or oew_lbm is None:
        return output_filename
    if re.search(r"_\d+_lbm_oew$", output_filename):
        return output_filename
    return f"{output_filename}_{oew_lbm}_lbm_oew"


def plot_optimization_history(case_recorder_filename, 
                            design_vars=None, design_var_units=None, design_var_labels=None,
                            objectives=None, objective_units=None, objective_labels=None,
                            constraints=None, constraint_units=None, constraint_labels=None,
                            recorded_outputs=None, recorded_output_units=None, recorded_output_labels=None,
                            figsize=(10, 8), use_unscaled=True, save_plots=False, output_dir=None,
                            payload_qty=None, turb_op_type=None, opt_objective=None):
    """
    Plot the evolution of specified design variables, objectives, and constraints during optimization
    
    Parameters
    ----------
    case_recorder_filename : str
        Path to the SQLite case recorder file
    design_vars : list, optional
        List of design variable names to plot
    design_var_units : list, optional
        List of units for design variables (must match design_vars length)
    design_var_labels : list, optional
        List of custom labels for design variables (must match design_vars length)
    objectives : list, optional
        List of objective names to plot
    objective_units : list, optional
        List of units for objectives (must match objectives length)
    objective_labels : list, optional
        List of custom labels for objectives (must match objectives length)
    constraints : list, optional
        List of constraint names to plot
    constraint_units : list, optional
        List of units for constraints (must match constraints length)
    constraint_labels : list, optional
        List of custom labels for constraints (must match constraints length)
    recorded_outputs : list, optional
        List of recorded output variable names to plot (non-design vars accessed via case.get_val)
    recorded_output_units : list, optional
        List of units for recorded outputs (must match recorded_outputs length)
    recorded_output_labels : list, optional
        List of custom labels for recorded outputs (must match recorded_outputs length)
    figsize : tuple, optional
        Figure size (width, height)
    use_unscaled : bool, optional
        If True, plot all variables (design variables, objectives, and constraints) 
        in their unscaled (original) values. If False, plot all variables in their 
        scaled values (default: True)
    save_plots : bool, optional
        If True, save all figures to files (default: False)
    output_dir : str, optional
        Directory to save plots. If None, saves to current directory (default: None)
    payload_qty : float, optional
        Payload quantity for filename generation (default: None)
    turb_op_type : str, optional
        Hybrid configuration type for filename generation (default: None)
    opt_objective : str, optional
        Optimization objective type (e.g., 'fuel' or 'range') for filename generation (default: None)
    """
    cr = om.CaseReader(case_recorder_filename)
    driver_cases = cr.get_cases('driver')
    
    # Initialize lists
    iterations = range(len(driver_cases))
    all_data = {}
    all_labels = {}
    all_units = {}
    
    # Process design variables
    if design_vars:
        if design_var_units is None:
            design_var_units = ['-'] * len(design_vars)
        elif len(design_var_units) != len(design_vars):
            raise ValueError("design_var_units length must match design_vars length")
        
        if design_var_labels is None:
            design_var_labels = [f"Design Var: {var_name}" for var_name in design_vars]
        elif len(design_var_labels) != len(design_vars):
            raise ValueError("design_var_labels length must match design_vars length")
            
        for i, var_name in enumerate(design_vars):
            all_data[var_name] = []
            all_labels[var_name] = design_var_labels[i]
            all_units[var_name] = design_var_units[i]
    
    # Process objectives
    if objectives:
        if objective_units is None:
            objective_units = ['-'] * len(objectives)
        elif len(objective_units) != len(objectives):
            raise ValueError("objective_units length must match objectives length")
        
        if objective_labels is None:
            objective_labels = [f"Objective: {obj_name}" for obj_name in objectives]
        elif len(objective_labels) != len(objectives):
            raise ValueError("objective_labels length must match objectives length")
            
        for i, obj_name in enumerate(objectives):
            all_data[obj_name] = []
            all_labels[obj_name] = objective_labels[i]
            all_units[obj_name] = objective_units[i]
    
    # Process constraints
    if constraints:
        if constraint_units is None:
            constraint_units = ['-'] * len(constraints)
        elif len(constraint_units) != len(constraints):
            raise ValueError("constraint_units length must match constraints length")
        
        if constraint_labels is None:
            constraint_labels = [f"Constraint: {con_name}" for con_name in constraints]
        elif len(constraint_labels) != len(constraints):
            raise ValueError("constraint_labels length must match constraints length")
            
        for i, con_name in enumerate(constraints):
            all_data[con_name] = []
            all_labels[con_name] = constraint_labels[i]
            all_units[con_name] = constraint_units[i]
    
    # Process recorded outputs (non-design variables accessed via get_val)
    if recorded_outputs:
        if recorded_output_units is None:
            recorded_output_units = ['-'] * len(recorded_outputs)
        elif len(recorded_output_units) != len(recorded_outputs):
            raise ValueError("recorded_output_units length must match recorded_outputs length")
        
        if recorded_output_labels is None:
            recorded_output_labels = [f"Output: {out_name}" for out_name in recorded_outputs]
        elif len(recorded_output_labels) != len(recorded_outputs):
            raise ValueError("recorded_output_labels length must match recorded_outputs length")
            
        for i, out_name in enumerate(recorded_outputs):
            all_data[out_name] = []
            all_labels[out_name] = recorded_output_labels[i]
            all_units[out_name] = recorded_output_units[i]
    
    # Extract data from cases
    for case in driver_cases:
        # Get design variables
        if design_vars:
            if use_unscaled:
                # Get unscaled design variables
                desvars = case.get_design_vars(scaled=False)
            else:
                # Get scaled design variables (original behavior)
                desvars = case.get_design_vars(scaled=True)
            
            for var_name in design_vars:
                if var_name in desvars:
                    all_data[var_name].append(desvars[var_name][0])
                else:
                    print(f"Warning: Design variable '{var_name}' not found in case")
        
        # Get objectives
        if objectives:
            
            if use_unscaled:
                # Get unscaled objectives
                objs = case.get_objectives(scaled=True)
            else:
                # Get scaled objectives (original behavior)
                objs = case.get_objectives(scaled=True)
            
            for obj_name in objectives:
                if obj_name in objs:
                    all_data[obj_name].append(objs[obj_name][0])
                else:
                    print(f"Warning: Objective '{obj_name}' not found in case")
        
        # Get constraints
        if constraints:
            for i, con_name in enumerate(constraints):
                try:
                    # Use get_val with units for proper unit conversion
                    unit = constraint_units[i] if constraint_units and i < len(constraint_units) and constraint_units[i] not in ['-', None] else None
                    val = case.get_val(con_name, units=unit)
                    # Try to extract scalar, handle various array shapes
                    if hasattr(val, '__len__'):
                        if len(val) == 1:
                            all_data[con_name].append(float(val[0]))
                        else:
                            # Multi-valued constraint - take max for upper bound constraints
                            all_data[con_name].append(float(val.max()))
                    else:
                        all_data[con_name].append(float(val))
                except KeyError:
                    # Fall back to get_constraints if not found as output
                    try:
                        if use_unscaled:
                            cons = case.get_constraints(scaled=False)
                        else:
                            cons = case.get_constraints(scaled=True)
                        if con_name in cons:
                            val = cons[con_name]
                            if hasattr(val, '__len__'):
                                all_data[con_name].append(float(val[0]) if len(val) == 1 else float(val.max()))
                            else:
                                all_data[con_name].append(float(val))
                        else:
                            print(f"Warning: Constraint '{con_name}' not found in case")
                    except Exception as e:
                        print(f"Warning: Error extracting constraint '{con_name}': {e}")
                except Exception as e:
                    print(f"Warning: Error extracting constraint '{con_name}': {e}")
        
        # Get recorded outputs (non-design variables)
        if recorded_outputs:
            for i, out_name in enumerate(recorded_outputs):
                try:
                    # Use units parameter to convert (e.g., m to NM)
                    unit = recorded_output_units[i] if recorded_output_units[i] not in ['-', None] else None
                    val = case.get_val(out_name, units=unit)
                    if val is not None:
                        all_data[out_name].append(val[0] if hasattr(val, '__len__') else val)
                    else:
                        print(f"Warning: Recorded output '{out_name}' returned None")
                except KeyError:
                    print(f"Warning: Recorded output '{out_name}' not found in case")
    
    # Create subplots
    num_plots = len(all_data)
    if num_plots == 0:
        print("No data to plot")
        return
    
    # Determine if we need multiple figures
    max_plots_per_figure = 4
    num_figures = (num_plots + max_plots_per_figure - 1) // max_plots_per_figure
    
    # Create multiple figures if needed
    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_plots_per_figure
        end_idx = min((fig_idx + 1) * max_plots_per_figure, num_plots)
        current_plots = end_idx - start_idx
        
        fig, axes = plt.subplots(current_plots, 1, figsize=figsize)
        if current_plots == 1:
            axes = [axes]
        
        # Get the variables for this figure
        var_items = list(all_data.items())[start_idx:end_idx]
        
        # Plot each variable in this figure
        for i, (var_name, data) in enumerate(var_items):
            if len(data) == 0:
                continue
                
            ax = axes[i]
            try:
                # Handle shape mismatches by truncating or padding
                import numpy as np
                data_arr = np.array(data).flatten()
                if len(data_arr) != len(iterations):
                    # Take every Nth element if data is longer, or use as-is if shorter
                    if len(data_arr) > len(iterations):
                        step = len(data_arr) // len(iterations)
                        data_arr = data_arr[::step][:len(iterations)]
                    print(f"Warning: Data shape mismatch for '{var_name}', adjusted from {len(data)} to {len(data_arr)}")
                ax.plot(iterations, data_arr, 'o-', linewidth=2, markersize=4)
            except Exception as e:
                print(f"Warning: Could not plot '{var_name}': {e}")
                ax.text(0.5, 0.5, f"Error plotting\n{var_name}", ha='center', va='center', transform=ax.transAxes)
            
            ax.set_ylabel(f'{all_labels[var_name]} [{all_units[var_name]}]')
            ax.grid(True, alpha=0.3)
            ax.set_title(all_labels[var_name])
            
            # Add zero line for constraints
            if 'constraint' in var_name.lower():
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Set x-label on bottom plot
        axes[-1].set_xlabel('Iteration')
        
        # Add figure title if multiple figures
        if num_figures > 1:
            fig.suptitle(f'Optimization History - Figure {fig_idx + 1} of {num_figures}', fontsize=14)
        
        plt.tight_layout()
        
        # Save plots if requested
        if save_plots:
            # Create output directory if it doesn't exist
            if output_dir is None:
                output_dir = '.'
            elif not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Extract timestamp from case recorder filename
            # Look for pattern like mission_analysis_20250929_182254.sql
            timestamp_match = re.search(r'(\d{8}_\d{6})', case_recorder_filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            else:
                # Fallback to current time if no timestamp found in filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build filename with payload and config info if provided
            base_name = 'optimization_history'
            if payload_qty is not None and turb_op_type is not None:
                payload_rounded = int(round(payload_qty, 0))
                base_name = f'{base_name}_{payload_rounded}_lbm_{turb_op_type}'
            elif payload_qty is not None:
                payload_rounded = int(round(payload_qty, 0))
                base_name = f'{base_name}_{payload_rounded}_lbm'
            elif turb_op_type is not None:
                base_name = f'{base_name}_{turb_op_type}'
            
            # Add optimization objective if provided
            if opt_objective is not None:
                base_name = f'{base_name}_{opt_objective}'
            
            if num_figures > 1:
                filename = f'{base_name}_fig_{fig_idx + 1}_of_{num_figures}_{timestamp}.png'
            else:
                filename = f'{base_name}_{timestamp}.png'
            
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        plt.show()


def plot_trajectory(
    prob, x_var, x_unit, y_vars, y_units, phases, x_label=None, y_labels=None, marker="o", plot_title="Trajectory", save_plots=False, output_filename=None
):
    if save_plots and output_filename is None:
        raise ValueError("output_filename must be provided when save_plots is True")

    val_list = []
    for phase in phases:
        val_list.append(prob.get_val(phase + "." + x_var, units=x_unit))
    x_vec = np.concatenate(val_list)

    for i, y_var in enumerate(y_vars):
        val_list = []
        for phase in phases:
            val_list.append(prob.get_val(phase + "." + y_var, units=y_units[i]))
        y_vec = np.concatenate(val_list)
        plt.figure()
        plt.plot(x_vec, y_vec, marker)
        if x_label is None:
            plt.xlabel(x_var)
        else:
            plt.xlabel(x_label)
        if y_labels is not None:
            if y_labels[i] is not None:
                plt.ylabel(y_labels[i])
        else:
            plt.ylabel(y_var)
        plt.title(plot_title)
        plt.grid(True, alpha=0.3)
    plt.show()
    

    if save_plots:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(f"{output_filename}_trajectory.pdf") as pdf:
            for fig in plt.get_fignums():
                pdf.savefig(plt.figure(fig))
            plt.close('all')
 

def extract_mission_data(prob, mission_config, nn=11, 
                         collect_time_data=False, 
                         collect_battery_data=False,
                         collect_gearbox_data=False,
                         collect_motor_elec_data=False,
                         collect_fuel_used_data=False):
    """
    Extract mission data from OpenMDAO problem for visualization.
    
    This function consolidates data extraction logic shared between plot_unsteady 
    and plot_spot_point_analysis to ensure consistency.
    
    Parameters
    ----------
    prob : OpenMDAO Problem object
        The problem containing the data to extract
    mission_config : dict
        Mission configuration dictionary containing phase information
    nn : int, optional
        Default number of nodes if not specified in mission config
    collect_time_data : bool, optional
        If True, create time vector and phase boundaries (default: False)
    collect_battery_data : bool, optional
        If True, collect battery SOC, voltage, and power data (default: False)
    collect_gearbox_data : bool, optional
        If True, collect gearbox power data (default: False)
    collect_motor_elec_data : bool, optional
        If True, collect motor electrical power and efficiency data (default: False)
    collect_fuel_used_data : bool, optional
        If True, collect fuel used data (default: False)
    
    Returns
    -------
    dict
        Dictionary containing all extracted data arrays and metadata:
        - Flight condition data: altitude_data, range_data, eas_data, tas_data, vs_data, gamma_data
        - Aerodynamic data: cl_data, cd_data, cd0_data, cd_lift_data, lift_data, drag_data
        - Weight data: weight_data, fuel_used_data (if requested)
        - Thrust data: unit_thrust_data, total_thrust_data
        - Power data: unit_gt_power_data, total_gt_power_data, gt_throttle_data,
                     unit_em_power_data, total_em_power_data, em_throttle_data,
                     unit_shaft_power_data, total_shaft_power_data, nac_throttle_data
        - Fuel data: unit_fuel_flow_data, total_fuel_flow_data
        - Jet thrust data: unit_jet_thrust_data
        - Propeller data: unit_adv_ratio_data, unit_cp_data, unit_eta_calc_data, unit_eta_data
        - Gearbox data (if requested): unit_gearbox_power_in_data, unit_gearbox_power_out_data
        - Motor electrical data (if requested): unit_em_elec_power_data, total_em_elec_power_data, unit_em_efficiency_data
        - Battery data (if requested): bat_soc_data, bat_voltage_data, bat_vline_cell_raw_data,
                                      bat_p_cell_data, bat_i_bat_data, bat_i_cell_raw_data, total_batt_power_data
        - Time data (if requested): time_vec, phase_boundaries, phase_names_list
        - Metadata: phase_names_data, phases, data_type
    """
    # Get phase information
    if "mission" in mission_config:
        data_type = "mission"
        phases = mission_config["mission"]["phase_names"]
    else:
        data_type = "seg"
        phases = ["seg"]
    
    # Initialize time tracking if requested
    time_vec = []
    current_time = 0.0
    phase_boundaries = []
    phase_names_list = []
    
    # Initialize data storage
    altitude_data = []
    range_data = []
    eas_data = []
    tas_data = []
    vs_data = []
    cl_data = []
    cd_data = []
    cd0_data = []
    cd_lift_data = []
    lift_data = []
    phase_names_data = []  # Store phase names for each time point/node
    # 2D arrays for nacelle-specific data: [num_nacelles, num_nodes]
    unit_thrust_data = []  # Will become 2D: [num_nacelles, num_nodes]
    total_thrust_data = []
    drag_data = []
    weight_data = []
    gamma_data = []
    # 2D arrays for nacelle-specific power data
    unit_gt_power_data = []  # Will become 2D: [num_nacelles, num_nodes]
    gt_throttle_data = []
    unit_jet_thrust_data = []  # Will become 2D: [num_nacelles, num_nodes] - jet thrust per nacelle
    unit_em_power_data = []  # Will become 2D: [num_nacelles, num_nodes]
    em_throttle_data = []
    total_em_power_data = []
    total_gt_power_data = []
    unit_fuel_flow_data = []  # Will become 2D: [num_nacelles, num_nodes]
    total_fuel_flow_data = []   
    total_shaft_power_data = []
    unit_shaft_power_data = []
    nac_throttle_data = []
    fuel_used_data = [] if collect_fuel_used_data else None
    
    # Optional data structures
    unit_gearbox_power_in_data = [] if collect_gearbox_data else None
    unit_gearbox_power_out_data = [] if collect_gearbox_data else None
    
    # Motor electrical power and efficiency data (for spot point analysis)
    unit_em_elec_power_data = [] if collect_motor_elec_data else None
    total_em_elec_power_data = [] if collect_motor_elec_data else None
    unit_em_efficiency_data = [] if collect_motor_elec_data else None
    
    # Propeller performance data - will be collected for each nacelle separately
    unit_adv_ratio_data = []  # Will become 2D: [num_nacelles, num_nodes]
    unit_cp_data = []  # Will become 2D: [num_nacelles, num_nodes] - power coefficient
    unit_eta_calc_data = []  # Will become 2D: [num_nacelles, num_nodes] - calculated efficiency
    unit_eta_data = []  # Will become 2D: [num_nacelles, num_nodes] - bounded efficiency
    unit_prop_thrust_data = []  # Will become 2D: [num_nacelles, num_nodes] - propeller thrust
    unit_prop_rpm_data = []  # Will become 2D: [num_nacelles, num_nodes] - propeller RPM
    
    # Collect data from each phase
    for phase in phases:
        if phase in mission_config:
            nn = mission_config[phase].get('num_nodes', nn)
            num_nacelles = mission_config[phase].get('num_nac', 4)  # Get number of nacelles
            
            # Time tracking (if requested)
            if collect_time_data:
                duration = prob.get_val(f"{phase}.duration", units='min')
                phase_boundaries.append(float(current_time))
                phase_names_list.append(phase)
                phase_time = np.linspace(current_time, current_time + duration, nn)
                time_vec.extend(phase_time)
                current_time += duration
            
            # Add phase name for each time point/node in this phase
            phase_names_data.extend([phase] * nn)
            
            # Altitude (ft)
            try:
                alt = prob.get_val(f"{phase}.fltcond|h", units='ft')
                altitude_data.extend(alt)
            except:
                altitude_data.extend([np.nan] * nn)
            
            # Range (NM) - cumulative
            try:
                range_val = prob.get_val(f"{phase}.range", units='NM')
                range_data.extend(range_val)
            except:
                range_data.extend([np.nan] * nn)
            
            # Airspeed EAS (knots)
            try:
                eas = prob.get_val(f"{phase}.fltcond|Ueas", units='kn')
                eas_data.extend(eas)
            except:
                eas_data.extend([np.nan] * nn)
            
            # Airspeed TAS (knots)
            try:
                tas = prob.get_val(f"{phase}.fltcond|Utrue", units='kn')
                tas_data.extend(tas)
            except:
                tas_data.extend([np.nan] * nn)
            
            # Lift coefficient
            try:
                cl = prob.get_val(f"{phase}.cl")
                cl_data.extend(cl)
            except:
                cl_data.extend([np.nan] * nn)
            
            # Total drag coefficient
            try:
                cd = prob.get_val(f"{phase}.cd_total")
                cd_data.extend(cd)
            except:
                cd_data.extend([np.nan] * nn)
            
            # Zero-lift drag coefficient
            try:
                cd0 = prob.get_val(f"{phase}.cd_0_total")
                cd0_data.extend(cd0)
            except:
                cd0_data.extend([np.nan] * nn)
            
            # Drag from lift coefficient
            try:
                cd_lift = prob.get_val(f"{phase}.cd_lift")
                cd_lift_data.extend(cd_lift)
            except:
                cd_lift_data.extend([np.nan] * nn)
            
            # Lift force (N)
            try:
                lift = prob.get_val(f"{phase}.lift", units='N')
                lift_data.extend(lift)
            except:
                lift_data.extend([np.nan] * nn)
            
            # Thrust (N) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                # Try two different variable names for thrust
                if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"]:
                    unit_thrust_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.add_thrust.thrust_out_calc", units="N")
                    total_thrust_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.total_thrust", units="N")
                else:
                    unit_thrust_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.thrust_out_calc", units="N")
                    total_thrust_phase = np.sum(unit_thrust_phase, axis=0)
            except:
                unit_thrust_phase = np.full((num_nacelles, nn), np.nan)
                total_thrust_phase = np.full(nn, np.nan)
                     
            unit_thrust_data.append(unit_thrust_phase)
            total_thrust_data.extend(total_thrust_phase)
            
            # Propeller performance data - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                # Advance ratio
                adv_ratio = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.adv_ratio", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if adv_ratio.shape[0] == nn:
                    unit_adv_ratio_phase = adv_ratio.T
                else:
                    unit_adv_ratio_phase = adv_ratio
            except:
                unit_adv_ratio_phase = np.full((num_nacelles, nn), np.nan)
            
            # Power coefficient (Cp) - only if nacelle_power_set is true
            if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"]:
                try:
                    cp = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.Cp", units=None)
                    # Transpose if needed to (num_nacelles, num_nodes)
                    if cp.shape[0] == nn:
                        unit_cp_phase = cp.T
                    else:
                        unit_cp_phase = cp
                except:
                    unit_cp_phase = np.full((num_nacelles, nn), np.nan)
            else:
                unit_cp_phase = np.full((num_nacelles, nn), np.nan)
        
            # Efficiency (eta_calc) - calculated efficiency
            try:
                eta_calc = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.eta_calc", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if eta_calc.shape[0] == nn:
                    unit_eta_calc_phase = eta_calc.T
                else:
                    unit_eta_calc_phase = eta_calc
            except:
                unit_eta_calc_phase = np.full((num_nacelles, nn), np.nan)
            
            # Efficiency (eta) - bounded efficiency
            try:
                eta = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.eta", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if eta.shape[0] == nn:
                    unit_eta_phase = eta.T
                else:
                    unit_eta_phase = eta
            except:
                unit_eta_phase = np.full((num_nacelles, nn), np.nan)
            
            # Propeller thrust - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                prop_thrust = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.compute_thrust.thrust_calc", units="N")
                # Transpose if needed to (num_nacelles, num_nodes)
                if prop_thrust.shape[0] == nn:
                    unit_prop_thrust_phase = prop_thrust.T
                else:
                    unit_prop_thrust_phase = prop_thrust
            except:
                unit_prop_thrust_phase = np.full((num_nacelles, nn), np.nan)
            
            # Propeller RPM - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                prop_rpm = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.prop.rpm", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if prop_rpm.shape[0] == nn:
                    unit_prop_rpm_phase = prop_rpm.T
                else:
                    unit_prop_rpm_phase = prop_rpm
            except:
                unit_prop_rpm_phase = np.full((num_nacelles, nn), np.nan)
        
            unit_adv_ratio_data.append(unit_adv_ratio_phase)
            unit_cp_data.append(unit_cp_phase)
            unit_eta_calc_data.append(unit_eta_calc_phase)
            unit_eta_data.append(unit_eta_phase)
            unit_prop_thrust_data.append(unit_prop_thrust_phase)
            unit_prop_rpm_data.append(unit_prop_rpm_phase)
            
            # Drag force (N)
            try:
                drag = prob.get_val(f"{phase}.drag", units='N')
                drag_data.extend(drag)
            except:
                drag_data.extend([np.nan] * nn)
            
            # Weight (kg)
            try:
                weight = prob.get_val(f"{phase}.weight", units='kg')
                weight_data.extend(weight)
            except:
                weight_data.extend([np.nan] * nn)
            
            # Fuel used (kg) - optional
            if collect_fuel_used_data:
                try:
                    fuel_used = prob.get_val(f"{phase}.fuel_used", units='kg')
                    fuel_used_data.extend(fuel_used)
                except:
                    fuel_used_data.extend([np.nan] * nn)
            
            # Vertical speed (ft/min)
            try:
                vs = prob.get_val(f"{phase}.fltcond|vs", units='ft/min')
                vs_data.extend(vs)
            except:
                vs_data.extend([np.nan] * nn)
            
            # Flight path angle (deg)
            try:
                sin_gamma = prob.get_val(f"{phase}.fltcond|singamma", units=None)
                cos_gamma = prob.get_val(f"{phase}.fltcond|cosgamma", units=None)
                gamma = np.arctan2(sin_gamma, cos_gamma) * 100
                gamma_data.extend(gamma)
            except:
                gamma_data.extend([np.nan] * nn)
            
            # Fuel flow (kg/h) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                # total_fuel_flow is a matrix: get and transpose if needed to (num_nacelles, num_nodes)
                total_fuel_flow_phase = prob.get_val(f"{phase}.total_fuel_flow", units='kg/h')
                if total_fuel_flow_phase.shape[0] == nn:  # If (num_nodes, num_nacelles), transpose
                    total_fuel_flow_phase = total_fuel_flow_phase.T
            except:
                total_fuel_flow_phase = np.full((num_nacelles, nn), np.nan)

            # Get per-nacelle fuel flow (already per nacelle, just transpose if needed)
            try:
                unit_fuel_flow = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.turb.fuel_flow", units='kg/h')
                if unit_fuel_flow.shape[0] == nn:  # If (num_nodes, num_nacelles), transpose
                    unit_fuel_flow_phase = unit_fuel_flow.T
                else:
                    unit_fuel_flow_phase = unit_fuel_flow
            except:
                unit_fuel_flow_phase = np.full((num_nacelles, nn), np.nan)

            unit_fuel_flow_data.append(unit_fuel_flow_phase)
            total_fuel_flow_data.append(total_fuel_flow_phase)
            
            # GT power (kW) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                turb_gt_power = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.turb.power", units="kW")
                turb_gt_throttle = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.turb.throttle", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if turb_gt_power.shape[0] == nn:
                    unit_gt_power_phase = turb_gt_power.T
                    gt_throttle_phase = turb_gt_throttle.T
                else:
                    unit_gt_power_phase = turb_gt_power
                    gt_throttle_phase = turb_gt_throttle
            except:
                unit_gt_power_phase = np.full((num_nacelles, nn), np.nan)
                gt_throttle_phase = np.full((num_nacelles, nn), np.nan)
            
            # total_gt_power - try to get it, or calculate from unit data
            try:
                total_gt_power_phase = prob.get_val(f"{phase}.total_gt_power", units="kW")
                if total_gt_power_phase.shape[0] == nn:  # If (num_nodes, num_nacelles), transpose
                    total_gt_power_phase = total_gt_power_phase.T
            except:
                # Calculate from unit data
                total_gt_power_phase = np.sum(unit_gt_power_phase, axis=0)

            unit_gt_power_data.append(unit_gt_power_phase)
            total_gt_power_data.append(total_gt_power_phase)
            gt_throttle_data.append(gt_throttle_phase)

            # Jet thrust (N) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                turb_jet_thrust = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.turb.jet_thrust", units="N")
                # Transpose if needed to (num_nacelles, num_nodes)
                if turb_jet_thrust.shape[0] == nn:
                    unit_jet_thrust_phase = turb_jet_thrust.T
                else:
                    unit_jet_thrust_phase = turb_jet_thrust
            except:
                unit_jet_thrust_phase = np.full((num_nacelles, nn), np.nan)
            
            unit_jet_thrust_data.append(unit_jet_thrust_phase)

            # EM power (kW) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                unit_em_power_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.motor.mech_power", units="kW")
                
                # Try to get throttle - use different paths depending on configuration
                try:
                    em_throttle_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.motor_power_to_throttle.throttle", units=None)
                except:
                    try:
                        motor_em_throttle = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.motor.throttle", units=None)
                        # Transpose if needed to (num_nacelles, num_nodes)
                        if motor_em_throttle.shape[0] == nn:
                            em_throttle_phase = motor_em_throttle.T
                        else:
                            em_throttle_phase = motor_em_throttle
                    except:
                        em_throttle_phase = np.full((num_nacelles, nn), np.nan)
                
                # Calculate total EM power
                total_em_power_phase = unit_em_power_phase.sum(axis=0)
            except:
                unit_em_power_phase = np.full((num_nacelles, nn), np.nan)
                total_em_power_phase = np.full((num_nacelles, nn), np.nan)
                em_throttle_phase = np.full((num_nacelles, nn), np.nan)
            
            unit_em_power_data.append(unit_em_power_phase)
            total_em_power_data.append(total_em_power_phase)
            em_throttle_data.append(em_throttle_phase)

            # Motor electrical power (kW) - optional, for spot point analysis
            if collect_motor_elec_data:
                try:
                    unit_em_elec_power_phase = prob.get_val(f"{phase}.p_train_elec", units="kW")
                    unit_em_efficiency_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.motor.eff", units=None)
                                       
                    total_em_elec_power_phase = unit_em_elec_power_phase.sum(axis=0)  # Sum along nacelles axis
                except:
                    unit_em_elec_power_phase = np.full((num_nacelles, nn), np.nan)
                    total_em_elec_power_phase = np.full(nn, np.nan)
                    unit_em_efficiency_phase = np.full((num_nacelles, nn), np.nan)
                
                unit_em_elec_power_data.append(unit_em_elec_power_phase)
                total_em_elec_power_data.extend(total_em_elec_power_phase)
                unit_em_efficiency_data.append(unit_em_efficiency_phase)

            # Shaft power (kW) - collect per nacelle (shape: num_nacelles, num_nodes)
            try:
                unit_shaft_power_phase = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.total_mech_power_out", units="kW")
            except:
                unit_shaft_power_phase = np.full((num_nacelles, nn), np.nan)
            total_shaft_power_phase = np.sum(unit_shaft_power_phase, axis=0)
            try:
                nac_throttle = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.throttle_nac", units=None)
                # Transpose if needed to (num_nacelles, num_nodes)
                if nac_throttle.shape[0] == nn:
                    nac_throttle_phase = nac_throttle.T
                else:
                    nac_throttle_phase = nac_throttle
            except:
                nac_throttle_phase = np.full((num_nacelles, nn), np.nan)
            total_shaft_power_data.append(total_shaft_power_phase)
            unit_shaft_power_data.append(unit_shaft_power_phase)
            nac_throttle_data.append(nac_throttle_phase)

            # Gearbox power (kW) - optional, for unsteady analysis
            if collect_gearbox_data:
                unit_gearbox_power_in_phase = np.zeros((num_nacelles, nn))  # 2D array: [nacelles, nodes]
                unit_gearbox_power_out_phase = np.zeros((num_nacelles, nn))  # 2D array: [nacelles, nodes]
                
                # Non-independent nacelles (combined)
                try:
                    gearbox_power_in = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.gb_comp.power_in", units="kW")
                    # Transpose if needed to (num_nacelles, num_nodes)
                    if gearbox_power_in.shape[0] == nn:
                        gearbox_power_in_t = gearbox_power_in.T
                    else:
                        gearbox_power_in_t = gearbox_power_in
                    # Broadcast to all nacelles (same value for all nacelles)
                    for nac_idx in range(num_nacelles):
                        unit_gearbox_power_in_phase[nac_idx, :] = gearbox_power_in_t[nac_idx, :] if gearbox_power_in_t.shape[0] > nac_idx else gearbox_power_in_t[0, :]
                except:
                    unit_gearbox_power_in_phase[:, :] = np.nan
                try:
                    gearbox_power_out = prob.get_val(f"{phase}.hy_parallel_ptrain.nacelles.gb_comp.power_out", units="kW")
                    # Transpose if needed to (num_nacelles, num_nodes)
                    if gearbox_power_out.shape[0] == nn:
                        gearbox_power_out_t = gearbox_power_out.T
                    else:
                        gearbox_power_out_t = gearbox_power_out
                    # Broadcast to all nacelles (same value for all nacelles)
                    for nac_idx in range(num_nacelles):
                        unit_gearbox_power_out_phase[nac_idx, :] = gearbox_power_out_t[nac_idx, :] if gearbox_power_out_t.shape[0] > nac_idx else gearbox_power_out_t[0, :]
                except:
                    unit_gearbox_power_out_phase[:, :] = np.nan
            
                unit_gearbox_power_in_data.append(unit_gearbox_power_in_phase)
                unit_gearbox_power_out_data.append(unit_gearbox_power_out_phase)
    
    # Convert to numpy arrays and concatenate phase data
    altitude_data = np.array(altitude_data)
    range_data = np.array(range_data)
    eas_data = np.array(eas_data)
    tas_data = np.array(tas_data)
    vs_data = np.array(vs_data)
    cl_data = np.array(cl_data)
    cd_data = np.array(cd_data)
    cd0_data = np.array(cd0_data)
    cd_lift_data = np.array(cd_lift_data)
    lift_data = np.array(lift_data)
    # Concatenate 2D arrays for nacelle-specific data (shape: num_nacelles, num_nodes)
    # Concatenate along axis 1 (nodes) to stack phases
    unit_thrust_data = np.hstack(unit_thrust_data) if unit_thrust_data else np.array([])
    total_thrust_data = np.array(total_thrust_data)
    drag_data = np.array(drag_data)
    weight_data = np.array(weight_data)
    gamma_data = np.array(gamma_data)
    
    # Concatenate 2D arrays for nacelle-specific power data (shape: num_nacelles, num_nodes)
    unit_gt_power_data = np.hstack(unit_gt_power_data) if unit_gt_power_data else np.array([])
    unit_jet_thrust_data = np.hstack(unit_jet_thrust_data) if unit_jet_thrust_data else np.array([])
    unit_em_power_data = np.hstack(unit_em_power_data) if unit_em_power_data else np.array([])
    unit_fuel_flow_data = np.hstack(unit_fuel_flow_data) if unit_fuel_flow_data else np.array([])
    
    # Handle different concatenation strategies for different data types
    # For unsteady: use hstack, for spot point: use vstack
    # We'll use hstack as default since it's more common
    total_em_power_data = np.hstack(total_em_power_data) if total_em_power_data else np.array([])
    total_gt_power_data = np.hstack(total_gt_power_data) if total_gt_power_data else np.array([])
    total_fuel_flow_data = np.hstack(total_fuel_flow_data) if total_fuel_flow_data else np.array([])
    unit_shaft_power_data = np.hstack(unit_shaft_power_data) if unit_shaft_power_data else np.array([])
    total_shaft_power_data = np.hstack(total_shaft_power_data) if total_shaft_power_data else np.array([])
    nac_throttle_data = np.hstack(nac_throttle_data) if nac_throttle_data else np.array([])
    gt_throttle_data = np.hstack(gt_throttle_data) if gt_throttle_data else np.array([])
    em_throttle_data = np.hstack(em_throttle_data) if em_throttle_data else np.array([])
    
    # Concatenate 2D arrays for nacelle-specific propeller data (shape: num_nacelles, num_nodes)
    unit_adv_ratio_data = np.hstack(unit_adv_ratio_data) if unit_adv_ratio_data else np.array([])
    unit_cp_data = np.hstack(unit_cp_data) if unit_cp_data else np.array([])
    unit_eta_calc_data = np.hstack(unit_eta_calc_data) if unit_eta_calc_data else np.array([])
    unit_eta_data = np.hstack(unit_eta_data) if unit_eta_data else np.array([])
    unit_prop_thrust_data = np.hstack(unit_prop_thrust_data) if unit_prop_thrust_data else np.array([])
    unit_prop_rpm_data = np.hstack(unit_prop_rpm_data) if unit_prop_rpm_data else np.array([])
    
    # Optional data - gearbox
    if collect_gearbox_data:
        unit_gearbox_power_in_data = np.hstack(unit_gearbox_power_in_data) if unit_gearbox_power_in_data else np.array([])
        unit_gearbox_power_out_data = np.hstack(unit_gearbox_power_out_data) if unit_gearbox_power_out_data else np.array([])
    
    # Optional data - motor electrical (for spot point analysis)
    if collect_motor_elec_data:
            # Note: these use hstack for proper time-series concatenation (unsteady analysis)
            # If spot point analysis (single phase), hstack and vstack behave similarly for (nacelle, nodes)
            unit_em_elec_power_data = np.hstack(unit_em_elec_power_data) if unit_em_elec_power_data else np.array([])
            total_em_elec_power_data = np.array(total_em_elec_power_data)
            unit_em_efficiency_data = np.hstack(unit_em_efficiency_data) if unit_em_efficiency_data else np.array([])
    
    # Optional data - fuel used
    if collect_fuel_used_data:
        fuel_used_data = np.array(fuel_used_data)
    
    # Optional data - battery (collected globally, not per-phase)
    bat_soc_data = None
    bat_voltage_data = None
    bat_vline_cell_raw_data = None
    bat_p_cell_data = None
    bat_i_bat_data = None
    bat_i_cell_raw_data = None
    bat_ocv_cell_data = None
    total_batt_power_data = None
    p_aux_elec_data = None
    bat_time_vec = None
    total_nodes_batt = None

    if collect_battery_data:
        try:
            # Collect battery data as vectors (global, not phase-specific)
            bat_soc_data = prob.get_val("batt1.soc", units=None)
            # bat_soc_data is shape (num_str, num_node), so get number of nodes from shape[1]
            if len(bat_soc_data.shape) > 1:
                total_nodes_batt = bat_soc_data.shape[1]  # Number of nodes (columns)
            else:
                total_nodes_batt = len(bat_soc_data)  # Fallback for 1D array
            
            # Collect battery voltage data (global vector)
            bat_voltage_data = prob.get_val("batt1.v_bat", units="V")
            
            # Collect additional battery data for detailed analysis
            bat_vline_cell_raw_data = prob.get_val("batt1.vline_cell", units="V")
            bat_p_cell_data = prob.get_val("batt1.p_cell", units="W")
            bat_i_bat_data = prob.get_val("batt1.i_bat", units="A")
            bat_i_cell_raw_data = prob.get_val("batt1.i_cell", units="A")
            bat_ocv_cell_data = prob.get_val("batt1.ocv_cell", units="V")
            total_batt_power_data = prob.get_val(f"batt1.p_bat", units="kW")
            p_aux_elec_data = prob.get_val(f"batt1.p_aux_elec", units="kW")

            # Create battery-specific time vector if time data is being collected
            if collect_time_data and len(time_vec) > 0:
                bat_time_vec = np.array(time_vec[:total_nodes_batt])
            
            # Convert to numpy arrays
            bat_soc_data = np.array(bat_soc_data)
            bat_voltage_data = np.array(bat_voltage_data)
            bat_vline_cell_raw_data = np.array(bat_vline_cell_raw_data)
            bat_p_cell_data = np.array(bat_p_cell_data)
            bat_i_bat_data = np.array(bat_i_bat_data)
            bat_i_cell_raw_data = np.array(bat_i_cell_raw_data)
            bat_ocv_cell_data = np.array(bat_ocv_cell_data)
            total_batt_power_data = np.array(total_batt_power_data)
            p_aux_elec_data = np.array(p_aux_elec_data)
        except:
            # If battery data collection fails, set to None
            pass
    
    # Convert time data to numpy arrays if collected
    if collect_time_data:
        time_vec = np.array(time_vec)
        phase_boundaries = np.array(phase_boundaries)
    
    # Return all extracted data in a dictionary
    return {
        # Flight condition data
        'altitude_data': altitude_data,
        'range_data': range_data,
        'eas_data': eas_data,
        'tas_data': tas_data,
        'vs_data': vs_data,
        'gamma_data': gamma_data,
        
        # Aerodynamic data
        'cl_data': cl_data,
        'cd_data': cd_data,
        'cd0_data': cd0_data,
        'cd_lift_data': cd_lift_data,
        'lift_data': lift_data,
        'drag_data': drag_data,
        
        # Weight data
        'weight_data': weight_data,
        'fuel_used_data': fuel_used_data,
        
        # Thrust data
        'unit_thrust_data': unit_thrust_data,
        'total_thrust_data': total_thrust_data,
        
        # Power data
        'unit_gt_power_data': unit_gt_power_data,
        'total_gt_power_data': total_gt_power_data,
        'gt_throttle_data': gt_throttle_data,
        'unit_em_power_data': unit_em_power_data,
        'total_em_power_data': total_em_power_data,
        'em_throttle_data': em_throttle_data,
        'unit_shaft_power_data': unit_shaft_power_data,
        'total_shaft_power_data': total_shaft_power_data,
        'nac_throttle_data': nac_throttle_data,
        
        # Fuel data
        'unit_fuel_flow_data': unit_fuel_flow_data,
        'total_fuel_flow_data': total_fuel_flow_data,
        
        # Jet thrust data
        'unit_jet_thrust_data': unit_jet_thrust_data,
        
        # Propeller data
        'unit_adv_ratio_data': unit_adv_ratio_data,
        'unit_cp_data': unit_cp_data,
        'unit_eta_calc_data': unit_eta_calc_data,
        'unit_eta_data': unit_eta_data,
        'unit_prop_thrust_data': unit_prop_thrust_data,
        'unit_prop_rpm_data': unit_prop_rpm_data,
        
        # Gearbox data (optional)
        'unit_gearbox_power_in_data': unit_gearbox_power_in_data,
        'unit_gearbox_power_out_data': unit_gearbox_power_out_data,
        
        # Motor electrical data (optional)
        'unit_em_elec_power_data': unit_em_elec_power_data,
        'total_em_elec_power_data': total_em_elec_power_data,
        'unit_em_efficiency_data': unit_em_efficiency_data,
        
        # Battery data (optional)
        'bat_soc_data': bat_soc_data,
        'bat_voltage_data': bat_voltage_data,
        'bat_vline_cell_raw_data': bat_vline_cell_raw_data,
        'bat_p_cell_data': bat_p_cell_data,
        'bat_i_bat_data': bat_i_bat_data,
        'bat_i_cell_raw_data': bat_i_cell_raw_data,
        'bat_ocv_cell_data': bat_ocv_cell_data,
        'total_batt_power_data': total_batt_power_data,
        'p_aux_elec_data': p_aux_elec_data,
        'bat_time_vec': bat_time_vec,
        'total_nodes_batt': total_nodes_batt,

        # Time data (optional)
        'time_vec': time_vec if collect_time_data else None,
        'phase_boundaries': phase_boundaries if collect_time_data else None,
        'phase_names_list': phase_names_list if collect_time_data else None,
        
        # Metadata
        'phase_names_data': phase_names_data,
        'phases': phases,
        'data_type': data_type,
    }


def plot_spot_point_analysis(prob, mission_config, save_plots=False, output_filename=None, payload_qty=None, turb_op_type=None, opt_objective=None, nn=11):
    """
    Plot kinematic and aerodynamic values for spot point analysis (no time tracking).
    
    Parameters
    ----------
    prob : OpenMDAO Problem object
        The problem containing the data to plot
    mission_config : dict
        Mission configuration dictionary containing phase information
    save_plots : bool, optional
        Whether to save plots to PDF instead of displaying
    output_filename : str, optional
        Base name for output PDF file (required if save_plots is True)
    payload_qty : float, optional
        Payload quantity for filename generation (default: None)
    turb_op_type : str, optional
        Hybrid configuration type for filename generation (default: None)
    opt_objective : str, optional
        Optimization objective type (e.g., 'fuel' or 'range') for filename generation (default: None)
    """
    if save_plots and output_filename is None:
        raise ValueError("output_filename must be provided when save_plots is True")
    
    # Add OEW (ac|weights|OEW) to saved output filenames, rounded to remove decimals
    output_filename_with_oew = _append_oew_to_output_filename(output_filename, _extract_oew_lbm_rounded(prob))

    # Extract mission data using shared function
    data = extract_mission_data(prob, mission_config, nn=nn,
                                collect_time_data=False,  # Spot point doesn't need time tracking
                                collect_battery_data=False,  # Spot point doesn't integrate battery
                                collect_gearbox_data=True,  # Always collect if available
                                collect_motor_elec_data=True,  # Always collect if available
                                collect_fuel_used_data=False)  # Spot point doesn't integrate fuel
    
    # Unpack the data dictionary into local variables
    altitude_data = data['altitude_data']
    range_data = data['range_data']
    eas_data = data['eas_data']
    tas_data = data['tas_data']
    vs_data = data['vs_data']
    gamma_data = data['gamma_data']
    cl_data = data['cl_data']
    cd_data = data['cd_data']
    cd0_data = data['cd0_data']
    cd_lift_data = data['cd_lift_data']
    lift_data = data['lift_data']
    drag_data = data['drag_data']
    weight_data = data['weight_data']
    unit_thrust_data = data['unit_thrust_data']
    total_thrust_data = data['total_thrust_data']
    unit_gt_power_data = data['unit_gt_power_data']
    total_gt_power_data = data['total_gt_power_data']
    gt_throttle_data = data['gt_throttle_data']
    unit_em_power_data = data['unit_em_power_data']
    total_em_power_data = data['total_em_power_data']
    em_throttle_data = data['em_throttle_data']
    unit_shaft_power_data = data['unit_shaft_power_data']
    total_shaft_power_data = data['total_shaft_power_data']
    nac_throttle_data = data['nac_throttle_data']
    unit_fuel_flow_data = data['unit_fuel_flow_data']
    total_fuel_flow_data = data['total_fuel_flow_data']
    unit_jet_thrust_data = data['unit_jet_thrust_data']
    unit_adv_ratio_data = data['unit_adv_ratio_data']
    unit_cp_data = data['unit_cp_data']
    unit_eta_calc_data = data['unit_eta_calc_data']
    unit_eta_data = data['unit_eta_data']
    unit_prop_thrust_data = data['unit_prop_thrust_data']
    unit_prop_rpm_data = data['unit_prop_rpm_data']
    unit_em_elec_power_data = data['unit_em_elec_power_data']
    total_em_elec_power_data = data['total_em_elec_power_data']
    unit_em_efficiency_data = data['unit_em_efficiency_data']
    unit_gearbox_power_in_data = data['unit_gearbox_power_in_data']
    unit_gearbox_power_out_data = data['unit_gearbox_power_out_data']
    phase_names_data = data['phase_names_data']
    phases = data['phases']
    data_type = data['data_type']
    
    # Note: For spot point analysis, some data needs different concatenation strategy
    # Convert vstack arrays to match expected format
    if unit_gt_power_data.size > 0 and len(unit_gt_power_data.shape) > 1:
        # Swap axes if needed (vstack gives different shape than hstack)
        unit_gt_power_data = np.vstack([unit_gt_power_data])  # Keep as is
    if unit_fuel_flow_data.size > 0 and len(unit_fuel_flow_data.shape) > 1:
        unit_fuel_flow_data = np.vstack([unit_fuel_flow_data])  # Keep as is
    if total_fuel_flow_data.size > 0 and len(total_fuel_flow_data.shape) > 1:
        total_fuel_flow_data = np.vstack([total_fuel_flow_data])  # Keep as is
    if unit_em_power_data.size > 0 and len(unit_em_power_data.shape) > 1:
        unit_em_power_data = np.vstack([unit_em_power_data])  # Keep as is
    if total_em_power_data.size > 0 and len(total_em_power_data.shape) > 1:
        total_em_power_data = np.vstack([total_em_power_data])  # Keep as is
    if total_gt_power_data.size > 0 and len(total_gt_power_data.shape) > 1:
        total_gt_power_data = np.vstack([total_gt_power_data])  # Keep as is
    if unit_shaft_power_data.size > 0 and len(unit_shaft_power_data.shape) > 1:
        unit_shaft_power_data = np.vstack([unit_shaft_power_data])  # Keep as is
    if total_shaft_power_data.size > 0 and len(total_shaft_power_data.shape) > 1:
        total_shaft_power_data = np.vstack([total_shaft_power_data])  # Keep as is
    
    # Define colors using tab10 colormap
    import matplotlib.cm as cm
    tab10_colors = cm.tab10.colors
    
    # Define colors for multiple values on same plot
    multi_colors = {
        'eas': tab10_colors[0],           # First tab10 color
        'tas': tab10_colors[1],           # Second tab10 color
        'vs': tab10_colors[0],            # First tab10 color (single line)
        'cl': tab10_colors[0],            # First tab10 color
        'cd': tab10_colors[1],            # Second tab10 color
        'cd0': tab10_colors[0],           # First tab10 color
        'cd_lift': tab10_colors[1],       # Second tab10 color
        'lift': tab10_colors[0],          # First tab10 color
        'thrust': tab10_colors[1],        # Second tab10 color
        'drag': tab10_colors[2],          # Third tab10 color
    }
    
    # Default color for single-value plots
    default_color = tab10_colors[0]  # First tab10 color

    # Create node vector (x-axis for spot point analysis)
    node_vec = np.arange(len(altitude_data))

    # Create main figure with subplots (4x2 grid)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Spot Point Analysis - Flight Conditions vs Nodes', fontsize=16, y=0.98)
    axes = axes.flatten()
    
    # Ensure we only use the first 8 subplots
    axes = axes[:8]
    
    # Plot 1: Altitude vs Nodes
    ax1 = axes[0]
    valid_mask = ~np.isnan(altitude_data)
    if np.any(valid_mask):
        ax1.plot(node_vec[valid_mask], altitude_data[valid_mask], 
                color=default_color, linewidth=2, label='Altitude')
    ax1.set_xlabel('Node')
    ax1.set_ylabel('Altitude (ft)')
    ax1.set_title('Altitude Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Range vs Nodes
    ax2 = axes[1]
    valid_mask = ~np.isnan(range_data)
    if np.any(valid_mask):
        ax2.plot(node_vec[valid_mask], range_data[valid_mask], 
                color=default_color, linewidth=2, label='Range')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Range (NM)')
    ax2.set_title('Range Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Airspeed vs Nodes (EAS and TAS on same plot)
    ax3 = axes[2]
    valid_mask_eas = ~np.isnan(eas_data)
    valid_mask_tas = ~np.isnan(tas_data)
    
    if np.any(valid_mask_eas):
        ax3.plot(node_vec[valid_mask_eas], eas_data[valid_mask_eas], 
                color=multi_colors['eas'], linewidth=2, label='EAS', linestyle='-')
    if np.any(valid_mask_tas):
        ax3.plot(node_vec[valid_mask_tas], tas_data[valid_mask_tas], 
                color=multi_colors['tas'], linewidth=2, label='TAS', linestyle='-')
    
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Airspeed (knots)')
    ax3.set_title('Airspeed Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Vertical Speed vs Nodes
    ax4 = axes[3]
    valid_mask = ~np.isnan(vs_data)
    if np.any(valid_mask):
        ax4.plot(node_vec[valid_mask], vs_data[valid_mask], 
                color=default_color, linewidth=2, label='Vertical Speed')
    ax4.set_xlabel('Node')
    ax4.set_ylabel('Vertical Speed (ft/min)')
    ax4.set_title('Vertical Speed Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 5: Aerodynamic Coefficients vs Nodes (CL and CD on same plot)
    ax5 = axes[4]
    valid_mask_cl = ~np.isnan(cl_data)
    valid_mask_cd = ~np.isnan(cd_data)
    
    if np.any(valid_mask_cl):
        ax5.plot(node_vec[valid_mask_cl], cl_data[valid_mask_cl], 
                color=multi_colors['cl'], linewidth=2, label='CL', linestyle='-')
    if np.any(valid_mask_cd):
        ax5.plot(node_vec[valid_mask_cd], cd_data[valid_mask_cd], 
                color=multi_colors['cd'], linewidth=2, label='CD', linestyle='-')
    
    ax5.set_xlabel('Node')
    ax5.set_ylabel('Aerodynamic Coefficients')
    ax5.set_title('Lift and Total Drag Coefficients')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Drag Components vs Nodes (CD0 and CD_lift on same plot)
    ax6 = axes[5]
    valid_mask_cd0 = ~np.isnan(cd0_data)
    valid_mask_cd_lift = ~np.isnan(cd_lift_data)
    
    if np.any(valid_mask_cd0):
        ax6.plot(node_vec[valid_mask_cd0], cd0_data[valid_mask_cd0], 
                color=multi_colors['cd0'], linewidth=2, label='CD0', linestyle='-')
    if np.any(valid_mask_cd_lift):
        ax6.plot(node_vec[valid_mask_cd_lift], cd_lift_data[valid_mask_cd_lift], 
                color=multi_colors['cd_lift'], linewidth=2, label='CD_induced', linestyle='-')
    
    ax6.set_xlabel('Node')
    ax6.set_ylabel('Drag Coefficients')
    ax6.set_title('Zero-Lift and Lift-Induced Drag')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Forces vs Nodes (Lift, Thrust, Drag on same plot)
    ax7 = axes[6]
    valid_mask_lift = ~np.isnan(lift_data)
    valid_mask_thrust = ~np.isnan(total_thrust_data)
    valid_mask_drag = ~np.isnan(drag_data)
    
    if np.any(valid_mask_thrust):
        ax7.plot(node_vec[valid_mask_thrust], total_thrust_data[valid_mask_thrust]/1000, 
                color=tab10_colors[1], linewidth=2, label='Thrust', linestyle='-')
    if np.any(valid_mask_drag):
        ax7.plot(node_vec[valid_mask_drag], drag_data[valid_mask_drag]/1000, 
                color=tab10_colors[3], linewidth=2, label='Drag', linestyle='-')
    
    ax7.set_xlabel('Node')
    ax7.set_ylabel('Force (kN)')
    ax7.set_title('Aerodynamic and Propulsive Forces')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Weight vs Nodes
    ax8 = axes[7]
    valid_mask = ~np.isnan(weight_data)
    if np.any(valid_mask):
        ax8.plot(node_vec[valid_mask], weight_data[valid_mask], 
                color=default_color, linewidth=2, label='Weight')
    ax8.set_xlabel('Node')
    ax8.set_ylabel('Weight (kg)')
    ax8.set_title('Aircraft Weight')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Adjust layout for main figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.0, w_pad=1.0)
    
    # Show main plots
    plt.show()
    
    # Create detailed analysis figure (2x2 grid)
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Detailed Spot Point Analysis', fontsize=16, y=0.98)
    axes2 = axes2.flatten()
    
    # Ensure we only use the first 4 subplots
    axes2 = axes2[:4]
    
    # Plot 1: Flight Path Angle vs Nodes
    ax1 = axes2[0]
    valid_mask = ~np.isnan(gamma_data)
    if np.any(valid_mask):
        ax1.plot(node_vec[valid_mask], gamma_data[valid_mask], 
                color=default_color, linewidth=2, label='Flight Path Angle')
    ax1.set_xlabel('Node')
    ax1.set_ylabel('Gradient (%)')
    ax1.set_title('Flight Path Angle')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 2: Fuel Flow vs Nodes
    ax2 = axes2[1]
    # total_fuel_flow_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_fuel_flow_data.size > 0 and len(total_fuel_flow_data.shape) > 1:
        total_fuel_flow_sum = total_fuel_flow_data.sum(axis=0)  # Sum along nacelles axis
        valid_mask = ~np.isnan(total_fuel_flow_sum)
        if np.any(valid_mask):
            ax2.plot(node_vec[valid_mask], total_fuel_flow_sum[valid_mask], 
                    color=default_color, linewidth=2, label='Total Fuel Flow')
    else:
        valid_mask = ~np.isnan(total_fuel_flow_data)
        if np.any(valid_mask):
            ax2.plot(node_vec[valid_mask], total_fuel_flow_data[valid_mask], 
                    color=default_color, linewidth=2, label='Fuel Flow')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Total Fuel Flow (kg/h)')
    ax2.set_title('Fuel Consumption Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Power vs Nodes
    ax3 = axes2[2]
    # total_em_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_em_power_data.size > 0 and len(total_em_power_data.shape) > 1:
        total_em_power_sum = total_em_power_data.sum(axis=0)  # Sum along nacelles axis
        valid_mask = ~np.isnan(total_em_power_sum)
        if np.any(valid_mask):
            ax3.plot(node_vec[valid_mask], total_em_power_sum[valid_mask], 
                    color=default_color, linewidth=2, label='Total EM Mech Power')
    else:
        valid_mask = ~np.isnan(total_em_power_data)
        if np.any(valid_mask):
            ax3.plot(node_vec[valid_mask], total_em_power_data[valid_mask], 
                    color=default_color, linewidth=2, label='Total EM Mech Power')
    # Add electrical power if available
    if total_em_elec_power_data.size > 0:
        valid_mask_elec = ~np.isnan(total_em_elec_power_data)
        if np.any(valid_mask_elec):
            ax3.plot(node_vec[valid_mask_elec], total_em_elec_power_data[valid_mask_elec], 
                    color=tab10_colors[1], linewidth=2, label='Total EM Elec Power', linestyle='--')
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Motor Power Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Shaft Power vs Nodes
    ax4 = axes2[3]
    # total_shaft_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_shaft_power_data.size > 0 and len(total_shaft_power_data.shape) > 1:
        total_shaft_power_sum = total_shaft_power_data.sum(axis=0)  # Sum along nacelles axis
        valid_mask = ~np.isnan(total_shaft_power_sum)
        if np.any(valid_mask):
            ax4.plot(node_vec[valid_mask], total_shaft_power_sum[valid_mask], 
                    color=default_color, linewidth=2, label='Total Shaft Power')
    else:
        valid_mask = ~np.isnan(total_shaft_power_data)
        if np.any(valid_mask):
            ax4.plot(node_vec[valid_mask], total_shaft_power_data[valid_mask], 
                    color=default_color, linewidth=2, label='Total Shaft Power')
    ax4.set_xlabel('Node')
    ax4.set_ylabel('Shaft Power (kW)')
    ax4.set_title('Total Shaft Power Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Adjust layout for detailed figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.0, w_pad=1.0)
    
    # Show detailed plots
    plt.show()
    
    # Create individual nacelle EM powers figure
    if unit_em_power_data.size > 0:
        fig3, ax3 = plt.subplots(figsize=(15, 8))
        fig3.suptitle('Per-Nacelle Electric Motor Powers', fontsize=16, y=0.98)
        
        num_nacelles = unit_em_power_data.shape[0] if len(unit_em_power_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_em_power = unit_em_power_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_em_power)
            if np.any(valid_mask):
                ax3.plot(node_vec[valid_mask], nac_em_power[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax3.set_xlabel('Node')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Per-Nacelle Electric Motor Power Profiles')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Create motor efficiency figure
    if unit_em_efficiency_data.size > 0:
        fig3_eff, ax3_eff = plt.subplots(figsize=(15, 8))
        fig3_eff.suptitle('Per-Nacelle Electric Motor Efficiencies', fontsize=16, y=0.98)
        
        num_nacelles = unit_em_efficiency_data.shape[0] if len(unit_em_efficiency_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_efficiency = unit_em_efficiency_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_efficiency)
            if np.any(valid_mask):
                ax3_eff.plot(node_vec[valid_mask], nac_efficiency[valid_mask] * 100, 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax3_eff.set_xlabel('Node')
        ax3_eff.set_ylabel('Efficiency (%)')
        ax3_eff.set_title('Per-Nacelle Electric Motor Efficiency Profiles')
        ax3_eff.grid(True, alpha=0.3)
        ax3_eff.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Create individual nacelle GT powers and thrust figure
    if unit_gt_power_data.size > 0 or unit_thrust_data or unit_jet_thrust_data.size > 0:
        fig4, axes = plt.subplots(3, 1, figsize=(15, 15))
        fig4.suptitle('Per-Nacelle Gas Turbine Powers and Thrust', fontsize=16, y=0.98)
        ax4, ax4_thrust, ax4_jet_thrust = axes
        
        # Plot per-nacelle GT powers
        num_nacelles = unit_gt_power_data.shape[0] if len(unit_gt_power_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_gt_power = unit_gt_power_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_gt_power)
            if np.any(valid_mask):
                ax4.plot(node_vec[valid_mask], nac_gt_power[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax4.set_xlabel('Node')
        ax4.set_ylabel('Power (kW)')
        ax4.set_title('Per-Nacelle Gas Turbine Power Profiles')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot individual nacelle thrust
        unit_thrust_array = np.array(unit_thrust_data)
        if unit_thrust_array.size > 0:
            num_nacelles = unit_thrust_array.shape[0] if len(unit_thrust_array.shape) > 1 else 0
            for nac_idx in range(num_nacelles):
                nac_thrust = unit_thrust_array[nac_idx, :]  # Row is nacelle, column is node
                valid_mask = ~np.isnan(nac_thrust)
                if np.any(valid_mask):
                    ax4_thrust.plot(node_vec[valid_mask], nac_thrust[valid_mask], 
                                    color=tab10_colors[nac_idx % len(tab10_colors)], 
                                    linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax4_thrust.set_xlabel('Node')
        ax4_thrust.set_ylabel('Thrust (N)')
        ax4_thrust.set_title('Individual Nacelle Thrust Profiles')
        ax4_thrust.grid(True, alpha=0.3)
        ax4_thrust.legend()
        
        # Plot individual nacelle jet thrust
        if unit_jet_thrust_data.size > 0:
            num_nacelles = unit_jet_thrust_data.shape[0] if len(unit_jet_thrust_data.shape) > 1 else 0
            for nac_idx in range(num_nacelles):
                nac_jet_thrust = unit_jet_thrust_data[nac_idx, :]  # Row is nacelle, column is node
                valid_mask = ~np.isnan(nac_jet_thrust)
                if np.any(valid_mask):
                    ax4_jet_thrust.plot(node_vec[valid_mask], nac_jet_thrust[valid_mask], 
                                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='--')
        
        ax4_jet_thrust.set_xlabel('Node')
        ax4_jet_thrust.set_ylabel('Jet Thrust (N)')
        ax4_jet_thrust.set_title('Individual Nacelle Jet Thrust Profiles')
        ax4_jet_thrust.grid(True, alpha=0.3)
        ax4_jet_thrust.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Create total power comparison figure
    fig5, ax5 = plt.subplots(figsize=(15, 8))
    fig5.suptitle('Total Power Comparison', fontsize=16, y=0.98)
    
    # Plot total EM power
    # total_em_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_em_power_data.size > 0 and len(total_em_power_data.shape) > 1:
        total_em_power_sum = total_em_power_data.sum(axis=0)  # Sum along nacelles axis
        valid_mask_em = ~np.isnan(total_em_power_sum)
        if np.any(valid_mask_em):
            ax5.plot(node_vec[valid_mask_em], total_em_power_sum[valid_mask_em], 
                    color=tab10_colors[0], linewidth=2, label='Total EM Power', linestyle='-')
    else:
        valid_mask_em = ~np.isnan(total_em_power_data)
        if np.any(valid_mask_em):
            ax5.plot(node_vec[valid_mask_em], total_em_power_data[valid_mask_em], 
                    color=tab10_colors[0], linewidth=2, label='Total EM Power', linestyle='-')
    
    # Plot total GT power
    # total_gt_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_gt_power_data.size > 0 and len(total_gt_power_data.shape) > 1:
        total_gt_power_sum = total_gt_power_data.sum(axis=0)  # Sum along nacelles axis
        valid_mask_gt = ~np.isnan(total_gt_power_sum)
        if np.any(valid_mask_gt):
            ax5.plot(node_vec[valid_mask_gt], total_gt_power_sum[valid_mask_gt], 
                    color=tab10_colors[1], linewidth=2, label='Total GT Power', linestyle='-')
    else:
        valid_mask_gt = ~np.isnan(total_gt_power_data)
        if np.any(valid_mask_gt):
            ax5.plot(node_vec[valid_mask_gt], total_gt_power_data[valid_mask_gt], 
                    color=tab10_colors[1], linewidth=2, label='Total GT Power', linestyle='-')
    
    # Plot total nacelle power (EM + GT)
    if total_em_power_data.size > 0 and total_gt_power_data.size > 0:
        if len(total_em_power_data.shape) > 1 and len(total_gt_power_data.shape) > 1:
            # Both are matrices: sum along nacelles axis (axis=0)
            total_em_power_sum = total_em_power_data.sum(axis=0)
            total_gt_power_sum = total_gt_power_data.sum(axis=0)
            total_nacelle_power = total_em_power_sum + total_gt_power_sum
            valid_mask = ~np.isnan(total_nacelle_power)
            if np.any(valid_mask):
                ax5.plot(node_vec[valid_mask], total_nacelle_power[valid_mask], 
                        color=tab10_colors[3], linewidth=2, label='Total Nacelle Power', linestyle='-')
        else:
            # Fallback for 1D arrays
            total_nacelle_power = total_em_power_data + total_gt_power_data
            valid_mask = ~np.isnan(total_nacelle_power)
            if np.any(valid_mask):
                ax5.plot(node_vec[valid_mask], total_nacelle_power[valid_mask], 
                        color=tab10_colors[3], linewidth=2, label='Total Nacelle Power', linestyle='-')
    
    ax5.set_xlabel('Node')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Total Power Comparison')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create individual motor and turbine throttles figure
    fig6, ax6 = plt.subplots(figsize=(15, 8))
    fig6.suptitle('Individual Motor and Turbine Throttles', fontsize=16, y=0.98)
    
    # Plot motor throttles (per nacelle)
    if em_throttle_data.size > 0:
        num_nacelles = em_throttle_data.shape[0] if len(em_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_em_throttle = em_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_em_throttle)
            if np.any(valid_mask):
                ax6.plot(node_vec[valid_mask], nac_em_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} EM', linestyle='-')
    
    # Plot turbine throttles (per nacelle)
    if gt_throttle_data.size > 0:
        num_nacelles = gt_throttle_data.shape[0] if len(gt_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_gt_throttle = gt_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_gt_throttle)
            if np.any(valid_mask):
                ax6.plot(node_vec[valid_mask], nac_gt_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} GT', linestyle='--')
    
    ax6.set_xlabel('Node')
    ax6.set_ylabel('Throttle')
    ax6.set_title('Individual Motor and Turbine Throttle Profiles')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create nacelle throttles figure
    fig7, ax7 = plt.subplots(figsize=(15, 8))
    fig7.suptitle('Nacelle Throttles', fontsize=16, y=0.98)
    
    if nac_throttle_data.size > 0:
        num_nacelles = nac_throttle_data.shape[0] if len(nac_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_throttle = nac_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_throttle)
            if np.any(valid_mask):
                ax7.plot(node_vec[valid_mask], nac_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax7.set_xlabel('Node')
    ax7.set_ylabel('Throttle')
    ax7.set_title('Nacelle Throttle Profiles')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create propeller performance analysis figure (2x2 grid)
    fig8, axes8 = plt.subplots(2, 2, figsize=(15, 12))
    fig8.suptitle('Propeller Performance Analysis', fontsize=16, y=0.98)
    axes8 = axes8.flatten()
    
    # Plot 1: Advance Ratio vs Nodes
    ax8_1 = axes8[0]
    if unit_adv_ratio_data.size > 0:
        num_nacelles = unit_adv_ratio_data.shape[0] if len(unit_adv_ratio_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_adv_ratio = unit_adv_ratio_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_adv_ratio)
            if np.any(valid_mask):
                ax8_1.plot(node_vec[valid_mask], nac_adv_ratio[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax8_1.set_xlabel('Node')
    ax8_1.set_ylabel('Advance Ratio')
    ax8_1.set_title('Propeller Advance Ratio')
    ax8_1.grid(True, alpha=0.3)
    ax8_1.legend()
    
    # Plot 2: Power Coefficient (Cp) vs Nodes
    ax8_2 = axes8[1]
    if unit_cp_data.size > 0:
        num_nacelles = unit_cp_data.shape[0] if len(unit_cp_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_cp = unit_cp_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_cp)
            if np.any(valid_mask):
                ax8_2.plot(node_vec[valid_mask], nac_cp[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax8_2.set_xlabel('Node')
    ax8_2.set_ylabel('Power Coefficient (Cp)')
    ax8_2.set_title('Propeller Power Coefficient')
    ax8_2.grid(True, alpha=0.3)
    ax8_2.legend()
    
    # Plot 3: Propeller Efficiency vs Nodes
    ax8_3 = axes8[2]
    if unit_eta_calc_data.size > 0:
        num_nacelles = unit_eta_calc_data.shape[0] if len(unit_eta_calc_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_eta_calc = unit_eta_calc_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_eta_calc)
            if np.any(valid_mask):
                ax8_3.plot(node_vec[valid_mask], nac_eta_calc[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} (calc)', linestyle='-')
    
    # Add bounded efficiency as dashed lines
    if unit_eta_data.size > 0:
        num_nacelles = unit_eta_data.shape[0] if len(unit_eta_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_eta = unit_eta_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_eta)
            if np.any(valid_mask):
                ax8_3.plot(node_vec[valid_mask], nac_eta[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} (bounded)', linestyle='--', alpha=0.5)
    
    ax8_3.set_xlabel('Node')
    ax8_3.set_ylabel('Efficiency')
    ax8_3.set_title('Propeller Efficiency')
    ax8_3.grid(True, alpha=0.3)
    ax8_3.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0, w_pad=1.5)
    plt.show()
    
    # Save plots and data if requested
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save plots to PDF
        # Add optimization objective to filename if provided
        if opt_objective is not None:
            pdf_filename = f"{output_filename_with_oew}_{opt_objective}_spot_point_{timestamp}.pdf"
        else:
            pdf_filename = f"{output_filename_with_oew}_spot_point_{timestamp}.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig2)
            if unit_em_power_data.size > 0:
                pdf.savefig(fig3)
            if unit_em_efficiency_data.size > 0:
                pdf.savefig(fig3_eff)
            if unit_gt_power_data.size > 0:
                pdf.savefig(fig4)
            pdf.savefig(fig5)
            if em_throttle_data.size > 0 or gt_throttle_data.size > 0:
                pdf.savefig(fig6)
            if nac_throttle_data.size > 0:
                pdf.savefig(fig7)
            if unit_adv_ratio_data.size > 0:
                pdf.savefig(fig8)
            plt.close('all')
        
        # Save all plotted data to CSV
        save_spot_point_data_to_csv(output_filename = output_filename_with_oew, timestamp = timestamp, node_vec = node_vec, 
                                   altitude_data = altitude_data, range_data = range_data, eas_data = eas_data, tas_data = tas_data, vs_data = vs_data, 
                                   cl_data = cl_data, cd_data = cd_data, cd0_data = cd0_data, cd_lift_data = cd_lift_data, lift_data = lift_data,
                                   unit_thrust_data = unit_thrust_data, total_thrust_data = total_thrust_data, drag_data = drag_data, weight_data = weight_data, 
                                   gamma_data = gamma_data, unit_gt_power_data = unit_gt_power_data, gt_throttle_data = gt_throttle_data, 
                                   unit_jet_thrust_data = unit_jet_thrust_data,
                                   unit_prop_thrust_data = unit_prop_thrust_data, unit_prop_rpm_data = unit_prop_rpm_data,
                                   unit_em_power_data = unit_em_power_data, em_throttle_data = em_throttle_data, total_em_power_data = total_em_power_data, 
                                   total_gt_power_data = total_gt_power_data, unit_fuel_flow_data = unit_fuel_flow_data, total_fuel_flow_data = total_fuel_flow_data, 
                                   unit_shaft_power_data = unit_shaft_power_data, total_shaft_power_data = total_shaft_power_data, nac_throttle_data = nac_throttle_data,
                                   unit_adv_ratio_data = unit_adv_ratio_data, unit_cp_data = unit_cp_data, unit_eta_calc_data = unit_eta_calc_data, unit_eta_data = unit_eta_data, 
                                   unit_em_elec_power_data = unit_em_elec_power_data, total_em_elec_power_data = total_em_elec_power_data, unit_em_efficiency_data = unit_em_efficiency_data,
                                   unit_gearbox_power_in_data = unit_gearbox_power_in_data, unit_gearbox_power_out_data = unit_gearbox_power_out_data,
                                   phase_names_data = phase_names_data, payload_qty = payload_qty, turb_op_type = turb_op_type, opt_objective = opt_objective)
        
        print(f"Spot point analysis plots saved to: {pdf_filename}")

    plt.close('all')


def plot_unsteady(prob, mission_config, save_plots=False, save_csv=True, output_filename=None, payload_qty=None, turb_op_type=None, opt_objective=None, nn=11):
    """
    Plot kinematic and aerodynamic values for unsteady flight phases.
    
    Parameters
    ----------
    prob : OpenMDAO Problem object
        The problem containing the data to plot
    phases : list
        List of phase names
    mission_config : dict
        Mission configuration dictionary containing phase information
    save_plots : bool, optional
        Whether to save plots to PDF instead of displaying
    save_csv : bool, optional
        Whether to save data to CSV file (default: True)
    output_filename : str, optional
        Base name for output PDF file (required if save_plots is True)
    opt_objective : str, optional
        Optimization objective type (e.g., 'fuel' or 'range') for filename generation (default: None)
    """
    if save_plots and output_filename is None:
        raise ValueError("output_filename must be provided when save_plots is True")
    
    # Add OEW (ac|weights|OEW) to saved output filenames, rounded to remove decimals
    output_filename_with_oew = _append_oew_to_output_filename(output_filename, _extract_oew_lbm_rounded(prob))

    # Extract mission data using shared function
    data = extract_mission_data(prob, mission_config, nn=nn,
                                collect_time_data=True,  # Unsteady analysis needs time data
                                collect_battery_data=True,  # Unsteady analysis integrates battery
                                collect_gearbox_data=True,  # Always collect if available
                                collect_motor_elec_data=True,  # Always collect if available
                                collect_fuel_used_data=True)  # Unsteady analysis integrates fuel
    
    # Unpack the data dictionary into local variables
    altitude_data = data['altitude_data']
    range_data = data['range_data']
    eas_data = data['eas_data']
    tas_data = data['tas_data']
    vs_data = data['vs_data']
    gamma_data = data['gamma_data']
    cl_data = data['cl_data']
    cd_data = data['cd_data']
    cd0_data = data['cd0_data']
    cd_lift_data = data['cd_lift_data']
    lift_data = data['lift_data']
    drag_data = data['drag_data']
    weight_data = data['weight_data']
    fuel_used_data = data['fuel_used_data']
    unit_thrust_data = data['unit_thrust_data']
    total_thrust_data = data['total_thrust_data']
    unit_gt_power_data = data['unit_gt_power_data']
    total_gt_power_data = data['total_gt_power_data']
    gt_throttle_data = data['gt_throttle_data']
    unit_em_power_data = data['unit_em_power_data']
    total_em_power_data = data['total_em_power_data']
    em_throttle_data = data['em_throttle_data']
    unit_shaft_power_data = data['unit_shaft_power_data']
    total_shaft_power_data = data['total_shaft_power_data']
    nac_throttle_data = data['nac_throttle_data']
    unit_fuel_flow_data = data['unit_fuel_flow_data']
    total_fuel_flow_data = data['total_fuel_flow_data']
    unit_jet_thrust_data = data['unit_jet_thrust_data']
    unit_adv_ratio_data = data['unit_adv_ratio_data']
    unit_cp_data = data['unit_cp_data']
    unit_eta_calc_data = data['unit_eta_calc_data']
    unit_eta_data = data['unit_eta_data']
    unit_prop_thrust_data = data['unit_prop_thrust_data']
    unit_prop_rpm_data = data['unit_prop_rpm_data']
    unit_gearbox_power_in_data = data['unit_gearbox_power_in_data']
    unit_gearbox_power_out_data = data['unit_gearbox_power_out_data']
    unit_em_elec_power_data = data['unit_em_elec_power_data']
    total_em_elec_power_data = data['total_em_elec_power_data']
    unit_em_efficiency_data = data['unit_em_efficiency_data']
    bat_soc_data = data['bat_soc_data']
    bat_voltage_data = data['bat_voltage_data']
    bat_vline_cell_raw_data = data['bat_vline_cell_raw_data']
    bat_p_cell_data = data['bat_p_cell_data']
    bat_i_bat_data = data['bat_i_bat_data']
    bat_i_cell_raw_data = data['bat_i_cell_raw_data']
    bat_ocv_cell_data = data['bat_ocv_cell_data']
    total_batt_power_data = data['total_batt_power_data']
    p_aux_elec_data = data['p_aux_elec_data']
    bat_time_vec = data['bat_time_vec']
    total_nodes_batt = data['total_nodes_batt']
    time_vec = data['time_vec']
    phase_boundaries = data['phase_boundaries']
    phase_names_list = data['phase_names_list']
    phase_names_data = data['phase_names_data']
    phases = data['phases']
    data_type = data['data_type']
    
    # Note: duration_boundary_data was in original but not in extract function, set to empty
    duration_boundary_data = np.array([])
    
    # Define colors using tab10 colormap
    import matplotlib.cm as cm
    tab10_colors = cm.tab10.colors
    
    # Define colors for multiple values on same plot
    multi_colors = {
        'eas': tab10_colors[0],           # First tab10 color
        'tas': tab10_colors[1],           # Second tab10 color
        'vs': tab10_colors[0],            # First tab10 color (single line)
        'cl': tab10_colors[0],            # First tab10 color
        'cd': tab10_colors[1],            # Second tab10 color
        'cd0': tab10_colors[0],           # First tab10 color
        'cd_lift': tab10_colors[1],       # Second tab10 color
        'lift': tab10_colors[0],          # First tab10 color
        'thrust': tab10_colors[1],        # Second tab10 color
        'drag': tab10_colors[2],          # Third tab10 color
    }
    
    # Default color for single-value plots
    default_color = tab10_colors[0]  # First tab10 color
    
    # Helper function to add phase boundary lines and labels
    def add_phase_boundaries(ax, phase_boundaries, phase_names_list, time_vec):
        """Add vertical dashed lines and phase labels to a plot"""
        for i, (boundary_time, phase_name) in enumerate(zip(phase_boundaries, phase_names_list)):
            if boundary_time > time_vec[0] and boundary_time < time_vec[-1]:  # Only show if within plot range
                ax.axvline(x=boundary_time, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                # Add phase name as text annotation - ensure it's within plot bounds
                y_min, y_max = ax.get_ylim()
                y_position = y_min + (y_max - y_min) * 0.95  # Position at 95% of the way up from bottom
                ax.text(boundary_time, y_position, f'{phase_name}\n{boundary_time:.1f}min', 
                    rotation=90, verticalalignment='top', horizontalalignment='right',
                    fontsize=8, alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Create main figure with subplots (4x2 grid)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Unsteady Flight Analysis', fontsize=16, y=0.98)
    axes = axes.flatten()
    
    # Ensure we only use the first 8 subplots
    axes = axes[:8]
    
    # Plot 1: Altitude vs Time
    ax1 = axes[0]
    valid_mask = ~np.isnan(altitude_data)
    if np.any(valid_mask):
        ax1.plot(time_vec[valid_mask], altitude_data[valid_mask], 
                color=default_color, linewidth=2, label='Altitude')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Altitude (ft)')
    ax1.set_title('Altitude Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    add_phase_boundaries(ax1, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 2: Range vs Time
    ax2 = axes[1]
    valid_mask = ~np.isnan(range_data)
    if np.any(valid_mask):
        ax2.plot(time_vec[valid_mask], range_data[valid_mask], 
                color=default_color, linewidth=2, label='Range')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Range (NM)')
    ax2.set_title('Range Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    add_phase_boundaries(ax2, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 3: Airspeed vs Time (EAS and TAS on same plot)
    ax3 = axes[2]
    valid_mask_eas = ~np.isnan(eas_data)
    valid_mask_tas = ~np.isnan(tas_data)
    
    if np.any(valid_mask_eas):
        ax3.plot(time_vec[valid_mask_eas], eas_data[valid_mask_eas], 
                color=multi_colors['eas'], linewidth=2, label='EAS', linestyle='-')
    if np.any(valid_mask_tas):
        ax3.plot(time_vec[valid_mask_tas], tas_data[valid_mask_tas], 
                color=multi_colors['tas'], linewidth=2, label='TAS', linestyle='-')
    
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Airspeed (knots)')
    ax3.set_title('Airspeed Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    add_phase_boundaries(ax3, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 4: Vertical Speed vs Time
    ax4 = axes[3]
    valid_mask = ~np.isnan(vs_data)
    if np.any(valid_mask):
        ax4.plot(time_vec[valid_mask], vs_data[valid_mask], 
                color=default_color, linewidth=2, label='Vertical Speed')
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Vertical Speed (ft/min)')
    ax4.set_title('Vertical Speed Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax4, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 5: Aerodynamic Coefficients vs Time (CL and CD on same plot)
    ax5 = axes[4]
    valid_mask_cl = ~np.isnan(cl_data)
    valid_mask_cd = ~np.isnan(cd_data)
    
    if np.any(valid_mask_cl):
        ax5.plot(time_vec[valid_mask_cl], cl_data[valid_mask_cl], 
                color=multi_colors['cl'], linewidth=2, label='CL', linestyle='-')
    if np.any(valid_mask_cd):
        ax5.plot(time_vec[valid_mask_cd], cd_data[valid_mask_cd], 
                color=multi_colors['cd'], linewidth=2, label='CD', linestyle='-')
    
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Aerodynamic Coefficients')
    ax5.set_title('Lift and Total Drag Coefficients')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    add_phase_boundaries(ax5, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 6: Drag Components vs Time (CD0 and CD_lift on same plot)
    ax6 = axes[5]
    valid_mask_cd0 = ~np.isnan(cd0_data)
    valid_mask_cd_lift = ~np.isnan(cd_lift_data)
    
    if np.any(valid_mask_cd0):
        ax6.plot(time_vec[valid_mask_cd0], cd0_data[valid_mask_cd0], 
                color=multi_colors['cd0'], linewidth=2, label='CD0', linestyle='-')
    if np.any(valid_mask_cd_lift):
        ax6.plot(time_vec[valid_mask_cd_lift], cd_lift_data[valid_mask_cd_lift], 
                color=multi_colors['cd_lift'], linewidth=2, label='CD_induced', linestyle='-')
    
    ax6.set_xlabel('Time (min)')
    ax6.set_ylabel('Drag Coefficients')
    ax6.set_title('Zero-Lift and Lift-Induced Drag')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    add_phase_boundaries(ax6, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 7: Forces vs Time (Lift, Thrust, Drag on same plot)
    ax7 = axes[6]
    valid_mask_lift = ~np.isnan(lift_data)
    valid_mask_thrust = ~np.isnan(total_thrust_data) # Use total_thrust_data for thrust
    valid_mask_drag = ~np.isnan(drag_data)
    
    #if np.any(valid_mask_lift):
    #    ax7.plot(time_vec[valid_mask_lift], lift_data[valid_mask_lift]/1000, 
    #            color=tab10_colors[0], linewidth=2, label='Lift', linestyle='-')
    if np.any(valid_mask_thrust):
        ax7.plot(time_vec[valid_mask_thrust], total_thrust_data[valid_mask_thrust]/1000, 
                color=tab10_colors[1], linewidth=2, label='Thrust', linestyle='-')
    if np.any(valid_mask_drag):
        ax7.plot(time_vec[valid_mask_drag], drag_data[valid_mask_drag]/1000, 
                color=tab10_colors[3], linewidth=2, label='Drag', linestyle='-')
    
    ax7.set_xlabel('Time (min)')
    ax7.set_ylabel('Force (kN)')
    ax7.set_title('Aerodynamic and Propulsive Forces')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    add_phase_boundaries(ax7, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 8: Weight vs Time
    ax8 = axes[7]
    valid_mask = ~np.isnan(weight_data)
    if np.any(valid_mask):
        ax8.plot(time_vec[valid_mask], weight_data[valid_mask], 
                color=default_color, linewidth=2, label='Weight')
    ax8.set_xlabel('Time (min)')
    ax8.set_ylabel('Weight (kg)')
    ax8.set_title('Aircraft Weight')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    add_phase_boundaries(ax8, phase_boundaries, phase_names_list, time_vec)
    
    # Adjust layout for main figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.0, w_pad=1.0)
    
    # Show main plots
    plt.show()
    
    # Create detailed analysis figure (2x2 grid)
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Detailed Unsteady Flight Analysis', fontsize=16, y=0.98)
    axes2 = axes2.flatten()
    
    # Ensure we only use the first 4 subplots
    axes2 = axes2[:4]
    
    # Plot 1: Flight Path Angle vs Time
    ax1 = axes2[0]
    valid_mask = ~np.isnan(gamma_data)
    if np.any(valid_mask):
        ax1.plot(time_vec[valid_mask], gamma_data[valid_mask], 
                color=default_color, linewidth=2, label='Flight Path Angle')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Gradient (%)')
    ax1.set_title('Flight Path Angle')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax1, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 2: Fuel Flow vs Time
    ax2 = axes2[1]
    # total_fuel_flow_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_fuel_flow_data.size > 0 and len(total_fuel_flow_data.shape) > 1:
        # Sum along axis 0 (nacelles) to get total fuel flow
        total_fuel_flow_sum = total_fuel_flow_data.sum(axis=0)
        valid_mask = ~np.isnan(total_fuel_flow_sum)
        if np.any(valid_mask):
            ax2.plot(time_vec[valid_mask], total_fuel_flow_sum[valid_mask], 
                    color=default_color, linewidth=2, label='Total Fuel Flow')
    else:
        valid_mask = ~np.isnan(total_fuel_flow_data)
        if np.any(valid_mask):
            ax2.plot(time_vec[valid_mask], total_fuel_flow_data[valid_mask], 
                    color=default_color, linewidth=2, label='Fuel Flow')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Total Fuel Flow (kg/h)')
    ax2.set_title('Fuel Consumption Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    add_phase_boundaries(ax2, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 3: Power vs Time
    ax3 = axes2[2]
    # total_em_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_em_power_data.size > 0 and len(total_em_power_data.shape) > 1:
        # Sum along axis 0 (nacelles) to get total EM power
        total_em_power_sum = total_em_power_data.sum(axis=0)
        valid_mask = ~np.isnan(total_em_power_sum)
        if np.any(valid_mask):
            ax3.plot(time_vec[valid_mask], total_em_power_sum[valid_mask], 
                    color=default_color, linewidth=2, label='Total EM Power')
    else:
        valid_mask = ~np.isnan(total_em_power_data)
        if np.any(valid_mask):
            ax3.plot(time_vec[valid_mask], total_em_power_data[valid_mask], 
                    color=default_color, linewidth=2, label='Total EM Power')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Power Profile')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    add_phase_boundaries(ax3, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 4: Fuel Used vs Time
    ax4 = axes2[3]
    # Convert lists to numpy arrays for boolean indexing
    time_vec_array = np.array(time_vec)
    fuel_used_data_array = np.array(fuel_used_data)
    valid_mask = ~np.isnan(fuel_used_data_array)
    if np.any(valid_mask):
        ax4.plot(time_vec_array[valid_mask], fuel_used_data_array[valid_mask], 
                color=default_color, linewidth=2, label='Fuel Used')
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Fuel Used (kg)')
    ax4.set_title('Cumulative Fuel Usage')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    add_phase_boundaries(ax4, phase_boundaries, phase_names_list, time_vec)
    
    # Adjust layout for detailed figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.0, w_pad=1.0)
    
    # Show detailed plots
    plt.show()
    
    # Create individual nacelle EM powers figure
    if unit_em_power_data.size > 0:
        fig3, ax3 = plt.subplots(figsize=(15, 8))
        fig3.suptitle('Per-Nacelle Electric Motor Powers', fontsize=16, y=0.98)
        
        num_nacelles = unit_em_power_data.shape[0] if len(unit_em_power_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_em_power = unit_em_power_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_em_power)
            if np.any(valid_mask):
                ax3.plot(time_vec[valid_mask], nac_em_power[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Per-Nacelle Electric Motor Power Profiles')
        ax3.grid(True, alpha=0.3)
    ax3.legend()
    add_phase_boundaries(ax3, phase_boundaries, phase_names_list, time_vec)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Create motor efficiency figure
    if unit_em_efficiency_data.size > 0:
        fig3_eff, ax3_eff = plt.subplots(figsize=(15, 8))
        fig3_eff.suptitle('Per-Nacelle Electric Motor Efficiencies', fontsize=16, y=0.98)
        
        num_nacelles = unit_em_efficiency_data.shape[0] if len(unit_em_efficiency_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_efficiency = unit_em_efficiency_data[nac_idx, :]  # Row is nacelle, column is node
            # Ensure the length matches time_vec
            min_len = min(len(nac_efficiency), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(nac_efficiency[:min_len])
                if np.any(valid_mask):
                    ax3_eff.plot(time_vec[:min_len][valid_mask], nac_efficiency[:min_len][valid_mask] * 100, 
                            color=tab10_colors[nac_idx % len(tab10_colors)], 
                            linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax3_eff.set_xlabel('Time (min)')
        ax3_eff.set_ylabel('Efficiency (%)')
        ax3_eff.set_title('Per-Nacelle Electric Motor Efficiency Profiles')
        ax3_eff.grid(True, alpha=0.3)
        ax3_eff.legend()
        add_phase_boundaries(ax3_eff, phase_boundaries, phase_names_list, time_vec)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Create individual nacelle GT powers and thrust figure
    if unit_gt_power_data.size > 0 or unit_thrust_data or unit_jet_thrust_data.size > 0:
        fig4, axes = plt.subplots(3, 1, figsize=(15, 15))
        fig4.suptitle('Per-Nacelle Gas Turbine Powers and Thrust', fontsize=16, y=0.98)
        ax4, ax4_thrust, ax4_jet_thrust = axes
        
        # Plot per-nacelle GT powers
        num_nacelles = unit_gt_power_data.shape[0] if len(unit_gt_power_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_gt_power = unit_gt_power_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_gt_power)
            if np.any(valid_mask):
                ax4.plot(time_vec[valid_mask], nac_gt_power[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Power (kW)')
        ax4.set_title('Individual Gas Turbine Power Profiles')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        add_phase_boundaries(ax4, phase_boundaries, phase_names_list, time_vec)
        
        # Plot individual nacelle thrust
        unit_thrust_array = np.array(unit_thrust_data)
        if unit_thrust_array.size > 0:
            num_nacelles = unit_thrust_array.shape[0] if len(unit_thrust_array.shape) > 1 else 0
            for nac_idx in range(num_nacelles):
                nac_thrust = unit_thrust_array[nac_idx, :]  # Row is nacelle, column is node
                valid_mask = ~np.isnan(nac_thrust)
                if np.any(valid_mask):
                    ax4_thrust.plot(time_vec[valid_mask], nac_thrust[valid_mask], 
                                    color=tab10_colors[nac_idx % len(tab10_colors)], 
                                    linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
        
        ax4_thrust.set_xlabel('Time (min)')
        ax4_thrust.set_ylabel('Thrust (N)')
        ax4_thrust.set_title('Individual Nacelle Thrust Profiles')
        ax4_thrust.grid(True, alpha=0.3)
        ax4_thrust.legend()
        add_phase_boundaries(ax4_thrust, phase_boundaries, phase_names_list, time_vec)
        
        # Plot individual nacelle jet thrust
        if unit_jet_thrust_data.size > 0:
            num_nacelles = unit_jet_thrust_data.shape[0] if len(unit_jet_thrust_data.shape) > 1 else 0
            for nac_idx in range(num_nacelles):
                nac_jet_thrust = unit_jet_thrust_data[nac_idx, :]  # Row is nacelle, column is node
                valid_mask = ~np.isnan(nac_jet_thrust)
                if np.any(valid_mask):
                    ax4_jet_thrust.plot(time_vec[valid_mask], nac_jet_thrust[valid_mask], 
                                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='--')
        
        ax4_jet_thrust.set_xlabel('Time (min)')
        ax4_jet_thrust.set_ylabel('Jet Thrust (N)')
        ax4_jet_thrust.set_title('Individual Nacelle Jet Thrust Profiles')
        ax4_jet_thrust.grid(True, alpha=0.3)
        ax4_jet_thrust.legend()
        add_phase_boundaries(ax4_jet_thrust, phase_boundaries, phase_names_list, time_vec)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Create total power comparison figure
    fig5, ax5 = plt.subplots(figsize=(15, 8))
    fig5.suptitle('Total Power Comparison', fontsize=16, y=0.98)
    
    # Plot total EM power
    # total_em_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_em_power_data.size > 0 and len(total_em_power_data.shape) > 1:
        total_em_power_sum = total_em_power_data.sum(axis=0)  # Sum along nacelles axis
        # Ensure the length matches time_vec
        min_len = min(len(total_em_power_sum), len(time_vec))
        if min_len > 0:
            valid_mask_em = ~np.isnan(total_em_power_sum[:min_len])
            if np.any(valid_mask_em):
                ax5.plot(time_vec[:min_len][valid_mask_em], total_em_power_sum[:min_len][valid_mask_em], 
                        color=tab10_colors[0], linewidth=2, label='Total EM Power', linestyle='-')
    else:
        if total_em_power_data.size > 0:
            min_len = min(len(total_em_power_data), len(time_vec))
            if min_len > 0:
                valid_mask_em = ~np.isnan(total_em_power_data[:min_len])
                if np.any(valid_mask_em):
                    ax5.plot(time_vec[:min_len][valid_mask_em], total_em_power_data[:min_len][valid_mask_em], 
                            color=tab10_colors[0], linewidth=2, label='Total EM Power', linestyle='-')
    
    # Plot total GT power
    # total_gt_power_data is now a matrix: shape (num_nacelles, num_nodes)
    if total_gt_power_data.size > 0 and len(total_gt_power_data.shape) > 1:
        total_gt_power_sum = total_gt_power_data.sum(axis=0)  # Sum along nacelles axis
        # Ensure the length matches time_vec
        min_len = min(len(total_gt_power_sum), len(time_vec))
        if min_len > 0:
            valid_mask_gt = ~np.isnan(total_gt_power_sum[:min_len])
            if np.any(valid_mask_gt):
                ax5.plot(time_vec[:min_len][valid_mask_gt], total_gt_power_sum[:min_len][valid_mask_gt], 
                        color=tab10_colors[1], linewidth=2, label='Total GT Power', linestyle='-')
    else:
        if total_gt_power_data.size > 0:
            min_len = min(len(total_gt_power_data), len(time_vec))
            if min_len > 0:
                valid_mask_gt = ~np.isnan(total_gt_power_data[:min_len])
                if np.any(valid_mask_gt):
                    ax5.plot(time_vec[:min_len][valid_mask_gt], total_gt_power_data[:min_len][valid_mask_gt], 
                            color=tab10_colors[1], linewidth=2, label='Total GT Power', linestyle='-')
    
    # Plot total nacelle power (EM + GT)
    if total_em_power_data.size > 0 and total_gt_power_data.size > 0:
        if len(total_em_power_data.shape) > 1 and len(total_gt_power_data.shape) > 1:
            # Both are matrices: sum along nacelles axis (axis=0)
            total_em_power_sum = total_em_power_data.sum(axis=0)
            total_gt_power_sum = total_gt_power_data.sum(axis=0)
            total_nacelle_power = total_em_power_sum + total_gt_power_sum
            # Ensure the length matches time_vec
            min_len = min(len(total_nacelle_power), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(total_nacelle_power[:min_len])
                if np.any(valid_mask):
                    ax5.plot(time_vec[:min_len][valid_mask], total_nacelle_power[:min_len][valid_mask], 
                            color=tab10_colors[3], linewidth=2, label='Total Nacelle Power', linestyle='-')
        else:
            # Fallback for 1D arrays
            total_nacelle_power = total_em_power_data + total_gt_power_data
            min_len = min(len(total_nacelle_power), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(total_nacelle_power[:min_len])
                if np.any(valid_mask):
                    ax5.plot(time_vec[:min_len][valid_mask], total_nacelle_power[:min_len][valid_mask], 
                            color=tab10_colors[3], linewidth=2, label='Total Nacelle Power', linestyle='-')
    
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Power (kW)')
    ax5.set_title('Total Power Comparison')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    add_phase_boundaries(ax5, phase_boundaries, phase_names_list, time_vec)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create individual motor and turbine throttles figure
    fig6, ax6 = plt.subplots(figsize=(15, 8))
    fig6.suptitle('Individual Motor and Turbine Throttles', fontsize=16, y=0.98)
    
    # Plot motor throttles (per nacelle)
    if em_throttle_data.size > 0:
        num_nacelles = em_throttle_data.shape[0] if len(em_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_em_throttle = em_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_em_throttle)
            if np.any(valid_mask):
                ax6.plot(time_vec[valid_mask], nac_em_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} EM', linestyle='-')
    
    # Plot turbine throttles (per nacelle)
    if gt_throttle_data.size > 0:
        num_nacelles = gt_throttle_data.shape[0] if len(gt_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_gt_throttle = gt_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_gt_throttle)
            if np.any(valid_mask):
                ax6.plot(time_vec[valid_mask], nac_gt_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1} GT', linestyle='--')
    
    ax6.set_xlabel('Time (min)')
    ax6.set_ylabel('Throttle')
    ax6.set_title('Individual Motor and Turbine Throttle Profiles')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    add_phase_boundaries(ax6, phase_boundaries, phase_names_list, time_vec)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create nacelle throttles figure
    fig7, ax7 = plt.subplots(figsize=(15, 8))
    fig7.suptitle('Nacelle Throttles', fontsize=16, y=0.98)
    
    if nac_throttle_data.size > 0:
        num_nacelles = nac_throttle_data.shape[0] if len(nac_throttle_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_throttle = nac_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            valid_mask = ~np.isnan(nac_throttle)
            if np.any(valid_mask):
                ax7.plot(time_vec[valid_mask], nac_throttle[valid_mask], 
                        color=tab10_colors[nac_idx % len(tab10_colors)], 
                        linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax7.set_xlabel('Time (min)')
    ax7.set_ylabel('Throttle')
    ax7.set_title('Nacelle Throttle Profiles')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    add_phase_boundaries(ax7, phase_boundaries, phase_names_list, time_vec)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Create battery analysis figure (2x2 grid)
    fig8, axes8 = plt.subplots(2, 2, figsize=(15, 12))
    fig8.suptitle('Battery Pack Analysis', fontsize=16, y=0.98)
    axes8 = axes8.flatten()
    
    # Plot 1: Battery SOC vs Time (using battery time vector) - per string
    ax8_1 = axes8[0]
    # bat_soc_data is now a matrix: shape (num_str, num_node)
    if bat_soc_data.size > 0 and len(bat_soc_data.shape) > 1:
        num_strings = bat_soc_data.shape[0]
        for str_idx in range(num_strings):
            str_soc = bat_soc_data[str_idx, :]  # Row is string, column is node
            valid_mask_soc = ~np.isnan(str_soc)
            if np.any(valid_mask_soc):
                ax8_1.plot(bat_time_vec[valid_mask_soc], str_soc[valid_mask_soc], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} SOC', linestyle='-')
    else:
        valid_mask_soc = ~np.isnan(bat_soc_data)
        if np.any(valid_mask_soc):
            ax8_1.plot(bat_time_vec[valid_mask_soc], bat_soc_data[valid_mask_soc], 
                    color=tab10_colors[0], linewidth=2, label='Battery SOC')
    ax8_1.set_xlabel('Time (min)')
    ax8_1.set_ylabel('State of Charge')
    ax8_1.set_title('Battery State of Charge Profile (Per String)')
    ax8_1.grid(True, alpha=0.3)
    ax8_1.legend()
    add_phase_boundaries(ax8_1, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 2: Battery Voltage vs Time (using battery time vector) - per string
    ax8_2 = axes8[1]
    # bat_voltage_data is now a matrix: shape (num_str, num_node)
    if bat_voltage_data.size > 0 and len(bat_voltage_data.shape) > 1:
        num_strings = bat_voltage_data.shape[0]
        for str_idx in range(num_strings):
            str_voltage = bat_voltage_data[str_idx, :]  # Row is string, column is node
            valid_mask_voltage = ~np.isnan(str_voltage)
            if np.any(valid_mask_voltage):
                ax8_2.plot(bat_time_vec[valid_mask_voltage], str_voltage[valid_mask_voltage], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Voltage', linestyle='-')
    else:
        valid_mask_voltage = ~np.isnan(bat_voltage_data)
        if np.any(valid_mask_voltage):
            ax8_2.plot(bat_time_vec[valid_mask_voltage], bat_voltage_data[valid_mask_voltage], 
                    color=tab10_colors[1], linewidth=2, label='Battery Pack Voltage')
    ax8_2.set_xlabel('Time (min)')
    ax8_2.set_ylabel('Battery Voltage (V)')
    ax8_2.set_title('Battery Pack Voltage (Per String)')
    ax8_2.grid(True, alpha=0.3)
    ax8_2.legend()
    add_phase_boundaries(ax8_2, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 3: Battery Current vs Time - per string
    ax8_3 = axes8[2]
    # bat_i_bat_data is now a matrix: shape (num_str, num_node)
    if bat_i_bat_data.size > 0 and len(bat_i_bat_data.shape) > 1:
        num_strings = bat_i_bat_data.shape[0]
        for str_idx in range(num_strings):
            str_current = bat_i_bat_data[str_idx, :]  # Row is string, column is node
            valid_mask_i_bat = ~np.isnan(str_current)
            if np.any(valid_mask_i_bat):
                ax8_3.plot(bat_time_vec[valid_mask_i_bat], str_current[valid_mask_i_bat], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Current', linestyle='-')
    else:
        valid_mask_i_bat = ~np.isnan(bat_i_bat_data)
        if np.any(valid_mask_i_bat):
            ax8_3.plot(bat_time_vec[valid_mask_i_bat], bat_i_bat_data[valid_mask_i_bat], 
                    color=tab10_colors[2], linewidth=2, label='Battery Pack Current', linestyle='-')

    ax8_3.set_xlabel('Time (min)')
    ax8_3.set_ylabel('Current (A)')
    ax8_3.set_title('Battery Pack Current (Per String)')
    ax8_3.grid(True, alpha=0.3)
    ax8_3.legend()
    ax8_3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax8_3, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 4: Battery Power vs Time - per string
    ax8_4 = axes8[3]
    # total_batt_power_data is now a matrix: shape (num_str, num_node)
    if total_batt_power_data.size > 0 and len(total_batt_power_data.shape) > 1:
        num_strings = total_batt_power_data.shape[0]
        for str_idx in range(num_strings):
            str_power = total_batt_power_data[str_idx, :]  # Row is string, column is node
            valid_mask_p_batt = ~np.isnan(str_power)
            if np.any(valid_mask_p_batt):
                ax8_4.plot(bat_time_vec[valid_mask_p_batt], str_power[valid_mask_p_batt], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Power', linestyle='-')
    else:
        valid_mask_p_batt = ~np.isnan(total_batt_power_data)
        if np.any(valid_mask_p_batt):
            ax8_4.plot(bat_time_vec[valid_mask_p_batt], total_batt_power_data[valid_mask_p_batt], 
                    color=tab10_colors[5], linewidth=2, label='Battery Pack Power')
    
    ax8_4.set_xlabel('Time (min)')
    ax8_4.set_ylabel('Power (kW)')
    ax8_4.set_title('Battery Power Profiles (Per String)')
    ax8_4.grid(True, alpha=0.3)
    ax8_4.legend()
    ax8_4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax8_4, phase_boundaries, phase_names_list, time_vec)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0, w_pad=1.5)
    plt.show()
    
    # Create detailed battery cell analysis figure
    fig9, axes9 = plt.subplots(2, 2, figsize=(15, 12))
    fig9.suptitle('Battery Cell Profiles', fontsize=16, y=0.98)
    axes9 = axes9.flatten()
    
    # Plot 1: Cell Voltage vs Time - per string
    ax9_1 = axes9[0]
    # bat_vline_cell_raw_data is now a matrix: shape (num_str, num_node)
    if bat_vline_cell_raw_data.size > 0 and len(bat_vline_cell_raw_data.shape) > 1:
        num_strings = bat_vline_cell_raw_data.shape[0]
        for str_idx in range(num_strings):
            str_vline_cell = bat_vline_cell_raw_data[str_idx, :]  # Row is string, column is node
            valid_mask_vline_cell = ~np.isnan(str_vline_cell)
            if np.any(valid_mask_vline_cell):
                ax9_1.plot(bat_time_vec[valid_mask_vline_cell], str_vline_cell[valid_mask_vline_cell], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Cell Voltage', linestyle='-')
    else:
        valid_mask_vline_cell = ~np.isnan(bat_vline_cell_raw_data)
        if np.any(valid_mask_vline_cell):
            ax9_1.plot(bat_time_vec[valid_mask_vline_cell], bat_vline_cell_raw_data[valid_mask_vline_cell], 
                    color=tab10_colors[0], linewidth=2, label='Cell Voltage')
    ax9_1.set_xlabel('Time (min)')
    ax9_1.set_ylabel('Voltage (V)')
    ax9_1.set_title('Cell Voltage (Per String)')
    ax9_1.grid(True, alpha=0.3)
    ax9_1.legend()
    add_phase_boundaries(ax9_1, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 2: Cell Power vs Time - per string
    ax9_2 = axes9[1]
    # bat_p_cell_data is now a matrix: shape (num_str, num_node)
    if bat_p_cell_data.size > 0 and len(bat_p_cell_data.shape) > 1:
        num_strings = bat_p_cell_data.shape[0]
        for str_idx in range(num_strings):
            str_p_cell = bat_p_cell_data[str_idx, :]  # Row is string, column is node
            valid_mask_p_cell_detailed = ~np.isnan(str_p_cell)
            if np.any(valid_mask_p_cell_detailed):
                ax9_2.plot(bat_time_vec[valid_mask_p_cell_detailed], str_p_cell[valid_mask_p_cell_detailed], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Cell Power', linestyle='-')
    else:
        valid_mask_p_cell_detailed = ~np.isnan(bat_p_cell_data)
        if np.any(valid_mask_p_cell_detailed):
            ax9_2.plot(bat_time_vec[valid_mask_p_cell_detailed], bat_p_cell_data[valid_mask_p_cell_detailed], 
                    color=tab10_colors[1], linewidth=2, label='Cell Power')
    ax9_2.set_xlabel('Time (min)')
    ax9_2.set_ylabel('Power (W)')
    ax9_2.set_title('Cell Power (Per String)')
    ax9_2.grid(True, alpha=0.3)
    ax9_2.legend()
    ax9_2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax9_2, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 3: Cell Current vs Time - per string
    ax9_3 = axes9[2]
    # bat_i_cell_raw_data is now a matrix: shape (num_str, num_node)
    if bat_i_cell_raw_data.size > 0 and len(bat_i_cell_raw_data.shape) > 1:
        num_strings = bat_i_cell_raw_data.shape[0]
        for str_idx in range(num_strings):
            str_i_cell = bat_i_cell_raw_data[str_idx, :]  # Row is string, column is node
            valid_mask_i_cell_detailed = ~np.isnan(str_i_cell)
            if np.any(valid_mask_i_cell_detailed):
                ax9_3.plot(bat_time_vec[valid_mask_i_cell_detailed], str_i_cell[valid_mask_i_cell_detailed], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Cell Current', linestyle='-')
    else:
        valid_mask_i_cell_detailed = ~np.isnan(bat_i_cell_raw_data)
        if np.any(valid_mask_i_cell_detailed):
            ax9_3.plot(bat_time_vec[valid_mask_i_cell_detailed], bat_i_cell_raw_data[valid_mask_i_cell_detailed], 
                    color=tab10_colors[2], linewidth=2, label='Cell Current')
    ax9_3.set_xlabel('Time (min)')
    ax9_3.set_ylabel('Current (A)')
    ax9_3.set_title('Cell Current (Per String)')
    ax9_3.grid(True, alpha=0.3)
    ax9_3.legend()
    ax9_3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax9_3, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 4: Battery Pack vs Cell Comparison - per string
    ax9_4 = axes9[3]
    # bat_vline_cell_raw_data is now a matrix: shape (num_str, num_node)
    if bat_vline_cell_raw_data.size > 0 and len(bat_vline_cell_raw_data.shape) > 1:
        num_strings = bat_vline_cell_raw_data.shape[0]
        for str_idx in range(num_strings):
            str_vline_cell = bat_vline_cell_raw_data[str_idx, :]  # Row is string, column is node
            valid_mask_cell_voltage = ~np.isnan(str_vline_cell)
            if np.any(valid_mask_cell_voltage):
                ax9_4.plot(bat_time_vec[valid_mask_cell_voltage], str_vline_cell[valid_mask_cell_voltage], 
                        color=tab10_colors[str_idx % len(tab10_colors)], 
                        linewidth=2, label=f'String {str_idx+1} Cell Voltage', linestyle='-')
    else:
        valid_mask_cell_voltage = ~np.isnan(bat_vline_cell_raw_data)
        if np.any(valid_mask_cell_voltage):
            ax9_4.plot(bat_time_vec[valid_mask_cell_voltage], bat_vline_cell_raw_data[valid_mask_cell_voltage], 
                    color=tab10_colors[4], linewidth=2, label='Cell Voltage')
    
    ax9_4.set_xlabel('Time (min)')
    ax9_4.set_ylabel('Voltage (V)')
    ax9_4.set_title('Cell Voltage (Per String)')
    ax9_4.grid(True, alpha=0.3)
    ax9_4.legend()
    add_phase_boundaries(ax9_4, phase_boundaries, phase_names_list, time_vec)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0, w_pad=1.5)
    plt.show()
    
    # Create propeller performance analysis figure (2x2 grid)
    fig10, axes10 = plt.subplots(2, 2, figsize=(15, 12))
    fig10.suptitle('Propeller Performance Analysis', fontsize=16, y=0.98)
    axes10 = axes10.flatten()
    
    # Plot 1: Advance Ratio vs Time
    ax10_1 = axes10[0]
    if unit_adv_ratio_data.size > 0:
        num_nacelles = unit_adv_ratio_data.shape[0] if len(unit_adv_ratio_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_adv_ratio = unit_adv_ratio_data[nac_idx, :]  # Row is nacelle, column is node
            # Ensure the length matches time_vec
            min_len = min(len(nac_adv_ratio), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(nac_adv_ratio[:min_len])
                if np.any(valid_mask):
                    ax10_1.plot(time_vec[:min_len][valid_mask], nac_adv_ratio[:min_len][valid_mask], 
                            color=tab10_colors[nac_idx % len(tab10_colors)], 
                            linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax10_1.set_xlabel('Time (min)')
    ax10_1.set_ylabel('Advance Ratio')
    ax10_1.set_title('Propeller Advance Ratio')
    ax10_1.grid(True, alpha=0.3)
    ax10_1.legend()
    add_phase_boundaries(ax10_1, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 2: Power Coefficient (Cp) vs Time
    ax10_2 = axes10[1]
    if unit_cp_data.size > 0:
        num_nacelles = unit_cp_data.shape[0] if len(unit_cp_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_cp = unit_cp_data[nac_idx, :]  # Row is nacelle, column is node
            # Ensure the length matches time_vec
            min_len = min(len(nac_cp), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(nac_cp[:min_len])
                if np.any(valid_mask):
                    ax10_2.plot(time_vec[:min_len][valid_mask], nac_cp[:min_len][valid_mask], 
                            color=tab10_colors[nac_idx % len(tab10_colors)], 
                            linewidth=2, label=f'Nacelle {nac_idx+1}', linestyle='-')
    
    ax10_2.set_xlabel('Time (min)')
    ax10_2.set_ylabel('Power Coefficient (Cp)')
    ax10_2.set_title('Propeller Power Coefficient')
    ax10_2.grid(True, alpha=0.3)
    ax10_2.legend()
    add_phase_boundaries(ax10_2, phase_boundaries, phase_names_list, time_vec)
    
    # Plot 3: Propeller Efficiency vs Time
    ax10_3 = axes10[2]
    if unit_eta_calc_data.size > 0:
        num_nacelles = unit_eta_calc_data.shape[0] if len(unit_eta_calc_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_eta_calc = unit_eta_calc_data[nac_idx, :]  # Row is nacelle, column is node
            # Ensure the length matches time_vec
            min_len = min(len(nac_eta_calc), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(nac_eta_calc[:min_len])
                if np.any(valid_mask):
                    ax10_3.plot(time_vec[:min_len][valid_mask], nac_eta_calc[:min_len][valid_mask], 
                            color=tab10_colors[nac_idx % len(tab10_colors)], 
                            linewidth=2, label=f'Nacelle {nac_idx+1} (calc)', linestyle='-')
    
    # Add bounded efficiency as dashed lines
    if unit_eta_data.size > 0:
        num_nacelles = unit_eta_data.shape[0] if len(unit_eta_data.shape) > 1 else 0
        for nac_idx in range(num_nacelles):
            nac_eta = unit_eta_data[nac_idx, :]  # Row is nacelle, column is node
            # Ensure the length matches time_vec
            min_len = min(len(nac_eta), len(time_vec))
            if min_len > 0:
                valid_mask = ~np.isnan(nac_eta[:min_len])
                if np.any(valid_mask):
                    ax10_3.plot(time_vec[:min_len][valid_mask], nac_eta[:min_len][valid_mask], 
                            color=tab10_colors[nac_idx % len(tab10_colors)], 
                            linewidth=2, label=f'Nacelle {nac_idx+1} (bounded)', linestyle='--', alpha=0.5)
    
    ax10_3.set_xlabel('Time (min)')
    ax10_3.set_ylabel('Efficiency')
    ax10_3.set_title('Propeller Efficiency')
    ax10_3.grid(True, alpha=0.3)
    ax10_3.legend()
    add_phase_boundaries(ax10_3, phase_boundaries, phase_names_list, time_vec)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0, w_pad=1.5)
    plt.show()

    # Create auxiliary electrical power figure
    fig11, ax11 = plt.subplots(1, 1, figsize=(12, 6))
    fig11.suptitle('Auxiliary Electrical Power', fontsize=16, y=0.98)

    # Plot p_aux_elec vs Time
    if p_aux_elec_data is not None and p_aux_elec_data.size > 0:
        # Ensure the length matches bat_time_vec
        min_len = min(len(p_aux_elec_data), len(bat_time_vec))
        if min_len > 0:
            valid_mask = ~np.isnan(p_aux_elec_data[:min_len])
            if np.any(valid_mask):
                ax11.plot(bat_time_vec[:min_len][valid_mask], p_aux_elec_data[:min_len][valid_mask],
                        color=tab10_colors[0], linewidth=2, label='Auxiliary Electrical Power', linestyle='-')

    ax11.set_xlabel('Time (min)')
    ax11.set_ylabel('Power (kW)')
    ax11.set_title('Auxiliary Electrical Power Profile')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    ax11.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    add_phase_boundaries(ax11, phase_boundaries, phase_names_list, time_vec)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Save plots and data if requested
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save plots to PDF
        # Note: output_filename already contains payload and config info, so we don't add it again
        # Add optimization objective to filename if provided
        if opt_objective is not None:
            pdf_filename = f"{output_filename_with_oew}_{opt_objective}_{timestamp}.pdf"
        else:
            pdf_filename = f"{output_filename_with_oew}_{timestamp}.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig2)
            if unit_em_power_data.size > 0:
                pdf.savefig(fig3)
            if unit_em_efficiency_data.size > 0:
                pdf.savefig(fig3_eff)
            if unit_gt_power_data.size > 0:
                pdf.savefig(fig4)
            pdf.savefig(fig5)
            if em_throttle_data.size > 0 or gt_throttle_data.size > 0:
                pdf.savefig(fig6)
            if nac_throttle_data.size > 0:
                pdf.savefig(fig7)
            pdf.savefig(fig8)
            pdf.savefig(fig9)
            if unit_adv_ratio_data.size > 0:
                pdf.savefig(fig10)
            if p_aux_elec_data is not None and p_aux_elec_data.size > 0:
                pdf.savefig(fig11)
            plt.close('all')
        
        # Save all plotted data to CSV (if requested)
        if save_csv:
            save_plot_data_to_csv(output_filename = output_filename_with_oew, timestamp = timestamp, time_vec = time_vec, 
                                 altitude_data = altitude_data, range_data = range_data, eas_data = eas_data, tas_data = tas_data, vs_data = vs_data, 
                                 cl_data = cl_data, cd_data = cd_data, cd0_data = cd0_data, cd_lift_data = cd_lift_data, lift_data = lift_data,
                                 unit_thrust_data = unit_thrust_data, total_thrust_data = total_thrust_data, drag_data = drag_data, weight_data = weight_data, 
                                 gamma_data = gamma_data, unit_gt_power_data = unit_gt_power_data, gt_throttle_data = gt_throttle_data, 
                                 unit_jet_thrust_data = unit_jet_thrust_data,
                                 unit_em_power_data = unit_em_power_data, em_throttle_data = em_throttle_data, total_em_power_data = total_em_power_data, 
                                 total_batt_power_data = total_batt_power_data, bat_soc_data = bat_soc_data, bat_voltage_data = bat_voltage_data, 
                                 total_gt_power_data = total_gt_power_data, unit_fuel_flow_data = unit_fuel_flow_data, total_fuel_flow_data = total_fuel_flow_data, 
                                 unit_shaft_power_data = unit_shaft_power_data, total_shaft_power_data = total_shaft_power_data, nac_throttle_data = nac_throttle_data,
                                 bat_vline_cell_raw_data = bat_vline_cell_raw_data, bat_p_cell_data = bat_p_cell_data, bat_i_bat_data = bat_i_bat_data, 
                                 bat_i_cell_raw_data = bat_i_cell_raw_data, bat_ocv_cell_data = bat_ocv_cell_data, phase_names_data = phase_names_data, duration_boundary_data = duration_boundary_data, 
                                 unit_adv_ratio_data = unit_adv_ratio_data, unit_cp_data = unit_cp_data, unit_eta_calc_data = unit_eta_calc_data, unit_eta_data = unit_eta_data, unit_em_efficiency_data = unit_em_efficiency_data, unit_prop_thrust_data = unit_prop_thrust_data, unit_prop_rpm_data = unit_prop_rpm_data,
                                 fuel_used_data = fuel_used_data, unit_gearbox_power_in_data = unit_gearbox_power_in_data, unit_gearbox_power_out_data = unit_gearbox_power_out_data, p_aux_elec_data = p_aux_elec_data, payload_qty = payload_qty, turb_op_type = turb_op_type, opt_objective = opt_objective)
        

    plt.close('all')

def save_spot_point_data_to_csv(output_filename = None, timestamp = None, node_vec = None,  
                               altitude_data = None, range_data = None, eas_data = None, tas_data = None, vs_data = None, 
                               cl_data = None, cd_data = None, cd0_data = None, cd_lift_data = None, lift_data = None,
                               unit_thrust_data = None, total_thrust_data = None, drag_data = None, weight_data = None, 
                               gamma_data = None, unit_gt_power_data = None, gt_throttle_data = None, 
                               unit_jet_thrust_data = None,
                               unit_em_power_data = None, em_throttle_data = None, total_em_power_data = None, 
                               total_gt_power_data = None, unit_fuel_flow_data = None, total_fuel_flow_data = None, 
                               unit_shaft_power_data = None, total_shaft_power_data = None, nac_throttle_data = None,
                               unit_adv_ratio_data = None, unit_prop_thrust_data = None, unit_prop_rpm_data = None, unit_cp_data = None, unit_eta_calc_data = None, unit_eta_data = None,  unit_ct_data = None,
                               unit_em_elec_power_data = None, total_em_elec_power_data = None, unit_em_efficiency_data = None,
                               unit_gearbox_power_in_data = None, unit_gearbox_power_out_data = None,
                               phase_names_data = None, payload_qty = None, turb_op_type = None, opt_objective = None):
    """
    Save all plotted spot point analysis data to CSV files for further analysis.
    
    Parameters
    ----------
    output_filename : str
        Base filename for output files
    timestamp : str
        Timestamp string for file naming
    node_vec : array
        Node vector (x-axis for spot point analysis)
    All other parameters are the data arrays from the plotting function
    """
    
    # Create main mission data DataFrame
    max_length = len(node_vec)
    
    # Helper function to safely pad 1D arrays
    def safe_pad_1d(data, target_length):
        """Safely pad a 1D array to target length, handling empty arrays and (n,1) shapes"""
        if data.size == 0:
            return np.full(target_length, np.nan)
        # Squeeze out any single-dimensional axes (e.g., (n,1) -> (n,))
        data = np.squeeze(data)
        # If it's still multi-dimensional after squeezing, handle matrices
        if len(data.shape) > 1:
            # For matrices (nodes, nacelles) or (nodes, num_strings), sum along axis 1
            # This gives total values across nacelles/strings
            data = data.sum(axis=1)
        
        # Handle integer arrays by converting to float first
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(float)
        
        return np.pad(data, (0, target_length - len(data)), constant_values=np.nan)
    
    # Helper function to safely pad lists (preserves string data)
    def safe_pad_list(data, target_length, fill_value=None):
        """Safely pad a list to target length, preserving data type"""
        if len(data) == 0:
            return [fill_value] * target_length
        # Pad with the fill value
        return data + [fill_value] * (target_length - len(data))
    
    # Prepare data dictionary for spot point analysis (only 1D arrays)
    spot_point_data = {
        'node': safe_pad_1d(node_vec, max_length),
        'phase_name': safe_pad_list(phase_names_data, max_length, fill_value=''),
        'altitude_ft': safe_pad_1d(altitude_data, max_length),
        'range_NM': safe_pad_1d(range_data, max_length),
        'eas_knots': safe_pad_1d(eas_data, max_length),
        'tas_knots': safe_pad_1d(tas_data, max_length),
        'vs_ft_min': safe_pad_1d(vs_data, max_length),
        'cl': safe_pad_1d(cl_data, max_length),
        'cd_total': safe_pad_1d(cd_data, max_length),
        'cd0': safe_pad_1d(cd0_data, max_length),
        'cd_lift': safe_pad_1d(cd_lift_data, max_length),
        'lift_N': safe_pad_1d(lift_data, max_length),
        'total_thrust_N': safe_pad_1d(total_thrust_data, max_length),
        'drag_N': safe_pad_1d(drag_data, max_length),
        'weight_kg': safe_pad_1d(weight_data, max_length),
        'gamma_percent': safe_pad_1d(gamma_data, max_length),
        'total_em_power_kW': safe_pad_1d(total_em_power_data, max_length),
        'total_gt_power_kW': safe_pad_1d(total_gt_power_data, max_length),
        'total_fuel_flow_kg_h': safe_pad_1d(total_fuel_flow_data, max_length),
        'total_shaft_power_kW': safe_pad_1d(total_shaft_power_data, max_length),
    }
    
    # Add individual nacelle thrust data
    if unit_thrust_data.size > 0 and len(unit_thrust_data.shape) > 1:
        num_nacelles = unit_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_thrust = unit_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_thrust_N'] = safe_pad_1d(nac_thrust, max_length)
    
    # Add individual nacelle shaft power data
    if unit_shaft_power_data.size > 0 and len(unit_shaft_power_data.shape) > 1:
        num_nacelles = unit_shaft_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_shaft_power = unit_shaft_power_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_shaft_power_kW'] = safe_pad_1d(nac_shaft_power, max_length)
    
    # Add individual nacelle throttle data
    if nac_throttle_data.size > 0 and len(nac_throttle_data.shape) > 1:
        num_nacelles = nac_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_throttle = nac_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_throttle'] = safe_pad_1d(nac_throttle, max_length)
    
    # Add individual nacelle GT power data
    if unit_gt_power_data.size > 0 and len(unit_gt_power_data.shape) > 1:
        num_nacelles = unit_gt_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_gt_power = unit_gt_power_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_gt_power_kW'] = safe_pad_1d(nac_gt_power, max_length)
    
    # Add individual nacelle GT throttle data
    if gt_throttle_data.size > 0 and len(gt_throttle_data.shape) > 1:
        num_nacelles = gt_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_gt_throttle = gt_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_gt_throttle'] = safe_pad_1d(nac_gt_throttle, max_length)
    
    # Add individual nacelle jet thrust data
    if unit_jet_thrust_data.size > 0 and len(unit_jet_thrust_data.shape) > 1:
        num_nacelles = unit_jet_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_jet_thrust = unit_jet_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_jet_thrust_N'] = safe_pad_1d(nac_jet_thrust, max_length)
    
    # Add individual nacelle EM power data
    if unit_em_power_data.size > 0 and len(unit_em_power_data.shape) > 1:
        num_nacelles = unit_em_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_power = unit_em_power_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_em_power_kW'] = safe_pad_1d(nac_em_power, max_length)
    
    # Add individual nacelle EM throttle data
    if em_throttle_data.size > 0 and len(em_throttle_data.shape) > 1:
        num_nacelles = em_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_throttle = em_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_em_throttle'] = safe_pad_1d(nac_em_throttle, max_length)
    
    # Add individual nacelle EM electrical power data
    if unit_em_elec_power_data.size > 0 and len(unit_em_elec_power_data.shape) > 1:
        num_nacelles = unit_em_elec_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_elec_power = unit_em_elec_power_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_em_elec_power_kW'] = safe_pad_1d(nac_em_elec_power, max_length)
    
    # Add individual nacelle EM efficiency data
    if unit_em_efficiency_data.size > 0 and len(unit_em_efficiency_data.shape) > 1:
        num_nacelles = unit_em_efficiency_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_efficiency = unit_em_efficiency_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_em_efficiency'] = safe_pad_1d(nac_em_efficiency, max_length)
    
    # Add individual nacelle fuel flow data
    if unit_fuel_flow_data.size > 0 and len(unit_fuel_flow_data.shape) > 1:
        num_nacelles = unit_fuel_flow_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_fuel_flow = unit_fuel_flow_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_fuel_flow_kg_h'] = safe_pad_1d(nac_fuel_flow, max_length)
    
    # Add individual propeller advance ratio data
    if unit_adv_ratio_data.size > 0 and len(unit_adv_ratio_data.shape) > 1:
        num_nacelles = unit_adv_ratio_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_adv_ratio = unit_adv_ratio_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_adv_ratio'] = safe_pad_1d(nac_adv_ratio, max_length)
    
    # Add individual propeller power coefficient data
    if unit_cp_data.size > 0 and len(unit_cp_data.shape) > 1:
        num_nacelles = unit_cp_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_cp = unit_cp_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_cp'] = safe_pad_1d(nac_cp, max_length)
    
    # Add individual propeller calculated efficiency data
    if unit_eta_calc_data.size > 0 and len(unit_eta_calc_data.shape) > 1:
        num_nacelles = unit_eta_calc_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_eta_calc = unit_eta_calc_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_eta_calc'] = safe_pad_1d(nac_eta_calc, max_length)
    
    # Add individual propeller bounded efficiency data
    if unit_eta_data.size > 0 and len(unit_eta_data.shape) > 1:
        num_nacelles = unit_eta_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_eta = unit_eta_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_eta'] = safe_pad_1d(nac_eta, max_length)
    
    # Add individual propeller thrust data
    if unit_prop_thrust_data.size > 0 and len(unit_prop_thrust_data.shape) > 1:
        num_nacelles = unit_prop_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_prop_thrust = unit_prop_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_prop_thrust_N'] = safe_pad_1d(nac_prop_thrust, max_length)
    
    # Add individual propeller RPM data
    if unit_prop_rpm_data.size > 0 and len(unit_prop_rpm_data.shape) > 1:
        num_nacelles = unit_prop_rpm_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_prop_rpm = unit_prop_rpm_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_prop_rpm'] = safe_pad_1d(nac_prop_rpm, max_length)
    
    # Add individual gearbox power in data
    if unit_gearbox_power_in_data is not None and unit_gearbox_power_in_data.size > 0 and len(unit_gearbox_power_in_data.shape) > 1:
        num_nacelles = unit_gearbox_power_in_data.shape[0]
        for nac_idx in range(num_nacelles):
            gearbox_power_in = unit_gearbox_power_in_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_gearbox_power_in_kW'] = safe_pad_1d(gearbox_power_in, max_length)
    
    # Add individual gearbox power out data
    if unit_gearbox_power_out_data is not None and unit_gearbox_power_out_data.size > 0 and len(unit_gearbox_power_out_data.shape) > 1:
        num_nacelles = unit_gearbox_power_out_data.shape[0]
        for nac_idx in range(num_nacelles):
            gearbox_power_out = unit_gearbox_power_out_data[nac_idx, :]  # Row is nacelle, column is node
            spot_point_data[f'nacelle{nac_idx+1}_gearbox_power_out_kW'] = safe_pad_1d(gearbox_power_out, max_length)
    
    # Create and save spot point analysis data CSV
    spot_point_df = pd.DataFrame(spot_point_data)
    # Add optimization objective to filename if provided
    if opt_objective is not None:
        spot_point_csv_filename = f"{output_filename}_{opt_objective}_spot_point_data_{timestamp}.csv"
    else:
        spot_point_csv_filename = f"{output_filename}_spot_point_data_{timestamp}.csv"
    spot_point_df.to_csv(spot_point_csv_filename, index=False)
    print(f"Spot point analysis data saved to: {spot_point_csv_filename}")


def save_plot_data_to_csv(output_filename=None, timestamp=None, time_vec=None, 
                         altitude_data=None, range_data=None, eas_data=None, tas_data=None, vs_data=None, 
                         cl_data=None, cd_data=None, cd0_data=None, cd_lift_data=None, lift_data=None,
                         unit_thrust_data=None, total_thrust_data=None, drag_data=None, weight_data=None, 
                         gamma_data=None, unit_gt_power_data=None, gt_throttle_data=None, 
                         unit_jet_thrust_data=None,
                         unit_em_power_data=None, em_throttle_data=None, total_em_power_data=None, 
                         total_batt_power_data=None, bat_soc_data=None, bat_voltage_data=None, 
                         total_gt_power_data=None, unit_fuel_flow_data=None, total_fuel_flow_data=None, 
                         unit_shaft_power_data=None, total_shaft_power_data=None, nac_throttle_data=None,
                         bat_vline_cell_raw_data=None, bat_p_cell_data=None, bat_i_bat_data=None, 
                         bat_i_cell_raw_data=None, bat_ocv_cell_data=None, phase_names_data=None, duration_boundary_data=None, 
                         unit_adv_ratio_data=None, unit_cp_data=None, unit_eta_calc_data=None, unit_eta_data=None, unit_em_efficiency_data=None, unit_prop_thrust_data=None, unit_prop_rpm_data=None,
                         fuel_used_data=None, unit_gearbox_power_in_data=None, unit_gearbox_power_out_data=None, p_aux_elec_data=None, payload_qty=None, turb_op_type=None, opt_objective=None):
    """
    Save all plotted data to CSV files for further analysis.
    
    Parameters
    ----------
    output_filename : str
        Base filename for output files
    timestamp : str
        Timestamp string for file naming
    All other parameters are the data arrays from the plotting function
    """
    
    # Create main mission data DataFrame
    max_length = len(time_vec)
    
    # Helper function to safely pad 1D arrays
    def safe_pad_1d(data, target_length):
        """Safely pad a 1D array to target length, handling empty arrays and (n,1) shapes"""
        if data.size == 0:
            return np.full(target_length, np.nan)
        # Squeeze out any single-dimensional axes (e.g., (n,1) -> (n,))
        data = np.squeeze(data)
        # If it's still multi-dimensional after squeezing, handle matrices
        if len(data.shape) > 1:
            # For matrices (nodes, nacelles) or (nodes, num_strings), sum along axis 1
            # This gives total values across nacelles/strings
            data = data.sum(axis=1)
        # Convert to float to allow NaN padding (NaN doesn't work with integer arrays)
        data = data.astype(float)
        return np.pad(data, (0, target_length - len(data)), constant_values=np.nan)
    
    # Helper function to safely pad lists (preserves string data)
    def safe_pad_list(data, target_length, fill_value=None):
        """Safely pad a list to target length, preserving data type"""
        if len(data) == 0:
            return [fill_value] * target_length
        # Pad with the fill value
        return data + [fill_value] * (target_length - len(data))
    
    # Prepare data dictionary for combined mission and battery data (only 1D arrays)
    mission_data = {
        'time_min': safe_pad_1d(time_vec, max_length),
        'phase_name': safe_pad_list(phase_names_data, max_length, fill_value=''),
        'altitude_ft': safe_pad_1d(altitude_data, max_length),
        'range_NM': safe_pad_1d(range_data, max_length),
        'eas_knots': safe_pad_1d(eas_data, max_length),
        'tas_knots': safe_pad_1d(tas_data, max_length),
        'vs_ft_min': safe_pad_1d(vs_data, max_length),
        'cl': safe_pad_1d(cl_data, max_length),
        'cd_total': safe_pad_1d(cd_data, max_length),
        'cd0': safe_pad_1d(cd0_data, max_length),
        'cd_lift': safe_pad_1d(cd_lift_data, max_length),
        'lift_N': safe_pad_1d(lift_data, max_length),
        'total_thrust_N': safe_pad_1d(total_thrust_data, max_length),
        'drag_N': safe_pad_1d(drag_data, max_length),
        'weight_kg': safe_pad_1d(weight_data, max_length),
        'gamma_percent': safe_pad_1d(gamma_data, max_length),
        'total_em_power_kW': safe_pad_1d(total_em_power_data, max_length),
        'total_gt_power_kW': safe_pad_1d(total_gt_power_data, max_length),
        'total_fuel_flow_kg_h': safe_pad_1d(total_fuel_flow_data, max_length),
        'fuel_used_kg': safe_pad_1d(fuel_used_data, max_length),
        'total_shaft_power_kW': safe_pad_1d(total_shaft_power_data, max_length),
        'phase_duration_min': safe_pad_1d(duration_boundary_data, max_length),
    }

    # Add auxiliary electrical power data
    if p_aux_elec_data is not None and p_aux_elec_data.size > 0:
        mission_data['p_aux_elec_kW'] = safe_pad_1d(p_aux_elec_data, max_length)

    # Add individual battery string data (shape: num_str, num_node)
    # Battery SOC data - per string
    if bat_soc_data.size > 0 and len(bat_soc_data.shape) > 1:
        num_strings = bat_soc_data.shape[0]
        for str_idx in range(num_strings):
            str_soc = bat_soc_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_soc'] = safe_pad_1d(str_soc, max_length)
    else:
        mission_data['soc'] = safe_pad_1d(bat_soc_data, max_length)
    
    # Battery voltage data - per string
    if bat_voltage_data.size > 0 and len(bat_voltage_data.shape) > 1:
        num_strings = bat_voltage_data.shape[0]
        for str_idx in range(num_strings):
            str_voltage = bat_voltage_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_pack_voltage_V'] = safe_pad_1d(str_voltage, max_length)
    else:
        mission_data['pack_voltage_V'] = safe_pad_1d(bat_voltage_data, max_length)
    
    # Battery current data - per string
    if bat_i_bat_data.size > 0 and len(bat_i_bat_data.shape) > 1:
        num_strings = bat_i_bat_data.shape[0]
        for str_idx in range(num_strings):
            str_current = bat_i_bat_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_pack_current_A'] = safe_pad_1d(str_current, max_length)
    else:
        mission_data['pack_current_A'] = safe_pad_1d(bat_i_bat_data, max_length)
    
    # Battery cell voltage data - per string
    if bat_vline_cell_raw_data.size > 0 and len(bat_vline_cell_raw_data.shape) > 1:
        num_strings = bat_vline_cell_raw_data.shape[0]
        for str_idx in range(num_strings):
            str_vline_cell = bat_vline_cell_raw_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_cell_voltage_V'] = safe_pad_1d(str_vline_cell, max_length)
    else:
        mission_data['cell_voltage_V'] = safe_pad_1d(bat_vline_cell_raw_data, max_length)
    
    # Battery cell power data - per string
    if bat_p_cell_data.size > 0 and len(bat_p_cell_data.shape) > 1:
        num_strings = bat_p_cell_data.shape[0]
        for str_idx in range(num_strings):
            str_p_cell = bat_p_cell_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_cell_power_W'] = safe_pad_1d(str_p_cell, max_length)
    else:
        mission_data['cell_power_W'] = safe_pad_1d(bat_p_cell_data, max_length)
    
    # Battery cell current data - per string
    if bat_i_cell_raw_data.size > 0 and len(bat_i_cell_raw_data.shape) > 1:
        num_strings = bat_i_cell_raw_data.shape[0]
        for str_idx in range(num_strings):
            str_i_cell = bat_i_cell_raw_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_cell_current_A'] = safe_pad_1d(str_i_cell, max_length)
    else:
        mission_data['cell_current_A'] = safe_pad_1d(bat_i_cell_raw_data, max_length)
    
    # Battery open circuit voltage data - per string
    if bat_ocv_cell_data is not None:
        if bat_ocv_cell_data.size > 0 and len(bat_ocv_cell_data.shape) > 1:
            num_strings = bat_ocv_cell_data.shape[0]
            for str_idx in range(num_strings):
                str_ocv_cell = bat_ocv_cell_data[str_idx, :]  # Row is string, column is node
                mission_data[f'string{str_idx+1}_cell_ocv_V'] = safe_pad_1d(str_ocv_cell, max_length)
        else:
            mission_data['cell_ocv_V'] = safe_pad_1d(bat_ocv_cell_data, max_length)

    # Battery power data - per string
    if total_batt_power_data.size > 0 and len(total_batt_power_data.shape) > 1:
        num_strings = total_batt_power_data.shape[0]
        for str_idx in range(num_strings):
            str_power = total_batt_power_data[str_idx, :]  # Row is string, column is node
            mission_data[f'string{str_idx+1}_batt_power_kW'] = safe_pad_1d(str_power, max_length)
    else:
        mission_data['total_batt_power_kW'] = safe_pad_1d(total_batt_power_data, max_length)
    
    # Add individual gearbox power data
    if unit_gearbox_power_in_data.size > 0 and len(unit_gearbox_power_in_data.shape) > 1:
        num_nacelles = unit_gearbox_power_in_data.shape[0]
        for nac_idx in range(num_nacelles):
            gearbox_power_in = unit_gearbox_power_in_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'gearbox_power_in_nacelle{nac_idx+1}_kW'] = safe_pad_1d(gearbox_power_in, max_length)
    
    if unit_gearbox_power_out_data.size > 0 and len(unit_gearbox_power_out_data.shape) > 1:
        num_nacelles = unit_gearbox_power_out_data.shape[0]
        for nac_idx in range(num_nacelles):
            gearbox_power_out = unit_gearbox_power_out_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'gearbox_power_out_nacelle{nac_idx+1}_kW'] = safe_pad_1d(gearbox_power_out, max_length)

    # Add individual nacelle EM efficiency data
    if unit_em_efficiency_data.size > 0 and len(unit_em_efficiency_data.shape) > 1:
        num_nacelles = unit_em_efficiency_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_efficiency = unit_em_efficiency_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_em_efficiency'] = safe_pad_1d(nac_em_efficiency, max_length)
    
    # Add individual nacelle thrust data
    if unit_thrust_data.size > 0 and len(unit_thrust_data.shape) > 1:
        num_nacelles = unit_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_thrust = unit_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_thrust_N'] = safe_pad_1d(nac_thrust, max_length)
    
    # Add individual nacelle shaft power data
    if unit_shaft_power_data.size > 0 and len(unit_shaft_power_data.shape) > 1:
        num_nacelles = unit_shaft_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_shaft_power = unit_shaft_power_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_shaft_power_kW'] = safe_pad_1d(nac_shaft_power, max_length)
    
    # Add individual nacelle throttle data
    if nac_throttle_data.size > 0 and len(nac_throttle_data.shape) > 1:
        num_nacelles = nac_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_throttle = nac_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_throttle'] = safe_pad_1d(nac_throttle, max_length)
    
    # Add individual nacelle GT power data
    if unit_gt_power_data.size > 0 and len(unit_gt_power_data.shape) > 1:
        num_nacelles = unit_gt_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_gt_power = unit_gt_power_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_gt_power_kW'] = safe_pad_1d(nac_gt_power, max_length)
    
    # Add individual nacelle GT throttle data
    if gt_throttle_data.size > 0 and len(gt_throttle_data.shape) > 1:
        num_nacelles = gt_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_gt_throttle = gt_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_gt_throttle'] = safe_pad_1d(nac_gt_throttle, max_length)
    
    # Add individual nacelle jet thrust data
    if unit_jet_thrust_data.size > 0 and len(unit_jet_thrust_data.shape) > 1:
        num_nacelles = unit_jet_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_jet_thrust = unit_jet_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_jet_thrust_N'] = safe_pad_1d(nac_jet_thrust, max_length)
    
    # Add individual nacelle EM power data
    if unit_em_power_data.size > 0 and len(unit_em_power_data.shape) > 1:
        num_nacelles = unit_em_power_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_power = unit_em_power_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_em_power_kW'] = safe_pad_1d(nac_em_power, max_length)
    
    # Add individual nacelle EM throttle data
    if em_throttle_data.size > 0 and len(em_throttle_data.shape) > 1:
        num_nacelles = em_throttle_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_em_throttle = em_throttle_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_em_throttle'] = safe_pad_1d(nac_em_throttle, max_length)
    
    # Add individual nacelle fuel flow data
    if unit_fuel_flow_data.size > 0 and len(unit_fuel_flow_data.shape) > 1:
        num_nacelles = unit_fuel_flow_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_fuel_flow = unit_fuel_flow_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_fuel_flow_kg_h'] = safe_pad_1d(nac_fuel_flow, max_length)
    
    # Add individual propeller advance ratio data
    if unit_adv_ratio_data.size > 0 and len(unit_adv_ratio_data.shape) > 1:
        num_nacelles = unit_adv_ratio_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_adv_ratio = unit_adv_ratio_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_adv_ratio'] = safe_pad_1d(nac_adv_ratio, max_length)
    
    # Add individual propeller power coefficient data
    if unit_cp_data.size > 0 and len(unit_cp_data.shape) > 1:
        num_nacelles = unit_cp_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_cp = unit_cp_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_cp'] = safe_pad_1d(nac_cp, max_length)
    
    # Add individual propeller calculated efficiency data
    if unit_eta_calc_data.size > 0 and len(unit_eta_calc_data.shape) > 1:
        num_nacelles = unit_eta_calc_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_eta_calc = unit_eta_calc_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_eta_calc'] = safe_pad_1d(nac_eta_calc, max_length)
    
    # Add individual propeller bounded efficiency data
    if unit_eta_data.size > 0 and len(unit_eta_data.shape) > 1:
        num_nacelles = unit_eta_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_eta = unit_eta_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_eta'] = safe_pad_1d(nac_eta, max_length)
    
    # Add individual propeller thrust data
    if unit_prop_thrust_data.size > 0 and len(unit_prop_thrust_data.shape) > 1:
        num_nacelles = unit_prop_thrust_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_prop_thrust = unit_prop_thrust_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_prop_thrust_N'] = safe_pad_1d(nac_prop_thrust, max_length)
    
    # Add individual propeller RPM data
    if unit_prop_rpm_data.size > 0 and len(unit_prop_rpm_data.shape) > 1:
        num_nacelles = unit_prop_rpm_data.shape[0]
        for nac_idx in range(num_nacelles):
            nac_prop_rpm = unit_prop_rpm_data[nac_idx, :]  # Row is nacelle, column is node
            mission_data[f'nacelle{nac_idx+1}_prop_rpm'] = safe_pad_1d(nac_prop_rpm, max_length)
    
    # Create and save combined mission and battery data CSV
    
    mission_df = pd.DataFrame(mission_data)
    # Note: output_filename already contains payload and config info, so we don't add it again
    # Add optimization objective to filename if provided
    if opt_objective is not None:
        mission_csv_filename = f"{output_filename}_{opt_objective}_combined_data_{timestamp}.csv"
    else:
        mission_csv_filename = f"{output_filename}_combined_data_{timestamp}.csv"
    mission_df.to_csv(mission_csv_filename, index=False)
    print(f"Combined mission and battery data saved to: {mission_csv_filename}")



def plot_trajectory_grid(
    cases,
    x_var,
    x_unit,
    y_vars,
    y_units,
    phases,
    x_label=None,
    y_labels=None,
    grid_layout=[5, 2],
    marker="o",
    savefig=None,
    figsize=None,
):
    """
    Plots multiple trajectories against each other
    Cases is a list of OpenMDAO CaseReader cases which act like OpenMDAO problems
    """
    x_vecs = []
    for case in cases:
        val_list = []
        for phase in phases:
            val_list.append(case.get_val(phase + "." + x_var, units=x_unit))
        x_vec = np.concatenate(val_list)
        x_vecs.append(x_vec)

    file_counter = -1
    counter_within_file = 0
    if figsize is None:
        figsize = (8.5, 11)
    for i, y_var in enumerate(y_vars):
        if counter_within_file % (grid_layout[0] * grid_layout[1]) == 0:
            if file_counter >= 0:
                # write the file
                if savefig is not None:
                    plt.savefig(savefig + "_" + str(file_counter) + ".pdf")
            fig, axs = plt.subplots(grid_layout[0], grid_layout[1], sharex=True, figsize=figsize)
            file_counter += 1
            counter_within_file = 0

        row_no = counter_within_file // grid_layout[1]
        col_no = counter_within_file % grid_layout[1]
        for j, case in enumerate(cases):
            val_list = []
            for phase in phases:
                val_list.append(case.get_val(phase + "." + y_var, units=y_units[i]))
            y_vec = np.concatenate(val_list)
            axs[row_no, col_no].plot(x_vecs[j], y_vec, marker)
        if row_no + 1 == grid_layout[0]:  # last row
            if x_label is None:
                axs[row_no, col_no].set(xlabel=x_var)
            else:
                axs[row_no, col_no].set(xlabel=x_label)
        if y_labels is not None:
            if y_labels[i] is not None:
                axs[row_no, col_no].set(ylabel=y_labels[i])
        else:
            axs[row_no, col_no].set(y_var)

        counter_within_file += 1

    if savefig is not None:
        fig.tight_layout()
        plt.savefig(savefig + "_" + str(file_counter) + ".pdf")


def plot_OAS_mesh(OAS_mesh, ax=None, set_xlim=True, turn_off_axis=True):
    """
    Plots the wing planform mesh from Atlas's OpenAeroStruct interface.

    Parameters
    ----------
    OAS_mesh : ndarray
        The mesh numpy array pulled out of the aerodynamics model (the output
        of the mesh component).
    ax : matplotlib axis object (optional)
        Axis on which to plot the wingbox. If not specified, this function
        creates and returns a figure and axis.
    set_xlim : bool (optional)
        Set the x limits on the axis to just fit the span of the wing.
    turn_off_axis : bool (optional)
        Turn off the spines, ticks, etc.

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        If ax is not specified, returns the created figure and axis.
    """
    # Duplicate the half mesh that is used by OAS
    mesh = np.hstack((OAS_mesh, OAS_mesh[:, -2::-1, :] * np.array([1, -1, 1])))
    chord_wing_front = mesh[0, :, 0]
    span_wing = mesh[0, :, 1]
    chord_wing_back = mesh[-1, :, 0]
    span = 2 * np.max(span_wing)

    return_ax = False
    if ax is None:
        # Figure out the size the plot should be
        y_range = abs(np.min(chord_wing_front) - np.max(chord_wing_back))
        x_size = 10
        y_size = y_range / span * x_size

        fig, ax = plt.subplots(figsize=(x_size, y_size))
        return_ax = True

    # Plot wing
    ax.fill_between(
        span_wing,
        -chord_wing_front,
        -chord_wing_back,
        facecolor="#d5e4f5",
        zorder=0,
        edgecolor="#919191",
        clip_on=False,
    )

    # Plot aerodynamic mesh
    x = mesh[:, :, 1]
    y = -mesh[:, :, 0]
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color="#919191", zorder=2, linewidth=0.3))
    ax.add_collection(LineCollection(segs2, color="#919191", zorder=2, linewidth=0.3))

    # Set final plot details
    ax.set_aspect("equal")
    if turn_off_axis:
        ax.set_axis_off()
    if set_xlim:
        ax.set_xlim((np.min(span_wing), np.max(span_wing)))

    if return_ax:
        return fig, ax


def plot_OAS_force_contours(
    OAS_mesh, panel_forces, ax=None, set_xlim=True, turn_off_axis=True, wing="both", force_dir=2, **contourf_kwargs
):
    """
    Plots contours of the force per area on the surface of the wing. The units are
    the force units of panel_forces divided by the square of the lenght units of OAS_mesh.

    Parameters
    ----------
    OAS_mesh : ndarray
        The mesh numpy array pulled out of the aerodynamics model (the output
        of the mesh component).
    panel_forces : ndarray
        Force from VLM for each panel (the panel_forces output of the VLM component).
    ax : matplotlib axis object (optional)
        Axis on which to plot the wingbox. If not specified, this function
        creates and returns a figure and axis.
    set_xlim : bool (optional)
        Set the x limits on the axis to just fit the span of the wing, by default True.
    turn_off_axis : bool (optional)
        Turn off the spines, ticks, etc., by default True.
    wing : str (optional)
        Which wing to plot, valid options are "left", "right", or "both", by default both.
    force_dir : int (optional)
        Force direction of which to plot contours. 0 is x force (drag direction),
        1 is y force (spanwise inward), and 2 is z force (lift direction).
    contourf_kwargs
        Any keyword arguments to pass to matplotlib's contourf function.

    Returns
    -------
    c : matplotlib QuadContourSet
        Return value from the call to contourf.
    fig, ax : matplotlib figure and axis objects
        If ax is not specified, returns the created figure and axis.
    """
    if wing not in ["left", "right", "both"]:
        raise ValueError(f'"{wing}" is not a valid value for wing, must be "left", "right", or "both"')
    if force_dir not in [0, 1, 2]:
        raise ValueError("force_dir must be either 0, 1, or 2")

    x_mesh = OAS_mesh[:, :, 0]
    y_mesh = OAS_mesh[:, :, 1]
    chord_wing_front = x_mesh[0, :]
    span_wing = y_mesh[0, :]
    chord_wing_back = x_mesh[-1, :]
    span = np.max(-span_wing)
    if wing == "both":
        span *= 2

    return_ax = False
    if ax is None:
        # Figure out the size the plot should be
        y_range = abs(np.min(chord_wing_front) - np.max(chord_wing_back))
        x_size = 10
        y_size = y_range / span * x_size

        fig, ax = plt.subplots(figsize=(x_size, y_size))
        return_ax = True

    # Compute the panel areas
    panel_front_widths = y_mesh[:-1, 1:] - y_mesh[:-1, :-1]
    panel_back_widths = y_mesh[1:, 1:] - y_mesh[1:, 1:]
    panel_avg_widths = 0.5 * (panel_front_widths + panel_back_widths)
    panel_left_side_lengths = x_mesh[1:, :-1] - x_mesh[:-1, :-1]
    panel_right_side_lengths = x_mesh[1:, 1:] - x_mesh[:-1, 1:]
    panel_avg_lengths = 0.5 * (panel_left_side_lengths + panel_right_side_lengths)
    panel_areas = panel_avg_widths * panel_avg_lengths

    # Force per area
    panel_pressures = panel_forces[:, :, force_dir] / panel_areas

    # Expand the panel forces to be at individual mesh points
    le_wt = 0.75 * 0.5
    te_wt = 0.25 * 0.5
    nodal_pressures = np.zeros_like(x_mesh)
    nodal_pressures[:-1, :-1] += panel_pressures * le_wt
    nodal_pressures[1:, :-1] += panel_pressures * te_wt
    nodal_pressures[1:, 1:] += panel_pressures * te_wt
    nodal_pressures[:-1, 1:] += panel_pressures * le_wt

    # Adjust the values at the nodes on the edges
    nodal_pressures[1:-1, [0, -1]] *= 2  # wing root and tip (but not corners)
    nodal_pressures[0, 1:-1] *= 0.5 / le_wt  # wing leading edge (not corners)
    nodal_pressures[-1, 1:-1] *= 0.5 / te_wt  # wing trailing edge (not corners)
    nodal_pressures[0, [0, -1]] *= 1 / le_wt  # leading edge corners
    nodal_pressures[-1, [0, -1]] *= 1 / te_wt  # trailing edge corners

    # Plot the contours
    if wing == "left":
        c = ax.contourf(y_mesh, -x_mesh, nodal_pressures, **contourf_kwargs)
    elif wing == "right":
        c = ax.contourf(-y_mesh, -x_mesh, nodal_pressures, **contourf_kwargs)
    else:
        y_mesh_merge = np.hstack((y_mesh, -y_mesh[:, ::-1]))
        x_mesh_merge = np.hstack((-x_mesh, -x_mesh[:, ::-1]))
        pressures_merge = np.hstack((nodal_pressures, nodal_pressures[:, ::-1]))
        c = ax.contourf(y_mesh_merge, x_mesh_merge, pressures_merge, **contourf_kwargs)

    # Set final plot details
    ax.set_aspect("equal")
    if turn_off_axis:
        ax.set_axis_off()
    if set_xlim:
        ax.set_xlim((np.min(span_wing), np.max(-span_wing)))

    if return_ax:
        return c, fig, ax
    return c


def plot_atlas_aviary_mission(
    prob,
    phases=("climb", "cruise", "descent"),
    save_plot=False,
    output_filename="atlas_aviary_mission_profile.png",
    show_plot=True,
):
    """
    Plot a compact mission summary for the Atlas-integrated Aviary run.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Solved Aviary problem object.
    phases : tuple[str]
        Ordered mission phases to stitch together.
    save_plot : bool
        If True, save the figure to ``output_filename``.
    output_filename : str
        Output image file when ``save_plot`` is True.
    show_plot : bool
        If True, display the figure via matplotlib.
    """

    def _concat(series_name, units=None):
        chunks = []
        for phase in phases:
            val = np.ravel(prob.get_val(f"traj.{phase}.timeseries.{series_name}", units=units))
            chunks.append(val)
        return np.concatenate(chunks)

    distance_nm = _concat("distance", units="NM")
    if np.ptp(distance_nm) < 1.0e-3:
        # In run_model-only workflows (no optimization), Dymos collocation states can
        # remain at initial guesses. Build an estimated mission distance from V(t).
        time_s = _concat("time", units="s")
        velocity_mps = _concat("velocity", units="m/s")
        dt = np.diff(time_s, prepend=time_s[0])
        dt = np.clip(dt, 0.0, None)
        est_distance_m = np.cumsum(velocity_mps * dt)
        x_data = est_distance_m / 1852.0
        x_label = "Estimated Distance (NM)"
    else:
        x_data = distance_nm
        x_label = "Distance (NM)"

    altitude_ft = _concat("altitude", units="ft")
    throttle = _concat("throttle", units="unitless")
    thrust_lbf = _concat("thrust_net_total", units="lbf")
    fuel_lbmph = _concat("fuel_flow_rate_negative_total", units="lbm/h")
    elec_kw = _concat("electric_power_in_total", units="kW")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Atlas Powertrain Aviary Mission")

    axes[0, 0].plot(x_data, altitude_ft, color="tab:blue")
    axes[0, 0].set_ylabel("Altitude (ft)")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_data, throttle, color="tab:orange")
    axes[0, 1].set_ylabel("Throttle (-)")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_data, thrust_lbf, color="tab:red")
    axes[1, 0].set_ylabel("Total Thrust (lbf)")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_data, -fuel_lbmph, color="tab:green", label="Fuel burn rate")
    ax2 = axes[1, 1].twinx()
    ax2.plot(x_data, elec_kw, color="tab:purple", label="Electric power")
    axes[1, 1].set_ylabel("Fuel Flow (lbm/h)")
    ax2.set_ylabel("Electric Power (kW)")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_plot:
        fig.savefig(output_filename, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig