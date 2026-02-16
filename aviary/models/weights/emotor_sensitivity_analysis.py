"""
Electric Motor Power Sensitivity Analysis

This script analyzes how electric motor rated power affects the weight of adjacent systems:
- Wing Weight (inertia relief from nacelles)
- Electric Engine
- Gearbox Weight  
- HV Wiring (HVWIS)
- Electrical Power Systems
- IPS TMS (Electric Motor + Turbine TMS)
- Nacelle Structure

Uses a stacked area chart to visualize the weight breakdown across motor power range.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from atlas.utils import DictIndepVarComp
from atlas.scenarios.setup_mission.load_ac_data import load_ac_data_from_excel
from atlas.scenarios.setup_mission.setup_ac_dvcomp import setup_ac_dv
from atlas.weights.compute_oew import OEWGroup
from atlas.weights.oew_analysis import connect_oew_inputs


def run_emotor_sensitivity(
    power_range_hp=(1200, 3000),
    n_points=20,
    plot_results=True,
    save_plot=True
):
    """
    Run electric motor power sensitivity analysis.
    
    Parameters
    ----------
    power_range_hp : tuple
        (min, max) electric motor power in horsepower per nacelle
    n_points : int
        Number of points to evaluate
    plot_results : bool
        Whether to display the plot
    save_plot : bool
        Whether to save the plot to file
        
    Returns
    -------
    dict
        Dictionary containing power values and weight arrays for each system
    """
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_dir = os.path.join(os.path.dirname(script_dir), 'scenarios', 'setup_mission')
    
    ac_filename = os.path.join(scenarios_dir, 'ac_data.xlsx')
    bat_filename = os.path.join(os.path.dirname(script_dir), 'propulsion', 'empirical_data',
                                'MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent.xlsx')
    cell_sheetname = 'BOL_cell_fct_CRate'
    config_sheetname = 'battery_config'
    
    # Load aircraft data from Excel
    print("Loading aircraft data from Excel...")
    ac_data = load_ac_data_from_excel(
        filename=ac_filename,
        bat_filename=bat_filename,
        cell_sheetname=cell_sheetname,
        config_sheetname=config_sheetname
    )

    # Add payload to ac_data (not in Excel file, needed by setup_ac_dv when compute_payload=False)
    ac_data['ac']['weights']['payload'] = {'value': 17100.0, 'units': 'lbm'}
    
    # Power range in kW (convert from hp)
    hp_to_kw = 0.7457
    power_values_hp = np.linspace(power_range_hp[0], power_range_hp[1], n_points)
    power_values_kw = power_values_hp * hp_to_kw
    
    # Storage for results
    results = {
        'power_hp': power_values_hp,
        'power_kw': power_values_kw,
        'wing_weight': np.zeros(n_points),
        'electric_engine_weight': np.zeros(n_points),
        'gearbox_weight': np.zeros(n_points),
        'hv_wiring_weight': np.zeros(n_points),
        'electrical_power_weight': np.zeros(n_points),
        'ips_tms_weight': np.zeros(n_points),
        'nacelle_weight': np.zeros(n_points),
        'oew': np.zeros(n_points),
    }
    
    print(f"\nRunning sensitivity analysis: {n_points} points from {power_range_hp[0]} to {power_range_hp[1]} HP")
    print("-" * 70)
    
    for i, (p_hp, p_kw) in enumerate(zip(power_values_hp, power_values_kw)):
        # Create OpenMDAO Problem for each power value
        prob = om.Problem(reports=False)
        model = prob.model
        
        # Add DictIndepVarComp with aircraft data
        dv_comp = DictIndepVarComp(ac_data)
        dv_comp = setup_ac_dv(dv_comp, compute_oew=True, compute_payload=False)
        model.add_subsystem('ac_data', dv_comp, promotes_outputs=['*'])
        
        # Add the OEW Group
        model.add_subsystem('oew', OEWGroup(), promotes_inputs=['*'], promotes_outputs=['*'])
        
        # Connect ac_data to OEW Group inputs
        connect_oew_inputs(model, compute_payload=False, compute_oew=True)
        
        # Setup
        prob.setup()
        
        # Override the motor power
        prob.set_val('ac|propulsion|motor|rating', p_kw, units='kW')
        
        # Run model
        prob.run_model()
        
        # Extract weights
        results['wing_weight'][i] = prob.get_val('wing_weight', units='lbm')[0]
        results['electric_engine_weight'][i] = prob.get_val('electric_motor_weight_per_nac', units='lbm')[0] * 4.0
        results['gearbox_weight'][i] = prob.get_val('gearbox_weight', units='lbm')[0] * 4.0
        results['hv_wiring_weight'][i] = prob.get_val('hvwis_weight', units='lbm')[0]
        results['electrical_power_weight'][i] = prob.get_val('electrical_power_system_weight', units='lbm')[0]
        
        # IPS TMS = Electric Motor TMS + Turbine TMS (per nacelle x 4)
        em_tms = prob.get_val('electric_motor_tms_weight_per_nac', units='lbm')[0] * 4.0
        turb_tms = prob.get_val('turbine_tms_weight_per_nac', units='lbm')[0] * 4.0
        results['ips_tms_weight'][i] = em_tms + turb_tms
        
        results['nacelle_weight'][i] = prob.get_val('total_nacelle_weight', units='lbm')[0]
        results['oew'][i] = prob.get_val('OEW', units='lbm')[0]
        
        # Print progress
        total_affected = (results['wing_weight'][i] + results['electric_engine_weight'][i] + 
                         results['gearbox_weight'][i] + results['hv_wiring_weight'][i] + 
                         results['electrical_power_weight'][i] + results['ips_tms_weight'][i] + 
                         results['nacelle_weight'][i])
        
        print(f"  {i+1:3d}/{n_points}: P={p_hp:7.1f} HP | "
              f"Wing={results['wing_weight'][i]:7.1f} | "
              f"E-Motor={results['electric_engine_weight'][i]:7.1f} | "
              f"Gearbox={results['gearbox_weight'][i]:7.1f} | "
              f"Total Affected={total_affected:8.1f} lbm")
        
        # Cleanup
        prob.cleanup()
    
    print("-" * 70)
    print("Sensitivity analysis complete.\n")
    
    # Create stacked area plot
    if plot_results or save_plot:
        create_stacked_plot(results, save_plot=save_plot)
    
    return results


def create_stacked_plot(results, save_plot=True):
    """
    Create stacked area chart showing weight breakdown vs motor power.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_emotor_sensitivity
    save_plot : bool
        Whether to save the plot to file
    """
    # Set up the figure with a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    power_hp = results['power_hp']
    
    # Define the systems (bottom to top in stack) - use default tab colors
    systems = [
        ('Wing Weight', results['wing_weight']),
        ('Electric Engine', results['electric_engine_weight']),
        ('Gearbox', results['gearbox_weight']),
        ('HV Wiring (HVWIS)', results['hv_wiring_weight']),
        ('Electrical Power System', results['electrical_power_weight']),
        ('IPS TMS', results['ips_tms_weight']),
        ('Nacelle Structure', results['nacelle_weight']),
    ]
    
    # Create stacked data
    y_stack = np.zeros((len(systems), len(power_hp)))
    labels = []
    
    for i, (name, data) in enumerate(systems):
        y_stack[i] = data
        labels.append(name)
    
    # Create stacked area plot (uses default tab colors with alpha)
    ax.stackplot(power_hp, y_stack, labels=labels, alpha=0.7)
    
    # Calculate total affected weight for annotation
    total_affected = np.sum(y_stack, axis=0)
    
    # Add total line on top
    ax.plot(power_hp, total_affected, 'k-', linewidth=2.5, label='Total Affected Weight')
    
    # Interpolate to find weights at reference points
    weight_at_2500 = np.interp(2500, power_hp, total_affected)
    weight_at_1967 = np.interp(1967, power_hp, total_affected)
    
    # Add vertical reference lines at 2500 HP and 1967 HP (thick black)
    ax.axvline(x=2500, color='black', linestyle='-', linewidth=3, alpha=0.9)
    ax.axvline(x=1967, color='black', linestyle='-', linewidth=3, alpha=0.9)
    
    # Add horizontal lines from reference points to y-axis
    ax.hlines(y=weight_at_2500, xmin=power_hp[0], xmax=2500, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.hlines(y=weight_at_1967, xmin=power_hp[0], xmax=1967, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add text labels for weights at reference points
    ax.text(power_hp[0] - 30, weight_at_2500, f'{weight_at_2500:.0f} lbm\n(2500 HP)', 
            fontsize=9, fontweight='bold', ha='right', va='center')
    ax.text(power_hp[0] - 30, weight_at_1967, f'{weight_at_1967:.0f} lbm\n(1967 HP)', 
            fontsize=9, fontweight='bold', ha='right', va='center')
    
    # Formatting
    ax.set_xlabel('Electric Motor Power (HP per nacelle)', fontsize=12, fontweight='bold')
    ax.set_ylabel('E-Motor Adjacent Systems Weight (lbm)', fontsize=12, fontweight='bold')
    ax.set_title('Electric Motor Power Sensitivity Analysis\nImpact on Adjacent System Weights', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits (y-axis starts at 6000 lbm, extend x-axis left for labels)
    ax.set_xlim(power_hp[0] - 150, power_hp[-1])
    ax.set_ylim(6000, total_affected.max() * 1.05)
    
    # Legend - place outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10,
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Calculate sensitivity (lbm per 100 HP and lbm per 100 kW)
    power_delta_hp = power_hp[-1] - power_hp[0]
    power_delta_kw = power_delta_hp * 0.7457
    delta_total = total_affected[-1] - total_affected[0]
    
    sensitivity_per_100hp = (delta_total / power_delta_hp) * 100
    sensitivity_per_100kw = (delta_total / power_delta_kw) * 100
    
    # Text box with sensitivity ratio only (bottom right)
    textstr = (f'Sensitivity:\n'
               f'  {sensitivity_per_100hp:+.1f} lbm/100HP\n'
               f'  {sensitivity_per_100kw:+.1f} lbm/100kW')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, fontfamily='monospace')
    
    # Calculate weight breakdown for each system at 1967 HP and 2500 HP
    wing_1967 = np.interp(1967, power_hp, results['wing_weight'])
    wing_2500 = np.interp(2500, power_hp, results['wing_weight'])
    emotor_1967 = np.interp(1967, power_hp, results['electric_engine_weight'])
    emotor_2500 = np.interp(2500, power_hp, results['electric_engine_weight'])
    gearbox_1967 = np.interp(1967, power_hp, results['gearbox_weight'])
    gearbox_2500 = np.interp(2500, power_hp, results['gearbox_weight'])
    hvwire_1967 = np.interp(1967, power_hp, results['hv_wiring_weight'])
    hvwire_2500 = np.interp(2500, power_hp, results['hv_wiring_weight'])
    elecpwr_1967 = np.interp(1967, power_hp, results['electrical_power_weight'])
    elecpwr_2500 = np.interp(2500, power_hp, results['electrical_power_weight'])
    tms_1967 = np.interp(1967, power_hp, results['ips_tms_weight'])
    tms_2500 = np.interp(2500, power_hp, results['ips_tms_weight'])
    nacelle_1967 = np.interp(1967, power_hp, results['nacelle_weight'])
    nacelle_2500 = np.interp(2500, power_hp, results['nacelle_weight'])
    
    # Build breakdown text (right side)
    delta_wing = wing_2500 - wing_1967
    delta_emotor = emotor_2500 - emotor_1967
    delta_gearbox = gearbox_2500 - gearbox_1967
    delta_hvwire = hvwire_2500 - hvwire_1967
    delta_elecpwr = elecpwr_2500 - elecpwr_1967
    delta_tms = tms_2500 - tms_1967
    delta_nacelle = nacelle_2500 - nacelle_1967
    delta_total_ref = weight_at_2500 - weight_at_1967
    
    breakdown_str = (f'Weight Δ (1967→2500 HP):\n'
                     f'{"─"*30}\n'
                     f'Wing:            {delta_wing:+7.1f} lbm\n'
                     f'E-Motor:         {delta_emotor:+7.1f} lbm\n'
                     f'Gearbox:         {delta_gearbox:+7.1f} lbm\n'
                     f'HV Wiring:       {delta_hvwire:+7.1f} lbm\n'
                     f'E Power Systems: {delta_elecpwr:+7.1f} lbm\n'
                     f'IPS TMS:         {delta_tms:+7.1f} lbm\n'
                     f'Nacelle Struct:  {delta_nacelle:+7.1f} lbm\n'
                     f'{"─"*30}\n'
                     f'TOTAL:       {delta_total_ref:+7.1f} lbm')
    
    props2 = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.95, edgecolor='gray')
    ax.text(1.02, 0.5, breakdown_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='left', bbox=props2, fontfamily='monospace')
    
    plt.tight_layout()
    
    # ==================== SECOND FIGURE: Bar comparison ====================
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    system_names = ['Wing', 'E-Motor', 'Gearbox', 'HV Wiring', 'E Power Sys', 'IPS TMS', 'Nacelle Struct']
    weights_1967 = [wing_1967, emotor_1967, gearbox_1967, hvwire_1967, elecpwr_1967, tms_1967, nacelle_1967]
    weights_2500 = [wing_2500, emotor_2500, gearbox_2500, hvwire_2500, elecpwr_2500, tms_2500, nacelle_2500]
    
    x = np.arange(len(system_names))
    bar_width = 0.35
    
    bars1 = ax2.bar(x - bar_width/2, weights_1967, bar_width, label='1967 HP', color='tab:blue', alpha=0.8)
    bars2 = ax2.bar(x + bar_width/2, weights_2500, bar_width, label='2500 HP', color='tab:orange', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('System', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight (lbm)', fontsize=12, fontweight='bold')
    ax2.set_title('System Weight Comparison: 1967 HP vs 2500 HP', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_names, rotation=15, ha='right')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_plot:
        # Save to results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), 
                                   'aircraft_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save first figure (stacked area chart)
        output_path = os.path.join(results_dir, 'emotor_power_sensitivity.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {output_path}")
        
        # Save second figure (bar chart)
        output_path2 = os.path.join(results_dir, 'emotor_power_comparison.png')
        fig2.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Comparison chart saved to: {output_path2}")
        
        # Also save CSV with data
        csv_path = os.path.join(results_dir, 'emotor_power_sensitivity.csv')
        with open(csv_path, 'w') as f:
            f.write("Power_HP,Power_kW,Wing_lbm,E_Motor_lbm,Gearbox_lbm,HV_Wiring_lbm,"
                    "Elec_Power_lbm,IPS_TMS_lbm,Nacelle_lbm,Total_Affected_lbm,OEW_lbm\n")
            for i in range(len(power_hp)):
                total = (results['wing_weight'][i] + results['electric_engine_weight'][i] +
                        results['gearbox_weight'][i] + results['hv_wiring_weight'][i] +
                        results['electrical_power_weight'][i] + results['ips_tms_weight'][i] +
                        results['nacelle_weight'][i])
                f.write(f"{results['power_hp'][i]:.1f},{results['power_kw'][i]:.2f},"
                       f"{results['wing_weight'][i]:.2f},{results['electric_engine_weight'][i]:.2f},"
                       f"{results['gearbox_weight'][i]:.2f},{results['hv_wiring_weight'][i]:.2f},"
                       f"{results['electrical_power_weight'][i]:.2f},{results['ips_tms_weight'][i]:.2f},"
                       f"{results['nacelle_weight'][i]:.2f},{total:.2f},{results['oew'][i]:.2f}\n")
        print(f"Data saved to: {csv_path}")
    
    plt.show()
    
    return fig, fig2


def print_sensitivity_table(results):
    """Print a formatted table of the sensitivity results."""
    print("\n" + "=" * 110)
    print("ELECTRIC MOTOR POWER SENSITIVITY ANALYSIS - DETAILED RESULTS")
    print("=" * 110)
    print(f"{'Power':>8} {'Wing':>10} {'E-Motor':>10} {'Gearbox':>10} {'HV Wire':>10} "
          f"{'Elec Pwr':>10} {'IPS TMS':>10} {'Nacelle':>10} {'Total':>10}")
    print(f"{'(HP)':>8} {'(lbm)':>10} {'(lbm)':>10} {'(lbm)':>10} {'(lbm)':>10} "
          f"{'(lbm)':>10} {'(lbm)':>10} {'(lbm)':>10} {'(lbm)':>10}")
    print("-" * 110)
    
    for i in range(len(results['power_hp'])):
        total = (results['wing_weight'][i] + results['electric_engine_weight'][i] +
                results['gearbox_weight'][i] + results['hv_wiring_weight'][i] +
                results['electrical_power_weight'][i] + results['ips_tms_weight'][i] +
                results['nacelle_weight'][i])
        
        print(f"{results['power_hp'][i]:>8.0f} {results['wing_weight'][i]:>10.1f} "
              f"{results['electric_engine_weight'][i]:>10.1f} {results['gearbox_weight'][i]:>10.1f} "
              f"{results['hv_wiring_weight'][i]:>10.1f} {results['electrical_power_weight'][i]:>10.1f} "
              f"{results['ips_tms_weight'][i]:>10.1f} {results['nacelle_weight'][i]:>10.1f} "
              f"{total:>10.1f}")
    
    print("=" * 110)
    
    # Print sensitivity summary
    print("\nSENSITIVITY SUMMARY (per 100 HP increase):")
    print("-" * 60)
    
    power_delta = results['power_hp'][-1] - results['power_hp'][0]
    scale = 100.0 / power_delta  # Scale to per 100 HP
    
    systems = [
        ('Wing Weight', results['wing_weight']),
        ('Electric Engine', results['electric_engine_weight']),
        ('Gearbox', results['gearbox_weight']),
        ('HV Wiring', results['hv_wiring_weight']),
        ('Electrical Power System', results['electrical_power_weight']),
        ('IPS TMS', results['ips_tms_weight']),
        ('Nacelle Structure', results['nacelle_weight']),
    ]
    
    total_sensitivity = 0
    for name, data in systems:
        delta = (data[-1] - data[0]) * scale
        total_sensitivity += delta
        print(f"  {name:<25}: {delta:+7.2f} lbm/100HP")
    
    print("-" * 60)
    print(f"  {'TOTAL AFFECTED':<25}: {total_sensitivity:+7.2f} lbm/100HP")
    print("=" * 60)


if __name__ == '__main__':
    # Run sensitivity analysis
    results = run_emotor_sensitivity(
        power_range_hp=(1200, 3000),  # HP per nacelle
        n_points=15,
        plot_results=True,
        save_plot=True
    )
    
    # Print detailed table
    print_sensitivity_table(results)

