"""
Q400 Mission Visualization

Simplified plotting functions for the Q400 turboprop aircraft.
Excludes battery and hybrid powertrain variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Standard matplotlib color cycle
COLORS = {
    'altitude': 'tab:blue',
    'range': 'tab:green',
    'cl': 'tab:blue',
    'cd': 'tab:red',
    'lift': 'tab:blue',
    'drag': 'tab:orange',
    'fuel_flow': 'tab:green',
    'fuel_used': 'tab:green',
    'thrust': 'tab:purple',
    'tas': 'tab:blue',
    'eas': 'tab:cyan',
    'weight': 'tab:green',
}

# Validation data from reference Q400 mission
# Range (NM), Fuel Flow per Engine (lb/h), and Cumulative Fuel Consumption (lb)
VALIDATION_RANGE_NM = np.array([
    0, 2, 5, 8, 10, 14, 17, 20, 24, 29, 34, 41, 49, 49,
    83.43943035, 117.8788607, 152.318291, 186.7577214, 221.1971517, 255.6365821,
    290.0760124, 324.5154428, 358.9548731, 393.3943035, 427.8337338, 427.8337338,
    439.8337338, 449.8337338, 455.8337338, 461.8337338, 467.8337338, 473.8337338,
    478.8337338, 483.8337338, 487.8337338, 492.8337338, 496.8337338, 500.8337338
])

VALIDATION_FUEL_FLOW_LBH = np.array([
    1966.666667, 1966.666667, 1966.666667, 1770, 2000, 1770, 1830, 1718.181818,
    1625, 1478.571429, 1460, 1350, 1350, 939.5, 939.5, 939.5, 939.5, 939.5,
    939.5, 939.5, 939.5, 939.5, 939.5, 939.5, 939.5, 885, 885, 970.5882353,
    840, 810, 810, 870, 870, 750, 540, 570, 540, 570
])

VALIDATION_FUEL_USED_LB = np.array([
    0, 59, 118, 177, 237, 296, 357, 420, 485, 554, 627, 708, 802.5, 802.5,
    1014.827, 1227.154, 1439.481, 1651.808, 1864.135, 2076.462, 2288.789,
    2501.116, 2713.443, 2925.77, 3138.097, 3138.097, 3167.597, 3195.097,
    3209.097, 3222.597, 3236.097, 3250.597, 3265.097, 3277.597, 3286.597,
    3296.097, 3305.097, 3314.597
])



def plot_q400_mission(prob, mission_config, save_plots=False, output_filename=None, 
                      payload_qty=None, nn=11):
    """
    Plot Q400 mission results - simplified for conventional turboprop.
    
    Shows:
    - Kinematics: altitude, range, time
    - Aerodynamics: CL, CD, lift, drag
    - Fuel: fuel flow, fuel used
    - Propulsion: total thrust
    
    Parameters
    ----------
    prob : OpenMDAO Problem
        Solved problem instance
    mission_config : dict
        Mission configuration dictionary
    save_plots : bool
        Whether to save plots to PDF
    output_filename : str
        Base filename for saving
    payload_qty : float
        Payload weight for title
    nn : int
        Number of nodes per phase
    """
    
    # Get phase names (exclude mission and reserve metadata)
    phase_names = [p for p in mission_config['mission']['phase_names'] 
                   if p not in ['mission', 'reserve']]
    
    # Create figure with subplots (5 rows x 2 cols = 10 plots)
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle(f'Q400 Mission Analysis', fontsize=14, fontweight='bold')
    
    # Initialize arrays to collect data across phases
    all_time = []
    all_range = []
    all_altitude = []
    all_cl = []
    all_cd = []
    all_lift = []
    all_drag = []
    all_fuel_flow = []
    all_fuel_used = []
    all_thrust = []
    all_weight = []
    all_tas = []
    all_eas = []
    all_torque_percent = []
    all_unit_fuel_flow = []
    all_rpm = []
    all_unit_shaft_power = []
    
    # PW150 max torque for shaft power calculation
    MAX_TORQUE_NM = 35404
    
    time_offset = 0
    
    for phase_name in phase_names:
        try:
            # Helper to safely get and flatten values
            def safe_get(path, units=None, default_val=np.nan):
                try:
                    val = prob.get_val(path, units=units) if units else prob.get_val(path)
                    # Handle matrix outputs (num_props, nn) by summing or taking first row
                    if val.ndim > 1:
                        return val.flatten()[:nn]  # Take first nn elements after flatten
                    return val.flatten()
                except Exception as e:
                    print(f"  Could not get {path}: {e}")
                    return np.full(nn, default_val)
            
            # Time - duration is promoted to phase.duration
            try:
                duration = prob.get_val(f'{phase_name}.duration', units='min')
                if hasattr(duration, '__len__'):
                    duration = duration[0]
            except:
                duration = prob.get_val(f'{phase_name}.ode_integ_phase.duration', units='min')[0]
            time_phase = np.linspace(0, float(duration), nn) + time_offset
            all_time.extend(time_phase)
            time_offset += float(duration)
            
            # Altitude - shape (nn,)
            altitude = safe_get(f'{phase_name}.fltcond|h', 'ft')
            all_altitude.extend(altitude)
            
            # Range - promoted to phase.range
            range_nm = safe_get(f'{phase_name}.range', 'NM')
            all_range.extend(range_nm)
            
            # Aerodynamics - shape (nn,)
            cl = safe_get(f'{phase_name}.cl')
            all_cl.extend(cl)
            
            cd = safe_get(f'{phase_name}.cd_total')
            all_cd.extend(cd)
            
            lift = safe_get(f'{phase_name}.lift', 'kN')
            all_lift.extend(lift)
            
            drag = safe_get(f'{phase_name}.drag', 'kN')
            all_drag.extend(drag)
            
            # Weight - shape (nn,)
            weight = safe_get(f'{phase_name}.weight', 'lbm')
            all_weight.extend(weight)
            
            # TAS - shape (nn,)
            tas = safe_get(f'{phase_name}.fltcond|Utrue', 'kn')
            all_tas.extend(tas)
            
            # EAS - shape (nn,)
            eas = safe_get(f'{phase_name}.fltcond|Ueas', 'kn')
            all_eas.extend(eas)
            
            # Fuel - shape (nn,)
            fuel_flow = safe_get(f'{phase_name}.total_fuel_flow', 'lbm/h')
            all_fuel_flow.extend(fuel_flow)
            
            fuel_used = safe_get(f'{phase_name}.fuel_used', 'lbm')
            all_fuel_used.extend(fuel_used)
            
            # Thrust - from Q400 powertrain
            # unit_thrust shape is (num_props, nn), sum across props (axis=0)
            try:
                unit_thrust = prob.get_val(f'{phase_name}.unit_thrust', units='kN')
                if unit_thrust.ndim > 1:
                    total_thrust = np.sum(unit_thrust, axis=0).flatten()
                else:
                    total_thrust = unit_thrust.flatten()
                all_thrust.extend(total_thrust)
            except Exception as e:
                print(f"  Could not get thrust for {phase_name}: {e}")
                all_thrust.extend([np.nan] * nn)
            
            # Torque percent - shape (num_props, nn), take mean across props
            try:
                torque_pct = prob.get_val(f'{phase_name}.torque_percent')
                if torque_pct.ndim > 1:
                    torque_pct_mean = np.mean(torque_pct, axis=0).flatten()
                else:
                    torque_pct_mean = torque_pct.flatten()
                all_torque_percent.extend(torque_pct_mean * 100)  # Convert to percentage
            except Exception as e:
                print(f"  Could not get torque_percent for {phase_name}: {e}")
                all_torque_percent.extend([np.nan] * nn)
            
            # Unit fuel flow per engine - shape (num_props, nn), take mean across engines
            try:
                unit_ff = prob.get_val(f'{phase_name}.unit_fuel_flow', units='lbm/h')
                if unit_ff.ndim > 1:
                    # Take first engine's fuel flow as representative
                    unit_ff_plot = unit_ff[0, :].flatten()
                else:
                    unit_ff_plot = unit_ff.flatten()
                all_unit_fuel_flow.extend(unit_ff_plot)
            except Exception as e:
                print(f"  Could not get unit_fuel_flow for {phase_name}: {e}")
                all_unit_fuel_flow.extend([np.nan] * nn)
            
            # RPM - shape (num_props, nn), take mean across props
            try:
                rpm = prob.get_val(f'{phase_name}.rpm', units='rpm')
                if rpm.ndim > 1:
                    rpm_mean = np.mean(rpm, axis=0).flatten()
                else:
                    rpm_mean = rpm.flatten()
                all_rpm.extend(rpm_mean)
            except Exception as e:
                print(f"  Could not get rpm for {phase_name}: {e}")
                all_rpm.extend([np.nan] * nn)
            
            # Unit shaft power calculation: P = omega * torque = (rpm / 60 * 2 * pi) * (torque_pct/100 * max_torque)
            # Result in kW
            try:
                rpm_vals = prob.get_val(f'{phase_name}.rpm', units='rpm')
                torque_pct_vals = prob.get_val(f'{phase_name}.torque_percent')
                if rpm_vals.ndim > 1:
                    rpm_mean = np.mean(rpm_vals, axis=0).flatten()
                else:
                    rpm_mean = rpm_vals.flatten()
                if torque_pct_vals.ndim > 1:
                    torque_pct_mean = np.mean(torque_pct_vals, axis=0).flatten()
                else:
                    torque_pct_mean = torque_pct_vals.flatten()
                # omega = rpm / 60 * 2 * pi (rad/s)
                # torque = torque_pct * max_torque (Nm)
                # power = omega * torque (W) -> / 1000 for kW
                omega = rpm_mean / 60 * 2 * np.pi
                torque_nm = torque_pct_mean * MAX_TORQUE_NM
                unit_shaft_power_kw = omega * torque_nm / 1000
                all_unit_shaft_power.extend(unit_shaft_power_kw)
            except Exception as e:
                print(f"  Could not compute unit_shaft_power for {phase_name}: {e}")
                all_unit_shaft_power.extend([np.nan] * nn)
                
        except Exception as e:
            print(f"Error processing phase {phase_name}: {e}")
            # Add NaN placeholders
            all_time.extend([np.nan] * nn)
            all_altitude.extend([np.nan] * nn)
            all_range.extend([np.nan] * nn)
            all_cl.extend([np.nan] * nn)
            all_cd.extend([np.nan] * nn)
            all_lift.extend([np.nan] * nn)
            all_drag.extend([np.nan] * nn)
            all_fuel_flow.extend([np.nan] * nn)
            all_fuel_used.extend([np.nan] * nn)
            all_thrust.extend([np.nan] * nn)
            all_weight.extend([np.nan] * nn)
            all_tas.extend([np.nan] * nn)
            all_eas.extend([np.nan] * nn)
            all_torque_percent.extend([np.nan] * nn)
            all_unit_fuel_flow.extend([np.nan] * nn)
            all_rpm.extend([np.nan] * nn)
            all_unit_shaft_power.extend([np.nan] * nn)
    
    # Convert to arrays
    all_time = np.array(all_time)
    all_range = np.array(all_range)
    all_altitude = np.array(all_altitude)
    all_cl = np.array(all_cl)
    all_cd = np.array(all_cd)
    all_lift = np.array(all_lift)
    all_drag = np.array(all_drag)
    all_fuel_flow = np.array(all_fuel_flow)
    all_fuel_used = np.array(all_fuel_used)
    all_thrust = np.array(all_thrust)
    all_weight = np.array(all_weight)
    all_tas = np.array(all_tas)
    all_eas = np.array(all_eas)
    all_torque_percent = np.array(all_torque_percent)
    all_unit_fuel_flow = np.array(all_unit_fuel_flow)
    all_rpm = np.array(all_rpm)
    all_unit_shaft_power = np.array(all_unit_shaft_power)
    
    # Plot 1: Altitude vs Range
    ax = axes[0, 0]
    ax.plot(all_range, all_altitude, color=COLORS['altitude'], linewidth=2)
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Altitude (ft)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 2: TAS and EAS vs Range
    ax = axes[0, 1]
    ax.plot(all_range, all_tas, color=COLORS['tas'], linewidth=2, label='TAS')
    ax.plot(all_range, all_eas, color=COLORS['eas'], linewidth=2, label='EAS')
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Airspeed (kn)')
    ax.set_title('True & Equivalent Airspeed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: CL and CD vs Range
    ax = axes[1, 0]
    ax.plot(all_range, all_cl, color=COLORS['cl'], linewidth=2, label='CL')
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('CL', color=COLORS['cl'])
    ax.tick_params(axis='y', labelcolor=COLORS['cl'])
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(all_range, all_cd, color=COLORS['cd'], linewidth=2, label='CD')
    ax2.set_ylabel('CD', color=COLORS['cd'])
    ax2.tick_params(axis='y', labelcolor=COLORS['cd'])
    ax2.legend(loc='upper right')
    ax.set_title('Lift & Drag Coefficients')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Lift and Drag vs Range
    ax = axes[1, 1]
    ax.plot(all_range, all_lift, color=COLORS['lift'], linewidth=2, label='Lift')
    ax.plot(all_range, all_drag, color=COLORS['drag'], linewidth=2, label='Drag')
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Force (kN)')
    ax.set_title('Lift & Drag Forces')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Fuel Flow per Engine vs Range (with validation data)
    ax = axes[2, 0]
    ax.plot(all_range, all_unit_fuel_flow, color='tab:red', linewidth=2, label='Computed')
    ax.plot(VALIDATION_RANGE_NM, VALIDATION_FUEL_FLOW_LBH, color='black', linewidth=2, 
           linestyle='--', label='Q400 AOM')
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Fuel Flow per Engine (lbm/h)')
    ax.set_title('Fuel Flow per Engine')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 6: Fuel Used vs Range (with validation data)
    ax = axes[2, 1]
    ax.plot(all_range, all_fuel_used, color=COLORS['fuel_used'], linewidth=2, label='Computed')
    ax.plot(VALIDATION_RANGE_NM, VALIDATION_FUEL_USED_LB, color='black', linewidth=2, 
           linestyle='--', label='Q400 AOM')
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Fuel Used (lbm)')
    ax.set_title('Cumulative Fuel Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 7: Total Thrust vs Range
    ax = axes[3, 0]
    ax.plot(all_range, all_thrust, color=COLORS['thrust'], linewidth=2)
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Total Thrust (kN)')
    ax.set_title('Total Thrust (All Engines)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 8: Weight vs Range
    ax = axes[3, 1]
    ax.plot(all_range, all_weight, color=COLORS['weight'], linewidth=2)
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Weight (lbm)')
    ax.set_title('Aircraft Weight')
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Torque Percent vs Range
    ax = axes[4, 0]
    ax.plot(all_range, all_torque_percent, color='tab:brown', linewidth=2)
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Torque (%)')
    ax.set_title('Engine Torque Setting')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 10: Unit Shaft Power vs Range
    ax = axes[4, 1]
    ax.plot(all_range, all_unit_shaft_power, color='tab:pink', linewidth=2)
    ax.set_xlabel('Range (NM)')
    ax.set_ylabel('Shaft Power per Engine (kW)')
    ax.set_title('Unit Shaft Power')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save if requested
    if save_plots and output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{output_filename}_{timestamp}.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        
        print(f"Saved Q400 mission plot to: {pdf_filename}")
    
    plt.show()
    
    # Print summary
    print_q400_summary(prob, mission_config, phase_names)
    
    return fig


def print_q400_summary(prob, mission_config, phase_names):
    """Print a summary of Q400 mission results."""
    
    def get_duration(phase):
        """Get duration, trying promoted name first."""
        try:
            dur = prob.get_val(f'{phase}.duration', units='min')
            return float(dur[0]) if hasattr(dur, '__len__') else float(dur)
        except:
            dur = prob.get_val(f'{phase}.ode_integ_phase.duration', units='min')
            return float(dur[0]) if hasattr(dur, '__len__') else float(dur)
    
    def get_range(phase):
        """Get range, trying promoted name first."""
        try:
            return prob.get_val(f'{phase}.range', units='NM')[-1]
        except:
            return prob.get_val(f'{phase}.ode_integ_phase.range', units='NM')[-1]
    
    print("\n" + "="*60)
    print("Q400 MISSION SUMMARY")
    print("="*60)
    
    # Get final values
    last_phase = phase_names[-1]
    
    try:
        total_range = get_range(last_phase)
        print(f"Total Range:        {total_range:.1f} NM")
    except:
        pass
    
    try:
        total_fuel = prob.get_val(f'{last_phase}.fuel_used', units='kg')[-1]
        print(f"Total Fuel Used:    {total_fuel:.1f} kg ({total_fuel*2.205:.1f} lbm)")
    except:
        pass
    
    try:
        # Sum durations
        total_time = 0
        for phase in phase_names:
            total_time += get_duration(phase)
        print(f"Total Flight Time:  {total_time:.1f} min ({total_time/60:.2f} h)")
    except:
        pass
    
    # Phase-by-phase breakdown
    print("\n" + "-"*60)
    print(f"{'Phase':<15} {'Duration':>10} {'Range':>10} {'Fuel':>10}")
    print(f"{'':15} {'(min)':>10} {'(NM)':>10} {'(kg)':>10}")
    print("-"*60)
    
    prev_range = 0
    prev_fuel = 0
    
    for phase in phase_names:
        try:
            dur = get_duration(phase)
            range_end = get_range(phase)
            fuel_end = prob.get_val(f'{phase}.fuel_used', units='kg')[-1]
            
            phase_range = range_end - prev_range
            phase_fuel = fuel_end - prev_fuel
            
            print(f"{phase:<15} {dur:>10.1f} {phase_range:>10.1f} {phase_fuel:>10.1f}")
            
            prev_range = range_end
            prev_fuel = fuel_end
        except Exception as e:
            print(f"{phase:<15} {'--':>10} {'--':>10} {'--':>10}")
    
    print("="*60 + "\n")


def save_q400_data_to_csv(prob, mission_config, output_filename, payload_qty=None, nn=11):
    """
    Save Q400 mission data to CSV file.
    
    Parameters
    ----------
    prob : OpenMDAO Problem
        Solved problem instance
    mission_config : dict
        Mission configuration dictionary
    output_filename : str
        Base filename for saving
    payload_qty : float
        Payload weight for metadata
    nn : int
        Number of nodes per phase
    """
    import pandas as pd
    
    # PW150 max torque for shaft power calculation
    MAX_TORQUE_NM = 35404
    
    phase_names = [p for p in mission_config['mission']['phase_names'] 
                   if p not in ['mission', 'reserve']]
    
    data_rows = []
    time_offset = 0
    
    for phase_name in phase_names:
        try:
            # Get duration - try promoted name first
            try:
                duration = prob.get_val(f'{phase_name}.duration', units='min')
                duration = float(duration[0]) if hasattr(duration, '__len__') else float(duration)
            except:
                duration = float(prob.get_val(f'{phase_name}.ode_integ_phase.duration', units='min')[0])
            
            time_phase = np.linspace(0, duration, nn) + time_offset
            
            # Get altitude
            altitude = prob.get_val(f'{phase_name}.fltcond|h', units='ft').flatten()
            
            # Get range - try promoted name first
            try:
                range_nm = prob.get_val(f'{phase_name}.range', units='NM').flatten()
            except:
                range_nm = prob.get_val(f'{phase_name}.ode_integ_phase.range', units='NM').flatten()
            
            # Get optional values with defaults, handling matrix vs vector shapes
            def safe_get(path, units, default=np.nan):
                try:
                    val = prob.get_val(path, units=units) if units else prob.get_val(path)
                    if val.ndim > 1:
                        return val.flatten()[:nn]
                    return val.flatten()
                except:
                    return np.full(nn, default)
            
            cl = safe_get(f'{phase_name}.cl', None)
            cd = safe_get(f'{phase_name}.cd_total', None)
            lift = safe_get(f'{phase_name}.lift', 'kN')
            drag = safe_get(f'{phase_name}.drag', 'kN')
            fuel_flow = safe_get(f'{phase_name}.total_fuel_flow', 'lbm/h')
            fuel_used = safe_get(f'{phase_name}.fuel_used', 'lbm')
            tas = safe_get(f'{phase_name}.fltcond|Utrue', 'kn')
            eas = safe_get(f'{phase_name}.fltcond|Ueas', 'kn')
            weight = safe_get(f'{phase_name}.weight', 'lbm')
            
            # Thrust - sum across props if matrix
            try:
                unit_thrust = prob.get_val(f'{phase_name}.unit_thrust', units='kN')
                if unit_thrust.ndim > 1:
                    total_thrust = np.sum(unit_thrust, axis=0).flatten()
                else:
                    total_thrust = unit_thrust.flatten()
            except:
                total_thrust = np.full(nn, np.nan)
            
            # Torque percent - mean across props if matrix
            try:
                torque_pct = prob.get_val(f'{phase_name}.torque_percent')
                if torque_pct.ndim > 1:
                    torque_pct = np.mean(torque_pct, axis=0).flatten() * 100
                else:
                    torque_pct = torque_pct.flatten() * 100
            except:
                torque_pct = np.full(nn, np.nan)
            
            # Unit fuel flow per engine
            try:
                unit_ff = prob.get_val(f'{phase_name}.unit_fuel_flow', units='lbm/h')
                if unit_ff.ndim > 1:
                    unit_ff = unit_ff[0, :].flatten()
                else:
                    unit_ff = unit_ff.flatten()
            except:
                unit_ff = np.full(nn, np.nan)
            
            # RPM - mean across props if matrix
            try:
                rpm = prob.get_val(f'{phase_name}.rpm', units='rpm')
                if rpm.ndim > 1:
                    rpm = np.mean(rpm, axis=0).flatten()
                else:
                    rpm = rpm.flatten()
            except:
                rpm = np.full(nn, np.nan)
            
            # Unit shaft power calculation: P = omega * torque = (rpm / 60 * 2 * pi) * (torque_pct * max_torque)
            # Result in kW
            try:
                rpm_vals = prob.get_val(f'{phase_name}.rpm', units='rpm')
                torque_pct_vals = prob.get_val(f'{phase_name}.torque_percent')
                if rpm_vals.ndim > 1:
                    rpm_mean = np.mean(rpm_vals, axis=0).flatten()
                else:
                    rpm_mean = rpm_vals.flatten()
                if torque_pct_vals.ndim > 1:
                    torque_pct_mean = np.mean(torque_pct_vals, axis=0).flatten()
                else:
                    torque_pct_mean = torque_pct_vals.flatten()
                # omega = rpm / 60 * 2 * pi (rad/s)
                # torque = torque_pct * max_torque (Nm)
                # power = omega * torque (W) -> / 1000 for kW
                omega = rpm_mean / 60 * 2 * np.pi
                torque_nm = torque_pct_mean * MAX_TORQUE_NM
                unit_shaft_power = omega * torque_nm / 1000
            except:
                unit_shaft_power = np.full(nn, np.nan)
            
            for i in range(nn):
                data_rows.append({
                    'phase': phase_name,
                    'time_min': time_phase[i],
                    'altitude_ft': altitude[i] if i < len(altitude) else np.nan,
                    'range_nm': range_nm[i] if i < len(range_nm) else np.nan,
                    'tas_kn': tas[i] if i < len(tas) else np.nan,
                    'eas_kn': eas[i] if i < len(eas) else np.nan,
                    'CL': cl[i] if i < len(cl) else np.nan,
                    'CD': cd[i] if i < len(cd) else np.nan,
                    'lift_kN': lift[i] if i < len(lift) else np.nan,
                    'drag_kN': drag[i] if i < len(drag) else np.nan,
                    'fuel_flow_lbm_h': fuel_flow[i] if i < len(fuel_flow) else np.nan,
                    'fuel_used_lbm': fuel_used[i] if i < len(fuel_used) else np.nan,
                    'unit_fuel_flow_lbm_h': unit_ff[i] if i < len(unit_ff) else np.nan,
                    'total_thrust_kN': total_thrust[i] if i < len(total_thrust) else np.nan,
                    'torque_percent': torque_pct[i] if i < len(torque_pct) else np.nan,
                    'weight_lbm': weight[i] if i < len(weight) else np.nan,
                    'rpm': rpm[i] if i < len(rpm) else np.nan,
                    'unit_shaft_power_kW': unit_shaft_power[i] if i < len(unit_shaft_power) else np.nan
                })
            
            time_offset += duration
            
        except Exception as e:
            print(f"Error processing phase {phase_name} for CSV: {e}")
    
    df = pd.DataFrame(data_rows)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_filename}_data_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved Q400 mission data to: {csv_filename}")
    
    return df

