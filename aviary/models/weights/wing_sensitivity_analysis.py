"""
Wing Weight Sensitivity Analysis

This script analyzes wing weight sensitivity with respect to:
- Thickness-to-chord ratio (t/c)
- Wing reference area (S_ref)

It uses the full OEW problem from compute_oew.py to capture all feedback effects,
including battery inertia relief, propulsion weight effects, etc.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import openmdao.api as om

# Import the OEW setup from compute_oew
from atlas.weights.compute_oew import OEWGroup
from atlas.weights.wing_geometry_comp import WingGeometryComp


def setup_oew_problem():
    """
    Set up the OEW problem similar to compute_oew.py main() function.
    Returns the problem and model for sensitivity analysis.
    """
    prob = om.Problem(reports=False)
    model = prob.model
    
    # Add IndepVarComp for all inputs
    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    
    # ========== FUSELAGE INPUTS ==========
    ivc.add_output('fuse_base', val=10.16, units='ft', desc='Fuselage width')
    ivc.add_output('fuse_height', val=9.87, units='ft', desc='Fuselage height')
    ivc.add_output('Vd', val=315.0, units='knot', desc='Dive speed')
    ivc.add_output('delta_p', val=9.25, units='psi', desc='Pressure differential')
    ivc.add_output('fuse_n_ult', val=1.5, units=None, desc='Ultimate factor')
    
    # ========== WINDOWS INPUTS ==========
    ivc.add_output('a_eff', val=6.0, units='inch', desc='Effective radius for window thickness calculation')
    ivc.add_output('base_pressure', val=6.0, units='psi', desc='Base pressure differential')
    ivc.add_output('window_area', val=175.0, units='inch**2', desc='Area of each passenger window')
    ivc.add_output('windshield_thickness', val=0.50, units='inch', desc='Thickness of windshield')
    ivc.add_output('windshield_surface_front', val=1100.0, units='inch**2', desc='Front surface area of windshield')
    ivc.add_output('windshield_surface_side', val=928.7, units='inch**2', desc='Side surface area of windshield')
    
    # ========== WING INPUTS ==========
    # These will be varied in the sensitivity analysis
    # Note: WingGeometryComp will compute derived geometry from S_ref (m²), AR, taper, toverc, c0_sweep
    ivc.add_output('wing_S_ref_m2', val=84.81, units='m**2', desc='Wing reference area (m²)')
    ivc.add_output('wing_taper_ratio', val=0.35, units=None, desc='Wing taper ratio')
    ivc.add_output('wing_AR', val=14.46, units=None, desc='Wing aspect ratio')
    ivc.add_output('wing_c0_sweep', val=4.854, units='deg', desc='Leading edge sweep angle')
    ivc.add_output('wing_apex_percentage', val=46.2, units=None, desc='Wing apex location as percentage of fuselage length')
    
    # Wing weight-specific inputs
    ivc.add_output('wing_S_ref_control_ratio', val=0.35393652, units=None, desc='Control surface area ratio')
    ivc.add_output('wing_toverc', val=0.18, units=None, desc='Thickness-to-chord ratio')
    ivc.add_output('strut_bracing_factor', val=0.0, units=None, desc='Strut bracing factor')
    ivc.add_output('aeroelastic_tailoring_factor', val=0.0, units=None, desc='Aeroelastic tailoring factor')
    ivc.add_output('gross_weight', val=86000.0, units='lbm', desc='Aircraft gross weight')
    ivc.add_output('composite_fraction', val=0.5, units=None, desc='Composite fraction')
    ivc.add_output('wing_n_ult', val=3.75, units=None, desc='Ultimate load factor')
    ivc.add_output('bending_material_weight_scaler', val=1.0, units=None, desc='Bending material weight scaler')
    ivc.add_output('load_fraction', val=1.0, units=None, desc='Load fraction')
    ivc.add_output('misc_weight_scaler', val=1.0, units=None, desc='Miscellaneous weight scaler')
    ivc.add_output('shear_control_weight_scaler', val=1.0, units=None, desc='Shear control weight scaler')
    ivc.add_output('var_sweep_weight_penalty', val=0.0, units=None, desc='Variable sweep weight penalty')
    ivc.add_output('weight_scaler', val=1.0, units=None, desc='Weight scaler')
    
    # ========== NACELLE INPUTS ==========
    ivc.add_output('nacelle_width_inboard', val=42.5/12.0, units='ft', desc='Inboard nacelle width')
    ivc.add_output('nacelle_height_inboard', val=92.5/12.0, units='ft', desc='Inboard nacelle height')
    ivc.add_output('nacelle_width_outboard', val=38.3/12.0, units='ft', desc='Outboard nacelle width')
    ivc.add_output('nacelle_height_outboard', val=56.0/12.0, units='ft', desc='Outboard nacelle height')
    ivc.add_output('dive_speed', val=315.0, units='knot', desc='Dive speed for nacelle calculation')
    
    # ========== EMPENNAGE INPUTS ==========
    ivc.add_output('hstab_S_ref', val=209.0, units='ft**2', desc='Horizontal tail surface area')
    ivc.add_output('vstab_S_ref', val=162.62, units='ft**2', desc='Vertical tail surface area')
    ivc.add_output('hstab_c2_sweep', val=4.1, units='deg', desc='Sweep angle at half chord for horizontal tail')
    ivc.add_output('vstab_c2_sweep', val=38.4, units='deg', desc='Sweep angle at half chord for vertical tail')
    
    # ========== LANDING GEAR INPUTS ==========
    ivc.add_output('landing_weight', val=86000.0, units='lbm', desc='Aircraft landing weight')
    ivc.add_output('mlg_height', val=54.74, units='inch', desc='Main landing gear height')
    ivc.add_output('nlg_height', val=65.40, units='inch', desc='Nose landing gear height')
    
    # ========== FLIGHT CONTROLS INPUTS ==========
    ivc.add_output('max_mach', val=0.69, units=None, desc='Maximum Mach number')
    # Note: wing_surface comes from wing_S_ref_m2 via connection (not set here)
    
    # ========== PROPULSION INPUTS ==========
    ivc.add_output('num_blades', val=6, units=None, desc='Number of propeller blades')
    ivc.add_output('blade_diameter', val=13.0, units='ft', desc='Propeller blade diameter')
    ivc.add_output('power_density', val=4.54, units='kW/lbm', desc='Power density')
    ivc.add_output('rated_power_em_per_nacelle', val=1341.0*2, units='hp', desc='Rated power electric motor per nacelle')
    ivc.add_output('rated_power_turbine', val=1200.0, units='hp', desc='Rated power turbine')
    ivc.add_output('num_em_per_nac', val=2, units=None, desc='Number of electric engines')
    ivc.add_output('electric_k', val=118.0, units=None, desc='Electric engine technology factor')
    ivc.add_output('electric_RPMin', val=12000.0, units='rpm', desc='Electric engine input RPM')
    ivc.add_output('electric_RPMout', val=1200.0, units='rpm', desc='Electric engine output RPM')
    ivc.add_output('num_turb_per_nac', val=1, units=None, desc='Number of turbines')
    ivc.add_output('turbine_k', val=118.0, units=None, desc='Turbine technology factor')
    ivc.add_output('turbine_RPMin', val=30000.0, units='rpm', desc='Turbine input RPM')
    ivc.add_output('turbine_RPMout', val=1200.0, units='rpm', desc='Turbine output RPM')
    
    # ========== PASSENGER INPUTS ==========
    ivc.add_output('num_passengers', val=76, units=None, desc='Number of passengers')
    
    # ========== AIR CONDITIONING INPUTS ==========
    ivc.add_output('max_passengers', val=88, units=None, desc='Maximum number of passengers for air conditioning calculation')
    
    # ========== BATTERY INPUTS ==========
    ivc.add_output('n_str', val=4.0)
    ivc.add_output('n_parallel_per_str', val=112.0)
    ivc.add_output('n_series_per_str', val=210.0)
    ivc.add_output('m_cell', val=0.070, units='kg')  # 70 grams per cell
    
    # ========== ELECTRICAL INPUTS ==========
    ivc.add_output('fuselage_width', val=10.16, units='ft', desc='Fuselage width for LVWIS')
    ivc.add_output('fuselage_height', val=9.87, units='ft', desc='Fuselage height for LVWIS')
    ivc.add_output('cabin_percent_cross_section', val=0.81, units=None, desc='Cabin cross section percentage')
    ivc.add_output('nacelle_distance', val=179.5, units='ft', desc='Distance between inboard and outboard nacelles')
    ivc.add_output('max_voltage', val=570.0, units='V', desc='Maximum voltage')
    
    # ========== THERMAL MANAGEMENT INPUTS ==========
    ivc.add_output('efficiency', val=0.95, units=None, desc='Electric motor efficiency')
    ivc.add_output('gearbox_heat', val=2500.0, units='W', desc='Gearbox heat generation')
    # Note: delta_T_em no longer needed (using kW/kg method)
    ivc.add_output('turbine_heat_per_nac', val=15000.0, units='W', desc='Turbine heat generation')
    # Note: delta_T_turb no longer needed (using kW/kg method)
    ivc.add_output('battery_heat_per_wing', val=50000.0, units='W', desc='Battery heat generation')
    # Note: delta_T_batt no longer needed (using kW/kg method)
    
    # ========== FURNISHING INPUTS ==========
    ivc.add_output('business_class_pax', val=0, units=None, desc='Number of business class passengers')
    
    # ========== PAYLOAD INPUTS ==========
    ivc.add_output('num_rows_fwd', val=9, units=None, desc='Number of rows forward of OWEED')
    ivc.add_output('pitch', val=32.0, units='inch', desc='Seat pitch (spacing between rows)')
    
    # Connect Power Values
    model.connect('rated_power_turbine', 'turbine_HP')
    
    # Add WingGeometryComp to compute all derived wing geometry
    # This ensures consistency and captures all second-order effects
    model.add_subsystem('wing_geom', WingGeometryComp(
        default_S_ref=84.81,
        default_AR=14.46,
        default_taper=0.35,
        default_toverc=0.18,
        default_c0_sweep=np.radians(4.854),
        k_max_thickness=0.3,
        toverc_shift=0.025
    ), promotes=[])
    
    # Connect WingGeometryComp inputs (OpenMDAO handles unit conversions automatically)
    model.connect('wing_S_ref_m2', 'wing_geom.S_ref')  # m² → m²
    model.connect('wing_AR', 'wing_geom.AR')  # unitless
    model.connect('wing_taper_ratio', 'wing_geom.taper')  # unitless
    model.connect('wing_toverc', 'wing_geom.toverc')  # unitless
    model.connect('wing_c0_sweep', 'wing_geom.c0_sweep')  # deg → rad (automatic)
    
    # Add the OEW Group
    model.add_subsystem('oew', OEWGroup(), promotes=['*'])
    
    # Connect fuselage inputs
    model.connect('fuse_base', ['fuse.base', 'fuse_parameter_links.base', 'bf'])
    model.connect('fuse_height', ['fuse.height', 'fuse_parameter_links.height', 'hf'])
    model.connect('fuse_n_ult', 'fuse.n_ult')
    
    # Connect wing_parameter_links inputs (using WingGeometryComp outputs)
    # OpenMDAO automatically converts units: m → inch, etc.
    model.connect('wing_taper_ratio', ['wing.taper_ratio', 'wing_parameter_links.taper_ratio'])
    model.connect('wing_geom.span', ['wing.span', 'wing_parameter_links.span'])  # m → inch (automatic)
    model.connect('wing_c0_sweep', 'wing_parameter_links.c0_sweep')  # deg → deg
    model.connect('wing_geom.MAC', 'MAC')  # m → inch (automatic)
    
    # Connect wing weight geometry inputs (using WingGeometryComp outputs)
    # OpenMDAO automatically converts: m² → ft², rad → deg
    model.connect('wing_S_ref_m2', 'wing.S_ref')  # m² → ft² (automatic)
    model.connect('wing_AR', 'wing.aspect_ratio')  # unitless
    model.connect('wing_geom.c4_sweep', 'wing.c4_sweep')  # rad → deg (automatic)
    model.connect('wing_S_ref_control_ratio', 'wing.S_ref_control_ratio')  # unitless
    model.connect('wing_toverc', 'wing.toverc')  # unitless
    model.connect('wing_n_ult', 'wing.n_ult')  # unitless
    
    # Also update wing_surface for flight controls (m² → ft², automatic)
    model.connect('wing_S_ref_m2', 'wing_surface')
    
    # Connect empennage inputs
    model.connect('hstab_S_ref', ['empennage.horizontal_tail.S_ref', 'empennage.vertical_tail.S_ref_hor'])
    model.connect('vstab_S_ref', 'empennage.vertical_tail.S_ref')
    model.connect('hstab_c2_sweep', 'empennage.horizontal_tail.c2_sweep')
    model.connect('vstab_c2_sweep', 'empennage.vertical_tail.c2_sweep')
    
    # Add Payload and Fuel (needed for complete problem)
    from atlas.weights.payload import PayloadWeight
    from atlas.weights.fuel import FuelWeight
    
    model.add_subsystem('payload', PayloadWeight(), promotes_inputs=['*'],
                       promotes_outputs=['payload_weight', 'cg_passenger', 'passenger_weight', 
                                       'cargo_weight', 'cg_cargo'])
    
    model.add_subsystem('fuel', FuelWeight(), promotes_inputs=['*'],
                       promotes_outputs=['fuel_weight', 'cg_fuel'])
    
    prob.setup()
    
    return prob




def main():
    """Run wing weight sensitivity analysis."""
    print("="*70)
    print("WING WEIGHT SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nThis analysis includes full feedback effects:")
    print("  - Battery inertia relief")
    print("  - Propulsion weight effects")
    print("  - All OEW component interactions")
    print()
    
    # Design space - 3D sweep: t/c, S_ref, AR
    toverc_range = np.linspace(0.12, 0.20, 20)
    S_ref_m2_range = np.linspace(80.0, 90.0, 20)
    AR_range = np.linspace(12.0, 15.0, 15)
    S_ref_ft2_range = S_ref_m2_range * 10.764  # Convert to ft² for plotting
    
    # Fixed parameters
    taper_ratio = 0.35
    c0_sweep = 4.854  # deg
    
    # Initialize 3D weight arrays: [AR_idx, S_idx, toverc_idx]
    n_AR = len(AR_range)
    n_S = len(S_ref_m2_range)
    n_tc = len(toverc_range)
    
    WING_WEIGHT_3D = np.zeros((n_AR, n_S, n_tc))
    OEW_3D = np.zeros((n_AR, n_S, n_tc))
    BATTERY_WEIGHT_3D = np.zeros((n_AR, n_S, n_tc))
    PROPULSION_WEIGHT_3D = np.zeros((n_AR, n_S, n_tc))
    
    # Set up the problem once
    print("Setting up OEW problem with WingGeometryComp...")
    prob = setup_oew_problem()
    print("✓ Problem setup complete\n")
    
    # Compute wing weight for each combination
    print("Computing 3D wing weight sensitivity (t/c, S_ref, AR)...")
    print("  (WingGeometryComp will compute all derived geometry automatically)")
    total_points = n_AR * n_S * n_tc
    point_count = 0
    
    for k in range(n_AR):
        AR = AR_range[k]
        for i in range(n_S):
            S_ref_m2 = S_ref_m2_range[i]
            for j in range(n_tc):
                toverc = toverc_range[j]
                point_count += 1
                if point_count % 200 == 0:
                    print(f"  Progress: {point_count}/{total_points} points ({100*point_count/total_points:.1f}%)")
                
                # Update problem inputs - WingGeometryComp will compute all derived geometry
                prob.set_val('wing_S_ref_m2', S_ref_m2, units='m**2')
                prob.set_val('wing_toverc', toverc)
                prob.set_val('wing_AR', AR)
                prob.set_val('wing_taper_ratio', taper_ratio)
                prob.set_val('wing_c0_sweep', c0_sweep, units='deg')
                
                # Run the model
                prob.run_model()
                
                # Extract results
                WING_WEIGHT_3D[k, i, j] = prob.get_val('wing_weight', units='lbm')[0]
                OEW_3D[k, i, j] = prob.get_val('OEW', units='lbm')[0]
                BATTERY_WEIGHT_3D[k, i, j] = prob.get_val('total_battery_weight', units='lbm')[0]
                PROPULSION_WEIGHT_3D[k, i, j] = prob.get_val('total_propulsion_weight', units='lbm')[0]
    
    print(f"✓ Completed {total_points} evaluations\n")
    
    # Create the plots
    print("Generating sensitivity plots...")
    
    # Save plot directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aircraft_performance_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    results_dir = os.path.join(os.path.dirname(aircraft_performance_dir), 'aircraft_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Reference point 1: AR=14.46
    ref_tc1 = 0.155
    ref_S_m2_1 = 84.81
    ref_S_ft2_1 = ref_S_m2_1 * 10.764
    ref_AR1 = 14.46
    
    # Reference point 2: AR=15.0
    ref_tc2 = 0.162
    ref_S_m2_2 = 86.2
    ref_S_ft2_2 = ref_S_m2_2 * 10.764
    ref_AR2 = 15.0
    
    # Find reference point indices
    ref_i1 = np.argmin(np.abs(S_ref_m2_range - ref_S_m2_1))
    ref_j1 = np.argmin(np.abs(toverc_range - ref_tc1))
    ref_k1 = np.argmin(np.abs(AR_range - ref_AR1))
    ref_i2 = np.argmin(np.abs(S_ref_m2_range - ref_S_m2_2))
    ref_j2 = np.argmin(np.abs(toverc_range - ref_tc2))
    ref_k2 = np.argmin(np.abs(AR_range - ref_AR2))
    
    # Sensitivity calculation parameters
    dtc = 0.005  # change per 0.005 t/c
    dS_ft2 = 50.0  # change per 50 ft²
    dS_m2 = dS_ft2 / 10.764
    dAR = 0.5  # change per 0.5 AR
    
    def calc_sensitivities_3d(ref_k, ref_i, ref_j):
        """Calculate sensitivities at a reference point using central differences in 3D."""
        # Find indices for t/c ± dtc
        tc_step = max(1, int(dtc / (toverc_range[1] - toverc_range[0])))
        tc_plus_idx = min(ref_j + tc_step, n_tc - 1)
        tc_minus_idx = max(ref_j - tc_step, 0)
        
        # Find indices for S ± dS
        S_step = max(1, int(dS_m2 / (S_ref_m2_range[1] - S_ref_m2_range[0])))
        S_plus_idx = min(ref_i + S_step, n_S - 1)
        S_minus_idx = max(ref_i - S_step, 0)
        
        # Find indices for AR ± dAR
        AR_step = max(1, int(dAR / (AR_range[1] - AR_range[0])))
        AR_plus_idx = min(ref_k + AR_step, n_AR - 1)
        AR_minus_idx = max(ref_k - AR_step, 0)
        
        # Calculate sensitivities (finite difference) - central difference
        # For t/c
        dWing_dtc = (WING_WEIGHT_3D[ref_k, ref_i, tc_plus_idx] - WING_WEIGHT_3D[ref_k, ref_i, tc_minus_idx]) / (toverc_range[tc_plus_idx] - toverc_range[tc_minus_idx])
        dOEW_dtc = (OEW_3D[ref_k, ref_i, tc_plus_idx] - OEW_3D[ref_k, ref_i, tc_minus_idx]) / (toverc_range[tc_plus_idx] - toverc_range[tc_minus_idx])
        
        # For S_ref
        dWing_dS = (WING_WEIGHT_3D[ref_k, S_plus_idx, ref_j] - WING_WEIGHT_3D[ref_k, S_minus_idx, ref_j]) / (S_ref_m2_range[S_plus_idx] - S_ref_m2_range[S_minus_idx])
        dOEW_dS = (OEW_3D[ref_k, S_plus_idx, ref_j] - OEW_3D[ref_k, S_minus_idx, ref_j]) / (S_ref_m2_range[S_plus_idx] - S_ref_m2_range[S_minus_idx])
        
        # For AR
        dWing_dAR = (WING_WEIGHT_3D[AR_plus_idx, ref_i, ref_j] - WING_WEIGHT_3D[AR_minus_idx, ref_i, ref_j]) / (AR_range[AR_plus_idx] - AR_range[AR_minus_idx])
        dOEW_dAR = (OEW_3D[AR_plus_idx, ref_i, ref_j] - OEW_3D[AR_minus_idx, ref_i, ref_j]) / (AR_range[AR_plus_idx] - AR_range[AR_minus_idx])
        
        # Convert dS sensitivity from per m² to per ft²
        dWing_dS_ft2 = dWing_dS / 10.764
        dOEW_dS_ft2 = dOEW_dS / 10.764
        
        # Calculate changes per specified increments
        dWing_per_tc = dWing_dtc * dtc
        dOEW_per_tc = dOEW_dtc * dtc
        dWing_per_S = dWing_dS_ft2 * dS_ft2
        dOEW_per_S = dOEW_dS_ft2 * dS_ft2
        dWing_per_AR = dWing_dAR * dAR
        dOEW_per_AR = dOEW_dAR * dAR
        
        return dWing_per_tc, dOEW_per_tc, dWing_per_S, dOEW_per_S, dWing_per_AR, dOEW_per_AR
    
    # Calculate sensitivities at both reference points
    dWing_per_tc_1, dOEW_per_tc_1, dWing_per_S_1, dOEW_per_S_1, dWing_per_AR_1, dOEW_per_AR_1 = calc_sensitivities_3d(ref_k1, ref_i1, ref_j1)
    dWing_per_tc_2, dOEW_per_tc_2, dWing_per_S_2, dOEW_per_S_2, dWing_per_AR_2, dOEW_per_AR_2 = calc_sensitivities_3d(ref_k2, ref_i2, ref_j2)
    
    # Create 2D meshgrids for plotting (at specific AR slices)
    TOVERC, S_REF_M2 = np.meshgrid(toverc_range, S_ref_m2_range)
    _, S_REF_FT2 = np.meshgrid(toverc_range, S_ref_ft2_range)
    
    # Extract 2D slices at each reference AR
    WING_WEIGHT_AR1 = WING_WEIGHT_3D[ref_k1, :, :]
    OEW_AR1 = OEW_3D[ref_k1, :, :]
    WING_WEIGHT_AR2 = WING_WEIGHT_3D[ref_k2, :, :]
    OEW_AR2 = OEW_3D[ref_k2, :, :]
    
    # ===== PLOT 1: Wing Weight - 2 subplots for different AR =====
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Common color scale for both AR plots
    wing_vmin = min(WING_WEIGHT_AR1.min(), WING_WEIGHT_AR2.min())
    wing_vmax = max(WING_WEIGHT_AR1.max(), WING_WEIGHT_AR2.max())
    wing_levels = np.linspace(wing_vmin, wing_vmax, 25)
    
    # Left: AR = ref_AR1
    contour1a = ax1a.contourf(TOVERC, S_REF_FT2, WING_WEIGHT_AR1, levels=wing_levels, cmap='viridis')
    contour_lines1a = ax1a.contour(TOVERC, S_REF_FT2, WING_WEIGHT_AR1, levels=15, colors='black', alpha=0.4, linewidths=0.8)
    ax1a.clabel(contour_lines1a, inline=True, fontsize=8, fmt='%d')
    ax1a.set_xlabel('Thickness-to-Chord Ratio (t/c)', fontsize=12, fontweight='bold')
    ax1a.set_ylabel('Wing Reference Area (ft²)', fontsize=12, fontweight='bold')
    ax1a.set_title(f'AR = {AR_range[ref_k1]:.2f} (Ref 1)', fontsize=13, fontweight='bold')
    ax1a.grid(True, alpha=0.3, linestyle='--')
    ax1a.plot(ref_tc1, ref_S_ft2_1, 'r*', markersize=16, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Right: AR = ref_AR2
    contour1b = ax1b.contourf(TOVERC, S_REF_FT2, WING_WEIGHT_AR2, levels=wing_levels, cmap='viridis')
    contour_lines1b = ax1b.contour(TOVERC, S_REF_FT2, WING_WEIGHT_AR2, levels=15, colors='black', alpha=0.4, linewidths=0.8)
    ax1b.clabel(contour_lines1b, inline=True, fontsize=8, fmt='%d')
    ax1b.set_xlabel('Thickness-to-Chord Ratio (t/c)', fontsize=12, fontweight='bold')
    ax1b.set_ylabel('Wing Reference Area (ft²)', fontsize=12, fontweight='bold')
    ax1b.set_title(f'AR = {AR_range[ref_k2]:.2f} (Ref 2)', fontsize=13, fontweight='bold')
    ax1b.grid(True, alpha=0.3, linestyle='--')
    ax1b.plot(ref_tc2, ref_S_ft2_2, 'c^', markersize=14, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Shared colorbar
    cbar1 = fig1.colorbar(contour1b, ax=[ax1a, ax1b], location='right', shrink=0.8)
    cbar1.set_label('Wing Weight (lbm)', fontsize=12, fontweight='bold')
    
    # Add sensitivity text box
    sensitivity_text1 = (
        f'Reference Points & Sensitivities:\n'
        f'══════════════════════════════════════════\n'
        f'Ref 1: t/c={ref_tc1}, S={ref_S_ft2_1:.0f}ft², AR={ref_AR1}\n'
        f'  Wing Wt: {WING_WEIGHT_3D[ref_k1, ref_i1, ref_j1]:.0f} lbm\n'
        f'  Δ per Δt/c=0.005: {dWing_per_tc_1:+.1f} lbm\n'
        f'  Δ per ΔS=50ft²:   {dWing_per_S_1:+.1f} lbm\n'
        f'  Δ per ΔAR=0.5:    {dWing_per_AR_1:+.1f} lbm\n'
        f'──────────────────────────────────────────\n'
        f'Ref 2: t/c={ref_tc2}, S={ref_S_ft2_2:.0f}ft², AR={ref_AR2}\n'
        f'  Wing Wt: {WING_WEIGHT_3D[ref_k2, ref_i2, ref_j2]:.0f} lbm\n'
        f'  Δ per Δt/c=0.005: {dWing_per_tc_2:+.1f} lbm\n'
        f'  Δ per ΔS=50ft²:   {dWing_per_S_2:+.1f} lbm\n'
        f'  Δ per ΔAR=0.5:    {dWing_per_AR_2:+.1f} lbm'
    )
    fig1.text(0.01, 0.02, sensitivity_text1, fontsize=9, fontfamily='monospace',
              verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    fig1.suptitle('Wing Weight Sensitivity Analysis\n(t/c, Wing Area, and Aspect Ratio)', 
                  fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_filename1 = os.path.join(results_dir, 'wing_weight_sensitivity_3d.png')
    plt.savefig(plot_filename1, dpi=150, bbox_inches='tight')
    print(f"✓ Wing Weight plot saved as '{plot_filename1}'")
    
    # ===== PLOT 2: OEW - 2 subplots for different AR =====
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Common color scale for both AR plots
    oew_vmin = min(OEW_AR1.min(), OEW_AR2.min())
    oew_vmax = max(OEW_AR1.max(), OEW_AR2.max())
    oew_levels = np.linspace(oew_vmin, oew_vmax, 25)
    
    # Left: AR = ref_AR1
    contour2a = ax2a.contourf(TOVERC, S_REF_FT2, OEW_AR1, levels=oew_levels, cmap='plasma')
    contour_lines2a = ax2a.contour(TOVERC, S_REF_FT2, OEW_AR1, levels=15, colors='black', alpha=0.4, linewidths=0.8)
    ax2a.clabel(contour_lines2a, inline=True, fontsize=8, fmt='%d')
    ax2a.set_xlabel('Thickness-to-Chord Ratio (t/c)', fontsize=12, fontweight='bold')
    ax2a.set_ylabel('Wing Reference Area (ft²)', fontsize=12, fontweight='bold')
    ax2a.set_title(f'AR = {AR_range[ref_k1]:.2f} (Ref 1)', fontsize=13, fontweight='bold')
    ax2a.grid(True, alpha=0.3, linestyle='--')
    ax2a.plot(ref_tc1, ref_S_ft2_1, 'r*', markersize=16, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Right: AR = ref_AR2
    contour2b = ax2b.contourf(TOVERC, S_REF_FT2, OEW_AR2, levels=oew_levels, cmap='plasma')
    contour_lines2b = ax2b.contour(TOVERC, S_REF_FT2, OEW_AR2, levels=15, colors='black', alpha=0.4, linewidths=0.8)
    ax2b.clabel(contour_lines2b, inline=True, fontsize=8, fmt='%d')
    ax2b.set_xlabel('Thickness-to-Chord Ratio (t/c)', fontsize=12, fontweight='bold')
    ax2b.set_ylabel('Wing Reference Area (ft²)', fontsize=12, fontweight='bold')
    ax2b.set_title(f'AR = {AR_range[ref_k2]:.2f} (Ref 2)', fontsize=13, fontweight='bold')
    ax2b.grid(True, alpha=0.3, linestyle='--')
    ax2b.plot(ref_tc2, ref_S_ft2_2, 'c^', markersize=14, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    # Shared colorbar
    cbar2 = fig2.colorbar(contour2b, ax=[ax2a, ax2b], location='right', shrink=0.8)
    cbar2.set_label('Operating Empty Weight (lbm)', fontsize=12, fontweight='bold')
    
    # Add sensitivity text box
    sensitivity_text2 = (
        f'Reference Points & Sensitivities:\n'
        f'══════════════════════════════════════════\n'
        f'Ref 1: t/c={ref_tc1}, S={ref_S_ft2_1:.0f}ft², AR={ref_AR1}\n'
        f'  OEW: {OEW_3D[ref_k1, ref_i1, ref_j1]:.0f} lbm\n'
        f'  Δ per Δt/c=0.005: {dOEW_per_tc_1:+.1f} lbm\n'
        f'  Δ per ΔS=50ft²:   {dOEW_per_S_1:+.1f} lbm\n'
        f'  Δ per ΔAR=0.5:    {dOEW_per_AR_1:+.1f} lbm\n'
        f'──────────────────────────────────────────\n'
        f'Ref 2: t/c={ref_tc2}, S={ref_S_ft2_2:.0f}ft², AR={ref_AR2}\n'
        f'  OEW: {OEW_3D[ref_k2, ref_i2, ref_j2]:.0f} lbm\n'
        f'  Δ per Δt/c=0.005: {dOEW_per_tc_2:+.1f} lbm\n'
        f'  Δ per ΔS=50ft²:   {dOEW_per_S_2:+.1f} lbm\n'
        f'  Δ per ΔAR=0.5:    {dOEW_per_AR_2:+.1f} lbm'
    )
    fig2.text(0.01, 0.02, sensitivity_text2, fontsize=9, fontfamily='monospace',
              verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    fig2.suptitle('OEW Sensitivity Analysis\n(t/c, Wing Area, and Aspect Ratio)', 
                  fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_filename2 = os.path.join(results_dir, 'oew_sensitivity_3d.png')
    plt.savefig(plot_filename2, dpi=150, bbox_inches='tight')
    print(f"✓ OEW plot saved as '{plot_filename2}'")
    
    # Print statistics
    print(f"\n" + "="*80)
    print("SENSITIVITY STATISTICS (3D: t/c, S_ref, AR)")
    print("="*80)
    
    print(f"\nDesign Space:")
    print(f"  t/c:  {toverc_range[0]:.3f} to {toverc_range[-1]:.3f}")
    print(f"  S:    {S_ref_m2_range[0]:.1f} to {S_ref_m2_range[-1]:.1f} m² ({S_ref_ft2_range[0]:.1f} to {S_ref_ft2_range[-1]:.1f} ft²)")
    print(f"  AR:   {AR_range[0]:.2f} to {AR_range[-1]:.2f}")
    
    print(f"\nWing Weight (across entire 3D space):")
    print(f"  Range: {WING_WEIGHT_3D.min():.1f} - {WING_WEIGHT_3D.max():.1f} lbm (Δ = {WING_WEIGHT_3D.max()-WING_WEIGHT_3D.min():.1f} lbm)")
    min_idx_wing = np.unravel_index(WING_WEIGHT_3D.argmin(), WING_WEIGHT_3D.shape)
    print(f"  Min: {WING_WEIGHT_3D.min():.1f} lbm at AR={AR_range[min_idx_wing[0]]:.2f}, t/c={toverc_range[min_idx_wing[2]]:.3f}, S={S_ref_ft2_range[min_idx_wing[1]]:.0f} ft²")
    max_idx_wing = np.unravel_index(WING_WEIGHT_3D.argmax(), WING_WEIGHT_3D.shape)
    print(f"  Max: {WING_WEIGHT_3D.max():.1f} lbm at AR={AR_range[max_idx_wing[0]]:.2f}, t/c={toverc_range[max_idx_wing[2]]:.3f}, S={S_ref_ft2_range[max_idx_wing[1]]:.0f} ft²")
    
    print(f"\nOperating Empty Weight (OEW):")
    print(f"  Range: {OEW_3D.min():.1f} - {OEW_3D.max():.1f} lbm (Δ = {OEW_3D.max()-OEW_3D.min():.1f} lbm)")
    min_idx_oew = np.unravel_index(OEW_3D.argmin(), OEW_3D.shape)
    print(f"  Min: {OEW_3D.min():.1f} lbm at AR={AR_range[min_idx_oew[0]]:.2f}, t/c={toverc_range[min_idx_oew[2]]:.3f}, S={S_ref_ft2_range[min_idx_oew[1]]:.0f} ft²")
    max_idx_oew = np.unravel_index(OEW_3D.argmax(), OEW_3D.shape)
    print(f"  Max: {OEW_3D.max():.1f} lbm at AR={AR_range[max_idx_oew[0]]:.2f}, t/c={toverc_range[max_idx_oew[2]]:.3f}, S={S_ref_ft2_range[max_idx_oew[1]]:.0f} ft²")
    
    # Sensitivity at reference points
    print(f"\n--- Reference Point 1 (t/c={ref_tc1}, S={ref_S_ft2_1:.0f} ft², AR={ref_AR1}) ---")
    print(f"  Wing weight:       {WING_WEIGHT_3D[ref_k1, ref_i1, ref_j1]:.1f} lbm")
    print(f"  OEW:               {OEW_3D[ref_k1, ref_i1, ref_j1]:.1f} lbm")
    print(f"  Battery weight:    {BATTERY_WEIGHT_3D[ref_k1, ref_i1, ref_j1]:.1f} lbm")
    print(f"  Propulsion weight: {PROPULSION_WEIGHT_3D[ref_k1, ref_i1, ref_j1]:.1f} lbm")
    print(f"  Sensitivity Ratios:")
    print(f"    Wing Weight: Δ/Δt/c=0.005: {dWing_per_tc_1:+.1f} lbm | Δ/ΔS=50ft²: {dWing_per_S_1:+.1f} lbm | Δ/ΔAR=0.5: {dWing_per_AR_1:+.1f} lbm")
    print(f"    OEW:         Δ/Δt/c=0.005: {dOEW_per_tc_1:+.1f} lbm | Δ/ΔS=50ft²: {dOEW_per_S_1:+.1f} lbm | Δ/ΔAR=0.5: {dOEW_per_AR_1:+.1f} lbm")
    
    print(f"\n--- Reference Point 2 (t/c={ref_tc2}, S={ref_S_ft2_2:.0f} ft², AR={ref_AR2}) ---")
    print(f"  Wing weight:       {WING_WEIGHT_3D[ref_k2, ref_i2, ref_j2]:.1f} lbm")
    print(f"  OEW:               {OEW_3D[ref_k2, ref_i2, ref_j2]:.1f} lbm")
    print(f"  Battery weight:    {BATTERY_WEIGHT_3D[ref_k2, ref_i2, ref_j2]:.1f} lbm")
    print(f"  Propulsion weight: {PROPULSION_WEIGHT_3D[ref_k2, ref_i2, ref_j2]:.1f} lbm")
    print(f"  Sensitivity Ratios:")
    print(f"    Wing Weight: Δ/Δt/c=0.005: {dWing_per_tc_2:+.1f} lbm | Δ/ΔS=50ft²: {dWing_per_S_2:+.1f} lbm | Δ/ΔAR=0.5: {dWing_per_AR_2:+.1f} lbm")
    print(f"    OEW:         Δ/Δt/c=0.005: {dOEW_per_tc_2:+.1f} lbm | Δ/ΔS=50ft²: {dOEW_per_S_2:+.1f} lbm | Δ/ΔAR=0.5: {dOEW_per_AR_2:+.1f} lbm")
    
    print("\n" + "="*80)
    plt.show()


if __name__ == '__main__':
    main()

