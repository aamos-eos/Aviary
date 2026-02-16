"""
Script to load aircraft data from ac_data.xlsx and compute Operating Empty Weight (OEW).

This script:
1. Loads aircraft data from Excel using load_ac_data_from_excel
2. Creates a DictIndepVarComp to parse the data into OpenMDAO variables
3. Connects the data to OEWGroup for weight and C.G. calculations
4. Exports results to a text file
"""

import os
import sys
import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from atlas.utils import DictIndepVarComp
from atlas.scenarios.setup_mission.load_ac_data import load_ac_data_from_excel
from atlas.scenarios.setup_mission.setup_ac_dvcomp import setup_ac_dv
from atlas.weights.compute_oew import OEWGroup, export_to_text_file
from atlas.weights.payload import PayloadWeight
from atlas.weights.fuel import FuelWeight


def connect_oew_inputs(model, compute_payload=False, compute_oew=False):
    """
    Connect DictIndepVarComp outputs to OEWGroup inputs.
    
    Parameters
    ----------
    model : openmdao.core.group.Group
        The OpenMDAO model/group to add connections to.
    """

    if compute_oew:
        # ========== CONNECTIONS: ac_data -> OEW Group ==========
        # Map DictIndepVarComp output names to OEWGroup input names
        
        # Fuselage (connect to non-promoted inputs)
        model.connect('ac|geom|fuselage|width', ['fuse.base', 'fuse_parameter_links.base','bf'])
        model.connect('ac|geom|fuselage|height', ['fuse.height', 'fuse_parameter_links.height','hf'])
        model.connect('ac|geom|cabin|cabin_length', 'fuse_parameter_links.cabin_length')
        model.connect('ac|geom|fuselage|nose_tail_length', 'fuse_parameter_links.nose_tail_length')
        model.connect('ac|geom|fuselage|dive_speed', 'Vd')
        model.connect('ac|geom|fuselage|pressure_diff', 'delta_p')
        model.connect('ac|geom|fuselage|n_ult', ['fuse.n_ult'])
        model.connect('ac|geom|fuselage|S_wet', 'Sg')
        
        # Windows
        model.connect('ac|geom|windows|a_eff', 'a_eff')
        model.connect('ac|geom|windows|base_pressure', 'base_pressure')
        model.connect('ac|geom|windows|area', 'window_area')
        model.connect('ac|geom|windows|windshield_thickness', 'windshield_thickness')
        model.connect('ac|geom|windows|windshield_surface_front', 'windshield_surface_front')
        model.connect('ac|geom|windows|windshield_surface_side', 'windshield_surface_side')
        
        # Wing geometry (from ac_data.xlsx / compute_wing_geometry)
        # Connect to both wing component and wing_parameter_links (non-promoted)
        # Use span_plan and S_ref for weight calculations, span_trap and S_trap for geometry
        model.connect('ac|geom|wing|taper', ['wing.taper_ratio', 'wing_parameter_links.taper_ratio'])
        model.connect('ac|geom|wing|S_ref', 'wing.S_ref')  # Planform area for weight
        model.connect('ac|geom|wing|span_plan', 'wing.span_plan')  # Planform span for weight
        model.connect('ac|geom|wing|AR', 'wing.aspect_ratio')
        model.connect('ac|geom|wing|c4_sweep', 'wing.c4_sweep')
        model.connect('ac|geom|wing|MAC', 'MAC')  # MAC from trapezoidal geometry
        model.connect('ac|geom|wing|span_trap', 'wing_parameter_links.span_trap')  # Trapezoidal span for LEMAC
        model.connect('ac|geom|wing|c0_sweep', 'wing_parameter_links.c0_sweep')
        model.connect('ac|geom|wing|apex_percentage', 'wing_apex_percentage')
        
        # Wing weight inputs (connect to non-promoted inputs)
        model.connect('ac|geom|wing|control_surface_area_ratio', 'wing.S_ref_control_ratio')
        model.connect('ac|geom|wing|toverc', 'wing.toverc')
        model.connect('ac|geom|wing|strut_bracing_factor', 'strut_bracing_factor')
        model.connect('ac|geom|wing|aeroelastic_tailoring_factor', 'aeroelastic_tailoring_factor')
        model.connect('ac|geom|wing|composite_fraction', 'composite_fraction')
        model.connect('ac|geom|wing|n_ult', 'wing.n_ult')
        model.connect('ac|geom|wing|bending_material_weight_scaler', 'bending_material_weight_scaler')
        model.connect('ac|geom|wing|load_fraction', 'load_fraction')
        model.connect('ac|geom|wing|misc_weight_scaler', 'misc_weight_scaler')
        model.connect('ac|geom|wing|shear_control_weight_scaler', 'shear_control_weight_scaler')
        model.connect('ac|geom|wing|var_sweep_weight_penalty', 'var_sweep_weight_penalty')
        model.connect('ac|geom|wing|weight_scaler', 'weight_scaler')
        
        # Weights
        model.connect('ac|weights|MTOW', 'gross_weight')
        model.connect('ac|weights|MLW', 'landing_weight')
        model.connect('ac|geom|wing|S_ref', 'wing_surface')
        
        # Nacelle Inboard
        model.connect('ac|geom|nacelle_inboard|width', 'nacelle_width_inboard')
        model.connect('ac|geom|nacelle_inboard|height', 'nacelle_height_inboard')
        
        # Nacelle Outboard
        model.connect('ac|geom|nacelle_outboard|width', 'nacelle_width_outboard')
        model.connect('ac|geom|nacelle_outboard|height', 'nacelle_height_outboard')
        model.connect('ac|geom|nacelle_outboard|dive_speed', 'dive_speed')
        
        # Empennage (connect to non-promoted inputs)
        # Horizontal tail
        model.connect('ac|geom|hstab|S_ref', 'empennage.horizontal_tail.surface')
        model.connect('ac|geom|hstab|c2_sweep', 'empennage.horizontal_tail.c2_sweep')
        # Vertical tail
        model.connect('ac|geom|vstab|S_ref', 'empennage.vertical_tail.surface')
        model.connect('ac|geom|hstab|S_ref', 'empennage.vertical_tail.surface_hor')  # Kv factor needs hstab surface
        model.connect('ac|geom|vstab|c2_sweep', 'empennage.vertical_tail.c2_sweep')
        
        # Landing Gear
        model.connect('ac|geom|landing_gear|mlg_height', 'mlg_height')
        model.connect('ac|geom|landing_gear|nlg_height', 'nlg_height')
        
        # Flight Controls
        model.connect('ac|geom|flight_controls|max_mach', 'max_mach')
        model.connect('ac|geom|wing|control_surface_area_ratio', 'control_surface_ratio')
        
        # Cabin
        model.connect('ac|geom|cabin|max_passengers', 'max_passengers')
        model.connect('ac|geom|cabin|business_class_pax', 'business_class_pax')
        model.connect('ac|geom|cabin|percent_cross_section', 'cabin_percent_cross_section')
        
        # Propulsion - Propeller
        model.connect('ac|propulsion|prop|diameter', 'blade_diameter')
        model.connect('ac|propulsion|prop|num_blades', 'num_blades')
        
        # Propulsion - Motor
        model.connect('ac|propulsion|motor|rating', ['rated_power_em_per_nacelle'])
        model.connect('ac|propulsion|motor|power_density', 'power_density')
        model.connect('ac|propulsion|motor|n_engines', 'num_em_per_nac')
        #model.connect('ac|propulsion|motor|HP', 'unit_rated_power_em')
        model.connect('ac|propulsion|motor|k', 'electric_k')
        model.connect('ac|propulsion|motor|RPMin', 'electric_RPMin')
        model.connect('ac|propulsion|motor|RPMout', 'electric_RPMout')
        
        # Propulsion - Turbine
        model.connect('ac|propulsion|turbine|rating', ['turbine_HP'])
        model.connect('ac|propulsion|nacelle|num_turb_per_nac', 'num_turb_per_nac')
        #model.connect('ac|propulsion|turbine|HP', 'turbine_HP')
        model.connect('ac|propulsion|turbine|k', 'turbine_k')
        model.connect('ac|propulsion|turbine|RPMin', 'turbine_RPMin')
        model.connect('ac|propulsion|turbine|RPMout', 'turbine_RPMout')
        
        # Propulsion - Gearbox
        model.connect('ac|propulsion|gearbox|counter_rotating', 'counter_rotating')
        
        # Battery (BatteryWeight uses batt_energy and batt_density from Excel)
        model.connect('ac|propulsion|battery|batt_energy', 'batt_energy')
        model.connect('ac|propulsion|battery|batt_density', 'batt_density')
        
        # Electrical
        #model.connect('ac|geom|electrical|engine_power', 'rated_power_em_per_nacelle')
        model.connect('ac|geom|fuselage|width', 'fuselage_width')
        model.connect('ac|geom|fuselage|height', 'fuselage_height')
        # Note: nacelle_distance and max_voltage no longer needed (HVWIS uses constant weight)
        #model.connect('ac|geom|electrical|nacelle_distance', 'nacelle_distance')
        #model.connect('ac|geom|electrical|max_voltage', 'max_voltage')
        
        # Thermal Management System
        model.connect('ac|propulsion|tms|motor_efficiency', 'efficiency')
        model.connect('ac|propulsion|tms|gearbox_heat', 'gearbox_heat')
        model.connect('ac|propulsion|tms|turbine_heat_per_nac', 'turbine_heat_per_nac')
        model.connect('ac|propulsion|battery|battery_heat_per_nacelle', 'battery_heat_per_nacelle')
        model.connect('ac|propulsion|tms|specific_cooling_capacity_em_tms', 'specific_cooling_capacity_em_tms')
        model.connect('ac|propulsion|tms|specific_cooling_capacity_turbine_tms', 'specific_cooling_capacity_turbine_tms')
        model.connect('ac|propulsion|tms|specific_cooling_capacity_battery_tms', 'specific_cooling_capacity_battery_tms')
        # Note: rated_power_em_per_nacelle is already connected above for propulsion
        model.connect('ac|geom|cabin|num_passengers', ['num_passengers'])

    if compute_payload:
        model.connect('ac|geom|cabin|num_rows_fwd', 'num_rows_fwd')
        model.connect('ac|geom|cabin|pitch', 'pitch')

        if not compute_oew: # Connect Parameter link values for payload CoG calculation
            model.connect('ac|geom|cabin|num_passengers', ['num_passengers'])
            model.connect('ac|geom|wing|span_trap', 'wing_parameter_links.span_trap')
            model.connect('ac|geom|wing|c0_sweep', 'wing_parameter_links.c0_sweep')
            model.connect('ac|geom|wing|taper', 'wing_parameter_links.taper_ratio')
            model.connect('ac|geom|fuselage|width', 'fuse_parameter_links.base')
            model.connect('ac|geom|fuselage|height', 'fuse_parameter_links.height')
            model.connect('ac|geom|cabin|cabin_length', 'fuse_parameter_links.cabin_length')
            model.connect('ac|geom|fuselage|nose_tail_length', 'fuse_parameter_links.nose_tail_length')
            model.connect('ac|geom|wing|apex_percentage', 'wing_apex_percentage')
            model.connect('ac|geom|wing|MAC', 'MAC')

def run_oew_analysis():
    """
    Main function to run OEW analysis using aircraft data from Excel.
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
    
    print("Creating OpenMDAO problem...")
    
    # Create OpenMDAO Problem
    prob = om.Problem(reports=False)
    model = prob.model
    
    # Add DictIndepVarComp with aircraft data
    dv_comp = DictIndepVarComp(ac_data)
    dv_comp = setup_ac_dv(dv_comp, compute_oew=True, compute_payload=True)
    model.add_subsystem('ac_data', dv_comp, promotes_outputs=['*'])
    
    # Add the OEW Group
    model.add_subsystem('oew', OEWGroup(), promotes_inputs=['*'], promotes_outputs=['*'])
    
    # Add Payload
    model.add_subsystem('payload', PayloadWeight(), 
                        promotes_inputs=['*'],
                        promotes_outputs=['payload_weight', 'cg_passenger', 'passenger_weight',
                                         'cargo_weight', 'cg_cargo'])
    
    # Add Fuel
    model.add_subsystem('fuel', FuelWeight(), 
                        promotes_inputs=['*'],
                        promotes_outputs=['fuel_weight', 'cg_fuel'])
    
    # Connect ac_data to OEW Group inputs
    connect_oew_inputs(model, compute_payload=True, compute_oew=True)
    
    # ========== CONNECTIONS: OEW outputs -> Payload/Fuel ==========
    # Payload needs MAC and LEMAC from wing_parameter_links (computed in OEW)
    
    # Setup and run
    print("Setting up problem...")
    prob.setup()
    om.n2(prob)

    print("Running model...")
    prob.run_model()

    # Print results
    print("\n" + "=" * 70)
    print("OEW ANALYSIS RESULTS (from ac_data.xlsx)")
    print("=" * 70)
    
    oew = prob.get_val('OEW', units='lbm')[0]
    cg_total = prob.get_val('cg_total', units='inch')[0]
    MAC = prob.get_val('ac|geom|wing|MAC', units='inch')[0]
    LEMAC = prob.get_val('LEMAC', units='inch')[0]
    
    oew_percent_mac = ((cg_total - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
    
    print(f"\nOperating Empty Weight (OEW): {oew:.2f} lbm")
    print(f"OEW C.G Location: {cg_total:.2f} in")
    print(f"OEW C.G %MAC: {oew_percent_mac:.2f}%")
    
    # Key component weights
    print("\nKey Component Weights:")
    print("-" * 50)
    print(f"  Fuselage:        {prob.get_val('Wf_total', units='lbm')[0]:.2f} lbm")
    print(f"  Wing:            {prob.get_val('wing_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Empennage:       {prob.get_val('empennage_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Nacelles:        {prob.get_val('total_nacelle_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Landing Gear:    {prob.get_val('landing_gear_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Propulsion:      {prob.get_val('total_propulsion_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Batteries:       {prob.get_val('total_battery_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Air Conditioning:{prob.get_val('air_conditioning_weight', units='lbm')[0]:.2f} lbm")
    print(f"  Interiors:       {prob.get_val('interiors_weight', units='lbm')[0]:.2f} lbm")
    
    # Calculate ZFW and TOW
    payload = prob.get_val('payload_weight', units='lbm')[0]
    fuel = prob.get_val('fuel_weight', units='lbm')[0]
    zfw = oew + payload
    tow = zfw + fuel
    
    print(f"\nPayload:           {payload:.2f} lbm")
    print(f"Fuel:              {fuel:.2f} lbm")
    print(f"\nZero Fuel Weight:  {zfw:.2f} lbm")
    print(f"Takeoff Weight:    {tow:.2f} lbm")
    print("=" * 70)
    
    # Export to text file (pass ac_data to include all input parameters)
    export_to_text_file(prob, ac_data=ac_data)
    
    # Create pie chart of ATA weights
    print("\nGenerating ATA weight breakdown pie chart...")
    
    # Collect all component weights
    weights = {
        'Fuselage': prob.get_val('Wf_total', units='lbm')[0],
        'Wing': prob.get_val('wing_weight', units='lbm')[0],
        'Empennage': prob.get_val('empennage_weight', units='lbm')[0],
        'Nacelles': prob.get_val('total_nacelle_weight', units='lbm')[0],
        'Landing Gear': prob.get_val('landing_gear_weight', units='lbm')[0],
        'Propulsion': prob.get_val('total_propulsion_weight', units='lbm')[0],
        'Batteries': prob.get_val('total_battery_weight', units='lbm')[0],
        'Air Conditioning': prob.get_val('air_conditioning_weight', units='lbm')[0],
        'Interiors': prob.get_val('interiors_weight', units='lbm')[0],
    }
    
    # Filter out zero weights and sort by value
    weights = {k: v for k, v in weights.items() if v > 0.01}
    sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = list(sorted_weights.keys())
    values = list(sorted_weights.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create pie chart with percentage labels
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                      startangle=90, colors=colors,
                                      textprops={'fontsize': 10})
    
    # Enhance percentage text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    # Add title
    ax.set_title(f'ATA Weight Breakdown\nOperating Empty Weight: {oew:.0f} lbm',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with absolute values
    legend_labels = [f'{k}: {v:.0f} lbm' for k, v in sorted_weights.items()]
    ax.legend(wedges, legend_labels, title="Component Weights",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(script_dir, '..', '..', '..', '..')
    output_path = os.path.join(output_dir, 'oew_weight_breakdown.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Pie chart saved to: {output_path}")
    
    return prob


if __name__ == '__main__':
    run_oew_analysis()
