import os
import sys
sys.path.append(os.path.dirname(__file__))
from datetime import datetime
import openmdao.api as om
import numpy as np

# Import all weight components
from atlas.weights.fuselage import FuselageMass
from atlas.weights.windows import WindowsMass
from atlas.weights.wing import WingWeight
from atlas.weights.nacelle import NacelleGroup
from atlas.weights.landing_gear import LandingGearWeight
from atlas.weights.flight_controls import FlightControlsWeight
from atlas.weights.empennage import EmpennageGroup
from atlas.weights.propulsion import PropulsionGroup, FuelInertingWeight, FuelSystemWeight
from atlas.weights.air_conditioning import AirConditioningWeight
from atlas.weights.batteries import BatteryWeight
from atlas.weights.avionics import AvionicsWeight
from atlas.weights.electrical import ElectricalGroup
from atlas.weights.payload import PayloadWeight
from atlas.weights.fuel import FuelWeight
from atlas.weights.thermal_management import TMSGroup
from atlas.weights.mechanical_systems import MechanicalSystemsGroup
from atlas.weights.furnishing import InteriorsWeight, PaintWeight, OperationalItemsWeight
from atlas.weights.parameter_links import NacelleParameterLinks, WingParameterLinks, FuselageParameterLinks, EMotorPowerLink
from atlas.utils import ElementMultiplyDivideComp

class TotalCGCalculator(om.ExplicitComponent):
    """
    Calculates total C.G as weighted average of all component C.Gs.
    """
    
    def setup(self):
        # All component weights
        self.add_input('Wf_total', val=0.0, units='lbm')
        self.add_input('total_nacelle_weight', val=0.0, units='lbm')
        self.add_input('empennage_weight', val=0.0, units='lbm')
        self.add_input('horizontal_tail_weight', val=0.0, units='lbm')
        self.add_input('vertical_tail_weight', val=0.0, units='lbm')
        self.add_input('W_total', val=0.0, units='lbm')
        self.add_input('wing_weight', val=0.0, units='lbm')
        self.add_input('air_conditioning_weight', val=0.0, units='lbm')
        self.add_input('battery_tms_weight_per_nac', val=0.0, units='lbm')
        self.add_input('fire_protection_weight', val=0.0, units='lbm')
        self.add_input('flight_controls_weight', val=0.0, units='lbm')
        self.add_input('ice_protection_weight', val=0.0, units='lbm')
        self.add_input('landing_gear_weight', val=0.0, units='lbm')
        self.add_input('nose_landing_gear_weight', val=0.0, units='lbm')
        self.add_input('main_landing_gear_weight', val=0.0, units='lbm')
        self.add_input('oxygen_system_weight', val=0.0, units='lbm')
        self.add_input('water_and_waste_weight', val=0.0, units='lbm')
        self.add_input('electrical_power_system_weight', val=0.0, units='lbm')
        self.add_input('avionics_weight', val=0.0, units='lbm')
        self.add_input('exterior_lighting_weight', val=0.0, units='lbm')
        self.add_input('lvwis_weight', val=0.0, units='lbm')
        self.add_input('hvwis_weight', val=0.0, units='lbm')
        self.add_input('fuel_system_weight', val=0.0, units='lbm')
        self.add_input('fuel_inerting_weight', val=0.0, units='lbm')
        self.add_input('total_propulsion_weight', val=0.0, units='lbm')
        self.add_input('propeller_weight', val=0.0, units='lbm')
        self.add_input('electric_motor_weight_per_nac', val=0.0, units='lbm')
        self.add_input('turbine_weight_per_nac', val=0.0, units='lbm')
        self.add_input('gearbox_weight', val=0.0, units='lbm')
        self.add_input('electric_motor_tms_weight_per_nac', val=0.0, units='lbm')
        self.add_input('turbine_tms_weight_per_nac', val=0.0, units='lbm')
        self.add_input('total_battery_weight', val=0.0, units='lbm')
        self.add_input('paint_weight', val=0.0, units='lbm')
        self.add_input('interiors_weight', val=0.0, units='lbm')
        self.add_input('operational_items_weight', val=0.0, units='lbm')
        self.add_input('OEW', val=0.0, units='lbm')
        
        # All component C.Gs
        self.add_input('cg_fuselage', val=0.0, units='inch')
        self.add_input('cg_total_nacelle', val=0.0, units='inch')
        self.add_input('cg_horizontal_tail', val=0.0, units='inch')
        self.add_input('cg_vertical_tail', val=0.0, units='inch')
        self.add_input('cg_windows', val=0.0, units='inch')
        self.add_input('cg_wing', val=0.0, units='inch')
        self.add_input('cg_air_conditioning', val=0.0, units='inch')
        self.add_input('cg_battery_tms', val=0.0, units='inch')
        self.add_input('cg_fire_protection', val=0.0, units='inch')
        self.add_input('cg_flight_controls', val=0.0, units='inch')
        self.add_input('cg_ice_protection', val=0.0, units='inch')
        self.add_input('cg_nose_landing_gear', val=0.0, units='inch')
        self.add_input('cg_main_landing_gear', val=0.0, units='inch')
        self.add_input('cg_oxygen_system', val=0.0, units='inch')
        self.add_input('cg_water_and_waste', val=0.0, units='inch')
        self.add_input('cg_electrical_power_system', val=0.0, units='inch')
        self.add_input('cg_avionics', val=0.0, units='inch')
        self.add_input('cg_exterior_lighting', val=0.0, units='inch')
        self.add_input('cg_lvwis', val=0.0, units='inch')
        self.add_input('cg_hvwis', val=0.0, units='inch')
        self.add_input('cg_fuel_system', val=0.0, units='inch')
        self.add_input('cg_fuel_inerting', val=0.0, units='inch')
        self.add_input('cg_propeller', val=0.0, units='inch')
        self.add_input('cg_electric_motor', val=0.0, units='inch')
        self.add_input('cg_turbine', val=0.0, units='inch')
        self.add_input('cg_gearbox', val=0.0, units='inch')
        self.add_input('cg_electric_motor_tms', val=0.0, units='inch')
        self.add_input('cg_turbine_tms', val=0.0, units='inch')
        self.add_input('cg_batteries', val=0.0, units='inch')
        self.add_input('cg_paint', val=0.0, units='inch')
        self.add_input('cg_interiors', val=0.0, units='inch')
        self.add_input('cg_operational_items', val=0.0, units='inch')
        
        self.add_output('cg_total', val=0.0, units='inch', desc='Total C.G location (weighted average)')
        
        # Declare partials for all inputs explicitly
        self.declare_partials('cg_total', [
            'Wf_total', 'total_nacelle_weight', 'empennage_weight', 'horizontal_tail_weight', 
            'vertical_tail_weight', 'W_total', 'wing_weight', 'air_conditioning_weight',
            'battery_tms_weight_per_nac', 'fire_protection_weight', 'flight_controls_weight',
            'ice_protection_weight', 'landing_gear_weight', 'nose_landing_gear_weight', 'main_landing_gear_weight', 'oxygen_system_weight',
            'water_and_waste_weight', 'electrical_power_system_weight', 'avionics_weight',
            'exterior_lighting_weight', 'lvwis_weight', 'hvwis_weight', 'fuel_system_weight',
            'fuel_inerting_weight', 'total_propulsion_weight', 'propeller_weight',
            'electric_motor_weight_per_nac', 'turbine_weight_per_nac', 'gearbox_weight',
            'electric_motor_tms_weight_per_nac', 'turbine_tms_weight_per_nac', 'total_battery_weight',
            'paint_weight', 'interiors_weight', 'operational_items_weight', 'OEW',
            'cg_fuselage', 'cg_total_nacelle', 'cg_horizontal_tail', 'cg_vertical_tail',
            'cg_windows', 'cg_wing', 'cg_air_conditioning', 'cg_battery_tms',
            'cg_fire_protection', 'cg_flight_controls', 'cg_ice_protection',
            'cg_nose_landing_gear', 'cg_main_landing_gear', 'cg_oxygen_system',
            'cg_water_and_waste', 'cg_electrical_power_system', 'cg_avionics',
            'cg_exterior_lighting', 'cg_lvwis', 'cg_hvwis', 'cg_fuel_system',
            'cg_fuel_inerting', 'cg_propeller', 'cg_electric_motor', 'cg_turbine',
            'cg_gearbox', 'cg_electric_motor_tms', 'cg_turbine_tms', 'cg_batteries',
            'cg_paint', 'cg_interiors', 'cg_operational_items'
        ])
    
    def compute(self, inputs, outputs):
        # Calculate weighted sum: sum(weight_i * cg_i)
        weighted_sum = (
            inputs['Wf_total'] * inputs['cg_fuselage'] +
            inputs['total_nacelle_weight'] * inputs['cg_total_nacelle'] +
            inputs['horizontal_tail_weight'] * inputs['cg_horizontal_tail'] +
            inputs['vertical_tail_weight'] * inputs['cg_vertical_tail'] +
            inputs['W_total'] * inputs['cg_windows'] +
            inputs['wing_weight'] * inputs['cg_wing'] +
            inputs['air_conditioning_weight'] * inputs['cg_air_conditioning'] +
            inputs['battery_tms_weight_per_nac'] * 4.0 * inputs['cg_battery_tms'] +
            inputs['fire_protection_weight'] * inputs['cg_fire_protection'] +
            inputs['flight_controls_weight'] * inputs['cg_flight_controls'] +
            inputs['ice_protection_weight'] * inputs['cg_ice_protection'] +
            # Landing gear: weighted average using actual weights
            inputs['nose_landing_gear_weight'] * inputs['cg_nose_landing_gear'] +
            inputs['main_landing_gear_weight'] * inputs['cg_main_landing_gear'] +
            inputs['oxygen_system_weight'] * inputs['cg_oxygen_system'] +
            inputs['water_and_waste_weight'] * inputs['cg_water_and_waste'] +
            inputs['electrical_power_system_weight'] * inputs['cg_electrical_power_system'] +
            inputs['avionics_weight'] * inputs['cg_avionics'] +
            inputs['exterior_lighting_weight'] * inputs['cg_exterior_lighting'] +
            inputs['lvwis_weight'] * inputs['cg_lvwis'] +
            inputs['hvwis_weight'] * inputs['cg_hvwis'] +
            inputs['fuel_system_weight'] * inputs['cg_fuel_system'] +
            inputs['fuel_inerting_weight'] * inputs['cg_fuel_inerting'] +
            # Propulsion: weighted average of components
            inputs['propeller_weight'] * 4.0 * inputs['cg_propeller'] +
            inputs['electric_motor_weight_per_nac'] * 4.0 * inputs['cg_electric_motor'] +
            inputs['turbine_weight_per_nac'] * 4.0 * inputs['cg_turbine'] +
            inputs['gearbox_weight'] * 4.0 * inputs['cg_gearbox'] +
            inputs['electric_motor_tms_weight_per_nac'] * 4.0 * inputs['cg_electric_motor_tms'] +
            inputs['turbine_tms_weight_per_nac'] * 4.0 * inputs['cg_turbine_tms'] +
            inputs['total_battery_weight'] * inputs['cg_batteries'] +
            inputs['paint_weight'] * inputs['cg_paint'] +
            inputs['interiors_weight'] * inputs['cg_interiors'] +
            inputs['operational_items_weight'] * inputs['cg_operational_items']
        )
        
        # Total C.G = weighted_sum / total_weight
        outputs['cg_total'] = weighted_sum / inputs['OEW']
    
    def compute_partials(self, inputs, partials):
        OEW = inputs['OEW']
        
        # Recalculate cg_total for partials
        weighted_sum = (
            inputs['Wf_total'] * inputs['cg_fuselage'] +
            inputs['total_nacelle_weight'] * inputs['cg_total_nacelle'] +
            inputs['horizontal_tail_weight'] * inputs['cg_horizontal_tail'] +
            inputs['vertical_tail_weight'] * inputs['cg_vertical_tail'] +
            inputs['W_total'] * inputs['cg_windows'] +
            inputs['wing_weight'] * inputs['cg_wing'] +
            inputs['air_conditioning_weight'] * inputs['cg_air_conditioning'] +
            inputs['battery_tms_weight_per_nac'] * 4.0 * inputs['cg_battery_tms'] +
            inputs['fire_protection_weight'] * inputs['cg_fire_protection'] +
            inputs['flight_controls_weight'] * inputs['cg_flight_controls'] +
            inputs['ice_protection_weight'] * inputs['cg_ice_protection'] +
            # Landing gear: weighted average using actual weights (same as compute)
            inputs['nose_landing_gear_weight'] * inputs['cg_nose_landing_gear'] +
            inputs['main_landing_gear_weight'] * inputs['cg_main_landing_gear'] +
            inputs['oxygen_system_weight'] * inputs['cg_oxygen_system'] +
            inputs['water_and_waste_weight'] * inputs['cg_water_and_waste'] +
            inputs['electrical_power_system_weight'] * inputs['cg_electrical_power_system'] +
            inputs['avionics_weight'] * inputs['cg_avionics'] +
            inputs['exterior_lighting_weight'] * inputs['cg_exterior_lighting'] +
            inputs['lvwis_weight'] * inputs['cg_lvwis'] +
            inputs['hvwis_weight'] * inputs['cg_hvwis'] +
            inputs['fuel_system_weight'] * inputs['cg_fuel_system'] +
            inputs['fuel_inerting_weight'] * inputs['cg_fuel_inerting'] +
            inputs['propeller_weight'] * 4.0 * inputs['cg_propeller'] +
            inputs['electric_motor_weight_per_nac'] * 4.0 * inputs['cg_electric_motor'] +
            inputs['turbine_weight_per_nac'] * 4.0 * inputs['cg_turbine'] +
            inputs['gearbox_weight'] * 4.0 * inputs['cg_gearbox'] +
            inputs['electric_motor_tms_weight_per_nac'] * 4.0 * inputs['cg_electric_motor_tms'] +
            inputs['turbine_tms_weight_per_nac'] * 4.0 * inputs['cg_turbine_tms'] +
            inputs['total_battery_weight'] * inputs['cg_batteries'] +
            inputs['paint_weight'] * inputs['cg_paint'] +
            inputs['interiors_weight'] * inputs['cg_interiors'] +
            inputs['operational_items_weight'] * inputs['cg_operational_items']
        )
        cg_total = weighted_sum / OEW
        
        # Partials w.r.t. weights
        partials['cg_total', 'Wf_total'] = (inputs['cg_fuselage'] - cg_total) / OEW
        partials['cg_total', 'total_nacelle_weight'] = (inputs['cg_total_nacelle'] - cg_total) / OEW
        partials['cg_total', 'horizontal_tail_weight'] = (inputs['cg_horizontal_tail'] - cg_total) / OEW
        partials['cg_total', 'vertical_tail_weight'] = (inputs['cg_vertical_tail'] - cg_total) / OEW
        partials['cg_total', 'W_total'] = (inputs['cg_windows'] - cg_total) / OEW
        partials['cg_total', 'wing_weight'] = (inputs['cg_wing'] - cg_total) / OEW
        partials['cg_total', 'air_conditioning_weight'] = (inputs['cg_air_conditioning'] - cg_total) / OEW
        partials['cg_total', 'battery_tms_weight_per_nac'] = (4.0 * inputs['cg_battery_tms'] - cg_total) / OEW
        partials['cg_total', 'fire_protection_weight'] = (inputs['cg_fire_protection'] - cg_total) / OEW
        partials['cg_total', 'flight_controls_weight'] = (inputs['cg_flight_controls'] - cg_total) / OEW
        partials['cg_total', 'ice_protection_weight'] = (inputs['cg_ice_protection'] - cg_total) / OEW
        # Landing gear partials: use actual weights (same as compute)
        partials['cg_total', 'nose_landing_gear_weight'] = (inputs['cg_nose_landing_gear'] - cg_total) / OEW
        partials['cg_total', 'main_landing_gear_weight'] = (inputs['cg_main_landing_gear'] - cg_total) / OEW
        # landing_gear_weight is sum of nose + main, so partial is sum of individual partials
        partials['cg_total', 'landing_gear_weight'] = partials['cg_total', 'nose_landing_gear_weight'] + partials['cg_total', 'main_landing_gear_weight']
        partials['cg_total', 'oxygen_system_weight'] = (inputs['cg_oxygen_system'] - cg_total) / OEW
        partials['cg_total', 'water_and_waste_weight'] = (inputs['cg_water_and_waste'] - cg_total) / OEW
        partials['cg_total', 'electrical_power_system_weight'] = (inputs['cg_electrical_power_system'] - cg_total) / OEW
        partials['cg_total', 'avionics_weight'] = (inputs['cg_avionics'] - cg_total) / OEW
        partials['cg_total', 'exterior_lighting_weight'] = (inputs['cg_exterior_lighting'] - cg_total) / OEW
        partials['cg_total', 'lvwis_weight'] = (inputs['cg_lvwis'] - cg_total) / OEW
        partials['cg_total', 'hvwis_weight'] = (inputs['cg_hvwis'] - cg_total) / OEW
        partials['cg_total', 'fuel_system_weight'] = (inputs['cg_fuel_system'] - cg_total) / OEW
        partials['cg_total', 'fuel_inerting_weight'] = (inputs['cg_fuel_inerting'] - cg_total) / OEW
        partials['cg_total', 'propeller_weight'] = (4.0 * inputs['cg_propeller'] - cg_total) / OEW
        partials['cg_total', 'electric_motor_weight_per_nac'] = (4.0 * inputs['cg_electric_motor'] - cg_total) / OEW
        partials['cg_total', 'turbine_weight_per_nac'] = (4.0 * inputs['cg_turbine'] - cg_total) / OEW
        partials['cg_total', 'gearbox_weight'] = (4.0 * inputs['cg_gearbox'] - cg_total) / OEW
        partials['cg_total', 'electric_motor_tms_weight_per_nac'] = (4.0 * inputs['cg_electric_motor_tms'] - cg_total) / OEW
        partials['cg_total', 'turbine_tms_weight_per_nac'] = (4.0 * inputs['cg_turbine_tms'] - cg_total) / OEW
        partials['cg_total', 'total_battery_weight'] = (inputs['cg_batteries'] - cg_total) / OEW
        partials['cg_total', 'paint_weight'] = (inputs['cg_paint'] - cg_total) / OEW
        partials['cg_total', 'interiors_weight'] = (inputs['cg_interiors'] - cg_total) / OEW
        partials['cg_total', 'operational_items_weight'] = (inputs['cg_operational_items'] - cg_total) / OEW
        partials['cg_total', 'OEW'] = -weighted_sum / (OEW ** 2)
        
        # Partials w.r.t. C.Gs
        partials['cg_total', 'cg_fuselage'] = inputs['Wf_total'] / OEW
        partials['cg_total', 'cg_total_nacelle'] = inputs['total_nacelle_weight'] / OEW
        partials['cg_total', 'cg_horizontal_tail'] = inputs['horizontal_tail_weight'] / OEW
        partials['cg_total', 'cg_vertical_tail'] = inputs['vertical_tail_weight'] / OEW
        partials['cg_total', 'cg_windows'] = inputs['W_total'] / OEW
        partials['cg_total', 'cg_wing'] = inputs['wing_weight'] / OEW
        partials['cg_total', 'cg_air_conditioning'] = inputs['air_conditioning_weight'] / OEW
        partials['cg_total', 'cg_battery_tms'] = inputs['battery_tms_weight_per_nac'] * 4.0 / OEW
        partials['cg_total', 'cg_fire_protection'] = inputs['fire_protection_weight'] / OEW
        partials['cg_total', 'cg_flight_controls'] = inputs['flight_controls_weight'] / OEW
        partials['cg_total', 'cg_ice_protection'] = inputs['ice_protection_weight'] / OEW
        partials['cg_total', 'cg_nose_landing_gear'] = inputs['nose_landing_gear_weight'] / OEW
        partials['cg_total', 'cg_main_landing_gear'] = inputs['main_landing_gear_weight'] / OEW
        partials['cg_total', 'cg_oxygen_system'] = inputs['oxygen_system_weight'] / OEW
        partials['cg_total', 'cg_water_and_waste'] = inputs['water_and_waste_weight'] / OEW
        partials['cg_total', 'cg_electrical_power_system'] = inputs['electrical_power_system_weight'] / OEW
        partials['cg_total', 'cg_avionics'] = inputs['avionics_weight'] / OEW
        partials['cg_total', 'cg_exterior_lighting'] = inputs['exterior_lighting_weight'] / OEW
        partials['cg_total', 'cg_lvwis'] = inputs['lvwis_weight'] / OEW
        partials['cg_total', 'cg_hvwis'] = inputs['hvwis_weight'] / OEW
        partials['cg_total', 'cg_fuel_system'] = inputs['fuel_system_weight'] / OEW
        partials['cg_total', 'cg_fuel_inerting'] = inputs['fuel_inerting_weight'] / OEW
        partials['cg_total', 'cg_propeller'] = inputs['propeller_weight'] * 4.0 / OEW
        partials['cg_total', 'cg_electric_motor'] = inputs['electric_motor_weight_per_nac'] * 4.0 / OEW
        partials['cg_total', 'cg_turbine'] = inputs['turbine_weight_per_nac'] * 4.0 / OEW
        partials['cg_total', 'cg_gearbox'] = inputs['gearbox_weight'] * 4.0 / OEW
        partials['cg_total', 'cg_electric_motor_tms'] = inputs['electric_motor_tms_weight_per_nac'] * 4.0 / OEW
        partials['cg_total', 'cg_turbine_tms'] = inputs['turbine_tms_weight_per_nac'] * 4.0 / OEW
        partials['cg_total', 'cg_batteries'] = inputs['total_battery_weight'] / OEW
        partials['cg_total', 'cg_paint'] = inputs['paint_weight'] / OEW
        partials['cg_total', 'cg_interiors'] = inputs['interiors_weight'] / OEW
        partials['cg_total', 'cg_operational_items'] = inputs['operational_items_weight'] / OEW


class OEWGroup(om.Group):
    """
    Operating Empty Weight (OEW) Group.
    
    This group contains all weight components and sums them together
    using AddSubtractComp to compute the total OEW.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Add all weight components/groups
        
        # FIRST: Add parameter link components that compute derived values needed by other components
        # Link fuselage parameters: computes fuselage_length from num_passengers, then lt and Sg from fuselage_length
        # Don't promote base/height - connect them explicitly as fuse.base, fuse.height
        self.add_subsystem('fuse_parameter_links', FuselageParameterLinks(), 
                          promotes_inputs=['num_passengers'],
                          promotes_outputs=['length_pax_cabin', 'fuselage_length', 'lt'])
        
        # Link wing geometry parameters to compute LEMAC and front_spar_location
        # Don't promote taper_ratio, span, c0_sweep - connect them explicitly
        # MAC is promoted so BatteryWeight can use it
        self.add_subsystem('wing_parameter_links', WingParameterLinks(), 
                          promotes_inputs=['wing_apex_percentage', 'fuselage_length', 'MAC'],
                          promotes_outputs=['LEMAC', 'front_spar_location'])
        
        # BatteryWeight uses batt_energy and batt_density from Excel, plus MAC and LEMAC
        # Use promotes_inputs=['*'] like other components to fix LEMAC issue
        self.add_subsystem('battery', BatteryWeight(), 
                          promotes_inputs=['*'],
                          promotes_outputs=['total_battery_weight', 'cg_batteries'])
        # Link energy to nacelle lengths
        self.add_subsystem('nacelle_geom_links', NacelleParameterLinks(), promotes=['*'])
        
        # Link nacelle-level motor power to unit motor power
        self.add_subsystem('emotor_power_link', EMotorPowerLink(), promotes_inputs=['*'],
                          promotes_outputs=['unit_rated_power_em'])
        
        # Structures
        # Fuselage: don't promote base/height - connect explicitly as fuse.base, fuse.height
        self.add_subsystem('fuse', FuselageMass(), 
                          promotes_inputs=['lt', 'fuselage_length', 'Vd', 'Sg', 'delta_p'],
                          promotes_outputs=['Wf_total', 'cg_fuselage'])
        
        self.add_subsystem('windows', WindowsMass(), promotes_inputs=['*'],
                          promotes_outputs=['W_total', 'cg_windows'])
        
        
        self.add_subsystem('propulsion', PropulsionGroup(), promotes_inputs=['*'],
                          promotes_outputs=['total_propulsion_weight', 'propeller_weight', 
                                           'electric_motor_weight_per_nac', 'turbine_weight_per_nac', 'gearbox_weight',
                                           'fuel_inerting_weight', 'fuel_system_weight',
                                           'cg_propeller', 'cg_electric_motor', 'cg_turbine', 'cg_gearbox',
                                           'cg_fuel_inerting', 'cg_fuel_system'])
        
        
        self.add_subsystem('nacelle', NacelleGroup(), promotes_inputs=['*'],
                          promotes_outputs=['total_nacelle_weight', 'nacelle_weight_inboard', 'nacelle_weight_outboard',
                                           'cg_total_nacelle'])
        
        self.add_subsystem('empennage', EmpennageGroup(), 
                          promotes_inputs=['Vd', 'fuselage_length'],
                          promotes_outputs=['empennage_weight', 'horizontal_tail_weight', 'vertical_tail_weight',
                                           'cg_horizontal_tail', 'cg_vertical_tail'])
        
        self.add_subsystem('landing_gear', LandingGearWeight(), promotes_inputs=['*'],
                          promotes_outputs=['landing_gear_weight', 'nose_landing_gear_weight', 'main_landing_gear_weight',
                                           'cg_nose_landing_gear', 'cg_main_landing_gear'])
        
        # Systems
        self.add_subsystem('flight_controls', FlightControlsWeight(), promotes_inputs=['*'],
                          promotes_outputs=['flight_controls_weight', 'cg_flight_controls'])
        


        self.add_subsystem('air_conditioning', AirConditioningWeight(), promotes_inputs=['*'],
                          promotes_outputs=['air_conditioning_weight', 'cg_air_conditioning'])
        
        #self.add_subsystem('battery', BatteryWeight(), promotes_inputs=['*'],
        #                  promotes_outputs=['total_battery_weight', 'cg_batteries'])


        
        # Compute full_nacelle_weight = nacelle structure + propulsion + batteries
        # This is used for the CAYE factor calculation in wing.py
        full_nacelle_comp = om.ExecComp(
            'full_nacelle_weight = total_nacelle_weight + total_propulsion_weight + total_battery_weight',
            total_nacelle_weight={'val': 0.0, 'units': 'lbm'},
            total_propulsion_weight={'val': 0.0, 'units': 'lbm'},
            total_battery_weight={'val': 0.0, 'units': 'lbm'},
            full_nacelle_weight={'val': 0.0, 'units': 'lbm'}
        )
        self.add_subsystem('full_nacelle_calc', full_nacelle_comp, promotes=['*'])
        
        # Wing: don't promote geometry inputs - connect explicitly as wing.<input>
        # full_nacelle_weight is calculated by full_nacelle_calc ExecComp which sums:
        #   total_nacelle_weight (from NacelleGroup) + 
        #   total_propulsion_weight (from PropulsionGroup: EM + Turbine + Gearbox + Propellers) + 
        #   total_battery_weight (from BatteryWeight)
        # Since full_nacelle_calc promotes all inputs/outputs, OpenMDAO automatically connects them
        self.add_subsystem('wing', WingWeight(), 
                          promotes_inputs=['strut_bracing_factor', 'aeroelastic_tailoring_factor',
                                          'gross_weight', 'bending_material_weight_scaler',
                                          'composite_fraction', 'load_fraction', 'misc_weight_scaler',
                                          'shear_control_weight_scaler', 'var_sweep_weight_penalty',
                                          'weight_scaler', 'LEMAC','MAC', 'full_nacelle_weight'],
                          promotes_outputs=['wing_weight', 'cg_wing'])


        self.add_subsystem('avionics', AvionicsWeight(), promotes_inputs=['*'],
                          promotes_outputs=['instruments_weight', 'communications_weight', 
                                           'navigation_weight', 'ima_weight', 'avionics_weight',
                                           'cg_avionics'])
        
        self.add_subsystem('electrical', ElectricalGroup(hvwis_weight_type='constant'), promotes_inputs=['*'],
                          promotes_outputs=['total_electrical_weight', 'exterior_lighting_weight',
                                           'electrical_power_system_weight', 'lvwis_weight', 
                                           'hvwis_weight', 'cabin_volume', 'electric_power',
                                           'cg_exterior_lighting', 'cg_lvwis', 'cg_hvwis',
                                           'cg_electrical_power_system'])
        
        self.add_subsystem('tms', TMSGroup(), promotes_inputs=['*'],
                          promotes_outputs=['total_tms_weight_per_nac', 'electric_motor_tms_weight_per_nac',
                                           'turbine_tms_weight_per_nac', 'battery_tms_weight_per_nac',
                                           'cg_electric_motor_tms', 'cg_turbine_tms', 'cg_battery_tms'])
        
        self.add_subsystem('mechanical_systems', MechanicalSystemsGroup(), promotes_inputs=['*'],
                          promotes_outputs=['total_mechanical_systems_weight', 'fire_protection_weight',
                                           'ice_protection_weight', 'oxygen_system_weight', 'water_and_waste_weight',
                                           'cg_fire_protection', 'cg_ice_protection', 'cg_oxygen_system',
                                           'cg_water_and_waste'])
        
        self.add_subsystem('interiors', InteriorsWeight(), promotes_inputs=['*'],
                          promotes_outputs=['interiors_weight', 'cg_interiors'])
        
        self.add_subsystem('paint', PaintWeight(), promotes_inputs=['*'],
                          promotes_outputs=['paint_weight', 'cg_paint'])
        
        self.add_subsystem('operational_items', OperationalItemsWeight(), promotes_inputs=['*'],
                          promotes_outputs=['operational_items_weight', 'cg_operational_items'])

        # Sum all weights using AddSubtractComp
        # Following ATA order for organization
        # Note: electric_motor_tms_weight_per_nac and turbine_tms_weight_per_nac are per-nacelle,
        #       so they need scaling_factor of 4.0 for 4 nacelles
        oew_adder = om.AddSubtractComp()
        oew_adder.add_equation(
            'OEW',
            input_names=[
                'Wf_total',                    # ATA 5300: Fuselage Structure
                'total_nacelle_weight',        # ATA 5400: Nacelle/Pylon Structure
                'empennage_weight',            # ATA 5500: Empennage Structure
                'W_total',                     # ATA 5600: Window/Windshield System
                'wing_weight',                 # ATA 5700: Wing Structure
                'air_conditioning_weight',     # ATA 2100: Air Conditioning (partial)
                'battery_tms_weight_per_nac',          # ATA 2100: Battery TMS (part of air conditioning)
                'fire_protection_weight',      # ATA 2600: Fire Protection System
                'flight_controls_weight',      # ATA 2700: Flight Control System
                'ice_protection_weight',       # ATA 3000: Ice/Rain Protection System
                'landing_gear_weight',         # ATA 3200: Landing Gear System
                'oxygen_system_weight',        # ATA 3500: Oxygen System
                'water_and_waste_weight',      # ATA 3800: Water and Waste System
                'electrical_power_system_weight',  # ATA 2400: Electrical Power System
                'instruments_weight',          # ATA 3100: Instruments
                'exterior_lighting_weight',    # ATA 3340: Exterior Lighting
                'lvwis_weight',                # ATA 9700: LVWIS
                'communications_weight',       # ATA 2300: Communications System
                'navigation_weight',           # ATA 3400: Navigation
                'ima_weight',                  # ATA 4200: IMA
                'fuel_system_weight',          # ATA 2800: Aircraft Fuel System
                'fuel_inerting_weight',        # ATA 4700: Fuel Inerting System
                'total_propulsion_weight',     # ATA 6100/7200/72E00: Propulsion (4x each component)
                'electric_motor_tms_weight_per_nac',   # ATA 7900: Electric Motor TMS (per nacelle, x4)
                'turbine_tms_weight_per_nac',          # ATA 7900: Turbine TMS (per nacelle, x4)
                'hvwis_weight',                # ATA 9710: HV Wiring (HVWIS)
                'total_battery_weight',              # ATA 8510: Batteries
                'paint_weight',                # ATA 1100: Paint
                'interiors_weight',            # ATA 2500: Cabin Equipment/furnishings
                'operational_items_weight',   # Operational Items
            ],
            scaling_factors=[
                1.0,  # Wf_total
                1.0,  # total_nacelle_weight
                1.0,  # empennage_weight
                1.0,  # W_total
                1.0,  # wing_weight
                1.0,  # air_conditioning_weight
                4.0,  # battery_tms_weight_per_nac (per nacelle x 4)
                1.0,  # fire_protection_weight
                1.0,  # flight_controls_weight
                1.0,  # ice_protection_weight
                1.0,  # landing_gear_weight
                1.0,  # oxygen_system_weight
                1.0,  # water_and_waste_weight
                1.0,  # electrical_power_system_weight
                1.0,  # instruments_weight
                1.0,  # exterior_lighting_weight
                1.0,  # lvwis_weight
                1.0,  # communications_weight
                1.0,  # navigation_weight
                1.0,  # ima_weight
                1.0,  # fuel_system_weight
                1.0,  # fuel_inerting_weight
                1.0,  # total_propulsion_weight
                4.0,  # electric_motor_tms_weight_per_nac (per nacelle x 4)
                4.0,  # turbine_tms_weight_per_nac (per nacelle x 4)
                1.0,  # hvwis_weight
                1.0,  # battery_weight
                1.0,  # paint_weight
                1.0,  # interiors_weight
                1.0,  # operational_items_weight
            ],
            units='lbm',
            desc='Operating Empty Weight'
        )
        self.add_subsystem('oew_sum', oew_adder, promotes=['*'])
        
        # Calculate total C.G as weighted average of all component C.Gs
        self.add_subsystem('cg_calculator', TotalCGCalculator(), promotes=['*'])


def export_to_text_file(prob, filename=None, ac_data=None, compute_payload_cg=True, compute_fuel_cg=True):
    """Export all analysis results to a text file.
    
    Parameters
    ----------
    prob : om.Problem
        The OpenMDAO problem containing the analysis results
    filename : str, optional
        Custom filename for the output file
    ac_data : dict, optional
        Aircraft data dictionary from load_ac_data_from_excel to include in output
    """
    import os
    
    # Create filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aircraft_weight_analysis_{timestamp}.txt"
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to aircraft_performance, then up one more level, then into aircraft_results
    # Path: weights -> atlas -> atlas -> models -> aircraft_performance -> (parent) -> aircraft_results
    aircraft_performance_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    results_dir = os.path.join(os.path.dirname(aircraft_performance_dir), 'aircraft_results')
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    # Create full path to save file in the aircraft_results directory
    filepath = os.path.join(results_dir, filename)
    
    # Extract all values from the problem
    Wf_total = prob.get_val('Wf_total', units='lbm')[0]
    W_total = prob.get_val('W_total', units='lbm')[0]
    wing_weight = prob.get_val('wing_weight', units='lbm')[0]
    total_nacelle_weight = prob.get_val('total_nacelle_weight', units='lbm')[0]
    empennage_weight = prob.get_val('empennage_weight', units='lbm')[0]
    landing_gear_weight = prob.get_val('landing_gear_weight', units='lbm')[0]
    flight_controls_weight = prob.get_val('flight_controls_weight', units='lbm')[0]
    total_propulsion_weight = prob.get_val('total_propulsion_weight', units='lbm')[0]
    air_conditioning_weight = prob.get_val('air_conditioning_weight', units='lbm')[0]
    battery_weight = prob.get_val('total_battery_weight', units='lbm')[0]
    
    # Avionics breakdown
    instruments_weight = prob.get_val('instruments_weight', units='lbm')[0]
    communications_weight = prob.get_val('communications_weight', units='lbm')[0]
    navigation_weight = prob.get_val('navigation_weight', units='lbm')[0]
    ima_weight = prob.get_val('ima_weight', units='lbm')[0]
    
    # Electrical breakdown
    exterior_lighting_weight = prob.get_val('exterior_lighting_weight', units='lbm')[0]
    electrical_power_system_weight = prob.get_val('electrical_power_system_weight', units='lbm')[0]
    lvwis_weight = prob.get_val('lvwis_weight', units='lbm')[0]
    hvwis_weight = prob.get_val('hvwis_weight', units='lbm')[0]
    total_electrical_weight = prob.get_val('total_electrical_weight', units='lbm')[0]
    
    # TMS breakdown
    electric_motor_tms_weight_per_nac = prob.get_val('electric_motor_tms_weight_per_nac', units='lbm')[0]
    turbine_tms_weight_per_nac = prob.get_val('turbine_tms_weight_per_nac', units='lbm')[0]
    battery_tms_weight_per_nac = prob.get_val('battery_tms_weight_per_nac', units='lbm')[0]
    
    # Mechanical systems breakdown
    fire_protection_weight = prob.get_val('fire_protection_weight', units='lbm')[0]
    ice_protection_weight = prob.get_val('ice_protection_weight', units='lbm')[0]
    oxygen_system_weight = prob.get_val('oxygen_system_weight', units='lbm')[0]
    water_and_waste_weight = prob.get_val('water_and_waste_weight', units='lbm')[0]
    
    # Furnishing breakdown
    interiors_weight = prob.get_val('interiors_weight', units='lbm')[0]
    paint_weight = prob.get_val('paint_weight', units='lbm')[0]
    operational_items_weight = prob.get_val('operational_items_weight', units='lbm')[0]
    
    # Propulsion breakdown
    propeller_weight = prob.get_val('propeller_weight', units='lbm')[0]
    electric_motor_weight_per_nac = prob.get_val('electric_motor_weight_per_nac', units='lbm')[0]
    turbine_weight_per_nac = prob.get_val('turbine_weight_per_nac', units='lbm')[0]
    gearbox_weight = prob.get_val('gearbox_weight', units='lbm')[0]
    fuel_inerting_weight = prob.get_val('fuel_inerting_weight', units='lbm')[0]
    fuel_system_weight = prob.get_val('fuel_system_weight', units='lbm')[0]
    
    # Nacelle breakdown
    nacelle_weight_inboard = prob.get_val('nacelle_weight_inboard', units='lbm')[0]
    nacelle_weight_outboard = prob.get_val('nacelle_weight_outboard', units='lbm')[0]
    
    # Payload and Fuel
    try:
        payload_weight = prob.get_val('payload_weight', units='lbm')[0]
    except Exception:
        payload_weight = prob.get_val('ac|weights|payload', units='lbm')[0]

    if compute_payload_cg:
        passenger_weight = prob.get_val('passenger_weight', units='lbm')[0]
        cargo_weight = prob.get_val('cargo_weight', units='lbm')[0]
        passenger_cg = prob.get_val('cg_passenger', units='inch')[0]
        cargo_cg = prob.get_val('cg_cargo', units='inch')[0]
        # Calculate total payload C.G (weighted average)
        total_payload_cg = (
            (passenger_weight * passenger_cg + cargo_weight * cargo_cg) / payload_weight
            if payload_weight > 0 else 0.0
        )
    else:
        passenger_weight = 0.0
        cargo_weight = 0.0
        passenger_cg = np.nan
        cargo_cg = np.nan
        total_payload_cg = np.nan
    
    try:
        fuel_weight = prob.get_val('fuel_weight', units='lbm')[0]
    except Exception:
        fuel_weight = prob.get_val('fuel_limit', units='lbm')[0]

    if compute_fuel_cg:
        fuel_cg = prob.get_val('cg_fuel', units='inch')[0]
    else:
        fuel_cg = np.nan
    
    # Empennage breakdown
    horizontal_tail_weight = prob.get_val('horizontal_tail_weight', units='lbm')[0]
    vertical_tail_weight = prob.get_val('vertical_tail_weight', units='lbm')[0]
    
    # Total OEW
    total_weight = prob.get_val('OEW', units='lbm')[0]
    total_cg = prob.get_val('cg_total', units='inch')[0]
    
    # Get MAC and LEMAC for %MAC calculations
    MAC = prob.get_val('MAC', units='inch')[0]
    LEMAC = prob.get_val('LEMAC', units='inch')[0]
    
    # Get C.G values
    cg_fuselage = prob.get_val('cg_fuselage', units='inch')[0]
    cg_total_nacelle = prob.get_val('cg_total_nacelle', units='inch')[0]
    cg_horizontal_tail = prob.get_val('cg_horizontal_tail', units='inch')[0]
    cg_vertical_tail = prob.get_val('cg_vertical_tail', units='inch')[0]
    cg_windows = prob.get_val('cg_windows', units='inch')[0]
    cg_wing = prob.get_val('cg_wing', units='inch')[0]
    cg_air_conditioning = prob.get_val('cg_air_conditioning', units='inch')[0]
    cg_battery_tms = prob.get_val('cg_battery_tms', units='inch')[0]
    cg_fire_protection = prob.get_val('cg_fire_protection', units='inch')[0]
    cg_flight_controls = prob.get_val('cg_flight_controls', units='inch')[0]
    cg_ice_protection = prob.get_val('cg_ice_protection', units='inch')[0]
    cg_nose_landing_gear = prob.get_val('cg_nose_landing_gear', units='inch')[0]
    cg_main_landing_gear = prob.get_val('cg_main_landing_gear', units='inch')[0]
    cg_oxygen_system = prob.get_val('cg_oxygen_system', units='inch')[0]
    cg_water_and_waste = prob.get_val('cg_water_and_waste', units='inch')[0]
    cg_electrical_power_system = prob.get_val('cg_electrical_power_system', units='inch')[0]
    cg_avionics = prob.get_val('cg_avionics', units='inch')[0]
    cg_exterior_lighting = prob.get_val('cg_exterior_lighting', units='inch')[0]
    cg_lvwis = prob.get_val('cg_lvwis', units='inch')[0]
    cg_hvwis = prob.get_val('cg_hvwis', units='inch')[0]
    cg_fuel_system = prob.get_val('cg_fuel_system', units='inch')[0]
    cg_fuel_inerting = prob.get_val('cg_fuel_inerting', units='inch')[0]
    cg_propeller = prob.get_val('cg_propeller', units='inch')[0]
    cg_electric_motor = prob.get_val('cg_electric_motor', units='inch')[0]
    cg_turbine = prob.get_val('cg_turbine', units='inch')[0]
    cg_gearbox = prob.get_val('cg_gearbox', units='inch')[0]
    cg_electric_motor_tms = prob.get_val('cg_electric_motor_tms', units='inch')[0]
    cg_turbine_tms = prob.get_val('cg_turbine_tms', units='inch')[0]
    cg_batteries = prob.get_val('cg_batteries', units='inch')[0]
    cg_paint = prob.get_val('cg_paint', units='inch')[0]
    cg_interiors = prob.get_val('cg_interiors', units='inch')[0]
    cg_operational_items = prob.get_val('cg_operational_items', units='inch')[0]
    
    # Calculate composite C.G values
    # Empennage C.G (weighted average of horizontal and vertical tail)
    emp_cg = (horizontal_tail_weight * cg_horizontal_tail + vertical_tail_weight * cg_vertical_tail) / (horizontal_tail_weight + vertical_tail_weight) if (horizontal_tail_weight + vertical_tail_weight) > 0 else 0.0
    
    # Air Conditioning C.G (weighted average of air conditioning and battery TMS)
    # Note: battery_tms_weight_per_nac is per nacelle, so multiply by 4 for total
    battery_tms_weight_per_nac_total = battery_tms_weight_per_nac * 4.0
    ac_cg = (air_conditioning_weight * cg_air_conditioning + battery_tms_weight_per_nac_total * cg_battery_tms) / (air_conditioning_weight + battery_tms_weight_per_nac_total) if (air_conditioning_weight + battery_tms_weight_per_nac_total) > 0 else 0.0
    
    # Landing Gear C.G (weighted average using actual weights)
    nose_lg_wt = prob.get_val('nose_landing_gear_weight', units='lbm')[0]
    main_lg_wt = prob.get_val('main_landing_gear_weight', units='lbm')[0]
    lg_cg = (nose_lg_wt * cg_nose_landing_gear + main_lg_wt * cg_main_landing_gear) / (nose_lg_wt + main_lg_wt) if (nose_lg_wt + main_lg_wt) > 0 else 0.0
    
    # Engine C.G (weighted average of turbine and gearbox)
    engine_cg = (turbine_weight_per_nac * cg_turbine + gearbox_weight * cg_gearbox) / (turbine_weight_per_nac + gearbox_weight) if (turbine_weight_per_nac + gearbox_weight) > 0 else 0.0
    
    # TMS C.G (weighted average of electric motor TMS and turbine TMS)
    tms_cg = (electric_motor_tms_weight_per_nac * cg_electric_motor_tms + turbine_tms_weight_per_nac * cg_turbine_tms) / (electric_motor_tms_weight_per_nac + turbine_tms_weight_per_nac) if (electric_motor_tms_weight_per_nac + turbine_tms_weight_per_nac) > 0 else 0.0
    
    # Get input parameters
    lt = prob.get_val('lt', units='ft')[0]
    bf = prob.get_val('bf', units='ft')[0]
    hf = prob.get_val('hf', units='ft')[0]
    Vd = prob.get_val('Vd', units='knot')[0]
    Sg = prob.get_val('Sg', units='ft**2')[0]
    delta_p = prob.get_val('delta_p', units='psi')[0]
    try:
        fuse_n_ult = prob.get_val('fuse_n_ult', units=None)[0]
    except:
        fuse_n_ult = prob.get_val('fuse.n_ult', units=None)[0]
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AIRCRAFT WEIGHT ANALYSIS RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Weight Summary Section (organized by ATA order)
        f.write("WEIGHT SUMMARY (ATA ORDER)\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'ATA':<8} {'Component':<42} {'Weight (lbm)':>15} {'C.G (in)':>15}\n")
        f.write("-" * 110 + "\n")
        
        # ATA 5300: Fuselage Structure (General)
        f.write(f"{'5300':<8} {'Fuselage Structure (General)':<42} {Wf_total:>15.2f} {cg_fuselage:>15.2f}\n")
        
        # ATA 5400: Nacelle/Pylon Structure
        f.write(f"{'5400':<8} {'Nacelle/Pylon Structure':<42} {total_nacelle_weight:>15.2f} {cg_total_nacelle:>15.2f}\n")
        
        # ATA 5500: Empennage Structure
        f.write(f"{'5500':<8} {'Empennage Structure':<42} {empennage_weight:>15.2f} {emp_cg:>15.2f}\n")
        
        # ATA 5600: Window/Windshield System
        f.write(f"{'5600':<8} {'Window/Windshield System':<42} {W_total:>15.2f} {cg_windows:>15.2f}\n")
        
        # ATA 5700: Wing Structure
        f.write(f"{'5700':<8} {'Wing Structure':<42} {wing_weight:>15.2f} {cg_wing:>15.2f}\n")
        
        # ATA 2100: Air Conditioning and Pressurization
        f.write(f"{'2100':<8} {'Air Conditioning and Pressurization':<42} {air_conditioning_weight + battery_tms_weight_per_nac_total:>15.2f} {ac_cg:>15.2f}\n")
        
        # ATA 2600: Fire Protection System
        f.write(f"{'2600':<8} {'Fire Protection System':<42} {fire_protection_weight:>15.2f} {cg_fire_protection:>15.2f}\n")
        
        # ATA 2700: Flight Control System
        f.write(f"{'2700':<8} {'Flight Control System':<42} {flight_controls_weight:>15.2f} {cg_flight_controls:>15.2f}\n")
        
        # ATA 3000: Ice/Rain Protection System
        f.write(f"{'3000':<8} {'Ice/Rain Protection System':<42} {ice_protection_weight:>15.2f} {cg_ice_protection:>15.2f}\n")
        
        # ATA 3200: Landing Gear System
        f.write(f"{'3200':<8} {'Landing Gear System':<42} {landing_gear_weight:>15.2f} {lg_cg:>15.2f}\n")
        
        # ATA 3500: Oxygen System
        f.write(f"{'3500':<8} {'Oxygen System':<42} {oxygen_system_weight:>15.2f} {cg_oxygen_system:>15.2f}\n")
        
        # ATA 3800: Water and Waste System
        f.write(f"{'3800':<8} {'Water and Waste System':<42} {water_and_waste_weight:>15.2f} {cg_water_and_waste:>15.2f}\n")
        
        # ATA 2400: Electrical Power System
        f.write(f"{'2400':<8} {'Electrical Power System':<42} {electrical_power_system_weight:>15.2f} {cg_electrical_power_system:>15.2f}\n")
        
        # ATA 3100: Instruments
        f.write(f"{'3100':<8} {'Instruments':<42} {instruments_weight:>15.2f} {cg_avionics:>15.2f}\n")
        
        # ATA 3340: Exterior Lighting
        f.write(f"{'3340':<8} {'Exterior Lighting':<42} {exterior_lighting_weight:>15.2f} {cg_exterior_lighting:>15.2f}\n")
        
        # ATA 9700: Electrical Interconnect (LVWIS)
        f.write(f"{'9700':<8} {'Electrical Interconnect (LVWIS)':<42} {lvwis_weight:>15.2f} {cg_lvwis:>15.2f}\n")
        
        # ATA 2300: Communications System
        f.write(f"{'2300':<8} {'Communications System':<42} {communications_weight:>15.2f} {cg_avionics:>15.2f}\n")
        
        # ATA 3400: Navigation
        f.write(f"{'3400':<8} {'Navigation':<42} {navigation_weight:>15.2f} {cg_avionics:>15.2f}\n")
        
        # ATA 4200: IMA
        f.write(f"{'4200':<8} {'IMA':<42} {ima_weight:>15.2f} {cg_avionics:>15.2f}\n")
        
        # ATA 2800: Aircraft Fuel System
        f.write(f"{'2800':<8} {'Aircraft Fuel System':<42} {fuel_system_weight:>15.2f} {cg_fuel_system:>15.2f}\n")
        
        # ATA 4700: Fuel Inerting System
        f.write(f"{'4700':<8} {'Fuel Inerting System':<42} {fuel_inerting_weight:>15.2f} {cg_fuel_inerting:>15.2f}\n")
        
        # ATA 6100: Propeller System
        f.write(f"{'6100':<8} {'Propeller System':<42} {propeller_weight * 4.0:>15.2f} {cg_propeller:>15.2f}\n")
        
        # ATA 7200: Engine (Turbine/Turboprop)
        f.write(f"{'7200':<8} {'Engine (Turbine/Turboprop)':<42} {(turbine_weight_per_nac * 4.0 + gearbox_weight * 4.0):>15.2f} {engine_cg:>15.2f}\n")
        
        # ATA 72E00: Electric Engine
        f.write(f"{'72E00':<8} {'Electric Engine':<42} {electric_motor_weight_per_nac * 4.0:>15.2f} {cg_electric_motor:>15.2f}\n")
        
        # ATA 7900: IPS TMS & Oil System (per nacelle x 4)
        f.write(f"{'7900':<8} {'IPS TMS & Oil System':<42} {(electric_motor_tms_weight_per_nac + turbine_tms_weight_per_nac) * 4.0:>15.2f} {tms_cg:>15.2f}\n")
        
        # ATA 9710: HV Wiring (HVWIS)
        f.write(f"{'9710':<8} {'HV Wiring (HVWIS)':<42} {hvwis_weight:>15.2f} {cg_hvwis:>15.2f}\n")
        
        # ATA 8510: Batteries
        f.write(f"{'8510':<8} {'Batteries':<42} {battery_weight:>15.2f} {cg_batteries:>15.2f}\n")
        
        # ATA 1100: Paint, Placards, and Markings
        f.write(f"{'1100':<8} {'Paint, Placards, and Markings':<42} {paint_weight:>15.2f} {cg_paint:>15.2f}\n")
        
        # ATA 2500: Cabin Equipment/furnishings
        f.write(f"{'2500':<8} {'Cabin Equipment/furnishings':<42} {interiors_weight:>15.2f} {cg_interiors:>15.2f}\n")
        
        # Operational Items
        f.write(f"{'':<8} {'Operational Items':<42} {operational_items_weight:>15.2f} {cg_operational_items:>15.2f}\n")
        f.write("-" * 100 + "\n")
        # Calculate %MAC for OEW
        oew_percent_mac = ((total_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
        f.write(f"{'TOTAL OPERATING EMPTY WEIGHT (OEW)':<50} {total_weight:>15.2f} {total_cg:>15.2f} {oew_percent_mac:>10.2f}%\n")
        f.write("=" * 80 + "\n\n")
        
        # Payload
        f.write("PAYLOAD\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'ATA':<8} {'Component':<42} {'Weight (lbm)':>15} {'C.G (in)':>15}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'':<8} {'Payload':<42} {payload_weight:>15.2f} {total_payload_cg:>15.2f}\n")
        f.write("-" * 100 + "\n")
        # Calculate ZFW (Zero Fuel Weight) = OEW + Payload
        zfw = total_weight + payload_weight
        zfw_cg = (total_weight * total_cg + payload_weight * total_payload_cg) / zfw if zfw > 0 else 0.0
        zfw_percent_mac = ((zfw_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
        f.write(f"{'ZERO FUEL WEIGHT (ZFW)':<50} {zfw:>15.2f} {zfw_cg:>15.2f} {zfw_percent_mac:>10.2f}%\n")
        f.write("=" * 80 + "\n\n")
        
        # Fuel
        f.write("FUEL\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'ATA':<8} {'Component':<42} {'Weight (lbm)':>15} {'C.G (in)':>15}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'':<8} {'Fuel':<42} {fuel_weight:>15.2f} {fuel_cg:>15.2f}\n")
        f.write("-" * 100 + "\n")
        # Calculate TOW (Takeoff Weight) = ZFW + Fuel = OEW + Payload + Fuel
        tow = zfw + fuel_weight
        tow_cg = (zfw * zfw_cg + fuel_weight * fuel_cg) / tow if tow > 0 else 0.0
        tow_percent_mac = ((tow_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
        f.write(f"{'TAKEOFF WEIGHT (TOW)':<50} {tow:>15.2f} {tow_cg:>15.2f} {tow_percent_mac:>10.2f}%\n")
        f.write("=" * 80 + "\n\n")
        
        # Input Parameters Section
        f.write("INPUT PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Parameter':<50} {'Value':>20} {'Units':>10}\n")
        f.write("-" * 80 + "\n")
        
        # Write legacy fuselage parameters from OpenMDAO problem
        f.write(f"{'Fuselage - Tail Arm (lt)':<50} {lt:>20.2f} {'ft':>10}\n")
        f.write(f"{'Fuselage - Width (bf)':<50} {bf:>20.2f} {'ft':>10}\n")
        f.write(f"{'Fuselage - Height (hf)':<50} {hf:>20.2f} {'ft':>10}\n")
        f.write(f"{'Fuselage - Dive Speed (Vd)':<50} {Vd:>20.2f} {'knots':>10}\n")
        f.write(f"{'Fuselage - Gross Shell Area (Sg)':<50} {Sg:>20.2f} {'ft²':>10}\n")
        f.write(f"{'Fuselage - Pressure Differential (delta_p)':<50} {delta_p:>20.2f} {'psi':>10}\n")
        f.write(f"{'Fuselage - Ultimate Factor (fuse_n_ult)':<50} {fuse_n_ult:>20.2f} {'unitless':>10}\n")
        
        # Write all ac_data parameters if provided
        if ac_data is not None:
            f.write("\n" + "=" * 80 + "\n")
            f.write("AIRCRAFT DATA (from ac_data.xlsx)\n")
            f.write("=" * 80 + "\n")
            _write_ac_data_to_file(f, ac_data)
    
    print(f"\nAnalysis results exported to {filepath}")
    return filepath


def _write_ac_data_to_file(f, data, prefix=""):
    """
    Recursively write aircraft data dictionary to file.
    
    Parameters
    ----------
    f : file handle
        Open file to write to
    data : dict
        Aircraft data dictionary or sub-dictionary
    prefix : str
        Current path prefix for nested keys
    """
    for key, value in data.items():
        current_path = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Check if this is a leaf node (has 'value' key)
            if 'value' in value:
                val = value['value']
                units = value.get('units', None)
                units_str = units if units else 'unitless'
                
                # Format the value based on type
                if isinstance(val, float):
                    if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                        val_str = f"{val:>20.4e}"
                    else:
                        val_str = f"{val:>20.4f}"
                elif isinstance(val, int):
                    val_str = f"{val:>20d}"
                else:
                    val_str = f"{str(val):>20}"
                
                f.write(f"{current_path:<50} {val_str} {units_str:>10}\n")
            else:
                # This is a nested dictionary, add section header and recurse
                f.write(f"\n--- {current_path.upper()} ---\n")
                _write_ac_data_to_file(f, value, f"{current_path}|")
        else:
            # Direct value (shouldn't happen with properly structured ac_data)
            f.write(f"{current_path:<50} {str(value):>20}\n")


def main():
    # Create OpenMDAO Problem
    prob = om.Problem(reports=False)
    model = prob.model
    
    # Add IndepVarComp for all inputs
    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    
    # ========== FUSELAGE INPUTS ==========
    # Note: lt and Sg are computed by FuselageParameterLinks from num_passengers, fuse_base, fuse_height
    # They are NOT set as independent variables here
    ivc.add_output('fuse_base', val=10.16, units='ft', desc='Fuselage width')
    ivc.add_output('fuse_height', val=9.87, units='ft', desc='Fuselage height')
    ivc.add_output('Sg', val=2728.0, units='ft**2', desc='Gross shell area')
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
    # Wing geometry from ac_data.xlsx (computed by compute_wing_geometry in load_ac_data.py)
    ivc.add_output('wing_taper_ratio', val=0.35, units=None, desc='Wing taper ratio')
    ivc.add_output('wing_S_ref', val=912.91, units='ft**2', desc='Wing reference area')
    ivc.add_output('wing_AR', val=14.46, units=None, desc='Wing aspect ratio')
    ivc.add_output('wing_c4_sweep', val=2.8, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('wing_MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
    ivc.add_output('wing_span_plan', val=1328.0, units='inch', desc='Planform wing span (includes winglets)')
    ivc.add_output('wing_span_trap', val=1328.0, units='inch', desc='Trapezoidal wing span (for geometry calculations)')
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
    
    # Wing weight needs full_nacelle_weight for CAYE (inertia relief) factor calculation
    # full_nacelle_weight is computed by full_nacelle_calc ExecComp which sums:
    #   total_nacelle_weight (from NacelleGroup) + 
    #   total_propulsion_weight (from PropulsionGroup: EM + Turbine + Gearbox + Propellers) + 
    #   total_battery_weight (from BatteryWeight)
    
    # ========== NACELLE INPUTS ==========
    # Inboard nacelle
    #ivc.add_output('engine_weight_inboard', val=2000.0, units='lbm', desc='Inboard engine weight')
    #ivc.add_output('batteries_weight_inboard', val=4851.25, units='lbm', desc='Inboard batteries weight')
    # nacelle_length_inboard is computed by NacelleParameterLinks from energy
    ivc.add_output('nacelle_width_inboard', val=42.5/12.0, units='ft', desc='Inboard nacelle width')
    ivc.add_output('nacelle_height_inboard', val=92.5/12.0, units='ft', desc='Inboard nacelle height')
    # Outboard nacelle
    #ivc.add_output('engine_weight_outboard', val=2000.0, units='lbm', desc='Outboard engine weight')
    #ivc.add_output('batteries_weight_outboard', val=4851.25, units='lbm', desc='Outboard batteries weight')
    # nacelle_length_outboard is computed by NacelleParameterLinks from energy
    ivc.add_output('nacelle_width_outboard', val=38.3/12.0, units='ft', desc='Outboard nacelle width')
    ivc.add_output('nacelle_height_outboard', val=56.0/12.0, units='ft', desc='Outboard nacelle height')
    # Shared nacelle input
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
    ivc.add_output('wing_surface', val=912.91, units='ft**2', desc='Wing reference area for flight controls')
    
    # ========== PROPULSION INPUTS ==========
    ivc.add_output('num_blades', val=6, units=None, desc='Number of propeller blades')
    ivc.add_output('blade_diameter', val=13.0, units='ft', desc='Propeller blade diameter')
    # rated_power_em_per_nacelle is defined in ELECTRICAL INPUTS section below
    ivc.add_output('power_density', val=4.54, units='kW/lbm', desc='Power density')
    ivc.add_output('rated_power_em_per_nacelle', val=1341.0*2, units='hp', desc='Rated power electric motor per nacelle')
    ivc.add_output('rated_power_turbine', val=1200.0, units='hp', desc='Rated power turbine')
    # Gearbox inputs
    ivc.add_output('num_em_per_nac', val=2, units=None, desc='Number of electric engines')
    #ivc.add_output('unit_rated_power_em', val=1341.0, units='hp', desc='Electric engine horsepower')
    ivc.add_output('electric_k', val=125.7, units=None, desc='Electric engine technology factor')
    ivc.add_output('electric_RPMin', val=12240.0, units='rpm', desc='Electric engine input RPM')
    ivc.add_output('electric_RPMout', val=1200.0, units='rpm', desc='Electric engine output RPM')
    ivc.add_output('num_turb_per_nac', val=1, units=None, desc='Number of turbines')
    #ivc.add_output('turbine_HP', val=1200.0, units='hp', desc='Turbine horsepower')
    ivc.add_output('turbine_k', val=125.7, units=None, desc='Turbine technology factor')
    ivc.add_output('counter_rotating', val=True, units=None, desc='Boolean: True if counter-rotating feature is enabled')
    ivc.add_output('turbine_RPMin', val=30000.0, units='rpm', desc='Turbine input RPM')
    ivc.add_output('turbine_RPMout', val=1200.0, units='rpm', desc='Turbine output RPM')
    
    # ========== PASSENGER INPUTS ==========
    ivc.add_output('num_passengers', val=76, units=None, desc='Number of passengers')
    
    # ========== AIR CONDITIONING INPUTS ==========
    # length_pax_cabin is computed by FuselageParameterLinks from num_passengers and cabin_length (used by other components)
    ivc.add_output('max_passengers', val=88, units=None, desc='Maximum number of passengers for air conditioning calculation')
    
    # ========== BATTERY INPUTS ==========
    ivc.add_output('n_str', val=4.0)
    ivc.add_output('n_parallel_per_str', val=112.0)
    ivc.add_output('n_series_per_str', val=210.0)
    ivc.add_output('m_cell', val=0.070, units='kg')  # 70 grams per cell
    
    #ivc.add_output('energy', val=2640588.0, units='W*h', desc='Battery energy capacity')
    # Note: energy_density for battery uses different units than propulsion
    # Battery uses W*h/lbm, so we need a separate input
    battery_energy_density = 300.0 / 2.20462  # Wh/lbm (converted from 300 Wh/kg)
    # We'll rename this to avoid conflict
    
    # ========== ELECTRICAL INPUTS ==========
    #ivc.add_output('rated_power_em_per_nacelle', val=1864000.0, units='W', desc='Electrical engine power')
    ivc.add_output('fuselage_width', val=10.16, units='ft', desc='Fuselage width for LVWIS')
    ivc.add_output('fuselage_height', val=9.87, units='ft', desc='Fuselage height for LVWIS')
    ivc.add_output('cabin_percent_cross_section', val=0.81, units=None, desc='Cabin cross section percentage')
    ivc.add_output('nacelle_distance', val=179.5, units='ft', desc='Distance between inboard and outboard nacelles')
    ivc.add_output('max_voltage', val=570.0, units='V', desc='Maximum voltage')
    
    # ========== THERMAL MANAGEMENT INPUTS ==========
    ivc.add_output('efficiency', val=0.95, units=None, desc='Electric motor efficiency')
    # rated_power_em_per_nacelle is defined in ELECTRICAL INPUTS section below
    ivc.add_output('gearbox_heat', val=2500.0, units='W', desc='Gearbox heat generation')
    # Note: delta_T_em no longer needed (using kW/kg method)
    ivc.add_output('turbine_heat_per_nac', val=15000.0, units='W', desc='Turbine heat generation')
    # Note: delta_T_turb no longer needed (using kW/kg method)
    ivc.add_output('battery_heat_per_wing', val=50000.0, units='W', desc='Battery heat generation')
    
    # ========== FURNISHING INPUTS ==========
    ivc.add_output('business_class_pax', val=0, units=None, desc='Number of business class passengers')
    
    # ========== PAYLOAD INPUTS ==========
    ivc.add_output('num_rows_fwd', val=9, units=None, desc='Number of rows forward of OWEED')
    ivc.add_output('pitch', val=32.0, units='inch', desc='Seat pitch (spacing between rows)')

    # Connect Power Values
    # rated_power_em connects to rated_power_em_per_nacelle; unit_rated_power_em is computed by EMotorPowerLink
    model.connect('rated_power_turbine', 'turbine_HP')
    
    # Add the OEW Group
    model.add_subsystem('oew', OEWGroup(), promotes=['*'])
    
    # Connect fuselage inputs (not promoted to avoid name conflicts)
    model.connect('fuse_base', ['fuse.base', 'fuse_parameter_links.base','bf'])
    model.connect('fuse_height', ['fuse.height', 'fuse_parameter_links.height','hf'])
    model.connect('fuse_n_ult', 'fuse.n_ult')
    
    # Connect wing_parameter_links inputs (not promoted)
    # Use trapezoidal span for geometry/LEMAC calculations
    model.connect('wing_taper_ratio', ['wing.taper_ratio', 'wing_parameter_links.taper_ratio'])
    model.connect('wing_span_trap', 'wing_parameter_links.span_trap')
    model.connect('wing_c0_sweep', 'wing_parameter_links.c0_sweep')
    model.connect('wing_MAC', 'MAC')
    
    # Connect wing weight geometry inputs (not promoted)
    # Use planform span and area for weight calculations
    model.connect('wing_span_plan', 'wing.span_plan')
    model.connect('wing_S_ref', 'wing.S_ref')
    model.connect('wing_AR', 'wing.aspect_ratio')
    model.connect('wing_c4_sweep', ['wing.c4_sweep'])
    model.connect('wing_S_ref_control_ratio', 'wing.S_ref_control_ratio')
    model.connect('wing_toverc', 'wing.toverc')
    model.connect('wing_n_ult', 'wing.n_ult')
    
    # Connect empennage inputs (not promoted to avoid name conflicts)
    model.connect('hstab_S_ref', ['empennage.horizontal_tail.S_ref', 'empennage.vertical_tail.S_ref_hor'])
    model.connect('vstab_S_ref', 'empennage.vertical_tail.S_ref')
    model.connect('hstab_c2_sweep', 'empennage.horizontal_tail.c2_sweep')
    model.connect('vstab_c2_sweep', 'empennage.vertical_tail.c2_sweep')
    
    # Add Payload (separate from OEW - payload is not part of Operating Empty Weight)
    model.add_subsystem('payload', PayloadWeight(), promotes_inputs=['*'],
                       promotes_outputs=['payload_weight', 'cg_passenger', 'passenger_weight', 
                                       'cargo_weight', 'cg_cargo'])
    
    # Add Fuel (separate from OEW - fuel is not part of Operating Empty Weight)
    model.add_subsystem('fuel', FuelWeight(), promotes_inputs=['*'],
                       promotes_outputs=['fuel_weight', 'cg_fuel'])
    
    # Setup and run the problem
    prob.setup()
    om.n2(prob)
    
    # Set battery energy density (needs to be done after setup due to unit conversion)
    # BatteryWeight expects 'energy_density' in W*h/lbm
    #prob.set_val('oew.battery.energy_density', battery_energy_density, units='W*h/lbm')
    
    prob.run_model()
    
    # Print results in ATA order
    print("\n" + "=" * 60)
    print("OPERATING EMPTY WEIGHT (OEW) ANALYSIS RESULTS")
    print("=" * 60)
    
    # ATA 5300: Fuselage Structure (General)
    print("\nATA 5300 - Fuselage Structure (General):")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('Wf_total', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_fuselage', units='inch')[0]:.2f} in")

    # ATA 5400: Nacelle/Pylon Structure
    print("\nATA 5400 - Nacelle/Pylon Structure:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('total_nacelle_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_total_nacelle', units='inch')[0]:.2f} in")

    # ATA 5500: Empennage Structure
    print("\nATA 5500 - Empennage Structure:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('empennage_weight', units='lbm')[0]:.2f} lbm")
    # Empennage C.G is weighted average of horizontal and vertical tail
    h_tail_wt = prob.get_val('horizontal_tail_weight', units='lbm')[0]
    v_tail_wt = prob.get_val('vertical_tail_weight', units='lbm')[0]
    h_tail_cg = prob.get_val('cg_horizontal_tail', units='inch')[0]
    v_tail_cg = prob.get_val('cg_vertical_tail', units='inch')[0]
    emp_cg = (h_tail_wt * h_tail_cg + v_tail_wt * v_tail_cg) / (h_tail_wt + v_tail_wt) if (h_tail_wt + v_tail_wt) > 0 else 0.0
    print(f"C.G Location:     {emp_cg:.2f} in")

    # ATA 5600: Window/Windshield System
    print("\nATA 5600 - Window/Windshield System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('W_total', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_windows', units='inch')[0]:.2f} in")

    # ATA 5700: Wing Structure
    print("\nATA 5700 - Wing Structure:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('wing_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_wing', units='inch')[0]:.2f} in")

    # ATA 2100: Air Conditioning and Pressurization
    air_cond = prob.get_val('air_conditioning_weight', units='lbm')[0]
    batt_tms = prob.get_val('battery_tms_weight_per_nac', units='lbm')[0]
    air_cond_cg = prob.get_val('cg_air_conditioning', units='inch')[0]
    batt_tms_cg = prob.get_val('cg_battery_tms', units='inch')[0]
    ac_cg = (air_cond * air_cond_cg + batt_tms * batt_tms_cg) / (air_cond + batt_tms) if (air_cond + batt_tms) > 0 else 0.0
    print("\nATA 2100 - Air Conditioning and Pressurization:")
    print("-" * 45)
    print(f"Total Weight:     {air_cond + batt_tms:.2f} lbm")
    print(f"C.G Location:     {ac_cg:.2f} in")

    # ATA 2600: Fire Protection System
    print("\nATA 2600 - Fire Protection System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('fire_protection_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_fire_protection', units='inch')[0]:.2f} in")

    # ATA 2700: Flight Control System
    print("\nATA 2700 - Flight Control System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('flight_controls_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_flight_controls', units='inch')[0]:.2f} in")

    # ATA 3000: Ice/Rain Protection System
    print("\nATA 3000 - Ice/Rain Protection System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('ice_protection_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_ice_protection', units='inch')[0]:.2f} in")

    # ATA 3200: Landing Gear System
    print("\nATA 3200 - Landing Gear System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('landing_gear_weight', units='lbm')[0]:.2f} lbm")
    # Landing gear C.G is weighted average using actual weights
    nose_lg_wt = prob.get_val('nose_landing_gear_weight', units='lbm')[0]
    main_lg_wt = prob.get_val('main_landing_gear_weight', units='lbm')[0]
    nose_lg_cg = prob.get_val('cg_nose_landing_gear', units='inch')[0]
    main_lg_cg = prob.get_val('cg_main_landing_gear', units='inch')[0]
    lg_cg = (nose_lg_wt * nose_lg_cg + main_lg_wt * main_lg_cg) / (nose_lg_wt + main_lg_wt) if (nose_lg_wt + main_lg_wt) > 0 else 0.0
    print(f"C.G Location:     {lg_cg:.2f} in")
    print(f"  Nose Landing Gear:  {nose_lg_wt:.2f} lbm at {nose_lg_cg:.2f} in")
    print(f"  Main Landing Gear:  {main_lg_wt:.2f} lbm at {main_lg_cg:.2f} in")

    # ATA 3500: Oxygen System
    print("\nATA 3500 - Oxygen System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('oxygen_system_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_oxygen_system', units='inch')[0]:.2f} in")

    # ATA 3800: Water and Waste System
    print("\nATA 3800 - Water and Waste System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('water_and_waste_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_water_and_waste', units='inch')[0]:.2f} in")

    # ATA 2400: Electrical Power System
    print("\nATA 2400 - Electrical Power System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('electrical_power_system_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_electrical_power_system', units='inch')[0]:.2f} in")

    # ATA 3100: Instruments (Avionics)
    avionics_wt = prob.get_val('avionics_weight', units='lbm')[0]
    print("\nATA 3100 - Instruments:")
    print("-" * 45)
    print(f"Total Weight:     {avionics_wt:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_avionics', units='inch')[0]:.2f} in")

    # ATA 3340: Exterior Lighting
    print("\nATA 3340 - Exterior Lighting:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('exterior_lighting_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_exterior_lighting', units='inch')[0]:.2f} in")

    # ATA 9700: Electrical Interconnect (LVWIS)
    print("\nATA 9700 - Electrical Interconnect (LVWIS):")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('lvwis_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_lvwis', units='inch')[0]:.2f} in")

    # ATA 2300: Communications System (part of avionics)
    print("\nATA 2300 - Communications System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('communications_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_avionics', units='inch')[0]:.2f} in")

    # ATA 3400: Navigation (part of avionics)
    print("\nATA 3400 - Navigation:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('navigation_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_avionics', units='inch')[0]:.2f} in")

    # ATA 4200: IMA (part of avionics)
    print("\nATA 4200 - IMA:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('ima_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_avionics', units='inch')[0]:.2f} in")

    # ATA 2800: Aircraft Fuel System
    print("\nATA 2800 - Aircraft Fuel System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('fuel_system_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_fuel_system', units='inch')[0]:.2f} in")

    # ATA 4700: Fuel Inerting System
    print("\nATA 4700 - Fuel Inerting System:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('fuel_inerting_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_fuel_inerting', units='inch')[0]:.2f} in")

    # ATA 6100: Propeller System
    propeller_weight = prob.get_val('propeller_weight', units='lbm')[0]
    print("\nATA 6100 - Propeller System:")
    print("-" * 45)
    print(f"Total Weight:     {propeller_weight * 4.0:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_propeller', units='inch')[0]:.2f} in")

    # ATA 7200: Engine (Turbine/Turboprop)
    turbine_wt = prob.get_val('turbine_weight_per_nac', units='lbm')[0]
    gearbox_wt = prob.get_val('gearbox_weight', units='lbm')[0]
    turbine_cg = prob.get_val('cg_turbine', units='inch')[0]
    gearbox_cg = prob.get_val('cg_gearbox', units='inch')[0]
    # Weighted average C.G for turbine + gearbox
    engine_cg = (turbine_wt * turbine_cg + gearbox_wt * gearbox_cg) / (turbine_wt + gearbox_wt) if (turbine_wt + gearbox_wt) > 0 else 0.0
    print("\nATA 7200 - Engine (Turbine/Turboprop):")
    print("-" * 45)
    print(f"Total Weight:     {(turbine_wt * 4.0 + gearbox_wt * 4.0):.2f} lbm")
    print(f"C.G Location:     {engine_cg:.2f} in")

    # ATA 72E00: Electric Engine
    electric_motor_wt = prob.get_val('electric_motor_weight_per_nac', units='lbm')[0]
    print("\nATA 72E00 - Electric Engine:")
    print("-" * 45)
    print(f"Total Weight:     {electric_motor_wt * 4.0:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_electric_motor', units='inch')[0]:.2f} in")

    # ATA 7900: IPS TMS & Oil System (per nacelle x 4)
    em_tms = prob.get_val('electric_motor_tms_weight_per_nac', units='lbm')[0]
    turb_tms = prob.get_val('turbine_tms_weight_per_nac', units='lbm')[0]
    em_tms_cg = prob.get_val('cg_electric_motor_tms', units='inch')[0]
    turb_tms_cg = prob.get_val('cg_turbine_tms', units='inch')[0]
    # Weighted average C.G for TMS
    tms_cg = (em_tms * em_tms_cg + turb_tms * turb_tms_cg) / (em_tms + turb_tms) if (em_tms + turb_tms) > 0 else 0.0
    print("\nATA 7900 - IPS TMS & Oil System:")
    print("-" * 45)
    print(f"Total Weight:     {(em_tms + turb_tms) * 4.0:.2f} lbm")
    print(f"C.G Location:     {tms_cg:.2f} in")

    # ATA 9710: HV Wiring (HVWIS)
    print("\nATA 9710 - HV Wiring (HVWIS):")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('hvwis_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_hvwis', units='inch')[0]:.2f} in")

    # ATA 8510: Batteries
    print("\nATA 8510 - Batteries:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('total_battery_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_batteries', units='inch')[0]:.2f} in")

    # ATA 1100: Paint, Placards, and Markings
    print("\nATA 1100 - Paint, Placards, and Markings:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('paint_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_paint', units='inch')[0]:.2f} in")

    # ATA 2500: Cabin Equipment/furnishings
    print("\nATA 2500 - Cabin Equipment/furnishings:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('interiors_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_interiors', units='inch')[0]:.2f} in")

    # Operational Items
    print("\nOperational Items:")
    print("-" * 45)
    print(f"Total Weight:     {prob.get_val('operational_items_weight', units='lbm')[0]:.2f} lbm")
    print(f"C.G Location:     {prob.get_val('cg_operational_items', units='inch')[0]:.2f} in")

    # Print Total OEW
    total_weight = prob.get_val('OEW', units='lbm')[0]
    total_cg = prob.get_val('cg_total', units='inch')[0]
    
    # Get MAC and LEMAC for %MAC calculations
    MAC = prob.get_val('MAC', units='inch')[0]
    LEMAC = prob.get_val('LEMAC', units='inch')[0]
    
    # Calculate %MAC for OEW
    oew_percent_mac = ((total_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("TOTAL OPERATING EMPTY WEIGHT (OEW)")
    print("=" * 70)
    print(f"Weight:           {total_weight:.2f} lbm")
    print(f"C.G Location:     {total_cg:.2f} in")
    print(f"C.G %MAC:         {oew_percent_mac:.2f}%")
    print("=" * 70)
    
    # Payload
    payload_weight = prob.get_val('payload_weight', units='lbm')[0]
    passenger_weight = prob.get_val('passenger_weight', units='lbm')[0]
    cargo_weight = prob.get_val('cargo_weight', units='lbm')[0]
    passenger_cg = prob.get_val('cg_passenger', units='inch')[0]
    cargo_cg = prob.get_val('cg_cargo', units='inch')[0]
    total_payload_cg = (passenger_weight * passenger_cg + cargo_weight * cargo_cg) / payload_weight if payload_weight > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("PAYLOAD")
    print("=" * 70)
    print(f"Total Weight:     {payload_weight:.2f} lbm")
    print(f"Total C.G:        {total_payload_cg:.2f} in")
    print("=" * 70)
    
    # Calculate ZFW (Zero Fuel Weight) = OEW + Payload
    zfw = total_weight + payload_weight
    zfw_cg = (total_weight * total_cg + payload_weight * total_payload_cg) / zfw if zfw > 0 else 0.0
    zfw_percent_mac = ((zfw_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("ZERO FUEL WEIGHT (ZFW)")
    print("=" * 70)
    print(f"Weight:           {zfw:.2f} lbm")
    print(f"C.G Location:     {zfw_cg:.2f} in")
    print(f"C.G %MAC:         {zfw_percent_mac:.2f}%")
    print("=" * 70)
    
    # Fuel
    fuel_weight = prob.get_val('fuel_weight', units='lbm')[0]
    fuel_cg = prob.get_val('cg_fuel', units='inch')[0]
    
    print("\n" + "=" * 70)
    print("FUEL")
    print("=" * 70)
    print(f"Total Weight:     {fuel_weight:.2f} lbm")
    print(f"Total C.G:        {fuel_cg:.2f} in")
    print("=" * 70)
    
    # Calculate TOW (Takeoff Weight) = ZFW + Fuel = OEW + Payload + Fuel
    tow = zfw + fuel_weight
    tow_cg = (zfw * zfw_cg + fuel_weight * fuel_cg) / tow if tow > 0 else 0.0
    tow_percent_mac = ((tow_cg - LEMAC) / MAC) * 100.0 if MAC > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("TAKEOFF WEIGHT (TOW)")
    print("=" * 70)
    print(f"Weight:           {tow:.2f} lbm")
    print(f"C.G Location:     {tow_cg:.2f} in")
    print(f"C.G %MAC:         {tow_percent_mac:.2f}%")
    print("=" * 70)

    # Export results to text file
    export_to_text_file(prob)
    
    # Run check_partials if desired (commented out for normal operation)
    # prob.check_partials(compact_print=True, method='fd')

    return prob


if __name__ == '__main__':
    main()
