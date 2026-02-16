import openmdao.api as om
import numpy as np


class ExteriorLightingWeight(om.ExplicitComponent):
    """
    Calculates exterior lighting weight.
    Constant weight value.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('exterior_lighting_weight', default=39.5, types=float,
                           desc='Exterior lighting weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_exterior_lighting_percentage', default=49.0, types=float,
                           desc='Exterior Lighting C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('exterior_lighting_weight', val=0.0, units='lbm', 
                       desc='Exterior lighting weight')
        self.add_output('cg_exterior_lighting', val=0.0, units='inch', desc='Exterior Lighting C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_exterior_lighting', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['exterior_lighting_weight'] = self.options['exterior_lighting_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_exterior_lighting_percentage']
        outputs['cg_exterior_lighting'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_exterior_lighting_percentage']
        partials['cg_exterior_lighting', 'fuselage_length'] = 12.0 * cg_percentage / 100.0



class ElectricalPowerSystemWeight(om.ExplicitComponent):
    """
    Calculates electrical power system weight.
    Constant weight value (like ExteriorLightingWeight).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('electrical_power_system_weight', default=628.8, types=float,
                           desc='Electrical power system weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('weight_fuselage_percentage', default=64.0, types=float,
                           desc='Percentage of electrical power system weight in fuselage')
        self.options.declare('cg_eps_fuselage_percentage', default=49.8, types=float,
                           desc='EPS Fuselage C.G location as percentage of fuselage length')
        self.options.declare('cg_eps_wing_percentage', default=16.8, types=float,
                           desc='EPS Wing C.G location as percentage of MAC')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        self.add_output('electrical_power_system_weight', val=0.0, units='lbm', 
                       desc='Electrical power system weight')
        self.add_output('cg_electrical_power_system', val=0.0, units='inch', desc='Electrical Power System total C.G location (weighted average of fuselage and wing portions)')
        
        # Declare partials
        self.declare_partials('cg_electrical_power_system', ['fuselage_length', 'MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get constant weight from options
        total_weight = self.options['electrical_power_system_weight']
        outputs['electrical_power_system_weight'] = total_weight
        
        # Split weight into fuselage and wing portions
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        # Calculate C.G for fuselage portion
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        cg_fuselage_pct = self.options['cg_eps_fuselage_percentage']
        cg_fuselage = fs_start + (fuselage_length_in * cg_fuselage_pct / 100.0)
        
        # Calculate C.G for wing portion
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_wing_pct = self.options['cg_eps_wing_percentage']
        cg_wing = LEMAC + (MAC * cg_wing_pct / 100.0)
        
        # Calculate weighted average total C.G
        outputs['cg_electrical_power_system'] = (weight_fuselage * cg_fuselage + weight_wing * cg_wing) / total_weight


    def compute_partials(self, inputs, partials):
        # Partials for C.G only (weight is constant)
        total_weight = self.options['electrical_power_system_weight']
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        cg_fuselage_pct = self.options['cg_eps_fuselage_percentage']
        
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_wing_pct = self.options['cg_eps_wing_percentage']
        
        # Partials w.r.t. fuselage_length (affects fuselage C.G only)
        partials['cg_electrical_power_system', 'fuselage_length'] = weight_fuselage * 12.0 * cg_fuselage_pct / 100.0 / total_weight
        
        # Partials w.r.t. MAC (affects wing C.G only)
        partials['cg_electrical_power_system', 'MAC'] = weight_wing * cg_wing_pct / 100.0 / total_weight
        
        # Partials w.r.t. LEMAC (affects wing C.G only)
        partials['cg_electrical_power_system', 'LEMAC'] = weight_wing / total_weight


class LVWISWeight(om.ExplicitComponent):
    """
    Calculates Low Voltage Wiring and Interconnect System (LVWIS) weight.
    
    First calculates cabin volume:
    Vpc = PI * (average(width, height) / 2)^2 * Length * cabin_percent_cross_section
    
    Then calculates electric power:
    Pel = 0.016 * Vpc
    
    Then calculates LVWIS weight:
    LVWIS = 36 * Pel * (1 - 0.033 * sqrt(Pel))
    
    NOTE: This component uses sqrt() which makes partials complex.
    Manual review recommended for compute_partials implementation.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('power_coefficient', default=0.016, types=float,
                           desc='Electric power coefficient (power per unit volume)')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_lvwis_percentage', default=47.2, types=float,
                           desc='LVWIS C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('length_pax_cabin', val=63.0, units='ft', desc='Length of passenger cabin')
        self.add_input('fuselage_width', val=10.16, units='ft', desc='Fuselage width')
        self.add_input('fuselage_height', val=9.87, units='ft', desc='Fuselage height')
        self.add_input('cabin_percent_cross_section', val=0.81, units=None,
                      desc='Percentage of cabin in fuselage cross section (0.81 = 81%)')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')

        # NOTE: Keeping intermediate outputs as they're used in export function
        self.add_output('cabin_volume', val=0.0, units='ft**3', desc='Passenger cabin volume')
        self.add_output('electric_power', val=0.0, units='W', desc='Electric power (Pel)')
        self.add_output('lvwis_weight', val=0.0, units='lbm', desc='LVWIS weight')
        self.add_output('cg_lvwis', val=0.0, units='inch', desc='LVWIS C.G location (Fuselage Station)')
        
        # Declare partials for all outputs
        cabin_inputs = ['length_pax_cabin', 'fuselage_width', 'fuselage_height', 'cabin_percent_cross_section']
        self.declare_partials('cabin_volume', cabin_inputs)
        self.declare_partials('electric_power', cabin_inputs)
        self.declare_partials('lvwis_weight', cabin_inputs)
        self.declare_partials('cg_lvwis', 'fuselage_length')

    def compute(self, inputs, outputs):
        # Calculate average diameter: average(width, height)
        avg_diameter = (inputs['fuselage_width'] + inputs['fuselage_height']) / 2.0
        # Calculate radius: average/2
        radius = avg_diameter / 2.0
        # Calculate cross-sectional area: PI * r^2
        cross_sectional_area = np.pi * (radius ** 2)
        # Calculate cabin volume: area * Length * cabin_percent_cross_section
        outputs['cabin_volume'] = cross_sectional_area * inputs['length_pax_cabin'] * inputs['cabin_percent_cross_section']
        
        # Calculate electric power: Pel = 0.016 * Vpc (in W)
        power_coeff = self.options['power_coefficient']
        outputs['electric_power'] = power_coeff * outputs['cabin_volume']
        
        # Calculate LVWIS weight: 36 * Pel * (1 - 0.033 * sqrt(Pel))
        Pel = outputs['electric_power']
        outputs['lvwis_weight'] = 36 * Pel * (1.0 - 0.033 * np.sqrt(Pel))
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_lvwis_percentage']
        outputs['cg_lvwis'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Intermediate calculations for weight partials
        avg_diameter = (inputs['fuselage_width'] + inputs['fuselage_height']) / 2.0
        radius = avg_diameter / 2.0
        cross_sectional_area = np.pi * (radius ** 2)
        cabin_volume = cross_sectional_area * inputs['length_pax_cabin'] * inputs['cabin_percent_cross_section']
        
        power_coeff = self.options['power_coefficient']
        Pel = power_coeff * cabin_volume
        
        # Partials for lvwis_weight: 36 * Pel * (1 - 0.033 * sqrt(Pel))
        sqrt_Pel = np.sqrt(Pel)
        d_lvwis_d_Pel = 36 * (1.0 - 0.0495 * sqrt_Pel)
        
        # Chain rule: d/d(input) = d/dPel * dPel/d(input)
        # dPel/d(input) = power_coeff * d(cabin_volume)/d(input)
        r = radius
        L = inputs['length_pax_cabin']
        pct = inputs['cabin_percent_cross_section']
        
        # Partials for cabin_volume
        d_cabin_volume_d_length = np.pi * (r ** 2) * pct
        d_cabin_volume_d_width = np.pi * L * pct * r / 2.0
        d_cabin_volume_d_height = np.pi * L * pct * r / 2.0
        d_cabin_volume_d_pct = np.pi * (r ** 2) * L
        
        partials['cabin_volume', 'length_pax_cabin'] = d_cabin_volume_d_length
        partials['cabin_volume', 'fuselage_width'] = d_cabin_volume_d_width
        partials['cabin_volume', 'fuselage_height'] = d_cabin_volume_d_height
        partials['cabin_volume', 'cabin_percent_cross_section'] = d_cabin_volume_d_pct
        
        # Partials for electric_power
        d_Pel_d_length = power_coeff * d_cabin_volume_d_length
        d_Pel_d_width = power_coeff * d_cabin_volume_d_width
        d_Pel_d_height = power_coeff * d_cabin_volume_d_height
        d_Pel_d_pct = power_coeff * d_cabin_volume_d_pct
        
        partials['electric_power', 'length_pax_cabin'] = d_Pel_d_length
        partials['electric_power', 'fuselage_width'] = d_Pel_d_width
        partials['electric_power', 'fuselage_height'] = d_Pel_d_height
        partials['electric_power', 'cabin_percent_cross_section'] = d_Pel_d_pct
        
        # Partials for lvwis_weight
        partials['lvwis_weight', 'length_pax_cabin'] = d_lvwis_d_Pel * d_Pel_d_length
        partials['lvwis_weight', 'fuselage_width'] = d_lvwis_d_Pel * d_Pel_d_width
        partials['lvwis_weight', 'fuselage_height'] = d_lvwis_d_Pel * d_Pel_d_height
        partials['lvwis_weight', 'cabin_percent_cross_section'] = d_lvwis_d_Pel * d_Pel_d_pct
        
        # Partials for C.G
        cg_percentage = self.options['cg_lvwis_percentage']
        partials['cg_lvwis', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class HVWISWeightParametric(om.ExplicitComponent):
    """
    Calculates High Voltage Wiring and Interconnect System (HVWIS) weight.
    
    Formula: HVWIS = alpha * P * L / V
    
    where:
    - alpha is a constant coefficient
    - P is the electrical engine power (W)
    - L is the distance between inboard and outboard nacelles (ft)
    - V is the maximum voltage (V)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('alpha', default=0.0017, types=float,
                           desc='HVWIS coefficient constant')
        self.options.declare('cg_hvwis_percentage', default=51.8, types=float,
                           desc='HVWIS C.G location as percentage of MAC')

    def setup(self):
        self.add_input('rated_power_em_per_nacelle', val=1864000.0, units='W', 
                      desc='Total electrical engine power')
        self.add_input('nacelle_distance', val=179.5, units='ft', 
                      desc='Distance between inboard and outboard nacelles')
        self.add_input('max_voltage', val=570.0, units='V', 
                      desc='Maximum voltage')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        self.add_output('hvwis_weight', val=0.0, units='lbm', desc='HVWIS weight')
        self.add_output('cg_hvwis', val=0.0, units='inch', desc='HVWIS C.G location')
        
        # Declare partials
        self.declare_partials('hvwis_weight', '*')
        self.declare_partials('cg_hvwis', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get alpha from options
        alpha = self.options['alpha']
        
        # Formula: HVWIS = alpha * P * L / V
        outputs['hvwis_weight'] = alpha * inputs['rated_power_em_per_nacelle'] * inputs['nacelle_distance'] / inputs['max_voltage']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_hvwis_percentage']
        outputs['cg_hvwis'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        alpha = self.options['alpha']
        P = inputs['rated_power_em_per_nacelle']
        L = inputs['nacelle_distance']
        V = inputs['max_voltage']
        
        # Partial w.r.t. rated_power_em_per_nacelle
        partials['hvwis_weight', 'rated_power_em_per_nacelle'] = alpha * L / V
        
        # Partial w.r.t. nacelle_distance
        partials['hvwis_weight', 'nacelle_distance'] = alpha * P / V
        
        # Partial w.r.t. max_voltage
        partials['hvwis_weight', 'max_voltage'] = -alpha * P * L / (V ** 2)
        
        # Partials for C.G
        cg_percentage = self.options['cg_hvwis_percentage']
        partials['cg_hvwis', 'MAC'] = cg_percentage / 100.0
        partials['cg_hvwis', 'LEMAC'] = 1.0



class HVWISWeightConstant(om.ExplicitComponent):
    """
    Calculates High Voltage Wiring and Interconnect System (HVWIS) weight.
    Constant weight value.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('hvwis_weight', default=963.90, types=float,
                           desc='HVWIS weight in lbm')
        self.options.declare('cg_hvwis_percentage', default=51.8, types=float,
                           desc='HVWIS C.G location as percentage of MAC')

    def setup(self):
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        self.add_output('hvwis_weight', val=0.0, units='lbm', desc='HVWIS weight')
        self.add_output('cg_hvwis', val=0.0, units='inch', desc='HVWIS C.G location')
        
        # Declare partials
        self.declare_partials('cg_hvwis', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        outputs['hvwis_weight'] = self.options['hvwis_weight']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_hvwis_percentage']
        outputs['cg_hvwis'] = LEMAC + (MAC * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_hvwis_percentage']
        partials['cg_hvwis', 'MAC'] = cg_percentage / 100.0
        partials['cg_hvwis', 'LEMAC'] = 1.0


class ElectricalGroup(om.Group):
    """
    Group that calculates total electrical system weight.
    
    Total = LVWIS + HVWIS + Electrical Power System + Exterior Lighting
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('hvwis_weight_type', default='parametric',
                           desc='Type of HVWIS weight calculation')

    def setup(self):
        # Add individual weight components
        self.add_subsystem('exterior_lighting', ExteriorLightingWeight(), promotes=['*'])
        self.add_subsystem('electrical_power', ElectricalPowerSystemWeight(), promotes=['*'])
        self.add_subsystem('lvwis', LVWISWeight(), promotes=['*'])
        if self.options['hvwis_weight_type'] == 'constant':
            self.add_subsystem('hvwis', HVWISWeightConstant(), promotes=['*'])
        elif self.options['hvwis_weight_type'] == 'parametric':
            self.add_subsystem('hvwis', HVWISWeightParametric(), promotes=['*'])

        # Use AddSubtractComp to sum all weights
        adder = om.AddSubtractComp()
        adder.add_equation(
            'total_electrical_weight',
            input_names=['lvwis_weight', 'hvwis_weight', 
                        'electrical_power_system_weight', 'exterior_lighting_weight'],
            units='lbm',
            desc='Total electrical system weight'
        )
        self.add_subsystem('total', adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # ElectricalPowerSystemWeight inputs (no longer needed - constant weight)
    # LVWISWeight inputs
    ivc.add_output('length_pax_cabin', val=63.0, units='ft')
    ivc.add_output('fuselage_width', val=10.16, units='ft')
    ivc.add_output('fuselage_height', val=9.87, units='ft')
    ivc.add_output('cabin_percent_cross_section', val=0.81, units=None)
    # HVWISWeight inputs
    ivc.add_output('nacelle_distance', val=179.5, units='ft')
    ivc.add_output('max_voltage', val=570.0, units='V')

    model.add_subsystem('electrical', ElectricalGroup(hvwis_weight_type='parametric'), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Exterior Lighting Weight:', prob.get_val('exterior_lighting_weight', units='lbm'))
    print('Electrical Power System Weight:', prob.get_val('electrical_power_system_weight', units='lbm'))
    print('LVWIS Weight:', prob.get_val('lvwis_weight', units='lbm'))
    print('HVWIS Weight:', prob.get_val('hvwis_weight', units='lbm'))
    print('Total Electrical Weight:', prob.get_val('total_electrical_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
