import openmdao.api as om
import numpy as np


class PropellerWeight(om.ExplicitComponent):
    """
    Calculates propeller weight based on number of blades and diameter.
    
    Configurations:
    - 5-blade: Fixed weight of 404 lbm (diameter ignored)
    - 6-blade: Weight based on diameter regression (12 ft = 402 lbm, 13 ft = 423 lbm)
    - 7-blade: Fixed weight of 450 lbm (diameter ignored)
    
    NOTE: This component has conditional logic (if/elif) which makes partials complex.
    Manual review recommended for compute_partials implementation.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Default values that can be overridden when instantiating the component
        self.options.declare('propeller_5blade_weight', default=404.0, types=float,
                           desc='Fixed propeller weight for 5-blade configuration in lbm')
        self.options.declare('propeller_7blade_weight', default=450.0, types=float,
                           desc='Fixed propeller weight for 7-blade configuration in lbm')
        self.options.declare('propeller_6blade_slope', default=21.0, types=float,
                           desc='6-blade propeller weight regression slope (lbm/ft)')
        self.options.declare('propeller_6blade_intercept', default=150.0, types=float,
                           desc='6-blade propeller weight regression intercept in lbm')
        self.options.declare('cg_propeller_inboard_percentage', default=-100.3, types=float,
                           desc='Propeller Inboard C.G location as percentage of MAC')
        self.options.declare('cg_propeller_outboard_percentage', default=-89.1, types=float,
                           desc='Propeller Outboard C.G location as percentage of MAC')

    def setup(self):
        self.add_input('num_blades', val=6, units=None, desc='Number of propeller blades (5, 6, or 7)')
        self.add_input('blade_diameter', val=12.0, units='ft', desc='Propeller blade diameter')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('propeller_weight', val=0.0, units='lbm', desc='Propeller weight (per unit)')
        self.add_output('cg_propeller', val=0.0, units='inch', desc='Propeller total C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials - NOTE: Complex due to conditional logic
        self.declare_partials('propeller_weight', '*')
        self.declare_partials('cg_propeller', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Input validation
        num_blades = inputs['num_blades']
        blade_diameter = inputs['blade_diameter']
        
        if num_blades < 4.5 or num_blades > 7.5:
            raise ValueError(f"num_blades must be between 4.5 and 7.5 (typically 5, 6, or 7), got {num_blades}")
        if blade_diameter <= 0:
            raise ValueError(f"blade_diameter must be positive, got {blade_diameter}")
        
        # Use continuous interpolation for optimization-friendly behavior
        # For integer values (5, 6, 7), use exact weights
        # For intermediate values, interpolate linearly
        
        n_blades_rounded = np.round(num_blades)
        
        if n_blades_rounded == 5:
            weight_5 = self.options['propeller_5blade_weight']
            # Interpolate between 5 and 6 if needed
            if num_blades < 5.5:
                outputs['propeller_weight'] = weight_5
            else:
                # Linear interpolation between 5-blade and 6-blade
                weight_6_base = self.options['propeller_6blade_slope'] * blade_diameter + self.options['propeller_6blade_intercept']
                alpha = (num_blades - 5.0) / 1.0  # 0 at 5, 1 at 6
                outputs['propeller_weight'] = (1.0 - alpha) * weight_5 + alpha * weight_6_base
        elif n_blades_rounded == 6:
            # 6-blade: Weight based on diameter regression
            if blade_diameter < 11.0 or blade_diameter > 15.0:
                raise ValueError(
                    f"Invalid diameter for 6-blade configuration: {blade_diameter} ft. "
                    f"Diameter must be between 11 ft and 15 ft."
                )
            slope = self.options['propeller_6blade_slope']
            intercept = self.options['propeller_6blade_intercept']
            weight_6 = slope * blade_diameter + intercept
            
            if num_blades < 6.5:
                # Interpolate between 5 and 6
                weight_5 = self.options['propeller_5blade_weight']
                alpha = (num_blades - 5.0) / 1.0
                outputs['propeller_weight'] = (1.0 - alpha) * weight_5 + alpha * weight_6
            else:
                # Interpolate between 6 and 7
                weight_7 = self.options['propeller_7blade_weight']
                alpha = (num_blades - 6.0) / 1.0
                outputs['propeller_weight'] = (1.0 - alpha) * weight_6 + alpha * weight_7
        elif n_blades_rounded == 7:
            weight_7 = self.options['propeller_7blade_weight']
            if num_blades < 7.5:
                # Interpolate between 6 and 7
                weight_6_base = self.options['propeller_6blade_slope'] * blade_diameter + self.options['propeller_6blade_intercept']
                alpha = (num_blades - 6.0) / 1.0
                outputs['propeller_weight'] = (1.0 - alpha) * weight_6_base + alpha * weight_7
            else:
                outputs['propeller_weight'] = weight_7
        else:
            raise ValueError(
                f"Invalid number of blades: {num_blades}. "
                f"Must be between 4.5 and 7.5 (typically 5, 6, or 7)."
            )
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_propeller_inboard_percentage']
        cg_outboard_pct = self.options['cg_propeller_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)
        
        # Weighted average: (2 * weight * cg_in + 2 * weight * cg_out) / (4 * weight) = (cg_in + cg_out) / 2
        outputs['cg_propeller'] = (cg_inboard + cg_outboard) / 2.0



    def compute_partials(self, inputs, partials):
        # Partials for continuous interpolation logic
        num_blades = inputs['num_blades']
        blade_diameter = inputs['blade_diameter']
        n_blades_rounded = np.round(num_blades)
        
        slope = self.options['propeller_6blade_slope']
        weight_5 = self.options['propeller_5blade_weight']
        weight_7 = self.options['propeller_7blade_weight']
        
        if n_blades_rounded == 5:
            if num_blades < 5.5:
                # Constant weight_5
                partials['propeller_weight', 'num_blades'] = 0.0
                partials['propeller_weight', 'blade_diameter'] = 0.0
            else:
                # Linear interpolation between 5 and 6
                weight_6_base = slope * blade_diameter + self.options['propeller_6blade_intercept']
                alpha = (num_blades - 5.0) / 1.0
                # d/d(num_blades) = (weight_6_base - weight_5) / 1.0
                partials['propeller_weight', 'num_blades'] = (weight_6_base - weight_5) / 1.0
                # d/d(blade_diameter) = alpha * slope
                partials['propeller_weight', 'blade_diameter'] = alpha * slope
        elif n_blades_rounded == 6:
            weight_6 = slope * blade_diameter + self.options['propeller_6blade_intercept']
            if num_blades < 6.5:
                # Interpolate between 5 and 6
                alpha = (num_blades - 5.0) / 1.0
                partials['propeller_weight', 'num_blades'] = (weight_6 - weight_5) / 1.0
                partials['propeller_weight', 'blade_diameter'] = alpha * slope
            else:
                # Interpolate between 6 and 7
                alpha = (num_blades - 6.0) / 1.0
                partials['propeller_weight', 'num_blades'] = (weight_7 - weight_6) / 1.0
                partials['propeller_weight', 'blade_diameter'] = (1.0 - alpha) * slope
        elif n_blades_rounded == 7:
            if num_blades < 7.5:
                # Interpolate between 6 and 7
                weight_6_base = slope * blade_diameter + self.options['propeller_6blade_intercept']
                alpha = (num_blades - 6.0) / 1.0
                partials['propeller_weight', 'num_blades'] = (weight_7 - weight_6_base) / 1.0
                partials['propeller_weight', 'blade_diameter'] = (1.0 - alpha) * slope
            else:
                # Constant weight_7
                partials['propeller_weight', 'num_blades'] = 0.0
                partials['propeller_weight', 'blade_diameter'] = 0.0
        else:
            partials['propeller_weight', 'num_blades'] = 0.0
            partials['propeller_weight', 'blade_diameter'] = 0.0
        
        # Partials for C.G
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_propeller_inboard_percentage']
        cg_outboard_pct = self.options['cg_propeller_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_propeller', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 0.5 * (1 + 1) = 1.0
        partials['cg_propeller', 'LEMAC'] = 1.0


class ElectricMotorWeight(om.ExplicitComponent):
    """
    Calculates electric motor weight based on energy density and energy needed.
    
    Weight = Energy (kW) / Energy Density (kW/lbm)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('cg_electric_motor_inboard_percentage', default=-64.1, types=float,
                           desc='Electric Motor Inboard C.G location as percentage of MAC')
        self.options.declare('cg_electric_motor_outboard_percentage', default=-53.0, types=float,
                           desc='Electric Motor Outboard C.G location as percentage of MAC')

    def setup(self):
        self.add_input('rated_power_em_per_nacelle', val=1864000.0, units='W', desc='Electrical engine power')
        self.add_input('power_density', val=4.54, units='kW/lbm', desc='Energy density constant')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('electric_motor_weight_per_nac', val=0.0, units='lbm', desc='Electric motor weight (per unit)')
        self.add_output('cg_electric_motor', val=0.0, units='inch', desc='Electric Motor total C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials
        self.declare_partials('electric_motor_weight_per_nac', '*')
        self.declare_partials('cg_electric_motor', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Convert power from W to kW for calculation
        power_kw = inputs['rated_power_em_per_nacelle'] / 1000.0
        # Weight = Power (kW) / Power Density (kW/lbm)
        outputs['electric_motor_weight_per_nac'] = power_kw / inputs['power_density']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_electric_motor_inboard_percentage']
        cg_outboard_pct = self.options['cg_electric_motor_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)
        
        # Weighted average: (2 * weight * cg_in + 2 * weight * cg_out) / (4 * weight) = (cg_in + cg_out) / 2
        outputs['cg_electric_motor'] = (cg_inboard + cg_outboard) / 2.0


    def compute_partials(self, inputs, partials):
        power_kw = inputs['rated_power_em_per_nacelle'] / 1000.0
        # Partial w.r.t. rated_power_em_per_nacelle (convert W to kW)
        partials['electric_motor_weight_per_nac', 'rated_power_em_per_nacelle'] = 1.0 / (inputs['power_density'] * 1000.0)
        
        # Partial w.r.t. power_density
        partials['electric_motor_weight_per_nac', 'power_density'] = -power_kw / (inputs['power_density'] ** 2)
        
        # Partials for C.G
        MAC = inputs['MAC']
        cg_inboard_pct = self.options['cg_electric_motor_inboard_percentage']
        cg_outboard_pct = self.options['cg_electric_motor_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_electric_motor', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 1.0
        partials['cg_electric_motor', 'LEMAC'] = 1.0


class TurbineWeight(om.ExplicitComponent):
    """
    Turbine weight component - constant weight from options.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('turbine_weight_per_nac', default=545.0, types=float, desc='Turbine weight in lbm')
        self.options.declare('cg_turbine_inboard_percentage', default=-36.2, types=float,
                           desc='Turbine Inboard C.G location as percentage of MAC')
        self.options.declare('cg_turbine_outboard_percentage', default=-25.0, types=float,
                           desc='Turbine Outboard C.G location as percentage of MAC')

    def setup(self):
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('turbine_weight_per_nac', val=0.0, units='lbm', desc='Turbine weight (per unit)')
        self.add_output('cg_turbine', val=0.0, units='inch', desc='Turbine total C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials
        self.declare_partials('cg_turbine', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        outputs['turbine_weight_per_nac'] = self.options['turbine_weight_per_nac']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_turbine_inboard_percentage']
        cg_outboard_pct = self.options['cg_turbine_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)
        
        # Weighted average: (2 * weight * cg_in + 2 * weight * cg_out) / (4 * weight) = (cg_in + cg_out) / 2
        outputs['cg_turbine'] = (cg_inboard + cg_outboard) / 2.0
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_inboard_pct = self.options['cg_turbine_inboard_percentage']
        cg_outboard_pct = self.options['cg_turbine_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_turbine', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 1.0
        partials['cg_turbine', 'LEMAC'] = 1.0




class FuelInertingWeight(om.ExplicitComponent):
    """
    Calculates fuel inerting system weight.
    Constant weight value.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fuel_inerting_weight', default=75.0, types=float,
                           desc='Fuel inerting system weight in lbm')
        self.options.declare('cg_fuel_inerting_percentage', default=44.4, types=float,
                           desc='Fuel Inerting C.G location as percentage of MAC')

    def setup(self):
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('fuel_inerting_weight', val=0.0, units='lbm', 
                       desc='Fuel inerting system weight')
        self.add_output('cg_fuel_inerting', val=0.0, units='inch', desc='Fuel Inerting C.G location')
        
        # Declare partials
        self.declare_partials('cg_fuel_inerting', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        outputs['fuel_inerting_weight'] = self.options['fuel_inerting_weight']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_fuel_inerting_percentage']
        outputs['cg_fuel_inerting'] = LEMAC + (MAC * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_fuel_inerting_percentage']
        partials['cg_fuel_inerting', 'MAC'] = cg_percentage / 100.0
        partials['cg_fuel_inerting', 'LEMAC'] = 1.0


class FuelSystemWeight(om.ExplicitComponent):
    """
    Calculates fuel system weight.
    Constant weight value.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fuel_system_weight', default=215.0, types=float,
                           desc='Fuel system weight in lbm')
        self.options.declare('cg_fuel_system_percentage', default=40.0, types=float,
                           desc='Fuel System C.G location as percentage of MAC')

    def setup(self):
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('fuel_system_weight', val=0.0, units='lbm', 
                       desc='Fuel system weight')
        self.add_output('cg_fuel_system', val=0.0, units='inch', desc='Fuel System C.G location')
        
        # Declare partials
        self.declare_partials('cg_fuel_system', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        outputs['fuel_system_weight'] = self.options['fuel_system_weight']
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_fuel_system_percentage']
        outputs['cg_fuel_system'] = LEMAC + (MAC * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_fuel_system_percentage']
        partials['cg_fuel_system', 'MAC'] = cg_percentage / 100.0
        partials['cg_fuel_system', 'LEMAC'] = 1.0



class GearboxWeight(om.ExplicitComponent):
    """
    Calculates gearbox weight from electric engines and turbine.
    
    Formula for each part: n * k * HP^0.76 * (RPMin^0.13) / (RPMout^0.89)
    
    After calculating both parts:
    - Sum the two parts
    - Add 11% for structures
    - Total = (electric_gearbox + turbine_gearbox) * 1.11
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('counter_rotating_weight', default=12.5, types=float,
                           desc='Additional weight for counter-rotating feature in lbm')
        self.options.declare('cg_gearbox_inboard_percentage', default=-79.1, types=float,
                           desc='Gearbox Inboard C.G location as percentage of MAC')
        self.options.declare('cg_gearbox_outboard_percentage', default=-67.9, types=float,
                           desc='Gearbox Outboard C.G location as percentage of MAC')

    def setup(self):
        # Electric engine gearbox inputs
        self.add_input('num_em_per_nac', val=2, units=None, 
                      desc='Number of electric engines')
        self.add_input('unit_rated_power_em', val=1052.7025, units='hp', 
                      desc='Electric engine horsepower')
        self.add_input('electric_k', val=125.687, units=None, 
                      desc='Electric engine technology factor')
        self.add_input('electric_RPMin', val=12240.0, units='rpm', 
                      desc='Electric engine input RPM')
        self.add_input('electric_RPMout', val=1200.0, units='rpm', 
                      desc='Electric engine output RPM')
        
        # Turbine gearbox inputs
        self.add_input('num_turb_per_nac', val=1, units=None, 
                      desc='Number of turbines')
        self.add_input('turbine_HP', val=1300.0, units='hp', 
                      desc='Turbine horsepower')
        self.add_input('turbine_k', val=125.687, units=None, 
                      desc='Turbine technology factor')
        self.add_input('turbine_RPMin', val=30000.0, units='rpm', 
                      desc='Turbine input RPM')
        self.add_input('turbine_RPMout', val=1200.0, units='rpm', 
                      desc='Turbine output RPM')
        
        self.add_input('counter_rotating', val=True, units=None,
                      desc='Boolean: True if counter-rotating feature is enabled')
        
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        # Only keep total weight output as requested
        self.add_output('gearbox_weight', val=0.0, units='lbm', 
                       desc='Total gearbox weight (per unit)')
        self.add_output('cg_gearbox', val=0.0, units='inch', desc='Gearbox total C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials
        self.declare_partials('gearbox_weight', '*')
        self.declare_partials('cg_gearbox', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Apply 10% factor to power before raising to 0.76
        # Electric engine gearbox: n * k * (1.1 * HP)^0.76 * (RPMin^0.13) / (RPMout^0.89)
        power_em_adjusted = 1.1 * inputs['unit_rated_power_em']
        electric_gearbox_weight = (inputs['num_em_per_nac'] * inputs['electric_k'] * 
                                   (power_em_adjusted ** 0.76) * 
                                   (inputs['electric_RPMin'] ** 0.13) / 
                                   (inputs['electric_RPMout'] ** 0.89))
        
        # Turbine gearbox: n * k * (1.1 * HP)^0.76 * (RPMin^0.13) / (RPMout^0.89)
        power_turb_adjusted = 1.0 * inputs['turbine_HP']
        turbine_gearbox_weight = (inputs['num_turb_per_nac'] * inputs['turbine_k'] * 
                                  (power_turb_adjusted ** 0.76) * 
                                  (inputs['turbine_RPMin'] ** 0.13) / 
                                  (inputs['turbine_RPMout'] ** 0.89))
        
        # Sum both parts
        base_gearbox_weight = electric_gearbox_weight + turbine_gearbox_weight
        
        # Add counter-rotating weight if enabled
        counter_rotating_weight = 0.0
        if inputs['counter_rotating']:
            counter_rotating_weight = self.options['counter_rotating_weight']
        
        outputs['gearbox_weight'] = base_gearbox_weight + counter_rotating_weight
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_gearbox_inboard_percentage']
        cg_outboard_pct = self.options['cg_gearbox_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)
        
        # Weighted average: (2 * weight * cg_in + 2 * weight * cg_out) / (4 * weight) = (cg_in + cg_out) / 2
        outputs['cg_gearbox'] = (cg_inboard + cg_outboard) / 2.0



    def compute_partials(self, inputs, partials):
        # Electric gearbox terms (with 10% power adjustment)
        elec_n = inputs['num_em_per_nac']
        elec_k = inputs['electric_k']
        elec_HP = inputs['unit_rated_power_em']
        elec_HP_adj = 1.1 * elec_HP  # 10% adjustment
        elec_RPMin = inputs['electric_RPMin']
        elec_RPMout = inputs['electric_RPMout']
        
        elec_base = elec_k * (elec_HP_adj ** 0.76) * (elec_RPMin ** 0.13) / (elec_RPMout ** 0.89)
        
        # Turbine gearbox terms (with 10% power adjustment)
        turb_n = inputs['num_turb_per_nac']
        turb_k = inputs['turbine_k']
        turb_HP = inputs['turbine_HP']
        turb_HP_adj = 1.1 * turb_HP  # 10% adjustment
        turb_RPMin = inputs['turbine_RPMin']
        turb_RPMout = inputs['turbine_RPMout']
        
        turb_base = turb_k * (turb_HP_adj ** 0.76) * (turb_RPMin ** 0.13) / (turb_RPMout ** 0.89)
        
        # Partials w.r.t. electric engine parameters
        partials['gearbox_weight', 'num_em_per_nac'] = elec_base
        partials['gearbox_weight', 'electric_k'] = elec_n * elec_base / elec_k
        # d/d(HP) of (1.1*HP)^0.76 = 0.76 * (1.1*HP)^-0.24 * 1.1 = 0.76 * 1.1^0.76 * HP^-0.24
        partials['gearbox_weight', 'unit_rated_power_em'] = elec_n * elec_k * 0.76 * (1.1 ** 0.76) * (elec_HP ** -0.24) * (elec_RPMin ** 0.13) / (elec_RPMout ** 0.89)
        partials['gearbox_weight', 'electric_RPMin'] = elec_n * elec_k * 0.13 * (elec_HP_adj ** 0.76) * (elec_RPMin ** -0.87) / (elec_RPMout ** 0.89)
        partials['gearbox_weight', 'electric_RPMout'] = -elec_n * elec_k * 0.89 * (elec_HP_adj ** 0.76) * (elec_RPMin ** 0.13) / (elec_RPMout ** 1.89)
        
        # Partials w.r.t. turbine parameters
        partials['gearbox_weight', 'num_turb_per_nac'] = turb_base
        partials['gearbox_weight', 'turbine_k'] = turb_n * turb_base / turb_k
        partials['gearbox_weight', 'turbine_HP'] = turb_n * turb_k * 0.76 * (1.1 ** 0.76) * (turb_HP ** -0.24) * (turb_RPMin ** 0.13) / (turb_RPMout ** 0.89)
        partials['gearbox_weight', 'turbine_RPMin'] = turb_n * turb_k * 0.13 * (turb_HP_adj ** 0.76) * (turb_RPMin ** -0.87) / (turb_RPMout ** 0.89)
        partials['gearbox_weight', 'turbine_RPMout'] = -turb_n * turb_k * 0.89 * (turb_HP_adj ** 0.76) * (turb_RPMin ** 0.13) / (turb_RPMout ** 1.89)
        
        # Partial w.r.t. counter_rotating (step function: derivative is 0 everywhere)
        # Note: For optimization, we treat this as a smooth function, but in practice it's a step
        # The derivative is 0 since it's a constant addition when True
        partials['gearbox_weight', 'counter_rotating'] = 0.0
        
        # Partials for C.G
        cg_inboard_pct = self.options['cg_gearbox_inboard_percentage']
        cg_outboard_pct = self.options['cg_gearbox_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_gearbox', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 1.0
        partials['cg_gearbox', 'LEMAC'] = 1.0


class PropulsionGroup(om.Group):
    """
    Group that calculates total propulsion weight for all nacelles.
    
    Total = 4 × (Propeller + Electric Motor + Turbine + Gearbox)
    (4 nacelles total)
    Note: Fuel Inerting and Fuel System are separate and not included in propulsion total.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Add individual weight components
        self.add_subsystem('propeller', PropellerWeight(), promotes=['*'])
        self.add_subsystem('electric_motor', ElectricMotorWeight(), promotes=['*'])
        self.add_subsystem('turbine', TurbineWeight(), promotes=['*'])
        self.add_subsystem('fuel_inerting', FuelInertingWeight(), promotes=['*'])
        self.add_subsystem('fuel_system', FuelSystemWeight(), promotes=['*'])
        self.add_subsystem('gearbox', GearboxWeight(), promotes=['*'])

        # Use AddSubtractComp to sum weights with scaling factor of 4 (4 nacelles)
        adder = om.AddSubtractComp()
        adder.add_equation(
            'total_propulsion_weight',
            input_names=['propeller_weight', 'electric_motor_weight_per_nac', 
                        'turbine_weight_per_nac', 'gearbox_weight'],
            scaling_factors=[4.0, 4.0, 4.0, 4.0],
            units='lbm',
            desc='Total propulsion weight (4 nacelles)'
        )
        self.add_subsystem('total', adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # PropellerWeight inputs
    ivc.add_output('num_blades', val=6, units=None)
    ivc.add_output('blade_diameter', val=12.0, units='ft')
    # ElectricMotorWeight inputs
    ivc.add_output('rated_power_em_per_nacelle', val=1864000.0, units='W')
    ivc.add_output('power_density', val=4.54, units='kW/lbm')
    # GearboxWeight inputs
    ivc.add_output('num_em_per_nac', val=2, units=None)
    ivc.add_output('unit_rated_power_em', val=1052.7025, units='hp')
    ivc.add_output('electric_k', val=125.687, units=None)
    ivc.add_output('electric_RPMin', val=12240.0, units='rpm')
    ivc.add_output('electric_RPMout', val=1200.0, units='rpm')
    ivc.add_output('num_turb_per_nac', val=1, units=None)
    ivc.add_output('turbine_HP', val=1300.0, units='hp')
    ivc.add_output('turbine_k', val=125.687, units=None)
    ivc.add_output('counter_rotating', val=True, units=None)
    ivc.add_output('turbine_RPMin', val=30000.0, units='rpm')
    ivc.add_output('turbine_RPMout', val=1200.0, units='rpm')

    model.add_subsystem('propulsion', PropulsionGroup(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Propeller Weight:', prob.get_val('propeller_weight', units='lbm'))
    print('Electric Motor Weight:', prob.get_val('electric_motor_weight_per_nac', units='lbm'))
    print('Turbine Weight:', prob.get_val('turbine_weight_per_nac', units='lbm'))
    print('Fuel Inerting Weight:', prob.get_val('fuel_inerting_weight', units='lbm'))
    print('Fuel System Weight:', prob.get_val('fuel_system_weight', units='lbm'))
    print('Gearbox Weight:', prob.get_val('gearbox_weight', units='lbm'))
    print('Total Propulsion Weight:', prob.get_val('total_propulsion_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
