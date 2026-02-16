import openmdao.api as om
import numpy as np


class ElectricMotorTMS(om.ExplicitComponent):
    """
    Calculates thermal management system (TMS) weight for both electric motors and gearbox using kW/kg method.
    
    Formula: Weight_per_nac = (1.2 * Q_heat) / (2.9 kW/kg) * conversion
    
    where:
    - Q_heat: heat generated = rated_power_em_per_nacelle * (1 - efficiency) + 0.4 * gearbox_heat
    - Only 40% of gearbox heat is used in the calculation
    - 20% margin is applied to heat load (multiply by 1.2)
    - 2.9 kW/kg: power density for electric motor TMS
    - conversion: kg to lbm (2.20462)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('heat_load_factor', default=1.2, types=float,
                           desc='Margin factor applied to heat load (20% margin = 1.2)')
        self.options.declare('cg_electric_motor_tms_percentage', default=-30.6, types=float,
                           desc='Electric Motor TMS C.G location as percentage of MAC (same for inboard and outboard)')

    def setup(self):
        self.add_input('rated_power_em_per_nacelle', val=1566000.0, units='W', desc='Electrical engine power per nacelle')
        self.add_input('efficiency', val=0.95, units=None, desc='Electric motor efficiency')
        self.add_input('gearbox_heat', val=1000.0, units='W', desc='Gearbox heat generation (1 kW = 1000 W)')
        self.add_input('specific_cooling_capacity_em_tms', val=2.9, units='kW/kg', desc='Power density for Electric Motor TMS in kW/kg')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('electric_motor_tms_weight_per_nac', val=0.0, units='lbm', desc='Electric motor and gearbox TMS weight (per nacelle)')
        self.add_output('cg_electric_motor_tms', val=0.0, units='inch', desc='Electric Motor TMS C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials
        self.declare_partials('electric_motor_tms_weight_per_nac', ['rated_power_em_per_nacelle', 'efficiency', 'gearbox_heat', 'specific_cooling_capacity_em_tms'])
        self.declare_partials('cg_electric_motor_tms', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get constants from options
        specific_cooling_capacity = inputs['specific_cooling_capacity_em_tms']  # kW/kg from input
        conversion = 2.20462  # kg to lbm
        gearbox_heat_factor = 0.4  # Only 40% of gearbox heat is used
        
        # Calculate heat generated: rated_power * (1 - efficiency) + 40% of gearbox_heat
        # Use heat generated directly (no 1.2 factor applied)
        heat_generated = inputs['rated_power_em_per_nacelle'] * (1.0 - inputs['efficiency']) + gearbox_heat_factor * inputs['gearbox_heat']  # W
        
        # Convert power density from kW/kg to W/kg
        specific_cooling_capacity_W_per_kg = specific_cooling_capacity * 1000.0  # W/kg
        
        # Calculate weight per nacelle: Weight = (heat_generated in W) / (specific_cooling_capacity in W/kg) = kg
        electric_motor_tms_weight_kg = heat_generated / specific_cooling_capacity_W_per_kg
        
        # Convert from kg to lbm (per nacelle)
        outputs['electric_motor_tms_weight_per_nac'] = electric_motor_tms_weight_kg * conversion
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        # Since inboard and outboard have same percentage, weighted average is just the same value
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_electric_motor_tms_percentage']
        outputs['cg_electric_motor_tms'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        specific_cooling_capacity = inputs['specific_cooling_capacity_em_tms']
        specific_cooling_capacity_W_per_kg = specific_cooling_capacity * 1000.0
        conversion = 2.20462
        gearbox_heat_factor = 0.4  # Only 40% of gearbox heat is used
        
        eff = inputs['efficiency']
        Pem = inputs['rated_power_em_per_nacelle']
        Q_gb = inputs['gearbox_heat']
        heat_generated = Pem * (1.0 - eff) + gearbox_heat_factor * Q_gb
        
        # Partial w.r.t. rated_power_em_per_nacelle
        # d/d(Pem) [(Pem * (1 - eff) + gearbox_heat_factor * Q_gb) / specific_cooling_capacity_W_per_kg * conversion]
        # = (1 - eff) / specific_cooling_capacity_W_per_kg * conversion
        partials['electric_motor_tms_weight_per_nac', 'rated_power_em_per_nacelle'] = (1.0 - eff) / specific_cooling_capacity_W_per_kg * conversion
        
        # Partial w.r.t. efficiency
        # d/d(eff) [(Pem * (1 - eff) + gearbox_heat_factor * Q_gb) / specific_cooling_capacity_W_per_kg * conversion]
        # = (-Pem) / specific_cooling_capacity_W_per_kg * conversion
        partials['electric_motor_tms_weight_per_nac', 'efficiency'] = -Pem / specific_cooling_capacity_W_per_kg * conversion
        
        # Partial w.r.t. gearbox_heat
        # d/d(Q_gb) [(Pem * (1 - eff) + gearbox_heat_factor * Q_gb) / specific_cooling_capacity_W_per_kg * conversion]
        # = gearbox_heat_factor / specific_cooling_capacity_W_per_kg * conversion
        partials['electric_motor_tms_weight_per_nac', 'gearbox_heat'] = gearbox_heat_factor / specific_cooling_capacity_W_per_kg * conversion
        
        # Partial w.r.t. specific_cooling_capacity_em_tms
        # d/d(specific_cooling_capacity) [heat_generated / (specific_cooling_capacity * 1000) * conversion]
        # = -heat_generated / (specific_cooling_capacity^2 * 1000) * conversion
        partials['electric_motor_tms_weight_per_nac', 'specific_cooling_capacity_em_tms'] = -heat_generated / (specific_cooling_capacity_W_per_kg * specific_cooling_capacity) * conversion
        
        # Partials for C.G
        cg_percentage = self.options['cg_electric_motor_tms_percentage']
        partials['cg_electric_motor_tms', 'MAC'] = cg_percentage / 100.0
        partials['cg_electric_motor_tms', 'LEMAC'] = 1.0


class TurbineTMS(om.ExplicitComponent):
    """
    Calculates turbine thermal management system (TMS) weight using kW/kg method.
    
    Formula: Weight_per_nac = (1.2 * Q_heat) / (1.7 kW/kg) * conversion
    
    where:
    - Q_heat: turbine heat generation per nacelle (W)
    - 20% margin is applied to heat load (multiply by 1.2)
    - 1.7 kW/kg: power density for turbine TMS
    - conversion: kg to lbm (2.20462)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('heat_load_factor', default=1.2, types=float,
                           desc='Margin factor applied to heat load (20% margin = 1.2)')
        self.options.declare('cg_turbine_tms_percentage', default=-30.6, types=float,
                           desc='Turbine TMS C.G location as percentage of MAC (same for inboard and outboard)')

    def setup(self):
        self.add_input('turbine_heat_per_nac', val=15000.0, units='W', desc='Turbine heat generation per nacelle (15 kW = 15000 W)')
        self.add_input('specific_cooling_capacity_turbine_tms', val=1.7, units='W/kg', desc='Power density for Turbine TMS in kW/kg')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('turbine_tms_weight_per_nac', val=0.0, units='lbm', desc='Turbine TMS weight (per nacelle)')
        self.add_output('cg_turbine_tms', val=0.0, units='inch', desc='Turbine TMS C.G location (weighted average of 2 inboard + 2 outboard)')
        
        # Declare partials
        self.declare_partials('turbine_tms_weight_per_nac', ['turbine_heat_per_nac', 'specific_cooling_capacity_turbine_tms'])
        self.declare_partials('cg_turbine_tms', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get constants from options
        specific_cooling_capacity = inputs['specific_cooling_capacity_turbine_tms']  # kW/kg from input
        conversion = 2.20462  # kg to lbm
        
        # Use turbine heat directly (no 1.2 factor applied)
        turbine_heat = inputs['turbine_heat_per_nac']  # W
        
        # Calculate weight per nacelle: Weight = (turbine_heat in W) / (specific_cooling_capacity in W/kg) = kg
        turbine_tms_weight_per_nac_kg = turbine_heat / specific_cooling_capacity 
        
        # Convert from kg to lbm (per nacelle)
        outputs['turbine_tms_weight_per_nac'] = turbine_tms_weight_per_nac_kg * conversion
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        # Since inboard and outboard have same percentage, weighted average is just the same value
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_turbine_tms_percentage']
        outputs['cg_turbine_tms'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        specific_cooling_capacity = inputs['specific_cooling_capacity_turbine_tms']
        specific_cooling_capacity_W_per_kg = specific_cooling_capacity * 1000.0
        conversion = 2.20462
        turbine_heat = inputs['turbine_heat_per_nac']
        
        # Partial w.r.t. turbine_heat_per_nac
        # d/d(Q_turb) [Q_turb / specific_cooling_capacity_W_per_kg * conversion]
        # = 1 / specific_cooling_capacity_W_per_kg * conversion
        partials['turbine_tms_weight_per_nac', 'turbine_heat_per_nac'] = 1.0 / specific_cooling_capacity_W_per_kg * conversion
        
        # Partial w.r.t. specific_cooling_capacity_turbine_tms
        # d/d(specific_cooling_capacity) [turbine_heat / (specific_cooling_capacity * 1000) * conversion]
        # = -turbine_heat / (specific_cooling_capacity^2 * 1000) * conversion
        partials['turbine_tms_weight_per_nac', 'specific_cooling_capacity_turbine_tms'] = -turbine_heat / (specific_cooling_capacity_W_per_kg * specific_cooling_capacity) * conversion
        
        # Partials for C.G
        cg_percentage = self.options['cg_turbine_tms_percentage']
        partials['cg_turbine_tms', 'MAC'] = cg_percentage / 100.0
        partials['cg_turbine_tms', 'LEMAC'] = 1.0


class BatteryTMS(om.ExplicitComponent):
    """
    Calculates battery thermal management system (TMS) weight using kW/kg method.
    
    Formula: Weight_per_nac = (1.3 * Q_heat) / (1 kW/kg) * conversion
    
    where:
    - Q_heat: battery heat generation per nacelle (W)
    - 30% margin is applied to heat load (multiply by 1.3)
    - 1 kW/kg: power density assumption
    - conversion: kg to lbm (2.20462)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('heat_load_factor', default=1.3, types=float,
                           desc='Margin factor applied to heat load (30% margin = 1.3)')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_battery_tms_percentage', default=52.6, types=float,
                           desc='Battery TMS C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('battery_heat_per_nacelle', val=60000.0, units='W', desc='Battery heat generation per nacelle (60 kW = 60000 W)')
        self.add_input('specific_cooling_capacity_battery_tms', val=1.0, units='kW/kg', desc='Power density for Battery TMS in kW/kg')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        self.add_output('battery_tms_weight_per_nac', val=0.0, units='lbm', desc='Battery TMS weight (per nacelle)')
        self.add_output('cg_battery_tms', val=0.0, units='inch', desc='Battery TMS C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('battery_tms_weight_per_nac', ['battery_heat_per_nacelle', 'specific_cooling_capacity_battery_tms'])
        self.declare_partials('cg_battery_tms', 'fuselage_length')

    def compute(self, inputs, outputs):
        # Get constants from options
        specific_cooling_capacity = inputs['specific_cooling_capacity_battery_tms']  # kW/kg from input
        conversion = 2.20462  # kg to lbm
        
        # Use battery heat directly (no 1.3 factor applied)
        battery_heat = inputs['battery_heat_per_nacelle']  # W
        
        # Convert power density from kW/kg to W/kg
        specific_cooling_capacity_W_per_kg = specific_cooling_capacity * 1000.0  # W/kg
        
        # Calculate weight per nacelle: Weight = (battery_heat in W) / (specific_cooling_capacity in W/kg) = kg
        battery_tms_weight_per_nac_kg = battery_heat / specific_cooling_capacity_W_per_kg
        
        # Convert from kg to lbm (per nacelle)
        outputs['battery_tms_weight_per_nac'] = battery_tms_weight_per_nac_kg * conversion
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_battery_tms_percentage']
        outputs['cg_battery_tms'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        specific_cooling_capacity = inputs['specific_cooling_capacity_battery_tms']
        specific_cooling_capacity_W_per_kg = specific_cooling_capacity * 1000.0
        conversion = 2.20462
        battery_heat = inputs['battery_heat_per_nacelle']
        
        # Partial w.r.t. battery_heat_per_nacelle
        # d/dQ [Q / specific_cooling_capacity_W_per_kg * conversion]
        # = 1 / specific_cooling_capacity_W_per_kg * conversion
        partials['battery_tms_weight_per_nac', 'battery_heat_per_nacelle'] = 1.0 / specific_cooling_capacity_W_per_kg * conversion
        
        # Partial w.r.t. specific_cooling_capacity_battery_tms
        # d/d(specific_cooling_capacity) [battery_heat / (specific_cooling_capacity * 1000) * conversion]
        # = -battery_heat / (specific_cooling_capacity^2 * 1000) * conversion
        partials['battery_tms_weight_per_nac', 'specific_cooling_capacity_battery_tms'] = -battery_heat / (specific_cooling_capacity_W_per_kg * specific_cooling_capacity) * conversion
        
        # Partials for C.G
        cg_percentage = self.options['cg_battery_tms_percentage']
        partials['cg_battery_tms', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class TMSGroup(om.Group):
    """
    Group that calculates total thermal management system (TMS) weight.
    
    Total = Electric Motor TMS + Turbine TMS + Battery TMS
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_nac', default=4, types=int,
                           desc='Number of nacelles')

    def setup(self):
        # Add individual TMS components
        self.add_subsystem('electric_motor_tms', ElectricMotorTMS(), promotes_inputs=[
            'rated_power_em_per_nacelle', 'efficiency', 'gearbox_heat', 'specific_cooling_capacity_em_tms', 'MAC', 'LEMAC'
        ], promotes_outputs=['electric_motor_tms_weight_per_nac', 'cg_electric_motor_tms'])
        
        self.add_subsystem('turbine_tms', TurbineTMS(), promotes_inputs=[
            'turbine_heat_per_nac', 'specific_cooling_capacity_turbine_tms', 'MAC', 'LEMAC'
        ], promotes_outputs=['turbine_tms_weight_per_nac', 'cg_turbine_tms'])
        
        self.add_subsystem('battery_tms', BatteryTMS(), promotes_inputs=[
            'battery_heat_per_nacelle', 'specific_cooling_capacity_battery_tms', 'fuselage_length'
        ], promotes_outputs=['battery_tms_weight_per_nac', 'cg_battery_tms'])

        # Use AddSubtractComp to sum all TMS weights
        adder = om.AddSubtractComp()
        adder.add_equation(
            'total_tms_weight_per_nac',
            input_names=['electric_motor_tms_weight_per_nac', 'turbine_tms_weight_per_nac', 'battery_tms_weight_per_nac'],
            units='lbm',
            desc='Total TMS weight'
        )
        self.add_subsystem('total', adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # ElectricMotorTMS inputs
    ivc.add_output('efficiency', val=0.95, units=None)
    ivc.add_output('rated_power_em_per_nacelle', val=1864000.0, units='W')
    ivc.add_output('gearbox_heat', val=2500.0, units='W')
    # Note: delta_T_em no longer needed (using kW/kg method)
    # TurbineTMS inputs
    ivc.add_output('turbine_heat_per_nac', val=15000.0, units='W')
    # Note: delta_T_turb no longer needed (using kW/kg method)
    # BatteryTMS inputs
    ivc.add_output('battery_heat_per_wing', val=50000.0, units='W')
    # Note: delta_T_batt no longer needed (using kW/kg method)

    model.add_subsystem('tms', TMSGroup(num_nac = 4), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Electric Motor TMS Weight:', prob.get_val('electric_motor_tms_weight_per_nac', units='lbm'))
    print('Turbine TMS Weight:', prob.get_val('turbine_tms_weight_per_nac', units='lbm'))
    print('Battery TMS Weight:', prob.get_val('battery_tms_weight_per_nac', units='lbm'))
    print('Total TMS Weight:', prob.get_val('total_tms_weight_per_nac', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
