import openmdao.api as om
import numpy as np


class FireProtectionWeight(om.ExplicitComponent):
    """
    Calculates fire protection system weight.
    Constant weight component.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fire_protection_weight', default=276.0, types=float, desc='Fire protection system weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_fire_protection_percentage', default=65.6, types=float,
                           desc='Fire Protection C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('fire_protection_weight', val=0.0, units='lbm', desc='Fire protection system weight')
        self.add_output('cg_fire_protection', val=0.0, units='inch', desc='Fire Protection C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_fire_protection', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['fire_protection_weight'] = self.options['fire_protection_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_fire_protection_percentage']
        outputs['cg_fire_protection'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_fire_protection_percentage']
        partials['cg_fire_protection', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class IceProtectionWeight(om.ExplicitComponent):
    """
    Calculates ice protection system weight.
    Constant weight component.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('ice_protection_weight', default=309.9, types=float, desc='Ice protection system weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('weight_fuselage_percentage', default=25.9, types=float,
                           desc='Percentage of ice protection weight in fuselage')
        self.options.declare('cg_ice_protection_fuselage_percentage', default=96.0, types=float,
                           desc='Ice Protection Fuselage C.G location as percentage of fuselage length')
        self.options.declare('cg_ice_protection_wing_percentage', default=-6.4, types=float,
                           desc='Ice Protection Wing C.G location as percentage of MAC')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('ice_protection_weight', val=0.0, units='lbm', desc='Ice protection system weight')
        self.add_output('cg_ice_protection', val=0.0, units='inch', desc='Ice Protection total C.G location (weighted average of fuselage and wing portions)')
        
        # Declare partials
        self.declare_partials('cg_ice_protection', ['fuselage_length', 'MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        total_weight = self.options['ice_protection_weight']
        outputs['ice_protection_weight'] = total_weight
        
        # Split weight into fuselage and wing portions
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        # Calculate C.G for fuselage portion
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        cg_fuselage_pct = self.options['cg_ice_protection_fuselage_percentage']
        cg_fuselage = fs_start + (fuselage_length_in * cg_fuselage_pct / 100.0)
        
        # Calculate C.G for wing portion
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_wing_pct = self.options['cg_ice_protection_wing_percentage']
        cg_wing = LEMAC + (MAC * cg_wing_pct / 100.0)
        
        # Calculate weighted average total C.G
        outputs['cg_ice_protection'] = (weight_fuselage * cg_fuselage + weight_wing * cg_wing) / total_weight
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        total_weight = self.options['ice_protection_weight']
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        cg_fuselage_pct = self.options['cg_ice_protection_fuselage_percentage']
        cg_wing_pct = self.options['cg_ice_protection_wing_percentage']
        
        # Partials w.r.t. lt (affects fuselage C.G only)
        partials['cg_ice_protection', 'fuselage_length'] = weight_fuselage * 12.0 * cg_fuselage_pct / 100.0 / total_weight
        
        # Partials w.r.t. MAC (affects wing C.G only)
        partials['cg_ice_protection', 'MAC'] = weight_wing * cg_wing_pct / 100.0 / total_weight
        
        # Partials w.r.t. LEMAC (affects wing C.G only)
        partials['cg_ice_protection', 'LEMAC'] = weight_wing / total_weight


class OxygenSystemWeight(om.ExplicitComponent):
    """
    Calculates oxygen system weight.
    Constant weight component.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('oxygen_system_weight', default=37.0, types=float, desc='Oxygen system weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_oxygen_system_percentage', default=7.0, types=float,
                           desc='Oxygen System C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('oxygen_system_weight', val=0.0, units='lbm', desc='Oxygen system weight')
        self.add_output('cg_oxygen_system', val=0.0, units='inch', desc='Oxygen System C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_oxygen_system', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['oxygen_system_weight'] = self.options['oxygen_system_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_oxygen_system_percentage']
        outputs['cg_oxygen_system'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_oxygen_system_percentage']
        partials['cg_oxygen_system', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class WaterAndWasteWeight(om.ExplicitComponent):
    """
    Calculates water and waste system weight.
    Constant weight component.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('water_and_waste_weight', default=167.0, types=float, desc='Water and waste system weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_water_and_waste_percentage', default=25.2, types=float,
                           desc='Water and Waste C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('water_and_waste_weight', val=0.0, units='lbm', desc='Water and waste system weight')
        self.add_output('cg_water_and_waste', val=0.0, units='inch', desc='Water and Waste C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_water_and_waste', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['water_and_waste_weight'] = self.options['water_and_waste_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_water_and_waste_percentage']
        outputs['cg_water_and_waste'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_water_and_waste_percentage']
        partials['cg_water_and_waste', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class MechanicalSystemsGroup(om.Group):
    """
    Group that calculates total mechanical systems weight.
    
    Total = Fire Protection + Ice Protection + Oxygen System + Water and Waste
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Add individual weight components
        self.add_subsystem('fire_protection', FireProtectionWeight(), promotes=['*'])
        self.add_subsystem('ice_protection', IceProtectionWeight(), promotes=['*'])
        self.add_subsystem('oxygen_system', OxygenSystemWeight(), promotes=['*'])
        self.add_subsystem('water_and_waste', WaterAndWasteWeight(), promotes=['*'])

        # Use AddSubtractComp to sum all weights
        adder = om.AddSubtractComp()
        adder.add_equation(
            'total_mechanical_systems_weight',
            input_names=['fire_protection_weight', 'ice_protection_weight', 
                        'oxygen_system_weight', 'water_and_waste_weight'],
            units='lbm',
            desc='Total mechanical systems weight'
        )
        self.add_subsystem('total', adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    model.add_subsystem('mechanical_systems', MechanicalSystemsGroup(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Fire Protection Weight:', prob.get_val('fire_protection_weight', units='lbm'))
    print('Ice Protection Weight:', prob.get_val('ice_protection_weight', units='lbm'))
    print('Oxygen System Weight:', prob.get_val('oxygen_system_weight', units='lbm'))
    print('Water and Waste Weight:', prob.get_val('water_and_waste_weight', units='lbm'))
    print('Total Mechanical Systems Weight:', prob.get_val('total_mechanical_systems_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
