import openmdao.api as om
import numpy as np


class FuelWeight(om.ExplicitComponent):
    """
    Calculates fuel weight and C.G location.
    
    Fuel weight is a constant value.
    Fuel C.G follows wing logic: located at a percentage of MAC from LEMAC.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fuel_weight', default=6000.0, types=float,
                           desc='Fuel weight in lbm')
        self.options.declare('cg_fuel_percentage', default=37.4, types=float,
                           desc='Fuel C.G location as percentage of MAC from LEMAC')

    def setup(self):
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('fuel_weight', val=0.0, units='lbm', desc='Fuel weight')
        self.add_output('cg_fuel', val=0.0, units='inch', desc='Fuel C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('fuel_weight', [])  # No inputs, constant weight
        self.declare_partials('cg_fuel', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Fuel weight is constant
        outputs['fuel_weight'] = self.options['fuel_weight']
        
        # Calculate Fuel C.G location using wing logic
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_fuel_percentage']
        outputs['cg_fuel'] = LEMAC + (MAC * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Fuel weight has no partials (constant)
        # Partials for C.G: cg = LEMAC + (MAC * percentage / 100)
        cg_percentage = self.options['cg_fuel_percentage']
        partials['cg_fuel', 'LEMAC'] = 1.0
        partials['cg_fuel', 'MAC'] = cg_percentage / 100.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('MAC', val=106.43, units='inch')
    ivc.add_output('LEMAC', val=949.11, units='inch')

    model.add_subsystem('fuel', FuelWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Fuel Weight:', prob.get_val('fuel_weight', units='lbm'))
    print('Fuel C.G:', prob.get_val('cg_fuel', units='inch'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)

