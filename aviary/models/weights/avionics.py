import openmdao.api as om
import numpy as np

class AvionicsWeight(om.ExplicitComponent):
    """
    Calculates avionics weight with separate components for Instruments,
    Communications, Navigation, and IMA (Integrated Modular Avionics).
    All weights are constant values that can be configured via options.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Default values that can be overridden when instantiating the component
        # Using types=float for options (units are specified on inputs/outputs, not options)
        self.options.declare('instruments_weight', default=63.0, types=float,
                           desc='Instruments weight in lbm')
        self.options.declare('communications_weight', default=41.53, types=float,
                           desc='Communications weight in lbm')
        self.options.declare('navigation_weight', default=153.0, types=float,
                           desc='Navigation weight in lbm')
        self.options.declare('ima_weight', default=60.5, types=float,
                           desc='IMA (Integrated Modular Avionics) weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_avionics_percentage', default=31.1, types=float,
                           desc='Avionics C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        # Outputs for each avionics component
        self.add_output('instruments_weight', val=0.0, units='lbm', desc='Instruments weight')
        self.add_output('communications_weight', val=0.0, units='lbm', desc='Communications weight')
        self.add_output('navigation_weight', val=0.0, units='lbm', desc='Navigation weight')
        self.add_output('ima_weight', val=0.0, units='lbm', desc='IMA (Integrated Modular Avionics) weight')
        self.add_output('avionics_weight', val=0.0, units='lbm', desc='Total avionics weight')
        self.add_output('cg_avionics', val=0.0, units='inch', desc='Avionics C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_avionics', 'fuselage_length')

    def compute(self, inputs, outputs):
        # Get weights from options (can be overridden at instantiation)
        instruments_weight = self.options['instruments_weight']
        communications_weight = self.options['communications_weight']
        navigation_weight = self.options['navigation_weight']
        ima_weight = self.options['ima_weight']
        
        outputs['instruments_weight'] = instruments_weight
        outputs['communications_weight'] = communications_weight
        outputs['navigation_weight'] = navigation_weight
        outputs['ima_weight'] = ima_weight
        
        # Total avionics weight
        outputs['avionics_weight'] = instruments_weight + communications_weight + navigation_weight + ima_weight
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_avionics_percentage']
        outputs['cg_avionics'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_avionics_percentage']
        partials['cg_avionics', 'fuselage_length'] = 12.0 * cg_percentage / 100.0

if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    # No inputs needed for AvionicsWeight (uses options for constant weights)
    model.add_subsystem('avionics', AvionicsWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)

    print('Instruments Weight:', prob.get_val('instruments_weight', units='lbm'))
    print('Communications Weight:', prob.get_val('communications_weight', units='lbm'))
    print('Navigation Weight:', prob.get_val('navigation_weight', units='lbm'))
    print('IMA Weight:', prob.get_val('ima_weight', units='lbm'))
    print('Total Avionics Weight:', prob.get_val('avionics_weight', units='lbm'))
    # No partials to check (no inputs)