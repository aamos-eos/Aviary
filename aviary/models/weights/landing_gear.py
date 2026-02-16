import openmdao.api as om
import numpy as np

class LandingGearWeight(om.ExplicitComponent):
    """
    Calculates landing gear weight with separate components for Main Landing Gear (MLG),
    Nose Landing Gear (NLG), and miscellaneous items.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Default values that can be overridden when instantiating the component
        self.options.declare('misc_items_weight', default=785.76, types=float,
                           desc='Landing gear miscellaneous items weight in lbm')
        self.options.declare('mlg_coefficient', default=0.0077, types=float,
                           desc='Main landing gear weight coefficient')
        self.options.declare('nlg_coefficient', default=0.03, types=float,
                           desc='Nose landing gear weight coefficient')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_nose_landing_gear_percentage', default=13.0, types=float,
                           desc='Nose Landing Gear C.G location as percentage of fuselage length')
        self.options.declare('cg_main_landing_gear_percentage', default=47.8, types=float,
                           desc='Main Landing Gear C.G location as percentage of MAC')

    def setup(self):
        self.add_input('landing_weight', val=0.0, units='lbm', desc='Aircraft landing weight')
        self.add_input('mlg_height', val=54.74, units='inch', desc='Main landing gear height')
        self.add_input('nlg_height', val=65.40, units='inch', desc='Nose landing gear height')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        # Outputs: individual weights (including misc items) and total
        self.add_output('nose_landing_gear_weight', val=0.0, units='lbm', desc='Nose Landing Gear weight (NLG + 14% misc items)')
        self.add_output('main_landing_gear_weight', val=0.0, units='lbm', desc='Main Landing Gear weight (MLG + 86% misc items)')
        self.add_output('landing_gear_weight', val=0.0, units='lbm', desc='Total landing gear weight')
        self.add_output('cg_nose_landing_gear', val=0.0, units='inch', desc='Nose Landing Gear C.G location (Fuselage Station)')
        self.add_output('cg_main_landing_gear', val=0.0, units='inch', desc='Main Landing Gear C.G location')
        
        # Declare partials
        self.declare_partials('nose_landing_gear_weight', ['landing_weight', 'nlg_height'])
        self.declare_partials('main_landing_gear_weight', ['landing_weight', 'mlg_height'])
        self.declare_partials('landing_gear_weight', ['landing_weight', 'mlg_height', 'nlg_height'])
        self.declare_partials('cg_nose_landing_gear', 'fuselage_length')
        self.declare_partials('cg_main_landing_gear', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get coefficients from options
        mlg_coeff = self.options['mlg_coefficient']
        nlg_coeff = self.options['nlg_coefficient']
        
        # Main Landing Gear weight: coefficient * Landing_Weight^0.95 * MLG_height^0.43
        mlg_weight = mlg_coeff * (inputs['landing_weight'] ** 0.95) * (inputs['mlg_height'] ** 0.43)
        
        # Nose Landing Gear weight: coefficient * Landing_Weight^0.67 * NLG_height^0.43
        nlg_weight = nlg_coeff * (inputs['landing_weight'] ** 0.67) * (inputs['nlg_height'] ** 0.43)
        
        # Miscellaneous items weight (constant)
        misc_items_weight = self.options['misc_items_weight']
        
        # Split misc items: 14% to NLG, 86% to MLG
        nlg_misc = 0.14 * misc_items_weight
        mlg_misc = 0.86 * misc_items_weight
        
        # Calculate totals (gear weight + misc items)
        outputs['nose_landing_gear_weight'] = nlg_weight + nlg_misc
        outputs['main_landing_gear_weight'] = mlg_weight + mlg_misc
        
        # Total landing gear weight
        outputs['landing_gear_weight'] = outputs['nose_landing_gear_weight'] + outputs['main_landing_gear_weight']
        
        # Calculate C.G location for Nose Landing Gear
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_nose_landing_gear_percentage']
        outputs['cg_nose_landing_gear'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
        
        # Calculate C.G location for Main Landing Gear
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_mlg_percentage = self.options['cg_main_landing_gear_percentage']
        outputs['cg_main_landing_gear'] = LEMAC + (MAC * cg_mlg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Get coefficients from options
        mlg_coeff = self.options['mlg_coefficient']
        nlg_coeff = self.options['nlg_coefficient']
        
        landing = inputs['landing_weight']
        mlg_h = inputs['mlg_height']
        nlg_h = inputs['nlg_height']
        
        # Partials for nose_landing_gear_weight = nlg_weight + 0.14 * misc_items
        # d/dW (nlg_coeff * W^0.67 * H^0.43) = nlg_coeff * 0.67 * W^-0.33 * H^0.43
        partials['nose_landing_gear_weight', 'landing_weight'] = nlg_coeff * 0.67 * (landing ** -0.33) * (nlg_h ** 0.43)
        # d/dH_nlg (nlg_coeff * W^0.67 * H^0.43) = nlg_coeff * W^0.67 * 0.43 * H^-0.57
        partials['nose_landing_gear_weight', 'nlg_height'] = nlg_coeff * (landing ** 0.67) * 0.43 * (nlg_h ** -0.57)
        
        # Partials for main_landing_gear_weight = mlg_weight + 0.86 * misc_items
        # d/dW (mlg_coeff * W^0.95 * H^0.43) = mlg_coeff * 0.95 * W^-0.05 * H^0.43
        partials['main_landing_gear_weight', 'landing_weight'] = mlg_coeff * 0.95 * (landing ** -0.05) * (mlg_h ** 0.43)
        # d/dH_mlg (mlg_coeff * W^0.95 * H^0.43) = mlg_coeff * W^0.95 * 0.43 * H^-0.57
        partials['main_landing_gear_weight', 'mlg_height'] = mlg_coeff * (landing ** 0.95) * 0.43 * (mlg_h ** -0.57)
        
        # Partials for landing_gear_weight = nose_landing_gear_weight + main_landing_gear_weight
        partials['landing_gear_weight', 'landing_weight'] = (mlg_coeff * 0.95 * (landing ** -0.05) * (mlg_h ** 0.43) +
                                                              nlg_coeff * 0.67 * (landing ** -0.33) * (nlg_h ** 0.43))
        partials['landing_gear_weight', 'mlg_height'] = mlg_coeff * (landing ** 0.95) * 0.43 * (mlg_h ** -0.57)
        partials['landing_gear_weight', 'nlg_height'] = nlg_coeff * (landing ** 0.67) * 0.43 * (nlg_h ** -0.57)
        
        # Partials for C.G
        cg_percentage = self.options['cg_nose_landing_gear_percentage']
        partials['cg_nose_landing_gear', 'fuselage_length'] = 12.0 * cg_percentage / 100.0
        
        # Partials for Main Landing Gear C.G
        cg_mlg_percentage = self.options['cg_main_landing_gear_percentage']
        partials['cg_main_landing_gear', 'MAC'] = cg_mlg_percentage / 100.0
        partials['cg_main_landing_gear', 'LEMAC'] = 1.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('landing_weight', val=75000.0, units='lbm')
    ivc.add_output('mlg_height', val=54.74, units='inch')
    ivc.add_output('nlg_height', val=65.40, units='inch')

    model.add_subsystem('landing_gear', LandingGearWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Landing Gear Weight:', prob.get_val('landing_gear_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)