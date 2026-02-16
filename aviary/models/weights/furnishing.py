import openmdao.api as om
import numpy as np


class InteriorsWeight(om.ExplicitComponent):
    """
    Calculates interiors weight based on cabin dimensions and business class passengers.
    
    Formula: 0.6847 * Length_of_Pax_Cabin * Fuselage_Width * Fuselage_Height + Business_Class_Pax * 50 + 480
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('coefficient', default=0.6847, types=float, desc='Interiors weight coefficient')
        self.options.declare('business_class_factor', default=50.0, types=float, desc='Weight factor per business class passenger')
        self.options.declare('base_weight', default=480.0, types=float, desc='Base interiors weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_interiors_percentage', default=37.0, types=float,
                           desc='Interiors C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('length_pax_cabin', val=63.0, units='ft', desc='Length of passenger cabin')
        self.add_input('fuselage_width', val=10.16, units='ft', desc='Fuselage width')
        self.add_input('fuselage_height', val=9.87, units='ft', desc='Fuselage height')
        self.add_input('business_class_pax', val=0, units=None, desc='Number of business class passengers')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        self.add_output('interiors_weight', val=0.0, units='lbm', desc='Interiors weight')
        self.add_output('cg_interiors', val=0.0, units='inch', desc='Interiors C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('interiors_weight', '*')
        self.declare_partials('cg_interiors', 'fuselage_length')

    def compute(self, inputs, outputs):
        coefficient = self.options['coefficient']
        business_class_factor = self.options['business_class_factor']
        base_weight = self.options['base_weight']
        
        # Formula: 0.6847 * Length * Width * Height + Business_Class_Pax * 50 + 480
        outputs['interiors_weight'] = (coefficient * inputs['length_pax_cabin'] * inputs['fuselage_width'] * inputs['fuselage_height'] + 
                                      inputs['business_class_pax'] * business_class_factor + 
                                      base_weight)
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_interiors_percentage']
        outputs['cg_interiors'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        coefficient = self.options['coefficient']
        business_class_factor = self.options['business_class_factor']
        
        # Partial w.r.t. length_pax_cabin
        partials['interiors_weight', 'length_pax_cabin'] = coefficient * inputs['fuselage_width'] * inputs['fuselage_height']
        
        # Partial w.r.t. fuselage_width
        partials['interiors_weight', 'fuselage_width'] = coefficient * inputs['length_pax_cabin'] * inputs['fuselage_height']
        
        # Partial w.r.t. fuselage_height
        partials['interiors_weight', 'fuselage_height'] = coefficient * inputs['length_pax_cabin'] * inputs['fuselage_width']
        
        # Partial w.r.t. business_class_pax
        partials['interiors_weight', 'business_class_pax'] = business_class_factor
        
        # Partials for C.G
        cg_percentage = self.options['cg_interiors_percentage']
        partials['cg_interiors', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class PaintWeight(om.ExplicitComponent):
    """
    Calculates paint weight.
    Constant weight component (reported separately in ATA 1100).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('paint_weight', default=289.0, types=float, desc='Paint weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_paint_percentage', default=51.7, types=float,
                           desc='Paint C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('paint_weight', val=0.0, units='lbm', desc='Paint weight')
        self.add_output('cg_paint', val=0.0, units='inch', desc='Paint C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_paint', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['paint_weight'] = self.options['paint_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_paint_percentage']
        outputs['cg_paint'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_paint_percentage']
        partials['cg_paint', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class OperationalItemsWeight(om.ExplicitComponent):
    """
    Calculates operational items weight.
    Constant weight component fixed at 1540.59 lbs.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('operational_items_weight', default=1562.59, types=float, desc='Operational items weight in lbm')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_operational_items_percentage', default=26.3, types=float,
                           desc='Operational Items C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_output('operational_items_weight', val=0.0, units='lbm', desc='Operational items weight')
        self.add_output('cg_operational_items', val=0.0, units='inch', desc='Operational Items C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('cg_operational_items', 'fuselage_length')

    def compute(self, inputs, outputs):
        outputs['operational_items_weight'] = self.options['operational_items_weight']
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_operational_items_percentage']
        outputs['cg_operational_items'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
    
    def compute_partials(self, inputs, partials):
        # Partials for C.G
        cg_percentage = self.options['cg_operational_items_percentage']
        partials['cg_operational_items', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('length_pax_cabin', val=63.0, units='ft')
    ivc.add_output('fuselage_width', val=10.16, units='ft')
    ivc.add_output('fuselage_height', val=9.87, units='ft')
    ivc.add_output('business_class_pax', val=0, units=None)

    model.add_subsystem('interiors', InteriorsWeight(), promotes=['*'])
    model.add_subsystem('paint', PaintWeight(), promotes=['*'])
    model.add_subsystem('operational_items', OperationalItemsWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Interiors Weight:', prob.get_val('interiors_weight', units='lbm'))
    print('Paint Weight:', prob.get_val('paint_weight', units='lbm'))
    print('Operational Items Weight:', prob.get_val('operational_items_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
