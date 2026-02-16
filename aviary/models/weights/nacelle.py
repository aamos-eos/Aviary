import openmdao.api as om
import numpy as np


class InboardNacelleWeight(om.ExplicitComponent):
    """Calculates the weight of inboard nacelles with 4 components."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, types=int,desc='number of props per aircraft')
        self.options.declare('cg_nacelle_inboard_percentage', default=26.7, types=float,
                           desc='Inboard Nacelle C.G location as percentage of MAC')

    def setup(self):
        self.add_input('total_battery_weight', val=1864000.0, units='lbm', desc='Total Battery weight')
        self.add_input('total_propulsion_weight', val=1864000.0, units='lbm', desc='Total Propulsion System weight')
        

        self.add_input('nacelle_length_inboard', val=322.39/12.0, units='ft')
        self.add_input('nacelle_width_inboard', val=42.5/12.0, units='ft')
        self.add_input('nacelle_height_inboard', val=92.5/12.0, units='ft')
        self.add_input('dive_speed', val=315.0, units='knot')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        # Only keep total weight output as requested
        self.add_output('nacelle_weight_inboard', val=0, units='lbm')
        self.add_output('cg_nacelle_inboard', val=0.0, units='inch', desc='Inboard Nacelle C.G location')
        
        # Declare partials
        self.declare_partials('nacelle_weight_inboard', '*')
        self.declare_partials('cg_nacelle_inboard', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        radius = (inputs['nacelle_width_inboard'] + inputs['nacelle_height_inboard']) / 4.0
        surface_area = 0.8 * (2.0 * np.pi * radius * inputs['nacelle_length_inboard'] + 2 * np.pi * radius**2)
        
        nac_engine_weight = inputs['total_propulsion_weight'] / self.options['num_props']
        nac_batteries_weight = inputs['total_battery_weight'] / self.options['num_props']


        engine_mounts_weight = 0.05 * nac_engine_weight
        batteries_mount_weight = 0.05 * nac_batteries_weight
        structure_fairings_weight = 0.0054 * (inputs['dive_speed'] ** 0.5) * (surface_area ** 1.3)

        base_total = engine_mounts_weight + batteries_mount_weight + structure_fairings_weight
        misc_items_weight = 0.1 * base_total

        outputs['nacelle_weight_inboard'] = engine_mounts_weight + batteries_mount_weight + structure_fairings_weight + misc_items_weight
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_nacelle_inboard_percentage']
        outputs['cg_nacelle_inboard'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        num_props = self.options['num_props']
        
        # Intermediate calculations
        radius = (inputs['nacelle_width_inboard'] + inputs['nacelle_height_inboard']) / 4.0
        L = inputs['nacelle_length_inboard']
        surface_area = 0.8 * (2.0 * np.pi * radius * L + 2 * np.pi * radius**2)
        
        # Partials w.r.t. total_propulsion_weight
        # engine_mounts_weight = 0.05 * total_propulsion_weight / num_props
        # contribution to total = engine_mounts_weight * 1.1 (including misc factor)
        partials['nacelle_weight_inboard', 'total_propulsion_weight'] = 0.05 / num_props * 1.1
        
        # Partials w.r.t. total_battery_weight
        # batteries_mount_weight = 0.05 * total_battery_weight / num_props
        partials['nacelle_weight_inboard', 'total_battery_weight'] = 0.05 / num_props * 1.1
        
        # Partials w.r.t. dive_speed
        V = inputs['dive_speed']
        partials['nacelle_weight_inboard', 'dive_speed'] = 0.0054 * 0.5 * (V ** -0.5) * (surface_area ** 1.3) * 1.1
        
        # Partials w.r.t. dimensions (affect surface_area)
        dSA_dr = 0.8 * (2.0 * np.pi * L + 4.0 * np.pi * radius)
        dSA_dw = dSA_dr * 0.25
        dSA_dh = dSA_dr * 0.25
        dSA_dL = 0.8 * 2.0 * np.pi * radius
        
        structure_partial_factor = 0.0054 * (V ** 0.5) * 1.3 * (surface_area ** 0.3) * 1.1
        
        partials['nacelle_weight_inboard', 'nacelle_width_inboard'] = structure_partial_factor * dSA_dw
        partials['nacelle_weight_inboard', 'nacelle_height_inboard'] = structure_partial_factor * dSA_dh
        partials['nacelle_weight_inboard', 'nacelle_length_inboard'] = structure_partial_factor * dSA_dL
        
        # Partials for C.G
        cg_percentage = self.options['cg_nacelle_inboard_percentage']
        partials['cg_nacelle_inboard', 'MAC'] = cg_percentage / 100.0
        partials['cg_nacelle_inboard', 'LEMAC'] = 1.0


class OutboardNacelleWeight(om.ExplicitComponent):
    """Calculates the weight of outboard nacelles with 4 components."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, types=int,desc='number of props per aircraft')
        self.options.declare('cg_nacelle_outboard_percentage', default=37.8, types=float,
                           desc='Outboard Nacelle C.G location as percentage of MAC')

    def setup(self):

        self.add_input('total_battery_weight', val=1864000.0, units='lbm', desc='Total Battery weight')
        self.add_input('total_propulsion_weight', val=1864000.0, units='lbm', desc='Total Propulsion System weight')

        self.add_input('nacelle_length_outboard', val=304.76/12.0, units='ft')
        self.add_input('nacelle_width_outboard', val=38.3/12.0, units='ft')
        self.add_input('nacelle_height_outboard', val=56.0/12.0, units='ft')
        self.add_input('dive_speed', val=315.0, units='knot')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        # Only keep total weight output as requested
        self.add_output('nacelle_weight_outboard', val=0.0, units='lbm')
        self.add_output('cg_nacelle_outboard', val=0.0, units='inch', desc='Outboard Nacelle C.G location')
        
        # Declare partials
        self.declare_partials('nacelle_weight_outboard', '*')
        self.declare_partials('cg_nacelle_outboard', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        radius = (inputs['nacelle_width_outboard'] + inputs['nacelle_height_outboard']) / 4.0
        surface_area = 2.0 * np.pi * radius * inputs['nacelle_length_outboard'] + 2 * np.pi * radius**2
        
        nac_engine_weight = inputs['total_propulsion_weight'] / self.options['num_props']
        nac_batteries_weight = inputs['total_battery_weight'] / self.options['num_props']
        
        engine_mounts_weight = 0.05 * nac_engine_weight
        batteries_mount_weight = 0.05 * nac_batteries_weight
        structure_fairings_weight = 0.0054 * (inputs['dive_speed'] ** 0.5) * (surface_area ** 1.3)
        
        base_total = engine_mounts_weight + batteries_mount_weight + structure_fairings_weight
        misc_items_weight = 0.1 * base_total
        
        outputs['nacelle_weight_outboard'] = engine_mounts_weight + batteries_mount_weight + structure_fairings_weight + misc_items_weight
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = self.options['cg_nacelle_outboard_percentage']
        outputs['cg_nacelle_outboard'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        num_props = self.options['num_props']
        
        # Intermediate calculations
        radius = (inputs['nacelle_width_outboard'] + inputs['nacelle_height_outboard']) / 4.0
        L = inputs['nacelle_length_outboard']
        surface_area = 2.0 * np.pi * radius * L + 2 * np.pi * radius**2
        
        # Partials w.r.t. total_propulsion_weight
        # engine_mounts_weight = 0.05 * total_propulsion_weight / num_props
        # contribution to total = engine_mounts_weight * 1.1 (including misc factor)
        partials['nacelle_weight_outboard', 'total_propulsion_weight'] = 0.05 / num_props * 1.1
        
        # Partials w.r.t. total_battery_weight
        # batteries_mount_weight = 0.05 * total_battery_weight / num_props
        partials['nacelle_weight_outboard', 'total_battery_weight'] = 0.05 / num_props * 1.1
        
        # Partials w.r.t. dive_speed
        V = inputs['dive_speed']
        partials['nacelle_weight_outboard', 'dive_speed'] = 0.0054 * 0.5 * (V ** -0.5) * (surface_area ** 1.3) * 1.1
        
        # Partials w.r.t. dimensions
        dSA_dr = 2.0 * np.pi * L + 4.0 * np.pi * radius
        dSA_dw = dSA_dr * 0.25
        dSA_dh = dSA_dr * 0.25
        dSA_dL = 2.0 * np.pi * radius
        
        structure_partial_factor = 0.0054 * (V ** 0.5) * 1.3 * (surface_area ** 0.3) * 1.1
        
        partials['nacelle_weight_outboard', 'nacelle_width_outboard'] = structure_partial_factor * dSA_dw
        partials['nacelle_weight_outboard', 'nacelle_height_outboard'] = structure_partial_factor * dSA_dh
        partials['nacelle_weight_outboard', 'nacelle_length_outboard'] = structure_partial_factor * dSA_dL
        
        # Partials for C.G
        cg_percentage = self.options['cg_nacelle_outboard_percentage']
        partials['cg_nacelle_outboard', 'MAC'] = cg_percentage / 100.0
        partials['cg_nacelle_outboard', 'LEMAC'] = 1.0


class NacelleGroup(om.Group):
    """
    Group that calculates total nacelle weight and C.G.
    
    Total = 2 × inboard nacelle weight + 2 × outboard nacelle weight
    Total C.G = weighted average of 2 inboard + 2 outboard
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, types=int,desc='number of props per aircraft')

    def setup(self):
        # Add individual weight components
        num_props = self.options['num_props']
        self.add_subsystem('inboard_nacelle', InboardNacelleWeight(num_props=self.options['num_props']), promotes=['*'])
        self.add_subsystem('outboard_nacelle', OutboardNacelleWeight(num_props=self.options['num_props']), promotes=['*'])

        # Use AddSubtractComp to sum weights with scaling factors (2 of each nacelle type)
        adder = om.AddSubtractComp()
        adder.add_equation(
            'total_nacelle_weight',
            input_names=['nacelle_weight_inboard', 'nacelle_weight_outboard'],
            scaling_factors=[num_props/2.0, num_props/2.0],
            units='lbm',
            desc='Total nacelle weight (2 inboard + 2 outboard)'
        )
        self.add_subsystem('total', adder, promotes=['*'])
        
        # Calculate weighted average C.G for total nacelle weight
        # C.G_total = (2 * W_in * cg_in + 2 * W_out * cg_out) / (2 * W_in + 2 * W_out)
        # = (W_in * cg_in + W_out * cg_out) / (W_in + W_out)
        # Add small epsilon to denominator to prevent division by zero
        cg_adder = om.ExecComp(
            'cg_total_nacelle = (nacelle_weight_inboard * cg_nacelle_inboard + nacelle_weight_outboard * cg_nacelle_outboard) / (nacelle_weight_inboard + nacelle_weight_outboard + 1e-10)',
            nacelle_weight_inboard={'val': 0.0, 'units': 'lbm'},
            nacelle_weight_outboard={'val': 0.0, 'units': 'lbm'},
            cg_nacelle_inboard={'val': 0.0, 'units': 'inch'},
            cg_nacelle_outboard={'val': 0.0, 'units': 'inch'},
            cg_total_nacelle={'val': 0.0, 'units': 'inch'}
        )
        self.add_subsystem('cg_calculator', cg_adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Total weight inputs (shared by both nacelle types)
    ivc.add_output('total_propulsion_weight', val=20000.0, units='lbm')
    ivc.add_output('total_battery_weight', val=4000.0, units='lbm')
    # Inboard nacelle geometry inputs
    ivc.add_output('nacelle_length_inboard', val=322.39/12.0, units='ft')
    ivc.add_output('nacelle_width_inboard', val=42.5/12.0, units='ft')
    ivc.add_output('nacelle_height_inboard', val=92.5/12.0, units='ft')
    # Outboard nacelle geometry inputs
    ivc.add_output('nacelle_length_outboard', val=304.76/12.0, units='ft')
    ivc.add_output('nacelle_width_outboard', val=38.3/12.0, units='ft')
    ivc.add_output('nacelle_height_outboard', val=56.0/12.0, units='ft')
    # Shared inputs
    ivc.add_output('dive_speed', val=315.0, units='knot')
    ivc.add_output('MAC', val=106.43, units='inch')
    ivc.add_output('LEMAC', val=949.11, units='inch')

    model.add_subsystem('nacelle', NacelleGroup(num_props=4), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Inboard Nacelle Weight:', prob.get_val('nacelle_weight_inboard', units='lbm'))
    print('Outboard Nacelle Weight:', prob.get_val('nacelle_weight_outboard', units='lbm'))
    print('Total Nacelle Weight:', prob.get_val('total_nacelle_weight', units='lbm'))
    print('Inboard Nacelle C.G:', prob.get_val('cg_nacelle_inboard', units='inch'))
    print('Outboard Nacelle C.G:', prob.get_val('cg_nacelle_outboard', units='inch'))
    print('Total Nacelle C.G:', prob.get_val('cg_total_nacelle', units='inch'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
