import openmdao.api as om
import numpy as np


class HorizontalTailWeight(om.ExplicitComponent):
    """
    Calculates the weight of the horizontal tail using Torenbeek formula.
    
    Formula: Wh = Kh * Sh * [4.35 * (Sh^0.2 * V_DEAS) / (1000 * sqrt(cos(Λc/2h))) - 0.287]
    
    Where:
    - Kh = 1.1 (constant factor associated with fixed incident h-stab)
    - Sh = horizontal tail surface area (ft²), provided as direct input
    - V_DEAS = dive speed in equivalent airspeed (knot)
    - Λc/2h = sweep angle at half chord (deg)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('Kh', default=1.1, types=float, desc='Factor associated with fixed incident h-stab')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_horizontal_tail_percentage', default=99.0, types=float,
                           desc='Horizontal tail C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('surface', val=209.0, units='ft**2', desc='Horizontal tail surface area')
        self.add_input('Vd', val=315.0, units='knot', desc='Dive speed in equivalent airspeed')
        self.add_input('c2_sweep', val=4.1, units='deg', desc='Sweep angle at half chord for horizontal tail')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        self.add_output('horizontal_tail_weight', val=0.0, units='lbm', desc='Horizontal tail weight')
        self.add_output('cg_horizontal_tail', val=0.0, units='inch', desc='Horizontal tail C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('horizontal_tail_weight', ['surface', 'Vd', 'c2_sweep'])
        self.declare_partials('cg_horizontal_tail', 'fuselage_length')

    def compute(self, inputs, outputs):
        Kh = self.options['Kh']
        Sh = inputs['surface']
        V_DEAS = inputs['Vd']
        sweep_rad = inputs['c2_sweep'] * np.pi / 180.0
        
        # Torenbeek formula: Wh = Kh * Sh * [4.35 * (Sh^0.2 * V_DEAS) / (1000 * sqrt(cos(Λc/2h))) - 0.287]
        sqrt_cos_sweep = np.sqrt(np.cos(sweep_rad))
        denominator = 1000.0 * sqrt_cos_sweep
        numerator = Sh**0.2 * V_DEAS
        
        bracket_value = 4.35 * (numerator / denominator) - 0.287
        outputs['horizontal_tail_weight'] = Kh * Sh * bracket_value
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_horizontal_tail_percentage']
        outputs['cg_horizontal_tail'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        Kh = self.options['Kh']
        Sh = inputs['surface']
        V_DEAS = inputs['Vd']
        sweep_rad = inputs['c2_sweep'] * np.pi / 180.0
        
        cos_sweep = np.cos(sweep_rad)
        sqrt_cos_sweep = np.sqrt(cos_sweep)
        denominator = 1000.0 * sqrt_cos_sweep
        
        bracket_value = 4.35 * (Sh**0.2 * V_DEAS) / denominator - 0.287
        d_bracket_dSh = 4.35 * (0.2 * Sh**(-0.8) * V_DEAS) / denominator
        
        # d(Wh)/d(Sh) = Kh * (bracket_value + Sh * d_bracket_dSh)
        dWh_dSh = Kh * (bracket_value + Sh * d_bracket_dSh)
        
        # Partial w.r.t. surface
        partials['horizontal_tail_weight', 'surface'] = dWh_dSh
        
        # Partial w.r.t. Vd
        d_bracket_dVd = 4.35 * (Sh**0.2) / denominator
        partials['horizontal_tail_weight', 'Vd'] = Kh * Sh * d_bracket_dVd
        
        # Partial w.r.t. c2_sweep
        d_sqrt_cos_dsweep = -0.5 * np.sin(sweep_rad) / sqrt_cos_sweep
        d_denominator_dsweep = 1000.0 * d_sqrt_cos_dsweep
        d_bracket_dsweep = -4.35 * (Sh**0.2 * V_DEAS) * d_denominator_dsweep / (denominator**2)
        # Convert from rad to deg
        partials['horizontal_tail_weight', 'c2_sweep'] = Kh * Sh * d_bracket_dsweep * np.deg2rad(1.0)
        
        # Partials for C.G
        cg_percentage = self.options['cg_horizontal_tail_percentage']
        partials['cg_horizontal_tail', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class VerticalTailWeight(om.ExplicitComponent):
    """
    Calculates the weight of the vertical tail using Torenbeek formula.
    
    Formula: Wv = Kv * Sv * [4.35 * (Sv^0.2 * V_DEAS) / (1000 * sqrt(cos(Λc/2v))) - 0.287]
    
    Where:
    - Kv = 1 + 0.15 * Sh/Sv (factor that depends on horizontal and vertical tail surfaces)
    - Sv = vertical tail surface area (ft²), provided as direct input
    - Sh = horizontal tail surface area (ft²), provided as direct input (for Kv calculation)
    - V_DEAS = dive speed in equivalent airspeed (knot)
    - Λc/2v = sweep angle at half chord (deg)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_vertical_tail_percentage', default=91.9, types=float,
                           desc='Vertical tail C.G location as percentage of fuselage length')

    def setup(self):
        self.add_input('surface', val=162.62, units='ft**2', desc='Vertical tail surface area')
        self.add_input('surface_hor', val=209.0, units='ft**2', desc='Horizontal tail surface area (for Kv calculation)')
        self.add_input('Vd', val=315.0, units='knot', desc='Dive speed in equivalent airspeed')
        self.add_input('c2_sweep', val=38.4, units='deg', desc='Sweep angle at half chord for vertical tail')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        self.add_output('vertical_tail_weight', val=0.0, units='lbm', desc='Vertical tail weight')
        self.add_output('cg_vertical_tail', val=0.0, units='inch', desc='Vertical tail C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('vertical_tail_weight', ['surface', 'surface_hor', 'Vd', 'c2_sweep'])
        self.declare_partials('cg_vertical_tail', 'fuselage_length')

    def compute(self, inputs, outputs):
        Sv = inputs['surface']
        Sh = inputs['surface_hor']
        V_DEAS = inputs['Vd']
        sweep_rad = inputs['c2_sweep'] * np.pi / 180.0
        
        # Kv = 1 + 0.15 * Sh/Sv
        Kv = 1.0 + 0.15 * Sh / Sv
        
        # Torenbeek formula: Wv = Kv * Sv * [4.35 * (Sv^0.2 * V_DEAS) / (1000 * sqrt(cos(Λc/2v))) - 0.287]
        sqrt_cos_sweep = np.sqrt(np.cos(sweep_rad))
        denominator = 1000.0 * sqrt_cos_sweep
        numerator = Sv**0.2 * V_DEAS
        
        bracket_value = 4.35 * (numerator / denominator) - 0.287
        outputs['vertical_tail_weight'] = Kv * Sv * bracket_value
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_vertical_tail_percentage']
        outputs['cg_vertical_tail'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        Sv = inputs['surface']
        Sh = inputs['surface_hor']
        V_DEAS = inputs['Vd']
        sweep_rad = inputs['c2_sweep'] * np.pi / 180.0
        
        Kv = 1.0 + 0.15 * Sh / Sv
        dKv_dSh = 0.15 / Sv
        dKv_dSv = -0.15 * Sh / (Sv**2)
        
        cos_sweep = np.cos(sweep_rad)
        sqrt_cos_sweep = np.sqrt(cos_sweep)
        denominator = 1000.0 * sqrt_cos_sweep
        
        bracket_value = 4.35 * (Sv**0.2 * V_DEAS) / denominator - 0.287
        d_bracket_dSv = 4.35 * (0.2 * Sv**(-0.8) * V_DEAS) / denominator
        
        # d(Wv)/d(Sv) = (dKv/dSv * Sv + Kv) * bracket + Kv * Sv * d_bracket/dSv
        dWv_dSv = (dKv_dSv * Sv + Kv) * bracket_value + Kv * Sv * d_bracket_dSv
        
        # Partial w.r.t. surface (Sv)
        partials['vertical_tail_weight', 'surface'] = dWv_dSv
        
        # Partial w.r.t. surface_hor (Sh)
        # d(Wv)/d(Sh) = dKv/dSh * Sv * bracket
        partials['vertical_tail_weight', 'surface_hor'] = dKv_dSh * Sv * bracket_value
        
        # Partial w.r.t. Vd
        d_bracket_dVd = 4.35 * (Sv**0.2) / denominator
        partials['vertical_tail_weight', 'Vd'] = Kv * Sv * d_bracket_dVd
        
        # Partial w.r.t. c2_sweep
        d_sqrt_cos_dsweep = -0.5 * np.sin(sweep_rad) / sqrt_cos_sweep
        d_denominator_dsweep = 1000.0 * d_sqrt_cos_dsweep
        d_bracket_dsweep = -4.35 * (Sv**0.2 * V_DEAS) * d_denominator_dsweep / (denominator**2)
        # Convert from rad to deg
        partials['vertical_tail_weight', 'c2_sweep'] = Kv * Sv * d_bracket_dsweep * np.deg2rad(1.0)
        
        # Partials for C.G
        cg_percentage = self.options['cg_vertical_tail_percentage']
        partials['cg_vertical_tail', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


class EmpennageGroup(om.Group):
    """
    Group that calculates total empennage weight (horizontal + vertical tail).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        # Add individual weight components - promote common inputs
        self.add_subsystem('horizontal_tail', HorizontalTailWeight(), 
                          promotes_inputs=['Vd', 'fuselage_length'],
                          promotes_outputs=['horizontal_tail_weight', 'cg_horizontal_tail'])
        self.add_subsystem('vertical_tail', VerticalTailWeight(), 
                          promotes_inputs=['Vd', 'fuselage_length'],
                          promotes_outputs=['vertical_tail_weight', 'cg_vertical_tail'])
        
        # Use AddSubtractComp to sum weights
        adder = om.AddSubtractComp()
        adder.add_equation(
            'empennage_weight',
            input_names=['horizontal_tail_weight', 'vertical_tail_weight'],
            units='lbm',
            desc='Total empennage weight'
        )
        self.add_subsystem('total', adder, promotes=['*'])


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('hstab_surface', val=209.0, units='ft**2')
    ivc.add_output('vstab_surface', val=162.62, units='ft**2')
    ivc.add_output('Vd', val=315.0, units='knot')
    ivc.add_output('hstab_c2_sweep', val=4.1, units='deg')
    ivc.add_output('vstab_c2_sweep', val=38.4, units='deg')
    ivc.add_output('fuselage_length', val=97.89, units='ft')

    model.add_subsystem('empennage', EmpennageGroup(), promotes=['*'])
    
    # Connect to non-promoted inputs
    model.connect('hstab_surface', 'empennage.horizontal_tail.surface')
    model.connect('hstab_surface', 'empennage.vertical_tail.surface_hor')  # Kv factor needs hstab surface
    model.connect('hstab_c2_sweep', 'empennage.horizontal_tail.c2_sweep')
    model.connect('vstab_surface', 'empennage.vertical_tail.surface')
    model.connect('vstab_c2_sweep', 'empennage.vertical_tail.c2_sweep')

    prob.setup()
    prob.run_model()

    print('Horizontal Tail Weight:', prob.get_val('horizontal_tail_weight', units='lbm'))
    print('Vertical Tail Weight:', prob.get_val('vertical_tail_weight', units='lbm'))
    print('Total Empennage Weight:', prob.get_val('empennage_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
