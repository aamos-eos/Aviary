import openmdao.api as om


class FuselageMass(om.ExplicitComponent):
    """
    Calculates fuselage weight with shell, stringer, frame, and total components.
    Uses only Wsk1 formula for shell weight (no max() function).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('regression_factor', default=2.4, types=float,
                           desc='Regression factor to account for additional weight components')
        self.options.declare('const_D2', default=0.00575, types=float,
                           desc='Shell weight constant D2')
        self.options.declare('const_D6', default=0.000635, types=float,
                           desc='Stringer/Longeron weight constant D6')
        self.options.declare('frame_factor', default=0.19, types=float,
                           desc='Frame weight factor (fraction of stringer + shell weight)')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_fuselage_percentage', default=47.2, types=float,
                           desc='Fuselage C.G location as percentage of fuselage length')

    def setup(self):
        # Inputs for shell, stringers and frame weight formulas
        self.add_input('lt', val=47.02, units='ft', desc='Tail arm (ft) - used in weight calculations')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft) - used for C.G calculation')
        self.add_input('base', val=10.16, units='ft', desc='Fuselage width (ft)')
        self.add_input('height', val=9.87, units='ft', desc='Fuselage height (ft)')
        self.add_input('Vd', val=315, units='knot', desc='Dive speed (knots)')
        self.add_input('Sg', val=2772.2, units='ft**2', desc='Gross shell area (ft²)')
        self.add_input('delta_p', val=9.25, units='psi', desc='Pressure differential (psi)')
        self.add_input('n_ult', val=1.5, units=None, desc='Ultimate factor')

        # Only keep total weight output as requested
        self.add_output('Wf_total', val=0.0, units='lbm', desc='Total fuselage weight')
        
        # C.G output: FS_start + (fuselage_length * percentage / 100)
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        self.add_output('cg_fuselage', val=0.0, units='inch', desc='Fuselage C.G location (Fuselage Station)')
        
        # FLAG: Complex partials due to max() - declare all
        self.declare_partials('Wf_total', '*')
        self.declare_partials('cg_fuselage', 'fuselage_length')

    def compute(self, inputs, outputs):
        # Input validation
        lt = inputs['lt']
        bf = inputs['base']
        hf = inputs['height']
        Vd = inputs['Vd']
        Sg = inputs['Sg']
        n_ult = inputs['n_ult']
        
        if lt <= 0:
            raise ValueError(f"lt must be positive, got {lt}")
        if bf <= 0 or hf <= 0:
            raise ValueError(f"bf and hf must be positive, got bf={bf}, hf={hf}")
        if Vd <= 0:
            raise ValueError(f"Vd must be positive, got {Vd}")
        if Sg <= 0:
            raise ValueError(f"Sg must be positive, got {Sg}")
        if n_ult <= 0:
            raise ValueError(f"n_ult must be positive, got {n_ult}")
        
        # Get constants from options
        const_D2 = self.options['const_D2']
        const_D6 = self.options['const_D6']

        # Shell weight (using only Wsk1 formula)
        k_lambda = 0.56 * (lt / (bf + hf)) ** 0.75
        Wsk = const_D2 * k_lambda * (Sg ** 1.07) * (Vd ** 0.743)

        # Stringer/Longeron weight
        Wstr = const_D6 * k_lambda * (Sg ** 1.45) * (Vd ** 0.39) * (n_ult ** 0.316)

        # Frame weight
        frame_factor = self.options['frame_factor']
        Wfr = frame_factor * (Wstr + Wsk)

        # Total fuselage weight (regression)
        regression_factor = self.options['regression_factor']
        outputs['Wf_total'] = (Wstr + Wsk + Wfr) * regression_factor
        
        # Calculate C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_fuselage_percentage']
        outputs['cg_fuselage'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Get constants from options
        const_D2 = self.options['const_D2']
        const_D6 = self.options['const_D6']
        regression_factor = self.options['regression_factor']
        frame_factor = self.options['frame_factor']
        
        # Extract inputs
        lt = inputs['lt']
        bf = inputs['base']
        hf = inputs['height']
        Vd = inputs['Vd']
        Sg = inputs['Sg']
        n_ult = inputs['n_ult']
        
        # Intermediate calculations
        bh = bf + hf  # sum of bf and hf
        k_lambda = 0.56 * (lt / bh) ** 0.75
        
        # Compute Wsk (only Wsk1 formula)
        Wsk = const_D2 * k_lambda * (Sg ** 1.07) * (Vd ** 0.743)
        
        # Compute Wstr
        Wstr = const_D6 * k_lambda * (Sg ** 1.45) * (Vd ** 0.39) * (n_ult ** 0.316)
        
        # Total factor: (1 + frame_factor) * regression_factor
        total_factor = (1.0 + frame_factor) * regression_factor
        
        # Partials of k_lambda:
        # k_lambda = 0.56 * (lt/bh)^0.75
        # dk/dlt = 0.75 * k_lambda / lt
        # dk/dbf = dk/dhf = -0.75 * k_lambda / bh
        
        # Partials of Wstr w.r.t. each input
        # Wstr = const_D6 * k_lambda * Sg^1.45 * Vd^0.39 * n_ult^0.316
        dWstr_dlt = 0.75 * Wstr / lt
        dWstr_dbf = -0.75 * Wstr / bh
        dWstr_dhf = -0.75 * Wstr / bh
        dWstr_dVd = 0.39 * Wstr / Vd
        dWstr_dSg = 1.45 * Wstr / Sg
        dWstr_ddelta_p = 0.0
        dWstr_dn_ult = 0.316 * Wstr / n_ult
        
        # Partials of Wsk (Wsk1 formula only)
        # Wsk = const_D2 * k_lambda * Sg^1.07 * Vd^0.743
        dWsk_dlt = 0.75 * Wsk / lt
        dWsk_dbf = -0.75 * Wsk / bh
        dWsk_dhf = -0.75 * Wsk / bh
        dWsk_dVd = 0.743 * Wsk / Vd
        dWsk_dSg = 1.07 * Wsk / Sg
        dWsk_ddelta_p = 0.0
        dWsk_dn_ult = 0.0
        
        # Total partials: Wf_total = (1 + frame_factor) * (Wstr + Wsk) * regression_factor
        partials['Wf_total', 'lt'] = total_factor * (dWstr_dlt + dWsk_dlt)
        partials['Wf_total', 'base'] = total_factor * (dWstr_dbf + dWsk_dbf)
        partials['Wf_total', 'height'] = total_factor * (dWstr_dhf + dWsk_dhf)
        partials['Wf_total', 'Vd'] = total_factor * (dWstr_dVd + dWsk_dVd)
        partials['Wf_total', 'Sg'] = total_factor * (dWstr_dSg + dWsk_dSg)
        partials['Wf_total', 'delta_p'] = total_factor * (dWstr_ddelta_p + dWsk_ddelta_p)
        partials['Wf_total', 'n_ult'] = total_factor * (dWstr_dn_ult + dWsk_dn_ult)
        
        # Partials for C.G: cg = fs_start + (fuselage_length * 12 * percentage/100)
        # d(cg)/d(fuselage_length) = (12 * percentage/100)
        cg_percentage = self.options['cg_fuselage_percentage']
        partials['cg_fuselage', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('lt', val=47.02, units='ft')
    ivc.add_output('fuselage_length', val=97.89, units='ft')
    ivc.add_output('fuse_base', val=10.16, units='ft')
    ivc.add_output('fuse_height', val=9.87, units='ft')
    ivc.add_output('Vd', val=315.0, units='knot')
    ivc.add_output('Sg', val=2728.0, units='ft**2')
    ivc.add_output('delta_p', val=9.25, units='psi')
    ivc.add_output('n_ult', val=1.5, units=None)

    model.add_subsystem('fuselage', FuselageMass(), 
                       promotes_inputs=['lt', 'fuselage_length', 'Vd', 'Sg', 'delta_p', 'n_ult'],
                       promotes_outputs=['*'])
    
    # Connect fuselage-specific inputs
    model.connect('fuse_base', 'fuselage.base')
    model.connect('fuse_height', 'fuselage.height')

    prob.setup()
    prob.run_model()

    print('Total Fuselage Weight:', prob.get_val('Wf_total', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
