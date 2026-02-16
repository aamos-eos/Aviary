import openmdao.api as om
import numpy as np


class WingBendingWeight(om.ExplicitComponent):
    """
    Calculates the weight of wing bending material. The methodology is
    based on the FLOPS weight equations.
    """

    def initialize(self):
        self.options.declare('num_fuselages', default=1, desc='Number of fuselages')
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.A1 = 8.80
        self.A2 = 6.25

    def setup(self):
        self.add_input('gross_weight', val=86000.0, units='lbm', desc='Aircraft gross weight')
        self.add_input('aeroelastic_tailoring_factor', val=0.0, units=None, desc='Aeroelastic tailoring factor')
        self.add_input('bending_material_factor', val=1.0, units=None, desc='Bending material factor')
        self.add_input('bending_material_weight_scaler', val=1.0, units=None, desc='Bending material weight scaler')
        self.add_input('composite_fraction', val=0.5, units=None, desc='Composite fraction')
        self.add_input('inertia_relief_factor', val=1.0, units=None, desc='Inertia relief factor')
        self.add_input('load_fraction', val=1.0, units=None, desc='Load fraction')
        self.add_input('misc_weight', val=300.0, units='lbm', desc='Miscellaneous weight')
        self.add_input('misc_weight_scaler', val=1.0, units=None, desc='Miscellaneous weight scaler')
        self.add_input('shear_control_weight', val=500.0, units='lbm', desc='Shear control weight')
        self.add_input('shear_control_weight_scaler', val=1.0, units=None, desc='Shear control weight scaler')
        self.add_input('span', val=115.825, units='ft', desc='Wing span')
        self.add_input('sweep', val=2.8, units='deg', desc='Wing sweep angle')
        self.add_input('ultimate_load_factor', val=3.75, units=None, desc='Ultimate load factor')
        self.add_input('var_sweep_weight_penalty', val=0.0, units=None, desc='Variable sweep weight penalty')

        self.add_output('bending_material_weight', val=0.0, units='lbm', desc='Bending material weight')
        
        self.declare_partials('bending_material_weight', '*')

    def compute(self, inputs, outputs):
        gross_weight = inputs['gross_weight']
        faert = inputs['aeroelastic_tailoring_factor']
        bt = inputs['bending_material_factor']
        scaler = inputs['bending_material_weight_scaler']
        comp_frac = inputs['composite_fraction']
        CAYE = inputs['inertia_relief_factor']
        pctl = inputs['load_fraction']
        misc_weight = inputs['misc_weight']
        misc_weight_scaler = inputs['misc_weight_scaler']
        shear_control_weight = inputs['shear_control_weight']
        shear_control_weight_scaler = inputs['shear_control_weight_scaler']
        span = inputs['span']
        sweep = inputs['sweep']
        ulf = inputs['ultimate_load_factor']
        varswp = inputs['var_sweep_weight_penalty']

        num_fuse = self.options['num_fuselages']

        W2 = shear_control_weight / shear_control_weight_scaler
        W3 = misc_weight / misc_weight_scaler

        vfact = 1.0 + varswp * (0.96 / np.cos(np.pi / 180.0 * sweep) - 1.0)
        cayf = 0.5 if num_fuse > 1 else 1.0

        W1NIR = (
            self.A1
            * bt
            * (1.0 + (self.A2 / span) ** 0.5)
            * ulf
            * span
            * (1.0 - 0.4 * comp_frac)
            * (1.0 - 0.1 * faert)
            * cayf
            * vfact
            * pctl
            * 1.0e-6
        )
        
        outputs['bending_material_weight'] = (
            ((gross_weight * CAYE * W1NIR + W2 + W3) / (1.0 + W1NIR) - W2 - W3)
            * scaler
        )

    def compute_partials(self, inputs, partials):
        gross_weight = inputs['gross_weight']
        faert = inputs['aeroelastic_tailoring_factor']
        bt = inputs['bending_material_factor']
        scaler = inputs['bending_material_weight_scaler']
        comp_frac = inputs['composite_fraction']
        CAYE = inputs['inertia_relief_factor']
        pctl = inputs['load_fraction']
        misc_weight = inputs['misc_weight']
        misc_weight_scaler = inputs['misc_weight_scaler']
        shear_control_weight = inputs['shear_control_weight']
        shear_control_weight_scaler = inputs['shear_control_weight_scaler']
        span = inputs['span']
        sweep = inputs['sweep']
        ulf = inputs['ultimate_load_factor']
        varswp = inputs['var_sweep_weight_penalty']

        num_fuse = self.options['num_fuselages']

        W2 = shear_control_weight / shear_control_weight_scaler
        W3 = misc_weight / misc_weight_scaler

        cos_sweep = np.cos(np.pi / 180.0 * sweep)
        vfact = 1.0 + varswp * (0.96 / cos_sweep - 1.0)
        cayf = 0.5 if num_fuse > 1 else 1.0

        # Base terms for W1NIR
        span_term = (1.0 + (self.A2 / span) ** 0.5)
        comp_term = (1.0 - 0.4 * comp_frac)
        faert_term = (1.0 - 0.1 * faert)
        
        W1NIR = (
            self.A1 * bt * span_term * ulf * span * comp_term * faert_term * cayf * vfact * pctl * 1.0e-6
        )

        # output = ((GW * CAYE * W1NIR + W2 + W3) / (1 + W1NIR) - W2 - W3) * scaler
        # Let N = GW * CAYE * W1NIR + W2 + W3
        # Let D = 1 + W1NIR
        # output = (N/D - W2 - W3) * scaler
        
        N = gross_weight * CAYE * W1NIR + W2 + W3
        D = 1.0 + W1NIR
        
        # d(output)/d(gross_weight) = scaler * CAYE * W1NIR / D
        partials['bending_material_weight', 'gross_weight'] = scaler * CAYE * W1NIR / D
        
        # d(output)/d(CAYE) = scaler * GW * W1NIR / D
        partials['bending_material_weight', 'inertia_relief_factor'] = scaler * gross_weight * W1NIR / D
        
        # d(output)/d(W1NIR) using quotient rule:
        # d/dW1NIR [(N/D)] = (dN/dW1NIR * D - N * dD/dW1NIR) / D^2
        #                  = (GW * CAYE * D - N * 1) / D^2
        #                  = (GW * CAYE - N/D) / D
        # d(output)/d(W1NIR) = scaler * (GW * CAYE - N/D) / D
        dout_dW1NIR = scaler * (gross_weight * CAYE - N / D) / D
        
        # Now chain rule for each variable that affects W1NIR
        # W1NIR = A1 * bt * span_term * ulf * span * comp_term * faert_term * cayf * vfact * pctl * 1e-6
        
        # d(W1NIR)/d(bt) = W1NIR / bt
        partials['bending_material_weight', 'bending_material_factor'] = dout_dW1NIR * W1NIR / bt
        
        # d(W1NIR)/d(ulf) = W1NIR / ulf
        partials['bending_material_weight', 'ultimate_load_factor'] = dout_dW1NIR * W1NIR / ulf
        
        # d(W1NIR)/d(pctl) = W1NIR / pctl
        partials['bending_material_weight', 'load_fraction'] = dout_dW1NIR * W1NIR / pctl
        
        # d(W1NIR)/d(comp_frac) = W1NIR * (-0.4) / comp_term
        partials['bending_material_weight', 'composite_fraction'] = dout_dW1NIR * W1NIR * (-0.4) / comp_term
        
        # d(W1NIR)/d(faert) = W1NIR * (-0.1) / faert_term
        partials['bending_material_weight', 'aeroelastic_tailoring_factor'] = dout_dW1NIR * W1NIR * (-0.1) / faert_term
        
        # d(W1NIR)/d(span): span appears in span_term and directly as span
        # span_term = 1 + (A2/span)^0.5
        # d(span_term)/d(span) = -0.5 * (A2/span)^0.5 / span = -0.5 * (A2/span)^0.5 / span
        d_span_term = -0.5 * (self.A2 / span) ** 0.5 / span
        # W1NIR = ... * span_term * span * ...
        # d(W1NIR)/d(span) = W1NIR * (d_span_term / span_term + 1/span)
        partials['bending_material_weight', 'span'] = dout_dW1NIR * W1NIR * (d_span_term / span_term + 1.0 / span)
        
        # d(W1NIR)/d(sweep) through vfact
        # vfact = 1 + varswp * (0.96/cos(sweep) - 1)
        # d(vfact)/d(sweep) = varswp * 0.96 * sin(sweep) / cos(sweep)^2 * (pi/180)
        sin_sweep = np.sin(np.pi / 180.0 * sweep)
        d_vfact_dsweep = varswp * 0.96 * sin_sweep / (cos_sweep ** 2) * (np.pi / 180.0)
        partials['bending_material_weight', 'sweep'] = dout_dW1NIR * W1NIR * d_vfact_dsweep / vfact if vfact != 0 else 0.0
        
        # d(W1NIR)/d(varswp)
        # vfact = 1 + varswp * (0.96/cos(sweep) - 1)
        # d(vfact)/d(varswp) = 0.96/cos(sweep) - 1
        d_vfact_dvarswp = 0.96 / cos_sweep - 1.0
        partials['bending_material_weight', 'var_sweep_weight_penalty'] = dout_dW1NIR * W1NIR * d_vfact_dvarswp / vfact if vfact != 0 else 0.0
        
        # d(output)/d(scaler) = output / scaler
        bending_weight = ((gross_weight * CAYE * W1NIR + W2 + W3) / (1.0 + W1NIR) - W2 - W3)
        partials['bending_material_weight', 'bending_material_weight_scaler'] = bending_weight
        
        # d(output)/d(shear_control_weight)
        # W2 = shear_control_weight / shear_control_weight_scaler
        # d(W2)/d(shear_control_weight) = 1 / shear_control_weight_scaler
        # output = scaler * (N/D - W2 - W3)
        # d(output)/d(W2) = scaler * (1/D - 1)
        dW2_dscw = 1.0 / shear_control_weight_scaler
        partials['bending_material_weight', 'shear_control_weight'] = scaler * (1.0 / D - 1.0) * dW2_dscw
        
        # d(output)/d(shear_control_weight_scaler)
        # W2 = shear_control_weight / shear_control_weight_scaler
        # d(W2)/d(scaler) = -shear_control_weight / shear_control_weight_scaler^2
        dW2_dscws = -shear_control_weight / (shear_control_weight_scaler ** 2)
        partials['bending_material_weight', 'shear_control_weight_scaler'] = scaler * (1.0 / D - 1.0) * dW2_dscws
        
        # d(output)/d(misc_weight)
        dW3_dmw = 1.0 / misc_weight_scaler
        partials['bending_material_weight', 'misc_weight'] = scaler * (1.0 / D - 1.0) * dW3_dmw
        
        # d(output)/d(misc_weight_scaler)
        dW3_dmws = -misc_weight / (misc_weight_scaler ** 2)
        partials['bending_material_weight', 'misc_weight_scaler'] = scaler * (1.0 / D - 1.0) * dW3_dmws


class WingShearControlWeight(om.ExplicitComponent):
    """
    Calculates the weight of wing shear control material. The methodology is
    based on the FLOPS weight equations.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.A3 = 0.68
        self.A4 = 0.34
        self.A5 = 0.60

    def setup(self):
        self.add_input('composite_fraction', val=0.5, units=None, desc='Composite fraction')
        self.add_input('control_surface_area', val=328.4, units='ft**2', desc='Control surface area')
        self.add_input('gross_weight', val=86000.0, units='lbm', desc='Aircraft gross weight')
        self.add_input('shear_control_weight_scaler', val=1.0, units=None, desc='Shear control weight scaler')

        self.add_output('shear_control_weight', val=0.0, units='lbm', desc='Shear control weight')
        
        self.declare_partials('shear_control_weight', '*')

    def compute(self, inputs, outputs):
        comp_frac = inputs['composite_fraction']
        ctrl_area = inputs['control_surface_area']
        gross_weight = inputs['gross_weight']
        scaler = inputs['shear_control_weight_scaler']

        outputs['shear_control_weight'] = (
            self.A3
            * (1.0 - 0.17 * comp_frac)
            * ctrl_area ** self.A4
            * gross_weight ** self.A5
            * scaler
        )

    def compute_partials(self, inputs, partials):
        comp_frac = inputs['composite_fraction']
        ctrl_area = inputs['control_surface_area']
        gross_weight = inputs['gross_weight']
        scaler = inputs['shear_control_weight_scaler']
        
        # W = A3 * (1 - 0.17*cf) * S^A4 * GW^A5 * scaler
        W = self.A3 * (1.0 - 0.17 * comp_frac) * ctrl_area ** self.A4 * gross_weight ** self.A5 * scaler
        
        # d(W)/d(comp_frac) = W * (-0.17) / (1 - 0.17*cf)
        partials['shear_control_weight', 'composite_fraction'] = W * (-0.17) / (1.0 - 0.17 * comp_frac)
        
        # d(W)/d(ctrl_area) = W * A4 / ctrl_area
        partials['shear_control_weight', 'control_surface_area'] = W * self.A4 / ctrl_area
        
        # d(W)/d(gross_weight) = W * A5 / gross_weight
        partials['shear_control_weight', 'gross_weight'] = W * self.A5 / gross_weight
        
        # d(W)/d(scaler) = W / scaler
        partials['shear_control_weight', 'shear_control_weight_scaler'] = W / scaler


class WingMiscWeight(om.ExplicitComponent):
    """
    Calculates the weight of wing miscellaneous material. The methodology is
    based on the FLOPS weight equations.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.A6 = 0.035
        self.A7 = 1.50

    def setup(self):
        self.add_input('composite_fraction', val=0.5, units=None, desc='Composite fraction')
        self.add_input('wing_area', val=912.91, units='ft**2', desc='Wing area')
        self.add_input('misc_weight_scaler', val=1.0, units=None, desc='Miscellaneous weight scaler')

        self.add_output('misc_weight', val=0.0, units='lbm', desc='Miscellaneous weight')
        
        self.declare_partials('misc_weight', '*')

    def compute(self, inputs, outputs):
        comp_frac = inputs['composite_fraction']
        area = inputs['wing_area']
        scaler = inputs['misc_weight_scaler']

        outputs['misc_weight'] = (
            self.A6 * (1.0 - 0.3 * comp_frac) * area ** self.A7 * scaler
        )

    def compute_partials(self, inputs, partials):
        comp_frac = inputs['composite_fraction']
        area = inputs['wing_area']
        scaler = inputs['misc_weight_scaler']
        
        # W = A6 * (1 - 0.3*cf) * S^A7 * scaler
        W = self.A6 * (1.0 - 0.3 * comp_frac) * area ** self.A7 * scaler
        
        # d(W)/d(comp_frac) = W * (-0.3) / (1 - 0.3*cf)
        partials['misc_weight', 'composite_fraction'] = W * (-0.3) / (1.0 - 0.3 * comp_frac)
        
        # d(W)/d(area) = W * A7 / area
        partials['misc_weight', 'wing_area'] = W * self.A7 / area
        
        # d(W)/d(scaler) = W / scaler
        partials['misc_weight', 'misc_weight_scaler'] = W / scaler


class WingTotalWeight(om.ExplicitComponent):
    """Computation of wing weight using FLOPS-based detailed wing weight equations."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    def setup(self):
        self.add_input('bending_material_weight', val=1000.0, units='lbm', desc='Bending material weight')
        self.add_input('shear_control_weight', val=500.0, units='lbm', desc='Shear control weight')
        self.add_input('misc_weight', val=300.0, units='lbm', desc='Miscellaneous weight')
        self.add_input('weight_scaler', val=1.0, units=None, desc='Weight scaler')

        self.add_output('wing_weight', val=0.0, units='lbm', desc='Total wing weight')
        
        self.declare_partials('wing_weight', '*')

    def compute(self, inputs, outputs):
        w1 = inputs['bending_material_weight']
        w2 = inputs['shear_control_weight']
        w3 = inputs['misc_weight']
        w_scaler = inputs['weight_scaler']

        outputs['wing_weight'] = (w1 + w2 + w3) * w_scaler

    def compute_partials(self, inputs, partials):
        w1 = inputs['bending_material_weight']
        w2 = inputs['shear_control_weight']
        w3 = inputs['misc_weight']
        w_scaler = inputs['weight_scaler']
        
        # d(W)/d(w1) = scaler
        partials['wing_weight', 'bending_material_weight'] = w_scaler
        
        # d(W)/d(w2) = scaler
        partials['wing_weight', 'shear_control_weight'] = w_scaler
        
        # d(W)/d(w3) = scaler
        partials['wing_weight', 'misc_weight'] = w_scaler
        
        # d(W)/d(scaler) = w1 + w2 + w3
        partials['wing_weight', 'weight_scaler'] = w1 + w2 + w3


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # WingShearControlWeight inputs
    ivc.add_output('composite_fraction', val=0.5, units=None)
    ivc.add_output('control_surface_area', val=328.4, units='ft**2')
    ivc.add_output('gross_weight', val=86000.0, units='lbm')
    ivc.add_output('shear_control_weight_scaler', val=1.0, units=None)
    # WingMiscWeight inputs
    ivc.add_output('wing_area', val=912.91, units='ft**2')
    ivc.add_output('misc_weight_scaler', val=1.0, units=None)
    # WingBendingWeight inputs
    ivc.add_output('aeroelastic_tailoring_factor', val=0.0, units=None)
    ivc.add_output('bending_material_factor', val=1.0, units=None)
    ivc.add_output('bending_material_weight_scaler', val=1.0, units=None)
    ivc.add_output('inertia_relief_factor', val=1.0, units=None)
    ivc.add_output('load_fraction', val=1.0, units=None)
    ivc.add_output('span', val=115.825, units='ft')
    ivc.add_output('sweep', val=2.8, units='deg')
    ivc.add_output('ultimate_load_factor', val=3.75, units=None)
    ivc.add_output('var_sweep_weight_penalty', val=0.0, units=None)
    # WingTotalWeight inputs
    ivc.add_output('weight_scaler', val=1.0, units=None)

    model.add_subsystem('shear_control', WingShearControlWeight(), promotes=['*'])
    model.add_subsystem('misc', WingMiscWeight(), promotes=['*'])
    model.add_subsystem('bending', WingBendingWeight(), promotes=['*'])
    model.add_subsystem('total', WingTotalWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Shear Control Weight:', prob.get_val('shear_control_weight', units='lbm'))
    print('Misc Weight:', prob.get_val('misc_weight', units='lbm'))
    print('Bending Material Weight:', prob.get_val('bending_material_weight', units='lbm'))
    print('Total Wing Weight:', prob.get_val('wing_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
