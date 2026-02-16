import openmdao.api as om
import numpy as np

class WingWeight(om.ExplicitComponent):
    """
    Calculates total wing weight using FLOPS-based detailed wing weight equations.
    Merges SimpleWingBendingFact, WingBendingWeight, WingShearControlWeight, 
    WingMiscWeight, and WingTotalWeight into a single component.
    """

    def initialize(self):
        self.options.declare('num_fuselages', default=1, desc='Number of fuselages')
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Constants from original components
        self.A1 = 8.80
        self.A2 = 6.25
        self.A3 = 0.68
        self.A4 = 0.34
        self.A5 = 0.60
        self.A6 = 0.035
        self.A7 = 1.50

    def setup(self):
        # Inputs from SimpleWingBendingFact
        # Use planform (ref) values for weight calculations
        self.add_input('S_ref', val=927.85, units='ft**2', desc='Wing reference area (planform, includes winglets)')
        self.add_input('span_plan', val=115.825, units='ft', desc='Wing span (planform, includes winglets)')
        self.add_input('taper_ratio', val=0.356, units=None, desc='Wing taper ratio')
        self.add_input('toverc', val=0.155, units=None, desc='Thickness-to-chord ratio')
        self.add_input('strut_bracing_factor', val=0.0, units=None, desc='Strut bracing factor')
        self.add_input('aeroelastic_tailoring_factor', val=0.0, units=None, desc='Aeroelastic tailoring factor')
        self.add_input('aspect_ratio', val=14.46, units=None, desc='Wing aspect ratio')
        self.add_input('c4_sweep', val=2.8, units='deg', desc='Wing sweep angle')
        
        # Additional inputs for CAYE calculation (new formula)
        # full_nacelle_weight = nacelle structure + propulsion + batteries
        # (propulsion includes: Electric motors + Turbine + Gearbox + Propellers)
        self.add_input('full_nacelle_weight', val=0.0, units='lbm', desc='Full nacelle weight (structure + propulsion + batteries)')
        
        # Additional inputs from WingBendingWeight
        self.add_input('gross_weight', val=86000.0, units='lbm', desc='Aircraft gross weight')
        self.add_input('bending_material_weight_scaler', val=1.0, units=None, desc='Bending material weight scaler')
        self.add_input('composite_fraction', val=0.5, units=None, desc='Composite fraction')
        self.add_input('load_fraction', val=1.0, units=None, desc='Load fraction')
        self.add_input('misc_weight_scaler', val=1.0, units=None, desc='Miscellaneous weight scaler')
        self.add_input('shear_control_weight_scaler', val=1.0, units=None, desc='Shear control weight scaler')
        self.add_input('n_ult', val=3.75, units=None, desc='Ultimate load factor')
        self.add_input('var_sweep_weight_penalty', val=0.0, units=None, desc='Variable sweep weight penalty')
        
        # Inputs from WingShearControlWeight
        self.add_input('S_ref_control_ratio', val=0.35393652, units=None, desc='Control surface area ratio')
        
        # Inputs from WingTotalWeight
        self.add_input('weight_scaler', val=1.0, units=None, desc='Weight scaler')
        
        # Inputs for C.G calculation
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        # Output: total wing weight only
        self.add_output('wing_weight', val=0.0, units='lbm', desc='Total wing weight')
        self.add_output('cg_wing', val=0.0, units='inch', desc='Wing C.G location')
        
        # Declare partials
        self.declare_partials('wing_weight', '*')
        self.declare_partials('cg_wing', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Step 1: Compute bending material factor and inertia relief factor (from SimpleWingBendingFact)
        fstrt = inputs['strut_bracing_factor']
        tr = inputs['taper_ratio']
        area = inputs['S_ref']
        tca = inputs['toverc']
        faert = inputs['aeroelastic_tailoring_factor']
        ar = inputs['aspect_ratio']
        span = inputs['span_plan']
        c4_sweep = inputs['c4_sweep']
        ctrl_area_ratio = inputs['S_ref_control_ratio']

        c4 = 1.0 - 0.5 * faert
        c6 = 0.5 * faert - 0.16 * fstrt

        # caya = ar - 5.0 (ar is always > 5.0)
        caya = ar - 5.0

        tlam = np.tan(np.pi / 180.0 * c4_sweep) - 2 * (1 - tr) / (ar * (1 + tr))
        slam = tlam / (1.0 + tlam**2) ** 0.5
        cayl = (1.0 - slam**2) * (1.0 + c6 * slam**2 + 0.03 * caya * c4 * slam)
        ems = 1.0 - 0.25 * fstrt

        bending_material_factor = (
            0.215 * (0.37 + 0.7 * tr) * (span**2 / area) ** ems / (cayl * tca)
        )

        # CAYE factor calculation using new formula:
        # CAYE = 1.3 * (NacelleW/DG)^2 - 1.26 * (NacelleW/DG) + 1
        # Where NacelleW = full_nacelle_weight (nacelle structure + propulsion + batteries)
        nacelle_weight = inputs['full_nacelle_weight']
        
        # Get gross_weight from inputs for CAYE calculation
        gross_weight = inputs['gross_weight']
        design_gross_weight = gross_weight
        nacelle_ratio = nacelle_weight / design_gross_weight if design_gross_weight > 0 else 0.0
        
        # CAYE = 1.3 * (NacelleW/DG)^2 - 1.26 * (NacelleW/DG) + 1
        CAYE = 1.3 * (nacelle_ratio ** 2) - 1.26 * nacelle_ratio + 1.0
        
        # Keep inertia_relief_factor for backward compatibility (same as CAYE now)
        inertia_relief_factor = CAYE

        # Step 2: Compute shear control weight (from WingShearControlWeight)
        comp_frac = inputs['composite_fraction']
        ctrl_area = area * ctrl_area_ratio
        shear_scaler = inputs['shear_control_weight_scaler']

        shear_control_weight = (
            self.A3
            * (1.0 - 0.17 * comp_frac)
            * ctrl_area**self.A4
            * gross_weight**self.A5
            * shear_scaler
        )

        # Step 3: Compute misc weight (from WingMiscWeight)
        misc_scaler = inputs['misc_weight_scaler']
        misc_weight = (
            self.A6 * (1.0 - 0.3 * comp_frac) * area**self.A7 * misc_scaler
        )

        # Step 4: Compute bending material weight (from WingBendingWeight)
        bt = bending_material_factor
        ulf = inputs['n_ult']
        CAYE = inertia_relief_factor
        scaler = inputs['bending_material_weight_scaler']
        pctl = inputs['load_fraction']
        varswp = inputs['var_sweep_weight_penalty']
        num_fuse = self.options['num_fuselages']

        W2 = shear_control_weight / shear_scaler
        W3 = misc_weight / misc_scaler

        vfact = 1.0 + varswp * (0.96 / np.cos(np.pi / 180.0 * c4_sweep) - 1.0)
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
        
        bending_material_weight = (
            ((gross_weight * CAYE * W1NIR + W2 + W3) / (1.0 + W1NIR) - W2 - W3)
            * scaler
        )

        # Step 5: Compute total wing weight (from WingTotalWeight)
        w_scaler = inputs['weight_scaler']
        outputs['wing_weight'] = (bending_material_weight + shear_control_weight + misc_weight) * w_scaler
        
        # Calculate C.G location
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_percentage = 45.0  # 45% from LEMAC
        outputs['cg_wing'] = LEMAC + (MAC * cg_percentage / 100.0)


    def compute_partials(self, inputs, partials):
        # Get all inputs
        fstrt = inputs['strut_bracing_factor']
        tr = inputs['taper_ratio']
        area = inputs['S_ref']
        tca = inputs['toverc']
        faert = inputs['aeroelastic_tailoring_factor']
        ar = inputs['aspect_ratio']
        span = inputs['span_plan']
        c4_sweep = inputs['c4_sweep']
        comp_frac = inputs['composite_fraction']
        ctrl_area_ratio = inputs['S_ref_control_ratio']
        ctrl_area = area * ctrl_area_ratio  # Compute control surface area from ratio
        gross_weight = inputs['gross_weight']
        shear_scaler = inputs['shear_control_weight_scaler']
        misc_scaler = inputs['misc_weight_scaler']
        ulf = inputs['n_ult']
        scaler = inputs['bending_material_weight_scaler']
        pctl = inputs['load_fraction']
        varswp = inputs['var_sweep_weight_penalty']
        w_scaler = inputs['weight_scaler']
        
        num_fuse = self.options['num_fuselages']
        c4_sweep_rad = np.pi / 180.0 * c4_sweep
        
        # ===== Recompute intermediates =====
        c4 = 1.0 - 0.5 * faert
        c6 = 0.5 * faert - 0.16 * fstrt
        caya = ar - 5.0
        
        tlam = np.tan(c4_sweep_rad) - 2.0 * (1.0 - tr) / (ar * (1.0 + tr))
        slam = tlam / np.sqrt(1.0 + tlam**2)
        cayl = (1.0 - slam**2) * (1.0 + c6 * slam**2 + 0.03 * caya * c4 * slam)
        ems = 1.0 - 0.25 * fstrt
        
        bt = 0.215 * (0.37 + 0.7 * tr) * (span**2 / area)**ems / (cayl * tca)
        
        # CAYE factor calculation using new formula:
        # CAYE = 1.3 * (NacelleW/DG)^2 - 1.26 * (NacelleW/DG) + 1
        # Where NacelleW = full_nacelle_weight (nacelle structure + propulsion + batteries)
        nacelle_weight = inputs['full_nacelle_weight']
        design_gross_weight = gross_weight
        nacelle_ratio = nacelle_weight / design_gross_weight if design_gross_weight > 0 else 0.0
        CAYE = 1.3 * (nacelle_ratio ** 2) - 1.26 * nacelle_ratio + 1.0
        
        W2 = self.A3 * (1.0 - 0.17 * comp_frac) * ctrl_area**self.A4 * gross_weight**self.A5
        W3 = self.A6 * (1.0 - 0.3 * comp_frac) * area**self.A7
        
        shear_control_weight = W2 * shear_scaler
        misc_weight = W3 * misc_scaler
        
        vfact = 1.0 + varswp * (0.96 / np.cos(c4_sweep_rad) - 1.0)
        cayf = 0.5 if num_fuse > 1 else 1.0
        
        W1NIR = (self.A1 * bt * (1.0 + (self.A2 / span)**0.5) * ulf * span 
                 * (1.0 - 0.4 * comp_frac) * (1.0 - 0.1 * faert) * cayf * vfact * pctl * 1.0e-6)
        
        bending_material_weight = ((gross_weight * CAYE * W1NIR + W2 + W3) / (1.0 + W1NIR) - W2 - W3) * scaler
        
        # ===== Partials for shear_control_weight =====
        dW2_dcomp_frac = self.A3 * (-0.17) * ctrl_area**self.A4 * gross_weight**self.A5
        dW2_dctrl_area = self.A3 * (1.0 - 0.17 * comp_frac) * self.A4 * ctrl_area**(self.A4 - 1) * gross_weight**self.A5
        dW2_dgross = self.A3 * (1.0 - 0.17 * comp_frac) * ctrl_area**self.A4 * self.A5 * gross_weight**(self.A5 - 1)
        
        dshear_dcomp_frac = dW2_dcomp_frac * shear_scaler
        dshear_dctrl_area = dW2_dctrl_area * shear_scaler
        dshear_dgross = dW2_dgross * shear_scaler
        dshear_dshear_scaler = W2
        
        # ===== Partials for misc_weight =====
        dW3_dcomp_frac = self.A6 * (-0.3) * area**self.A7
        dW3_darea = self.A6 * (1.0 - 0.3 * comp_frac) * self.A7 * area**(self.A7 - 1)
        
        dmisc_dcomp_frac = dW3_dcomp_frac * misc_scaler
        dmisc_darea = dW3_darea * misc_scaler
        dmisc_dmisc_scaler = W3
        
        # ===== Partials for bending_material_factor (bt) =====
        # bt = 0.215 * (0.37 + 0.7*tr) * (span^2/area)^ems / (cayl * tca)
        span2_over_area = span**2 / area
        base_bt = 0.215 * (0.37 + 0.7 * tr)
        
        # d(bt)/d(tr) 
        dbt_dtr_direct = 0.215 * 0.7 * span2_over_area**ems / (cayl * tca)
        
        # d(bt)/d(tca)
        dbt_dtca = -bt / tca
        
        # d(bt)/d(span): bt contains span^(2*ems) / area^ems
        dbt_dspan = bt * 2.0 * ems / span
        
        # d(bt)/d(area): bt contains span^(2*ems) / area^ems
        dbt_darea = -bt * ems / area
        
        # d(bt)/d(fstrt): affects ems and c6/cayl
        dems_dfstrt = -0.25
        dc6_dfstrt = -0.16
        # For ems effect: d/d(fstrt) of (span2/area)^ems = (span2/area)^ems * ln(span2/area) * d(ems)/d(fstrt)
        dbt_dfstrt_ems = bt * np.log(span2_over_area) * dems_dfstrt
        
        # For c6/cayl effect - need d(cayl)/d(c6)
        dcayl_dc6 = (1.0 - slam**2) * slam**2
        dbt_dfstrt_c6 = -bt / cayl * dcayl_dc6 * dc6_dfstrt
        dbt_dfstrt = dbt_dfstrt_ems + dbt_dfstrt_c6
        
        # d(bt)/d(faert): affects c4 and c6
        dc4_dfaert = -0.5
        dc6_dfaert = 0.5
        dcayl_dc4 = (1.0 - slam**2) * 0.03 * caya * slam
        dcayl_dfaert = dcayl_dc4 * dc4_dfaert + dcayl_dc6 * dc6_dfaert
        dbt_dfaert = -bt / cayl * dcayl_dfaert
        
        # d(bt)/d(ar): affects caya and tlam/slam/cayl
        dcaya_dar = 1.0
        dcayl_dcaya = (1.0 - slam**2) * 0.03 * c4 * slam
        
        # d(tlam)/d(ar)
        dtlam_dar = 2.0 * (1.0 - tr) / (ar**2 * (1.0 + tr))
        # d(slam)/d(tlam)
        dslam_dtlam = 1.0 / (1.0 + tlam**2)**1.5
        # d(cayl)/d(slam)
        dcayl_dslam = (-2.0 * slam * (1.0 + c6 * slam**2 + 0.03 * caya * c4 * slam) 
                       + (1.0 - slam**2) * (2.0 * c6 * slam + 0.03 * caya * c4))
        
        dbt_dar = -bt / cayl * (dcayl_dcaya * dcaya_dar + dcayl_dslam * dslam_dtlam * dtlam_dar)
        
        # d(bt)/d(sweep): affects tlam
        dtlam_dsweep = (1.0 / np.cos(c4_sweep_rad)**2) * (np.pi / 180.0)
        dbt_dsweep = -bt / cayl * dcayl_dslam * dslam_dtlam * dtlam_dsweep
        
        # d(bt)/d(tr): also affects tlam
        dtlam_dtr = 2.0 / (ar * (1.0 + tr)**2)
        dbt_dtr_tlam = -bt / cayl * dcayl_dslam * dslam_dtlam * dtlam_dtr
        dbt_dtr = dbt_dtr_direct + dbt_dtr_tlam
        
        # ===== Partials for CAYE factor =====
        # CAYE = 1.3 * (NacelleW/DG)^2 - 1.26 * (NacelleW/DG) + 1
        # d(CAYE)/d(NacelleW) = 2 * 1.3 * (NacelleW/DG) * (1/DG) - 1.26 * (1/DG)
        #                     = (2.6 * NacelleW/DG - 1.26) / DG
        #                     = (2.6 * ratio - 1.26) / DG
        nacelle_weight = inputs['full_nacelle_weight']
        
        design_gross_weight = gross_weight
        nacelle_ratio = nacelle_weight / design_gross_weight if design_gross_weight > 0 else 0.0
        
        # d(CAYE)/d(NacelleW) = (2.6 * ratio - 1.26) / DG
        dCAYE_dNacelleW = (2.6 * nacelle_ratio - 1.26) / design_gross_weight if design_gross_weight > 0 else 0.0
        
        # d(CAYE)/d(DG) = d/d(DG) [1.3 * (NacelleW/DG)^2 - 1.26 * (NacelleW/DG) + 1]
        #                = 1.3 * 2 * (NacelleW/DG) * (-NacelleW/DG^2) - 1.26 * (-NacelleW/DG^2)
        #                = -2.6 * NacelleW^2 / DG^3 + 1.26 * NacelleW / DG^2
        #                = NacelleW / DG^2 * (1.26 - 2.6 * NacelleW / DG)
        #                = NacelleW / DG^2 * (1.26 - 2.6 * ratio)
        dCAYE_dDG = (nacelle_weight / (design_gross_weight ** 2)) * (1.26 - 2.6 * nacelle_ratio) if design_gross_weight > 0 else 0.0
        
        # Partials w.r.t. full_nacelle_weight
        dCAYE_dfull_nacelle = dCAYE_dNacelleW  # d(NacelleW)/d(full_nacelle_weight) = 1
        
        # ===== Partials for W1NIR =====
        # W1NIR = A1 * bt * (1 + (A2/span)^0.5) * ulf * span * (1-0.4*cf) * (1-0.1*faert) * cayf * vfact * pctl * 1e-6
        base_W1NIR = W1NIR / bt if bt != 0 else 0.0
        
        sqrt_term = (1.0 + (self.A2 / span)**0.5)
        dsqrt_dspan = -0.5 * (self.A2 / span)**0.5 / span
        
        dW1NIR_dbt = W1NIR / bt if bt != 0 else 0.0
        dW1NIR_dspan = (self.A1 * bt * dsqrt_dspan * ulf * span * (1.0 - 0.4 * comp_frac) 
                        * (1.0 - 0.1 * faert) * cayf * vfact * pctl * 1.0e-6
                        + self.A1 * bt * sqrt_term * ulf * (1.0 - 0.4 * comp_frac) 
                        * (1.0 - 0.1 * faert) * cayf * vfact * pctl * 1.0e-6)
        dW1NIR_dulf = W1NIR / ulf if ulf != 0 else 0.0
        dW1NIR_dcomp_frac = (self.A1 * bt * sqrt_term * ulf * span * (-0.4) 
                             * (1.0 - 0.1 * faert) * cayf * vfact * pctl * 1.0e-6)
        dW1NIR_dfaert = (self.A1 * bt * sqrt_term * ulf * span * (1.0 - 0.4 * comp_frac) 
                         * (-0.1) * cayf * vfact * pctl * 1.0e-6)
        dW1NIR_dpctl = W1NIR / pctl if pctl != 0 else 0.0
        
        # d(vfact)/d(varswp)
        dvfact_dvarswp = 0.96 / np.cos(c4_sweep_rad) - 1.0
        dW1NIR_dvarswp = (self.A1 * bt * sqrt_term * ulf * span * (1.0 - 0.4 * comp_frac) 
                          * (1.0 - 0.1 * faert) * cayf * dvfact_dvarswp * pctl * 1.0e-6)
        
        # d(vfact)/d(sweep)
        dvfact_dsweep = varswp * 0.96 * np.sin(c4_sweep_rad) / np.cos(c4_sweep_rad)**2 * (np.pi / 180.0)
        dW1NIR_dsweep_vfact = (self.A1 * bt * sqrt_term * ulf * span * (1.0 - 0.4 * comp_frac) 
                               * (1.0 - 0.1 * faert) * cayf * dvfact_dsweep * pctl * 1.0e-6)
        
        # ===== Partials for bending_material_weight =====
        # bmw = ((gw * CAYE * W1NIR + W2 + W3) / (1 + W1NIR) - W2 - W3) * scaler
        denom = 1.0 + W1NIR
        numer = gross_weight * CAYE * W1NIR + W2 + W3
        
        # d(bmw)/d(scaler)
        bmw_unscaled = (numer / denom - W2 - W3)
        dbmw_dscaler = bmw_unscaled
        
        # d(bmw)/d(gross_weight)
        dbmw_dgross = CAYE * W1NIR / denom * scaler
        
        # d(bmw)/d(CAYE)
        dbmw_dCAYE = gross_weight * W1NIR / denom * scaler
        
        # d(bmw)/d(W1NIR)
        dbmw_dW1NIR = (gross_weight * CAYE * denom - numer) / denom**2 * scaler
        
        # d(bmw)/d(W2)
        dbmw_dW2 = (1.0 / denom - 1.0) * scaler
        
        # d(bmw)/d(W3)
        dbmw_dW3 = (1.0 / denom - 1.0) * scaler
        
        # ===== Total wing weight partials =====
        # wing_weight = (bmw + scw + mw) * w_scaler
        
        # d(wing_weight)/d(w_scaler)
        partials['wing_weight', 'weight_scaler'] = bending_material_weight + shear_control_weight + misc_weight
        
        # d(wing_weight)/d(bending_material_weight_scaler)
        partials['wing_weight', 'bending_material_weight_scaler'] = bmw_unscaled * w_scaler
        
        # d(wing_weight)/d(shear_control_weight_scaler)
        # scw = W2 * shear_scaler, and W2 also appears in bmw through d(bmw)/d(W2)
        # d(W2)/d(shear_scaler) = 0 (W2 doesn't depend on shear_scaler)
        partials['wing_weight', 'shear_control_weight_scaler'] = W2 * w_scaler
        
        # d(wing_weight)/d(misc_weight_scaler)
        partials['wing_weight', 'misc_weight_scaler'] = W3 * w_scaler
        
        # d(wing_weight)/d(gross_weight) - Note: gross_weight affects CAYE through DG term
        # This will be updated below after CAYE partials are calculated
        
        # d(wing_weight)/d(composite_fraction)
        dW1NIR_total_cf = dW1NIR_dcomp_frac
        partials['wing_weight', 'composite_fraction'] = (
            (dbmw_dW1NIR * dW1NIR_total_cf + dbmw_dW2 * dW2_dcomp_frac + dbmw_dW3 * dW3_dcomp_frac)
            + dshear_dcomp_frac + dmisc_dcomp_frac
        ) * w_scaler
        
        # d(wing_weight)/d(S_ref_control_ratio)
        # ctrl_area = area * ctrl_area_ratio, so d(ctrl_area)/d(ctrl_area_ratio) = area
        dctrl_area_dratio = area
        partials['wing_weight', 'S_ref_control_ratio'] = (dbmw_dW2 * dW2_dctrl_area + dshear_dctrl_area) * dctrl_area_dratio * w_scaler
        
        # d(wing_weight)/d(S_ref)
        # S_ref affects: bt (bending factor), misc_weight (W3), and ctrl_area (through ctrl_area = area * ratio)
        dW1NIR_darea = dW1NIR_dbt * dbt_darea
        dctrl_area_darea = ctrl_area_ratio  # d(area * ratio)/d(area) = ratio
        partials['wing_weight', 'S_ref'] = (
            dbmw_dW1NIR * dW1NIR_darea 
            + dbmw_dW3 * dW3_darea 
            + dmisc_darea
            + (dbmw_dW2 * dW2_dctrl_area + dshear_dctrl_area) * dctrl_area_darea
        ) * w_scaler
        
        # d(wing_weight)/d(span_plan)
        dW1NIR_total_span = dW1NIR_dbt * dbt_dspan + dW1NIR_dspan
        partials['wing_weight', 'span_plan'] = dbmw_dW1NIR * dW1NIR_total_span * w_scaler
        
        # d(wing_weight)/d(taper_ratio)
        dW1NIR_dtr = dW1NIR_dbt * dbt_dtr
        partials['wing_weight', 'taper_ratio'] = dbmw_dW1NIR * dW1NIR_dtr * w_scaler
        
        # d(wing_weight)/d(thickness_to_chord)
        dW1NIR_dtca = dW1NIR_dbt * dbt_dtca
        partials['wing_weight', 'toverc'] = dbmw_dW1NIR * dW1NIR_dtca * w_scaler
        
        # d(wing_weight)/d(aspect_ratio)
        dW1NIR_dar = dW1NIR_dbt * dbt_dar
        partials['wing_weight', 'aspect_ratio'] = dbmw_dW1NIR * dW1NIR_dar * w_scaler
        
        # d(wing_weight)/d(sweep)
        dW1NIR_total_sweep = dW1NIR_dbt * dbt_dsweep + dW1NIR_dsweep_vfact
        partials['wing_weight', 'c4_sweep'] = dbmw_dW1NIR * dW1NIR_total_sweep * w_scaler
        
        # d(wing_weight)/d(strut_bracing_factor)
        dW1NIR_dfstrt = dW1NIR_dbt * dbt_dfstrt
        partials['wing_weight', 'strut_bracing_factor'] = dbmw_dW1NIR * dW1NIR_dfstrt * w_scaler
        
        # d(wing_weight)/d(aeroelastic_tailoring_factor)
        dW1NIR_total_faert = dW1NIR_dbt * dbt_dfaert + dW1NIR_dfaert
        partials['wing_weight', 'aeroelastic_tailoring_factor'] = dbmw_dW1NIR * dW1NIR_total_faert * w_scaler
        
        # d(wing_weight)/d(ultimate_load_factor)
        partials['wing_weight', 'n_ult'] = dbmw_dW1NIR * dW1NIR_dulf * w_scaler
        
        # d(wing_weight)/d(load_fraction)
        partials['wing_weight', 'load_fraction'] = dbmw_dW1NIR * dW1NIR_dpctl * w_scaler
        
        # d(wing_weight)/d(var_sweep_weight_penalty)
        partials['wing_weight', 'var_sweep_weight_penalty'] = dbmw_dW1NIR * dW1NIR_dvarswp * w_scaler
        
        # d(wing_weight)/d(full_nacelle_weight)
        partials['wing_weight', 'full_nacelle_weight'] = dbmw_dCAYE * dCAYE_dfull_nacelle * w_scaler
        
        # d(wing_weight)/d(gross_weight) - includes dCAYE/dDG term
        partials['wing_weight', 'gross_weight'] = (dbmw_dgross + dbmw_dCAYE * dCAYE_dDG + dbmw_dW2 * dW2_dgross + dshear_dgross) * w_scaler
        
        # Partials for C.G
        cg_percentage = 45.0
        partials['cg_wing', 'MAC'] = cg_percentage / 100.0
        partials['cg_wing', 'LEMAC'] = 1.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # SimpleWingBendingFact inputs
    ivc.add_output('S_ref', val=912.91, units='ft**2')
    ivc.add_output('span_plan', val=115.825, units='ft')
    ivc.add_output('taper_ratio', val=0.35, units=None)
    ivc.add_output('toverc', val=0.155, units=None)
    ivc.add_output('strut_bracing_factor', val=0.0, units=None)
    ivc.add_output('aeroelastic_tailoring_factor', val=0.0, units=None)
    ivc.add_output('aspect_ratio', val=14.46, units=None)
    ivc.add_output('c4_sweep', val=2.8, units='deg')
    ivc.add_output('full_nacelle_weight', val=27300.0, units='lbm')  # Example: nacelle + propulsion + batteries
    # WingBendingWeight additional inputs
    ivc.add_output('gross_weight', val=86000.0, units='lbm')
    ivc.add_output('bending_material_weight_scaler', val=1.0, units=None)
    ivc.add_output('composite_fraction', val=0.5, units=None)
    ivc.add_output('load_fraction', val=1.0, units=None)
    ivc.add_output('misc_weight_scaler', val=1.0, units=None)
    ivc.add_output('shear_control_weight_scaler', val=1.0, units=None)
    ivc.add_output('n_ult', val=3.75, units=None)
    ivc.add_output('var_sweep_weight_penalty', val=0.0, units=None)
    # WingShearControlWeight inputs
    ivc.add_output('S_ref_control_ratio', val=0.35393652, units=None)  # Control surface area ratio
    # WingTotalWeight inputs
    ivc.add_output('weight_scaler', val=1.0, units=None)

    model.add_subsystem('wing', WingWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Total Wing Weight:', prob.get_val('wing_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)