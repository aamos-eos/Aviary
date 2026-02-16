import openmdao.api as om
import numpy as np


class FlightControlsWeight(om.ExplicitComponent):
    """
    Calculates flight controls weight:
      W_fc = Mach^0.52 * S_controls^0.6 * Gross_weight^0.32

    where S_controls = control_surface_ratio * wing_surface (computed internally)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        # Default values that can be overridden when instantiating the component
        self.options.declare('base_coefficient', default=1.1, types=float,
                           desc='Base flight controls weight coefficient')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('weight_fuselage_percentage', default=31.5, types=float,
                           desc='Percentage of flight controls weight in fuselage')
        self.options.declare('cg_flight_controls_fuselage_percentage', default=98.4, types=float,
                           desc='Flight Controls Fuselage C.G location as percentage of fuselage length')
        self.options.declare('cg_flight_controls_wing_percentage', default=64.3, types=float,
                           desc='Flight Controls Wing C.G location as percentage of MAC')

    def setup(self):
        self.add_input('max_mach', val=0.69, units=None, desc='Maximum Mach number')
        self.add_input('wing_surface', val=927.85, units='ft**2', desc='Wing reference area')
        self.add_input('gross_weight', val=86000.0, units='lbm', desc='Aircraft gross weight')
        self.add_input('control_surface_ratio', val=0.35393652, units=None, desc='Control surface area ratio of wing surface')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')

        self.add_output('flight_controls_weight', val=0.0, units='lbm', desc='Total flight controls weight')
        self.add_output('cg_flight_controls', val=0.0, units='inch', desc='Flight Controls total C.G location (weighted average of fuselage and wing portions)')
        
        # Declare partials
        self.declare_partials('flight_controls_weight', '*')
        self.declare_partials('cg_flight_controls', ['max_mach', 'wing_surface', 'gross_weight', 'control_surface_ratio', 'fuselage_length', 'MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Get constants from options
        base_coeff = self.options['base_coefficient']
        
        # Get control_surface_ratio from inputs (from ac_data.xlsx)
        control_ratio = inputs['control_surface_ratio']
        
        # Movable surface area = control_surface_ratio * wing_reference_area
        s_controls = control_ratio * inputs['wing_surface']
        total_weight = base_coeff * (inputs['max_mach'] ** 0.52) * (s_controls ** 0.6) * (inputs['gross_weight'] ** 0.32)
        # Subtract 123.75 from the total weight
        outputs['flight_controls_weight'] = total_weight - 123.75
        
        # Split weight into fuselage and wing portions
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        # Calculate C.G for fuselage portion
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        cg_fuselage_pct = self.options['cg_flight_controls_fuselage_percentage']
        cg_fuselage = fs_start + (fuselage_length_in * cg_fuselage_pct / 100.0)
        
        # Calculate C.G for wing portion
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_wing_pct = self.options['cg_flight_controls_wing_percentage']
        cg_wing = LEMAC + (MAC * cg_wing_pct / 100.0)
        
        # Calculate weighted average total C.G
        outputs['cg_flight_controls'] = (weight_fuselage * cg_fuselage + weight_wing * cg_wing) / total_weight

    def compute_partials(self, inputs, partials):
        # Get constants from options
        base_coeff = self.options['base_coefficient']
        
        # Get control_surface_ratio from inputs
        control_ratio = inputs['control_surface_ratio']
        
        s_controls = control_ratio * inputs['wing_surface']
        mach = inputs['max_mach']
        gross = inputs['gross_weight']
        
        # Partial w.r.t. max_mach: d/dM (base_coeff * M^0.52 * S^0.6 * W^0.32)
        # = base_coeff * 0.52 * M^-0.48 * S^0.6 * W^0.32
        partials['flight_controls_weight', 'max_mach'] = (base_coeff * 0.52 * (mach ** -0.48) * 
                                                          (s_controls ** 0.6) * (gross ** 0.32))
        
        # Partial w.r.t. wing_surface: d/dS (base_coeff * M^0.52 * (control_ratio*S)^0.6 * W^0.32)
        # = base_coeff * M^0.52 * 0.6 * (control_ratio*S)^-0.4 * control_ratio * W^0.32
        # = base_coeff * M^0.52 * 0.6 * control_ratio^0.6 * S^-0.4 * W^0.32
        partials['flight_controls_weight', 'wing_surface'] = (base_coeff * (mach ** 0.52) * 0.6 * 
                                                               (control_ratio ** 0.6) * (inputs['wing_surface'] ** -0.4) * 
                                                               (gross ** 0.32))
        
        # Partial w.r.t. gross_weight: d/dW (base_coeff * M^0.52 * S^0.6 * W^0.32)
        # = base_coeff * M^0.52 * S^0.6 * 0.32 * W^-0.68
        partials['flight_controls_weight', 'gross_weight'] = (base_coeff * (mach ** 0.52) * 
                                                               (s_controls ** 0.6) * 0.32 * (gross ** -0.68))
        
        # Partial w.r.t. control_surface_ratio: d/dR (base_coeff * M^0.52 * (R*S)^0.6 * W^0.32)
        # = base_coeff * M^0.52 * 0.6 * (R*S)^-0.4 * S * W^0.32
        # = base_coeff * M^0.52 * 0.6 * R^-0.4 * S^0.6 * W^0.32
        partials['flight_controls_weight', 'control_surface_ratio'] = (base_coeff * (mach ** 0.52) * 0.6 * 
                                                                       (control_ratio ** -0.4) * 
                                                                       (inputs['wing_surface'] ** 0.6) * 
                                                                       (gross ** 0.32))
        
        # Partials for C.G
        total_weight = base_coeff * (mach ** 0.52) * (s_controls ** 0.6) * (gross ** 0.32)
        weight_fuselage_pct = self.options['weight_fuselage_percentage']
        weight_wing_pct = 100.0 - weight_fuselage_pct
        weight_fuselage = total_weight * weight_fuselage_pct / 100.0
        weight_wing = total_weight * weight_wing_pct / 100.0
        
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        cg_fuselage_pct = self.options['cg_flight_controls_fuselage_percentage']
        cg_fuselage = fs_start + (fuselage_length_in * cg_fuselage_pct / 100.0)
        
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_wing_pct = self.options['cg_flight_controls_wing_percentage']
        cg_wing = LEMAC + (MAC * cg_wing_pct / 100.0)
        cg_total = (weight_fuselage * cg_fuselage + weight_wing * cg_wing) / total_weight
        
        # Partials w.r.t. inputs that affect weight (and thus C.G through weighted average)
        dW_dM = base_coeff * 0.52 * (mach ** -0.48) * (s_controls ** 0.6) * (gross ** 0.32)
        dW_dS = base_coeff * (mach ** 0.52) * 0.6 * (control_ratio ** 0.6) * (inputs['wing_surface'] ** -0.4) * (gross ** 0.32)
        dW_dG = base_coeff * (mach ** 0.52) * (s_controls ** 0.6) * 0.32 * (gross ** -0.68)
        dW_dR = base_coeff * (mach ** 0.52) * 0.6 * (control_ratio ** -0.4) * (inputs['wing_surface'] ** 0.6) * (gross ** 0.32)
        
        dWf_dW = weight_fuselage_pct / 100.0
        dWw_dW = weight_wing_pct / 100.0
        
        # d(cg_total)/d(input) = d/d(input)[(Wf*cg_f + Ww*cg_w)/W]
        # = (dWf/d(input)*cg_f + Wf*dcg_f/d(input) + dWw/d(input)*cg_w + Ww*dcg_w/d(input))/W - cg_total*dW/d(input)/W
        partials['cg_flight_controls', 'max_mach'] = (dWf_dW * cg_fuselage + dWw_dW * cg_wing - cg_total) * dW_dM / total_weight
        partials['cg_flight_controls', 'wing_surface'] = (dWf_dW * cg_fuselage + dWw_dW * cg_wing - cg_total) * dW_dS / total_weight
        partials['cg_flight_controls', 'gross_weight'] = (dWf_dW * cg_fuselage + dWw_dW * cg_wing - cg_total) * dW_dG / total_weight
        partials['cg_flight_controls', 'control_surface_ratio'] = (dWf_dW * cg_fuselage + dWw_dW * cg_wing - cg_total) * dW_dR / total_weight
        
        # Partials w.r.t. lt (affects fuselage C.G only)
        partials['cg_flight_controls', 'fuselage_length'] = weight_fuselage * 12.0 * cg_fuselage_pct / 100.0 / total_weight
        
        # Partials w.r.t. MAC (affects wing C.G only)
        partials['cg_flight_controls', 'MAC'] = weight_wing * cg_wing_pct / 100.0 / total_weight
        
        # Partials w.r.t. LEMAC (affects wing C.G only)
        partials['cg_flight_controls', 'LEMAC'] = weight_wing / total_weight


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('max_mach', val=0.7, units=None)
    ivc.add_output('wing_surface', val=800.0, units='ft**2')
    ivc.add_output('gross_weight', val=80000.0, units='lbm')

    model.add_subsystem('flight_controls', FlightControlsWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Flight Controls Weight:', prob.get_val('flight_controls_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)