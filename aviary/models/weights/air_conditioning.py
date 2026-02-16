import openmdao.api as om
import numpy as np


class AirConditioningWeight(om.ExplicitComponent):
    """
    Calculates air conditioning weight based on Equation 113:
      WAC = ((3.2 × (FPAREA × DF)^0.6 + 9 × NPASS^0.83) × VMAX + 0.075 × WAVONC) × 1.4 + compressor_weight
    
    where:
    - FPAREA = fuselage_length × fuselage_width (fuselage planform area in ft²)
    - DF = fuselage_height (maximum fuselage depth in ft)
    - NPASS = maximum number of passengers
    - VMAX = maximum Mach number
    - WAVONC = weight of avionics system group (lb)
    - compressor_weight = compressor fixed weight (lb)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_air_conditioning_percentage', default=52.6, types=float,
                           desc='Air Conditioning C.G location as percentage of fuselage length')
        self.options.declare('vmax', default=0.69, types=float,
                           desc='Maximum Mach number')
        self.options.declare('wavonc', default=259.0, types=float,
                           desc='Weight of avionics system group (lbm)')
        self.options.declare('compressor_weight', default=160.15, types=float,
                           desc='Compressor fixed weight (lbm)')

    def setup(self):
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('bf', val=10.16, units='ft', desc='Fuselage width (ft)')
        self.add_input('hf', val=9.87, units='ft', desc='Fuselage height (ft)')
        self.add_input('max_passengers', val=88, units=None, desc='Maximum number of passengers')
        
        self.add_output('air_conditioning_weight', val=0.0, units='lbm', desc='Air conditioning weight')
        self.add_output('cg_air_conditioning', val=0.0, units='inch', desc='Air Conditioning C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('air_conditioning_weight', ['fuselage_length', 'bf', 'hf', 'max_passengers'])
        self.declare_partials('cg_air_conditioning', 'fuselage_length')

    def compute(self, inputs, outputs):
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_ft = inputs['fuselage_length']
        
        # Calculate fuselage planform area: FPAREA = fuselage_length × fuselage_width
        FPAREA = fuselage_length_ft * inputs['bf']
        
        # Maximum fuselage depth: DF = fuselage_height
        DF = inputs['hf']
        
        # Maximum number of passengers
        NPASS = inputs['max_passengers']
        
        # Get constants from options
        VMAX = self.options['vmax']
        WAVONC = self.options['wavonc']
        compressor_weight = self.options['compressor_weight']
        
        # Calculate weight: WAC = ((3.2 × (FPAREA × DF)^0.6 + 9 × NPASS^0.83) × VMAX + 0.075 × WAVONC) × 1.4 + compressor_weight
        term1 = 3.2 * ((FPAREA * DF) ** 0.6)
        term2 = 9.0 * (NPASS ** 0.83)
        outputs['air_conditioning_weight'] = ((term1 + term2) * VMAX + 0.075 * WAVONC) * 1.4 + compressor_weight
        
        # Calculate C.G location
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_air_conditioning_percentage']
        outputs['cg_air_conditioning'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)

    def compute_partials(self, inputs, partials):
        # Get constants
        VMAX = self.options['vmax']
        WAVONC = self.options['wavonc']
        
        # Calculate intermediate values
        fuselage_length_ft = inputs['fuselage_length']
        FPAREA = fuselage_length_ft * inputs['bf']
        DF = inputs['hf']
        NPASS = inputs['max_passengers']
        
        # Calculate partials for weight
        # WAC = ((3.2 × (FPAREA × DF)^0.6 + 9 × NPASS^0.83) × VMAX + 0.075 × WAVONC) × 1.4 + compressor_weight
        
        # Partial w.r.t. fuselage_length (affects FPAREA)
        # d(FPAREA)/d(fuselage_length) = bf
        # d((FPAREA × DF)^0.6)/d(FPAREA) = 0.6 × (FPAREA × DF)^(-0.4) × DF
        dFPAREA_dfuselage_length = inputs['bf']
        d_term1_dFPAREA = 3.2 * 0.6 * ((FPAREA * DF) ** (-0.4)) * DF
        partials['air_conditioning_weight', 'fuselage_length'] = d_term1_dFPAREA * dFPAREA_dfuselage_length * VMAX * 1.4
        
        # Partial w.r.t. bf (affects FPAREA)
        # d(FPAREA)/d(bf) = fuselage_length_ft
        dFPAREA_dbf = fuselage_length_ft
        partials['air_conditioning_weight', 'bf'] = d_term1_dFPAREA * dFPAREA_dbf * VMAX * 1.4
        
        # Partial w.r.t. hf (affects DF)
        # d((FPAREA × DF)^0.6)/d(DF) = 0.6 × (FPAREA × DF)^(-0.4) × FPAREA
        d_term1_dDF = 3.2 * 0.6 * ((FPAREA * DF) ** (-0.4)) * FPAREA
        partials['air_conditioning_weight', 'hf'] = d_term1_dDF * VMAX * 1.4
        
        # Partial w.r.t. max_passengers
        # d(9 × NPASS^0.83)/d(NPASS) = 9 × 0.83 × NPASS^(-0.17)
        partials['air_conditioning_weight', 'max_passengers'] = 9.0 * 0.83 * (NPASS ** (-0.17)) * VMAX * 1.4
        
        # Partials for C.G
        cg_percentage = self.options['cg_air_conditioning_percentage']
        partials['cg_air_conditioning', 'fuselage_length'] = 12.0 * cg_percentage / 100.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('fuselage_length', val=97.89, units='ft')
    ivc.add_output('bf', val=10.16, units='ft')
    ivc.add_output('hf', val=9.87, units='ft')
    ivc.add_output('max_passengers', val=88, units=None)

    model.add_subsystem('air_conditioning', AirConditioningWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Air Conditioning Weight:', prob.get_val('air_conditioning_weight', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)