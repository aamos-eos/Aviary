import openmdao.api as om
import numpy as np


class WingParameterLinks(om.ExplicitComponent):
    """
    Computes wing position parameters: LEMAC and front_spar_location.
    
    Other wing geometry (taper_ratio, wing_area, aspect_ratio, MAC, etc.) is computed 
    by compute_wing_geometry in load_ac_data.py and passed as inputs.
    
    Outputs:
    - LEMAC (Leading Edge Mean Aerodynamic Chord) = computed from wing geometry and apex location
    - front_spar_location = apex_location + (front_spar_percentage / 100) * MAC
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('front_spar_percentage', default=21.22, types=float,
                           desc='Front spar location as percentage of MAC from apex')
    
    def setup(self):
        # Inputs - wing geometry from ac_data.xlsx (computed by compute_wing_geometry)
        # Use trapezoidal span for LEMAC calculation (geometry-based)
        self.add_input('taper_ratio', val=0.35, units=None, desc='Wing taper ratio (ct/cr)')
        self.add_input('span_trap', val=1328.0, units='inch', desc='Trapezoidal wing span (for geometry calculations)')
        self.add_input('c0_sweep', val=4.854, units='deg', desc='Leading edge sweep angle')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord (from trapezoidal geometry)')
        self.add_input('wing_apex_percentage', val=46.2, units=None, 
                      desc='Wing apex location as percentage of fuselage length')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        
        # Outputs - computed position parameters
        self.add_output('LEMAC', val=0.0, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        self.add_output('front_spar_location', val=0.0, units='inch', desc='Front spar location (Fuselage Station)')
        
        self.declare_partials('LEMAC', ['taper_ratio', 'span_trap', 'c0_sweep', 
                                       'wing_apex_percentage', 'fuselage_length'])
        self.declare_partials('front_spar_location', ['MAC', 'wing_apex_percentage', 'fuselage_length'])
    
    def compute(self, inputs, outputs):
        taper = inputs['taper_ratio']
        b_inch = inputs['span_trap']
        sweep_LE = inputs['c0_sweep']
        MAC = inputs['MAC']
        
        # Calculate apex location in inches
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        fs_start = self.options['fs_start']
        apex_percentage = inputs['wing_apex_percentage']
        apex_location = fs_start + (fuselage_length_in * apex_percentage / 100.0)
        
        # LEMAC = b * (1+2*λ)/(6*(1+λ)) * tan(sweep_LE) + apex_location
        LEMAC_coeff = (1.0 + 2.0 * taper) / (6.0 * (1.0 + taper))
        LEMAC_term = b_inch * LEMAC_coeff * np.tan(sweep_LE * np.pi / 180.0)
        outputs['LEMAC'] = LEMAC_term + apex_location
        
        # Front spar = apex_location + (front_spar_percentage / 100) * MAC
        front_spar_pct = self.options['front_spar_percentage']
        outputs['front_spar_location'] = apex_location + (front_spar_pct / 100.0) * MAC
    
    def compute_partials(self, inputs, partials):
        taper = inputs['taper_ratio']
        b_inch = inputs['span_trap']
        sweep_LE = inputs['c0_sweep']
        MAC = inputs['MAC']
        
        fuselage_length_ft = inputs['fuselage_length']
        fuselage_length_in = fuselage_length_ft * 12.0
        apex_percentage = inputs['wing_apex_percentage']
        
        tan_sweep = np.tan(sweep_LE * np.pi / 180.0)
        LEMAC_coeff = (1.0 + 2.0 * taper) / (6.0 * (1.0 + taper))
        
        # d(LEMAC)/d(taper_ratio)
        # d(coeff)/d(taper) = d[(1+2λ)/(6(1+λ))]/dλ = [2(1+λ) - (1+2λ)] / [6(1+λ)²] = 1 / [6(1+λ)²]
        d_coeff_d_taper = 1.0 / (6.0 * (1.0 + taper)**2)
        partials['LEMAC', 'taper_ratio'] = b_inch * d_coeff_d_taper * tan_sweep
        
        # d(LEMAC)/d(span_trap)
        partials['LEMAC', 'span_trap'] = LEMAC_coeff * tan_sweep
        
        # d(LEMAC)/d(c0_sweep)
        d_tan_d_sweep = (1.0 / np.cos(np.deg2rad(sweep_LE)))**2 * np.deg2rad(1.0)
        partials['LEMAC', 'c0_sweep'] = b_inch * LEMAC_coeff * d_tan_d_sweep
        
        # d(LEMAC)/d(wing_apex_percentage)
        partials['LEMAC', 'wing_apex_percentage'] = fuselage_length_in / 100.0
        
        # d(LEMAC)/d(fuselage_length)
        partials['LEMAC', 'fuselage_length'] = 12.0 * apex_percentage / 100.0
        
        # Partials for front_spar_location
        front_spar_pct = self.options['front_spar_percentage']
        front_spar_factor = front_spar_pct / 100.0
        
        partials['front_spar_location', 'MAC'] = front_spar_factor
        partials['front_spar_location', 'wing_apex_percentage'] = fuselage_length_in / 100.0
        partials['front_spar_location', 'fuselage_length'] = 12.0 * apex_percentage / 100.0


class FuselageParameterLinks(om.ExplicitComponent):
    """
    Links fuselage and cabin parameters and calculates gross shell area and tail arm.
    
    Calculates:
    1. Cabin length from number of passengers:
       - Base cabin length from ac|geom|cabin|cabin_length (ft)
       - For every 4 passengers change: ±31 inches (default)
       - Formula: length_pax_cabin = cabin_length + (num_passengers - threshold_pax) // 4 * length_per_4_pax
    
    2. Fuselage length:
       - fuselage_length = length_pax_cabin + nose_tail_length
       - nose_tail_length from ac|geom|fuselage|nose_tail_length (ft)
    
    3. Tail arm (lt):
       - lt = tail_arm_fraction * fuselage_length
       - Maintains geometric proportions as fuselage scales
    
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('threshold_pax', default=73, types=(int, float),
                           desc='Threshold passenger count for length calculation')
        self.options.declare('length_increment_in', default=31.0, types=float,
                           desc='Length increment per 4 passengers (inches)')
        self.options.declare('tail_arm_fraction', default=0.485775, types=float,
                           desc='Tail arm as fraction of fuselage length')
        self.options.declare('shell_factor', default=0.9, types=float,
                           desc='Shell area factor (accounts for nose/tail shrink)')
    
    def setup(self):
        self.add_input('num_passengers', val=76, units=None, desc='Number of passengers')
        self.add_input('cabin_length', val=63.0, units='ft', desc='Base cabin length (ft) at threshold passengers')
        self.add_input('nose_tail_length', val=34.89, units='ft', desc='Fuselage length extension beyond cabin (ft)')
        self.add_input('base', val=10.16, units='ft', desc='Fuselage width')
        self.add_input('height', val=9.87, units='ft', desc='Fuselage height')
        
        self.add_output('length_pax_cabin', val=63.0, units='ft', desc='Length of passenger cabin')
        self.add_output('fuselage_length', val=97.89, units='ft', desc='Fuselage length')
        self.add_output('lt', val=47.02, units='ft', desc='Tail arm')
        
        self.declare_partials('length_pax_cabin', ['num_passengers', 'cabin_length'])
        self.declare_partials('fuselage_length', ['num_passengers', 'cabin_length', 'nose_tail_length'])
        self.declare_partials('lt', ['num_passengers', 'cabin_length', 'nose_tail_length'])
    
    def compute(self, inputs, outputs):
        num_pax = inputs['num_passengers']
        bf = inputs['base']
        hf = inputs['height']
        
        cabin_length = inputs['cabin_length']
        nose_tail_length = inputs['nose_tail_length']
        threshold_pax = self.options['threshold_pax']
        length_per_4_pax = self.options['length_increment_in'] / 12.0  # Convert to ft
        tail_arm_fraction = self.options['tail_arm_fraction']
        shell_factor = self.options['shell_factor']
        
        # Calculate cabin length
        steps = (num_pax - threshold_pax) // 4
        length_pax_cabin = cabin_length + steps * length_per_4_pax
        outputs['length_pax_cabin'] = length_pax_cabin
        
        # Calculate fuselage length
        fuselage_length = length_pax_cabin + nose_tail_length
        outputs['fuselage_length'] = fuselage_length
        
        outputs['lt'] = tail_arm_fraction * fuselage_length
        
    
    def compute_partials(self, inputs, partials):
        num_pax = inputs['num_passengers']
        bf = inputs['base']
        hf = inputs['height']
        
        cabin_length = inputs['cabin_length']
        nose_tail_length = inputs['nose_tail_length']
        threshold_pax = self.options['threshold_pax']
        length_per_4_pax = self.options['length_increment_in'] / 12.0
        tail_arm_fraction = self.options['tail_arm_fraction']
        shell_factor = self.options['shell_factor']
        
        # Surrogate gradient for floor division
        d_length_d_pax = length_per_4_pax / 4.0
        
        # Partials for length_pax_cabin
        partials['length_pax_cabin', 'num_passengers'] = d_length_d_pax
        partials['length_pax_cabin', 'cabin_length'] = 1.0
        
        # Partials for fuselage_length
        partials['fuselage_length', 'num_passengers'] = d_length_d_pax
        partials['fuselage_length', 'cabin_length'] = 1.0
        partials['fuselage_length', 'nose_tail_length'] = 1.0
        
        # Partials for lt
        partials['lt', 'num_passengers'] = tail_arm_fraction * d_length_d_pax
        partials['lt', 'cabin_length'] = tail_arm_fraction
        partials['lt', 'nose_tail_length'] = tail_arm_fraction
        

class NacelleParameterLinks(om.ExplicitComponent):
    """
    Computes nacelle lengths from battery energy capacity.
    
    Nacelle length increases in steps as battery energy increases:
    - Every 2 modules of energy, alternating between inboard and outboard nacelles
    - Inboard nacelles get the first increment at threshold crossings
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, types=int,desc='number of props per aircraft')
        self.options.declare('base_energy', default=2640588.0, types=float,
                           desc='Base battery energy capacity (W*h)')
        self.options.declare('module_energy', default=47200.0, types=float,
                           desc='Energy per battery module (W*h)')
        self.options.declare('length_increment_in', default=8.0, types=float,
                           desc='Nacelle length increment (inches)')
        self.options.declare('cabin_length_inboard_in', default=322.39, types=float,
                           desc='Base inboard nacelle length (inches)')
        self.options.declare('cabin_length_outboard_in', default=304.76, types=float,
                           desc='Base outboard nacelle length (inches)')
    
    def setup(self):
        self.add_input('batt_energy', val=2640588.0, units='W*h', desc='Battery energy capacity')

        cabin_length_inboard = self.options['cabin_length_inboard_in'] / 12.0
        cabin_length_outboard = self.options['cabin_length_outboard_in'] / 12.0
        
        self.add_output('nacelle_length_inboard', val=cabin_length_inboard, units='ft', desc='Inboard nacelle length')
        self.add_output('nacelle_length_outboard', val=cabin_length_outboard, units='ft', desc='Outboard nacelle length')
        self.add_output('nac_engine_weight', units='lbm', desc='IPS weight per nacelle')
        self.add_output('nac_batteries_weight', units='lbm', desc='Batteries weight per nacelle')
        
        self.declare_partials('nacelle_length_inboard', 'batt_energy')
        self.declare_partials('nacelle_length_outboard', 'batt_energy')
    
    def compute(self, inputs, outputs):
        batt_energy = inputs['batt_energy']
        
        base_energy = self.options['base_energy']
        module_energy = self.options['module_energy']
        threshold_increment = 2.0 * module_energy
        length_increment = self.options['length_increment_in'] / 12.0  # Convert to ft
        
        cabin_length_inboard = self.options['cabin_length_inboard_in'] / 12.0
        cabin_length_outboard = self.options['cabin_length_outboard_in'] / 12.0
        
        energy_diff = batt_energy - base_energy
        thresholds = energy_diff / threshold_increment
        
        inboard_increments = int((thresholds + 1) // 2)
        outboard_increments = int(thresholds // 2)

        # TODO: Update nacelle length calculation for wetted area. Rectangular prism overestimates...
        
        outputs['nacelle_length_inboard'] = cabin_length_inboard + inboard_increments * length_increment
        outputs['nacelle_length_outboard'] = cabin_length_outboard + outboard_increments * length_increment
    
    def compute_partials(self, inputs, partials):
        module_energy = self.options['module_energy']
        threshold_increment = 2.0 * module_energy
        length_increment = self.options['length_increment_in'] / 12.0
        
        # Surrogate gradients for floor division
        d_thresholds_d_energy = 1.0 / threshold_increment
        d_inboard_d_thresholds = 0.5
        d_outboard_d_thresholds = 0.5
        
        partials['nacelle_length_inboard', 'batt_energy'] = d_inboard_d_thresholds * d_thresholds_d_energy * length_increment
        partials['nacelle_length_outboard', 'batt_energy'] = d_outboard_d_thresholds * d_thresholds_d_energy * length_increment


class EMotorPowerLink(om.ExplicitComponent):
    """
    Computes unit electric motor rated power from nacelle-level power.
    
    This component calculates:
    - unit_rated_power_em = rated_power_em_per_nacelle / num_em_per_nac
    
    This is needed because the propulsion weight calculations expect per-motor
    power ratings, but the system-level inputs are often specified per nacelle.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
    
    def setup(self):
        self.add_input('rated_power_em_per_nacelle', val=1341.0, units='hp', 
                      desc='Rated power per nacelle (total for all motors in nacelle)')
        self.add_input('num_em_per_nac', val=1.0, units=None,
                      desc='Number of electric motors per nacelle')
        
        self.add_output('unit_rated_power_em', val=1341.0, units='hp',
                       desc='Rated power per individual electric motor')
        
        self.declare_partials('unit_rated_power_em', ['rated_power_em_per_nacelle', 'num_em_per_nac'])
    
    def compute(self, inputs, outputs):
        rated_power = inputs['rated_power_em_per_nacelle']
        num_em = inputs['num_em_per_nac']
        
        outputs['unit_rated_power_em'] = rated_power / num_em
    
    def compute_partials(self, inputs, partials):
        rated_power = inputs['rated_power_em_per_nacelle']
        num_em = inputs['num_em_per_nac']
        
        partials['unit_rated_power_em', 'rated_power_em_per_nacelle'] = 1.0 / num_em
        partials['unit_rated_power_em', 'num_em_per_nac'] = -rated_power / (num_em ** 2)


if __name__ == "__main__":
    # Example usage - Wing parameter links
    prob_wing = om.Problem(reports=False)
    model_wing = prob_wing.model
    
    # Add wing geometry inputs (from ac_data.xlsx / compute_wing_geometry)
    ivc_wing = model_wing.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    
    ivc_wing.add_output('taper_ratio', val=0.35, units=None, desc='Wing taper ratio')
    ivc_wing.add_output('span', val=1328.0, units='inch', desc='Total wing span')
    ivc_wing.add_output('c0_sweep', val=4.854, units='deg', desc='Leading edge sweep angle')
    ivc_wing.add_output('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
    ivc_wing.add_output('wing_apex_percentage', val=46.2, units=None, desc='Wing apex location as percentage of fuselage length')
    ivc_wing.add_output('fuselage_length', val=97.89, units='ft', desc='Fuselage length')
    
    # Add parameter links
    model_wing.add_subsystem('wing_params', WingParameterLinks(), promotes=['*'])
    
    prob_wing.setup()
    prob_wing.run_model()
    prob_wing.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
    
    print("=" * 60)
    print("WING PARAMETER LINKS DEMONSTRATION")
    print("=" * 60)
    print(f"Inputs (from ac_data.xlsx / compute_wing_geometry):")
    print(f"  taper_ratio = {prob_wing.get_val('taper_ratio', units=None)[0]:.4f}")
    print(f"  span = {prob_wing.get_val('span', units='inch')[0]:.2f} inch")
    print(f"  c0_sweep = {prob_wing.get_val('c0_sweep', units='deg')[0]:.3f} deg")
    print(f"  MAC = {prob_wing.get_val('MAC', units='inch')[0]:.2f} inch")
    print(f"  wing_apex_percentage = {prob_wing.get_val('wing_apex_percentage', units=None)[0]:.2f}%")
    print(f"  fuselage_length = {prob_wing.get_val('fuselage_length', units='ft')[0]:.2f} ft")
    print()
    print(f"Outputs:")
    print(f"  LEMAC (Leading Edge MAC) = {prob_wing.get_val('LEMAC', units='inch')[0]:.2f} inch")
    print(f"  front_spar_location = {prob_wing.get_val('front_spar_location', units='inch')[0]:.2f} inch")
    print("=" * 60)
    print()
    
    # Test fuselage parameter links
    prob_fuselage = om.Problem(reports=False)
    model_fuselage = prob_fuselage.model
    
    # Add fuselage inputs
    ivc_fuselage = model_fuselage.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc_fuselage.add_output('num_passengers', val=76, units=None, desc='Number of passengers')
    ivc_fuselage.add_output('fuse_base', val=10.16, units='ft', desc='Fuselage width')
    ivc_fuselage.add_output('fuse_height', val=9.87, units='ft', desc='Fuselage height')
    ivc_fuselage.add_output('cabin_length', val=63.0, units='ft', desc='Base cabin length')
    ivc_fuselage.add_output('nose_tail_length', val=34.89, units='ft', desc='Fuselage length extension')
    
    # Add fuselage parameter links (don't promote base/height/cabin_length/nose_tail_length)
    model_fuselage.add_subsystem('fuse_params', FuselageParameterLinks(), 
                                 promotes_inputs=['num_passengers'],
                                 promotes_outputs=['*'])
    model_fuselage.connect('fuse_base', 'fuse_params.base')
    model_fuselage.connect('fuse_height', 'fuse_params.height')
    model_fuselage.connect('cabin_length', 'fuse_params.cabin_length')
    model_fuselage.connect('nose_tail_length', 'fuse_params.nose_tail_length')
    
    prob_fuselage.setup()
    prob_fuselage.run_model()
    prob_fuselage.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
    
    print("=" * 60)
    print("FUSELAGE PARAMETER LINKS DEMONSTRATION")
    print("=" * 60)
    print(f"Inputs:")
    print(f"  num_passengers = {prob_fuselage.get_val('num_passengers', units=None)[0]:.0f}")
    print(f"  base (fuselage width) = {prob_fuselage.get_val('fuse_base', units='ft')[0]:.2f} ft")
    print(f"  height (fuselage height) = {prob_fuselage.get_val('fuse_height', units='ft')[0]:.2f} ft")
    print(f"  cabin_length = {prob_fuselage.get_val('cabin_length', units='ft')[0]:.2f} ft")
    print(f"  nose_tail_length = {prob_fuselage.get_val('nose_tail_length', units='ft')[0]:.2f} ft")
    print()
    print(f"Derived outputs:")
    print(f"  length_pax_cabin = {prob_fuselage.get_val('length_pax_cabin', units='ft')[0]:.2f} ft")
    print(f"  fuselage_length = {prob_fuselage.get_val('fuselage_length', units='ft')[0]:.2f} ft")
    print(f"  lt (tail arm) = {prob_fuselage.get_val('lt', units='ft')[0]:.2f} ft (48% of length)")
    print(f"  Sg (gross shell area) = {prob_fuselage.get_val('Sg', units='ft**2')[0]:.2f} ft²")
    print("=" * 60)
    print()
    
    prob_nacelle = om.Problem(reports=False)
    model_nacelle = prob_nacelle.model
    
    ivc_nacelle = model_nacelle.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc_nacelle.add_output('batt_energy', val=2640588.0, units='W*h', desc='Battery energy capacity')
    
    model_nacelle.add_subsystem('nacelle_params', NacelleParameterLinks(), promotes=['*'])
    
    prob_nacelle.setup()
    prob_nacelle.run_model()
    prob_nacelle.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
    
    print("=" * 60)
    print("NACELLE PARAMETER LINKS DEMONSTRATION")
    print("=" * 60)
    print(f"Inputs:")
    print(f"  energy = {prob_nacelle.get_val('batt_energy', units='W*h')[0]:.0f} Wh ({prob_nacelle.get_val('batt_energy', units='kW*h')[0]:.2f} kWh)")
    print()
    print(f"Derived outputs (used by nacelle.py):")
    print(f"  nacelle_length_inboard = {prob_nacelle.get_val('nacelle_length_inboard', units='ft')[0]:.2f} ft")
    print(f"  nacelle_length_outboard = {prob_nacelle.get_val('nacelle_length_outboard', units='ft')[0]:.2f} ft")
    print("=" * 60)
    print()
    
