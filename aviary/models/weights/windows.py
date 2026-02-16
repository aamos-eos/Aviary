import openmdao.api as om
import numpy as np

class WindowsMass(om.ExplicitComponent):
    """
    Calculate the weight of aircraft windows and windshield.
    
    Windows use pressure-based thickness calculation:
    t = a_eff * sqrt((k * delta_P) / sigma_allow)
    
    Then weight per window = 2 layers * thickness * area * density
    Total windows weight = weight per window * number of windows
    
    Windshield weight = thickness * density * (surface_front + surface_side) * 2 (per side)
    
    NOTE: This component uses sqrt() which makes partials complex.
    Manual review recommended for compute_partials implementation.
    """

    def initialize(self):
        # Constants for windows calculation
        self.options.declare('window_density', default=0.043, types=float,
                           desc='Window glass density (lbm/in³)')
        self.options.declare('k', default=0.4, types=float,
                           desc='Window pressure coefficient (between 0.35 and 0.47)')
        self.options.declare('safety_factor', default=1.3, types=float,
                           desc='Safety factor for pressure (delta_P = base_pressure * safety_factor)')
        self.options.declare('sigma_allow', default=2900.0, types=float,
                           desc='Allowable stress (equivalent to 20 MPa) in psi')
        self.options.declare('n_windows', default=54, types=int,
                           desc='Number of passenger windows')
        
        # Constants for windshield calculation
        self.options.declare('windshield_density', default=0.093, types=float,
                           desc='Windshield glass density (lbm/in³)')

    def setup(self):
        # Inputs for windows calculation (pressure-based)
        self.add_input('a_eff', val=6.0, units='inch',
                      desc='Effective radius (small radius) for window thickness calculation')
        self.add_input('base_pressure', val=6.0, units='psi',
                      desc='Base pressure differential (before safety factor)')
        self.add_input('window_area', val=175.0, units='inch**2',
                      desc='Area of each passenger window (surface)')

        # Inputs for windshield calculation
        self.add_input('windshield_thickness', val=0.50, units='inch',
                      desc='Thickness of windshield')
        self.add_input('windshield_surface_front', val=1100.0, units='inch**2',
                      desc='Front surface area of windshield')
        self.add_input('windshield_surface_side', val=928.7, units='inch**2',
                      desc='Side surface area of windshield')

        # Only keep total weight output as requested
        self.add_output('W_total', val=0.0, units='lbm',
                       desc='Total weight of all windows and windshield')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        self.add_output('cg_windows', val=0.0, units='inch', desc='Windows C.G location (same as LEMAC)')
        
        # FLAG: Complex partials due to sqrt() - declare all
        self.declare_partials('W_total', '*')
        self.declare_partials('cg_windows', 'LEMAC')

    def compute(self, inputs, outputs):
        window_density = self.options['window_density']
        windshield_density = self.options['windshield_density']
        k = self.options['k']
        safety_factor = self.options['safety_factor']
        sigma_allow = self.options['sigma_allow']
        n_windows = self.options['n_windows']
        
        # Calculate pressure differential with safety factor
        delta_P = inputs['base_pressure'] * safety_factor
        
        # Calculate window thickness using pressure-based formula
        # t = a_eff * sqrt((k * delta_P) / sigma_allow)
        window_thickness = inputs['a_eff'] * np.sqrt((k * delta_P) / sigma_allow)
        
        # Calculate weight per window (2 pressure bearing layers)
        # Apply 1.1 factor to weight per layer
        weight_per_layer = window_thickness * inputs['window_area'] * window_density * 1.1
        weight_per_window = weight_per_layer * 2.0  # Two layers
        
        # Calculate total windows weight (multiply by number of windows)
        W_windows = n_windows * weight_per_window

        # Calculate windshield weight
        # Weight = thickness * density * (surface_front + surface_side) * 2 (per side)
        windshield_total_surface = inputs['windshield_surface_front'] + inputs['windshield_surface_side']
        windshield_weight_per_side = inputs['windshield_thickness'] * windshield_total_surface * windshield_density
        W_windshield = windshield_weight_per_side * 2.0  # Multiply by 2 for both sides

        # Calculate total weight
        outputs['W_total'] = W_windows + W_windshield
        
        # Calculate C.G location - same as LEMAC
        outputs['cg_windows'] = inputs['LEMAC']

    def compute_partials(self, inputs, partials):
        # FLAG: Complex partials due to sqrt() - Manual review needed
        window_density = self.options['window_density']
        windshield_density = self.options['windshield_density']
        k = self.options['k']
        safety_factor = self.options['safety_factor']
        sigma_allow = self.options['sigma_allow']
        n_windows = self.options['n_windows']
        
        delta_P = inputs['base_pressure'] * safety_factor
        window_thickness = inputs['a_eff'] * np.sqrt((k * delta_P) / sigma_allow)
        
        # Partials for windows weight
        # d/d(a_eff): affects window_thickness linearly
        d_thickness_d_a_eff = np.sqrt((k * delta_P) / sigma_allow)
        d_windows_d_a_eff = n_windows * 2.0 * 1.1 * window_density * inputs['window_area'] * d_thickness_d_a_eff
        
        # d/d(base_pressure): affects delta_P, which affects sqrt
        # d/d(delta_P) of sqrt((k*delta_P)/sigma) = 0.5 * sqrt(sigma/(k*delta_P)) * k/sigma
        # = 0.5 * k / (sigma * sqrt((k*delta_P)/sigma)) = 0.5 * k / (sqrt(k*delta_P*sigma))
        d_thickness_d_delta_P = inputs['a_eff'] * 0.5 * k / (sigma_allow * np.sqrt((k * delta_P) / sigma_allow))
        d_windows_d_base_pressure = n_windows * 2.0 * 1.1 * window_density * inputs['window_area'] * d_thickness_d_delta_P * safety_factor
        
        # d/d(window_area): linear
        d_windows_d_window_area = n_windows * 2.0 * 1.1 * window_density * window_thickness
        
        # Partials for windshield weight
        windshield_total_surface = inputs['windshield_surface_front'] + inputs['windshield_surface_side']
        d_windshield_d_thickness = windshield_total_surface * windshield_density * 2.0
        d_windshield_d_front = inputs['windshield_thickness'] * windshield_density * 2.0
        d_windshield_d_side = inputs['windshield_thickness'] * windshield_density * 2.0
        
        # Total partials
        partials['W_total', 'a_eff'] = d_windows_d_a_eff
        partials['W_total', 'base_pressure'] = d_windows_d_base_pressure
        partials['W_total', 'window_area'] = d_windows_d_window_area
        partials['W_total', 'windshield_thickness'] = d_windshield_d_thickness
        partials['W_total', 'windshield_surface_front'] = d_windshield_d_front
        partials['W_total', 'windshield_surface_side'] = d_windshield_d_side
        
        # Partials for C.G
        partials['cg_windows', 'LEMAC'] = 1.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Windows inputs
    ivc.add_output('a_eff', val=6.0, units='inch')
    ivc.add_output('base_pressure', val=6.0, units='psi')
    ivc.add_output('window_area', val=175.0, units='inch**2')
    # Windshield inputs
    ivc.add_output('windshield_thickness', val=0.50, units='inch')
    ivc.add_output('windshield_surface_front', val=1100.0, units='inch**2')
    ivc.add_output('windshield_surface_side', val=928.7, units='inch**2')

    model.add_subsystem('windows', WindowsMass(), promotes=['*'])

    prob.setup()
    prob.run_model()

    print('Total Windows Weight:', prob.get_val('W_total', units='lbm'))
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)