import numpy as np
import jax.numpy as jnp
import openmdao.api as om
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator, griddata
from aviary.utils.dvlabel import DVLabel
from aviary.utils.math_components.add_subtract_comp import AddSubtractComp
from aviary.utils.math_components.multiply_divide_comp import ElementMultiplyDivideComp
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
import time

# Import the global data store
from aviary.models.engines.propulsion.prop.propeller_data import PropellerData

from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp

Debug = True


class ComputeThrustPowerRelation(om.ExplicitComponent):
    """
    Bidirectional component to compute either:
    - Thrust from power: T = P * eta / V (direction='power_in')
    - Power from thrust: P = T * V / eta (direction='thrust_in')
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        self.options.declare('direction', default='power_in', values=['power_in', 'thrust_in'], 
                           desc='Compute thrust from power (power_in) or power from thrust (thrust_in)')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        direction = self.options['direction']
        
        # Common inputs
        self.add_input('eta', shape=(npp, nn), desc='Efficiency')
        self.add_input('Utrue', shape=(nn,), units='m/s', desc='True airspeed')
        
        # Direction-specific inputs and outputs
        if direction == 'power_in':
            self.add_input('power', shape=(npp, nn), units='W', desc='Power input')
            self.add_output('thrust_calc', shape=(npp, nn), units='N', desc='Calculated thrust')
        else:  # direction == 'thrust_in'
            self.add_input('thrust', shape=(npp, nn), units='N', desc='Thrust input')
            self.add_output('power_calc', shape=(npp, nn), units='W', desc='Calculated power', upper=2e6)
        
        # Declare partials with rows and cols
        n = npp * nn
        rows = np.arange(n)
        cols_main = np.arange(n)  # For power or thrust
        cols_eta = np.arange(n)
        # For Utrue broadcasting: each output element (i,j) depends on Utrue[j]
        cols_Utrue = np.tile(np.arange(nn), npp)
        
        if direction == 'power_in':
            self.declare_partials('thrust_calc', 'power', rows=rows, cols=cols_main)
            self.declare_partials('thrust_calc', 'eta', rows=rows, cols=cols_eta)
            self.declare_partials('thrust_calc', 'Utrue', rows=rows, cols=cols_Utrue)
        else:  # direction == 'thrust_in'
            self.declare_partials('power_calc', 'thrust', rows=rows, cols=cols_main)
            self.declare_partials('power_calc', 'eta', rows=rows, cols=cols_eta)
            self.declare_partials('power_calc', 'Utrue', rows=rows, cols=cols_Utrue)
    
    def compute(self, inputs, outputs):
        direction = self.options['direction']
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        eta = inputs['eta']
        Utrue = inputs['Utrue']
        
        # Broadcast Utrue to match matrix shape (npp, nn)
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        
        if direction == 'power_in':
            # T = P * eta / V
            #print(f"Power (kW): {inputs['power']/1000}")
            power = inputs['power']
            thrust = power * eta / Utrue_broadcast
            #print(f"Thrust (N): {thrust}")
            #print(f"eta: {eta}")
            outputs['thrust_calc'] = thrust
        else:  # direction == 'thrust_in'
            # P = T * V / eta
            thrust = inputs['thrust']
            outputs['power_calc'] = thrust * Utrue_broadcast / eta
    
    def compute_partials(self, inputs, partials):
        direction = self.options['direction']
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        eta = inputs['eta']
        Utrue = inputs['Utrue']
        
        # Broadcast Utrue to match matrix shape (npp, nn)
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        
        if direction == 'power_in':
            # T = P * eta / V
            power = inputs['power']
            
            # dT/dP = eta / V
            partials['thrust_calc', 'power'] = (eta / Utrue_broadcast).flatten()
            
            # dT/deta = P / V
            partials['thrust_calc', 'eta'] = (power / Utrue_broadcast).flatten()
            
            # dT/dV = -P * eta / V^2
            partials['thrust_calc', 'Utrue'] = (-power * eta / (Utrue_broadcast**2)).flatten()
            
        else:  # direction == 'thrust_in'
            # P = T * V / eta
            thrust = inputs['thrust']
            
            # dP/dT = V / eta
            partials['power_calc', 'thrust'] = (Utrue_broadcast / eta).flatten()
            
            # dP/deta = -T * V / eta^2
            partials['power_calc', 'eta'] = (-thrust * Utrue_broadcast / (eta**2)).flatten()
            
            # dP/dV = T / eta
            partials['power_calc', 'Utrue'] = (thrust / eta).flatten() 


class ThrustCoefficient(om.ExplicitComponent):
    """
    Computes thrust coefficient: Ct = T / (rho * (rpm/60)^2 * D^4)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        
        self.options.declare('min_Ct', default=1e-6, desc='minimum thrust coefficient value')
        self.options.declare('max_Ct', default=0.283, desc='maximum thrust coefficient value')
        self.options.declare('mu', default=0.001, desc='smoothing parameter')

    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        self.add_input('thrust', shape=(npp, nn), units='N')
        self.add_input('rpm', shape=(npp, nn), units='rpm')
        self.add_input('diameter', shape=(npp, nn), units='m')
        self.add_input('rho', shape=(nn,), units='kg/m**3')

        self.add_output('Ct', shape=(npp, nn), lower = self.options['min_Ct']*1.001, upper = self.options['max_Ct']*0.999)
        
        # Declare partials with rows and cols
        n = npp * nn
        rows = np.arange(n)
        cols_thrust = np.arange(n)
        cols_rpm = np.arange(n)
        cols_diameter = np.arange(n)
        # For rho broadcasting: each output element (i,j) depends on rho[j]

        cols_rho = np.tile(np.arange(nn), npp)
        self.declare_partials('Ct', 'thrust', rows=rows, cols=cols_thrust)
        self.declare_partials('Ct', 'rpm', rows=rows, cols=cols_rpm)
        self.declare_partials('Ct', 'diameter', rows=rows, cols=cols_diameter)
        self.declare_partials('Ct', 'rho', rows=rows, cols=cols_rho)
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        thrust = inputs['thrust']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        rho = inputs['rho']
        
        # Broadcast rho to match matrix shape
                
        # Convert rpm to rev/s and compute Ct
        n = rpm / 60.0  # revolutions per second
        Ct = thrust / (rho * n**2 * diameter**4)
  
        outputs['Ct'] = Ct
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        thrust = inputs['thrust']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        rho = inputs['rho']
        
        # Convert rpm to rev/s
        n = rpm / 60.0
        
        # dCt/dthrust
        dCt_dT = 1.0 / (rho * n**2 * diameter**4)
        
        # dCt/drpm
        dCt_drpm = -2 * thrust / (rho * n**3 * diameter**4) / 60.0
        
        # dCt/ddiameter
        dCt_ddiam = -4 * thrust / (rho * n**2 * diameter**5)
        
        # dCt/drho (need to account for broadcasting)
        dCt_drho = -thrust / (rho**2 * n**2 * diameter**4)
        
        partials['Ct', 'thrust'] = dCt_dT.flatten()
        partials['Ct', 'rpm'] = dCt_drpm.flatten()
        partials['Ct', 'diameter'] = dCt_ddiam.flatten()
        partials['Ct', 'rho'] = dCt_drho.flatten()

class PowerCoefficient(om.ExplicitComponent):
    """
    Computes power coefficient: Cp = P / (rho * (rpm/60)^3 * D^5)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        #self.options.declare('min_power_W', default=1e-6, desc='minimum power value')
        #self.options.declare('max_power_W', default=30000, desc='maximum power value')
        #self.options.declare('mu', default=0.00001, desc='smoothing parameter')
        self.options.declare('min_Cp', default=1e-6, desc='minimum power coefficient value')
        self.options.declare('max_Cp', default=0.283, desc='maximum power coefficient value')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        self.add_input('power', shape=(npp, nn), units='W')
        self.add_input('rpm', shape=(npp, nn), units='rpm')
        self.add_input('diameter', shape=(npp, nn), units='m')
        self.add_input('rho', shape=(nn,), units='kg/m**3')

        self.add_output('Cp', shape=(npp, nn), lower = self.options['min_Cp']*1.001, upper = self.options['max_Cp']*0.999)
        # Declare partials with rows and cols
        n = npp * nn
        rows = np.arange(n)
        cols_power = np.arange(n)
        cols_rpm = np.arange(n)
        cols_diameter = np.arange(n)
        cols_rho = np.tile(np.arange(nn), npp)
        self.declare_partials('Cp', 'power', rows=rows, cols=cols_power)
        self.declare_partials('Cp', 'rpm', rows=rows, cols=cols_rpm)
        self.declare_partials('Cp', 'diameter', rows=rows, cols=cols_diameter)
        self.declare_partials('Cp', 'rho', rows=rows, cols=cols_rho)
    
    def compute(self, inputs, outputs):

        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        power = inputs['power']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        rho = inputs['rho']
        
        # Convert rpm to rev/s and compute Cp
        n = rpm / 60.0  # revolutions per second
        Cp = power / (rho* n**3 * diameter**5)
        
        # Clip Cp
        #Cp_clipped = np.clip(Cp, self.options['min_Cp'], self.options['max_Cp'])
        outputs['Cp'] = Cp
        

    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        power = inputs['power']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        rho = inputs['rho']
        
        # Broadcast rho to match matrix shape (using newaxis like in compute)
        
        # Convert rpm to rev/s
        n = rpm / 60.0
        
        dCp_dP = 1.0 / (rho * n**3 * diameter**5)
        dCp_drpm = -3 * power / (rho * n**4 * diameter**5) / 60.0
        dCp_ddiam = -5 * power / (rho * n**3 * diameter**6)
        dCp_drho = -power / (rho**2 * n**3 * diameter**5)
        
        # No clipping masks needed
        partials['Cp', 'power'] = dCp_dP.flatten()
        partials['Cp', 'rpm'] = dCp_drpm.flatten()
        partials['Cp', 'diameter'] = dCp_ddiam.flatten()
        partials['Cp', 'rho'] = dCp_drho.flatten()


class tipMach(om.ExplicitComponent):
    """
    Computes propeller tip Mach number.
    
    tipMach = sqrt(Utrue^2 + (omega * r_tip)^2) / a
    
    where omega = rpm * 2 * pi / 60 and r_tip = diameter / 2
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        self.add_input('Utrue', shape=(nn,), units='m/s', desc='True airspeed')
        self.add_input('rpm', shape=(npp, nn), units='rpm', desc='Propeller rotational speed')
        self.add_input('diameter', shape=(npp, nn), units='m', desc='Propeller diameter')
        self.add_input('a', shape=(nn,), units='m/s', desc='Speed of sound')
        
        self.add_output('tipMach', shape=(npp, nn), desc='Tip Mach number')
        
        # Declare partials
        n = npp * nn
        rows = np.arange(n)
        cols_rpm = np.arange(n)
        cols_diameter = np.arange(n)
        cols_Utrue = np.tile(np.arange(nn), npp)
        cols_a = np.tile(np.arange(nn), npp)
        
        self.declare_partials('tipMach', 'Utrue', rows=rows, cols=cols_Utrue)
        self.declare_partials('tipMach', 'rpm', rows=rows, cols=cols_rpm)
        self.declare_partials('tipMach', 'diameter', rows=rows, cols=cols_diameter)
        self.declare_partials('tipMach', 'a', rows=rows, cols=cols_a)
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        Utrue = inputs['Utrue']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        a = inputs['a']
        
        # Broadcast Utrue and a to match matrix shape
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        a_broadcast = np.broadcast_to(a, (npp, nn))
        
        # Calculate tip speed
        omega = rpm * 2 * np.pi / 60  # rad/s
        r_tip = diameter / 2  # m
        v_tip_rotational = omega * r_tip  # m/s
        
        # Total tip velocity (vector sum of forward and rotational)
        v_tip_total = np.sqrt(Utrue_broadcast**2 + v_tip_rotational**2)
        
        # Tip Mach number
        outputs['tipMach'] = v_tip_total / a_broadcast
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        Utrue = inputs['Utrue']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        a = inputs['a']
        
        # Broadcast to match matrix shape
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        a_broadcast = np.broadcast_to(a, (npp, nn))
        
        # Intermediate calculations
        omega = rpm * 2 * np.pi / 60
        r_tip = diameter / 2
        v_tip_rotational = omega * r_tip
        v_tip_total = np.sqrt(Utrue_broadcast**2 + v_tip_rotational**2)
        
        # d(tipMach)/d(Utrue) = Utrue / (v_tip_total * a)
        dtipMach_dUtrue = Utrue_broadcast / (v_tip_total * a_broadcast)
        
        # d(tipMach)/d(rpm) = (v_tip_rotational * (2*pi/60) * r_tip) / (v_tip_total * a)
        #                   = v_tip_rotational * (2*pi/60) * r_tip / (v_tip_total * a)
        dtipMach_drpm = (v_tip_rotational * (2 * np.pi / 60) * r_tip) / (v_tip_total * a_broadcast)
        
        # d(tipMach)/d(diameter) = (v_tip_rotational * omega * 0.5) / (v_tip_total * a)
        dtipMach_ddiameter = (v_tip_rotational * omega * 0.5) / (v_tip_total * a_broadcast)
        
        # d(tipMach)/d(a) = -v_tip_total / a^2
        dtipMach_da = -v_tip_total / (a_broadcast**2)
        
        partials['tipMach', 'Utrue'] = dtipMach_dUtrue.flatten()
        partials['tipMach', 'rpm'] = dtipMach_drpm.flatten()
        partials['tipMach', 'diameter'] = dtipMach_ddiameter.flatten()
        partials['tipMach', 'a'] = dtipMach_da.flatten()


class AdvanceRatio(om.ExplicitComponent):
    """
    Computes advance ratio: J = V / (n * D) = V / ((rpm/60) * D)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        self.options.declare('min_J', default=0.1287, desc='minimum advance ratio value')
        self.options.declare('max_J', default=2.5, desc='maximum advance ratio value')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        self.add_input('Utrue', shape=(nn,), units='m/s')
        self.add_input('rpm', shape=(npp, nn), units='rpm')
        self.add_input('diameter', shape=(npp, nn), units='m')

        self.add_output('adv_ratio', shape=(npp, nn), lower = self.options['min_J'], upper = self.options['max_J'])
        # Declare partials with rows and cols
        n = npp * nn
        rows = np.arange(n)
        cols_Utrue = np.tile(np.arange(nn), npp)
        cols_rpm = np.arange(n)
        cols_diameter = np.arange(n)
        self.declare_partials('adv_ratio', 'Utrue', rows=rows, cols=cols_Utrue)
        self.declare_partials('adv_ratio', 'rpm', rows=rows, cols=cols_rpm)
        self.declare_partials('adv_ratio', 'diameter', rows=rows, cols=cols_diameter)
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        Utrue = inputs['Utrue']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        
        # Broadcast Utrue to match matrix shape
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        
        # Convert rpm to rev/s and compute J
        n = rpm / 60.0  # revolutions per second
        adv_ratio = Utrue_broadcast / (n * diameter)
        
        # Clip advance ratio
        adv_ratio_clipped = np.clip(adv_ratio, self.options['min_J'], self.options['max_J'])
        
        outputs['adv_ratio'] = adv_ratio_clipped
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        Utrue = inputs['Utrue']
        rpm = inputs['rpm']
        diameter = inputs['diameter']
        
        # Broadcast Utrue to match matrix shape
        Utrue_broadcast = np.broadcast_to(Utrue, (npp, nn))
        
        n = rpm / 60.0
        adv_ratio = Utrue_broadcast / (n * diameter)
        
        # Derivatives of clipping function
        clip_mask = (adv_ratio >= self.options['min_J']) & (adv_ratio <= self.options['max_J'])
        
        # dJ/dUtrue
        dJ_dUtrue = 1.0 / (n * diameter)
        
        # dJ/drpm
        dJ_drpm = -Utrue_broadcast / (n**2 * diameter) / 60.0
        
        # dJ/ddiameter
        dJ_ddiam = -Utrue_broadcast / (n * diameter**2)
        
        # Apply clipping mask
        partials['adv_ratio', 'Utrue'] = (dJ_dUtrue * clip_mask).flatten()
        partials['adv_ratio', 'rpm'] = (dJ_drpm * clip_mask).flatten()
        partials['adv_ratio', 'diameter'] = (dJ_ddiam * clip_mask).flatten()


class EmpiricalPropellerCoeffMMMtip(om.Group):

    _grid_data_built = False
    _config_data_loaded = False
    _data_loaded = False
    
    # Cache file path for pre-computed grid data
    _cache_file = 'aviary/models/engines/propulsion/empirical_data/propeller_grid_cache.npz'

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        self.options.declare('thrust_set', default=True, desc='the thrust is set (input)')
        self.options.declare('power_set', default=False, desc='the power is set (input)')
        self.options.declare('rpm_set', default=False, desc='the rpm is set (input)')
        self.options.declare('test_interp', default=False, desc='test the interpolator against nearest neighbour')
        self.options.declare('turbine_eff', default=0.2, desc='propeller `efficiency` in turbine mode')
        self.options.declare('wmill_thresh_hp', default=200.0, desc='Power threshold for windmilling drag (HP)')
        self.options.declare('kwmdDrag', default=0.1, desc='Windmilling drag coefficient')
        
        start_init = time.time()
        self._load_data()
        self._load_config_data()
        self._prepare_grid_data()
        elapsed = time.time() - start_init
        #print(f"  -> Propeller model initialized in {elapsed:.1f} seconds")
    
    @classmethod
    def _load_data(cls):
        start_load_data = time.time()
        if cls._data_loaded:
            return
        PropellerData.load_data(prop_filename='aviary/models/engines/propulsion/empirical_data/DOWTY_prop_CCA8_6blade_13ft_cleaned.xlsx', sheet_name='data')
        PropellerData.load_rpm_schedule(prop_filename='aviary/models/engines/propulsion/empirical_data/DOWTY_prop_CCA8_6blade_13ft_cleaned.xlsx', sheet_name='RPM_schedule')
        elapsed = time.time() - start_load_data
        print(f"  -> Data loaded in {elapsed:.1f} seconds")
        cls._data_loaded = True
    
    @classmethod
    def _load_config_data(cls):
        start_load_config_data = time.time()
        if cls._config_data_loaded:
            return
        PropellerData.load_config_data(prop_filename='aviary/models/engines/propulsion/empirical_data/DOWTY_prop_CCA8_6blade_13ft_cleaned.xlsx', sheet_name='INFO')
        cls._KwmdDrag = PropellerData._KwmdDrag
        cls._f_slipstream = PropellerData._slipstream_data
        cls._config_data_loaded = True
        elapsed = time.time() - start_load_config_data
        print(f"  -> Config data loaded in {elapsed:.1f} seconds")

    @classmethod
    def _load_from_cache(cls):
        """
        Load pre-computed grid data from cache file.
        Returns True if cache was loaded successfully, False otherwise.
        """
        import os
        
        if not os.path.exists(cls._cache_file):
            return False
        
        try:
            start_cache_load = time.time()
            print(f"Loading propeller grid from cache: {cls._cache_file}")
            data = np.load(cls._cache_file)
            
            # Load all cached arrays
            cls._vectJ = data['vectJ']
            cls._vectCp = data['vectCp']
            cls._vectCt = data['vectCt']
            cls._vecttipMach = data['vecttipMach']
            cls._mapEta_from_Cp = data['mapEta_from_Cp']
            cls._mapEta_from_Ct = data['mapEta_from_Ct']
            cls._mapEta_from_tipMach = data['mapEta_from_tipMach']
            cls._min_power_W = float(data['min_power_W'])
            cls._max_power_W = float(data['max_power_W'])
            cls._min_thrust_N = float(data['min_thrust_N'])
            cls._max_thrust_N = float(data['max_thrust_N'])
            cls._min_Cp = float(data['min_Cp'])
            cls._max_Cp = float(data['max_Cp'])
            cls._min_Ct = float(data['min_Ct'])
            cls._max_Ct = float(data['max_Ct'])
            cls._min_tipMach = float(data['min_tipMach'])
            cls._max_tipMach = float(data['max_tipMach'])
            cls._min_J = float(data['min_J'])
            cls._max_J = float(data['max_J'])
            
            print("  -> Cache loaded successfully!")
            elapsed = time.time() - start_cache_load
            print(f"  -> Cache loaded in {elapsed:.1f} seconds")
            return True
            
        except Exception as e:
            print(f"  -> Failed to load cache: {e}")
            return False
    
    @classmethod
    def _save_to_cache(cls):
        """
        Save computed grid data to cache file for fast loading next time.
        """
        import os
        
        # Ensure directory exists
        cache_dir = os.path.dirname(cls._cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        print(f"Saving propeller grid to cache: {cls._cache_file}")
        np.savez(cls._cache_file,
                 vectJ=cls._vectJ,
                 vectCp=cls._vectCp,
                 vectCt=cls._vectCt,
                 vecttipMach=cls._vecttipMach,
                 mapEta_from_Cp=cls._mapEta_from_Cp,
                 mapEta_from_Ct=cls._mapEta_from_Ct,
                 mapEta_from_tipMach=cls._mapEta_from_tipMach,
                 min_power_W=cls._min_power_W,
                 max_power_W=cls._max_power_W,
                 min_thrust_N=cls._min_thrust_N,
                 max_thrust_N=cls._max_thrust_N,
                 min_Cp=cls._min_Cp,
                 max_Cp=cls._max_Cp,
                 min_Ct=cls._min_Ct,
                 max_Ct=cls._max_Ct,
                 min_tipMach=cls._min_tipMach,
                 max_tipMach=cls._max_tipMach,
                 min_J=cls._min_J,
                 max_J=cls._max_J)
        print("  -> Cache saved successfully!")

    @classmethod
    def _prepare_grid_data(cls):

        if cls._grid_data_built:
            return
        
        # Try to load from cache first (fast path)
        if cls._load_from_cache():
            cls._grid_data_built = True
            return
        
        # Cache miss - compute the grid (slow path, only happens once)
        print("Computing propeller structured grid (one-time operation)...")
        print("  This may take 30-60 seconds...")
        
        import time
        start_time = time.time()
        
        J_data = PropellerData.dyn_J_data
        Cp_data = PropellerData.dyn_Cp_data
        eta_data = PropellerData.dyn_eta_data/100
        Ct_data = PropellerData.dyn_Ct_data
        power_data_W = PropellerData.dyn_power_data__W
        thrust_data_N = PropellerData.dyn_thrust_data__N
        tipMach_data = PropellerData.dyn_tip_mach_data
        min_power_W = np.min(power_data_W)
        max_power_W = np.max(power_data_W)
        min_thrust_N = np.min(thrust_data_N)
        max_thrust_N = np.max(thrust_data_N)

        minJ = np.min(J_data)
        maxJ = np.max(J_data)
        minCp = np.min(Cp_data)
        maxCp = np.max(Cp_data)
        minCt = np.min(Ct_data)
        maxCt = np.max(Ct_data)
        mintipMach = np.min(tipMach_data)
        maxtipMach = np.max(tipMach_data)

        nJ = 100
        nCp = 100
        nCt = 100
        ntipMach = 100

        vectJ = np.linspace(minJ,maxJ,nJ)
        vectCp = np.linspace(minCp,maxCp,nCp)
        vectCt = np.linspace(minCt,maxCt,nCt)
        vecttipMach = np.linspace(mintipMach,maxtipMach,ntipMach)

        mapEta_from_Cp = np.zeros((nJ,nCp)) 
        mapEta_from_Ct = np.zeros((nJ,nCt)) 
        mapEta_from_tipMach = np.zeros((nJ,ntipMach)) 

        print("  Computing Cp-based eta grid...")
        J_grid, Cp_grid, tipMach_grid = np.meshgrid(vectJ, vectCp, vecttipMach, indexing='ij')
        grid_points_Cp = np.column_stack([J_grid.flatten(), Cp_grid.flatten(), tipMach_grid.flatten()])
        mapEta_flat_Cp = griddata(np.column_stack([J_data, Cp_data, tipMach_data]), eta_data, grid_points_Cp, method='linear', fill_value=0.3)
        mapEta_from_Cp = mapEta_flat_Cp.reshape(len(vectJ), len(vectCp), len(vecttipMach))

        print("  Computing Ct-based eta grid...")
        J_grid, Ct_grid, tipMach_grid = np.meshgrid(vectJ, vectCt, vecttipMach, indexing='ij')
        grid_points_Ct = np.column_stack([J_grid.flatten(), Ct_grid.flatten(), tipMach_grid.flatten()])
        mapEta_flat_Ct = griddata(np.column_stack([J_data, Ct_data, tipMach_data]), eta_data, grid_points_Ct, method='linear', fill_value=0.3)
        mapEta_from_Ct = mapEta_flat_Ct.reshape(len(vectJ), len(vectCt), len(vecttipMach))

        cls._vectJ = vectJ
        cls._vectCp = vectCp
        cls._vectCt = vectCt
        cls._vecttipMach = vecttipMach
        cls._mapEta_from_Cp = mapEta_from_Cp
        cls._mapEta_from_Ct = mapEta_from_Ct
        cls._mapEta_from_tipMach = mapEta_from_tipMach
        cls._min_power_W = min_power_W
        cls._max_power_W = max_power_W
        cls._min_thrust_N = min_thrust_N
        cls._max_thrust_N = max_thrust_N
        cls._min_Cp = minCp
        cls._max_Cp = maxCp
        cls._min_Ct = minCt
        cls._max_Ct = maxCt
        cls._min_tipMach = mintipMach
        cls._max_tipMach = maxtipMach

        cls._min_J = minJ + 3e-6
        cls._max_J = maxJ - 3e-6
        
        elapsed = time.time() - start_time
        print(f"  Grid computation completed in {elapsed:.1f} seconds")
        
        # Save to cache for next time
        cls._save_to_cache()

        cls._grid_data_built = True

    def setup(self):

        #print("Setting up Propeller")
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        power_set = self.options['power_set']
        thrust_set = self.options['thrust_set']
        rpm_set = self.options['rpm_set']
        test_interp = self.options['test_interp']



        if not rpm_set:

            #self.set_input_defaults('rpm_schedule_interp.power', val = 1500.0 * np.ones(nn), units = 'kW')
            # Convert matrix power input to vector for RPM schedule MetaModel
            self.add_subsystem('matrix_to_vector_rpm_power', 
                              MatrixToVectorConverter(
                                  num_nodes=nn, 
                                  num_comps=npp,
                                  input_names=['power'],
                                  units={'power': 'W'}
                              ), 
                              promotes_inputs=[])


            # Connect the vector output from matrix_to_vector through clipper to MetaModel input
            self.connect('matrix_to_vector_rpm_power.power_vect', 'lim_min_power.input_array')
            self.connect('lim_min_power.output', 'lim_max_power.input_array')
            self.connect('lim_max_power.output', 'rpm_schedule_interp.power')

            self.add_subsystem('lim_min_power', SmoothMaxComp(num_nodes=nn * npp,  mode='limit', units='W', limit_val=self._min_power_W, n_comps=1), promotes_inputs=[], promotes_outputs=[])
            self.add_subsystem('lim_max_power', SmoothMinComp(num_nodes=nn * npp, mode='limit', units='W', limit_val=self._max_power_W, n_comps=1), promotes_inputs=[], promotes_outputs=[])
            #self.add_subsystem('lim_min_power', SmoothMaxComp(num_nodes=nn * npp,  mode='limit', units='W', limit_val=self._min_power_W, n_comps=1, expected_min=self._min_power_W, expected_max=self._max_power_W), promotes_inputs=[], promotes_outputs=[])
            #self.add_subsystem('lim_max_power', SmoothMinComp(num_nodes=nn * npp, mode='limit', units='W', limit_val=self._max_power_W, n_comps=1, expected_min=self._min_power_W, expected_max=self._max_power_W), promotes_inputs=[], promotes_outputs=[])


            # Create RPM schedule metamodel
            rpm_schedule_interp = om.MetaModelStructuredComp(vec_size=nn * npp, method='akima')
            rpm_schedule_interp.add_input('power', 1000.0, training_data=PropellerData.rpm_sched_data__power_W, units='W')
            rpm_schedule_interp.add_output('rpm', 1000.0, training_data=PropellerData.rpm_sched_data__rpm, units='rpm', upper = np.max(PropellerData.rpm_sched_data__rpm), lower = np.min(PropellerData.rpm_sched_data__rpm))
            rpm_schedule_interp.options['extrapolate'] = True
            self.add_subsystem('rpm_schedule_interp', rpm_schedule_interp, promotes_inputs=[], promotes_outputs=[])

            # Convert vector output back to matrix
            self.add_subsystem('vector_to_matrix_rpm', 
                              VectorToMatrixConverter(
                                  num_nodes=nn, 
                                  num_comps=npp,
                                  input_names=['rpm_vect'],
                                  output_names=['rpm'],
                                  units={'rpm_vect': 'rpm'}
                              ), 
                              promotes_outputs=['rpm'])

            
            # Connect the vector output from MetaModel to vector_to_matrix input
            self.connect('rpm_schedule_interp.rpm', 'vector_to_matrix_rpm.rpm_vect')

        if not test_interp:            
            self.add_subsystem('advance_ratio', AdvanceRatio(num_nodes=nn, num_props=npp, min_J=self._min_J, max_J=self._max_J), promotes = ['*'])
            self.add_subsystem('tip_mach', tipMach(num_nodes=nn, num_props=npp), promotes = ['*'])

        if power_set:

            if not test_interp:

                self.add_subsystem('power_coefficient', PowerCoefficient(num_nodes=nn, num_props=npp, min_Cp=self._min_Cp, max_Cp=self._max_Cp), promotes = ['*'])

                # Convert matrix inputs to vectors for MetaModel
                self.add_subsystem('matrix_to_vector_cp', 
                                MatrixToVectorConverter(
                                    num_nodes=nn, 
                                    num_comps=npp,  # Single propeller case
                                    input_names=['adv_ratio', 'Cp', 'tipMach'],
                                    units={'adv_ratio': None, 'Cp': None, 'tipMach': None}
                                ), 
                                promotes_inputs=['adv_ratio', 'Cp', 'tipMach'])

                self.connect('matrix_to_vector_cp.Cp_vect', 'lim_min_Cp.input_array')
                self.connect('lim_min_Cp.output', 'lim_max_Cp.input_array')
                self.connect('lim_max_Cp.output', 'Cp_eta_interp.Cp_vect')

                self.add_subsystem('lim_min_Cp', SmoothMaxComp(num_nodes=nn * npp,  mode='limit', units=None, limit_val=self._min_Cp, n_comps=1), promotes_inputs=[], promotes_outputs=[])
                self.add_subsystem('lim_max_Cp', SmoothMinComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=self._max_Cp, n_comps=1), promotes_inputs=[], promotes_outputs=[])
                #self.add_subsystem('lim_min_Cp', SmoothMaxComp(num_nodes=nn * npp,  mode='limit', units=None, limit_val=self._min_Cp, n_comps=1, expected_min=self._min_Cp, expected_max=self._max_Cp), promotes_inputs=[], promotes_outputs=[])
                #self.add_subsystem('lim_max_Cp', SmoothMinComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=self._max_Cp, n_comps=1, expected_min=self._min_Cp, expected_max=self._max_Cp), promotes_inputs=[], promotes_outputs=[])

            
            # Create regular grid interpolator instance with vectorization
            Cp_eta_interp = om.MetaModelStructuredComp(vec_size=nn * npp, method='scipy_quintic')
            
            # set up inputs and outputs with vectorization using global data
            Cp_eta_interp.add_input('adv_ratio_vect',1.5, training_data= self._vectJ)
            Cp_eta_interp.add_input('Cp_vect', 0.02, training_data=self._vectCp)
            Cp_eta_interp.add_input('tipMach_vect', 0.9, training_data=self._vecttipMach)
            Cp_eta_interp.add_output('eta_calc_vect', 0.8, training_data=self._mapEta_from_Cp, units=None)
            Cp_eta_interp.options['extrapolate'] = True
            self.add_subsystem('Cp_eta_interp', Cp_eta_interp, promotes_inputs=[], promotes_outputs=[])

            if not test_interp:
                # Convert vector output back to matrix
                self.add_subsystem('vector_to_matrix_cp', 
                                VectorToMatrixConverter(
                                    num_nodes=nn, 
                                    num_comps=npp,  # Single propeller case
                                    input_names=['eta_calc_vect'],
                                    output_names=['eta_calc'],
                                    units={'eta_calc_vect': None}
                                ), 
                                promotes_outputs=['eta_calc'])


                self.connect('lim_min_eta.output', 'lim_max_eta.input_array')

                self.add_subsystem('lim_min_eta', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=1e-6, n_comps=npp), promotes_inputs=[('input_array','eta_calc')], promotes_outputs=[])
                self.add_subsystem('lim_max_eta', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1.0 - 1e-6, n_comps=npp), promotes_inputs=[], promotes_outputs=[('output', 'eta')])
                #self.add_subsystem('lim_min_eta', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=1e-6, n_comps=npp, expected_min=1e-6, expected_max=1.0 - 1e-6), promotes_inputs=[('input_array','eta_calc')], promotes_outputs=[])
                #self.add_subsystem('lim_max_eta', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1.0 - 1e-6, n_comps=npp, expected_min=1e-6, expected_max=1.0 - 1e-6), promotes_inputs=[], promotes_outputs=[('output', 'eta')])

                # Connect the vector outputs from matrix_to_vector through clipper to MetaModel inputs
                self.connect('matrix_to_vector_cp.adv_ratio_vect', 'Cp_eta_interp.adv_ratio_vect')
                self.connect('matrix_to_vector_cp.tipMach_vect', 'Cp_eta_interp.tipMach_vect')
        
                
                # Connect the vector output from MetaModel to vector_to_matrix input
                self.connect('Cp_eta_interp.eta_calc_vect', 'vector_to_matrix_cp.eta_calc_vect')

                self.add_subsystem("compute_thrust", ComputeThrustPowerRelation(num_nodes=nn, num_props=npp, direction='power_in'), 
                                promotes_inputs=["eta", "Utrue","power"], promotes_outputs=[])

                # Add windmilling drag component
                self.add_subsystem('windmilling_drag', 
                                WindmillingDrag(num_nodes=nn, num_props=npp,
                                            wmill_thresh_hp=self.options['wmill_thresh_hp'],
                                            kwmdDrag=self._KwmdDrag,
                                            f_slipstream=self._f_slipstream,thrust_set =thrust_set),
                                promotes_inputs=['power', 'rho', 'Utrue'],
                                promotes_outputs=['*'])
                
                # Connect thrust_calc to windmilling_drag input
                self.connect('compute_thrust.thrust_calc', 'windmilling_drag.thrust_in')

        if thrust_set:

            if not test_interp: 
                self.add_subsystem('windmilling_drag', 
                                WindmillingDrag(num_nodes=nn, num_props=npp,
                                            wmill_thresh_hp=self.options['wmill_thresh_hp'],
                                            kwmdDrag=self._KwmdDrag,
                                            f_slipstream=self._f_slipstream,
                                            thrust_set = thrust_set),
                                promotes_inputs= ['rho', 'Utrue'],
                                promotes_outputs=[])

                self.connect('power_calc', ['windmilling_drag.power'])
            if not rpm_set:
                self.connect('power_calc', 'matrix_to_vector_rpm_power.power')

            if not test_interp: 

                # Connect thrust to windmilling_drag input
                self.connect('windmilling_drag.thrust_calc', ['thrust_coefficient.thrust','compute_power.thrust'])

                self.add_subsystem('thrust_coefficient', ThrustCoefficient(num_nodes=nn, num_props=npp, min_Ct=self._min_Ct, max_Ct=self._max_Ct), 
                promotes_inputs=["rpm", "diameter", "rho"], promotes_outputs=["*"])

                # Convert matrix inputs to vectors for MetaModel
                self.add_subsystem('matrix_to_vector_ct', 
                                MatrixToVectorConverter(
                                    num_nodes=nn, 
                                    num_comps=npp,  # Single propeller case
                                    input_names=['adv_ratio', 'Ct'],
                                    units={'adv_ratio': None, 'Ct': None}
                                ), 
                                promotes_inputs=['adv_ratio', 'Ct'])


            # Create regular grid interpolator instance with vectorization
            Ct_eta_interp = om.MetaModelStructuredComp(vec_size=nn * npp, method='scipy_quintic')
            
            # set up inputs and outputs with vectorization using global data
            Ct_eta_interp.add_input('adv_ratio_vect',1.5, training_data= self._vectJ)
            Ct_eta_interp.add_input('Ct_vect', 0.02, training_data=self._vectCt)
            Ct_eta_interp.add_input('tipMach_vect', 0.9, training_data=self._vecttipMach)
            Ct_eta_interp.add_output('eta_calc_vect', 0.8, training_data=self._mapEta_from_Ct, units=None, lower = 1e-6, upper = 1.0 - 1e-6)
            Ct_eta_interp.options['extrapolate'] = False
            self.add_subsystem('Ct_eta_interp', Ct_eta_interp, promotes_inputs=[], promotes_outputs=[])

            if not test_interp:

                # Convert vector output back to matrix
                self.add_subsystem('vector_to_matrix_ct', 
                                VectorToMatrixConverter(
                                    num_nodes=nn, 
                                    num_comps=npp,  # Single propeller case
                                    input_names=['eta_calc_vect'],
                                    output_names=['eta_calc'],
                                    units={'eta_calc_vect': None}
                                ), 
                                promotes_outputs=['eta_calc'])


                self.connect('lim_min_eta.output', 'lim_max_eta.input_array')

                self.add_subsystem('lim_min_eta', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=1e-6, n_comps=npp), promotes_inputs=[('input_array','eta_calc')], promotes_outputs=[])
                self.add_subsystem('lim_max_eta', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1.0 - 1e-6, n_comps=npp), promotes_inputs=[], promotes_outputs=[('output', 'eta')])
                #self.add_subsystem('lim_min_eta', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=1e-6, n_comps=npp, expected_min=1e-6, expected_max=1.0 - 1e-6), promotes_inputs=[('input_array','eta_calc')], promotes_outputs=[])
                #self.add_subsystem('lim_max_eta', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1.0 - 1e-6, n_comps=npp, expected_min=1e-6, expected_max=1.0 - 1e-6), promotes_inputs=[], promotes_outputs=[('output', 'eta')])

                # Connect the vector outputs from matrix_to_vector to MetaModel inputs
                self.connect('matrix_to_vector_ct.adv_ratio_vect', 'Ct_eta_interp.adv_ratio_vect')
                self.connect('matrix_to_vector_ct.Ct_vect', 'Ct_eta_interp.Ct_vect')
                self.connect('matrix_to_vector_ct.tipMach_vect', 'Ct_eta_interp.tipMach_vect')
                
                # Connect the vector output from MetaModel to vector_to_matrix input
                self.connect('Ct_eta_interp.eta_calc_vect', 'vector_to_matrix_ct.eta_calc_vect')

                self.add_subsystem("compute_power", ComputeThrustPowerRelation(num_nodes=nn, num_props=npp, direction='thrust_in'), 
                                promotes_inputs=["eta","Utrue"], promotes_outputs=["*"])

        # end 

        if power_set and not test_interp:
            self.set_input_defaults('power', val = 1000, units = 'kW')

    # end 




        
def run_prop_model(test_interp = False, thrust_set = False, power_set = True, rpm_set = True,plot_results = False
):

    import time
    start_time = time.time()


    nn = 5
    npp = 2

    model = om.Group()

    # Accelerating velocity profile (start at 50 m/s, end at 150 m/s)
    velocity_profile = np.linspace(243, 252, nn) * 0.5144

    # Convert atmosphere to density 
    alt_m =  np.linspace(14500, 15000, nn) * 0.3048
    temp_degC = 15.04 - 0.00649 * alt_m
    press_kPa = 101.29 * ((temp_degC + 273.15) / 288.08) ** 5.256
    rho_kgpm3 = press_kPa / (0.2869 * (temp_degC + 273.15))

    a = np.sqrt(1.4*287*(temp_degC+273.15))


    # Calculate thrust required
    cd0 = 0.03
    sref_m2 = 913 * 0.3048**2
    q_inf = 0.5 * rho_kgpm3 * velocity_profile**2
    weight_N = 37009 * 9.81
    CL = 0.5 * weight_N / (q_inf * sref_m2)
    wingspan_m = 111 * 0.3048
    ar = wingspan_m**2 / sref_m2
    e = 0.8
    cd = cd0 + CL**2 / (np.pi * ar * e)
    drag_N = cd * q_inf * sref_m2
    num_engines = 4
    thrust_profile_unit = drag_N / num_engines* 0
    
    # RPM
    #rpm_constant = 1000.0
    rpm_profile = 997
    
    # power
    power_profile = np.linspace(850e3, 850e3, nn)


    ivc = om.IndepVarComp()
    if rpm_set:
        ivc.add_output('rpm', val = rpm_profile * np.ones((npp, nn)), units = 'rpm')
    if power_set:
        ivc.add_output('power', val = power_profile * np.ones((npp, nn)), units = 'W')
    if thrust_set:
        ivc.add_output('thrust', val = thrust_profile_unit * np.ones((npp, nn)), units = 'N')

    ivc.add_output('diameter', val = 13 * 0.3048 * np.ones((npp, nn)), units = 'm')
    ivc.add_output('Utrue', val = velocity_profile, units = 'm/s')
    ivc.add_output('rho', val = rho_kgpm3, units = 'kg/m**3')
    ivc.add_output('a', val = a, units = 'm/s')

    model.add_subsystem('ivc', ivc, promotes = ['*'])
    model.add_subsystem('solve_propeller', EmpiricalPropellerCoeffMMMtip(num_nodes=nn, 
                                                            num_props=npp,
                                                          thrust_set = thrust_set, 
                                                          power_set = power_set, 
                                                          rpm_set = rpm_set), promotes = ['*'])

    prob = om.Problem(model, reports=False)


    if thrust_set:
        prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver.options["maxiter"] = 100
        prob.model.nonlinear_solver.options["atol"] = 1e-6
        prob.model.nonlinear_solver.options["rtol"] = 1e-6
        prob.model.connect('thrust', 'windmilling_drag.thrust_in')
    # end

    if power_set and not rpm_set:
        prob.model.connect('power', ['matrix_to_vector_rpm_power.power'])


    prob.setup()
    om.n2(prob)
    run_start_time = time.time()

    
    #om.n2(prob)
    #prob.check_partials(compact_print=True)
    prob.run_model()
    end_time = time.time()
    print(f"Setup time: {run_start_time - start_time:.2f} seconds")
    print(f"Execution time: {end_time - run_start_time:.2f} seconds")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")

    #prob.run_driver()
    nodes = np.arange(0, nn)

    # Get results
    J_profile = prob.get_val('adv_ratio').flatten()
    
    rpm_profile = prob.get_val('rpm').flatten()
    eta_profile = prob.get_val('eta').flatten()

    if thrust_set:
        thrust_calc_profile = prob.get_val('thrust').flatten()
        power_back_calc_kW = thrust_calc_profile/0.98 * velocity_profile / eta_profile / 1000
        back_calc_error_kW = np.abs(power_back_calc_kW - power_profile)
        print(f"Back Calculation Error Power: {back_calc_error_kW} kW")
    else:
        thrust_calc_profile = prob.get_val('thrust_calc').flatten()

    if power_set:
        power_profile = prob.get_val('power', units='kW').flatten()
        thrust_back_calc_N = eta_profile * power_profile*1000 / np.tile(velocity_profile, npp)
        back_calc_error_N = np.abs(thrust_back_calc_N - thrust_calc_profile/0.98)
        print(f"Back Calculation Error Thrust: {back_calc_error_N} N")
    else:
        power_profile = prob.get_val('power_calc', units='kW').flatten()

    try:
        eta_profile = prob.get_val('eta').flatten()
    except:
        eta_profile = thrust_calc_profile * velocity_profile / (power_profile*1000)
    #end


    if plot_results:

        if test_interp:
            j_val = prob.get_val('adv_ratio').flatten()
            tipMach_val = prob.get_val('tipMach').flatten()

            if thrust_set: 
                ct_val = prob.get_val('Ct').flatten()
                J_ct_interpolator_nearest = NearestNDInterpolator(np.column_stack([PropellerData.dyn_J_data, PropellerData.dyn_Ct_data, PropellerData.dyn_tip_mach_data]), PropellerData.dyn_eta_data)
                eta_nearest = J_ct_interpolator_nearest(j_val, ct_val, tipMach_val)/ 100
                power_calc_nearest = thrust_calc_profile * np.tile(velocity_profile, npp) / eta_nearest / 1000
            elif power_set:
                cp_val = prob.get_val('Cp').flatten()
                J_cp_interpolator_nearest = NearestNDInterpolator(np.column_stack([PropellerData.dyn_J_data, PropellerData.dyn_Cp_data, PropellerData.dyn_tip_mach_data]), PropellerData.dyn_eta_data)
                eta_nearest = J_cp_interpolator_nearest(j_val, cp_val, tipMach_val) / 100
                thrust_calc_nearest = eta_nearest * power_profile*1000 / np.tile(velocity_profile, npp)
            # end
        # end

        
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Velocity profile
        plt.subplot(3, 3, 1)
        plt.plot(nodes, velocity_profile * 1.944, 'b-', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Velocity (kts)')
        plt.title('Acceleration Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Altitude profile
        plt.subplot(3, 3, 2)
        plt.plot(nodes, alt_m/0.3048, 'g-', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Altitude (ft)')
        plt.title('Altitude Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: RPM profile
        plt.subplot(3, 3, 3)
        for i in range(npp):
            plt.plot(nodes, prob.get_val('rpm')[i,:], 'r-', linewidth=2, marker='o', label='Interpolated')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('RPM')
        plt.title('RPM Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Power profile
        plt.subplot(3, 3, 4)
        for i in range(npp):
            plt.plot(nodes, prob.get_val('power')[i,:], 'purple', linewidth=2, marker='o', alpha=1.0, label=f'Metamodel Prop{i+1}')
        plt.legend()
        if test_interp and thrust_set:
            for i in range(npp):
                plt.plot(nodes, power_calc_nearest[i*len(nodes):(i+1)*len(nodes)], 'purple', linewidth=2, marker='o', alpha=0.2, label=f'Nearest Neighbour Prop{i+1}')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('Power (kW)')
        plt.title('Power Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Thrust profile
        plt.subplot(3, 3, 5)
        for i in range(npp):
            plt.plot(nodes, prob.get_val('thrust_calc')[i,:]/1000, 'orange', linewidth=2, marker='o', alpha=1.0, label=f'Metamodel Prop{i+1}')
        plt.legend()
        if test_interp and power_set:
            for i in range(npp):
                plt.plot(nodes, thrust_calc_nearest[i*len(nodes):(i+1)*len(nodes)]/1000, 'orange', linewidth=2, marker='o', alpha=0.2, label=f'Nearest Neighbour Prop{i+1}')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('Thrust (kN)')
        plt.title('Thrust Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Advance ratio profile
        plt.subplot(3, 3, 6)
        for i in range(npp):
            plt.plot(nodes, prob.get_val('adv_ratio')[i,:], 'brown', linewidth=2, marker='o')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('Advance Ratio (J)')
        plt.title('Advance Ratio Profile')
        plt.grid(True, alpha=0.3)

        # Plot 7: Efficiency profile
        plt.subplot(3, 3, 7)
        for i in range(npp):
            plt.plot(nodes, prob.get_val('eta')[i,:], 'cyan', linewidth=2, marker='o', label='Metamodel')
        if test_interp:
            for i in range(npp):
                plt.plot(nodes, eta_nearest[i*len(nodes):(i+1)*len(nodes)], 'cyan', linewidth=2, marker='o', alpha=0.2, label='Nearest Neighbour')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('Efficiency (%)')
        plt.title('Propeller Efficiency Profile')
        plt.grid(True, alpha=0.3)
        

        plt.tight_layout()
        plt.show()
    # end 



def _get_propeller_interpolation_results():
    """
    Helper function to get propeller interpolation results for testing.
    Uses EmpiricalPropellerCoeffMMMtip twice: once for power_set and once for thrust_set.
    Returns: interpolated_values_Cp, interpolated_values_Ct, actual_values, relative_errors_Cp, relative_errors_Ct
    """
    # Get the data
    prop_data = PropellerData.get_data(prop_filename='aviary/models/engines/propulsion/empirical_data/DOWTY_prop_CCA8_6blade_13ft_cleaned.xlsx', sheet_name='data')
    
    # Test with a subset of data points to avoid overfitting test
    # Use every 10th point to create test data that's different from training
    test_indices = np.arange(0, len(prop_data.dyn_eta_data), 10)
    nn = len(test_indices)
    npp = 1  # Single propeller for testing
    
    # Get test data
    J_test = prop_data.dyn_J_data[test_indices]
    cp_test = prop_data.dyn_Cp_data[test_indices]
    ct_test = prop_data.dyn_Ct_data[test_indices]
    tipMach_test = prop_data.dyn_tip_mach_data[test_indices]
    eta_test = prop_data.dyn_eta_data[test_indices] / 100
    
    # Get propeller diameter from config
    PropellerData.load_config_data(prop_filename='aviary/models/engines/propulsion/empirical_data/DOWTY_prop_CCA8_6blade_13ft_cleaned.xlsx', sheet_name='INFO')

    
    # ===== Test 1: power_set=True (Cp-based interpolation) =====
    model_power = om.Group()
    ivc_power = om.IndepVarComp()
    
    # Reshape to (npp, nn) format expected by EmpiricalPropellerCoeffMMMtip
    ivc_power.add_output('Cp', cp_test, units=None, desc='Power coefficient')
    ivc_power.add_output('adv_ratio', J_test, units=None, desc='Advance ratio')
    ivc_power.add_output('tipMach', tipMach_test, units=None, desc='Tip Mach')


    model_power.add_subsystem('ivc', ivc_power, promotes=['*'])
    model_power.add_subsystem('prop', EmpiricalPropellerCoeffMMMtip(
        num_nodes=nn, 
        num_props=npp, 
        power_set=True, 
        thrust_set=False, 
        rpm_set=True,
        test_interp=True
    ), promotes=['*'])

    model_power.connect('Cp', 'Cp_eta_interp.Cp_vect')
    model_power.connect('adv_ratio', 'Cp_eta_interp.adv_ratio_vect')
    model_power.connect('tipMach', 'Cp_eta_interp.tipMach_vect')

    prob_power = om.Problem(model_power, reports=False)


    prob_power.setup()
    om.n2(prob_power)
    prob_power.run_model()
    
    # Get Cp-based efficiency results
    interp_eta_Cp = prob_power.get_val('Cp_eta_interp.eta_calc_vect').flatten()
    
    # ===== Test 2: thrust_set=True (Ct-based interpolation) =====
    model_thrust = om.Group()
    ivc_thrust = om.IndepVarComp()
    
    # Reshape to (npp, nn) format expected by EmpiricalPropellerCoeffMMMtip
    ivc_thrust.add_output('Ct', ct_test, units=None, desc='Thrust coefficient')
    ivc_thrust.add_output('adv_ratio', J_test, units=None, desc='Advance ratio')
    ivc_thrust.add_output('tipMach', tipMach_test, units=None, desc='Tip Mach')
    
    model_thrust.add_subsystem('ivc', ivc_thrust, promotes=['*'])
    model_thrust.add_subsystem('prop', EmpiricalPropellerCoeffMMMtip(
        num_nodes=nn, 
        num_props=npp, 
        power_set=False, 
        thrust_set=True, 
        rpm_set=True,
        test_interp=True
    ), promotes=['*'])
    
    model_thrust.connect('Ct', 'Ct_eta_interp.Ct_vect')
    model_thrust.connect('adv_ratio', 'Ct_eta_interp.adv_ratio_vect')
    model_thrust.connect('tipMach', 'Ct_eta_interp.tipMach_vect')

    prob_thrust = om.Problem(model_thrust, reports=False)

    


    prob_thrust.setup()
    om.n2(prob_thrust)
    prob_thrust.run_model()
    
    # Get Ct-based efficiency results
    interp_eta_Ct = prob_thrust.get_val('Ct_eta_interp.eta_calc_vect').flatten()
    
    # Calculate statistics
    errors_Cp = np.abs(interp_eta_Cp - eta_test)
    errors_Ct = np.abs(interp_eta_Ct - eta_test)
    relative_errors_Cp = errors_Cp / np.maximum(eta_test, 1e-6)  # Avoid division by zero
    relative_errors_Ct = errors_Ct / np.maximum(eta_test, 1e-6)  # Avoid division by zero
    
    return interp_eta_Cp, interp_eta_Ct, eta_test, relative_errors_Cp, relative_errors_Ct


def test_cp_interpolation_max_error(tolerance=1.0):
    """Test Cp-based interpolation maximum relative error"""
    _, _, _, relative_errors_Cp, _ = _get_propeller_interpolation_results()
    max_relative_error = np.max(relative_errors_Cp)
    print(f"Cp interpolation max relative error (%): {max_relative_error:.6f}")
    assert max_relative_error < tolerance, f"Cp interpolation max relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"


def test_cp_interpolation_mean_error(tolerance=0.5):
    """Test Cp-based interpolation mean relative error"""
    _, _, _, relative_errors_Cp, _ = _get_propeller_interpolation_results()
    mean_relative_error = np.mean(relative_errors_Cp)
    print(f"Cp interpolation mean relative error (%): {mean_relative_error:.6f}")
    assert mean_relative_error < tolerance, f"Cp interpolation mean relative error {mean_relative_error:.6f} exceeds tolerance {tolerance}"


def test_ct_interpolation_max_error(tolerance=1.0):
    """Test Ct-based interpolation maximum relative error"""
    _, _, _, _, relative_errors_Ct = _get_propeller_interpolation_results()
    max_relative_error = np.max(relative_errors_Ct)
    print(f"Ct interpolation max relative error (%): {max_relative_error:.6f}")
    assert max_relative_error < tolerance, f"Ct interpolation max relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"


def test_ct_interpolation_mean_error(tolerance=0.5):
    """Test Ct-based interpolation mean relative error"""
    _, _, _, _, relative_errors_Ct = _get_propeller_interpolation_results()
    mean_relative_error = np.mean(relative_errors_Ct)
    print(f"Ct interpolation mean relative error (%): {mean_relative_error:.6f}")
    assert mean_relative_error < tolerance, f"Ct interpolation mean relative error {mean_relative_error:.6f} exceeds tolerance {tolerance}"


def test_propeller_efficiency_bounds():
    """Test that interpolated efficiency values are within reasonable bounds"""
    interpolated_values_Cp, interpolated_values_Ct, _, _, _ = _get_propeller_interpolation_results()
    assert np.all(interpolated_values_Cp >= 0.0), f"Some Cp-based efficiency values are negative: min = {np.min(interpolated_values_Cp)}"
    assert np.all(interpolated_values_Cp <= 1.0), f"Some Cp-based efficiency values exceed 1.0: max = {np.max(interpolated_values_Cp)}"
    assert np.all(interpolated_values_Ct >= 0.0), f"Some Ct-based efficiency values are negative: min = {np.min(interpolated_values_Ct)}"
    assert np.all(interpolated_values_Ct <= 1.0), f"Some Ct-based efficiency values exceed 1.0: max = {np.max(interpolated_values_Ct)}"


def run_interpolation_accuracy_test(assert_test = True):
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    """
    print("Testing propeller interpolation accuracy...")
    
    # Get the interpolation results
    interpolated_values_Cp, interpolated_values_Ct, actual_values, relative_errors_Cp, relative_errors_Ct = _get_propeller_interpolation_results()
    
    # Calculate absolute errors for display
    errors_Cp = np.abs(interpolated_values_Cp - actual_values)
    errors_Ct = np.abs(interpolated_values_Ct - actual_values)
    
    print(f"Mean absolute error (Cp): {np.mean(errors_Cp * 100):.2f}%")
    print(f"Max absolute error (Cp): {np.max(errors_Cp * 100):.2f}%")
    print(f"Mean relative error (Cp): {np.mean(relative_errors_Cp * 100):.2f}%")
    print(f"Max relative error (Cp): {np.max(relative_errors_Cp * 100):.2f}%")

    print(f"Mean absolute error (Ct): {np.mean(errors_Ct * 100):.2f}%")
    print(f"Max absolute error (Ct): {np.max(errors_Ct * 100):.2f}%")
    print(f"Mean relative error (Ct): {np.mean(relative_errors_Ct * 100):.2f}%")
    print(f"Max relative error (Ct): {np.max(relative_errors_Ct * 100):.2f}%")
    
    if assert_test:
        # Call individual test functions
        test_cp_interpolation_max_error(tolerance=2.0)
        test_cp_interpolation_mean_error(tolerance=0.5)
        test_ct_interpolation_max_error(tolerance=2.0)
        test_ct_interpolation_mean_error(tolerance=0.5)
        test_propeller_efficiency_bounds()
    else:
        print("Skipping assert tests")

    
    # Create the comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot actual vs interpolated
    plt.subplot(2, 2, 1)
    plt.scatter(actual_values, interpolated_values_Cp, alpha=0.6, s=20)
    plt.scatter(actual_values, interpolated_values_Ct, alpha=0.6, s=20)
    plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Interpolated Efficiency')
    plt.title('Dynamic Conditions: Actual vs Interpolated Efficiency')
    plt.grid(True, alpha=0.3)
    
    # Plot error distribution
    plt.subplot(2, 2, 2)
    plt.hist(errors_Cp, bins=30, alpha=0.7, edgecolor='black')
    plt.hist(errors_Ct, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot relative error distribution
    plt.subplot(2, 2, 3)
    plt.hist(relative_errors_Cp, bins=30, alpha=0.7, edgecolor='black')
    plt.hist(relative_errors_Ct, bins=30, alpha=0.7, edgecolor='black')

    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot error vs actual value
    plt.subplot(2, 2, 4)
    plt.scatter(actual_values, relative_errors_Cp, alpha=0.6, s=20)
    plt.scatter(actual_values, relative_errors_Ct, alpha=0.6, s=20)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Relative Error (%)')
    plt.title('Error vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return interpolated_values_Cp, interpolated_values_Ct, actual_values, errors_Cp, errors_Ct, relative_errors_Cp, relative_errors_Ct





class WindmillingDrag(om.JaxExplicitComponent):
    """
    Component that computes windmilling drag for propellers.
    
    When power is below the windmilling threshold, this component subtracts
    windmilling drag from the input thrust:
    
    thrust_out = thrust_in - kwmdDrag * 0.5 * rho * v^2
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points
    wmill_thresh_hp : float
        Power threshold below which windmilling drag is applied (default 200 HP)
    kwmdDrag : float
        Windmilling drag coefficient (default 0.1)
    """
    _data_loaded = False

    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of props')
        self.options.declare('wmill_thresh_hp', default=200.0, 
                           desc='Power threshold for windmilling drag (HP)')
        self.options.declare('kwmdDrag', default=0.2415, 
                           desc='Windmilling drag coefficient')
        self.options.declare('f_slipstream', default=0.98,desc='Slipstream factor')               
        self.options.declare('thrust_set', default=False, desc='Thrust is set (input). If False, power is set (input), and trhust is an output.')


    


    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        thrust_set = self.options['thrust_set']
        
        # Inputs
        self.add_input('power', units='hp', shape=(npp,nn), 
                      desc='Power input to propeller')
        self.add_input('thrust_in', units='N', shape=(npp,nn), 
                      desc='Thrust input to propeller')
        self.add_input('rho', units='kg/m**3', shape=(nn,), 
                      desc='Air density')
        self.add_input('Utrue', units='m/s', shape=(nn), 
                      desc='True airspeed')
        
        # Outputs
        self.add_output('thrust_calc', units='N', shape=(npp,nn), 
                       desc='Output thrust adjusted for windmilling drag')
        
        # Declare partials with rows and cols
        n = npp * nn
        rows = np.arange(n)
        # For matrix inputs (npp, nn): diagonal structure
        cols_power = np.arange(n)
        cols_thrust_in = np.arange(n)
        # For vector inputs (nn,) that broadcast: each output element (i,j) depends on input[j]
        # Flattened array: [thrust_calc[0,0], thrust_calc[0,1], ..., thrust_calc[0,nn-1], thrust_calc[1,0], ...]
        # So cols should be: [0, 1, 2, ..., nn-1, 0, 1, 2, ..., nn-1, ...]
        cols_rho = np.tile(np.arange(nn), npp)
        cols_Utrue = np.tile(np.arange(nn), npp)
        #self.declare_partials('thrust_calc', 'power', rows=rows, cols=cols_power, method='exact')
        #self.declare_partials('thrust_calc', 'thrust_in', rows=rows, cols=cols_thrust_in, method='exact')
        #self.declare_partials('thrust_calc', 'rho', rows=rows, cols=cols_rho, method='exact')
        #self.declare_partials('thrust_calc', 'Utrue', rows=rows, cols=cols_Utrue, method='exact')
    
    def compute_primal(self, power, thrust_in, rho, Utrue):

        threshold_hp = self.options['wmill_thresh_hp']
        kwmd_drag = self.options['kwmdDrag']
        thrust_set = self.options['thrust_set']
        f_slipstream = self.options['f_slipstream']

        
        # Convert threshold from HP to W and determine windmilling condition
        windmilling_active = power < threshold_hp
        
        # Calculate windmilling drag
        # Dynamic pressure: q = 0.5 * rho * v^2
        dynamic_pressure = 0.5 * rho * Utrue**2
        
        # Windmilling drag force
        windmilling_drag = kwmd_drag * dynamic_pressure
        
        # Apply windmilling drag only when active
        drag_wmill = jnp.where(windmilling_active, 
                              windmilling_drag,
                             0)

        if thrust_set:
            # If thrust is calculated to balance drag, raise drag (and thus thrust required)
            thrust_calc = thrust_in/f_slipstream + drag_wmill
        else:
            # if power is the input and thrust is the output, reduce thrust generated by windmilling drag
            thrust_calc = thrust_in * f_slipstream - drag_wmill

        return thrust_calc

    def self_get_statics(self):
        return (self.options['num_nodes'], self.options['f_slipstream'], self.options['kwmdDrag'], self.options['wmill_thresh_hp'])




def regenerate_propeller_cache():
    """
    Utility function to force regeneration of the propeller grid cache.
    Call this if the propeller data file has been updated.
    
    Usage:
        from aviary.models.engines.propulsion.prop.propeller_empirical_mtip_mm import regenerate_propeller_cache
        regenerate_propeller_cache()
    """
    import os
    
    cache_file = EmpiricalPropellerCoeffMMMtip._cache_file
    
    # Delete existing cache if it exists
    if os.path.exists(cache_file):
        print(f"Deleting existing cache: {cache_file}")
        os.remove(cache_file)
    
    # Reset class state
    EmpiricalPropellerCoeffMMMtip._grid_data_built = False
    EmpiricalPropellerCoeffMMMtip._data_loaded = False
    EmpiricalPropellerCoeffMMMtip._config_data_loaded = False
    
    # Load data and regenerate cache
    EmpiricalPropellerCoeffMMMtip._load_data()
    EmpiricalPropellerCoeffMMMtip._load_config_data()
    EmpiricalPropellerCoeffMMMtip._prepare_grid_data()
    
    print("Cache regeneration complete!")


if __name__ == "__main__":
    # Test interpolation accuracy using EmpiricalPropellerCoeffMMMtip
    # run_interpolation_accuracy_test(assert_test=False)
    
    # Test Propeller Model based on Cp/Ct for Thrust 
    run_prop_model(test_interp=True, thrust_set=True, power_set=False, rpm_set=True, plot_results=True)
    
    # To regenerate cache (e.g., after updating propeller data), uncomment:
    # regenerate_propeller_cache()



