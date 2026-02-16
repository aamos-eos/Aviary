import numpy as np
import jax.numpy as jnp
import openmdao.api as om
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator, griddata

from aviary.utils.dvlabel import DVLabel
from aviary.utils.math.add_subtract_comp import AddSubtractComp
from aviary.utils.math.multiply_divide_comp import ElementMultiplyDivideComp
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
from aviary.utils.tiler import Tiler

# Import the global data store
from .propeller_data import PropellerData

from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp

Debug = True


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

        self.add_output('adv_ratio', shape=(npp, nn), lower=self.options['min_J'], upper=self.options['max_J'])
        
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


class EmpiricalPropellerMM(om.Group):
    """
    Propeller model using direct interpolation from (velocity, rpm, power) -> thrust.
    
    This model interpolates thrust directly from shaft power using a 3D metamodel
    based on advance ratio (J), RPM, and power.
    
    Inputs
    ------
    Utrue : float
        True airspeed (m/s), shape (nn,)
    rpm : float
        Propeller RPM, shape (npp, nn)
    power : float
        Shaft power (W), shape (npp, nn)
    diameter : float
        Propeller diameter (m), shape (npp, nn)
    rho : float
        Air density (kg/m^3), shape (nn,)
        
    Outputs
    -------
    thrust_calc : float
        Calculated thrust (N), shape (npp, nn)
    adv_ratio : float
        Advance ratio J, shape (npp, nn)
    """

    _grid_data_built = False
    _config_data_loaded = False
    _data_loaded = False

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_props', default=4, desc='number of props')
        self.options.declare('rpm_set', default=False, desc='the rpm is set (input), if False use RPM schedule')
        self.options.declare('wmill_thresh_hp', default=200.0, desc='Power threshold for windmilling drag (HP)')
        self.options.declare('kwmdDrag', default=0.1, desc='Windmilling drag coefficient')
        
        self._load_data()
        self._load_config_data()
        self._prepare_grid_data()

    @classmethod
    def _load_data(cls):
        if cls._data_loaded:
            return
        PropellerData.load_data(prop_filename='models/atlas/atlas/propulsion/empirical_data/DOWTY_prop_6blade_13ft.xlsx', sheet_name='data')
        PropellerData.load_rpm_schedule(prop_filename='models/atlas/atlas/propulsion/empirical_data/DOWTY_prop_6blade_13ft.xlsx', sheet_name='RPM_schedule')
        
        cls._data_loaded = True
    
    @classmethod
    def _load_config_data(cls):
        if cls._config_data_loaded:
            return
        PropellerData.load_config_data(prop_filename='models/atlas/atlas/propulsion/empirical_data/DOWTY_prop_6blade_13ft.xlsx', sheet_name='INFO')
        cls._KwmdDrag = PropellerData._KwmdDrag
        cls._f_slipstream = PropellerData._slipstream_data
        cls._config_data_loaded = True

    @classmethod
    def _prepare_grid_data(cls):
        """
        Prepare grid data for direct interpolation: (velocity, rpm, power) -> thrust.
        
        Creates a 3D structured grid for MetaModelStructuredComp interpolation.
        """
        if cls._grid_data_built:
            return
        
        # Get raw data from PropellerData
        velocity_data_mps = PropellerData.dyn_tas_data__mps
        rpm_data = PropellerData.dyn_rpm_data__rpm
        power_data_W = PropellerData.dyn_power_data__W
        alt_data_m = PropellerData.dyn_dens_alt_data__m
        thrust_data_N = PropellerData.dyn_thrust_data__N
        
        # Compute data ranges
        min_velocity = np.min(velocity_data_mps)
        max_velocity = np.max(velocity_data_mps)
        min_rpm = np.min(rpm_data)
        max_rpm = np.max(rpm_data)
        min_power_W = np.min(power_data_W)
        max_power_W = np.max(power_data_W)
        min_thrust_N = np.min(thrust_data_N)
        max_thrust_N = np.max(thrust_data_N)
        min_alt_m = np.min(alt_data_m)
        max_alt_m = np.max(alt_data_m)
        
        # Grid dimensions
        n_velocity = 20
        n_rpm = 5
        n_power = 20
        n_alt = 10
        
        # Create 1D vectors for grid axes
        vect_velocity = np.linspace(min_velocity, max_velocity, n_velocity)
        vect_rpm = np.linspace(min_rpm, max_rpm, n_rpm)
        vect_power = np.linspace(min_power_W, max_power_W, n_power)
        vect_alt = np.linspace(min_alt_m, max_alt_m, n_alt)
        
        # Create 3D meshgrid for interpolation
        velocity_grid, rpm_grid, power_grid, alt_grid = np.meshgrid(
            vect_velocity, vect_rpm, vect_power, vect_alt, indexing='ij'
        )
        
        # Flatten grid points for griddata interpolation
        grid_points = np.column_stack([
            velocity_grid.flatten(), 
            rpm_grid.flatten(), 
            power_grid.flatten(),
            alt_grid.flatten()
        ])
        
        # Original scattered data points
        scatter_points = np.column_stack([velocity_data_mps, rpm_data, power_data_W, alt_data_m])
        
        # Interpolate thrust onto regular grid
        # Use linear interpolation with nearest neighbor fill for extrapolation
        thrust_grid_flat = griddata(
            scatter_points, 
            thrust_data_N, 
            grid_points, 
            method='linear', 
            fill_value=np.nan
        )
        
        # Fill NaN values with nearest neighbor interpolation'
        
        nan_mask = np.isnan(thrust_grid_flat)
        if np.any(nan_mask):
            nearest_interp = NearestNDInterpolator(scatter_points, thrust_data_N)
            thrust_grid_flat[nan_mask] = nearest_interp(grid_points[nan_mask])
        
        # Reshape to nd grid
    
        thrust_grid = thrust_grid_flat.reshape(n_velocity, n_rpm, n_power, n_alt)
        
        # Store grid data as class attributes
        cls._vect_velocity = vect_velocity
        cls._vect_rpm = vect_rpm
        cls._vect_power = vect_power
        cls._vect_alt = vect_alt
        cls._thrust_grid = thrust_grid
        
        # Store ranges for clamping/bounds
        cls._min_velocity = min_velocity
        cls._max_velocity = max_velocity
        cls._min_rpm = min_rpm
        cls._max_rpm = max_rpm
        cls._min_power_W = min_power_W
        cls._max_power_W = max_power_W
        cls._min_thrust_N = min_thrust_N
        cls._max_thrust_N = max_thrust_N
        
        cls._grid_data_built = True
        
        print(f"Propeller grid data prepared:")
        print(f"  Velocity range: {min_velocity:.1f} to {max_velocity:.1f} m/s")
        print(f"  RPM range: {min_rpm:.0f} to {max_rpm:.0f}")
        print(f"  Power range: {min_power_W/1000:.1f} to {max_power_W/1000:.1f} kW")
        print(f"  Thrust range: {min_thrust_N:.1f} to {max_thrust_N:.1f} N")
        print(f"  Grid size: {n_velocity} x {n_rpm} x {n_power} = {n_velocity*n_rpm*n_power} points")

    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        rpm_set = self.options['rpm_set']

        # RPM Schedule (if not set externally)
        if not rpm_set:
            # Convert matrix power input to vector for RPM schedule MetaModel
            self.add_subsystem('matrix_to_vector_rpm_power', 
                              MatrixToVectorConverter(
                                  num_nodes=nn, 
                                  num_comps=npp,
                                  input_names=['power'],
                                  units={'power': 'W'}
                              ), 
                              promotes_inputs=['power'])

            # Clamp power for RPM schedule interpolation
            self.connect('matrix_to_vector_rpm_power.power_vect', 'lim_min_power.input_array')
            self.connect('lim_min_power.output', 'lim_max_power.input_array')
            self.connect('lim_max_power.output', 'rpm_schedule_interp.power')

            self.add_subsystem('lim_min_power', 
                              SmoothMaxComp(num_nodes=nn * npp, mode='limit', units='W', 
                                          limit_val=self._min_power_W, n_comps=1, expected_min=self._min_power_W, expected_max=self._max_power_W), 
                              promotes_inputs=[], promotes_outputs=[])
            self.add_subsystem('lim_max_power', 
                              SmoothMinComp(num_nodes=nn * npp, mode='limit', units='W', 
                                          limit_val=self._max_power_W, n_comps=1, expected_min=self._min_power_W, expected_max=self._max_power_W), 
                              promotes_inputs=[], promotes_outputs=[])

            # Create RPM schedule metamodel
            rpm_schedule_interp = om.MetaModelStructuredComp(vec_size=nn * npp, method='akima')
            rpm_schedule_interp.add_input('power', 1000.0, 
                                         training_data=PropellerData.rpm_sched_data__power_W, units='W')
            rpm_schedule_interp.add_output('rpm', 1000.0, 
                                          training_data=PropellerData.rpm_sched_data__rpm, units='rpm',
                                          upper=np.max(PropellerData.rpm_sched_data__rpm), 
                                          lower=np.min(PropellerData.rpm_sched_data__rpm))
            rpm_schedule_interp.options['extrapolate'] = True
            self.add_subsystem('rpm_schedule_interp', rpm_schedule_interp, 
                              promotes_inputs=[], promotes_outputs=[])

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
            
            self.connect('rpm_schedule_interp.rpm', 'vector_to_matrix_rpm.rpm_vect')

        # Convert matrix inputs to vectors for thrust interpolation MetaModel
        self.add_subsystem('matrix_to_vector_inputs', 
                          MatrixToVectorConverter(
                              num_nodes=nn, 
                              num_comps=npp,
                              input_names=['power', 'rpm'],
                              units={'power': 'W', 'rpm': 'rpm'}
                          ), 
                          promotes_inputs=['power', 'rpm'])

        # Broadcast velocity to matrix (same for all props) using Tiler
        self.add_subsystem('velocity_tile',
                          Tiler(num_nodes=nn, n_comps=npp, tile_option='matrix',
                                input_names=['Utrue', 'alt'], output_names=['Utrue_mat', 'alt_mat'],
                                input_units={'Utrue': 'm/s', 'alt': 'm'}, output_units={'Utrue_mat': 'm/s', 'alt_mat': 'm'}),
                          promotes_inputs=['Utrue', 'alt'],
                          promotes_outputs=['Utrue_mat', 'alt_mat'])

        # Convert velocity matrix to vector for MetaModel
        self.add_subsystem('velocity_matrix_to_vector',
                          MatrixToVectorConverter(
                              num_nodes=nn,
                              num_comps=npp,
                              input_names=['Utrue_mat', 'alt_mat'],
                              output_names=['Utrue_vect', 'alt_vect'],
                              units={'Utrue_mat': 'm/s', 'alt_mat': 'm'}
                          ),
                          promotes_inputs=['Utrue_mat', 'alt_mat'], promotes_outputs=[])
        
        #self.connect('velocity_tile.Utrue_mat', 'velocity_matrix_to_vector.Utrue_mat')
        #self.connect('velocity_tile.alt_mat', 'velocity_matrix_to_vector.alt_mat')

        # Create thrust interpolation metamodel (3D: velocity, rpm, power -> thrust)
        thrust_interp = om.MetaModelStructuredComp(vec_size=nn * npp, method='scipy_cubic')
        thrust_interp.add_input('velocity', 50.0, training_data=self._vect_velocity, units='m/s')
        thrust_interp.add_input('rpm', 1000.0, training_data=self._vect_rpm, units='rpm')
        thrust_interp.add_input('power', 500000.0, training_data=self._vect_power, units='W')
        thrust_interp.add_input('alt', 5000, training_data=self._vect_alt, units='m')
        thrust_interp.add_output('thrust_raw', 5000.0, training_data=self._thrust_grid, units='N',
                                lower=self._min_thrust_N * 0.5, upper=self._max_thrust_N * 1.5)
        thrust_interp.options['extrapolate'] = True
        self.add_subsystem('thrust_interp', thrust_interp, promotes_inputs=[], promotes_outputs=[])
        
        # Connect inputs directly to thrust interpolator
        self.connect('velocity_matrix_to_vector.Utrue_vect', 'thrust_interp.velocity')
        self.connect('matrix_to_vector_inputs.rpm_vect', 'thrust_interp.rpm')
        self.connect('matrix_to_vector_inputs.power_vect', 'thrust_interp.power')
        self.connect('velocity_matrix_to_vector.alt_vect', 'thrust_interp.alt')

        # Convert vector output back to matrix
        self.add_subsystem('vector_to_matrix_thrust', 
                          VectorToMatrixConverter(
                              num_nodes=nn, 
                              num_comps=npp,
                              input_names=['thrust_raw_vect'],
                              output_names=['thrust_raw'],
                              units={'thrust_raw_vect': 'N'}
                          ), 
                          promotes_outputs=['thrust_raw'])
        
        self.connect('thrust_interp.thrust_raw', 'vector_to_matrix_thrust.thrust_raw_vect')

        # Add windmilling drag component
        self.add_subsystem('windmilling_drag', 
                          WindmillingDrag(num_nodes=nn, num_props=npp,
                                         wmill_thresh_hp=self.options['wmill_thresh_hp'],
                                         kwmdDrag=self._KwmdDrag,
                                         f_slipstream=self._f_slipstream),
                          promotes_inputs=['power', 'rho', 'Utrue'],
                          promotes_outputs=['thrust_calc'])
        
        # Connect raw thrust to windmilling drag component
        self.connect('thrust_raw', 'windmilling_drag.thrust_in')

        # Set default input values
        self.set_input_defaults('power', val=500000.0, units='W')


class WindmillingDrag(om.JaxExplicitComponent):
    """
    Component that computes windmilling drag for propellers.
    
    When power is below the windmilling threshold, this component subtracts
    windmilling drag from the input thrust:
    
    thrust_out = thrust_in * f_slipstream - windmilling_drag
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points
    num_props : int
        Number of propellers
    wmill_thresh_hp : float
        Power threshold below which windmilling drag is applied (default 200 HP)
    kwmdDrag : float
        Windmilling drag coefficient (default 0.1)
    f_slipstream : float
        Slipstream factor (default 0.98)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of props')
        self.options.declare('wmill_thresh_hp', default=200.0, 
                           desc='Power threshold for windmilling drag (HP)')
        self.options.declare('kwmdDrag', default=0.2415, 
                           desc='Windmilling drag coefficient')
        self.options.declare('f_slipstream', default=0.98, desc='Slipstream factor')

    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # Inputs
        self.add_input('power', units='hp', shape=(npp, nn), 
                      desc='Power input to propeller')
        self.add_input('thrust_in', units='N', shape=(npp, nn), 
                      desc='Thrust input to propeller')
        self.add_input('rho', units='kg/m**3', shape=(nn,), 
                      desc='Air density')
        self.add_input('Utrue', units='m/s', shape=(nn,), 
                      desc='True airspeed')
        
        # Outputs
        self.add_output('thrust_calc', units='N', shape=(npp, nn), 
                       desc='Output thrust adjusted for windmilling drag')
    
    def compute_primal(self, power, thrust_in, rho, Utrue):
        threshold_hp = self.options['wmill_thresh_hp']
        kwmd_drag = self.options['kwmdDrag']
        f_slipstream = self.options['f_slipstream']
        
        # Determine windmilling condition
        windmilling_active = power < threshold_hp
        
        # Calculate windmilling drag
        # Dynamic pressure: q = 0.5 * rho * v^2
        dynamic_pressure = 0.5 * rho * Utrue**2
        
        # Windmilling drag force
        windmilling_drag = kwmd_drag * dynamic_pressure
        
        # Apply windmilling drag only when active
        drag_wmill = jnp.where(windmilling_active, windmilling_drag, 0)

        # Power is input, thrust is output - reduce thrust by windmilling drag
        thrust_calc = thrust_in * f_slipstream - drag_wmill

        return thrust_calc


def run_prop_model(rpm_set=True, plot_results=True, test_interp=False):
    """
    Test the direct interpolation propeller model.
    """
    import time
    start_time = time.time()

    nn = 10
    npp = 2

    model = om.Group()

    # Velocity profile
    velocity_profile = np.linspace(243, 252, nn) * 0.5144 # Convert from knots to m/s

    # Compute density from altitude
    alt_m = np.linspace(14500, 15000, nn) * 0.3048
    temp_degC = 15.04 - 0.00649 * alt_m
    press_kPa = 101.29 * ((temp_degC + 273.15) / 288.08) ** 5.256
    rho_kgpm3 = press_kPa / (0.2869 * (temp_degC + 273.15))

    # Power profile
    power_profile = np.linspace(850e3, 850e3, nn)
    
    # RPM profile (if rpm_set)
    rpm_profile = np.linspace(997, 997, nn)

    ivc = om.IndepVarComp()
    if rpm_set:
        ivc.add_output('rpm', val=rpm_profile * np.ones((npp, 1)), units='rpm')
    ivc.add_output('power', val=power_profile * np.ones((npp, 1)), units='W')
    ivc.add_output('diameter', val=13 * 0.3048 * np.ones((npp, nn)), units='m')
    ivc.add_output('Utrue', val=velocity_profile, units='m/s')
    ivc.add_output('rho', val=rho_kgpm3, units='kg/m**3')
    ivc.add_output('alt', val=alt_m, units='m')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('propeller', EmpiricalPropellerMM(
        num_nodes=nn, 
        num_props=npp,
        rpm_set=rpm_set
    ), promotes=['*'])

    prob = om.Problem(model, reports=False)
    prob.setup()
    #om.n2(prob)
    
    run_start_time = time.time()
    prob.run_model()
    end_time = time.time()
    
    print(f"\nSetup time: {run_start_time - start_time:.2f} seconds")
    print(f"Execution time: {end_time - run_start_time:.2f} seconds")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")

    # Get results
    thrust_calc = prob.get_val('thrust_calc')
    thrust_raw = prob.get_val('thrust_raw')
    rpm_result = prob.get_val('rpm')
    power_result = prob.get_val('power', units='kW')

    eta_calc = thrust_raw * velocity_profile / (power_result * 1000)

    print(f"\nResults:")
    print(f"  Thrust raw (N): {thrust_raw}")
    print(f"  Thrust calc (N): {thrust_calc}")
    print(f"  RPM: {rpm_result}")
    print(f"  Power (kW): {power_result}")

    if plot_results:
        nodes = np.arange(nn)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Velocity profile
        plt.subplot(3, 3, 1)
        plt.plot(nodes, velocity_profile * 1.944, 'b-', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Velocity (kts)')
        plt.title('Velocity Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Altitude / Density profile
        plt.subplot(3, 3, 2)
        plt.plot(nodes, alt_m / 0.3048, 'g-', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Altitude (ft)')
        plt.title('Altitude Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: RPM profile
        plt.subplot(3, 3, 3)
        for i in range(npp):
            plt.plot(nodes, rpm_result[i, :], linewidth=2, marker='o', label=f'Prop {i+1}')
        plt.xlabel('Nodes')
        plt.ylabel('RPM')
        plt.title('RPM Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Power profile
        plt.subplot(3, 3, 4)
        for i in range(npp):
            plt.plot(nodes, power_result[i, :], linewidth=2, marker='o', label=f'Prop {i+1}')
        plt.xlabel('Nodes')
        plt.ylabel('Power (kW)')
        plt.title('Power Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Thrust profile (raw)
        plt.subplot(3, 3, 5)
        for i in range(npp):
            plt.plot(nodes, thrust_raw[i, :] / 1000, linewidth=2, marker='o', label=f'Prop {i+1}')
        plt.xlabel('Nodes')
        plt.ylabel('Thrust Raw (kN)')
        plt.title('Raw Interpolated Thrust')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Thrust profile (with windmilling)
        plt.subplot(3, 3, 6)
        for i in range(npp):
            plt.plot(nodes, thrust_calc[i, :] / 1000, linewidth=2, marker='o', label=f'Prop {i+1}')
        plt.xlabel('Nodes')
        plt.ylabel('Thrust Calc (kN)')
        plt.title('Thrust (with windmilling)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Density profile
        plt.subplot(3, 3, 7)
        plt.plot(nodes, rho_kgpm3, 'cyan', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Density (kg/m³)')
        plt.title('Air Density Profile')
        plt.grid(True, alpha=0.3)

        # Plot 8: Efficiency profile

        plt.subplot(3, 3, 8)
        for i in range(npp):
            plt.plot(nodes, eta_calc[i, :], 'magenta', linewidth=2, marker='o', label=f'Prop {i+1}')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('Efficiency (%)')
        plt.title('Efficiency Profile')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return prob


def test_interpolation_accuracy():
    """
    Test the interpolation accuracy against the original data.
    """
    print("Testing propeller direct interpolation accuracy...")
    
    # Ensure data is loaded
    PropellerData.load_data(
        prop_filename='models/atlas/atlas/propulsion/empirical_data/DOWTY_prop_6blade_13ft.xlsx', 
        sheet_name='data'
    )
    
    # Get test data (use a subset)
    test_indices = np.arange(0, len(PropellerData.dyn_thrust_data__N), 20)
    nn = len(test_indices)
    npp = 1
    
    velocity_test = PropellerData.dyn_tas_data__mps[test_indices]
    rpm_test = PropellerData.dyn_rpm_data__rpm[test_indices]
    power_test = PropellerData.dyn_power_data__W[test_indices]
    thrust_actual = PropellerData.dyn_thrust_data__N[test_indices]
    
    # Compute density from altitude
    dens_alt_m = PropellerData.dyn_dens_alt_data__m[test_indices]
    temp_degC = 15.04 - 0.00649 * dens_alt_m
    press_kPa = 101.29 * ((temp_degC + 273.15) / 288.08) ** 5.256
    rho_test = press_kPa / (0.2869 * (temp_degC + 273.15))
    
    # Build and run the model
    model = om.Group()
    
    ivc = om.IndepVarComp()
    ivc.add_output('rpm', val=rpm_test.reshape(npp, nn), units='rpm')
    ivc.add_output('power', val=power_test.reshape(npp, nn), units='W')
    ivc.add_output('diameter', val=13 * 0.3048 * np.ones((npp, nn)), units='m')
    ivc.add_output('Utrue', val=velocity_test, units='m/s')
    ivc.add_output('rho', val=rho_test, units='kg/m**3')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('propeller', EmpiricalPropellerMM(
        num_nodes=nn, 
        num_props=npp,
        rpm_set=True
    ), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    prob.run_model()
    
    # Get interpolated thrust
    thrust_interp = prob.get_val('thrust_raw').flatten()
    
    # Calculate errors
    errors = np.abs(thrust_interp - thrust_actual)
    relative_errors = errors / np.maximum(thrust_actual, 1.0)
    
    print(f"\nInterpolation Results:")
    print(f"  Number of test points: {nn}")
    print(f"  Mean absolute error: {np.mean(errors):.1f} N")
    print(f"  Max absolute error: {np.max(errors):.1f} N")
    print(f"  Mean relative error: {np.mean(relative_errors)*100:.2f}%")
    print(f"  Max relative error: {np.max(relative_errors)*100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(thrust_actual, thrust_interp, alpha=0.6, s=20)
    plt.plot([thrust_actual.min(), thrust_actual.max()], 
             [thrust_actual.min(), thrust_actual.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Thrust (N)')
    plt.ylabel('Interpolated Thrust (N)')
    plt.title('Actual vs Interpolated Thrust')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(relative_errors * 100, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return thrust_interp, thrust_actual, errors, relative_errors


if __name__ == "__main__":
    # Test interpolation accuracy
    #test_interpolation_accuracy()
    
    # Run the propeller model
    run_prop_model(rpm_set=True, plot_results=True)
