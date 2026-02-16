import numpy as np
import openmdao.api as om
import jax.numpy as jnp
import openmdao.jax as omj
import numpy as np
import matplotlib.pyplot as plt
from .import_sorted_turbo_data import SortedTurboData

from aviary.utils.math.multiply_divide_comp import ElementMultiplyDivideComp
from aviary.utils.math.integrals import Integrator
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
from aviary.utils.tiler import Tiler
from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp


class ComputeMaxPower(om.JaxExplicitComponent):
    """
    JAX component to convert power to throttle fraction using max power lookup.
    
    This component takes power input and converts it to throttle fraction
    using the max power matrix for each operating condition.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_turbs', default=4, desc='Number of propellers/nacelles')
        self.options.declare('turb_type', default='PT6', desc='Turbine type: PT6 or ACCE')
        self.options.declare('disa_unique', default=None, desc='Unique DISA values')
        self.options.declare('alt_unique', default=None, desc='Unique altitude values')
        self.options.declare('mach_unique', default=None, desc='Unique mach values')
        self.options.declare('max_power_matrix', default=None, desc='Max power matrix')
    
    def setup(self):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']

        # Add inputs
        self.add_input('disa', shape=(nn,), units='degC', desc='DISA temperature')
        self.add_input('altitude', shape=(nn,), units='m', desc='Altitude')
        self.add_input('mach', shape=(nn,), units=None, desc='Mach number')
        
        # Add outputs
        self.add_output('max_power', shape=(nt, nn), units='kW', desc='Max power')
        
        # Declare partials
        self.declare_partials('*', '*', method='exact')
    
    def compute_primal(self, disa, altitude, mach):
        """
        Compute max power using max power lookup.
        
        Parameters:
        -----------
        disa : jax.numpy.ndarray
            DISA temperature in degC
        altitude : jax.numpy.ndarray
            Altitude in meters
        mach : jax.numpy.ndarray
            Mach number
            
        Returns:
        --------
        jax.numpy.ndarray : Throttle fraction (0.0 to 1.0)
        
        """
        # Find the closest operating condition indices for each element in the input vectors
        disa_unique = jnp.array(self.options['disa_unique'])
        alt_unique = jnp.array(self.options['alt_unique'])
        mach_unique = jnp.array(self.options['mach_unique'])

        # For each input, find the index of the closest unique value (vectorized)
        disa_idx = jnp.argmin(omj.smooth_abs(disa_unique[None, :] - disa[:, None],0.001), axis=1)
        alt_idx = jnp.argmin(omj.smooth_abs(alt_unique[None, :] - altitude[:, None],0.001), axis=1)
        mach_idx = jnp.argmin(omj.smooth_abs(mach_unique[None, :] - mach[:, None],0.001), axis=1)
        
        # Get the max power for this operating condition using JAX indexing
        max_power_matrix = jnp.array(self.options['max_power_matrix'])
        max_power = max_power_matrix[disa_idx, alt_idx, mach_idx]

        # Broadcast max power to the number of props
        max_power = jnp.tile(max_power, (self.options['num_turbs'], 1))
        return max_power

class NegateFuelFlow(om.ExplicitComponent):
    """
    Negates fuel flow if throttle is less than or equal to throttle_min.
    Used to set fuel flow to 0 if GT is off, rather than idle 
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_turbs', default=4, desc='Number of propellers/nacelles')
        self.options.declare('mu', default=0.001, desc='Smoothing parameter')
        self.options.declare('throttle_min', default=1e-2, desc='min throttle designated as idle fuel flow')

    def setup(self):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']

        self.add_input('throttle_clipped', shape=(nt, nn), units=None, desc='clipped throttle')
        self.add_input('fuel_flow_raw', shape=(nt, nn), units="kg/h", desc='Clipped throttle')
        self.add_output('fuel_flow', shape=(nt, nn), units="kg/h", desc='Clipped throttle')

        self.declare_partials('fuel_flow', 'throttle_clipped', dependent= False)
        self.declare_partials('fuel_flow', 'fuel_flow_raw', rows=np.arange(nt * nn), cols=np.arange(nt * nn), method='exact')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']
        throttle_min = self.options['throttle_min']
        throttle_clipped = inputs['throttle_clipped']
        fuel_flow_raw = inputs['fuel_flow_raw']


        fuel_flow = np.where(throttle_clipped <= throttle_min, 0, fuel_flow_raw)

        outputs['fuel_flow'] = fuel_flow

    def compute_partials(self, inputs, partials):
        throttle_min = self.options['throttle_min']
        throttle_clipped = inputs['throttle_clipped']
        
        # Flatten for OpenMDAO
        x = throttle_clipped.flatten()
        
        # fuel_flow = 0                       if x <= throttle_min
        # fuel_flow = fuel_flow_raw           if x > throttle_min
        
        # ∂fuel_flow / ∂fuel_flow_raw
        df_draw = np.where(x > throttle_min, 1.0, 0.0)

        # ∂fuel_flow / ∂throttle_clipped
        # Discontinuous step → treat as 0 for stability
        df_dthr = np.zeros_like(x)

        partials['fuel_flow', 'fuel_flow_raw'] = df_draw
        #partials['fuel_flow', 'throttle_clipped'] = df_dthr

class ComputeTurbThrottle(om.ExplicitComponent):
    """
    Compute throttle from power and max power: throttle = power / max_power
    Handles matrix operations for multiple propellers/nacelles
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_turbs', default=4, desc='Number of propellers/nacelles')

    def setup(self):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']
        
        # Inputs
        self.add_input('power', shape=(nt, nn), units='kW', desc='Power input')
        self.add_input('max_power', shape=(nt, nn), units='kW', desc='Maximum power')
        
        # Output
        self.add_output('throttle', shape=(nt, nn), desc='Throttle ratio')
        
        # Declare partials
        self.declare_partials('throttle', ['power', 'max_power'], rows=np.arange(nt * nn), cols=np.arange(nt * nn), method='exact')
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']
        power = inputs['power']
        max_power = inputs['max_power']

        outputs['throttle'] = power / max_power
    
    def compute_partials(self, inputs, partials):

        nn = self.options['num_nodes']
        nt = self.options['num_turbs']
        power = inputs['power']
        max_power = inputs['max_power']
        # throttle = power / max_power
        partials['throttle', 'power'] = 1 / max_power.flatten()
        partials['throttle', 'max_power'] = -(power / max_power**2).flatten()
    

class TurboMission(om.Group):
    """
    Group containing turbo fuel flow component and mission integrator.
    
    This group combines the turbo engine model with mission integration
    to calculate total fuel consumption over a time period.
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points (default: 100)
    duration : float
        Time interval for integration in seconds (default: 3600)
    throttle_set : bool
        Set throttle or power for gas turbine (default: True)
    """
    _data_loaded = False
    _turb_type = None

    def initialize(self):
        self.options.declare('num_nodes', default=100, desc='Number of analysis points')
        self.options.declare('num_turbs', default=4, desc='Number of propellers/nacelles')
        self.options.declare('throttle_set', default=True, desc='Set throttle or power for gas turbine')
        self.options.declare('idle_allowed', default=True, desc='allow idle mode')
        self.options.declare('turb_type', default='PT6', desc='Turbine type: PT6 or ACCE')

    @classmethod
    def _load_data(cls, turb_type='PT6'):
        if cls._data_loaded and cls._turb_type == turb_type:
            return
        SortedTurboData.load_data(csv_filename=f'models/atlas/atlas/propulsion/empirical_data/sorted_turbo_dataset_{turb_type}.csv')
        cls._data_loaded = True
        cls._turb_type = turb_type


    def setup(self):
        nn = self.options['num_nodes']
        nt = self.options['num_turbs']
        throttle_set = self.options['throttle_set']
        idle_allowed = self.options['idle_allowed']
        turb_type = self.options['turb_type']

        self._load_data(turb_type=turb_type)




        if not throttle_set:
            self.add_subsystem("compute_turb_throttle", ComputeTurbThrottle(num_nodes=nn, num_turbs=nt), 
                              promotes_inputs=["*"], promotes_outputs=["*"])
            #self.connect("throttle", "lim_min_throttle.input_array")

        self.add_subsystem('lim_min_throttle', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=0, n_comps=nt), promotes_inputs=[('input_array', 'throttle')], promotes_outputs=[])
        self.add_subsystem('lim_max_throttle', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1, n_comps=nt), promotes_inputs=[], promotes_outputs=[("output", "throttle_clipped")])
        #self.add_subsystem('lim_min_throttle', SmoothMaxComp(num_nodes=nn,  mode='limit', units=None, limit_val=0, n_comps=nt, expected_min=0, expected_max=1), promotes_inputs=[('input_array', 'throttle')], promotes_outputs=[])
        #self.add_subsystem('lim_max_throttle', SmoothMinComp(num_nodes=nn, mode='limit', units=None, limit_val=1, n_comps=nt, expected_min=0, expected_max=1), promotes_inputs=[], promotes_outputs=[("output", "throttle_clipped")])
        self.connect('lim_min_throttle.output', 'lim_max_throttle.input_array')


        # Tile flight condition variables for MetaModel
        self.add_subsystem('tile_flight_conditions', Tiler(
            num_nodes=nn,
            n_comps=nt,
            tile_option='vector',
            input_names=['mach', 'disa', 'altitude'],
            output_names=['mach_tiled', 'disa_tiled', 'altitude_tiled'],
            input_units={'mach': None, 'disa': 'degC', 'altitude': 'm'},
            output_units={'mach_tiled': None, 'disa_tiled': 'degC', 'altitude_tiled': 'm'},
            input_desc={'mach': 'Mach number', 'disa': 'DISA temperature', 'altitude': 'Altitude'},
            output_desc={'mach_tiled': 'Tiled mach number', 'disa_tiled': 'Tiled DISA temperature', 'altitude_tiled': 'Tiled altitude'}
        ), promotes_inputs=['mach', 'disa', 'altitude'])

        if throttle_set:
            # Convert matrix inputs to vectors for MetaModel
            self.add_subsystem('matrix_to_vector_pow', 
                              MatrixToVectorConverter(
                                  num_nodes=nn, 
                                  num_comps=nt,
                                  input_names=['throttle_clipped'],
                                  units={'throttle_clipped': None}
                              ), 
                              promotes_inputs=['throttle_clipped'])
            
            #print("turbo throttle interp")
            throttle_to_pow_interp = om.MetaModelStructuredComp(vec_size=nn*nt, method='lagrange2')

            # TODO: See if this can be removed and replaced with throttle * max_power.... 
            
            # set up inputs and outputs with vectorization using global data
            throttle_to_pow_interp.add_input('disa_tiled', 1.0, training_data=SortedTurboData.disa_unique, units="degC")
            throttle_to_pow_interp.add_input('altitude_tiled', 1.0, training_data=SortedTurboData.alt_unique, units="m")
            throttle_to_pow_interp.add_input('mach_tiled', 1.0, training_data=SortedTurboData.mach_unique, units=None)
            throttle_to_pow_interp.add_input('throttle_clipped_vect', 1.0, training_data=SortedTurboData.throttle_unique, units=None)
            throttle_to_pow_interp.add_output('power_vect', 1.0, training_data=SortedTurboData.power_matrix, units="kW", lower = 1e-6)
            throttle_to_pow_interp.options['extrapolate'] = True

            self.add_subsystem('throttle_to_pow_interp', throttle_to_pow_interp, promotes_inputs=[], promotes_outputs=[])

            # Convert vector output back to matrix
            self.add_subsystem('vector_to_matrix_pow', 
                              VectorToMatrixConverter(
                                  num_nodes=nn, 
                                  num_comps=nt,
                                  input_names=['power_vect'],
                                  output_names=['power'],
                                  units={'power_vect': 'kW'}
                              ), 
                              promotes_outputs=['power'])

            # Connect the vector outputs from matrix_to_vector to MetaModel inputs
            self.connect('matrix_to_vector_pow.throttle_clipped_vect', 'throttle_to_pow_interp.throttle_clipped_vect')
            
            # Connect the tiled flight condition variables to MetaModel inputs
            self.connect('tile_flight_conditions.mach_tiled', 'throttle_to_pow_interp.mach_tiled')
            self.connect('tile_flight_conditions.disa_tiled', 'throttle_to_pow_interp.disa_tiled')
            self.connect('tile_flight_conditions.altitude_tiled', 'throttle_to_pow_interp.altitude_tiled')
            
            # Connect the vector output from MetaModel to vector_to_matrix input
            self.connect('throttle_to_pow_interp.power_vect', 'vector_to_matrix_pow.power_vect')


        # Convert matrix inputs to vectors for MetaModel
        self.add_subsystem('matrix_to_vector_ff', 
                          MatrixToVectorConverter(
                              num_nodes=nn, 
                              num_comps=nt,
                              input_names=['throttle_clipped'],
                              units={'throttle_clipped': None}
                          ), 
                          promotes_inputs=['throttle_clipped'])

        #print("turbo fuel flow interp")
        # Create regular grid interpolator instance with vectorization
        throttle_to_ff_interp = om.MetaModelStructuredComp(vec_size=nn * nt, method='akima')
        
        # set up inputs and outputs with vectorization using global data
        throttle_to_ff_interp.add_input('disa_tiled', 1.0, training_data=SortedTurboData.disa_unique, units="degC")
        throttle_to_ff_interp.add_input('altitude_tiled', 1.0, training_data=SortedTurboData.alt_unique, units="m")
        throttle_to_ff_interp.add_input('mach_tiled', 1.0, training_data=SortedTurboData.mach_unique, units=None)
        throttle_to_ff_interp.add_input('throttle_clipped_vect', 1.0, training_data=SortedTurboData.throttle_unique, units=None)
        throttle_to_ff_interp.add_output('jet_thrust_vect', 1.0, training_data=SortedTurboData.jet_thrust_matrix, units="N", lower = 1e-6)
        throttle_to_ff_interp.options['extrapolate'] = True

        if idle_allowed:
            throttle_to_ff_interp.add_output('fuel_flow_vect', 1.0, training_data=SortedTurboData.fuel_flow_throttle_matrix, units="kg/h",lower = 1e-6)
        else:
            throttle_to_ff_interp.add_output('fuel_flow_raw_vect', 1.0, training_data=SortedTurboData.fuel_flow_throttle_matrix, units="kg/h", lower = 1e-6)
        # end 
        #throttle_to_ff_interp.options['extrapolate'] = True

        self.add_subsystem('throttle_to_ff_interp', throttle_to_ff_interp, promotes_inputs=[], promotes_outputs=[])

        # Convert vector outputs back to matrix
        if idle_allowed:
            self.add_subsystem('vector_to_matrix_ff', 
                              VectorToMatrixConverter(
                                  num_nodes=nn, 
                                  num_comps=nt,
                                  input_names=['jet_thrust_vect', 'fuel_flow_vect'],
                                  output_names=['jet_thrust', 'fuel_flow'],
                                  units={'jet_thrust_vect': 'N', 'fuel_flow_vect': 'kg/h'}
                              ), 
                              promotes_outputs=['jet_thrust', 'fuel_flow'])
        else:
            self.add_subsystem('vector_to_matrix_ff', 
                              VectorToMatrixConverter(
                                  num_nodes=nn, 
                                  num_comps=nt,
                                  input_names=['jet_thrust_vect', 'fuel_flow_raw_vect'],
                                  output_names=['jet_thrust', 'fuel_flow_raw'],
                                  units={'jet_thrust_vect': 'N', 'fuel_flow_raw_vect': 'kg/h'}
                              ), 
                              promotes_outputs=['jet_thrust', 'fuel_flow_raw'])

        # Connect the vector outputs from matrix_to_vector to MetaModel inputs
        self.connect('matrix_to_vector_ff.throttle_clipped_vect', 'throttle_to_ff_interp.throttle_clipped_vect')
        
        # Connect the tiled flight condition variables to MetaModel inputs
        self.connect('tile_flight_conditions.mach_tiled', 'throttle_to_ff_interp.mach_tiled')
        self.connect('tile_flight_conditions.disa_tiled', 'throttle_to_ff_interp.disa_tiled')
        self.connect('tile_flight_conditions.altitude_tiled', 'throttle_to_ff_interp.altitude_tiled')
        
        # Connect the vector output from MetaModel to vector_to_matrix input
        self.connect('throttle_to_ff_interp.jet_thrust_vect', 'vector_to_matrix_ff.jet_thrust_vect')
        if idle_allowed:
            self.connect('throttle_to_ff_interp.fuel_flow_vect', 'vector_to_matrix_ff.fuel_flow_vect')
        else:
            self.connect('throttle_to_ff_interp.fuel_flow_raw_vect', 'vector_to_matrix_ff.fuel_flow_raw_vect')

        # AFter calculating fuel flow, if idle mode is off, set fuel flow to 0. GT is off..          
        if not idle_allowed:
            self.add_subsystem('idle_off', NegateFuelFlow(num_nodes=nn, num_turbs=nt), promotes_inputs=["*"], promotes_outputs=["*"])


class obj_func(om.ExplicitComponent):
    """
    Objective function for fuel consumption optimization
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('fuel_consumption', val=0.0, units='kg', desc='Fuel consumption', shape=(nn,))
        self.add_output('obj_func_val', val=0.0, desc='Objective function')
        
        self.declare_partials('obj_func_val', 'fuel_consumption', method='fd')
        
    def compute(self, inputs, outputs):
        outputs['obj_func_val'] = inputs['fuel_consumption'][-1]
        
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        partials['obj_func_val', 'fuel_consumption'] = np.zeros((nn))
        partials['obj_func_val', 'fuel_consumption'][-1] = 1.0


def run_turbomission(turb_type='PT6'):
    """
    Test function to verify TurboMission integrator is working correctly
    """
    print("="*60)
    print("TESTING TURBO MISSION INTEGRATOR")
    print("="*60)
    import time
    start_time = time.time()

    # Set up a simple mission
    nn = 11  # 10 time points
    duration = 1.0  # 1 hour mission
    nt = 4  # 4 propellers and/or nacelles
    
    # Create the model
    model = om.Group()

    altitude = np.linspace(10000, 15000, nn)
    mach = np.linspace(0.2, 0.4, nn)
    throttle = np.linspace(0.0, 0.9, nn) * np.ones((nt, nn))
    disa = np.linspace(0.0, 0.0, nn)
    power = np.linspace(0, 800, nn) * np.ones((nt, nn))
    max_power = np.linspace(872, 872, nn)

    throttle_set = False
    idle_allowed = False
    # Add independent variables
    ivc = om.IndepVarComp()
    # Repeat altitude, mach, and disa to match flattened matrix dimensions (nn * nt)
    ivc.add_output('altitude', val= altitude, units='ft', desc='Altitude')
    ivc.add_output('mach', val= mach, desc='Mach number')
    ivc.add_output('disa', val= disa, desc='DISA', units='degC')
    ivc.add_output('duration', val=duration, units='h', desc='Mission duration')
    ivc.add_output('max_power', val=max_power * np.ones((nt, nn)), units='kW', desc='Max power')

    if throttle_set:
        ivc.add_output('throttle', val=throttle, desc='Throttle fraction')
    else:
        ivc.add_output('power', val=power, units='kW', desc='Power')
    # end
       
    
    model.add_subsystem('ivc', ivc, promotes=['*'])

    
    # Add the turbo mission group
    model.add_subsystem('turbo_mission', TurboMission(
        num_nodes=nn,
        num_turbs=nt,
        throttle_set=throttle_set,
        idle_allowed=idle_allowed,
        turb_type=turb_type,
    ), promotes=['*'])
    
    # Set up the problem
    prob = om.Problem(model, reports=False)

    # Flight condition variables are now connected through tiler components
    # No need for direct connections to MetaModel inputs

    if throttle_set:
        # Flight condition variables are now connected through tiler components
        # No need for direct connections to MetaModel inputs
        pass
    # end 

    prob.setup()
    om.n2(prob)
    plot_results = True
    
    # Run the model
    print("Running TurboMission test...")
    prob.run_model()
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    nodes = np.arange(nn)

    # Get results
    fuel_flow = prob.get_val('fuel_flow')
    jet_thrust = prob.get_val('jet_thrust')
    
    if plot_results:


        # Formatting Results
        print(f"\nResults:")
        print(f"  Mission duration:")
        print(duration)
        print(f"  Number of time points:")
        print(nn)
        print(f"  Altitude:")
        print(prob['altitude'])
        print(f"  Mach:")
        print(prob['mach'])
        if throttle_set:    
            throttle = prob['throttle']
            print(f"  Throttle:")
            print(prob['throttle'])
        else:
            print(f"  Power:")
            print(prob['power'])
        print(f"  Average fuel flow:")
        print(np.mean(fuel_flow))
        print(f"  Average jet thrust:")
        print(np.mean(jet_thrust))
        

        plt.figure(figsize=(12, 8))
        
        # Plot altitude
        plt.subplot(2, 3, 1)
        plt.plot(nodes, altitude, 'b-', linewidth=2, marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Altitude (ft)')
        plt.title('Altitude Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot Mach
        plt.subplot(2, 3, 2)
        plt.plot(nodes, mach, 'r-', linewidth=2, marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Mach Number')
        plt.title('Mach Profile')
        plt.grid(True, alpha=0.3)

        if throttle_set:
            # Plot throttle
            plt.subplot(2, 3, 3)
            plt.plot(nodes, throttle[0,:], 'g-', linewidth=2, marker='o')
            plt.xlabel('Time (hours)')
            plt.ylabel('Throttle')
            plt.title('Throttle Profile')
            plt.grid(True, alpha=0.3)
        # end 
        
        # Plot fuel flow
        plt.subplot(2, 3, 4)
        plt.plot(nodes, fuel_flow[0,:], 'orange', linewidth=2, label='Interpolated', marker='o')
        #if test_interp :
        #    plt.plot(time_points, fuel_flow_nearest, 'orange', linewidth=2, alpha=0.5, label='Nearest Neighbor', marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Fuel Flow (kg/h)')
        plt.title('Fuel Flow Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot power
        plt.subplot(2, 3, 5)
        plt.plot(nodes, power[0,:], 'b-', linewidth=2, label='Interpolated', marker='o')
        #if test_interp and throttle_set:
        #    plt.plot(time_points, power_nearest, 'b-', linewidth=2, alpha=0.5, label='Nearest Neighbor', marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot jet thrust
        plt.subplot(2, 3, 6)
        plt.plot(nodes, jet_thrust[0,:], 'purple', linewidth=2, label='Interpolated', marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Jet Thrust (N)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("TURBO MISSION TEST COMPLETED")
        print("="*60)
        print("If the integration results make sense, the TurboMission integrator is working correctly!")
    # end


def _get_turbo_interpolation_results(plot_error=False, turb_type='PT6'):
    """
    Helper function to get turbo interpolation results for testing
    Parameters:
    -----------
    plot_error : bool, optional
        If True, create plots showing interpolation accuracy (default: False)

    Returns: interpolated_values_ff, actual_values_ff, relative_errors_ff, interpolated_values_power, actual_values_power, relative_errors_power
    """
    # Get the data
    SortedTurboData.load_data(csv_filename=f'models/atlas/atlas/propulsion/empirical_data/sorted_turbo_dataset_{turb_type}.csv')
    
    # Use a subset of data points to avoid overfitting test
    # Use every 10th point to create test data that's different from training
    n_total_points = len(SortedTurboData.fuel_flow_data__kgph)
    test_indices = np.arange(0, n_total_points, 100)
    n_test_points = len(test_indices)
    
    # Set up the OpenMDAO model for testing
    nn = n_test_points
    idle_allowed = True
    nt = 1
    
    model = om.Group()
    ivc = om.IndepVarComp()

    throttle_set = False
    
    # Add independent variables with the test data points
    ivc.add_output('altitude', SortedTurboData.alt_data__m[test_indices], units='m', desc='Altitude in meters')
    ivc.add_output('mach', SortedTurboData.mach_data[test_indices], desc='Mach number')
    ivc.add_output('disa', SortedTurboData.disa_data__degC[test_indices], units='degC', desc='DISA in degrees Celsius')
    if throttle_set:
        ivc.add_output('max_power', 872 * np.ones(nn), units='kW', desc='Max power in kW')

        ivc.add_output('throttle', SortedTurboData.frac_data[test_indices], desc='Throttle fraction')
    else:
        ivc.add_output('power', SortedTurboData.power_data__kW[test_indices], units='kW', desc='Power in kW')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])

    if not throttle_set:
        # Initialize ComputeMaxPower with required data
        compute_max_power = ComputeMaxPower(
            num_nodes=nn,
            num_turbs=nt,
            disa_unique=SortedTurboData.disa_unique,
            alt_unique=SortedTurboData.alt_unique,
            mach_unique=SortedTurboData.mach_unique,
            max_power_matrix=SortedTurboData.max_power_matrix,
            turb_type=turb_type
        )
        model.add_subsystem('turb_max_power', compute_max_power, promotes=["*"])
    # end

    model.add_subsystem('dyn_turbo_group', TurboMission(num_nodes=nn,num_turbs=nt,
    throttle_set=throttle_set, idle_allowed=idle_allowed, turb_type=turb_type), promotes=["*"])
    
    prob = om.Problem(model, reports=False)

    if throttle_set:
        # Flight condition variables are now connected through tiler components
        # No need for direct connections to MetaModel inputs
        pass
    # end 

    # Flight condition variables are now connected through tiler components
    # No need for direct connections to MetaModel inputs
    prob.setup()
    #om.n2(prob)
    # Run the model with the test data points
    prob.run_model()
    interpolated_ff = prob.get_val('fuel_flow').flatten()
    actual_values_ff = SortedTurboData.fuel_flow_data__kgph[test_indices]
    
    # Calculate statistics
    errors_ff = np.abs(interpolated_ff - actual_values_ff)
    relative_errors_ff = errors_ff / np.maximum(actual_values_ff, 1e-6)  # Avoid division by zero
    
    # Initialize power variables
    interpolated_values_power = None
    actual_values_power = None
    relative_errors_power = None
    
    # Add power analysis if throttle_set is true
    if throttle_set:
        interpolated_values_power = prob.get_val('power')
        actual_values_power = SortedTurboData.power_data__kW[test_indices]
        actual_values_power[actual_values_power==0] = 1e-6
        errors_power = np.abs(interpolated_values_power - actual_values_power)
        relative_errors_power = errors_power / np.maximum(actual_values_power, 1e-6)  # Avoid division by zero

        # Remove infs and nans from the interpolated power values
        valid_mask = np.isfinite(interpolated_values_power)
        interpolated_values_power = interpolated_values_power[valid_mask]
        actual_values_power = actual_values_power[valid_mask]
        relative_errors_power = relative_errors_power[valid_mask]
    
    # Create plots if requested
    if plot_error:
        # Create the comparison plot
        plt.figure(figsize=(10, 8))
        
        # Plot actual vs interpolated
        plt.subplot(2, 2, 1)
        plt.scatter(actual_values_ff, interpolated_ff, alpha=0.6, s=20)
        plt.plot([actual_values_ff.min(), actual_values_ff.max()], [actual_values_ff.min(), actual_values_ff.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Fuel Flow (kg/h)')
        plt.ylabel('Interpolated Fuel Flow (kg/h)')
        plt.title('Dynamic Conditions: Actual vs Interpolated (Subset of Points)')
        plt.grid(True, alpha=0.3)
        
        # Plot error distribution
        plt.subplot(2, 2, 2)
        plt.hist(errors_ff, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error (kg/h)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot relative error distribution
        plt.subplot(2, 2, 3)
        plt.hist(relative_errors_ff * 100, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Relative Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot error vs actual value
        plt.subplot(2, 2, 4)
        plt.scatter(actual_values_ff, relative_errors_ff * 100, alpha=0.6, s=20)
        plt.xlabel('Actual Fuel Flow (kg/h)')
        plt.ylabel('Relative Error (%)')
        plt.title('Error vs Actual Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        # Add power analysis if throttle_set is true
        if throttle_set and interpolated_values_power is not None and relative_errors_power is not None:
            errors_power = np.abs(interpolated_values_power - actual_values_power)
            
            plt.figure(figsize=(10, 8))
            # Plot actual vs interpolated power
            plt.subplot(2, 2, 1)
            plt.scatter(actual_values_power, interpolated_values_power, alpha=0.6, s=20)
            plt.plot([actual_values_power.min(), actual_values_power.max()], [actual_values_power.min(), actual_values_power.max()], 'r--', linewidth=2)
            plt.xlabel('Actual Power (kW)')
            plt.ylabel('Interpolated Power (kW)')
            plt.title('Actual vs Interpolated Power')
            plt.grid(True, alpha=0.3)

            # Plot error distribution for power
            plt.subplot(2, 2, 2)
            plt.hist(errors_power, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Absolute Error (kW)')
            plt.ylabel('Frequency')
            plt.title('Power Error Distribution')
            plt.grid(True, alpha=0.3)

            # Plot relative error distribution for power
            plt.subplot(2, 2, 3)
            plt.hist(relative_errors_power * 100, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Frequency')
            plt.title('Power Relative Error Distribution')
            plt.grid(True, alpha=0.3)

            # Plot error vs actual value for power
            plt.subplot(2, 2, 4)
            plt.scatter(actual_values_power, relative_errors_power * 100, alpha=0.6, s=20)
            plt.xlabel('Actual Power (kW)')
            plt.ylabel('Relative Error (%)')
            plt.title('Power Error vs Actual Value')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        # Add fuel flow error vs throttle/power plots
        plt.figure(figsize=(15, 6))
        
        # Get throttle and power data for the test points
        throttle_test_data = SortedTurboData.frac_data[test_indices]
        power_test_data = SortedTurboData.power_data__kW[test_indices]
        
        if throttle_set:
            # Plot fuel flow error vs throttle
            plt.subplot(1, 3, 1)
            plt.scatter(throttle_test_data, relative_errors_ff * 100, alpha=0.6, s=20, c='blue')
            plt.xlabel('Throttle Fraction')
            plt.ylabel('Fuel Flow Relative Error (%)')
            plt.title('Fuel Flow Error vs Throttle')
            plt.grid(True, alpha=0.3)
            
            # Plot fuel flow error vs power
            plt.subplot(1, 3, 2)
            plt.scatter(power_test_data, relative_errors_ff * 100, alpha=0.6, s=20, c='red')
            plt.xlabel('Power (kW)')
            plt.ylabel('Fuel Flow Relative Error (%)')
            plt.title('Fuel Flow Error vs Power')
            plt.grid(True, alpha=0.3)
        else:
            # When throttle_set=False, focus on power-based analysis
            # Plot fuel flow error vs power (primary x-axis)
            plt.subplot(1, 3, 1)
            plt.scatter(power_test_data, relative_errors_ff * 100, alpha=0.6, s=20, c='red')
            plt.xlabel('Power (kW)')
            plt.ylabel('Fuel Flow Relative Error (%)')
            plt.title('Fuel Flow Error vs Power')
            plt.grid(True, alpha=0.3)
            
            # Plot fuel flow error vs throttle (secondary)
            plt.subplot(1, 3, 2)
            plt.scatter(throttle_test_data, relative_errors_ff * 100, alpha=0.6, s=20, c='blue')
            plt.xlabel('Throttle Fraction')
            plt.ylabel('Fuel Flow Relative Error (%)')
            plt.title('Fuel Flow Error vs Throttle')
            plt.grid(True, alpha=0.3)
        
        # Plot fuel flow error vs actual fuel flow (common to both cases)
        plt.subplot(1, 3, 3)
        plt.scatter(actual_values_ff, relative_errors_ff * 100, alpha=0.6, s=20, c='green')
        plt.xlabel('Actual Fuel Flow (kg/h)')
        plt.ylabel('Fuel Flow Relative Error (%)')
        plt.title('Fuel Flow Error vs Actual Fuel Flow')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed analysis plots
        plt.figure(figsize=(15, 10))
        
        if throttle_set:
            # Create 2D scatter plots with color-coded error (throttle-focused)
            # Fuel flow error vs throttle with color-coded power
            plt.subplot(2, 3, 1)
            scatter = plt.scatter(throttle_test_data, actual_values_ff, c=relative_errors_ff * 100, 
                                 alpha=0.7, s=30, cmap='viridis')
            plt.colorbar(scatter, label='Relative Error (%)')
            plt.xlabel('Throttle Fraction')
            plt.ylabel('Actual Fuel Flow (kg/h)')
            plt.title('Fuel Flow vs Throttle (Color = Error %)')
            plt.grid(True, alpha=0.3)
            
            # Fuel flow error vs power with color-coded throttle
            plt.subplot(2, 3, 2)
            scatter = plt.scatter(power_test_data, actual_values_ff, c=throttle_test_data, 
                                 alpha=0.7, s=30, cmap='plasma')
            plt.colorbar(scatter, label='Throttle Fraction')
            plt.xlabel('Power (kW)')
            plt.ylabel('Actual Fuel Flow (kg/h)')
            plt.title('Fuel Flow vs Power (Color = Throttle)')
            plt.grid(True, alpha=0.3)
            
            # Error distribution by throttle ranges
            plt.subplot(2, 3, 3)
            throttle_ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
            colors = ['blue', 'green', 'red']
            for i, (min_throttle, max_throttle) in enumerate(throttle_ranges):
                mask = (throttle_test_data >= min_throttle) & (throttle_test_data < max_throttle)
                if np.any(mask):
                    plt.hist(relative_errors_ff[mask] * 100, bins=15, alpha=0.6, 
                            label=f'Throttle {min_throttle}-{max_throttle}', color=colors[i])
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution by Throttle Range')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Create 2D scatter plots with color-coded error (power-focused)
            # Fuel flow error vs power with color-coded throttle
            plt.subplot(2, 3, 1)
            scatter = plt.scatter(power_test_data, actual_values_ff, c=relative_errors_ff * 100, 
                                 alpha=0.7, s=30, cmap='viridis')
            plt.colorbar(scatter, label='Relative Error (%)')
            plt.xlabel('Power (kW)')
            plt.ylabel('Actual Fuel Flow (kg/h)')
            plt.title('Fuel Flow vs Power (Color = Error %)')
            plt.grid(True, alpha=0.3)
            
            # Fuel flow error vs throttle with color-coded power
            plt.subplot(2, 3, 2)
            scatter = plt.scatter(throttle_test_data, actual_values_ff, c=power_test_data, 
                                 alpha=0.7, s=30, cmap='plasma')
            plt.colorbar(scatter, label='Power (kW)')
            plt.xlabel('Throttle Fraction')
            plt.ylabel('Actual Fuel Flow (kg/h)')
            plt.title('Fuel Flow vs Throttle (Color = Power)')
            plt.grid(True, alpha=0.3)
            
            # Error distribution by power ranges
            plt.subplot(2, 3, 3)
            power_ranges = [(0, 200), (200, 400), (400, 600)]
            colors = ['blue', 'green', 'red']
            for i, (min_power, max_power) in enumerate(power_ranges):
                mask = (power_test_data >= min_power) & (power_test_data < max_power)
                if np.any(mask):
                    plt.hist(relative_errors_ff[mask] * 100, bins=15, alpha=0.6, 
                            label=f'Power {min_power}-{max_power}kW', color=colors[i])
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution by Power Range')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Error vs altitude
        plt.subplot(2, 3, 4)
        altitude_test_data = SortedTurboData.alt_data__m[test_indices]
        plt.scatter(altitude_test_data, relative_errors_ff * 100, alpha=0.6, s=20)
        plt.xlabel('Altitude (m)')
        plt.ylabel('Fuel Flow Relative Error (%)')
        plt.title('Fuel Flow Error vs Altitude')
        plt.grid(True, alpha=0.3)
        
        # Error vs Mach number
        plt.subplot(2, 3, 5)
        mach_test_data = SortedTurboData.mach_data[test_indices]
        plt.scatter(mach_test_data, relative_errors_ff * 100, alpha=0.6, s=20)
        plt.xlabel('Mach Number')
        plt.ylabel('Fuel Flow Relative Error (%)')
        plt.title('Fuel Flow Error vs Mach Number')
        plt.grid(True, alpha=0.3)
        
        # Error vs DISA
        plt.subplot(2, 3, 6)
        disa_test_data = SortedTurboData.disa_data__degC[test_indices]
        plt.scatter(disa_test_data, relative_errors_ff * 100, alpha=0.6, s=20)
        plt.xlabel('DISA (°C)')
        plt.ylabel('Fuel Flow Relative Error (%)')
        plt.title('Fuel Flow Error vs DISA')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return interpolated_ff, actual_values_ff, relative_errors_ff, interpolated_values_power, actual_values_power, relative_errors_power

def test_fuel_flow_interpolation_max_error(tolerance=2.0, turb_type='PT6'):
    """Test fuel flow interpolation maximum relative error"""
    _, _, relative_errors_ff, _, _, _ = _get_turbo_interpolation_results(turb_type=turb_type)
    max_relative_error = np.max(relative_errors_ff)
    print(f"Fuel flow interpolation max relative error: {max_relative_error:.6f}")
    assert max_relative_error < tolerance, f"Fuel flow interpolation max relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"

def test_fuel_flow_interpolation_mean_error(tolerance=0.5, turb_type='PT6'):
    """Test fuel flow interpolation mean relative error"""
    _, _, relative_errors_ff, _, _, _ = _get_turbo_interpolation_results(turb_type=turb_type)
    mean_relative_error = np.mean(relative_errors_ff)
    print(f"Fuel flow interpolation mean relative error: {mean_relative_error:.6f}")
    assert mean_relative_error < tolerance, f"Fuel flow interpolation mean relative error {mean_relative_error:.6f} exceeds tolerance {tolerance}"

def test_power_interpolation_max_error(tolerance=2.0, turb_type='PT6'):
    """Test power interpolation maximum relative error (if throttle_set=True)"""
    _, _, _, interpolated_values_power, _, relative_errors_power = _get_turbo_interpolation_results(turb_type=turb_type)
    if interpolated_values_power is not None and relative_errors_power is not None:
        max_relative_error = np.max(relative_errors_power)
        print(f"Power interpolation max relative error: {max_relative_error:.6f}")
        assert max_relative_error < tolerance, f"Power interpolation max relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"

def test_power_interpolation_mean_error(tolerance=0.5, turb_type='PT6'):
    """Test power interpolation mean relative error (if throttle_set=True)"""
    _, _, _, interpolated_values_power, _, relative_errors_power = _get_turbo_interpolation_results(turb_type=turb_type)
    if interpolated_values_power is not None and relative_errors_power is not None:
        mean_relative_error = np.mean(relative_errors_power)
        print(f"Power interpolation mean relative error: {mean_relative_error:.6f}")
        assert mean_relative_error < tolerance, f"Power interpolation mean relative error {mean_relative_error:.6f} exceeds tolerance {tolerance}"

def run_interpolation_accuracy_test(plot_error=True, turb_type='PT6'):
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    Parameters:
    -----------
    plot_error : bool, optional
        If True, create plots showing interpolation accuracy (default: True)
    """
    print("Testing turbo interpolation accuracy...")

    # Get the interpolation results with plotting
    interpolated_ff, actual_values_ff, relative_errors_ff, interpolated_values_power, actual_values_power, relative_errors_power = _get_turbo_interpolation_results(plot_error=plot_error, turb_type=turb_type)
    
    # Calculate absolute errors for display
    errors_ff = np.abs(interpolated_ff - actual_values_ff)
    
    print(f"Mean absolute error (fuel flow): {np.mean(errors_ff):.2f} kg/h")
    print(f"Max absolute error (fuel flow): {np.max(errors_ff):.2f} kg/h")
    print(f"Mean relative error (fuel flow): {np.mean(relative_errors_ff * 100):.2f}%")
    print(f"Max relative error (fuel flow): {np.max(relative_errors_ff * 100):.2f}%")
    
    if interpolated_values_power is not None and relative_errors_power is not None:
        errors_power = np.abs(interpolated_values_power - actual_values_power)
        print(f"Mean absolute error (power): {np.mean(errors_power):.2f} kW")
        print(f"Max absolute error (power): {np.max(errors_power):.2f} kW")
        print(f"Mean relative error (power): {np.mean(relative_errors_power * 100):.2f}%")
        print(f"Max relative error (power): {np.max(relative_errors_power * 100):.2f}%")
    
    # Call individual test functions
    test_fuel_flow_interpolation_max_error(tolerance=2.0, turb_type=turb_type)
    test_fuel_flow_interpolation_mean_error(tolerance=0.5, turb_type=turb_type)
    test_power_interpolation_max_error(tolerance=2.0, turb_type=turb_type)
    test_power_interpolation_mean_error(tolerance=0.5, turb_type=turb_type)
    

if __name__ == "__main__":

    # Test interpolation accuracy (Linear ND)
    #test_neural_net = True
    #run_interpolation_accuracy_test(turb_type='PT6')
    
    # Test TurboMission integrator
    run_turbomission(turb_type='ACCE')

