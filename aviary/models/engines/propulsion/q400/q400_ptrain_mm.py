"""
Q400 Powertrain Metamodel Component

This module provides a structured metamodel component for Q400 powertrain performance
interpolation. It uses data loaded by Q400PtrainData to interpolate fuel flow and
thrust based on throttle setting, altitude, and true airspeed.

Usage:
    from aviary.models.engines.propulsion.q400.q400_ptrain_mm import Q400Ptrain
    
    # Add to your model
    model.add_subsystem('powertrain', Q400Ptrain(num_nodes=nn), 
                        promotes_inputs=['throttle_nac', 'altitude', 'Utrue'],
                        promotes_outputs=['unit_fuel_flow', 'unit_thrust'])
"""

import numpy as np
import openmdao.api as om
import os
from openmdao.api import IndepVarComp
from aviary.utils.tiler import Tiler
from aviary.utils.matrix_vector_converter import VectorToMatrixConverter, MatrixToVectorConverter
from aviary.models.engines.propulsion.systems.parallel_hybrid import SumAlongAxis
from aviary.models.engines.propulsion.q400.q400_ptrain_data import Q400PtrainData
from aviary.models.engines.propulsion.systems.parallel_hybrid import DetermineNominalThrottle, ApplyThrustShare
from aviary.utils.dvlabel import DVLabel

class Q400Ptrain(om.Group):
    """
    Q400 Powertrain performance metamodel using structured interpolation.
    
    This component interpolates powertrain performance data to compute
    fuel flow and thrust based on operating conditions.
    
    Inputs
    ------
    throttle_nac : float (vec)
        Throttle/torque setting as percentage (0-100%)
    altitude : float (vec)
        Altitude in meters
    Utrue : float (vec)
        True airspeed in m/s
    
    Outputs
    -------
    fuel_flow : float (vec)
        Total fuel flow for both engines in kg/h
    unit_thrust : float (vec)
        Thrust per propeller/engine in Newtons
    
    Options
    -------
    num_nodes : int
        Number of analysis points (default: 1)
    data_file : str
        Path to the Excel data file (default: predefined path)
    sheet_name : str
        Sheet name in Excel file (default: 'cruise_data')
    surrogate_type : str
        Surrogate model type: 'response_surface', 'kriging', 'nearest_neighbor'
        (default: 'response_surface')
    """
    
    _data_loaded = False
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('data_file', default=None, 
                           desc='Path to Excel data file. If None, uses default location.')
        self.options.declare('sheet_name', default='data',
                           desc='Sheet name in Excel file')
        self.options.declare('surrogate_type', default='response_surface',
                           desc='Surrogate type: response_surface, kriging, nearest_neighbor')
        self.options.declare('num_props', default=4, desc='number of props')
        self.options.declare('max_torque', default=35404, desc='PW150 max torque in Nm')

        self.options.declare('prop_thrust_set', default=False, desc='propelelr thrust is specified')
        self.options.declare('phase_name', default='cruise', desc='phase name')
        self.options.declare('nacelle_power_set', default=False, desc='nacelle power is specified')
        # Load data on initialization
        self._load_data()
    
    def _load_data(self):
        """Load powertrain data if not already loaded."""
        if Q400Ptrain._data_loaded:
            return
        
        data_file = self.options['data_file']
        sheet_name = self.options['sheet_name']
        
        # Use default path if not specified
        if data_file is None:
            data_file = os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'propulsion', 'empirical_data', 'q400_mission_data.xlsx'
            )
        
        # Load the data
        Q400PtrainData.load_data(data_file, sheet_name)
        Q400Ptrain._data_loaded = True
    
    
    def setup(self):
        nn = self.options['num_nodes']
        surrogate_type = self.options['surrogate_type']
        prop_thrust_set = self.options['prop_thrust_set']
        num_props = self.options['num_props']
        phase_name = self.options['phase_name']
        
        # Select surrogate type
        if surrogate_type == 'kriging':
            surrogate = om.KrigingSurrogate()
        elif surrogate_type == 'nearest_neighbor':
            surrogate = om.NearestNeighbor()
        else:
            surrogate = om.ResponseSurface()  # Default: polynomial response surface
        # end

        dvlist = [
            ["fltcond|h", "altitude", 0* np.ones(nn), "m"],
            ["fltcond|Utrue", "Utrue", 90* np.ones(nn), "m/s"],
        ]

        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])


        if prop_thrust_set:

            self.add_subsystem("throttle_balance", DetermineNominalThrottle(num_nodes=nn, num_props=num_props, phase_name=phase_name), promotes_inputs=["total_thrust_required"], promotes_outputs=["nominal_throttle"])
            #self.add_subsystem("throttle_balance", SmoothThrustBalanceImplicit(num_nodes=nn, num_props=npp, phase_name=phase_name), promotes_inputs=[], promotes_outputs=["throttle_nac"])

            # Apply thrust share to get per-nacelle throttle
            self.add_subsystem("apply_thrust_share",
                ApplyThrustShare(num_nodes=nn, num_props=num_props),
                promotes_inputs=['nominal_throttle', 'thrust_share'],
                promotes_outputs=['throttle_nac'])

            # Connect the determined throttle to the nacelles
            self.connect(f"total_thrust", f"throttle_balance.total_thrust_calculated")
            self.connect("throttle_nac", f"torque_percent")



        self.add_subsystem('matrix_to_vector_eng_deck', 
                            MatrixToVectorConverter(
                                num_nodes=nn, 
                                num_comps=num_props,
                                input_names=['torque_percent', 'rpm'],
                                units={'torque_percent': None, 'rpm': 'rpm'},
                            ), 
        promotes_inputs=['*'], promotes_outputs=['*'])


        # Tile flight condition variables for MetaModel
        self.add_subsystem('tile_flight_conditions', Tiler(
            num_nodes=nn,
            n_comps=num_props,
            tile_option='vector',
            input_names=['altitude', 'Utrue'],
            output_names=['altitude_tiled', 'Utrue_tiled'],
            input_units={'altitude': 'm', 'Utrue': 'm/s'},
            output_units={'altitude_tiled': 'm', 'Utrue_tiled': 'm/s'},
            input_desc={'altitude': 'Altitude', 'Utrue': 'True airspeed'},
            output_desc={'altitude_tiled': 'Tiled altitude', 'Utrue_tiled': 'Tiled true airspeed'}
        ), promotes_inputs=['altitude', 'Utrue'])

        self.connect('tile_flight_conditions.altitude_tiled', 'nacelles.altitude')
        self.connect('tile_flight_conditions.Utrue_tiled', 'nacelles.Utrue')


        # Create UNstructured metamodel - works directly with scattered data
        # No need to create a regular grid first!
        q400_eng_deck = om.MetaModelUnStructuredComp(vec_size=nn * num_props, default_surrogate=surrogate)
        q400_eng_deck.add_input('torque_percent_vect', val=0.5 * np.ones(nn * num_props), units=None, training_data=Q400PtrainData.throttle_torque, desc='Throttle/torque setting (%)')
        q400_eng_deck.add_input('rpm_vect', val=900.0 * np.ones(nn * num_props), units='rpm', training_data=Q400PtrainData.RPM, desc='RPM')
        q400_eng_deck.add_input('altitude', val=5000.0 * np.ones(nn * num_props), units='m', training_data=Q400PtrainData.altitude_m, desc='Altitude')
        q400_eng_deck.add_input('Utrue', val=150.0 * np.ones(nn * num_props), units='m/s', training_data=Q400PtrainData.true_airspeed_mps, desc='True airspeed')
        q400_eng_deck.add_output('unit_fuel_flow', val=1000.0 * np.ones(nn * num_props), units='kg/h', training_data=Q400PtrainData.unit_fuel_flow_kgph, desc='Unit fuel flow (both engines)')
        q400_eng_deck.add_output('unit_thrust', val=7000.0 * np.ones(nn * num_props), units='N', training_data=Q400PtrainData.unit_thrust_N, desc='Thrust per engine')
        #q400_eng_deck.options['extrapolate'] = False
        

        self.add_subsystem('nacelles', q400_eng_deck, promotes_inputs=['torque_percent_vect', 'rpm_vect'], promotes_outputs=[])

        self.connect('nacelles.unit_fuel_flow', 'vector_to_matrix_eng_deck.unit_fuel_flow_vect')
        self.connect('nacelles.unit_thrust', 'vector_to_matrix_eng_deck.unit_thrust_vect')

        self.add_subsystem('vector_to_matrix_eng_deck', VectorToMatrixConverter(
            num_nodes=nn,
            num_comps=num_props,
            input_names=['unit_fuel_flow_vect', 'unit_thrust_vect'],
            units={'unit_fuel_flow_vect': 'kg/h', 'unit_thrust_vect': 'N'},
            output_names=['unit_fuel_flow', 'unit_thrust'],
        ), promotes_outputs=['*'])


        self.add_subsystem("add_thrust", SumAlongAxis(
                            num_nodes=nn, 
                            num_comps=num_props,
                            input_name="unit_thrust",
                            output_name="total_thrust",
                            input_units="N",
                            output_units="N",
                            input_desc="Thrust from all propellers/nacelles",
                            output_desc="Total thrust summed over all propellers",
                        ), promotes_inputs=["unit_thrust"], promotes_outputs=["total_thrust"])



def test_interpolator_accuracy(plot_results=True):
    """
    Test the interpolator accuracy by comparing metamodel outputs
    against actual data points.
    
    Parameters
    ----------
    plot_results : bool
        If True, create plots showing interpolated vs actual values
        
    Returns
    -------
    dict
        Dictionary containing error statistics
    """
    import time
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing Q400 Powertrain Interpolator Accuracy")
    print("=" * 60)
    
    # Ensure data is loaded
    data_file = os.path.join(
        os.path.dirname(__file__),
        '..','..', 'propulsion','empirical_data', 'q400_mission_data.xlsx'
    )
    
    Q400PtrainData.load_data(data_file, sheet_name='data')
    
    # Get actual data points
    throttle_actual = Q400PtrainData.throttle_torque
    altitude_actual = Q400PtrainData.altitude_m
    tas_actual = Q400PtrainData.true_airspeed_mps
    fuel_flow_actual = Q400PtrainData.unit_fuel_flow_kgph
    thrust_actual = Q400PtrainData.unit_thrust_N
    power_actual = Q400PtrainData._unit_shaft_power_kW
    
    nn = len(throttle_actual)
    
    print(f"Testing with {nn} data points from loaded dataset")
    
    # Create test model using the metamodel
    model = om.Group()
    
    # Add independent variables with actual data points
    ivc = om.IndepVarComp()
    ivc.add_output('torque_percent', val=throttle_actual, units=None)
    ivc.add_output('altitude', val=altitude_actual, units='m')
    ivc.add_output('Utrue', val=tas_actual, units='m/s')

    prop_thrust_set = False

    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('powertrain', Q400Ptrain(num_nodes=nn, num_props=4,nacelle_power_set=True, prop_thrust_set=False), promotes=['*'])
    
    # Create and setup problem
    prob = om.Problem(model, reports=False)
    
    start_time = time.time()
    prob.setup()
    setup_time = time.time() - start_time
    
    start_time = time.time()
    prob.run_model()
    run_time = time.time() - start_time
    
    print(f"\nSetup time: {setup_time:.3f} seconds")
    print(f"Run time: {run_time:.3f} seconds")
    
    # Get interpolated values
    fuel_flow_interp = prob.get_val('unit_fuel_flow', units='kg/h')
    thrust_interp = prob.get_val('unit_thrust', units='N')
    
    # Calculate errors
    fuel_flow_errors = fuel_flow_interp - fuel_flow_actual
    thrust_errors = thrust_interp - thrust_actual
    
    fuel_flow_rel_errors = np.abs(fuel_flow_errors) / np.maximum(np.abs(fuel_flow_actual), 1e-6) * 100
    thrust_rel_errors = np.abs(thrust_errors) / np.maximum(np.abs(thrust_actual), 1e-6) * 100
    
    # Print statistics
    print(f"\nFuel Flow Interpolation Errors:")
    print(f"  Mean absolute error: {np.mean(np.abs(fuel_flow_errors)):.2f} kg/h")
    print(f"  Max absolute error: {np.max(np.abs(fuel_flow_errors)):.2f} kg/h")
    print(f"  Mean relative error: {np.mean(fuel_flow_rel_errors):.2f}%")
    print(f"  Max relative error: {np.max(fuel_flow_rel_errors):.2f}%")
    
    print(f"\nThrust Interpolation Errors:")
    print(f"  Mean absolute error: {np.mean(np.abs(thrust_errors)):.2f} N")
    print(f"  Max absolute error: {np.max(np.abs(thrust_errors)):.2f} N")
    print(f"  Mean relative error: {np.mean(thrust_rel_errors):.2f}%")
    print(f"  Max relative error: {np.max(thrust_rel_errors):.2f}%")
    
    # Create plots if requested
    if plot_results:
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Fuel flow - actual vs interpolated
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(fuel_flow_actual, fuel_flow_interp, alpha=0.5, s=10)
        min_val = min(fuel_flow_actual.min(), fuel_flow_interp.min())
        max_val = max(fuel_flow_actual.max(), fuel_flow_interp.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        plt.xlabel('Actual Fuel Flow (kg/h)')
        plt.ylabel('Interpolated Fuel Flow (kg/h)')
        plt.title('Fuel Flow: Actual vs Interpolated')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Fuel flow error distribution
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(fuel_flow_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Fuel Flow Error (kg/h)')
        plt.ylabel('Frequency')
        plt.title('Fuel Flow Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Fuel flow relative error vs actual
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(fuel_flow_actual, fuel_flow_rel_errors, alpha=0.5, s=10)
        plt.xlabel('Actual Fuel Flow (kg/h)')
        plt.ylabel('Relative Error (%)')
        plt.title('Fuel Flow Relative Error')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Thrust - actual vs interpolated
        ax4 = plt.subplot(2, 3, 4)
        plt.scatter(thrust_actual, thrust_interp, alpha=0.5, s=10)
        min_val = min(thrust_actual.min(), thrust_interp.min())
        max_val = max(thrust_actual.max(), thrust_interp.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        plt.xlabel('Actual Thrust (N)')
        plt.ylabel('Interpolated Thrust (N)')
        plt.title('Thrust: Actual vs Interpolated')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Thrust error distribution
        ax5 = plt.subplot(2, 3, 5)
        plt.hist(thrust_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Thrust Error (N)')
        plt.ylabel('Frequency')
        plt.title('Thrust Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Thrust relative error vs actual
        ax6 = plt.subplot(2, 3, 6)
        plt.scatter(thrust_actual, thrust_rel_errors, alpha=0.5, s=10)
        plt.xlabel('Actual Thrust (N)')
        plt.ylabel('Relative Error (%)')
        plt.title('Thrust Relative Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Error vs input parameters
        fig2 = plt.figure(figsize=(15, 5))
        
        # Error vs throttle
        ax7 = plt.subplot(1, 3, 1)
        plt.scatter(throttle_actual, fuel_flow_rel_errors, alpha=0.5, s=10, label='Fuel Flow')
        plt.scatter(throttle_actual, thrust_rel_errors, alpha=0.5, s=10, label='Thrust')
        plt.xlabel('Throttle Torque (%)')
        plt.ylabel('Relative Error (%)')
        plt.title('Error vs Throttle')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error vs altitude (in ft for display)
        ax8 = plt.subplot(1, 3, 2)
        altitude_ft = altitude_actual / 0.3048
        plt.scatter(altitude_ft, fuel_flow_rel_errors, alpha=0.5, s=10, label='Fuel Flow')
        plt.scatter(altitude_ft, thrust_rel_errors, alpha=0.5, s=10, label='Thrust')
        plt.xlabel('Altitude (ft)')
        plt.ylabel('Relative Error (%)')
        plt.title('Error vs Altitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Error vs true airspeed
        ax9 = plt.subplot(1, 3, 3)
        plt.scatter(tas_actual, fuel_flow_rel_errors, alpha=0.5, s=10, label='Fuel Flow')
        plt.scatter(tas_actual, thrust_rel_errors, alpha=0.5, s=10, label='Thrust')
        plt.xlabel('True Airspeed (m/s)')
        plt.ylabel('Relative Error (%)')
        plt.title('Error vs True Airspeed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()
    
    return {
        'fuel_flow_errors': fuel_flow_errors,
        'thrust_errors': thrust_errors,
        'fuel_flow_rel_errors': fuel_flow_rel_errors,
        'thrust_rel_errors': thrust_rel_errors,
        'fuel_flow_actual': fuel_flow_actual,
        'thrust_actual': thrust_actual,
        'fuel_flow_interp': fuel_flow_interp,
        'thrust_interp': thrust_interp,
        'prob': prob
    }


def run_q400_ptrain(nacelle_power_set=True, prop_thrust_set=False):
    """Test the Q400 powertrain metamodel with a simple example."""
    import time
    
    print("=" * 60)
    print("Testing Q400 Powertrain Metamodel")
    print("=" * 60)
    
    nn = 5
    num_props = 2
    
    # Create test model
    model = om.Group()
    
    # Add independent variables
    ivc = om.IndepVarComp()
    ivc.add_output('fltcond|h', val=np.linspace(3000, 7000, nn), units='m')
    ivc.add_output('fltcond|Utrue', val=np.linspace(130, 160, nn), units='m/s')
    ivc.add_output('rpm', val=np.ones((num_props, nn)) * 850, units='rpm')


    if nacelle_power_set:
        ivc.add_output('torque_percent', val=np.linspace(0.4, 0.6, nn)[np.newaxis,:] * np.ones((num_props, nn)), units=None)

    if prop_thrust_set:
        
        thrust_share_array = np.array([1.0, 1.0])  # Nacelle 2 disabled
        thrust_share = np.tile(thrust_share_array[:, np.newaxis], (1, nn))  # (npp, nn)
        
        ivc.add_output('total_thrust_required', val=np.ones(nn) *2* 9000, units='N')
        ivc.add_output('thrust_share', thrust_share, units=None)

    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('powertrain', Q400Ptrain(num_nodes=nn, num_props=2, phase_name='cruise', nacelle_power_set=nacelle_power_set, prop_thrust_set=prop_thrust_set), promotes=['*'])
    
    # Create and setup problem
    prob = om.Problem(model, reports=False)


    om.n2(prob)
    
    start_time = time.time()
    prob.setup()

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 50
    prob.model.nonlinear_solver.options['atol'] = 1e-3
    prob.model.nonlinear_solver.options['rtol'] = 1e-3
    # Use ScipyKrylov instead of DirectSolver to avoid memory issues
    prob.model.linear_solver = om.DirectSolver()

    setup_time = time.time() - start_time
    
    
    start_time = time.time()
    prob.run_model()
    run_time = time.time() - start_time
    
    print(f"\nSetup time: {setup_time:.3f} seconds")
    print(f"Run time: {run_time:.3f} seconds")
    
    # Print results
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    
    throttle = prob.get_val('torque_percent').flatten()
    altitude = prob.get_val('altitude', units='m').flatten()
    tas = prob.get_val('Utrue', units='m/s').flatten()
    fuel_flow = prob.get_val('unit_fuel_flow', units='kg/h').flatten()
    thrust = prob.get_val('unit_thrust', units='N').flatten()
    
    print(f"{'Node':<6} {'Torque Throttle':<10} {'Alt (m)':<10} {'TAS (m/s)':<12} {'FF (kg/h)':<12} {'Thrust (N)':<12}")
    print("-" * 60)
    for i in range(nn):
        print(f"{i:<6} {float(throttle[i]):<10.1f} {float(altitude[i]):<10.0f} {float(tas[i]):<12.1f} {float(fuel_flow[i]):<12.1f} {float(thrust[i]):<12.1f}")
    
    print("\nTest completed successfully!")
    
    return prob


if __name__ == "__main__":
    # Run accuracy test with plots
    run_q400_ptrain(nacelle_power_set=False, prop_thrust_set=True)
    #test_interpolator_accuracy(plot_results=True)

