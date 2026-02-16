import numpy as np
import openmdao.api as om
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
import jax 
import jax.numpy as jnp
import openmdao.jax as omj
import openmdao.api as om
import numpy as np
import cProfile
import matplotlib.pyplot as plt
import json
from aviary.utils.math.multiply_divide_comp import ElementMultiplyDivideComp
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
import time

# Import the global data store
from .h3x_motor_data_web import MotorDataEffMap # Torque to efficiency lookup table
from .h3x_motor_data_web import MotorDataPowerVoltCurve # Power to voltage lookup table
from .h3x_motor_data_rfi import MotorDataPowerEffCurve # Power to efficiency lookup table



def _get_motor_interpolation_results(plot_error=False):
    """
    Helper function to get motor interpolation results for testing using EmpiricalMotor group
    Parameters:
    -----------
    test_neural_net : bool, optional
        If True, use neural network model (default: False)
    plot_error : bool, optional
        If True, create plots showing interpolation accuracy (default: False)
    
    Returns: efficiency_results, voltage_power_results, power_efficiency_results
    """
    # Load the actual data for comparison
    MotorDataEffMap.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM_2300_eff.xlsx')
    MotorDataPowerVoltCurve.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM_2300_volts.xlsx')
    MotorDataPowerEffCurve.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')
    
    # Test 1: Torque/RPM to Efficiency Interpolation using EmpiricalMotor
    n_test_points = min(20, len(MotorDataEffMap.rpm_data))
    np.random.seed(42)
    test_indices = np.random.choice(len(MotorDataEffMap.rpm_data), n_test_points, replace=False)
    
    # Set up the model using EmpiricalMotor group
    model = om.Group()
    ivc = om.IndepVarComp()
    
    test_rpm = MotorDataEffMap.rpm_data[test_indices]
    test_torque = MotorDataEffMap.torque_data[test_indices]
    actual_eff = MotorDataEffMap.eff_data[test_indices]
    
    ivc.add_output('rpm', test_rpm, units='rpm', desc='Motor speed')
    ivc.add_output('torque', test_torque, units='N*m', desc='Motor torque')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor', EmpiricalMotor(
        num_nodes=n_test_points,
        torque_rpm_set=True,  # Use torque/RPM mode
        num_motors=1,
    ), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    om.n2(prob)
    prob.run_model()
    
    interpolated_eff = prob.get_val('eff').flatten()
    eff_errors = np.abs(interpolated_eff - actual_eff)
    eff_relative_errors = eff_errors / actual_eff * 100
    

    # Test 3: Power/Voltage to Efficiency Interpolation using EmpiricalMotor
    n_test_points_power_eff = min(20, len(MotorDataPowerEffCurve.voltage_data))
    np.random.seed(456)
    test_indices_power_eff = np.random.choice(len(MotorDataPowerEffCurve.voltage_data), n_test_points_power_eff, replace=False)
    
    model3 = om.Group()
    ivc3 = om.IndepVarComp()
    
    test_voltage_power_eff = MotorDataPowerEffCurve.voltage_data[test_indices_power_eff]
    test_power_power_eff = MotorDataPowerEffCurve.power_data__W[test_indices_power_eff] / 1000  # Convert to kW
    actual_eff_power_eff = MotorDataPowerEffCurve.eff_data[test_indices_power_eff]
    
    ivc3.add_output('voltage', test_voltage_power_eff, units='V', desc='Motor voltage')
    ivc3.add_output('mech_power', test_power_power_eff, units='kW', desc='Motor power')
    
    model3.add_subsystem('ivc', ivc3, promotes=['*'])
    model3.add_subsystem('motor', EmpiricalMotor(
        num_nodes=n_test_points_power_eff,
        torque_rpm_set=False,  # Use power/voltage mode
        throttle_set=False,
        num_motors=1,
    ), promotes=['*'])
    
    prob3 = om.Problem(model3, reports=False)
    prob3.setup()
    prob3.run_model()
    
    interpolated_eff_power_eff = prob3.get_val('eff').flatten()
    eff_power_eff_errors = np.abs(interpolated_eff_power_eff - actual_eff_power_eff)
    eff_power_eff_relative_errors = eff_power_eff_errors / actual_eff_power_eff * 100
    
    # Test 2: Voltage/Power Limit Interpolation using direct MetaModelStructuredComp
    n_test_points_volts = min(20, len(MotorDataPowerVoltCurve.rpm_data))
    np.random.seed(123)
    test_indices_volts = np.random.choice(len(MotorDataPowerVoltCurve.rpm_data), n_test_points_volts, replace=False)
    
    # Load the data and create the interpolation grid (same as nacelle_splitter.py)
    rpm_data = MotorDataPowerVoltCurve.rpm_data
    voltage_data = MotorDataPowerVoltCurve.voltage_data
    power_kW_data = MotorDataPowerVoltCurve.power_data

    min_rpm = np.min(rpm_data)
    max_rpm = np.max(rpm_data)
    min_voltage = np.min(voltage_data)
    max_voltage = np.max(voltage_data)

    n_rpm = 50
    n_voltage = 50
    
    vect_rpm = np.linspace(min_rpm, max_rpm, n_rpm)
    vect_voltage = np.linspace(min_voltage, max_voltage, n_voltage)

    map_power_from_rpm_voltage = np.zeros((n_rpm, n_voltage))
    rpm_grid, voltage_grid = np.meshgrid(vect_rpm, vect_voltage, indexing='ij')
    grid_points_rpm_voltage = np.column_stack([rpm_grid.flatten(), voltage_grid.flatten()])
    map_power_from_rpm_voltage_flat = griddata(np.column_stack([rpm_data, voltage_data]), power_kW_data, grid_points_rpm_voltage, method='cubic', fill_value=0.2)
    map_power_from_rpm_voltage = map_power_from_rpm_voltage_flat.reshape(len(vect_rpm), len(vect_voltage))
    
    model2 = om.Group()
    ivc2 = om.IndepVarComp()
    
    test_rpm_volts = MotorDataPowerVoltCurve.rpm_data[test_indices_volts]
    test_voltage = MotorDataPowerVoltCurve.voltage_data[test_indices_volts]
    actual_power_lim = MotorDataPowerVoltCurve.power_data[test_indices_volts]
    
    ivc2.add_output('rpm', test_rpm_volts, units='rpm', desc='Motor speed')
    ivc2.add_output('voltage', test_voltage, units='V', desc='Motor voltage')
    
    model2.add_subsystem('ivc', ivc2, promotes=['*'])
    
    #print("motor rpm volt pow interp")
    # Create the MetaModelStructuredComp directly (like in nacelle_splitter.py)
    motor_rpm_volt_pow_interp = om.MetaModelStructuredComp(vec_size=n_test_points_volts)
    motor_rpm_volt_pow_interp.add_input('rpm', 1790, training_data=vect_rpm, units="rpm", shape=(n_test_points_volts,))
    motor_rpm_volt_pow_interp.add_input('voltage', 800, training_data=vect_voltage, units="V", shape=(n_test_points_volts,))
    motor_rpm_volt_pow_interp.add_output('max_power_em', 2300, training_data=map_power_from_rpm_voltage, units="kW", shape=(n_test_points_volts,))
    motor_rpm_volt_pow_interp.options['extrapolate'] = False
    model2.add_subsystem('motor_rpm_volt_pow_interp', motor_rpm_volt_pow_interp, promotes_inputs=['*'], promotes_outputs=['*'])
    
    prob2 = om.Problem(model2, reports=False)
    prob2.setup()
    prob2.run_model()
    
    interpolated_power_lim = prob2.get_val('max_power_em', units='kW')
    power_errors = np.abs(interpolated_power_lim - actual_power_lim)
    power_relative_errors = power_errors / actual_power_lim * 100
    
    # Create plots if requested
    if plot_error:
        # Figure 1: Error Analysis Plots (3x3 grid)
        plt.figure(figsize=(22, 16))
        
        # Torque/RPM to Efficiency Error Analysis
        plt.subplot(3, 3, 1)
        plt.scatter(test_rpm, eff_relative_errors, alpha=0.6, s=30, c='blue')
        plt.xlabel('RPM')
        plt.ylabel('Efficiency Relative Error (%)')
        plt.title('Efficiency Error vs RPM')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.scatter(test_torque, eff_relative_errors, alpha=0.6, s=30, c='red')
        plt.xlabel('Torque (N⋅m)')
        plt.ylabel('Efficiency Relative Error (%)')
        plt.title('Efficiency Error vs Torque')
        plt.grid(True, alpha=0.3)
        
        # Error distribution by RPM ranges
        plt.subplot(3, 3, 3)
        rpm_ranges = [(0, 1000), (1000, 2000), (2000, 3000)]
        colors = ['blue', 'green', 'red']
        for i, (min_rpm, max_rpm) in enumerate(rpm_ranges):
            mask = (test_rpm >= min_rpm) & (test_rpm < max_rpm)
            if np.any(mask):
                plt.hist(eff_relative_errors[mask], bins=10, alpha=0.6, 
                        label=f'RPM {min_rpm}-{max_rpm}', color=colors[i])
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution by RPM Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Power/Voltage to Efficiency Error Analysis
        plt.subplot(3, 3, 4)
        plt.scatter(test_voltage_power_eff, eff_power_eff_relative_errors, alpha=0.6, s=30, c='purple')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Efficiency Relative Error (%)')
        plt.title('Efficiency Error vs Voltage')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.scatter(test_power_power_eff, eff_power_eff_relative_errors, alpha=0.6, s=30, c='orange')
        plt.xlabel('Power (kW)')
        plt.ylabel('Efficiency Relative Error (%)')
        plt.title('Efficiency Error vs Power')
        plt.grid(True, alpha=0.3)
        
        # Error distribution by power ranges
        plt.subplot(3, 3, 6)
        power_ranges = [(0, 200), (200, 400), (400, 600)]
        colors = ['blue', 'green', 'red']
        for i, (min_power, max_power) in enumerate(power_ranges):
            mask = (test_power_power_eff >= min_power) & (test_power_power_eff < max_power)
            if np.any(mask):
                plt.hist(eff_power_eff_relative_errors[mask], bins=10, alpha=0.6, 
                        label=f'Power {min_power}-{max_power}kW', color=colors[i])
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution by Power Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Voltage/Power Limit Error Analysis
        plt.subplot(3, 3, 7)
        plt.scatter(test_rpm_volts, power_relative_errors, alpha=0.6, s=30, c='green')
        plt.xlabel('RPM')
        plt.ylabel('Power Limit Relative Error (%)')
        plt.title('Power Limit Error vs RPM')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        plt.scatter(test_voltage, power_relative_errors, alpha=0.6, s=30, c='brown')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power Limit Relative Error (%)')
        plt.title('Power Limit Error vs Voltage')
        plt.grid(True, alpha=0.3)
        
        # Error distribution by voltage ranges
        plt.subplot(3, 3, 9)
        voltage_ranges = [(400, 600), (600, 800), (800, 1000)]
        colors = ['blue', 'green', 'red']
        for i, (min_voltage, max_voltage) in enumerate(voltage_ranges):
            mask = (test_voltage >= min_voltage) & (test_voltage < max_voltage)
            if np.any(mask):
                plt.hist(power_relative_errors[mask], bins=10, alpha=0.6, 
                        label=f'Voltage {min_voltage}-{max_voltage}V', color=colors[i])
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution by Voltage Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=4.5)
        plt.show()
        
        # Figure 2: Validation and Comparison Plots (2x3 grid)
        plt.figure(figsize=(18, 12))
        
        # Actual vs Interpolated comparison plots
        plt.subplot(2, 3, 1)
        plt.scatter(actual_eff, interpolated_eff, alpha=0.6, s=30, c='blue')
        plt.plot([actual_eff.min(), actual_eff.max()], [actual_eff.min(), actual_eff.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Efficiency')
        plt.ylabel('Interpolated Efficiency')
        plt.title('Torque/RPM: Actual vs Interpolated')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.scatter(actual_eff_power_eff, interpolated_eff_power_eff, alpha=0.6, s=30, c='red')
        plt.plot([actual_eff_power_eff.min(), actual_eff_power_eff.max()], 
                [actual_eff_power_eff.min(), actual_eff_power_eff.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Efficiency')
        plt.ylabel('Interpolated Efficiency')
        plt.title('Power/Voltage: Actual vs Interpolated')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.scatter(actual_power_lim, interpolated_power_lim, alpha=0.6, s=30, c='green')
        plt.plot([actual_power_lim.min(), actual_power_lim.max()], 
                [actual_power_lim.min(), actual_power_lim.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Power Limit (kW)')
        plt.ylabel('Interpolated Power Limit (kW)')
        plt.title('Voltage/RPM: Actual vs Interpolated')
        plt.grid(True, alpha=0.3)
        
        # Error histograms
        plt.subplot(2, 3, 4)
        plt.hist(eff_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Torque/RPM Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.hist(eff_power_eff_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='red')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Power/Voltage Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.hist(power_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Voltage/RPM Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.show()
    
    return {
        'efficiency': {
            'interpolated': interpolated_eff,
            'actual': actual_eff,
            'errors': eff_errors,
            'relative_errors': eff_relative_errors,
            'rpm': test_rpm,
            'torque': test_torque
        },
        'power_efficiency': {
            'interpolated': interpolated_eff_power_eff,
            'actual': actual_eff_power_eff,
            'errors': eff_power_eff_errors,
            'relative_errors': eff_power_eff_relative_errors,
            'voltage': test_voltage_power_eff,
            'power': test_power_power_eff
        },
        'voltage_power': {
            'interpolated': interpolated_power_lim,
            'actual': actual_power_lim,
            'errors': power_errors,
            'relative_errors': power_relative_errors,
            'rpm': test_rpm_volts,
            'voltage': test_voltage
        }
    }


def test_motor_efficiency_interpolation_max_error(tolerance=1):
    """Test motor efficiency interpolation maximum relative error"""
    results = _get_motor_interpolation_results()
    max_relative_error = np.max(results['efficiency']['relative_errors'])
    print(f"Motor efficiency interpolation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Motor efficiency interpolation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_motor_efficiency_interpolation_mean_error(tolerance=0.5):
    """Test motor efficiency interpolation mean relative error"""
    results = _get_motor_interpolation_results()
    mean_relative_error = np.mean(results['efficiency']['relative_errors'])
    print(f"Motor efficiency interpolation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Motor efficiency interpolation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"



def test_motor_power_efficiency_interpolation_max_error(tolerance=1):
    """Test motor power/efficiency interpolation maximum relative error"""
    results = _get_motor_interpolation_results()
    max_relative_error = np.max(results['power_efficiency']['relative_errors'])
    print(f"Motor power/efficiency interpolation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Motor power/efficiency interpolation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_motor_power_efficiency_interpolation_mean_error(tolerance=0.5):
    """Test motor power/efficiency interpolation mean relative error"""
    results = _get_motor_interpolation_results()
    mean_relative_error = np.mean(results['power_efficiency']['relative_errors'])
    print(f"Motor power/efficiency interpolation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Motor power/efficiency interpolation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_motor_voltage_power_interpolation_max_error(tolerance=1):
    """Test motor voltage/power interpolation maximum relative error"""
    results = _get_motor_interpolation_results()
    max_relative_error = np.max(results['voltage_power']['relative_errors'])
    print(f"Motor voltage/power interpolation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Motor voltage/power interpolation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_motor_voltage_power_interpolation_mean_error(tolerance=0.5):
    """Test motor voltage/power interpolation mean relative error"""
    results = _get_motor_interpolation_results()
    mean_relative_error = np.mean(results['voltage_power']['relative_errors'])
    print(f"Motor voltage/power interpolation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Motor voltage/power interpolation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_motor_efficiency_bounds():
    """Test that motor efficiency values are within reasonable bounds"""
    results = _get_motor_interpolation_results()
    interpolated_eff = results['efficiency']['interpolated']
    actual_eff = results['efficiency']['actual']
    
    # Check that interpolated values are within reasonable bounds
    assert np.all(interpolated_eff >= 0.0), "Motor efficiency interpolation produced negative values"
    assert np.all(interpolated_eff <= 1.0), "Motor efficiency interpolation produced values > 100%"
    assert np.all(actual_eff >= 0.0), "Motor efficiency actual data contains negative values"
    assert np.all(actual_eff <= 1.0), "Motor efficiency actual data contains values > 100%"


def test_motor_power_bounds():
    """Test that motor power values are within reasonable bounds"""
    results = _get_motor_interpolation_results()
    interpolated_power = results['voltage_power']['interpolated']
    actual_power = results['voltage_power']['actual']
    
    # Check that interpolated values are within reasonable bounds
    assert np.all(interpolated_power >= 0.0), "Motor power interpolation produced negative values"
    assert np.all(actual_power >= 0.0), "Motor power actual data contains negative values"


def run_motor_interpolation_accuracy_test( plot_error=True):
    """
    Test interpolation accuracy by comparing interpolated vs actual values
    """
    print("Testing motor interpolation accuracy...")
    
    # Get the interpolation results with plotting
    results = _get_motor_interpolation_results(plot_error)
    
    # Print summary statistics
    print(f"\n=== Motor Interpolation Results Summary ===")
    print(f"Efficiency Interpolation (Torque/RPM):")
    print(f"  Mean relative error: {np.mean(results['efficiency']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['efficiency']['relative_errors']):.2f}%")
    
    print(f"Power/Efficiency Interpolation (Power/Voltage):")
    print(f"  Mean relative error: {np.mean(results['power_efficiency']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['power_efficiency']['relative_errors']):.2f}%")
    
    print(f"Voltage/Power Limit Interpolation (RPM/Voltage):")
    print(f"  Mean relative error: {np.mean(results['voltage_power']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['voltage_power']['relative_errors']):.2f}%")
    
    # Call individual test functions
    test_motor_efficiency_interpolation_max_error(tolerance=0.1)
    test_motor_efficiency_interpolation_mean_error(tolerance=0.05)
    test_motor_power_efficiency_interpolation_max_error(tolerance=0.1)
    test_motor_power_efficiency_interpolation_mean_error(tolerance=0.05)
    test_motor_voltage_power_interpolation_max_error(tolerance=5.0)
    test_motor_voltage_power_interpolation_mean_error(tolerance=2.0)
    test_motor_efficiency_bounds()
    test_motor_power_bounds()
    
    return results


class ComputeMotorPower(om.ExplicitComponent):
    """
    Computes the mechanical power output of the motor given torque, rpm, and voltage command.
    Limits power based on voltage constraints and updates torque if needed.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_motors', default=4, desc='number of props')
        # Load data immediately when module is imported - this ensures it's available for all components

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_motors = self.options['num_motors']
        
        # Inputs
        self.add_input('torque', val=np.ones((num_motors, num_nodes)), units='N*m', 
                      desc='Commanded torque')
        self.add_input('rpm', val=np.ones((num_motors, num_nodes)), units='rpm', 
                      desc='Motor rotational speed')

        # Outputs
        self.add_output('mech_power', val=np.ones((num_motors, num_nodes)), units='W', 
                       desc='Mechanical power output')

        rows = np.arange(num_motors * num_nodes)
        cols_torque = np.arange(num_motors * num_nodes)
        cols_rpm = np.arange(num_motors * num_nodes)
        self.declare_partials('mech_power', 'torque', rows=rows, cols=cols_torque, method='exact')
        self.declare_partials('mech_power', 'rpm', rows=rows, cols=cols_rpm, method='exact')
        # Declare partials

    def compute(self, inputs, outputs):
        torque = inputs['torque']
        rpm = inputs['rpm']
        mech_power = outputs['mech_power']
        # Convert rpm to rad/s
        omega = rpm * 2 * np.pi / 60
        # Calculate requested mechanical power
        mech_power = torque * omega
        outputs['mech_power'] = mech_power
    
    def compute_partials(self, inputs, partials):

        torque = inputs['torque']
        rpm = inputs['rpm']
        # Convert rpm to rad/s
        omega = rpm * 2 * np.pi / 60
        # Calculate requested mechanical power
        mech_power = torque * omega
        partials['mech_power', 'torque'] = omega
        partials['mech_power', 'rpm'] = torque * 2 * np.pi / 60


class ComputeMotorElecDraw(om.ExplicitComponent):
    """
    Computes the electrical power draw of the motor based on mechanical power and efficiency.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_motors', default=4, desc='number of props')
        self.options.declare('gen_eff_drop', default=0.1, desc='Percentage drop degradation of motor operating in genearting mode')
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        num_motors = self.options['num_motors']
        
        # Inputs
        self.add_input('mech_power', val=np.ones((num_motors, num_nodes)), units='W', 
                      desc='Mechanical power output')
        self.add_input('eff', val=np.ones((num_motors, num_nodes)), units=None, 
                      desc='Motor efficiency')
        
        # Outputs
        self.add_output('p_train_elec', val=np.ones((num_motors, num_nodes)), units='W', 
                       desc='Electrical power draw')


        rows = np.arange(num_motors * num_nodes)
        cols_mech_power = np.arange(num_motors * num_nodes)
        cols_eff = np.arange(num_motors * num_nodes)
        self.declare_partials('p_train_elec', 'mech_power', rows=rows, cols=cols_mech_power, method='exact')
        self.declare_partials('p_train_elec', 'eff', rows=rows, cols=cols_eff, method='exact')
        
    def compute(self, inputs, outputs):

        mech_power = inputs['mech_power']
        eff = inputs['eff']
        p_train_elec = outputs['p_train_elec']
        gen_eff_drop = self.options['gen_eff_drop']
        # Avoid division by zero and negative efficiency
        eff_safe = np.clip(eff, 1e-6, 1.0 - 1e-6)
        # Discount efficiency for generating mode
        eff_safe = np.where(mech_power < 0, eff_safe * (1 - gen_eff_drop), eff_safe)    
        # Electrical power = mechanical power / efficiency
        p_train_elec = mech_power / (eff_safe)

        outputs['p_train_elec'] = p_train_elec

    def compute_partials(self, inputs, partials):

        mech_power = inputs['mech_power'].flatten()
        eff = inputs['eff'].flatten()

        gen_eff_drop = self.options['gen_eff_drop']
        
        # Compute eff_safe the same way as in compute()
        eff_clipped = np.clip(eff, 1e-6, 1.0 - 1e-6)
        
        # Derivative of clip function: 1 if within bounds, 0 if at bounds
        d_clip_d_eff = np.where((eff > 1e-6) & (eff < 1.0 - 1e-6), 1.0, 0.0)
        
        # Apply conditional scaling for generator mode
        is_generating = mech_power < 0
        scale_factor = np.where(is_generating, 1 - gen_eff_drop, 1.0)
        eff_safe = eff_clipped * scale_factor
        
        # ∂(p_train_elec)/∂(mech_power) = 1 / eff_safe
        # (ignoring discontinuity at mech_power = 0)
        partials['p_train_elec', 'mech_power'] = 1.0 / eff_safe
        
        # ∂(p_train_elec)/∂(eff) = -mech_power / eff_safe^2 * ∂(eff_safe)/∂(eff)
        # where ∂(eff_safe)/∂(eff) = d_clip_d_eff * scale_factor
        d_eff_safe_d_eff = d_clip_d_eff * scale_factor
        partials['p_train_elec', 'eff'] = -mech_power / (eff_safe**2) * d_eff_safe_d_eff
    
class EmpiricalMotor(om.Group):
    """
    Complete motor group that combines interpolation with power computation.
    """

    _data_loaded = False
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_motors', default=4, desc='number of motors')
        self.options.declare('torque_rpm_set', default=False, desc='True when torque and RPM are set by the user and power is an output')
        self.options.declare('throttle_set', default=False, desc='Specify absolute power or motor throttle')
        self.options.declare('gen_eff_drop', default=0.1, desc='Percentage drop degradation of motor operating in genearting mode')

        self._load_data()

    @classmethod
    def _load_data(cls):

        if cls._data_loaded:
            return
        
        #print("loading motor data")
        # Load motor data
        MotorDataEffMap.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM_2300_eff.xlsx')
        MotorDataPowerEffCurve.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')

        cls._rpm_tq_eff_rpm = MotorDataEffMap.rpm_data
        cls._rpm_tq_eff_torque = MotorDataEffMap.torque_data
        cls._rpm_tq_eff_eff = MotorDataEffMap.eff_data

        _pow_eff_volts_unique = np.unique(MotorDataPowerEffCurve.voltage_data)
        cls._pow_eff_volts_unique =_pow_eff_volts_unique
        _pow_eff_power_kW_unique = np.unique(MotorDataPowerEffCurve.power_data__W/1000)
        cls._pow_eff_power_kW_unique = _pow_eff_power_kW_unique
        _pow_eff_eff_unique = MotorDataPowerEffCurve.eff_data
        cls._pow_eff_eff_unique = _pow_eff_eff_unique.reshape(len(_pow_eff_volts_unique),len(_pow_eff_power_kW_unique)).transpose()

        cls._data_loaded = True

        #print("Motor Data loaded.")
    

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_motors = self.options['num_motors']
        torque_rpm_set = self.options['torque_rpm_set']
        throttle_set = self.options['throttle_set']
        gen_eff_drop = self.options['gen_eff_drop']

        # Add all subsystems
        if torque_rpm_set:
            self.add_subsystem('compute_power', ComputeMotorPower(num_nodes=num_nodes, num_motors=num_motors), promotes=['*'])

            # Convert matrix inputs to vectors for MetaModel
            self.add_subsystem('matrix_to_vector', 
                              MatrixToVectorConverter(
                                  num_nodes=num_nodes, 
                                  num_comps=num_motors,
                                  input_names=['rpm', 'torque'],
                                  units={'rpm': 'rpm', 'torque': 'N*m'}
                              ), 
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])


            # MetaModel for efficiency interpolation
            motor_rpm_tq_eff_interp = om.MetaModelUnStructuredComp(
                vec_size=num_nodes*num_motors, 
                default_surrogate=om.ResponseSurface()
            )
            motor_rpm_tq_eff_interp.add_input('rpm_vect', 1.0, training_data=self._rpm_tq_eff_rpm, units="rpm", shape=(num_motors*num_nodes,))
            motor_rpm_tq_eff_interp.add_input('torque_vect', 1.0, training_data=self._rpm_tq_eff_torque, units="N*m", shape=(num_motors*num_nodes,))
            motor_rpm_tq_eff_interp.add_output('eff_vect', 1.0, training_data=self._rpm_tq_eff_eff, units=None, shape=(num_motors*num_nodes,))

            self.add_subsystem('motor_rpm_tq_eff_interp', motor_rpm_tq_eff_interp, promotes_inputs=['*'], promotes_outputs=['*'])

            # Convert vector output back to matrix
            self.add_subsystem('vector_to_matrix', 
                              VectorToMatrixConverter(
                                  num_nodes=num_nodes, 
                                  num_comps=num_motors,
                                  input_names=['eff_vect'],
                                  output_names=['eff'],
                                  units={'eff_vect': None}
                              ), 
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])


        else:
            if throttle_set:

                nacelle_throttle_to_power = ElementMultiplyDivideComp()
                nacelle_throttle_to_power.add_equation(output_name="mech_power", 
                                        input_names=["max_power", "throttle"], 
                                        vec_size=[num_motors, num_nodes],
                                        input_units=["kW", None],
                                        divide=[False, False])
                self.add_subsystem("nacelle_throttle_to_power", nacelle_throttle_to_power, promotes_inputs=["*"], promotes_outputs=["*"])   
                
                # Convert matrix inputs to vectors for MetaModel (throttle_set=True case)
                self.add_subsystem('matrix_to_vector_pow_eff', 
                                  MatrixToVectorConverter(
                                      num_nodes=num_nodes, 
                                      num_comps=num_motors,
                                      input_names=['mech_power', 'voltage'],
                                      units={'mech_power': 'kW', 'voltage': 'V'}
                                  ), 
                                  promotes_inputs=['*'], promotes_outputs=['*'])
                
            else:
                # Convert matrix inputs to vectors for MetaModel (throttle_set=False case)
                self.add_subsystem('matrix_to_vector_pow_eff', 
                                  MatrixToVectorConverter(
                                      num_nodes=num_nodes, 
                                      num_comps=num_motors,
                                      input_names=['mech_power', 'voltage'],
                                      units={'mech_power': 'kW', 'voltage': 'V'}
                                  ), 
                                  promotes_inputs=['*'], promotes_outputs=['*'])

            # print("motor pow eff interp")
            # MetaModel for power/voltage efficiency interpolation
            motor_pow_eff_interp = om.MetaModelStructuredComp(
                vec_size=num_nodes*num_motors, 
                method='akima'
            )
            motor_pow_eff_interp.add_input('mech_power_vect', 1500, training_data=self._pow_eff_power_kW_unique, units="kW", shape=(num_motors*num_nodes,))
            motor_pow_eff_interp.add_input('voltage_vect', 800, training_data=self._pow_eff_volts_unique, units="V", shape=(num_motors*num_nodes,))
            motor_pow_eff_interp.add_output('eff_vect', 0.95, training_data=self._pow_eff_eff_unique, units=None, shape=(num_motors*num_nodes,))
            motor_pow_eff_interp.options['extrapolate'] = True
            self.add_subsystem('motor_pow_eff_interp', motor_pow_eff_interp, promotes_inputs=['*'], promotes_outputs=['*'])

            # Convert vector output back to matrix
            self.add_subsystem('vector_to_matrix_pow_eff', 
                              VectorToMatrixConverter(
                                  num_nodes=num_nodes, 
                                  num_comps=num_motors,
                                  input_names=['eff_vect'],
                                  output_names=['eff'],
                                  units={'eff_vect': None}
                              ), promotes_inputs=['*'],
                              promotes_outputs=['*'])
            

            self.set_input_defaults('voltage', 850 * np.ones((num_motors, num_nodes)), units = 'V')
        # end 

        # Add the electrical power computation
        self.add_subsystem('compute_elec', ComputeMotorElecDraw(num_nodes=num_nodes, num_motors=num_motors, gen_eff_drop=gen_eff_drop), promotes=['*'])

      # end

def test_motor_components():
    """
    Test the new motor power computation components
    """
    print("Testing motor power computation components...")
    
    num_nodes = 5
    num_motors = 1
    start_time = time.time()

    
    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()

    torque_rpm_set = False
    test_interp = True  # whether to test the linear nd interpolation accuracy 
    throttle_set = False # whether to command the motor via throttle or absolute power

    voltage_val = 500 * np.ones((num_motors, num_nodes))
    mech_power_val = 1500 * np.ones((num_motors, num_nodes))  # kW
    
    ivc.add_output('voltage', voltage_val, units='V', desc='Motor voltage')

    # Add independent variables
    if torque_rpm_set:
        ivc.add_output('torque', 5000 * np.ones((num_motors, num_nodes)), units='N*m', desc='Commanded torque')
    else:
        if throttle_set:
            ivc.add_output('max_power', 2300e3 , units='W', desc='Max power')
            ivc.add_output('throttle', 0.5 * np.ones((num_motors, num_nodes)), desc='Throttle fraction')
        else:
            ivc.add_output('mech_power', mech_power_val, units='kW', desc='Commanded power')
    ivc.add_output('rpm', 1000 * np.ones((num_motors, num_nodes)), units='rpm', desc='Motor speed')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('motor', EmpiricalMotor(num_nodes=num_nodes, 
                                                torque_rpm_set=torque_rpm_set, 
                                                throttle_set = throttle_set,
                                                num_motors = num_motors,
                                                ), promotes=['*'])
    
    prob = om.Problem(model, reports=False)
    prob.setup()
    run_start_time = time.time()
    #om.n2(prob)
    
    # Run the model
    prob.run_model()

    end_time = time.time()

    print(f"Setup time: {run_start_time - start_time:.2f} seconds")
    print(f"Execution time: {end_time - run_start_time:.2f} seconds")
    print(f"Total time: {end_time - start_time:.2f} seconds")


    
    # Get results
    mech_power = prob.get_val('mech_power', units='W')
    p_train_elec = prob.get_val('p_train_elec', units='W')
    eff = prob.get_val('eff')


    
    print(f"Results for {num_nodes} nodes:")
    if torque_rpm_set:
        print(f"Commanded torque: {prob.get_val('torque', units='N*m')}")
        print(f"RPM: {prob.get_val('rpm', units='rpm')}")
        print(f"Voltage: {prob.get_val('voltage', units='V')}")

    print(f"Commanded Mechanical power: {prob.get_val('mech_power', units='kW')}")

    print(f"Efficiency: {eff}")

    # Build NearestNDInterpolator directly for comparison (like run_prop_model)
    if test_interp:
        # Load data for nearest neighbor interpolation
        MotorDataPowerEffCurve.load_data(motor_filename='models/atlas/atlas/propulsion/empirical_data/H3X_HPDM-XXXX_1MW.xlsx')
        
        # Build nearest neighbor interpolator as a simple function
        voltage_power_eff_interpolator_nearest = NearestNDInterpolator(
            np.column_stack([MotorDataPowerEffCurve.voltage_data, MotorDataPowerEffCurve.power_data__W]),
            MotorDataPowerEffCurve.eff_data
        )
        
        # Query the nearest neighbor interpolator with the same inputs
        # Flatten voltage and power for query, then reshape back
        voltage_flat = voltage_val.flatten()
        power_flat = mech_power_val.flatten() * 1000  # Convert kW to W
        eff_nearest = voltage_power_eff_interpolator_nearest(voltage_flat, power_flat)
        eff_nearest = eff_nearest.reshape(num_motors, num_nodes)
        
        print(f"Nearest Neighbour Efficiency: {eff_nearest}")
    
    print(f"Electrical power: {p_train_elec} W")

    
    # Check partials
    #print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    

    # Create visualization
    nodes = np.arange(0,num_nodes)
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Power profile
    plt.subplot(3, 3, 1)
    plt.plot(nodes, prob.get_val('mech_power', units='kW').flatten() , 'b-', linewidth=2, marker='o')
    plt.xlabel('Nodes')
    plt.ylabel('Mechanical Power (kW)')
    plt.grid(True, alpha=0.3)
    
    if torque_rpm_set:

        # Plot 2: Torque Profile
        plt.subplot(3, 3, 2)
        plt.plot(nodes, prob.get_val('torque', units='N*m').flatten(), 'g-', linewidth=2, marker='o')
        plt.xlabel('Nodes')
        plt.ylabel('Torque (Nm)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: RPM profile
        plt.subplot(3, 3, 3)
        plt.plot(nodes, prob.get_val('rpm', units='rpm').flatten(), 'r-', linewidth=2, marker='o')
        plt.legend()
        plt.xlabel('Nodes')
        plt.ylabel('RPM')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency profile
    plt.subplot(3, 3, 4)
    for i in range(num_motors):
        plt.plot(nodes, eff[i, :], 'purple', linewidth=2, marker='o', alpha=1.0, label=f'Metamodel Motor{i+1}')
    if test_interp:
        for i in range(num_motors):
            plt.plot(nodes, eff_nearest[i, :], 'purple', linewidth=2, marker='o', alpha=0.2, label=f'Nearest Neighbour Motor{i+1}')

    plt.legend()
    plt.xlabel('Nodes')
    plt.ylabel('Efficiency (%)')
    plt.grid(True, alpha=0.3)

    plt.show()

    

    



if __name__ == "__main__":
    
    # Test interpolation accuracy with new test structure
    test_neural_net = False

    #_get_motor_interpolation_results(test_neural_net=test_neural_net)
    run_motor_interpolation_accuracy_test(plot_error=True)

    # Test motor components
    #test_motor_components()
    #print("Completed")
