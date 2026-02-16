import numpy as np
import openmdao.api as om
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp
# Import the existing battery data




class BatteryMMInterpolationGroup(om.Group):
    """
    Group containing battery interpolation components with vectorization support.
    Uses MetaModelStructuredComp to interpolate from SOC and power to line voltage and current.
    """
    
    _training_data_loaded = False
    _loaded_datasheet_name = None
    _boundary_clipper_func = None
    
    @classmethod
    def _load_training_data(cls, battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105'):
        """Load and prepare training data for the battery interpolation."""
        # Check if we already have this datasheet loaded
        if cls._training_data_loaded and cls._loaded_datasheet_name == battery_datasheet_name:
            return
            
        # Load the clean battery database
        try:
            df = pd.read_csv('models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '_battery_power_database_full.csv')
            #print(f"Loaded battery database with {len(df)} points")
        except FileNotFoundError:   
            raise FileNotFoundError("' + battery_datasheet_name + '_battery_power_database_full.csv' not found. Please run build_battery_database.py to create the database.")
        
        # Get unique SOC and power values for training data
        soc_values = np.sort(df['soc'].unique())
        power_values = np.sort(df['power'].unique())
        
        #print(f"Training data ranges:")
        #print(f"  SOC: {soc_values.min():.3f} to {soc_values.max():.3f} ({len(soc_values)} points)")
        #print(f"  Power: {power_values.min():.0f}W to {power_values.max():.0f}W ({len(power_values)} points)")
        
        # Create training data grids
        soc_mesh, power_mesh = np.meshgrid(soc_values, power_values, indexing='ij')
        
        # Prepare points for griddata interpolation
        grid_points = np.column_stack([soc_mesh.flatten(), power_mesh.flatten()])
        data_points = np.column_stack([df['soc'], df['power']])
        
        # Use griddata to interpolate line voltage and current data
        line_voltage_training = griddata(data_points, df['line_voltage'], 
                                        grid_points, method='cubic', fill_value=3.5).reshape(soc_mesh.shape)
        current_training = griddata(data_points, df['current'], 
                                   grid_points, method='cubic', fill_value=200).reshape(soc_mesh.shape)
        
        # Store training data as class attributes
        cls._soc_values = soc_values
        cls._power_values = power_values
        cls._line_voltage_training = line_voltage_training
        cls._current_training = current_training
        
        # Track which datasheet is loaded
        cls._loaded_datasheet_name = battery_datasheet_name
        
        # Create boundary clipper function once during initialization
        # cls._create_boundary_clipper()
               
        cls._training_data_loaded = True

    
    def initialize(self):
        # Declare battery_datasheet_name option
        self.options.declare('battery_datasheet_name', 
                            default='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105',
                            desc='Name of the battery datasheet to use for interpolation')
        
        self.options.declare('nn', default=1, desc='number of nodes to evaluate')
        # These will be computed in setup() after loading training data
        self.options.declare('v_min', default=None, desc='Minimum simulated voltage for the battery cell')
        self.options.declare('v_max', default=None, desc='Maximum simulated voltage for the battery cell')
        self.options.declare('i_min', default=None, desc='Minimum simulated current for the battery cell')
        self.options.declare('i_max', default=None, desc='Maximum simulated current for the battery cell')
        self.options.declare('mu', default=0.001, desc='Smoothing parameter for the max/min functions')

    def setup(self):
        # Load training data first, using the option value (now available in setup)
        battery_datasheet_name = self.options['battery_datasheet_name']
        self._load_training_data(battery_datasheet_name=battery_datasheet_name)
        
        # Unpack Options
        nn = self.options['nn']
        # Compute v_min, v_max, i_min, i_max from training data if not provided
        v_min = self.options['v_min'] if self.options['v_min'] is not None else np.min(self._line_voltage_training)
        v_max = self.options['v_max'] if self.options['v_max'] is not None else np.max(self._line_voltage_training)
        i_min = self.options['i_min'] if self.options['i_min'] is not None else np.min(self._current_training)
        i_max = self.options['i_max'] if self.options['i_max'] is not None else np.max(self._current_training)
        mu = self.options['mu']
        
        # Create line voltage interpolator
        # Using 'scipy_cubic' instead of 'scipy_quintic' to avoid oscillatory behavior
        line_voltage_interp = om.MetaModelStructuredComp(vec_size=nn, method='scipy_slinear')
        line_voltage_interp.add_input('soc_clipped', 0.5, training_data=self._soc_values, units=None, shape=(nn,))
        line_voltage_interp.add_input('p_cell', 750, training_data=self._power_values, units='W', shape=(nn,))
        line_voltage_interp.add_output('vline_cell_raw', 3.5, training_data=self._line_voltage_training, units='V', shape=(nn,), upper = v_max, lower = np.min(self._line_voltage_training))
        line_voltage_interp.options['extrapolate'] = True
        line_voltage_interp.options['always_opt'] = True

        self.add_subsystem('line_voltage_interp', line_voltage_interp, promotes_inputs=['soc_clipped', 'p_cell'], promotes_outputs=[('vline_cell_raw', 'vline_cell')])
        
        # TO-DO: Set up interpoltion accuracy test for line voltage interpolation

        # Create current interpolator
        # Using 'scipy_cubic' instead of 'scipy_quintic' to avoid oscillatory behavior
        current_interp = om.MetaModelStructuredComp(vec_size=nn, method='scipy_slinear')
        current_interp.add_input('soc_clipped', 0.5, training_data=self._soc_values, units=None, shape=(nn,))
        current_interp.add_input('p_cell', 750, training_data=self._power_values, units='W', shape=(nn,))
        current_interp.add_output('i_cell_raw', 50, training_data=self._current_training, units='A', shape=(nn,), upper = i_max, lower = np.min(self._current_training))
        current_interp.options['extrapolate'] = True
        current_interp.options['always_opt'] = True
        self.add_subsystem('current_interp', current_interp, promotes_inputs=['soc_clipped', 'p_cell'], promotes_outputs= [])
        
        # Add simple voltage limiter component

        #self.add_subsystem('limit_min_voltage', SmoothMaxComp(num_nodes=nn, mode='limit', units='V', limit_val=2.5, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_voltage', SmoothMinComp(num_nodes=nn, mode='limit', units='V', limit_val=v_max, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'vline_cell')])

        #self.add_subsystem('limit_min_voltage', DiscontMaxComp(num_nodes=nn, mode='limit', units='V', limit_val=v_min, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_voltage', DiscontMinComp(num_nodes=nn, mode='limit', units='V', limit_val=v_max, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'vline_cell')])


        #self.connect('line_voltage_interp.vline_cell_raw', 'limit_min_voltage.input_array')
        #self.connect('limit_min_voltage.output', 'limit_max_voltage.input_array')

        self.add_subsystem('limit_min_current', SmoothMaxComp(num_nodes=nn, mode='limit', units='A', limit_val=i_min, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('limit_max_current', SmoothMinComp(num_nodes=nn, mode='limit', units='A', limit_val=i_max, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'i_cell')])
        #self.add_subsystem('limit_min_current', SmoothMaxComp(num_nodes=nn, mode='limit', units='A', limit_val=i_min, n_comps=1, expected_min=i_min, expected_max=i_max), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_current', SmoothMinComp(num_nodes=nn, mode='limit', units='A', limit_val=i_max, n_comps=1, expected_min=i_min, expected_max=i_max), promotes_inputs=[], promotes_outputs=[('output', 'i_cell')])

        #self.add_subsystem('limit_min_current', DiscontMaxComp(num_nodes=nn, mode='limit', units='A', limit_val=i_min, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_current', DiscontMinComp(num_nodes=nn, mode='limit', units='A', limit_val=i_max, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'i_cell')])

        self.connect('current_interp.i_cell_raw', 'limit_min_current.input_array')
        self.connect('limit_min_current.output', 'limit_max_current.input_array')
        # Connect voltage interpolator to voltage limiter


        
      

def _get_battery_interpolation_results(plot_error=False, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Get battery interpolation results with comprehensive error analysis."""
    print("Testing battery interpolation group...")
    
    # Create test model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Load actual data points from the database for testing
    df = pd.read_csv('models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '_battery_power_database_full.csv')
    
    # Set up random number generator and pick random test points
    np.random.seed(42)  # For reproducible results
    n_test_points = min(50, len(df))  # Limit to 50 points for faster testing
    test_indices = np.random.choice(len(df), size=n_test_points, replace=False)
    test_data = df.iloc[test_indices]
    
    nn = len(test_data)
    ivc.add_output('soc_clipped', test_data['soc'].values, units=None)
    ivc.add_output('p_cell', test_data['power'].values, units='W')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('battery_interp', BatteryMMInterpolationGroup(nn=nn, battery_datasheet_name=battery_datasheet_name), promotes=['*'])
    
    # Setup and run
    prob = om.Problem(model, reports = False)
    prob.setup()
    prob.run_model()
    
    # Get all values at once
    soc_vals = prob.get_val('soc_clipped')
    power_vals = prob.get_val('p_cell')
    line_voltage_vals = prob.get_val('vline_cell')
    current_vals = prob.get_val('i_cell')
    calculated_power_vals = line_voltage_vals * current_vals
    
    # Get original values from database for comparison
    original_line_voltage_vals = test_data['line_voltage'].values
    original_current_vals = test_data['current'].values
    
    # Calculate errors
    power_errors = np.abs(calculated_power_vals - power_vals)
    line_voltage_errors = np.abs(line_voltage_vals - original_line_voltage_vals)
    current_errors = np.abs(current_vals - original_current_vals)
    
    # Calculate relative errors
    line_voltage_relative_errors = line_voltage_errors / original_line_voltage_vals * 100
    current_relative_errors = current_errors / np.abs(original_current_vals) * 100
    power_relative_errors = power_errors / power_vals * 100
    
    # Create plots if requested
    if plot_error:
        # Figure 1: Error Analysis Plots (2x3 grid)
        plt.figure(figsize=(18, 12))
        
        # Line Voltage Error Analysis
        plt.subplot(2, 3, 1)
        plt.scatter(soc_vals, line_voltage_relative_errors, alpha=0.6, s=30, c='blue')
        plt.xlabel('SOC')
        plt.ylabel('Line Voltage Relative Error (%)')
        plt.title('Line Voltage Error vs SOC')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.scatter(power_vals, line_voltage_relative_errors, alpha=0.6, s=30, c='red')
        plt.xlabel('Power (W)')
        plt.ylabel('Line Voltage Relative Error (%)')
        plt.title('Line Voltage Error vs Power')
        plt.grid(True, alpha=0.3)
        
        # Error distribution by SOC ranges
        plt.subplot(2, 3, 3)
        soc_ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        colors = ['blue', 'green', 'red']
        for i, (min_soc, max_soc) in enumerate(soc_ranges):
            mask = (soc_vals >= min_soc) & (soc_vals < max_soc)
            if np.any(mask):
                plt.hist(line_voltage_relative_errors[mask], bins=10, alpha=0.6, 
                        label=f'SOC {min_soc}-{max_soc}', color=colors[i])
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Line Voltage Error Distribution by SOC Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Current Error Analysis
        plt.subplot(2, 3, 4)
        plt.scatter(soc_vals, current_relative_errors, alpha=0.6, s=30, c='purple')
        plt.xlabel('SOC')
        plt.ylabel('Current Relative Error (%)')
        plt.title('Current Error vs SOC')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.scatter(power_vals, current_relative_errors, alpha=0.6, s=30, c='orange')
        plt.xlabel('Power (W)')
        plt.ylabel('Current Relative Error (%)')
        plt.title('Current Error vs Power')
        plt.grid(True, alpha=0.3)
        
        # Error distribution by power ranges
        plt.subplot(2, 3, 6)
        power_ranges = [(0, 200), (200, 500), (500, 1000)]
        colors = ['blue', 'green', 'red']
        for i, (min_power, max_power) in enumerate(power_ranges):
            mask = (power_vals >= min_power) & (power_vals < max_power)
            if np.any(mask):
                plt.hist(current_relative_errors[mask], bins=10, alpha=0.6, 
                        label=f'Power {min_power}-{max_power}W', color=colors[i])
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Current Error Distribution by Power Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        # Figure 2: Validation and Comparison Plots (2x3 grid)
        plt.figure(figsize=(18, 12))
        
        # Actual vs Interpolated comparison plots
        plt.subplot(2, 3, 1)
        plt.scatter(original_line_voltage_vals, line_voltage_vals, alpha=0.6, s=30, c='blue')
        plt.plot([original_line_voltage_vals.min(), original_line_voltage_vals.max()], 
                [original_line_voltage_vals.min(), original_line_voltage_vals.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Line Voltage (V)')
        plt.ylabel('Interpolated Line Voltage (V)')
        plt.title('Line Voltage: Actual vs Interpolated')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.scatter(original_current_vals, current_vals, alpha=0.6, s=30, c='red')
        plt.plot([original_current_vals.min(), original_current_vals.max()], 
                [original_current_vals.min(), original_current_vals.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Current (A)')
        plt.ylabel('Interpolated Current (A)')
        plt.title('Current: Actual vs Interpolated')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.scatter(power_vals, calculated_power_vals, alpha=0.6, s=30, c='green')
        plt.plot([power_vals.min(), power_vals.max()], 
                [power_vals.min(), power_vals.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Power (W)')
        plt.ylabel('Calculated Power (W)')
        plt.title('Power Conservation Check')
        plt.grid(True, alpha=0.3)
        
        # Error histograms
        plt.subplot(2, 3, 4)
        plt.hist(line_voltage_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Line Voltage Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.hist(current_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='red')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Current Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.hist(power_relative_errors, bins=15, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Power Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.show()
    
    return {
        'line_voltage': {
            'interpolated': line_voltage_vals,
            'actual': original_line_voltage_vals,
            'errors': line_voltage_errors,
            'relative_errors': line_voltage_relative_errors,
            'soc': soc_vals,
            'power': power_vals
        },
        'current': {
            'interpolated': current_vals,
            'actual': original_current_vals,
            'errors': current_errors,
            'relative_errors': current_relative_errors,
            'soc': soc_vals,
            'power': power_vals
        },
        'power_conservation': {
            'calculated': calculated_power_vals,
            'actual': power_vals,
            'errors': power_errors,
            'relative_errors': power_relative_errors
        }
    }


def test_battery_interpolation(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test the battery interpolation group."""
    print("Testing battery interpolation group...")
    
    # Create test model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Load actual data points from the database for testing
    df = pd.read_csv('models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '_battery_power_database_full.csv')
    
    # Set up random number generator and pick random test points
    np.random.seed(42)  # For reproducible results
    n_test_points = len(df['soc'])
    test_indices = np.random.choice(len(df), size=n_test_points, replace=False)
    test_data = df.iloc[test_indices]
    
    nn = len(test_data)
    ivc.add_output('soc_clipped', test_data['soc'].values, units=None)
    ivc.add_output('p_cell', test_data['power'].values, units='W')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('battery_interp', BatteryMMInterpolationGroup(nn=nn, battery_datasheet_name=battery_datasheet_name), promotes=['*'])
    
    # Setup and run
    prob = om.Problem(model, reports = False)
    prob.setup()
    #om.n2(prob)
    prob.run_model()
    
    # Print results
    print("\nTest results (using actual database points):")
    
    # Get all values at once
    soc_vals = prob.get_val('soc_clipped')
    power_vals = prob.get_val('p_cell')
    line_voltage_vals = prob.get_val('vline_cell')
    current_vals = prob.get_val('i_cell')
    calculated_power_vals = line_voltage_vals * current_vals
    
    # Get original values from database for comparison
    original_line_voltage_vals = test_data['line_voltage'].values
    original_current_vals = test_data['current'].values
    
    # Calculate errors
    power_errors = np.abs(calculated_power_vals - power_vals)
    line_voltage_errors = np.abs(line_voltage_vals - original_line_voltage_vals)
    current_errors = np.abs(current_vals - original_current_vals)
    
    # Print summary
    print(f"  Tested {nn} points from database")
    print(f"  SOC range: {soc_vals.min():.3f} to {soc_vals.max():.3f}")
    print(f"  Power range: {power_vals.min():.0f}W to {power_vals.max():.0f}W")
    print(f"  Line voltage range: {line_voltage_vals.min():.3f}V to {line_voltage_vals.max():.3f}V")
    print(f"  Current range: {current_vals.min():.3f}A to {current_vals.max():.3f}A")
    print(f"  Max power error: {power_errors.max():.2f}W")
    print(f"  Max line voltage error: {line_voltage_errors.max():.3f}V")
    print(f"  Max current error: {current_errors.max():.3f}A")
    print(f"  Mean power error: {power_errors.mean():.2f}W")
    print(f"  Mean line voltage error: {line_voltage_errors.mean():.3f}V")
    print(f"  Mean current error: {current_errors.mean():.3f}A")
    
    return prob


def test_battery_line_voltage_interpolation_max_error(tolerance=5.0, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery line voltage interpolation maximum relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    max_relative_error = np.max(results['line_voltage']['relative_errors'])
    print(f"Battery line voltage interpolation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Battery line voltage interpolation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_line_voltage_interpolation_mean_error(tolerance=2.0, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery line voltage interpolation mean relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    mean_relative_error = np.mean(results['line_voltage']['relative_errors'])
    print(f"Battery line voltage interpolation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Battery line voltage interpolation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_current_interpolation_max_error(tolerance=10.0, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery current interpolation maximum relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    max_relative_error = np.max(results['current']['relative_errors'])
    print(f"Battery current interpolation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Battery current interpolation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_current_interpolation_mean_error(tolerance=5.0, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery current interpolation mean relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    mean_relative_error = np.mean(results['current']['relative_errors'])
    print(f"Battery current interpolation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Battery current interpolation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_power_conservation_max_error(tolerance=1.0, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery power conservation maximum relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    max_relative_error = np.max(results['power_conservation']['relative_errors'])
    print(f"Battery power conservation max relative error: {max_relative_error:.6f}%")
    assert max_relative_error < tolerance, f"Battery power conservation max relative error {max_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_power_conservation_mean_error(tolerance=0.5, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test battery power conservation mean relative error"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    mean_relative_error = np.mean(results['power_conservation']['relative_errors'])
    print(f"Battery power conservation mean relative error: {mean_relative_error:.6f}%")
    assert mean_relative_error < tolerance, f"Battery power conservation mean relative error {mean_relative_error:.6f}% exceeds tolerance {tolerance}%"


def test_battery_voltage_bounds(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test that battery voltage values are within reasonable bounds"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    voltage_vals = results['line_voltage']['interpolated']
    
    # Check voltage bounds (typical Li-ion cell voltage range)
    assert np.all(voltage_vals >= 2.5), f"Battery voltage below minimum (2.5V): {voltage_vals.min():.3f}V"
    assert np.all(voltage_vals <= 4.2), f"Battery voltage above maximum (4.2V): {voltage_vals.max():.3f}V"
    
    print(f"✓ Battery voltage bounds check passed: {voltage_vals.min():.3f}V to {voltage_vals.max():.3f}V")


def test_battery_current_bounds(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Test that battery current values are within reasonable bounds"""
    results = _get_battery_interpolation_results(battery_datasheet_name=battery_datasheet_name)
    current_vals = results['current']['interpolated']
    
    # Check current bounds (reasonable for battery applications)
    assert np.all(current_vals >= -500), f"Battery current below minimum (-500A): {current_vals.min():.3f}A"
    assert np.all(current_vals <= 500), f"Battery current above maximum (500A): {current_vals.max():.3f}A"
    
    print(f"✓ Battery current bounds check passed: {current_vals.min():.3f}A to {current_vals.max():.3f}A")


def run_battery_interpolation_accuracy_test(plot_error=False, battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """Run comprehensive battery interpolation accuracy tests with optional plotting."""
    print("Testing battery interpolation accuracy...")
    
    # Get the interpolation results with plotting
    results = _get_battery_interpolation_results(plot_error, battery_datasheet_name=battery_datasheet_name)
    
    # Print summary statistics
    print(f"\n=== Battery Interpolation Results Summary ===")
    print(f"Line Voltage Interpolation:")
    print(f"  Mean relative error: {np.mean(results['line_voltage']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['line_voltage']['relative_errors']):.2f}%")
    
    print(f"Current Interpolation:")
    print(f"  Mean relative error: {np.mean(results['current']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['current']['relative_errors']):.2f}%")
    
    print(f"Power Conservation:")
    print(f"  Mean relative error: {np.mean(results['power_conservation']['relative_errors']):.2f}%")
    print(f"  Max relative error: {np.max(results['power_conservation']['relative_errors']):.2f}%")
    
    # Call individual test functions
    test_battery_line_voltage_interpolation_max_error(tolerance=5.0, battery_datasheet_name=battery_datasheet_name)
    test_battery_line_voltage_interpolation_mean_error(tolerance=2.0, battery_datasheet_name=battery_datasheet_name)
    test_battery_current_interpolation_max_error(tolerance=10.0, battery_datasheet_name=battery_datasheet_name)
    test_battery_current_interpolation_mean_error(tolerance=5.0, battery_datasheet_name=battery_datasheet_name)
    test_battery_power_conservation_max_error(tolerance=1.0, battery_datasheet_name=battery_datasheet_name)
    test_battery_power_conservation_mean_error(tolerance=0.5, battery_datasheet_name=battery_datasheet_name)
    
    return results


def create_3d_battery_visualization(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """
    Create 3D visualizations of battery cell voltage and current across SOC and power values.
    Sweeps SOC from 0 to 1 and power from 0 to 500W.
    Also plots actual data points from the database for comparison.
    """
    print("Creating 3D battery visualization...")

    power_max_W = 1250
    
    # Load actual data points from the database for comparison
    try:
        df = pd.read_csv('models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '_battery_power_database_full.csv')
        print(f"Loaded {len(df)} actual data points from database")
    except FileNotFoundError:
        print("Warning: Could not load database file. Only showing interpolation surface.")
        df = None
    
    # Filter actual data points to the range of interest (0-1 SOC, 0-500W power)
    if df is not None:
        mask = (df['soc'] >= 0) & (df['soc'] <= 1) & (df['power'] >= 0) & (df['power'] <= power_max_W)
        filtered_df = df[mask]
        print(f"Filtered to {len(filtered_df)} points within SOC [0,1] and power [0,500W]")
    
    # Create test model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Create fine grid for visualization
    soc_points = 50  # Number of SOC points
    power_points = 50  # Number of power points
    
    soc_range = np.linspace(1e-6, 1, soc_points)
    power_range = np.linspace(1e-6, power_max_W, power_points)
    
    # Create meshgrid for 3D plotting
    soc_mesh, power_mesh = np.meshgrid(soc_range, power_range, indexing='ij')
    
    # Flatten for OpenMDAO
    soc_flat = soc_mesh.flatten()
    power_flat = power_mesh.flatten()
    nn = len(soc_flat)
    
    ivc.add_output('soc_clipped', soc_flat, units=None)
    ivc.add_output('p_cell', power_flat, units='W')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    model.add_subsystem('battery_interp', BatteryMMInterpolationGroup(nn=nn, battery_datasheet_name=battery_datasheet_name), promotes=['*'])
    
    # Setup and run
    prob = om.Problem(model, reports=False)
    prob.setup()
    prob.run_model()
    
    # Get results
    soc_vals = prob.get_val('soc_clipped')
    power_vals = prob.get_val('p_cell')
    line_voltage_vals = prob.get_val('vline_cell')
    current_vals = prob.get_val('i_cell')
    
    # Reshape back to 2D for plotting
    soc_2d = soc_vals.reshape(soc_points, power_points)
    power_2d = power_vals.reshape(soc_points, power_points)
    line_voltage_2d = line_voltage_vals.reshape(soc_points, power_points)
    current_2d = current_vals.reshape(soc_points, power_points)
    
    # Create separate figures for each plot
    
    # Figure 1: Line Voltage vs SOC and Power
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Plot the interpolation surface
    surf1 = ax1.plot_surface(soc_2d, power_2d, line_voltage_2d, cmap='viridis', alpha=0.7, label='Interpolation Surface')
    
    # Plot actual data points if available
    if df is not None and len(filtered_df) > 0:
        scatter1 = ax1.scatter(filtered_df['soc'], filtered_df['power'], filtered_df['line_voltage'], 
                              c='red', s=20, alpha=0.8, label='Actual Data Points')
    
    ax1.set_xlabel('SOC')
    ax1.set_ylabel('Power (W)')
    ax1.set_zlabel('Line Voltage (V)')
    ax1.set_title('Battery Cell Line Voltage vs SOC and Power\n(Interpolation Surface + Actual Data Points)')
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Add legend
    if df is not None and len(filtered_df) > 0:
        ax1.legend()
    
    plt.show()
    
    # Figure 2: Current vs SOC and Power
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Plot the interpolation surface
    surf2 = ax2.plot_surface(soc_2d, power_2d, current_2d, cmap='plasma', alpha=0.7, label='Interpolation Surface')
    
    # Plot actual data points if available
    if df is not None and len(filtered_df) > 0:
        scatter2 = ax2.scatter(filtered_df['soc'], filtered_df['power'], filtered_df['current'], 
                              c='red', s=20, alpha=0.8, label='Actual Data Points')
    
    ax2.set_xlabel('SOC')
    ax2.set_ylabel('Power (W)')
    ax2.set_zlabel('Current (A)')
    ax2.set_title('Battery Cell Current vs SOC and Power\n(Interpolation Surface + Actual Data Points)')
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # Add legend
    if df is not None and len(filtered_df) > 0:
        ax2.legend()
    
    plt.show()
    
    # Print summary statistics
    print(f"\n3D Visualization Summary:")
    print(f"  SOC range: {soc_vals.min():.3f} to {soc_vals.max():.3f}")
    print(f"  Power range: {power_vals.min():.0f}W to {power_vals.max():.0f}W")
    print(f"  Line voltage range: {line_voltage_vals.min():.3f}V to {line_voltage_vals.max():.3f}V")
    print(f"  Current range: {current_vals.min():.3f}A to {current_vals.max():.3f}A")
    
    # Calculate and verify power conservation
    calculated_power = line_voltage_vals * current_vals
    power_errors = np.abs(calculated_power - power_vals)
    print(f"  Max power error: {power_errors.max():.2f}W")
    print(f"  Mean power error: {power_errors.mean():.2f}W")
    
    # If we have actual data, calculate interpolation errors at those points
    if df is not None and len(filtered_df) > 0:
        print(f"\nInterpolation Accuracy at Actual Data Points:")
        print(f"  Number of comparison points: {len(filtered_df)}")
        
        # Create a test model with actual data points
        test_model = om.Group()
        test_ivc = om.IndepVarComp()
        
        test_ivc.add_output('soc_clipped', filtered_df['soc'].values, units=None)
        test_ivc.add_output('p_cell', filtered_df['power'].values, units='W')
        
        test_model.add_subsystem('ivc', test_ivc, promotes=['*'])
        test_model.add_subsystem('battery_interp', BatteryMMInterpolationGroup(nn=len(filtered_df), battery_datasheet_name=battery_datasheet_name), promotes=['*'])
        
        test_prob = om.Problem(test_model, reports=False)
        test_prob.setup()
        test_prob.run_model()
        
        # Get interpolated values
        interp_voltage = test_prob.get_val('vline_cell')
        interp_current = test_prob.get_val('i_cell')
        
        # Calculate absolute errors
        voltage_errors_abs = np.abs(interp_voltage - filtered_df['line_voltage'].values)
        current_errors_abs = np.abs(interp_current - filtered_df['current'].values)
        
        # Calculate percentage errors (relative to actual values)
        actual_voltage = filtered_df['line_voltage'].values
        actual_current = filtered_df['current'].values
        
        # Avoid division by zero - use small epsilon for very small values
        voltage_errors_pct = np.abs(interp_voltage - actual_voltage) / np.maximum(np.abs(actual_voltage), 1e-6) * 100
        current_errors_pct = np.abs(interp_current - actual_current) / np.maximum(np.abs(actual_current), 1e-6) * 100
        
        print(f"  Max voltage error: {voltage_errors_abs.max():.3f}V ({voltage_errors_pct.max():.2f}%)")
        print(f"  Mean voltage error: {voltage_errors_abs.mean():.3f}V ({voltage_errors_pct.mean():.2f}%)")
        print(f"  Max current error: {current_errors_abs.max():.3f}A ({current_errors_pct.max():.2f}%)")
        print(f"  Mean current error: {current_errors_abs.mean():.3f}A ({current_errors_pct.mean():.2f}%)")
        
        # Create 2D contour plots of interpolation errors (percentage)
        # Interpolate errors onto a regular grid for contour plotting
        # Create regular grid for contour plot
        soc_grid = np.linspace(filtered_df['soc'].min(), filtered_df['soc'].max(), 100)
        power_grid = np.linspace(filtered_df['power'].min(), filtered_df['power'].max(), 100)
        soc_grid_mesh, power_grid_mesh = np.meshgrid(soc_grid, power_grid, indexing='ij')
        
        # Prepare data points for interpolation
        data_points = np.column_stack([filtered_df['soc'].values, filtered_df['power'].values])
        grid_points = np.column_stack([soc_grid_mesh.flatten(), power_grid_mesh.flatten()])
        
        # Interpolate voltage percentage errors onto grid
        voltage_error_grid = griddata(data_points, voltage_errors_pct, grid_points, 
                                     method='cubic', fill_value=0).reshape(soc_grid_mesh.shape)
        
        # Interpolate current percentage errors onto grid
        current_error_grid = griddata(data_points, current_errors_pct, grid_points, 
                                     method='cubic', fill_value=0).reshape(soc_grid_mesh.shape)
        
        # Figure 3: Voltage Error Contour Plot (Percentage)
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        contour3 = ax3.contourf(soc_grid_mesh, power_grid_mesh, voltage_error_grid, 
                               levels=20, cmap='Reds', extend='max')
        ax3.set_xlabel('SOC', fontsize=12)
        ax3.set_ylabel('Power (W)', fontsize=12)
        ax3.set_title('Line Voltage Interpolation Error\n(Percentage Error: |Predicted - Actual| / |Actual| × 100%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(contour3, ax=ax3, label='Voltage Error (%)')
        plt.tight_layout()
        plt.show()
        
        # Figure 4: Current Error Contour Plot (Percentage)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        contour4 = ax4.contourf(soc_grid_mesh, power_grid_mesh, current_error_grid, 
                               levels=20, cmap='Reds', extend='max')
        ax4.set_xlabel('SOC', fontsize=12)
        ax4.set_ylabel('Power (W)', fontsize=12)
        ax4.set_title('Current Interpolation Error\n(Percentage Error: |Predicted - Actual| / |Actual| × 100%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        cbar4 = plt.colorbar(contour4, ax=ax4, label='Current Error (%)')
        plt.tight_layout()
        plt.show()
    


def run_battery_interpolation_results(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):
    """
    Test function that calls BatteryMMInterpolationGroup with a mesh sweep of SOC and power values.
    """
    import time
    
    print("="*60)
    print("BATTERY INTERPOLATION MESH SWEEP")
    print("="*60)
    
    start_time = time.time()
    
    # Create mesh parameters
    nn = 30  # Number of points in each direction
    soc_range = np.linspace(0.001, 1, nn)  # SOC from 0.001 to 0.3 (avoid exact 0)
    power_range = np.linspace(50, 500, nn)  # Power from 50W to 500W
    
    # Create meshgrid
    soc_mesh, power_mesh = np.meshgrid(soc_range, power_range, indexing='ij')
    
    # Flatten for OpenMDAO
    soc_flat = soc_mesh.flatten()
    power_flat = power_mesh.flatten()
    total_nodes = len(soc_flat)
    
    print(f"Creating mesh with {total_nodes} points:")
    print(f"  SOC range: {soc_range.min():.3f} to {soc_range.max():.3f} ({nn} points)")
    print(f"  Power range: {power_range.min():.0f}W to {power_range.max():.0f}W ({nn} points)")
    
    # Create OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add inputs
    ivc.add_output('soc_clipped', soc_flat, units=None, desc='State of charge array')
    ivc.add_output('p_cell', power_flat, units='W', desc='Cell power array')
    
    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add battery interpolation group
    model.add_subsystem('battery_interp', 
                       BatteryMMInterpolationGroup(nn=total_nodes, battery_datasheet_name=battery_datasheet_name), 
                       promotes=['*'])
    
    # Create problem
    prob = om.Problem(model, reports=False)
    prob.setup()
    
    # Run the model
    run_start_time = time.time()
    prob.run_model()
    end_time = time.time()
    
    print(f"Setup time: {run_start_time - start_time:.4f} seconds")
    print(f"Execution time: {end_time - run_start_time:.4f} seconds")
    print(f"Total Runtime: {end_time - start_time:.4f} seconds")
    
    # Get results
    vline_cell = prob.get_val('vline_cell', units='V')
    i_cell = prob.get_val('i_cell', units='A')
    
    # Calculate power conservation
    calculated_power = vline_cell * i_cell
    power_errors = np.abs(calculated_power - power_flat)
    power_error_percent = (power_errors / power_flat) * 100
    
    # Reshape for analysis
    soc_2d = soc_flat.reshape(nn, nn)
    power_2d = power_flat.reshape(nn, nn)
    voltage_2d = vline_cell.reshape(nn, nn)
    current_2d = i_cell.reshape(nn, nn)
    power_error_2d = power_errors.reshape(nn, nn)
    power_error_percent_2d = power_error_percent.reshape(nn, nn)
    
    # Print comprehensive analysis
    print("\n" + "="*60)
    print("MESH SWEEP RESULTS")
    print("="*60)
    
    print(f"Voltage Analysis:")
    print(f"  Min voltage: {vline_cell.min():.4f} V")
    print(f"  Max voltage: {vline_cell.max():.4f} V")
    print(f"  Mean voltage: {vline_cell.mean():.4f} V")
    print(f"  Std voltage: {vline_cell.std():.4f} V")
    
    print(f"\nCurrent Analysis:")
    print(f"  Min current: {i_cell.min():.4f} A")
    print(f"  Max current: {i_cell.max():.4f} A")
    print(f"  Mean current: {i_cell.mean():.4f} A")
    print(f"  Std current: {i_cell.std():.4f} A")
    
    print(f"\nPower Conservation Analysis:")
    print(f"  Max power error: {power_errors.max():.4f} W")
    print(f"  Mean power error: {power_errors.mean():.4f} W")
    print(f"  Max power error %: {power_error_percent.max():.4f}%")
    print(f"  Mean power error %: {power_error_percent.mean():.4f}%")
    
    # Identify problematic regions
    high_error_mask = power_error_percent > 5.0  # More than 5% error
    if np.any(high_error_mask):
        print(f"\nProblematic Regions (>5% power error):")
        print(f"  Number of problematic points: {np.sum(high_error_mask)}")
        print(f"  Percentage of total: {100*np.sum(high_error_mask)/total_nodes:.2f}%")
        
        # Show worst cases
        worst_indices = np.argsort(power_error_percent)[-10:]  # Top 10 worst
        print(f"\nTop 10 worst cases:")
        for i, idx in enumerate(worst_indices):
            print(f"  {i+1:2d}. SOC={soc_flat[idx]:.3f}, P={power_flat[idx]:.0f}W, "
                  f"Error={power_error_percent[idx]:.2f}%")
    
    # Create diagnostic plots
    print(f"\nCreating diagnostic plots...")
    
    # Figure 1: Voltage and Current Surfaces
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    
    # Voltage surface
    im1 = ax1.contourf(soc_2d, power_2d, voltage_2d, levels=20, cmap='viridis')
    # Add cutoff voltage line at 2.5V
    ax1.contour(soc_2d, power_2d, voltage_2d, levels=[2.5], colors='red', linewidths=3, linestyles='--')
    ax1.set_xlabel('SOC', fontsize=12)
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Interpolated Voltage Surface\n(Red dashed line = 2.5V cutoff)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    plt.colorbar(im1, ax=ax1, label='Voltage (V)')
    
    # Current surface
    im2 = ax2.contourf(soc_2d, power_2d, current_2d, levels=20, cmap='plasma')
    ax2.set_xlabel('SOC', fontsize=12)
    ax2.set_ylabel('Power (W)', fontsize=12)
    ax2.set_title('Interpolated Current Surface', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    plt.colorbar(im2, ax=ax2, label='Current (A)')
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    
    print("="*60)
    print("MESH SWEEP ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'soc_mesh': soc_2d,
        'power_mesh': power_2d,
        'voltage_mesh': voltage_2d,
        'current_mesh': current_2d,
        'power_errors': power_error_2d,
        'power_error_percent': power_error_percent_2d,
        'high_error_mask': high_error_mask,
        'statistics': {
            'voltage_range': (vline_cell.min(), vline_cell.max()),
            'current_range': (i_cell.min(), i_cell.max()),
            'max_power_error': power_errors.max(),
            'mean_power_error': power_errors.mean(),
            'max_power_error_percent': power_error_percent.max(),
            'mean_power_error_percent': power_error_percent.mean(),
            'problematic_points': np.sum(high_error_mask),
            'problematic_percentage': 100*np.sum(high_error_mask)/total_nodes
        }
    }

if __name__ == "__main__":
    # Run comprehensive battery interpolation accuracy tests
    #run_battery_interpolation_accuracy_test(plot_error=True, battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105')
    
    # Simplified test function
    #test_battery_interpolation()
    
    # Run battery interpolatioCn test
    print("\n" + "="*80)
    #run_battery_interpolation_results()
    

    # Uncomment the line below to create 3D visualization
    create_3d_battery_visualization(battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105')


