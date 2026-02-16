import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SortedTurboData:
    """
    Class to store sorted turbo data vectors and matrices for direct loading into metamodels.
    Similar to TurboData class but with pre-sorted and matrix-formatted data.
    """
    
    # Class variables to store the data
    disa_data__degC = None
    alt_data__m = None
    mach_data = None
    frac_data = None
    power_data__kW = None
    fuel_flow_data__kgph = None
    jet_thrust_data__N = None
    
    # Matrix data (4D arrays)
    fuel_flow_throttle_matrix = None  # Fuel flow based on throttle input
    power_matrix = None               # Power output based on throttle input
    max_power_matrix = None           # Max power for each (DISA, Alt, Mach) combination
    jet_thrust_matrix = None          # Jet thrust based on throttle input

    
    # Dimension vectors
    disa_unique = None
    alt_unique = None
    mach_unique = None
    throttle_unique = None
    
    # Data loading flag
    _data_loaded = False
    
    @classmethod
    def load_data(cls, csv_filename):
        """
        Load sorted turbo data from CSV and convert to vectors and matrices.
        
        Parameters:
        -----------
        csv_filename : str
            Path to the sorted turbo data CSV file
        """
        if cls._data_loaded:
            return
            
        #print(f"Loading sorted turbo data from: {csv_filename}")
        
        # Load the sorted data from CSV
        sorted_data = pd.read_csv(csv_filename)
        
        # Define column names
        input_columns = ['DISA', 'Altitude', 'Mach', 'FRAC']
        output_columns = ['power_kW', 'FuelFlow_kgph', 'JetThrust_N']
        
        # Sort the data to make input columns monotonically increasing
        sorted_data = sorted_data.sort_values(by=input_columns).reset_index(drop=True)
        
        
        # Get unique values for each input dimension
        cls.disa_unique = sorted(sorted_data['DISA'].unique())
        cls.alt_unique = sorted(sorted_data['Altitude'].unique())
        cls.mach_unique = sorted(sorted_data['Mach'].unique())
        cls.throttle_unique = sorted(sorted_data['FRAC'].unique())
        
        # Create 4D matrices efficiently using pandas operations
        # Set up the data for reshaping
        sorted_data_indexed = sorted_data.set_index(['DISA', 'Altitude', 'Mach', 'FRAC'])
        
        # Create fuel flow matrix based on throttle input
        fuel_flow_throttle_unstacked = sorted_data_indexed['FuelFlow_kgph'].unstack(level=-1)
        cls.fuel_flow_throttle_matrix = fuel_flow_throttle_unstacked.values.reshape(
            len(cls.disa_unique), len(cls.alt_unique), 
            len(cls.mach_unique), len(cls.throttle_unique)
        )
        
        # Create power matrix (throttle-based)
        power_unstacked = sorted_data_indexed['power_kW'].unstack(level=-1)
        cls.power_matrix = power_unstacked.values.reshape(
            len(cls.disa_unique), len(cls.alt_unique), 
            len(cls.mach_unique), len(cls.throttle_unique)
        )

        # Create jet thrust matrix based on throttle input
        jet_thrust_unstacked = sorted_data_indexed['JetThrust_N'].unstack(level=-1)
        cls.jet_thrust_matrix = jet_thrust_unstacked.values.reshape(
            len(cls.disa_unique), len(cls.alt_unique), 
            len(cls.mach_unique), len(cls.throttle_unique)
        )
        
        # Create max power matrix for each (DISA, Alt, Mach) combination
        # Since FRAC cycles from 0 to 1 repeatedly, we need to find the actual max power for each operating condition
        #print(f"  Creating max power matrix...")
        
        # Use numpy's max function along the throttle axis (axis=3)
        cls.max_power_matrix = np.max(cls.power_matrix, axis=3)
        
        #print(f"    Max power matrix shape: {cls.max_power_matrix.shape}")
        #print(f"    Max power range: {np.min(cls.max_power_matrix):.1f} to {np.max(cls.max_power_matrix):.1f} kW")
        
        # Verify we have max power values for all operating conditions
        zero_power_count = np.sum(cls.max_power_matrix <= 0)
        """
        if zero_power_count > 0:
            print(f"    WARNING: Found {zero_power_count} operating conditions with zero or negative max power!")
        else:
            print(f"    ✓ All operating conditions have positive max power values")
        """
        # Store the original vector data for compatibility with existing code
        cls.disa_data__degC = sorted_data['DISA'].values
        cls.alt_data__m = sorted_data['Altitude'].values
        cls.mach_data = sorted_data['Mach'].values
        cls.frac_data = sorted_data['FRAC'].values
        cls.power_data__kW = sorted_data['power_kW'].values
        cls.fuel_flow_data__kgph = sorted_data['FuelFlow_kgph'].values
        cls.jet_thrust_data__N = sorted_data['JetThrust_N'].values
        
        cls._data_loaded = True
        """
        
        print(f"Data loaded successfully:")
        print(f"  Vector data shape: {len(cls.disa_data__degC)} points")
        print(f"  Fuel flow throttle matrix shape: {cls.fuel_flow_throttle_matrix.shape}")
        print(f"  Power matrix shape: {cls.power_matrix.shape}")
        print(f"  Max power matrix shape: {cls.max_power_matrix.shape}")
        print(f"  Unique values:")
        print(f"    DISA: {len(cls.disa_unique)} values")
        print(f"    Altitude: {len(cls.alt_unique)} values")
        print(f"    Mach: {len(cls.mach_unique)} values")
        print(f"    Throttle: {len(cls.throttle_unique)} values")
        """
    
    @classmethod
    def power_to_throttle(cls, disa, altitude, mach, power_kw):
        """
        Convert power to throttle fraction using power ratio.
        
        Parameters:
        -----------
        disa : float
            DISA temperature in degC
        altitude : float
            Altitude in meters
        mach : float
            Mach number
        power_kw : float
            Power in kW
            
        Returns:
        --------
        float : Throttle fraction (0.0 to 1.0)
        """
        # Find the closest operating condition indices
        disa_idx = np.argmin(np.abs(np.array(cls.disa_unique) - disa))
        alt_idx = np.argmin(np.abs(np.array(cls.alt_unique) - altitude))
        mach_idx = np.argmin(np.abs(np.array(cls.mach_unique) - mach))
        
        # Get the max power for this operating condition
        max_power = cls.max_power_matrix[disa_idx, alt_idx, mach_idx]
        
        # Calculate throttle as ratio of power to max power
        throttle_val = power_kw / max_power
        
        # Clamp to valid range
        throttle_val = np.clip(throttle_val, 0.0, 1.0)
        
        return throttle_val

def import_sorted_turbo_data(csv_filename):
    """
    Import sorted turbo data from CSV and convert to multidimensional matrix format.
    
    Parameters:
    -----------
    csv_filename : str
        Path to the sorted turbo data CSV file
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'sorted_data': pandas DataFrame with sorted data
        - 'fuel_flow_matrix': 4D numpy array (disa, alt, mach, throttle)
        - 'power_matrix': 4D numpy array (disa, alt, mach, throttle)
        - 'dimensions': dict with dimension info
        - 'input_columns': list of input column names
        - 'output_columns': list of output column names
    """
    
    # Load the data using the class
    SortedTurboData.load_data(csv_filename)
    
    # Create dimension mapping
    dimensions = {
        'disa': SortedTurboData.disa_unique,
        'alt': SortedTurboData.alt_unique,
        'mach': SortedTurboData.mach_unique,
        'throttle': SortedTurboData.throttle_unique
    }
    
    # Define column names
    input_columns = ['DISA', 'Altitude', 'Mach', 'FRAC']
    output_columns = ['power_kW', 'FuelFlow_kgph', 'JetThrust_N']
    
    # Create sorted DataFrame for compatibility
    sorted_data = pd.DataFrame({
        'DISA': SortedTurboData.disa_data__degC,
        'Altitude': SortedTurboData.alt_data__m,
        'Mach': SortedTurboData.mach_data,
        'FRAC': SortedTurboData.frac_data,
        'power_kW': SortedTurboData.power_data__kW,
        'FuelFlow_kgph': SortedTurboData.fuel_flow_data__kgph,
        'JetThrust_N': SortedTurboData.jet_thrust_data__N
    })
    
    return {
        'sorted_data': sorted_data,
        'fuel_flow_throttle_matrix': SortedTurboData.fuel_flow_throttle_matrix,
        'power_matrix': SortedTurboData.power_matrix,
        'max_power_matrix': SortedTurboData.max_power_matrix,
        'jet_thrust_matrix': SortedTurboData.jet_thrust_matrix,
        'dimensions': dimensions,
        'input_columns': input_columns,
        'output_columns': output_columns
    }

def print_matrix_info(data_dict):
    """
    Print information about the loaded and processed data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary returned by import_sorted_turbo_data
    """
    
    print("="*60)
    print("TURBO DATA MATRIX INFORMATION")
    print("="*60)
    
    print(f"Original data shape: {data_dict['sorted_data'].shape}")
    print(f"Fuel flow throttle matrix shape: {data_dict['fuel_flow_throttle_matrix'].shape}")
    print(f"Power matrix shape: {data_dict['power_matrix'].shape}")
    print(f"Max power matrix shape: {data_dict['max_power_matrix'].shape}")
    print(f"Jet thrust matrix shape: {data_dict['jet_thrust_matrix'].shape}")

    print("\nDimension sizes:")
    for dim_name, dim_values in data_dict['dimensions'].items():
        print(f"  {dim_name}: {len(dim_values)} values - {dim_values}")
    
    print(f"\nInput columns: {data_dict['input_columns']}")
    print(f"Output columns: {data_dict['output_columns']}")
    
    # Check for NaN values
    fuel_throttle_nan_count = np.isnan(data_dict['fuel_flow_throttle_matrix']).sum()
    power_nan_count = np.isnan(data_dict['power_matrix']).sum()
    max_power_nan_count = np.isnan(data_dict['max_power_matrix']).sum()
    jet_thrust_nan_count = np.isnan(data_dict['jet_thrust_matrix']).sum()
    
    print(f"\nNaN values in fuel flow throttle matrix: {fuel_throttle_nan_count}")
    print(f"NaN values in power matrix: {power_nan_count}")
    print(f"NaN values in max power matrix: {max_power_nan_count}")
    print(f"NaN values in jet thrust matrix: {jet_thrust_nan_count}")
    
    # Show some statistics
    print(f"\nFuel flow throttle statistics:")
    print(f"  Min: {np.nanmin(data_dict['fuel_flow_throttle_matrix']):.2f} kg/h")
    print(f"  Max: {np.nanmax(data_dict['fuel_flow_throttle_matrix']):.2f} kg/h")
    print(f"  Mean: {np.nanmean(data_dict['fuel_flow_throttle_matrix']):.2f} kg/h")
    
    print(f"\nMax power statistics:")
    print(f"  Min: {np.nanmin(data_dict['max_power_matrix']):.2f} kW")
    print(f"  Max: {np.nanmax(data_dict['max_power_matrix']):.2f} kW")
    print(f"  Mean: {np.nanmean(data_dict['max_power_matrix']):.2f} kW")
    
    print(f"\nPower statistics:")
    print(f"  Min: {np.nanmin(data_dict['power_matrix']):.2f} kW")
    print(f"  Max: {np.nanmax(data_dict['power_matrix']):.2f} kW")
    print(f"  Mean: {np.nanmean(data_dict['power_matrix']):.2f} kW")

    print(f"\nJet thrust statistics:")
    print(f"  Min: {np.nanmin(data_dict['jet_thrust_matrix']):.2f} N")
    print(f"  Max: {np.nanmax(data_dict['jet_thrust_matrix']):.2f} N")
    print(f"  Mean: {np.nanmean(data_dict['jet_thrust_matrix']):.2f} N")

# Example usage
if __name__ == "__main__":
    csv_filename = 'aviary/models/engines/propulsion/empirical_data/sorted_turbo_dataset.csv'
    
    print("Loading sorted turbo data from CSV...")
    data_dict = import_sorted_turbo_data(csv_filename)
    
    print_matrix_info(data_dict)
    
    # Test direct access to class data
    print("\n" + "="*60)
    print("TESTING DIRECT CLASS ACCESS")
    print("="*60)
    print(f"Vector data available:")
    print(f"  DISA data shape: {SortedTurboData.disa_data__degC.shape}")
    print(f"  Altitude data shape: {SortedTurboData.alt_data__m.shape}")
    print(f"  Mach data shape: {SortedTurboData.mach_data.shape}")
    print(f"  Throttle data shape: {SortedTurboData.frac_data.shape}")
    print(f"  Power data shape: {SortedTurboData.power_data__kW.shape}")
    print(f"  Fuel flow data shape: {SortedTurboData.fuel_flow_data__kgph.shape}")
    print(f"  Jet thrust data shape: {SortedTurboData.jet_thrust_data__N.shape}")

    print(f"\nMatrix data available:")
    print(f"  Fuel flow throttle matrix shape: {SortedTurboData.fuel_flow_throttle_matrix.shape}")
    print(f"  Power matrix shape: {SortedTurboData.power_matrix.shape}")
    print(f"  Max power matrix shape: {SortedTurboData.max_power_matrix.shape}")
    print(f"  Jet thrust matrix shape: {SortedTurboData.jet_thrust_matrix.shape}")
    
    # Test power_to_throttle conversion accuracy
    print(f"\nTesting power_to_throttle conversion accuracy:")
    
    # Get a sample operating condition
    sample_disa = SortedTurboData.disa_unique[len(SortedTurboData.disa_unique)//2]
    sample_alt = SortedTurboData.alt_unique[len(SortedTurboData.alt_unique)//2]
    sample_mach = SortedTurboData.mach_unique[len(SortedTurboData.mach_unique)//2]
    
    # Get the power curve for this condition
    disa_idx = SortedTurboData.disa_unique.index(sample_disa)
    alt_idx = SortedTurboData.alt_unique.index(sample_alt)
    mach_idx = SortedTurboData.mach_unique.index(sample_mach)
    
    power_curve = SortedTurboData.power_matrix[disa_idx, alt_idx, mach_idx, :]
    throttle_curve = SortedTurboData.throttle_unique
    
    print(f"Sample condition: DISA={sample_disa}°C, Alt={sample_alt}m, Mach={sample_mach}")
    
    # Test conversion at various power levels
    test_powers = np.linspace(power_curve[0], power_curve[-1], 20)
    
    # Get actual throttles from data
    power_indices = np.argmin(np.abs(power_curve[:, None] - test_powers), axis=0)
    actual_throttles = np.array(throttle_curve)[power_indices]
    
    # Convert power to throttle using our function
    converted_throttles = np.array([SortedTurboData.power_to_throttle(sample_disa, sample_alt, sample_mach, power) for power in test_powers])
    
    # Print results
    for i, power in enumerate(test_powers):
        print(f"  Power={power:.1f}kW: Nearest={actual_throttles[i]:.3f}, Converted={converted_throttles[i]:.3f}")
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(test_powers, actual_throttles, 'b-o', label='Nearest Throttle', markersize=4)
    plt.plot(test_powers, converted_throttles, 'r--s', label='Converted Throttle', markersize=4)
    plt.xlabel('Power (kW)')
    plt.ylabel('Throttle Fraction')
    plt.title('Power to Throttle Conversion Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
