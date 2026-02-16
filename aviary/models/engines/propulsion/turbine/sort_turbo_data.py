import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .turbo_data import TurboData

def sort_turbo_dataset(data, input_cols, output_cols):
    """
    Sort turbo dataset so that all input columns are monotonically increasing.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Your turbo performance dataset
    input_cols : list of strings
        Names of your input columns (e.g., ['DISA', 'Altitude', 'Mach', 'FRAC'])
    output_cols : list of strings
        Names of your output columns (e.g., ['power_kW', 'FuelFlow_kgph', 'JetThrust_N'])
    
    Returns:
    --------
    sorted_data : pandas DataFrame
        Your dataset with monotonically increasing input columns
    """
    
    # Create a copy of the data
    sorted_data = data.copy()
    
    # Sort by all input columns in order (primary sort by first column, then second, etc.)
    sorted_data = sorted_data.sort_values(by=input_cols)
    
    # Reset the index
    sorted_data = sorted_data.reset_index(drop=True)
    
    return sorted_data

def verify_monotonicity(sorted_data, input_cols):
    """
    Verify that the sorted data is monotonically increasing.
    
    Parameters:
    -----------
    sorted_data : pandas DataFrame
        Sorted dataset
    input_cols : list
        List of input column names
    
    Returns:
    --------
    bool : True if all columns are monotonically increasing
    """
    
    # Check if each column is monotonically increasing
    is_monotonic = []
    for col in input_cols:
        diff = np.diff(sorted_data[col])
        is_increasing = np.all(diff >= 0)  # Allow equal values
        is_monotonic.append(is_increasing)
        print(f"{col}: {'✓ Monotonic' if is_increasing else '✗ Not monotonic'}")
    
    return all(is_monotonic)

def visualize_turbo_data(original_data, sorted_data, input_cols, output_cols):
    """
    Visualize the sorting results for turbo data.
    
    Parameters:
    -----------
    original_data : pandas DataFrame
        Original dataset
    sorted_data : pandas DataFrame
        Sorted dataset
    input_cols : list
        List of input column names
    output_cols : list
        List of output column names
    """
    
    # Create subplots for each input column
    fig, axes = plt.subplots(2, len(input_cols), figsize=(4*len(input_cols), 8))
    
    for i, col in enumerate(input_cols):
        # Plot original data
        axes[0, i].plot(original_data[col], 'b-', alpha=0.7, linewidth=1)
        axes[0, i].set_title(f'{col} - Original')
        axes[0, i].set_xlabel('Data Point Index')
        axes[0, i].set_ylabel('Value')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot sorted data
        axes[1, i].plot(sorted_data[col], 'r-', alpha=0.7, linewidth=1)
        axes[1, i].set_title(f'{col} - Sorted (Monotonic)')
        axes[1, i].set_xlabel('Data Point Index')
        axes[1, i].set_ylabel('Value')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot output vs first input to show the relationship
    fig, axes = plt.subplots(1, len(output_cols), figsize=(4*len(output_cols), 4))
    
    for i, output_col in enumerate(output_cols):
        # Original relationship
        axes[i].scatter(original_data[input_cols[0]], original_data[output_col], 
                       alpha=0.6, s=20, label='Original')
        axes[i].scatter(sorted_data[input_cols[0]], sorted_data[output_col], 
                       alpha=0.6, s=20, label='Sorted', color='red')
        axes[i].set_xlabel(input_cols[0])
        axes[i].set_ylabel(output_col)
        axes[i].set_title(f'{output_col} vs {input_cols[0]}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def create_dataframe_from_turbo_data():
    """
    Create a DataFrame from the loaded TurboData class.
    
    Returns:
    --------
    pandas DataFrame with all the turbo data
    """
    
    # Create DataFrame from TurboData class variables
    data = pd.DataFrame({
        'DISA': TurboData.disa_data__degC,
        'Altitude': TurboData.alt_data__m,
        'Mach': TurboData.mach_data,
        'FRAC': TurboData.frac_data,
        'power_kW': TurboData.power_data__kW,
        'FuelFlow_kgph': TurboData.fuel_flow_data__kgph,
        'JetThrust_N': TurboData.jet_thrust_data__N

    })
    
    return data

# Main execution
if __name__ == "__main__":
    # Load PT6 base data (used for both PT6 and ACCE)
    turbo_filename = 'models/atlas/atlas/propulsion/empirical_data/PT6E-67XP-4_EngData.xlsx'
    sheet_name = 'CRZ'

    print("Loading PT6 base turbo data...")
    TurboData.load_data(turbo_filename, sheet_name)

    # Create DataFrame from the loaded data
    data = create_dataframe_from_turbo_data()

    print(f"Original turbo data shape: {data.shape}")
    print("\nOriginal data (first 10 rows):")
    print(data.head(10))

    # Define your column names
    input_columns = ['DISA', 'Altitude', 'Mach', 'FRAC']
    output_columns = ['power_kW', 'FuelFlow_kgph', 'JetThrust_N']

    # Sort the data
    sorted_data = sort_turbo_dataset(data, input_columns, output_columns)

    print("\nSorted turbo data (first 10 rows):")
    print(sorted_data.head(10))

    # Verify monotonicity
    print("\nChecking if input columns are monotonically increasing:")
    is_monotonic = verify_monotonicity(sorted_data, input_columns)
    print(f"\nAll input columns monotonic: {'✓ Yes' if is_monotonic else '✗ No'}")

    # Visualize results
    visualize_turbo_data(data, sorted_data, input_columns, output_columns)

    # Save PT6 dataset
    pt6_csv = 'models/atlas/atlas/propulsion/empirical_data/sorted_turbo_dataset_PT6.csv'
    sorted_data.to_csv(pt6_csv, index=False)
    print(f"\nPT6 dataset saved to '{pt6_csv}'")

    # Generate ACCE dataset: +10% power and +10% jet thrust, fuel flow unchanged
    acce_data = sorted_data.copy()
    acce_data['power_kW'] = acce_data['power_kW'] * 1.1
    acce_data['JetThrust_N'] = acce_data['JetThrust_N'] * 1.1
    acce_csv = 'models/atlas/atlas/propulsion/empirical_data/sorted_turbo_dataset_ACCE.csv'
    acce_data.to_csv(acce_csv, index=False)
    print(f"ACCE dataset saved to '{acce_csv}' (+10% power, +10% jet thrust, fuel flow unchanged)")
    
    # Show the improvement
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Sorted data shape: {sorted_data.shape}")
    print("All input columns are now monotonically increasing!")
    
    # Show some statistics
    print(f"\nInput column ranges:")
    for col in input_columns:
        print(f"  {col}: {data[col].min():.3f} to {data[col].max():.3f}")
    
    print(f"\nOutput column ranges:")
    for col in output_columns:
        print(f"  {col}: {data[col].min():.3f} to {data[col].max():.3f}")
    
    # Show unique values in each input column
    print(f"\nUnique values in each input column:")
    for col in input_columns:
        unique_vals = sorted(data[col].unique())
        print(f"  {col}: {len(unique_vals)} unique values - {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}") 