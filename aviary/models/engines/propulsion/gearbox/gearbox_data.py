import pandas as pd
import numpy as np

class GearboxData:
    """Global data store for gearbox training data - loads once, accessible everywhere"""
    
    # Configuration data loading flag
    _data_loaded = False
    
    # Gearbox efficiency data
    throttle_data = None
    efficiency_data = None
    
    @classmethod
    def load_data(cls, gearbox_filename, sheet_name='data'):
        """Load data once and store it in the class variables"""
        if cls.throttle_data is None:  # Only load if not already loaded
            try:
                # Read the Excel file
                df = pd.read_excel(gearbox_filename, sheet_name=sheet_name)
                
                # Drop any rows with NaN values
                df = df.dropna()
                
                # Extract throttle and efficiency data
                throttle_data = df['throttle'].values
                efficiency_data = df['efficiency'].values
                
                # Ensure data is sorted by throttle
                sort_indices = np.argsort(throttle_data)
                cls.throttle_data = throttle_data[sort_indices]
                cls.efficiency_data = efficiency_data[sort_indices]
                
            except Exception as e:
                print(f"Warning: Could not load gearbox data from {gearbox_filename}. Using default data.")
                print(f"Error: {e}")
                
                # Default gearbox efficiency data (typical values)
                cls.throttle_data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                cls.efficiency_data = np.array([0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.99])
    
    @classmethod
    def get_data(cls, gearbox_filename, sheet_name='data'):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.throttle_data is None:
            cls.load_data(gearbox_filename, sheet_name)
        return cls
