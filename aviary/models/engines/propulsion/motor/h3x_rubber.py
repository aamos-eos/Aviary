import pandas as pd
import numpy as np

class MotorDataThrottleEffCurve:
    """Global data store for motor training data - loads once, accessible everywhere"""
    voltage_data = None
    power_data__W = None
    eff_data = None
    throttle_data = None
    
    @classmethod
    def load_data(cls, motor_filename):
        """Load data once and store it in the class variables"""
        if cls.voltage_data is None:  # Only load if not already loaded
            # Read the Excel file (assumes headers are in the first row)
            df = pd.read_excel(motor_filename)
            
            # Drop any rows with NaN values
            #print(f"Original dataframe shape: {df.shape}")
            df = df.dropna()
            #print(f"Dataframe shape after dropping NaNs: {df.shape}")
            
            # Assign data from columns
            cls.voltage_data = df['Voltage'].values
            cls.throttle_data = df['throttle'].values
            cls.eff_data = df['eff'].values 
            
            # Define your input columns
            input_cols = ['Voltage','throttle','eff']  # replace with your actual input columns
            
            # Find all rows where the input combination is duplicated (singular points)
            df.drop_duplicates(subset=input_cols, keep='first')
                
    @classmethod
    def get_data(cls, motor_filename):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.voltage_data is None:
            cls.load_data(motor_filename)
        return cls 