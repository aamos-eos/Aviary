import pandas as pd
import numpy as np

class TurboData:
    """Global data store for turbo training data - loads once, accessible everywhere"""
    # Dynamic data
    disa_data__degC = None
    alt_data__ft = None
    alt_data__m = None
    mach_data = None
    frac_data = None
    power_data__kW = None
    power_data__W = None
    fuel_flow_data__kgph = None
    jet_thrust_data__N = None
    
    @classmethod
    def load_data(cls, turbo_filename, sheet_name='CRZ'):
        """Load data once and store it in the class variables"""
        if cls.disa_data__degC is None:  # Only load if not already loaded
            # Read the Excel file
            df = pd.read_excel(turbo_filename, sheet_name=sheet_name)
            
            # Rename columns
            df.columns = [
                'DISA', 'Altitude_ft', 'Mach', 'FRAC', 'power_SHP', 'FuelFlow_pph', 'JetThrust_lbf',
                'Altitude', 'power_kW', 'FuelFlow_kgph', 'JetThrust_N'
            ]

            # Define your input columns
            input_cols = ['DISA','Altitude','Mach','FRAC','power_kW','FuelFlow_kgph','JetThrust_N']  # replace with your actual input columns
            
            # Find all rows where the input combination is duplicated (singular points)
            df.drop_duplicates(subset=input_cols, keep='first')

            # Drop any rows with NaN values
            df = df.dropna()

        
            # Assign dynamic data from columns
            cls.disa_data__degC = df['DISA']
            #cls.alt_data__ft = df['Altitude_ft']
            cls.mach_data = df['Mach']
            cls.frac_data = df['FRAC']
            cls.alt_data__m = df['Altitude']
            cls.power_data__kW = df['power_kW']
            cls.fuel_flow_data__kgph = df['FuelFlow_kgph']
            cls.jet_thrust_data__N = df['JetThrust_N']


    
    @classmethod
    def get_data(cls, turbo_filename, sheet_name='CRZ'):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.disa_data__degC is None:
            cls.load_data(turbo_filename, sheet_name)
        return cls
