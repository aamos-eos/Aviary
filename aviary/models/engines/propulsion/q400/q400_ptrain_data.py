"""
Q400 Powertrain Data Loader

This module provides a data loader class for Q400 powertrain performance data,
following the same pattern as PropellerData in the propulsion module.

The data includes cruise, climb, and descent performance tables from the Q400 AOM.

Inputs to metamodels:
    - throttle_torque: Torque percentage (0-100%)
    - altitude_m: Altitude in meters
    - true_airspeed_mps: True airspeed in m/s

Outputs from metamodels:
    - fuel_flow_kgph: Fuel flow in kg/h (both engines total)
    - unit_thrust_N: Thrust per engine in Newtons
"""

import pandas as pd
import numpy as np


class Q400PtrainData:
    """Global data store for Q400 powertrain training data - loads once, accessible everywhere"""
    
    # Data loading flag
    _data_loaded = False
    
    # Raw data arrays (in original units)
    _table_type = None
    _data_type = None
    _OAT_C = None
    _altitude_ft = None
    _KIAS = None
    _KTAS = None
    _throttle_torque = None
    _unit_fuel_flow_lbph = None
    _total_fuel_flow_lbph = None
    _RPM = None
    _unit_shaft_power_kW = None
    _unit_thrust_N = None
    
    # SI unit data arrays (these are passed directly to MetaModelUnStructuredComp)
    altitude_m = None
    true_airspeed_mps = None
    throttle_torque = None  # throttle_torque (already dimensionless)
    fuel_flow_kgph = None
    unit_thrust_N = None
    
    # Data bounds
    min_throttle = None
    max_throttle = None
    min_altitude_m = None
    max_altitude_m = None
    min_tas_mps = None
    max_tas_mps = None
    
    @classmethod
    def load_data(cls, filename, sheet_name='data'):
        """
        Load Q400 powertrain data from Excel file and convert to SI units.
        
        Parameters
        ----------
        filename : str
            Path to the Excel file containing powertrain data
        sheet_name : str
            Name of the sheet to load (default: 'data')
        """
        if cls._data_loaded:
            return
        
        print(f"Loading Q400 powertrain data from: {filename}")
        
        # Read the Excel file
        df = pd.read_excel(filename, sheet_name=sheet_name)
        
        # Drop any rows with NaN values in critical columns
        critical_cols = ['throttle_torque', 'altitude_ft', 'KTAS', 'fuel_flow_lbph', 'unit_thrust_N']
        available_cols = [col for col in critical_cols if col in df.columns]
        if available_cols:
            df = df.dropna(subset=available_cols)
        
        print(f"  Loaded {len(df)} data points")
        
        # Store raw data
        if 'table' in df.columns:
            cls._table_type = df['table'].values
        if 'data_type' in df.columns:
            cls._data_type = df['data_type'].values
        if 'OAT_C' in df.columns:
            cls._OAT_C = df['OAT_C'].values
        if 'altitude_ft' in df.columns:
            cls._altitude_ft = df['altitude_ft'].values
        if 'KIAS' in df.columns:
            cls._KIAS = df['KIAS'].values
        if 'KTAS' in df.columns:
            cls._KTAS = df['KTAS'].values
        if 'throttle_torque' in df.columns:
            cls._throttle_torque = df['throttle_torque'].values/100 # convert to percentage
        if 'unit_fuel_flow_lbph' in df.columns:
            cls._unit_fuel_flow_lbph = df['unit_fuel_flow_lbph'].values
        if 'total_fuel_flow_lbph' in df.columns:
            cls._total_fuel_flow_lbph = df['total_fuel_flow_lbph'].values
        if 'RPM' in df.columns:
            cls._RPM = df['RPM'].values
        if 'unit_shaft_power_kW' in df.columns:
            cls._unit_shaft_power_kW = df['unit_shaft_power_kW'].values
        if 'unit_thrust_N' in df.columns:
            cls._unit_thrust_N = df['unit_thrust_N'].values
        
        # Convert to SI units

        if cls._RPM is not None:
            cls.RPM = cls._RPM
        # Altitude: ft -> m
        if cls._altitude_ft is not None:
            cls.altitude_m = cls._altitude_ft * 0.3048
        
        # True airspeed: KTAS -> m/s
        if cls._KTAS is not None:
            cls.true_airspeed_mps = cls._KTAS * 0.514444
        
        # Throttle/torque: already in percent (0-100)
        if cls._throttle_torque is not None:
            cls.throttle_torque = cls._throttle_torque
        
        # Fuel flow: lb/h -> kg/h (1 lb = 0.453592 kg)
        if cls._unit_fuel_flow_lbph is not None:
            cls.unit_fuel_flow_kgph = cls._unit_fuel_flow_lbph * 0.453592

        if cls._total_fuel_flow_lbph is not None:
            cls.total_fuel_flow_kgph = cls._total_fuel_flow_lbph * 0.453592
        
        # Thrust: already in Newtons
        if cls._unit_thrust_N is not None:
            cls.unit_thrust_N = cls._unit_thrust_N
        
        # Calculate data bounds
        if cls.throttle_torque is not None:
            cls.min_throttle = np.min(cls.throttle_torque)
            cls.max_throttle = np.max(cls.throttle_torque)
        if cls.altitude_m is not None:
            cls.min_altitude_m = np.min(cls.altitude_m)
            cls.max_altitude_m = np.max(cls.altitude_m)
        if cls.true_airspeed_mps is not None:
            cls.min_tas_mps = np.min(cls.true_airspeed_mps)
            cls.max_tas_mps = np.max(cls.true_airspeed_mps)
        
        print(f"  Throttle range: {cls.min_throttle*100:.2f}% - {cls.max_throttle*100:.2f}%")
        print(f"  Altitude range: {cls.min_altitude_m:.0f}m - {cls.max_altitude_m:.0f}m")
        print(f"  TAS range: {cls.min_tas_mps:.1f}m/s - {cls.max_tas_mps:.1f}m/s")
        
        cls._data_loaded = True
    
    @classmethod
    def get_data(cls, filename, sheet_name='data'):
        """
        Ensure data is loaded and return the class reference.
        
        Parameters
        ----------
        filename : str
            Path to the Excel file
        sheet_name : str
            Name of the sheet to load
            
        Returns
        -------
        class
            Reference to Q400PtrainData class with loaded data
        """
        if not cls._data_loaded:
            cls.load_data(filename, sheet_name)
        return cls
    
    @classmethod
    def reset(cls):
        """Reset all class data (useful for testing or reloading different data)."""
        cls._data_loaded = False
        cls._table_type = None
        cls._data_type = None
        cls._OAT_C = None
        cls._altitude_ft = None
        cls._KIAS = None
        cls._KTAS = None
        cls._throttle_torque = None
        cls._fuel_flow_lbph = None
        cls._RPM = None
        cls._unit_shaft_power_kW = None
        cls._unit_thrust_N = None
        cls.altitude_m = None
        cls.true_airspeed_mps = None
        cls.throttle_torque = None
        cls.unit_fuel_flow_kgph = None
        cls.total_fuel_flow_kgph = None
        cls.unit_thrust_N = None
        cls.min_throttle = None
        cls.max_throttle = None
        cls.min_altitude_m = None
        cls.max_altitude_m = None
        cls.min_tas_mps = None
        cls.max_tas_mps = None


if __name__ == "__main__":
    # Test the data loader
    import os
    
    # Path to the data file
    data_file = os.path.join(
        os.path.dirname(__file__),
        '..', 'predefined_missions', 'q400_mission_data.xlsx'
    )
    
    # Load data
    Q400PtrainData.load_data(data_file)
    
    print("\nData loaded successfully!")
    print(f"Number of data points: {len(Q400PtrainData.throttle_torque)}")
    print(f"Unit fuel flow range: {np.min(Q400PtrainData.unit_fuel_flow_kgph):.1f} - {np.max(Q400PtrainData.unit_fuel_flow_kgph):.1f} kg/h")
    print(f"Total fuel flow range: {np.min(Q400PtrainData.total_fuel_flow_kgph):.1f} - {np.max(Q400PtrainData.total_fuel_flow_kgph):.1f} kg/h")
    print(f"Thrust range: {np.min(Q400PtrainData.unit_thrust_N):.1f} - {np.max(Q400PtrainData.unit_thrust_N):.1f} N")
    
    # Show sample data points
    print(f"\nSample data (first 5 points):")
    print(f"  Throttle: {Q400PtrainData.throttle_torque[:5]}")
    print(f"  Altitude (m): {Q400PtrainData.altitude_m[:5]}")
    print(f"  TAS (m/s): {Q400PtrainData.true_airspeed_mps[:5]}")
    print(f"  Unit fuel flow (kg/h): {Q400PtrainData.unit_fuel_flow_kgph[:5]}")
    print(f"  Total fuel flow (kg/h): {Q400PtrainData.total_fuel_flow_kgph[:5]}")
    print(f"  Thrust (N): {Q400PtrainData.unit_thrust_N[:5]}")

