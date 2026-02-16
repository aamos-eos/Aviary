import pandas as pd
import numpy as np
import os
import sys

class BatteryData:
    """Global data store for battery training data - loads once, accessible everywhere"""
    soc_data = None
    c_rate_cur_data = None
    ir0_data = None
    ocv_data = None
    line_voltage_data = None
    m_cell = None
    Cp_cell = None
    cell_Ah_capacity = None
    limit_min_temp = None
    limit_max_temp = None
    
    # Meshgrid data for 2D interpolation
    soc_mesh_flat = None
    c_rate_cur_mesh_flat = None
    line_voltage_flat = None
    ir0_flat = None
    ocv_data_flipped = None
    soc_data_flipped = None
    
    @classmethod
    def load_data(cls,bat_filename, cell_sheetname, config_sheetname):
        """Load data once and store it in the class variables"""
        if cls.soc_data is None:  # Only load if not already loaded

            df_cell = pd.read_excel(bat_filename, sheet_name=cell_sheetname)
            df_batconfig = pd.read_excel(bat_filename, sheet_name=config_sheetname)
            
            soc_data = df_cell.SOC.to_numpy()
            # Round any 0 SOC values up to 1e-6
            soc_data[soc_data == 0] = 1e-6
            ocv_data = df_cell.OCV.to_numpy()
            ir0_data = df_cell[df_cell.Crate_col.dropna()].to_numpy()/1000

            # If starting SOC is greater than ending SOC, 
            # OCV, ir0, also decreases with decreasing SOC
            # Flip data for monotonic increase for interpolation
            if len(soc_data) > 1 and soc_data[0] > soc_data[-1]:
                soc_data_incr = np.flip(soc_data)
                ir0_data_incr = np.flip(ir0_data)
                ocv_data_incr = np.flip(ocv_data)

                soc_data_decr = soc_data
                ir0_data_decr = ir0_data
                ocv_data_decr = ocv_data
            else:
                soc_data_incr = soc_data
                ir0_data_incr = ir0_data
                ocv_data_incr = ocv_data

                soc_data_decr = np.flip(soc_data)
                ir0_data_decr = np.flip(ir0_data)
                ocv_data_decr = np.flip(ocv_data)

            # Store flipped SOC and OCV data to be used for interpolation
            cls.soc_data_incr = soc_data_incr
            
            # C-rate data is always increasing in table
            c_rate_cur_data = df_cell.Crate_value.dropna().to_numpy()
            c_rate_cur_data[c_rate_cur_data == 0] = 1e-6

            cls.c_rate_cur_data_incr = c_rate_cur_data
            cls.ir0_data_incr = ir0_data_incr

            
            # Check for duplicate columns in ir0_data and remove them
            # ir0_data is a matrix (SOC x C-rate), c_rate_data is a vector
            #if cls.ir0_data.shape[1] > 1:  # Only check if there are multiple columns
            #    # Find unique columns and their indices
            #    _, unique_indices = np.unique(cls.ir0_data, axis=1, return_index=True)
            #    
            #    # Remove duplicate columns from ir0_data and corresponding elements from c_rate_data
            #    if len(unique_indices) < cls.ir0_data.shape[1]:
            #        cls.ir0_data = cls.ir0_data[:, unique_indices]
            #        cls.c_rate_data = cls.c_rate_data[unique_indices]
            #        print(f"Removed {cls.ir0_data.shape[1] - len(unique_indices)} duplicate C-rate columns")
            

            cls.ocv_data_incr = ocv_data_incr

            # Force initial calculation with decreasing OCV with respect to SOC, and 
            # ir0 increasing with c-rate
            line_voltage_data = ocv_data_decr.reshape((-1,1)) - np.multiply(ir0_data_decr, cls.c_rate_cur_data_incr)
            cls.line_voltage_data_incr = line_voltage_data[::-1]

            cls.power_data_incr = cls.line_voltage_data_incr * cls.c_rate_cur_data_incr

            cls.m_cell = df_cell.mass_cell.iloc[0]
            cls.cp_cell = df_cell.Cp_cell.iloc[0]
            cls.cell_Ah_capacity = df_cell.capacity_Ah.iloc[0]
            cls.limit_min_temp = df_cell.min_temp.iloc[0]
            cls.limit_max_temp = df_cell.max_temp.iloc[0]

            cls.n_str = df_batconfig.Group.iloc[-1]
            cls.n_parallel_per_str = df_batconfig.nparallels.iloc[0]
            cls.n_series_per_str = df_batconfig.nseries.iloc[0]
            #cls.n_motors = df_batconfig.motor_id.iloc[0][-1]

            cls.soc_init = soc_data[0]

            # Store decreasing data
            cls.soc_data_decr = soc_data_decr
            cls.ir0_data_decr = ir0_data_decr
            cls.ocv_data_decr = ocv_data_decr


            # Generate meshgrid data for 2D interpolation
            #cls._generate_meshgrid_data()
            
            #print(f"SOC range: {cls.soc_data.min():.2e} to {cls.soc_data.max():.2f}")
    

    @classmethod
    def get_data(cls,bat_filename, cell_sheetname, config_sheetname):
        """Ensure data is loaded and return all data as a tuple"""
        if cls.soc_data is None:
            cls.load_data(bat_filename, cell_sheetname, config_sheetname)
        return cls


    
if __name__ == '__main__':
    BatteryData.load_data(bat_filename='aviary/models/engines/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx', 
                            cell_sheetname='BOL_cell_fct_CRate', 
                            config_sheetname='battery_config')

    print("Increasing data:") 
    print("OCV:")                       
    print(BatteryData.ocv_data_incr)
    print("IR0:")
    print(BatteryData.ir0_data_incr)
    print("Line Voltage:")
    print(BatteryData.line_voltage_data_incr)
    print("Power [kW]:")
    print(BatteryData.power_data_incr/1000)
    print("SOC:")
    print(BatteryData.soc_data_incr)
    print("C-rate:")
    print(BatteryData.c_rate_cur_data_incr)
    print("--------------------------------")
    print("Decreasing data:")
    print("SOC:")
    print(BatteryData.soc_data_decr)
    print("OCV:")
    print(BatteryData.ocv_data_decr)
    print("IR0:")
    print(BatteryData.ir0_data_decr)
    print("--------------------------------")
