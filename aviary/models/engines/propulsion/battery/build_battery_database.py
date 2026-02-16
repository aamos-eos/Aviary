import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import the existing battery data
from .battery_data import BatteryData

class BatterySolver:
    """
    Smart battery solver using scipy interpolation functions with existing BatteryData.
    """
    
    def __init__(self, bat_filename, cell_sheetname, config_sheetname):
        """
        Initialize the battery solver with existing battery data.
        
        Parameters
        ----------
        bat_filename : str
            Path to the battery data Excel file
        cell_sheetname : str
            Name of the cell data sheet
        config_sheetname : str
            Name of the configuration sheet
        """
        # Load the battery data
        BatteryData.load_data(bat_filename, cell_sheetname, config_sheetname)
        
        # Get the data
        self.soc_data = BatteryData.soc_data_incr
        self.c_rate_cur_data = BatteryData.c_rate_cur_data_incr
        self.ocv_data = BatteryData.ocv_data_incr
        self.ir0_data = BatteryData.ir0_data_incr
        self.cell_Ah_capacity = BatteryData.cell_Ah_capacity
        
        # Create interpolators
        self._create_interpolators()
        
        print(f"Battery solver initialized with:")
        print(f"  SOC range: {self.soc_data.min():.3f} to {self.soc_data.max():.3f}")
        print(f"  Current Range: {self.c_rate_cur_data.min():.2f} to {self.c_rate_cur_data.max():.2f} A")
        print(f"  OCV range: {self.ocv_data.min():.2f} to {self.ocv_data.max():.2f} V")
        print(f"  IR0 range: {self.ir0_data.min()*1000:.2f} to {self.ir0_data.max()*1000:.2f} mΩ (stored in ohms)")
        print(f"  Cell capacity: {self.cell_Ah_capacity:.1f} Ah")
        
    def _create_interpolators(self):
        """Create scipy interpolators for OCV and IR0."""
        
        # OCV interpolation (1D: SOC -> OCV)
        self.ocv_interp = interp1d(self.soc_data, self.ocv_data, 
                                   kind='cubic', 
                                   bounds_error=False, 
                                   fill_value=(self.ocv_data[0], self.ocv_data[-1]))
        
        # IR0 interpolation (2D: (SOC, C-rate) -> IR0)
        self.ir0_interp = RegularGridInterpolator(
            (self.soc_data, self.c_rate_cur_data), 
            self.ir0_data,
            method='cubic',
            bounds_error=False,
            fill_value=None  # Will use nearest value for out-of-bounds
        )
        
    def solve_quadratic_current(self, ocv, ir0, power):
        """
        Solve quadratic equation for current.
        I₂ = (OCV - √(OCV² - 4·R·P)) / (2·R)
        """
        discriminant = ocv**2 - 4 * ir0 * power
        
        if discriminant < 0:
            #print(discriminant)
            raise ValueError(f"No solution exists for discriminant ")
        
        # Use negative root (current flows from battery to load)
        current = (ocv - np.sqrt(discriminant)) / (2 * ir0)
        
        return current
    
    def solve_battery_state(self, soc, power, max_iterations=50, tolerance=1e-6):
        """
        Iteratively solve for battery current and voltage.
        
        Parameters
        ----------
        soc : float
            State of charge (0 to 1)
        power : float
            Power demand (W)
        max_iterations : int
            Maximum iterations for convergence
        tolerance : float
            Convergence tolerance for C-rate
            
        Returns
        -------
        dict
            Battery state solution
        """
        # Validate inputs
        if not (0 <= soc <= 1):
            raise ValueError(f"SOC must be between 0 and 1, got {soc}")
        #if power < 0:
        #    raise ValueError(f"Power must be non-negative, got {power}")
        
        # Handle zero power case
        if power == 0:
            ocv = self.ocv_interp(soc)
            return {
                'current': 0.0,
                'voltage': ocv,
                'ocv': ocv,
                'ir0': self.ir0_interp(np.column_stack([soc, 0.0])).item(),
                'c_rate': 0.0,
                'iterations': 1,
                'converged': True,
                'error': None
            }
        
        # Get OCV
        ocv = self.ocv_interp(soc)
        
        # Initial guess for C-rate
        c_rate_old = 1.0
        cell_capacity = self.cell_Ah_capacity
        
        # Iterative solution
        for iteration in range(max_iterations):
            # Get IR0 for current C-rate

            ir0 = self.ir0_interp(np.column_stack([soc, cell_capacity * c_rate_old])).item()
            
            # Solve for current
            try:
                current = self.solve_quadratic_current(ocv, ir0, power)
            except ValueError as e:
                return {
                    'current': None,
                    'voltage': None,
                    'ocv': ocv,
                    'ir0': ir0,
                    'c_rate': c_rate_old,
                    'iterations': iteration + 1,
                    'converged': False,
                    'error': str(e)
                }
            
            # Calculate new C-rate
            c_rate_new = abs(current) / self.cell_Ah_capacity

            if current < 0:
                current = np.maximum(current, -700 / 27)
            
            # Check convergence
            if abs(c_rate_new - c_rate_old) < tolerance:
                voltage = ocv - current * ir0

                # Limit Charging Voltage
                if current < 0:
                    voltage = np.minimum(voltage, 860/208)
                
                # Debug: Calculate and verify power
                calculated_power = current * voltage
                power_error = abs(calculated_power - power)
                power_error_pct = abs(power_error / power * 100)
                
                # Debug output for high errors
                if power_error_pct > 1.0:  # More than 1% error
                    print(f"DEBUG: High power error detected!")
                    print(f"  SOC: {soc:.3f}, Requested power: {power:.2f} W")
                    print(f"  OCV: {ocv:.3f} V, IR0: {ir0:.6f} Ω ({ir0*1000:.2f} mΩ)")
                    print(f"  Current: {current:.3f} A, C-rate: {c_rate_new:.3f}C")
                    print(f"  Voltage: {voltage:.3f} V")
                    print(f"  Calculated power: {calculated_power:.2f} W")
                    print(f"  Power error: {power_error:.2f} W ({power_error_pct:.2f}%)")
                    print(f"  Iterations: {iteration + 1}")
                    converged = False
                else:
                    converged = True
                
                return {
                    'current': current,
                    'voltage': voltage,
                    'ocv': ocv,
                    'ir0': ir0,
                    'c_rate': c_rate_new,
                    'iterations': iteration + 1,
                    'converged': converged,
                    'error': None
                }
            
            c_rate_old = c_rate_new
        
        # If we get here, didn't converge
        return {
            'current': None,
            'voltage': None,
            'ocv': ocv,
            'ir0': ir0,
            'c_rate': c_rate_old,
            'iterations': max_iterations,
            'converged': False,
            'error': f"Failed to converge after {max_iterations} iterations"
        }
    
    def build_database(self, soc_values, power_values):
        """
        Build battery database for given SOC and power values.
        
        Parameters
        ----------
        soc_values : array
            Array of SOC values
        power_values : array
            Array of power values
            
        Returns
        -------
        pandas.DataFrame
            Database of solutions
        """
        results = []
        
        total_points = len(soc_values) * len(power_values)
        current_point = 0
        
        print(f"Building database with {total_points} points...")
        
        for soc in soc_values:
            for power in power_values:
                current_point += 1
                if current_point % 100 == 0:
                    print(f"Progress: {current_point}/{total_points} ({current_point/total_points*100:.1f}%)")
                
                solution = self.solve_battery_state(soc, power)
                
                results.append({
                    'soc': soc,
                    'power': power,
                    'current': solution['current'],
                    'ocv': solution['ocv'],
                    'ir0': solution['ir0'],
                    'c_rate': solution['c_rate'],
                    'line_voltage': solution['voltage'] if solution['current'] is not None else None,
                    'power_out': solution['voltage'] * solution['current'] if solution['current'] is not None else None,
                    'deltaP': power - solution['voltage'] * solution['current'] if solution['current'] is not None else None,
                    'converged': solution['converged'],
                    'iterations': solution['iterations'],
                    'error': solution['error']
                })
        
        df = pd.DataFrame(results)
        
        # Print statistics
        converged_count = df['converged'].sum()
        total_count = len(df)
        print(f"\nDatabase complete!")
        print(f"Converged: {converged_count}/{total_count} ({converged_count/total_count*100:.1f}%)")
        
        if converged_count > 0:
            avg_iterations = df[df['converged']]['iterations'].mean()
            print(f"Average iterations: {avg_iterations:.1f}")
        
        return df
    
    def plot_database(self, df, save_path=None):
        """
        Create visualizations of the database.
        """
        # Filter converged solutions
        df_converged = df[df['converged']].copy()
        
        if len(df_converged) == 0:
            print("No converged solutions to plot!")
            return
        
        # Ensure numeric types for plotting columns
        for col in ['current', 'ocv', 'c_rate']:
            df_converged[col] = pd.to_numeric(df_converged[col], errors='coerce')
        
        # Create pivot tables for plotting
        pivot_current = df_converged.pivot(index='soc', columns='power', values='current')
        pivot_voltage = df_converged.pivot(index='soc', columns='power', values='ocv')
        pivot_c_rate = df_converged.pivot(index='soc', columns='power', values='c_rate')
        
        """
        # Fill NaN values with 0 for plotting
        pivot_current = pivot_current.fillna(0)
        pivot_voltage = pivot_voltage.fillna(0)
        pivot_c_rate = pivot_c_rate.fillna(0)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Current map
        im1 = axes[0,0].contourf(X_current, Y_current, Z_current, levels=20)
        axes[0,0].set_xlabel('Power (W)')
        axes[0,0].set_ylabel('SOC')
        axes[0,0].set_title('Current (A)')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Voltage map
        im2 = axes[0,1].contourf(X_voltage, Y_voltage, Z_voltage, levels=20)
        axes[0,1].set_xlabel('Power (W)')
        axes[0,1].set_ylabel('SOC')
        axes[0,1].set_title('Voltage (V)')
        plt.colorbar(im2, ax=axes[0,1])
        
        # C-rate map
        im3 = axes[1,0].contourf(X_crate, Y_crate, Z_crate, levels=20)
        axes[1,0].set_xlabel('Power (W)')
        axes[1,0].set_ylabel('SOC')
        axes[1,0].set_title('C-rate (1/h)')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Convergence map
        pivot_converged = df.pivot(index='soc', columns='power', values='converged')
        X_conv = pivot_converged.columns.values.astype(float)
        Y_conv = pivot_converged.index.values.astype(float)
        Z_conv = pivot_converged.values.astype(float)
        im4 = axes[1,1].contourf(X_conv, Y_conv, Z_conv, levels=2)
        axes[1,1].set_xlabel('Power (W)')
        axes[1,1].set_ylabel('SOC')
        axes[1,1].set_title('Convergence')
        plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        """

    def test_ir0_interpolation(self, plot=True):
        """
        Test the IR0 interpolator against the actual training data.
        """
        print("\nTesting IR0 interpolation against training data...")
        
        # Create meshgrid covering all training data points
        soc_mesh, c_rate_cur_mesh = np.meshgrid(self.soc_data, self.c_rate_cur_data, indexing='ij')
        
        # Flatten for interpolation
        soc_flat = soc_mesh.flatten()
        c_rate_cur_flat = c_rate_cur_mesh.flatten()
        
        # Get interpolated values for all points
        points = np.column_stack([soc_flat, c_rate_cur_flat])
        interpolated_ir0 = self.ir0_interp(points)
        
        # Get actual values
        actual_ir0 = self.ir0_data.flatten()
        
        # Calculate errors
        errors = np.abs(interpolated_ir0 - actual_ir0)
        relative_errors = errors / actual_ir0 * 100
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        print(f"  Total points tested: {len(actual_ir0)}")
        print(f"  Max error: {max_error:.8f} Ω")
        print(f"  Mean error: {mean_error:.8f} Ω")
        print(f"  Max relative error: {np.max(relative_errors):.4f}%")
        
        if plot:
            # Reshape errors back to 2D for plotting
            errors_2d = errors.reshape(soc_mesh.shape)
            relative_errors_2d = relative_errors.reshape(soc_mesh.shape)
            interpolated_2d = interpolated_ir0.reshape(soc_mesh.shape)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('IR0 Interpolation Test', fontsize=14)
            
            # Plot actual IR0 data
            im1 = axes[0,0].contourf(self.c_rate_cur_data, self.soc_data, self.ir0_data * 1000, levels=20, cmap='viridis')
            axes[0,0].set_xlabel('Current (A)')
            axes[0,0].set_ylabel('SOC')
            axes[0,0].set_title('Actual IR0 (mΩ)')
            plt.colorbar(im1, ax=axes[0,0])
            
            # Plot interpolated IR0 data
            im2 = axes[0,1].contourf(self.c_rate_cur_data, self.soc_data, interpolated_2d * 1000, levels=20, cmap='viridis')
            axes[0,1].set_xlabel('Current (A)')
            axes[0,1].set_ylabel('SOC')
            axes[0,1].set_title('Interpolated IR0 (mΩ)')
            plt.colorbar(im2, ax=axes[0,1])
            
            # Plot absolute error
            im3 = axes[1,0].contourf(self.c_rate_cur_data, self.soc_data, errors_2d * 1000, levels=20, cmap='Reds')
            axes[1,0].set_xlabel('Current (A)')
            axes[1,0].set_ylabel('SOC')
            axes[1,0].set_title('Absolute Error (mΩ)')
            plt.colorbar(im3, ax=axes[1,0])
            
            # Plot relative error
            im4 = axes[1,1].contourf(self.c_rate_cur_data, self.soc_data, relative_errors_2d, levels=20, cmap='Reds')
            axes[1,1].set_xlabel('Current (A)')
            axes[1,1].set_ylabel('SOC')
            axes[1,1].set_title('Relative Error (%)')
            plt.colorbar(im4, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig('ir0_interpolation_test.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot error visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reshape errors for 2D plotting
        errors_2d = errors.reshape(soc_mesh.shape)
        relative_errors = (errors / actual_ir0) * 100
        relative_errors_2d = relative_errors.reshape(soc_mesh.shape)
        
        # Absolute error contour
        im1 = axes[0, 0].contourf(c_rate_cur_mesh, soc_mesh, errors_2d, levels=20, cmap='Reds')
        axes[0, 0].set_xlabel('C-rate (Current)')
        axes[0, 0].set_ylabel('SOC')
        axes[0, 0].set_title('IR0 Absolute Error (Ω)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Relative error contour
        im2 = axes[0, 1].contourf(c_rate_cur_mesh, soc_mesh, relative_errors_2d, levels=20, cmap='Reds')
        axes[0, 1].set_xlabel('C-rate (Current)')
        axes[0, 1].set_ylabel('SOC')
        axes[0, 1].set_title('IR0 Relative Error (%)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Error histogram
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error (Ω)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('IR0 Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Interpolated scatter
        axes[1, 1].scatter(actual_ir0, interpolated_ir0, alpha=0.5, s=10)
        min_val = min(actual_ir0.min(), interpolated_ir0.min())
        max_val = max(actual_ir0.max(), interpolated_ir0.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        axes[1, 1].set_xlabel('Actual IR0 (Ω)')
        axes[1, 1].set_ylabel('Interpolated IR0 (Ω)')
        axes[1, 1].set_title('IR0: Actual vs Interpolated')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ir0_interpolation_error_plots.png', dpi=150, bbox_inches='tight')
        print("  Saved error plots to: ir0_interpolation_error_plots.png")
        plt.show()
        
        print("IR0 interpolation test complete!")
        
    def test_ocv_interpolation(self, plot=True):
        """
        Test the OCV interpolator against the actual training data.
        """
        print("\nTesting OCV interpolation against training data...")
        
        # Get interpolated values for all SOC points
        interpolated_ocv = self.ocv_interp(self.soc_data)
        
        # Get actual values
        actual_ocv = self.ocv_data
        
        # Calculate errors
        errors = np.abs(interpolated_ocv - actual_ocv)
        relative_errors = errors / actual_ocv * 100
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        print(f"  Total points tested: {len(actual_ocv)}")
        print(f"  Max error: {max_error:.6f} V")
        print(f"  Mean error: {mean_error:.6f} V")
        print(f"  Max relative error: {np.max(relative_errors):.4f}%")
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('OCV Interpolation Test', fontsize=14)
            
            # Plot OCV vs SOC comparison
            axes[0,0].plot(self.soc_data, actual_ocv, 'b-', linewidth=2, label='Actual')
            axes[0,0].plot(self.soc_data, interpolated_ocv, 'r--', linewidth=2, label='Interpolated')
            axes[0,0].set_xlabel('SOC')
            axes[0,0].set_ylabel('OCV (V)')
            axes[0,0].set_title('OCV vs SOC')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot absolute error vs SOC
            axes[0,1].plot(self.soc_data, errors * 1000, 'r-', linewidth=2)
            axes[0,1].set_xlabel('SOC')
            axes[0,1].set_ylabel('Absolute Error (mV)')
            axes[0,1].set_title('Absolute Error vs SOC')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Plot relative error vs SOC
            axes[1,0].plot(self.soc_data, relative_errors, 'r-', linewidth=2)
            axes[1,0].set_xlabel('SOC')
            axes[1,0].set_ylabel('Relative Error (%)')
            axes[1,0].set_title('Relative Error vs SOC')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Plot residuals (actual - interpolated)
            residuals = actual_ocv - interpolated_ocv
            axes[1,1].scatter(self.soc_data, residuals * 1000, c='blue', s=30, alpha=0.7)
            axes[1,1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1,1].set_xlabel('SOC')
            axes[1,1].set_ylabel('Residual (mV)')
            axes[1,1].set_title('Residuals (Actual - Interpolated)')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ocv_interpolation_test.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot error visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        relative_errors = (errors / actual_ocv) * 100
        
        # Error vs SOC
        axes[0, 0].plot(self.soc_data, errors, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('SOC')
        axes[0, 0].set_ylabel('Absolute Error (V)')
        axes[0, 0].set_title('OCV Absolute Error vs SOC')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Relative error vs SOC
        axes[0, 1].plot(self.soc_data, relative_errors, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_xlabel('SOC')
        axes[0, 1].set_ylabel('Relative Error (%)')
        axes[0, 1].set_title('OCV Relative Error vs SOC')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error histogram
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error (V)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('OCV Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Interpolated
        axes[1, 1].scatter(actual_ocv, interpolated_ocv, alpha=0.7, s=50)
        min_val = min(actual_ocv.min(), interpolated_ocv.min())
        max_val = max(actual_ocv.max(), interpolated_ocv.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        axes[1, 1].set_xlabel('Actual OCV (V)')
        axes[1, 1].set_ylabel('Interpolated OCV (V)')
        axes[1, 1].set_title('OCV: Actual vs Interpolated')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ocv_interpolation_error_plots.png', dpi=150, bbox_inches='tight')
        print("  Saved error plots to: ocv_interpolation_error_plots.png")
        plt.show()
        
        print("OCV interpolation test complete!")

def build_database(save_database=True, plot_database=True, battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105'):
    """Main function to build the battery database."""
    
    # Battery data file path
    battery_original_filename = 'models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name 
    bat_filename = battery_original_filename + '.xlsx'
    cell_sheetname = 'BOL_cell_fct_CRate'
    config_sheetname = 'battery_config'
    
    # Create battery solver
    print("Initializing battery solver...")
    solver = BatterySolver(bat_filename, cell_sheetname, config_sheetname)
    
    # Run interpolation tests
    solver.test_ocv_interpolation()
    solver.test_ir0_interpolation()
    
    # Define SOC and power ranges
    soc_values = np.linspace(1e-6, 1, 40)  # 20 SOC points from 0% to 100%
    power_values = np.arange(-100, 1550, 25)  # Power from 0W to 1500W in 50W increments
    
    print(f"\nBuilding database with:")
    print(f"  SOC points: {len(soc_values)} (range: {soc_values.min():.2f} to {soc_values.max():.2f})")
    print(f"  Power points: {len(power_values)} (range: {power_values.min():.0f}W to {power_values.max():.0f}W)")
    print(f"  Total points: {len(soc_values) * len(power_values)}")
    
    # Build database
    df = solver.build_database(soc_values, power_values)

    if save_database:
        # Save database to CSV
        output_path = battery_original_filename + '_battery_power_database.csv'
        df.to_csv(output_path, index=False)
        print(f"\nDatabase saved to: {output_path}")
        
        # Create smaller dataframe with only converged solutions
        df_converged = df[df['converged']].copy()
        df_clean = df_converged[['soc', 'power', 'ocv', 'line_voltage', 'c_rate', 'current', 'ir0']].copy()
        
        # Save clean database to CSV
        clean_output_path = battery_original_filename + '_battery_power_database_clean.csv'
        df_clean.to_csv(clean_output_path, index=False)
        print(f"Clean database saved to: {clean_output_path}")
        print(f"Clean database contains {len(df_clean)} converged solutions")
        
        # Create full database with non-converged solutions set to 0
        df_full = df.copy()
        # Set current and line_voltage to 0 for non-converged solutions
        df_full.loc[~df_full['converged'], 'current'] = 1e-6
        df_full.loc[~df_full['converged'], 'line_voltage'] = 1.0
        
        # Keep only the same columns as df_clean
        df_full = df_full[['soc', 'power', 'ocv', 'line_voltage', 'c_rate', 'current', 'ir0']].copy()
        
        # Save full database to CSV
        full_output_path = battery_original_filename + '_battery_power_database_full.csv'
        df_full.to_csv(full_output_path, index=False)
        print(f"Full database saved to: {full_output_path}")
        print(f"Full database contains {len(df_full)} total solutions")
        
    # Print statistics about non-converged solutions
    non_converged_count = (~df['converged']).sum()
    print(f"Non-converged solutions set to 0: {non_converged_count}")
    

    if plot_database:
        # Create visualizations
        print("\nCreating visualizations...")
        solver.plot_database(df, save_path='battery_database_plots.png')
    
    
    # Create visualizations
    print("\nCreating visualizations...")
    solver.plot_database(df, save_path='battery_database_plots.png')
    
    # Print some example results
    print("\nExample results:")
    example_soc = 0.8
    example_power = 1000
    result = solver.solve_battery_state(example_soc, example_power)
    print(f"SOC: {example_soc}, Power: {example_power}W")
    
    if result['converged'] and result['current'] is not None:
        print(f"  Current: {result['current']:.2f} A")
        print(f"  OCV: {result['ocv']:.2f} V")
        print(f"  IR0: {result['ir0']:.6f} Ω ({result['ir0']*1000:.2f} mΩ)")
        print(f"  Line voltage: {result['ocv'] - result['current'] * result['ir0']:.2f} V")
        print(f"  C-rate: {result['c_rate']:.2f}C")
        print(f"  Converged: {result['converged']}")
        
        # Verify the calculation
        calculated_power = result['current'] * (result['ocv'] - result['current'] * result['ir0'])
        power_error = abs(calculated_power - example_power)
        print(f"  Power verification: {calculated_power:.2f} W (error: {power_error:.2f} W)")
    else:
        print(f"  Converged: {result['converged']}")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        if result['ocv'] is not None:
            print(f"  OCV: {result['ocv']:.2f} V")
        if result['ir0'] is not None:
            print(f"  IR0: {result['ir0']:.6f} Ω ({result['ir0']*1000:.2f} mΩ)")
        print(f"  Iterations: {result.get('iterations', 'N/A')}")
    
    
    return solver, df


def _get_battery_solver():
    """
    Helper function to create and return a BatterySolver instance.
    Returns: solver instance
    """
    battery_original_filename = 'models/atlas/atlas/propulsion/empirical_data/inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle'
    bat_filename = battery_original_filename + '.xlsx'
    cell_sheetname = 'BOL_cell_fct_CRate'
    config_sheetname = 'battery_config'
    
    return BatterySolver(bat_filename, cell_sheetname, config_sheetname)


def _get_ocv_interpolation_errors():
    """
    Helper function to calculate OCV interpolation errors.
    Returns: max_error, mean_error, max_relative_error
    """
    solver = _get_battery_solver()
    
    # Get interpolated values for all SOC points
    interpolated_ocv = solver.ocv_interp(solver.soc_data)
    actual_ocv = solver.ocv_data
    
    # Calculate errors
    errors = np.abs(interpolated_ocv - actual_ocv)
    relative_errors = errors / actual_ocv * 100
    
    return np.max(errors), np.mean(errors), np.max(relative_errors)


def _get_ir0_interpolation_errors():
    """
    Helper function to calculate IR0 interpolation errors.
    Returns: max_error, mean_error, max_relative_error
    """
    solver = _get_battery_solver()
    
    # Create meshgrid covering all training data points
    soc_mesh, c_rate_cur_mesh = np.meshgrid(solver.soc_data, solver.c_rate_cur_data, indexing='ij')
    
    # Flatten for interpolation
    soc_flat = soc_mesh.flatten()
    c_rate_cur_flat = c_rate_cur_mesh.flatten()
    
    # Get interpolated values for all points
    points = np.column_stack([soc_flat, c_rate_cur_flat])
    interpolated_ir0 = solver.ir0_interp(points)
    
    # Get actual values
    actual_ir0 = solver.ir0_data.flatten()
    
    # Calculate errors
    errors = np.abs(interpolated_ir0 - actual_ir0)
    relative_errors = errors / actual_ir0 * 100
    
    return np.max(errors), np.mean(errors), np.max(relative_errors)


# Pytest-compatible test functions
def test_ocv_interpolation_max_error(tolerance=1e-6):
    """Test OCV interpolation maximum error"""
    max_error, _, _ = _get_ocv_interpolation_errors()
    assert max_error < tolerance, f"OCV max error {max_error:.8f} V exceeds tolerance {tolerance}"


def test_ocv_interpolation_mean_error(tolerance=1e-6):
    """Test OCV interpolation mean error"""
    _, mean_error, _ = _get_ocv_interpolation_errors()
    assert mean_error < tolerance, f"OCV mean error {mean_error:.8f} V exceeds tolerance {tolerance}"


def test_ocv_interpolation_relative_error(tolerance=0.01):
    """Test OCV interpolation relative error"""
    _, _, max_relative_error = _get_ocv_interpolation_errors()
    assert max_relative_error < tolerance, f"OCV max relative error {max_relative_error:.4f}% exceeds tolerance {tolerance}%"


def test_ir0_interpolation_max_error(tolerance=1e-6):
    """Test IR0 interpolation maximum error"""
    max_error, _, _ = _get_ir0_interpolation_errors()
    assert max_error < tolerance, f"IR0 max error {max_error:.10f} Ω exceeds tolerance {tolerance}"


def test_ir0_interpolation_mean_error(tolerance=1e-6):
    """Test IR0 interpolation mean error"""
    _, mean_error, _ = _get_ir0_interpolation_errors()
    assert mean_error < tolerance, f"IR0 mean error {mean_error:.10f} Ω exceeds tolerance {tolerance}"


def test_ir0_interpolation_relative_error(tolerance=0.05):
    """Test IR0 interpolation relative error"""
    _, _, max_relative_error = _get_ir0_interpolation_errors()
    assert max_relative_error < tolerance, f"IR0 max relative error {max_relative_error:.4f}% exceeds tolerance {tolerance}%"


if __name__ == "__main__":
    solver, df = build_database(save_database=True, plot_database=False, battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105') 