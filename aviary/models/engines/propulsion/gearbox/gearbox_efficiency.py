"""
Gearbox efficiency interpolation using structured metamodel.

This module provides a component to interpolate gearbox efficiency
based on nacelle throttle_nac using empirical data from gearbox.xlsx.
"""
from aviary.utils.discont_minmax import DiscontMaxComp, DiscontMinComp

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import openmdao.jax as omj
# Import the global data store
from .gearbox_data import GearboxData
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp

class GearboxPowerCalculator(om.ExplicitComponent):
    """
    Component that calculates power output or input based on gearbox efficiency.
    
    Parameters
    ----------
    num_nodes : int
        Number of analysis points
    num_props : int
        Number of propellers/nacelles
    mode : str
        Either 'power_set' or 'thrust_set'
        - 'power_set': power_in is input, power_out = power_in * efficiency
        - 'thrust_set': power_out is input, power_in = power_out / efficiency
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('mode', default='power_set', desc='Mode: power_set or thrust_set')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        mode = self.options['mode']
        
        # Add efficiency input (always needed) - now comes as vector from converter
        self.add_input('gearbox_efficiency', shape=(npp, nn), val=0.99 * np.ones((npp, nn)), units=None, 
                      desc='Gearbox efficiency (0-1) per nacelle per time point')
        
        if mode == 'power_set':
            # Power in is input, power out is output
            self.add_input('power_in', shape=(npp, nn), units='W', 
                          desc='Input power to gearbox per nacelle')
            self.add_output('power_out', shape=(npp, nn), units='W', 
                           desc='Output power from gearbox per nacelle')
            
            # Declare partial derivatives
            rows = np.arange(npp * nn)
            cols = np.arange(npp * nn)
            self.declare_partials('power_out', 'power_in', rows=rows, cols=cols, method='exact')
            self.declare_partials('power_out', 'gearbox_efficiency', rows=rows, cols=cols, method='exact')
            
        elif mode == 'thrust_set':
            # Power out is input, power in is output
            self.add_input('power_out', shape=(npp, nn), units='W', 
                          desc='Output power from gearbox per nacelle')
            self.add_output('power_in', shape=(npp, nn), units='W', 
                           desc='Input power to gearbox per nacelle')
            
            # Declare partial derivatives
            rows = np.arange(npp * nn)
            cols = np.arange(npp * nn)
            self.declare_partials('power_in', 'power_out', rows=rows, cols=cols, method='exact')
            self.declare_partials('power_in', 'gearbox_efficiency', rows=rows, cols=cols, method='exact')
        else:
            raise ValueError(f"Mode must be 'power_set' or 'thrust_set', got '{mode}'")
    
    def compute(self, inputs, outputs):
        efficiency = inputs['gearbox_efficiency']  # Shape (npp, nn)
        
        if self.options['mode'] == 'power_set':
            power_in = inputs['power_in']  # Shape (npp, nn)
            power_out = power_in * efficiency  # Element-wise: (npp, nn) * (npp, nn) -> (npp, nn)
            outputs['power_out'] = power_out

        elif self.options['mode'] == 'thrust_set':
            power_out = inputs['power_out']  # Shape (npp, nn)
            # Avoid division by zero
            efficiency_safe = np.maximum(efficiency, 1e-6)  # Shape (npp, nn)
            power_in = power_out / efficiency_safe  # Element-wise: (npp, nn) / (npp, nn) -> (npp, nn)
            outputs['power_in'] = power_in
    
    def compute_partials(self, inputs, partials):
        efficiency = inputs['gearbox_efficiency']  # Shape (npp, nn)
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        if self.options['mode'] == 'power_set':
            power_in = inputs['power_in']  # Shape (npp, nn)
            # d(power_out)/d(power_in): efficiency for each element
            partials['power_out', 'power_in'] = efficiency.flatten()
            # d(power_out)/d(efficiency): power_in for each element
            partials['power_out', 'gearbox_efficiency'] = power_in.flatten()
            
        elif self.options['mode'] == 'thrust_set':
            power_out = inputs['power_out']  # Shape (npp, nn)
            efficiency_safe = np.maximum(efficiency, 1e-6)  # Shape (npp, nn)
            # d(power_in)/d(power_out): 1/efficiency for each element
            partials['power_in', 'power_out'] = (1.0 / efficiency_safe).flatten()
            # d(power_in)/d(efficiency): -power_out/efficiency^2 for each element
            partials['power_in', 'gearbox_efficiency'] = (-power_out / (efficiency_safe**2)).flatten()

class GearboxEfficiencyGroup(om.Group):
    """
    OpenMDAO group for gearbox efficiency calculation.
    
    This group contains the gearbox efficiency interpolator and can be
    easily integrated into larger propulsion system models.
    """
    _data_loaded = False
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of nodes')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('gearbox_filename', default='gearbox.xlsx', 
                           desc='Path to gearbox data file')
        self.options.declare('mode', default='power_set', 
                           desc='Mode: power_set or thrust_set')
        
        self._load_data()
    
    @classmethod
    def _load_data(cls):
        if cls._data_loaded:
            return
        GearboxData.load_data(gearbox_filename='models/atlas/atlas/propulsion/empirical_data/gearbox.xlsx', sheet_name='data')
        cls._data_loaded = True
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        mode = self.options['mode']
        
        # Add matrix to vector converter for throttle input
        self.add_subsystem('throttle_matrix_to_vector', 
                          MatrixToVectorConverter(
                              num_nodes=nn,
                              num_comps=npp,
                              input_names=['throttle_nac'],
                              output_names=['throttle_nac_vect'],
                              units={'throttle_nac': None}
                          ), promotes_inputs=['throttle_nac'], promotes_outputs=['throttle_nac_vect'])

        

        self.connect('lim_min_throttle.output', 'lim_max_throttle.input_array')

        #self.add_subsystem('lim_min_throttle', SmoothMaxComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=0, n_comps=1), promotes_inputs=[('input_array', 'throttle_nac_vect')], promotes_outputs=[])
        #self.add_subsystem('lim_max_throttle', SmoothMinComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=1, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'throttle_nac_clipped')])
        self.add_subsystem('lim_min_throttle', DiscontMaxComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=0, n_comps=1), promotes_inputs=[('input_array', 'throttle_nac_vect')], promotes_outputs=[])
        self.add_subsystem('lim_max_throttle', DiscontMinComp(num_nodes=nn * npp, mode='limit', units=None, limit_val=1, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'throttle_nac_clipped')])


        #print("gb efficiency interp")
        # Create gearbox efficiency metamodel
        gearbox_efficiency_interp = om.MetaModelStructuredComp(vec_size=npp*nn, method='akima')
        gearbox_efficiency_interp.add_input('throttle_nac_clipped', 0.7, training_data=GearboxData.throttle_data, units=None)
        gearbox_efficiency_interp.add_output('gearbox_efficiency_vect', 0.99, training_data=GearboxData.efficiency_data, units=None)
        gearbox_efficiency_interp.options['extrapolate'] = True
        
        self.add_subsystem('gearbox_efficiency_interp', gearbox_efficiency_interp, 
                          promotes_inputs=['throttle_nac_clipped'], promotes_outputs=['gearbox_efficiency_vect'])
        
                
        # Add vector to matrix converter for efficiency output
        self.add_subsystem('efficiency_vector_to_matrix', 
                          VectorToMatrixConverter(
                              num_nodes=nn,
                              num_comps=npp,
                              input_names=['gearbox_efficiency_vect'],
                              output_names=['gearbox_efficiency'],
                              units={'gearbox_efficiency_vect': None}
                          ),promotes_inputs=['gearbox_efficiency_vect'], promotes_outputs=['gearbox_efficiency'])
        
        # Add power calculator component
        self.add_subsystem('power_calculator', 
                          GearboxPowerCalculator(num_nodes=nn, num_props=npp, mode=mode),
                          promotes_inputs=['*'], 
                          promotes_outputs=['*'])

        #self.set_input_defaults("throttle_nac", 0.5, units = None)


def create_gearbox_efficiency_test():
    """
    Create a test using the same throttle values from experimental data.
    """
    # Load experimental data to get the throttle values
    GearboxData.load_data(gearbox_filename='models/atlas/atlas/propulsion/empirical_data/gearbox.xlsx', sheet_name='data')
    exp_throttle = GearboxData.throttle_data
    exp_efficiency = GearboxData.efficiency_data
    
    # Create test case using experimental throttle values
    prob = om.Problem(reports=False)
    
    # Add independent variable using experimental throttle values
    ivc = om.IndepVarComp()
    ivc.add_output('throttle_nac', val=exp_throttle, units=None)
    # Create power values for testing (scaled to match number of throttle points)
    # For test, use single nacelle (npp=1), so shape is (1, nn)
    power_values = np.linspace(1000.0, 5000.0, len(exp_throttle)).reshape(1, -1)
    ivc.add_output('power_in', val=power_values, units='W')
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Add gearbox efficiency group
    prob.model.add_subsystem('gearbox',
                           GearboxEfficiencyGroup(num_nodes=len(exp_throttle), num_props=1, mode='power_set'),
                           promotes=['*'])
    
    # Setup and run
    prob.setup(force_alloc_complex=True)
    prob.run_model()

    prob.check_partials(compact_print=True,method='cs')
    
    # Print results
    print("Gearbox Efficiency Test Results:")
    print("Throttle:", prob['throttle_nac'])
    print("Efficiency:", prob['gearbox_efficiency'])
    print("Power In (W):", prob['power_in'])
    print("Power Out (W):", prob['power_out'])
    
    return prob


def plot_gearbox_efficiency_test():
    """
    Create and plot gearbox efficiency test results with experimental data comparison.
    """
    # Create test problem
    prob = create_gearbox_efficiency_test()
    
    # Extract interpolated data (flatten to 1D for plotting)
    throttle = prob['throttle_nac'].flatten()
    efficiency = prob['gearbox_efficiency'].flatten()
    power_in = prob['power_in'].flatten()
    power_out = prob['power_out'].flatten()
    
    # Get experimental data for comparison
    exp_throttle = GearboxData.throttle_data
    exp_efficiency = GearboxData.efficiency_data
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Efficiency vs Throttle (with experimental data)
    axes[0, 0].plot(throttle, efficiency, 'bo-', linewidth=2, markersize=6, label='Interpolated')
    axes[0, 0].plot(exp_throttle, exp_efficiency, 'rs-', linewidth=2, markersize=8, 
                    label='Experimental Data', alpha=0.8)
    axes[0, 0].set_xlabel('Throttle')
    axes[0, 0].set_ylabel('Gearbox Efficiency')
    axes[0, 0].set_title('Gearbox Efficiency vs Throttle')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    
    # Plot 2: Power In vs Power Out
    axes[0, 1].plot(power_in, power_out, 'ro-', linewidth=2, markersize=6, label='Actual')
    axes[0, 1].plot([0, max(power_in)], [0, max(power_in)], 'k--', alpha=0.5, label='Perfect Efficiency')
    axes[0, 1].set_xlabel('Power In (W)')
    axes[0, 1].set_ylabel('Power Out (W)')
    axes[0, 1].set_title('Power Out vs Power In')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Power Loss vs Throttle
    power_loss = power_in - power_out
    axes[1, 0].plot(throttle, power_loss, 'go-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Throttle')
    axes[1, 0].set_ylabel('Power Loss (W)')
    axes[1, 0].set_title('Power Loss vs Throttle')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Efficiency vs Power In
    axes[1, 1].plot(power_in, efficiency, 'mo-', linewidth=2, markersize=6, label='Interpolated')
    # Create experimental efficiency vs power curve (assuming constant power levels)
    exp_power_levels = np.linspace(min(power_in), max(power_in), len(exp_efficiency))
    axes[1, 1].plot(exp_power_levels, exp_efficiency, 'cs-', linewidth=2, markersize=8, 
                    label='Experimental Data', alpha=0.8)
    axes[1, 1].set_xlabel('Power In (W)')
    axes[1, 1].set_ylabel('Gearbox Efficiency')
    axes[1, 1].set_title('Efficiency vs Power In')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    # Plot 5: Relative Error vs Throttle (direct comparison - same throttle values)
    if len(exp_throttle) > 1:
        # Calculate relative error directly since throttle values are the same
        relative_errors = []
        for i in range(len(throttle)):
            if exp_efficiency[i] > 0:  # Avoid division by zero
                rel_error = np.abs(efficiency[i] - exp_efficiency[i]) / exp_efficiency[i] * 100
                relative_errors.append(rel_error)
        
        if relative_errors:
            axes[1, 2].plot(throttle, relative_errors, 'ko-', linewidth=2, markersize=6)
            axes[1, 2].set_xlabel('Throttle')
            axes[1, 2].set_ylabel('Relative Error (%)')
            axes[1, 2].set_title('Relative Error vs Throttle')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # Add statistics text
            mean_error = np.mean(relative_errors)
            max_error = np.max(relative_errors)
            axes[1, 2].text(0.05, 0.95, f'Mean Error: {mean_error:.2f}%\nMax Error: {max_error:.2f}%', 
                            transform=axes[1, 2].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[1, 2].text(0.5, 0.5, 'No valid experimental data\nfor error calculation', 
                            transform=axes[1, 2].transAxes, ha='center', va='center')
            axes[1, 2].set_title('Relative Error vs Throttle')
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient experimental data\nfor error calculation', 
                        transform=axes[1, 2].transAxes, ha='center', va='center')
        axes[1, 2].set_title('Relative Error vs Throttle')
    
    plt.tight_layout()
    plt.savefig('gearbox_efficiency_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison statistics
    print(f"\nGearbox Efficiency Comparison:")
    print(f"Experimental throttle range: {min(exp_throttle):.2f} to {max(exp_throttle):.2f}")
    print(f"Experimental efficiency range: {min(exp_efficiency):.3f} to {max(exp_efficiency):.3f}")
    print(f"Interpolated throttle range: {min(throttle):.2f} to {max(throttle):.2f}")
    print(f"Interpolated efficiency range: {min(efficiency):.3f} to {max(efficiency):.3f}")
    
    # Calculate interpolation accuracy at experimental points
    if len(exp_throttle) > 1:
        interp_func = interp1d(throttle, efficiency, kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_at_exp = interp_func(exp_throttle)
        mae = np.mean(np.abs(interp_at_exp - exp_efficiency))
        print(f"Mean Absolute Error at experimental points: {mae:.4f}")
        
        # Calculate relative error statistics (direct comparison - same throttle values)
        relative_errors = []
        for i in range(len(throttle)):
            if exp_efficiency[i] > 0:
                rel_error = np.abs(efficiency[i] - exp_efficiency[i]) / exp_efficiency[i] * 100
                relative_errors.append(rel_error)
        
        if relative_errors:
            print(f"Relative Error Statistics:")
            print(f"  Mean Relative Error: {np.mean(relative_errors):.2f}%")
            print(f"  Max Relative Error: {np.max(relative_errors):.2f}%")
            print(f"  Min Relative Error: {np.min(relative_errors):.2f}%")
            print(f"  Std Relative Error: {np.std(relative_errors):.2f}%")
    
    print(f"\nGearbox efficiency test plots saved as 'gearbox_efficiency_test.png'")
    
    return prob


if __name__ == "__main__":
    # Run test and create plots

    # Run Model and Print Results
    #create_gearbox_efficiency_test()

    # Plot results
    prob = plot_gearbox_efficiency_test()
    