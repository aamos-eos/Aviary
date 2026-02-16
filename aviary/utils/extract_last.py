"""Definition of the ExtractLast Component."""

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
import openmdao.api as om


class ExtractLast(ExplicitComponent):
    """
    Extract the last value(s) from an input vector or matrix.
    
    Modes:
    - 'extract_scalar': Extract last element from a vector (nn,) -> scalar (1,)
    - 'extract_vector': Extract last column from a matrix (n_comps, nn) -> vector (n_comps,)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=11, desc='Number of nodes (time points) in the input')
        self.options.declare('n_comps', default=1, desc='Number of components (rows) in the input matrix')
        self.options.declare('units', default=None, desc='Units of the input')
        self.options.declare('mode', default='extract_scalar', values=['extract_scalar', 'extract_vector'],
                           desc="Mode: 'extract_scalar' for vector input, 'extract_vector' for matrix input")
    
    def setup(self):
        """
        Set up the component inputs and outputs.
        """
        nn = self.options['num_nodes']
        n_comps = self.options['n_comps']
        units = self.options['units']
        mode = self.options['mode']
        
        if mode == 'extract_scalar':
            # Input: vector (nn,), Output: scalar (1,)
            self.add_input('input_vector', shape=(nn,), units=units, desc='Input vector')
            self.add_output('output_scalar', shape=(1,), units=units, desc='Last value of input vector')
            
            # Declare partial derivatives - output only depends on last element
            self.declare_partials('output_scalar', 'input_vector', rows=[0], cols=[nn-1], val=[1.0])
            
        elif mode == 'extract_vector':
            # Input: matrix (n_comps, nn), Output: vector (n_comps,)
            self.add_input('input_matrix', shape=(n_comps, nn), units=units, desc='Input matrix')
            self.add_output('output_vector', shape=(n_comps,), units=units, desc='Last column of input matrix')
            
            # Declare partial derivatives - each output element depends on last element of its row
            # output[i] = input[i, nn-1]
            # So d(output[i])/d(input[i, nn-1]) = 1.0
            rows = np.arange(n_comps, dtype=int)
            cols = np.arange(n_comps, dtype=int) * nn + (nn - 1)  # Flattened index of [i, nn-1]
            self.declare_partials('output_vector', 'input_matrix', rows=rows, cols=cols, val=np.ones(n_comps))
        
    def compute(self, inputs, outputs):
        """
        Compute the last value(s) from the input.
        """
        mode = self.options['mode']
        
        if mode == 'extract_scalar':
            #print(f"Extracting last value from vector: {inputs['input_vector']}")
            outputs['output_scalar'] = inputs['input_vector'][-1]
        elif mode == 'extract_vector':
            # Extract last column from matrix
            outputs['output_vector'] = inputs['input_matrix'][:, -1]


def test_extract_last_scalar():
    """Test ExtractLast in extract_scalar mode."""
    print("\n" + "="*60)
    print("Testing ExtractLast - Extract Scalar Mode")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ivc.add_output('input_vector', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             ExtractLast(num_nodes=10, n_comps=1, mode='extract_scalar', units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output_scalar')
    expected = test_vector[-1]
    print(f"Input vector: {test_vector}")
    print(f"Extracted last value: {result[0]:.6f}")
    print(f"Expected: {expected:.6f}")
    print(f"Match: {np.allclose(result[0], expected)}")
    
    # Check partials
    print("\nChecking partials:")
    partials = prob.check_partials(compact_print=True, method='cs')
    print("✓ ExtractLast Extract Scalar test passed")
    
    return partials


def test_extract_last_vector():
    """Test ExtractLast in extract_vector mode."""
    print("\n" + "="*60)
    print("Testing ExtractLast - Extract Vector Mode")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Matrix: 4 components (strings), 5 nodes (time points)
    test_matrix = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],      # component 0
        [10.0, 20.0, 30.0, 40.0, 50.0], # component 1
        [100.0, 200.0, 300.0, 400.0, 500.0], # component 2
        [0.5, 0.6, 0.7, 0.8, 0.9]       # component 3
    ])
    ivc.add_output('input_matrix', val=test_matrix, units='V')
    
    prob.model.add_subsystem('comp', 
                             ExtractLast(num_nodes=5, n_comps=4, mode='extract_vector', units='V'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output_vector')
    expected = test_matrix[:, -1]  # Last column
    print(f"Input matrix (n_comps=4, num_nodes=5):\n{test_matrix}")
    print(f"Extracted last vector: {result}")
    print(f"Expected (last column): {expected}")
    print(f"Match: {np.allclose(result, expected)}")
    
    # Check partials
    print("\nChecking partials:")
    partials = prob.check_partials(compact_print=True, method='cs')
    print("✓ ExtractLast Extract Vector test passed")
    
    return partials


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# RUNNING EXTRACT LAST COMPONENT TESTS")
    print("#"*60)
    
    # Test both modes
    test_extract_last_scalar()
    test_extract_last_vector()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)
