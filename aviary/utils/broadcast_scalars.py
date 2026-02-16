import numpy as np
import openmdao.api as om

class ScalarToMatrixBroadcast(om.ExplicitComponent):
    """
    Broadcast a scalar variable to a vector for all time nodes.
    This ensures all time nodes have the same variable.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of time nodes')
        self.options.declare('num_comps', default=1, desc='Number of components')
        self.options.declare('units', default=None, desc='Units of the variable')
    
    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        num_comps = self.options['num_comps']

        # Input: scalar variable
        self.add_input('scalar', val=0.5, units=units, desc='Variable (scalar)')
        
        # Output: vector variable
        self.add_output('matrix', val=np.ones((num_comps, nn))*0.5, units=units, 
                       shape=(num_comps, nn), desc='Variable (matrix)')
        
        # Declare partials structure (sparse format: each output element depends on scalar)
        # All output elements (num_comps * nn) depend on the single scalar input (index 0)
        self.declare_partials('matrix', 'scalar', rows=np.arange(num_comps*nn), cols=np.zeros(num_comps*nn, dtype=int))
    
    def compute(self, inputs, outputs):
        scalar = inputs['scalar']
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        units = self.options['units']
        #print(f"Broadcasted Scalar: {scalar} ({units})")
        outputs['matrix'] = np.full((num_comps, num_nodes), scalar)

    def compute_partials(self, inputs, partials):
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        # All partials are 1.0 (scalar broadcasts to all matrix elements)
        # Must be flattened to match the sparse format (rows/cols)
        partials['matrix', 'scalar'] = np.ones(num_comps * num_nodes)

class ScalarToVectorBroadcast(om.ExplicitComponent):
    """
    Broadcast a scalar variable to a vector for all time nodes.
    This ensures all time nodes have the same variable.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of time nodes')
        self.options.declare('units', default=None, desc='Units of the variable')
    
    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']

        # Input: scalar variable
        self.add_input('scalar', val=0.5, units=units, desc='Variable (scalar)')
        
        # Output: vector variable
        self.add_output('vector', val=np.ones(nn)*0.5, units=units, 
                       shape=(nn,), desc='Variable (vector)')
        
        # Declare partials
        self.declare_partials('vector', 'scalar', 
                             val=np.ones(nn), rows=np.arange(nn), cols=np.zeros(nn))
    
    def compute(self, inputs, outputs):
        scalar = inputs['scalar']
        units = self.options['units']
        #print(f"Broadcasted Scalar: {scalar} ({units})")
        outputs['vector'] = np.full(self.options['num_nodes'], scalar)


def test_scalar_to_vector_broadcast():
    """Test ScalarToVectorBroadcast component."""
    print("\n" + "="*60)
    print("Testing ScalarToVectorBroadcast")
    print("="*60)
    
    nn = 10
    
    prob = om.Problem()
    prob.model.add_subsystem('comp', ScalarToVectorBroadcast(num_nodes=nn, units='kg'))
    prob.setup()
    
    # Test with a scalar value
    prob.set_val('comp.scalar', 42.5)
    
    prob.run_model()
    
    print(f"Input scalar: {prob.get_val('comp.scalar')}")
    print(f"Output vector: {prob.get_val('comp.vector')}")
    print(f"Expected: All elements should be {prob.get_val('comp.scalar')[0]}")
    
    # Verify all elements are equal to the scalar
    output_vector = prob.get_val('comp.vector')
    all_equal = np.all(output_vector == prob.get_val('comp.scalar')[0])
    print(f"All elements equal to scalar: {all_equal}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ ScalarToVectorBroadcast test passed")


def test_scalar_to_matrix_broadcast():
    """Test ScalarToMatrixBroadcast component."""
    print("\n" + "="*60)
    print("Testing ScalarToMatrixBroadcast")
    print("="*60)
    
    nn = 10
    num_comps = 3

    prob = om.Problem()
    prob.model.add_subsystem('comp', ScalarToMatrixBroadcast(num_nodes=nn, num_comps=num_comps, units='kg'))
    prob.setup()
    
    # Test with a scalar value
    prob.set_val('comp.scalar', 42.5)

    prob.run_model()
    
    print(f"Input scalar: {prob.get_val('comp.scalar')}")
    print(f"Output matrix: {prob.get_val('comp.matrix')}")
    print(f"Expected: All elements should be {prob.get_val('comp.scalar')[0]}")
    
    # Verify all elements are equal to the scalar
    output_matrix = prob.get_val('comp.matrix')
    all_equal = np.all(output_matrix == prob.get_val('comp.scalar')[0])
    print(f"All elements equal to scalar: {all_equal}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ ScalarToMatrixBroadcast test passed")

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# RUNNING BROADCAST_SCALARS COMPONENT TESTS")
    print("#"*60)
    
    #test_scalar_to_vector_broadcast()
    test_scalar_to_matrix_broadcast()
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)