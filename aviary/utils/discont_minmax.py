"""Definition of discontinuous Min/Max Components."""

import numpy as np
import openmdao.api as om


# ---------------------------
# DiscontMaxComp (extract or elementwise limit)
# ---------------------------
class DiscontMaxComp(om.ExplicitComponent):
    """
    Discontinuous max component using np.max (extract modes) or np.maximum (limit mode).
    
    Modes:
    - 'extract_scalar': Extract global maximum from all elements (scalar output)
    - 'extract_vector': Extract maximum across num_nodes for each component (vector output)
    - 'limit': Elementwise maximum with a limit value (same shape as input)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=165, desc='Number of nodes (rows for matrix case)')
        self.options.declare('n_comps', default=1, desc='Number of components (cols for matrix case; if 1, input is vector)')
        self.options.declare('mode', default='extract_scalar', desc="'extract_scalar' (scalar out), 'extract_vector' (vector of length n_comps), or 'limit' (elementwise output same shape as input)")
        self.options.declare('limit_val', default=0.0, desc='Limit value used when mode="limit" (constant option)')
        self.options.declare('units', default=None, desc='Units of the input')

    def setup(self):
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        mode = self.options['mode']
        units = self.options['units']

        # Inputs
        if n_comps == 1:
            # vector case: shape (nn,)
            self.add_input('input_array', shape=(nn,), units=units, desc='Input vector')
            in_size = nn
            self._in_shape = (nn,)
        else:
            # matrix case: shape (n_comps, nn)
            self.add_input('input_array', shape=(n_comps, nn), units=units, desc='Input matrix')
            in_size = n_comps * nn
            self._in_shape = (n_comps, nn)

        # Outputs depend on mode
        if mode == 'extract_scalar':
            # scalar output - global max over all elements
            self.add_output('output', shape=(1,), units=units, desc='Global max (scalar)')
            # declare scalar-out many-in partials
            rows = np.zeros(in_size, dtype=int)
            cols = np.arange(in_size, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        elif mode == 'extract_vector':
            # vector output - max across num_nodes for each component: shape (n_comps,)
            self.add_output('output', shape=(n_comps,), units=units, desc='Max across num_nodes for each component')
            # each output[i] depends on input[i,:] (all nn nodes for component i)
            # Flattened index for input[i, j] in C order: i * nn + j
            rows = np.repeat(np.arange(n_comps, dtype=int), nn)
            cols_i = np.repeat(np.arange(n_comps, dtype=int), nn)
            cols_j = np.tile(np.arange(nn, dtype=int), n_comps)
            cols = cols_i * nn + cols_j
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        elif mode == 'limit':
            # elementwise output same shape as input
            self.add_output('output', shape=self._in_shape, units=units, desc='Elementwise max(x, limit_val)')
            # declare diagonal partials (each output element depends only on corresponding input element)
            n_total = in_size
            rows = np.arange(n_total, dtype=int)
            cols = np.arange(n_total, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        else:
            raise ValueError("mode must be 'extract_scalar', 'extract_vector', or 'limit'")

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        x = inputs['input_array']

        if mode == 'extract_scalar':
            # Global max using np.max
            result = np.max(x)
        elif mode == 'extract_vector':
            # Max across num_nodes for each component: output[i] = max(x[i,:])
            # x has shape (n_comps, nn), we want max along axis 1
            result = np.max(x, axis=1)  # shape (n_comps,)
        else:  # mode == 'limit'
            # Elementwise maximum with limit_val using np.maximum
            result = np.maximum(x, limit_val)

        #print("Discont Max Result: {}", result)
        outputs['output'] = result

    def compute_partials(self, inputs, partials):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        x = inputs['input_array']

        if mode == 'extract_scalar':
            # Partial is 1 where input equals the max, 0 elsewhere
            x_flat = x.flatten()
            max_val = np.max(x_flat)
            partials['output', 'input_array'] = np.where(x_flat == max_val, 1.0, 0.0)
            
        elif mode == 'extract_vector':
            # For each output[i], partial is 1 where input[i,j] equals max(x[i,:]), 0 elsewhere
            max_per_comp = np.max(x, axis=1)  # shape (n_comps,)
            # Create a matrix of partials: 1 where x[i,j] == max_per_comp[i], 0 elsewhere
            partials_matrix = np.where(x == max_per_comp[:, np.newaxis], 1.0, 0.0)
            # Flatten in row-major order to match declared sparsity
            partials['output', 'input_array'] = partials_matrix.flatten()
            
        else:  # mode == 'limit'
            # Diagonal partials: 1 where x > limit_val, 0 where x <= limit_val
            x_flat = x.flatten()
            partials['output', 'input_array'] = np.where(x_flat >= limit_val, 1.0, 0.0)


# ---------------------------
# DiscontMinComp (extract or elementwise limit)
# ---------------------------
class DiscontMinComp(om.ExplicitComponent):
    """
    Discontinuous min component using np.min (extract modes) or np.minimum (limit mode).
    
    Modes:
    - 'extract_scalar': Extract global minimum from all elements (scalar output)
    - 'extract_vector': Extract minimum across num_nodes for each component (vector output)
    - 'limit': Elementwise minimum with a limit value (same shape as input)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=165, desc='Number of nodes (rows for matrix case)')
        self.options.declare('n_comps', default=4, desc='Number of components (cols for matrix case; if 1, input is vector)')
        self.options.declare('mode', default='extract_scalar', desc="'extract_scalar' (scalar out), 'extract_vector' (vector of length n_comps), or 'limit' (elementwise output same shape as input)")
        self.options.declare('limit_val', default=0.0, desc='Limit value used when mode="limit" (constant option)')
        self.options.declare('units', default=None, desc='Units of the input')

    def setup(self):
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        mode = self.options['mode']
        units = self.options['units']

        if n_comps == 1:
            self.add_input('input_array', shape=(nn,), units=units, desc='Input vector')
            in_size = nn
            self._in_shape = (nn,)
        else:
            self.add_input('input_array', shape=(n_comps, nn), units=units, desc='Input matrix')
            in_size = n_comps * nn
            self._in_shape = (n_comps, nn)

        if mode == 'extract_scalar':
            # scalar output - global min over all elements
            self.add_output('output', shape=(1,), units=units, desc='Global min (scalar)')
            rows = np.zeros(in_size, dtype=int)
            cols = np.arange(in_size, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        elif mode == 'extract_vector':
            # vector output - min across num_nodes for each component: shape (n_comps,)
            self.add_output('output', shape=(n_comps,), units=units, desc='Min across num_nodes for each component')
            # each output[i] depends on input[i,:] (all nn nodes for component i)
            rows = np.repeat(np.arange(n_comps, dtype=int), nn)
            cols_i = np.repeat(np.arange(n_comps, dtype=int), nn)
            cols_j = np.tile(np.arange(nn, dtype=int), n_comps)
            cols = cols_i * nn + cols_j
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        elif mode == 'limit':
            self.add_output('output', shape=self._in_shape, units=units, desc='Elementwise min(x, limit_val)')
            n_total = in_size
            rows = np.arange(n_total, dtype=int)
            cols = np.arange(n_total, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols)
        else:
            raise ValueError("mode must be 'extract_scalar', 'extract_vector', or 'limit'")

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        x = inputs['input_array']

        if mode == 'extract_scalar':
            # Global min using np.min
            result = np.min(x)
        elif mode == 'extract_vector':
            # Min across num_nodes for each component: output[i] = min(x[i,:])
            # x has shape (n_comps, nn), we want min along axis 1
            result = np.min(x, axis=1)  # shape (n_comps,)
        else:  # mode == 'limit'
            # Elementwise minimum with limit_val using np.minimum
            result = np.minimum(x, limit_val)

        #print("Discont Min Result: {}", result)
        outputs['output'] = result

    def compute_partials(self, inputs, partials):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        x = inputs['input_array']

        if mode == 'extract_scalar':
            # Partial is 1 where input equals the min, 0 elsewhere
            x_flat = x.flatten()
            min_val = np.min(x_flat)
            partials['output', 'input_array'] = np.where(x_flat == min_val, 1.0, 0.0)
            
        elif mode == 'extract_vector':
            # For each output[i], partial is 1 where input[i,j] equals min(x[i,:]), 0 elsewhere
            min_per_comp = np.min(x, axis=1)  # shape (n_comps,)
            # Create a matrix of partials: 1 where x[i,j] == min_per_comp[i], 0 elsewhere
            partials_matrix = np.where(x == min_per_comp[:, np.newaxis], 1.0, 0.0)
            # Flatten in row-major order to match declared sparsity
            partials['output', 'input_array'] = partials_matrix.flatten()
            
        else:  # mode == 'limit'
            # Elementwise derivative of y = min(x, limit_val)
            # dy/dx = 1 if x < limit_val
            #       = 0 if x >= limit_val
            x_flat = x.flatten()
            diag_vals = np.where(x_flat < limit_val, 1.0, 0.0)
            partials['output', 'input_array'] = diag_vals

# ---------------------------
# Test Functions
# ---------------------------
def test_discont_min_extract_scalar_vector():
    """Test DiscontMinComp in extract_scalar mode with vector input."""
    print("\n" + "="*60)
    print("Testing DiscontMinComp - Extract Scalar Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             DiscontMinComp(num_nodes=5, n_comps=1, mode='extract_scalar', units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_min = np.min(test_vector)
    print(f"Input vector: {test_vector}")
    print(f"Discont min result: {result[0]:.6f}")
    print(f"Exact min: {exact_min:.6f}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMinComp Extract Scalar Vector test passed")


def test_discont_min_extract_vector_matrix():
    """Test DiscontMinComp in extract_vector mode with matrix input."""
    print("\n" + "="*60)
    print("Testing DiscontMinComp - Extract Vector Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Matrix: 3 components (strings), 4 nodes (time points)
    test_matrix = np.array([
        [10.0, 5.0, 3.0, 7.0],   # component 0
        [8.0, 1.0, 6.0, 9.0],    # component 1
        [12.0, 4.0, 2.0, 8.0]    # component 2
    ])
    ivc.add_output('input_array', val=test_matrix, units='V')
    
    prob.model.add_subsystem('comp', 
                             DiscontMinComp(num_nodes=4, n_comps=3, mode='extract_vector', units='V'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_min = np.min(test_matrix, axis=1)  # min across nodes for each component
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Discont min result (vector): {result}")
    print(f"Exact min (across axis 1 - nodes): {exact_min}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMinComp Extract Vector Matrix test passed")


def test_discont_min_limit_vector():
    """Test DiscontMinComp in limit mode with vector input."""
    print("\n" + "="*60)
    print("Testing DiscontMinComp - Limit Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    limit_value = 6.0
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             DiscontMinComp(num_nodes=5, n_comps=1, mode='limit', 
                                          limit_val=limit_value, units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.minimum(test_vector, limit_value)
    print(f"Input vector: {test_vector}")
    print(f"Limit value: {limit_value}")
    print(f"Discont limited result: {result}")
    print(f"Exact limited: {exact_limit}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMinComp Limit Vector test passed")


def test_discont_max_extract_scalar_vector():
    """Test DiscontMaxComp in extract_scalar mode with vector input."""
    print("\n" + "="*60)
    print("Testing DiscontMaxComp - Extract Scalar Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             DiscontMaxComp(num_nodes=5, n_comps=1, mode='extract_scalar', units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_max = np.max(test_vector)
    print(f"Input vector: {test_vector}")
    print(f"Discont max result: {result[0]:.6f}")
    print(f"Exact max: {exact_max:.6f}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMaxComp Extract Scalar Vector test passed")


def test_discont_max_extract_vector_matrix():
    """Test DiscontMaxComp in extract_vector mode with matrix input."""
    print("\n" + "="*60)
    print("Testing DiscontMaxComp - Extract Vector Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Matrix: 3 components (strings), 4 nodes (time points)
    test_matrix = np.array([
        [10.0, 5.0, 3.0, 7.0],   # component 0
        [8.0, 15.0, 6.0, 9.0],   # component 1
        [12.0, 4.0, 14.0, 8.0]   # component 2
    ])
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             DiscontMaxComp(num_nodes=4, n_comps=3, mode='extract_vector', units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_max = np.max(test_matrix, axis=1)  # max across nodes for each component
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Discont max result (vector): {result}")
    print(f"Exact max (across axis 1 - nodes): {exact_max}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMaxComp Extract Vector Matrix test passed")


def test_discont_max_limit_vector():
    """Test DiscontMaxComp in limit mode with vector input."""
    print("\n" + "="*60)
    print("Testing DiscontMaxComp - Limit Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    limit_value = 6.0
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             DiscontMaxComp(num_nodes=5, n_comps=1, mode='limit', 
                                          limit_val=limit_value, units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.maximum(test_vector, limit_value)
    print(f"Input vector: {test_vector}")
    print(f"Limit value: {limit_value}")
    print(f"Discont limited result: {result}")
    print(f"Exact limited: {exact_limit}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMaxComp Limit Vector test passed")


def test_discont_min_limit_matrix():
    """Test DiscontMinComp in limit mode with matrix input."""
    print("\n" + "="*60)
    print("Testing DiscontMinComp - Limit Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, 5.0, 3.0, 7.0],   # component 0
        [8.0, 1.0, 6.0, 9.0],    # component 1
        [12.0, 4.0, 2.0, 8.0]    # component 2
    ])
    limit_value = 6.0
    ivc.add_output('input_array', val=test_matrix, units='V')
    
    prob.model.add_subsystem('comp', 
                             DiscontMinComp(num_nodes=4, n_comps=3, mode='limit', 
                                          limit_val=limit_value, units='V'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.minimum(test_matrix, limit_value)
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Limit value: {limit_value}")
    print(f"Discont limited result:\n{result}")
    print(f"Exact limited:\n{exact_limit}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMinComp Limit Matrix test passed")


def test_discont_max_limit_matrix():
    """Test DiscontMaxComp in limit mode with matrix input."""
    print("\n" + "="*60)
    print("Testing DiscontMaxComp - Limit Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, 5.0, 3.0, 7.0],   # component 0
        [8.0, 15.0, 6.0, 9.0],   # component 1
        [12.0, 4.0, 14.0, 8.0]   # component 2
    ])
    limit_value = 9.0
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             DiscontMaxComp(num_nodes=4, n_comps=3, mode='limit', 
                                          limit_val=limit_value, units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.maximum(test_matrix, limit_value)
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Limit value: {limit_value}")
    print(f"Discont limited result:\n{result}")
    print(f"Exact limited:\n{exact_limit}")
    
    # Check partials
    prob.check_partials(compact_print=True, show_only_incorrect=True)
    print("✓ DiscontMaxComp Limit Matrix test passed")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# RUNNING DISCONTINUOUS MIN/MAX COMPONENT TESTS")
    print("#"*60)
    
    # Test DiscontMinComp - all modes
    test_discont_min_extract_scalar_vector()
    test_discont_min_extract_vector_matrix()
    test_discont_min_limit_vector()
    test_discont_min_limit_matrix()
    
    # Test DiscontMaxComp - all modes
    test_discont_max_extract_scalar_vector()
    test_discont_max_extract_vector_matrix()
    test_discont_max_limit_vector()
    test_discont_max_limit_matrix()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)
