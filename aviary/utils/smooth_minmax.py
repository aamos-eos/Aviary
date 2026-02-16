"""Definition of the ExtractMin Component."""

import numpy as np
import openmdao.api as om
import jax

import openmdao.api as om
import numpy as np
from scipy.special import logsumexp

import numpy as np
import openmdao.api as om
from scipy.special import logsumexp

# ---------------------------
# SmoothMaxComp (extract or elementwise limit)
# ---------------------------
class SmoothMaxComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=165, desc='Number of nodes (rows for matrix case)')
        self.options.declare('n_comps', default=1, desc='Number of components (cols for matrix case; if 1, input is vector)')
        self.options.declare('mode', default='extract_scalar', desc="'extract_scalar' (scalar out), 'extract_vector' (vector of length num_nodes), or 'limit' (elementwise output same shape as input)")
        self.options.declare('limit_val', default=0.0, desc='Limit value used when mode="limit" (constant option)')
        self.options.declare('units', default=None, desc='Units of the input')
        self.options.declare('default_input', default=1.0, desc='Default value for input (val parameter)')
        self.options.declare('default_output', default=1.0, desc='Default value for output (val parameter)')

    def setup(self):
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        mode = self.options['mode']
        units = self.options['units']
        default_input = float(self.options['default_input'])
        default_output = float(self.options['default_output'])

        # Inputs
        if n_comps == 1:
            # vector case: shape (nn,)
            self.add_input('input_array', shape=(nn,), units=units, val=default_input, desc='Input vector')
            in_size = nn
            self._in_shape = (nn,)
        else:
            # matrix case: shape (n_comps, nn)
            self.add_input('input_array', shape=(n_comps, nn), units=units, val=default_input, desc='Input matrix')
            in_size = n_comps * nn
            self._in_shape = (n_comps, nn)

        # Outputs depend on mode
        if mode == 'extract_scalar':
            # scalar output - global max over all elements
            self.add_output('output', shape=(1,), units=units, val=default_output, desc='Smoothed global max (scalar)')
            # declare scalar-out many-in partials
            rows = np.zeros(in_size, dtype=int)
            cols = np.arange(in_size, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        elif mode == 'extract_vector':
            # vector output - max across num_nodes for each component: shape (n_comps,)
            self.add_output('output', shape=(n_comps,), units=units, val=default_output, desc='Smoothed max across num_nodes for each component')
            # each output[i] depends on input[i,:] (all nn nodes for component i)
            # Flattened index for input[i, j] in C order: i * nn + j
            rows = np.repeat(np.arange(n_comps, dtype=int), nn)  # [0,0,0,..., 1,1,1,..., 2,2,2,...] for nn nodes
            cols_i = np.repeat(np.arange(n_comps, dtype=int), nn)  # [0,0,0,..., 1,1,1,..., 2,2,2,...]
            cols_j = np.tile(np.arange(nn, dtype=int), n_comps)  # [0,1,2,...,nn-1, 0,1,2,...,nn-1, ...]
            cols = cols_i * nn + cols_j  # Flattened indices
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        elif mode == 'limit':
            # elementwise output same shape as input
            self.add_output('output', shape=self._in_shape, units=units, val=default_output, desc='Smoothed elementwise max(x, limit_val)')
            # declare diagonal partials (each output element depends only on corresponding input element)
            n_total = in_size
            rows = np.arange(n_total, dtype=int)
            cols = np.arange(n_total, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        else:
            raise ValueError("mode must be 'extract_scalar', 'extract_vector', or 'limit'")

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])

        x = inputs['input_array']
        #print("Input Array: {}", x)
    
        # flatten row-major for extract_scalar; keep original shape for elementwise limit output
        if x.ndim == 1:
            x_flat = x
        else:
            x_flat = x.flatten()

        # Calculate mu automatically based on input magnitude
        # Use range of input values, with fallback to max absolute value
        x_range = np.max(x_flat) - np.min(x_flat)
        x_max_abs = np.max(np.abs(x_flat))
        # mu scales with input magnitude (1e-3 factor), with minimum to avoid numerical issues
        mu = 1e-3 * max(x_range, x_max_abs * 1e-6, 1e-10)

        if mode == 'extract_scalar':
            result = mu * logsumexp(x_flat / mu)
        elif mode == 'extract_vector':
            # max across num_nodes for each component: output[i] = smooth_max(x[i,:])
            # x has shape (n_comps, nn), we want max along axis 1
            # For extract_vector, calculate mu per component based on that component's range
            if n_comps > 1:
                # Calculate mu for each component separately
                mu_per_comp = np.zeros(n_comps)
                for i in range(n_comps):
                    comp_range = np.max(x[i, :]) - np.min(x[i, :])
                    comp_max_abs = np.max(np.abs(x[i, :]))
                    mu_per_comp[i] = 1e-3 * max(comp_range, comp_max_abs * 1e-6, 1e-10)
                # Use average mu for consistency (or could use per-component mu)
                mu = np.mean(mu_per_comp)
            result = mu * logsumexp(x / mu, axis=1)  # shape (n_comps,)
        else:  # mode == 'limit'
            # elementwise smooth max between x_i and limit_val
            limit_scaled = limit_val / mu
            # stack (2, N) then logsumexp along axis 0
            stacked = np.vstack((x_flat / mu, np.full_like(x_flat / mu, limit_scaled)))
            lse_pair = logsumexp(stacked, axis=0)   # shape (in_size,)
            out_flat = mu * lse_pair
            result = out_flat.reshape(self._in_shape)

        #print("Smooth Max Result: {}", result)
        outputs['output'] = result

    def compute_partials(self, inputs, partials):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])

        x = inputs['input_array']
        if x.ndim == 1:
            x_flat = x
        else:
            x_flat = x.flatten()
        in_size = x_flat.size

        # Calculate mu automatically based on input magnitude (same as in compute)
        x_range = np.max(x_flat) - np.min(x_flat)
        x_max_abs = np.max(np.abs(x_flat))
        mu = 1e-3 * max(x_range, x_max_abs * 1e-6, 1e-10)

        if mode == 'extract_scalar':
            # softmax over all entries
            lse = logsumexp(x_flat / mu)
            grad_flat = np.exp(x_flat / mu - lse)  # shape (in_size,)
            partials['output', 'input_array'] = grad_flat.reshape(1, -1)

        elif mode == 'extract_vector':
            # for each output[i], compute softmax over x[i,:] (nn elements)
            # output[i] = mu * logsumexp(x[i,:] / mu)
            # d output[i] / d x[i,j] = exp(x[i,j]/mu) / sum_k(exp(x[i,k]/mu))
            # Calculate mu per component for consistency
            if n_comps > 1:
                mu_per_comp = np.zeros(n_comps)
                for i in range(n_comps):
                    comp_range = np.max(x[i, :]) - np.min(x[i, :])
                    comp_max_abs = np.max(np.abs(x[i, :]))
                    mu_per_comp[i] = 1e-3 * max(comp_range, comp_max_abs * 1e-6, 1e-10)
                mu = np.mean(mu_per_comp)
            lse_per_comp = logsumexp(x / mu, axis=1)  # shape (n_comps,)
            # grad[i,j] = exp(x[i,j]/mu - lse[i])
            grad_matrix = np.exp(x / mu - lse_per_comp[:, np.newaxis])  # shape (n_comps, nn)
            # flatten in row-major order to match declared sparsity
            # declared as: rows = repeat(arange(n_comps), nn), cols = repeat(arange(n_comps), nn) * nn + tile(arange(nn), n_comps)
            # so partials array should be ordered as: [d_out[0]/d_x[0,0], d_out[0]/d_x[0,1], ..., d_out[1]/d_x[1,0], ...]
            partials['output', 'input_array'] = grad_matrix.flatten()

        else:  # mode == 'limit' : diagonal partials
            # for each element i: derivative wrt x_i is
            # d/dx_i mu*log( exp(x_i/mu) + exp(limit/mu) ) = exp(x_i/mu) / (exp(x_i/mu) + exp(limit/mu))
            # compute in stable form:
            x_scaled = x_flat / mu
            limit_scaled = limit_val / mu
            # to avoid overflow, we use exp(x_scaled - max) etc; but since only 2 elements, stable direct:
            # p_i = exp(x_scaled) / (exp(x_scaled) + exp(limit_scaled)) = 1 / (1 + exp(limit_scaled - x_scaled))
            denom = 1.0 + np.exp(limit_scaled - x_scaled)
            diag_vals = 1.0 / denom   # shape (in_size,)
            # assign flattened vector of diagonal entries corresponding to rows/cols declared
            partials['output', 'input_array'] = diag_vals.reshape(-1)

# ---------------------------
# SmoothMinComp (extract or elementwise limit)
# ---------------------------
class SmoothMinComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=165, desc='Number of nodes (rows for matrix case)')
        self.options.declare('n_comps', default=4, desc='Number of components (cols for matrix case; if 1, input is vector)')
        self.options.declare('mode', default='extract_scalar', desc="'extract_scalar' (scalar out), 'extract_vector' (vector of length num_nodes), or 'limit' (elementwise output same shape as input)")
        self.options.declare('limit_val', default=0.0, desc='Limit value used when mode="limit" (constant option)')
        self.options.declare('units', default=None, desc='Units of the input')
        self.options.declare('default_input', default=1.0, desc='Default value for input (val parameter)')
        self.options.declare('default_output', default=1.0, desc='Default value for output (val parameter)')

    def setup(self):
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])
        mode = self.options['mode']
        units = self.options['units']
        default_input = float(self.options['default_input'])
        default_output = float(self.options['default_output'])

        if n_comps == 1:
            self.add_input('input_array', shape=(nn,), units=units, val=default_input, desc='Input vector')
            in_size = nn
            self._in_shape = (nn,)
        else:
            self.add_input('input_array', shape=(n_comps, nn), units=units, val=default_input, desc='Input matrix')
            in_size = n_comps * nn
            self._in_shape = (n_comps, nn)

        if mode == 'extract_scalar':
            # scalar output - global min over all elements
            self.add_output('output', shape=(1,), units=units, val=default_output, desc='Smoothed global min (scalar)')
            rows = np.zeros(in_size, dtype=int)
            cols = np.arange(in_size, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        elif mode == 'extract_vector':
            # vector output - min across num_nodes for each component: shape (n_comps,)
            self.add_output('output', shape=(n_comps,), units=units, val=default_output, desc='Smoothed min across num_nodes for each component')
            # each output[i] depends on input[i,:] (all nn nodes for component i)
            # Flattened index for input[i, j] in C order: i * nn + j
            rows = np.repeat(np.arange(n_comps, dtype=int), nn)  # [0,0,0,..., 1,1,1,..., 2,2,2,...] for nn nodes
            cols_i = np.repeat(np.arange(n_comps, dtype=int), nn)  # [0,0,0,..., 1,1,1,..., 2,2,2,...]
            cols_j = np.tile(np.arange(nn, dtype=int), n_comps)  # [0,1,2,...,nn-1, 0,1,2,...,nn-1, ...]
            cols = cols_i * nn + cols_j  # Flattened indices
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        elif mode == 'limit':
            self.add_output('output', shape=self._in_shape, units=units, val=default_output, desc='Smoothed elementwise min(x, limit_val)')
            n_total = in_size
            rows = np.arange(n_total, dtype=int)
            cols = np.arange(n_total, dtype=int)
            self.declare_partials('output', 'input_array', rows=rows, cols=cols, method='exact')
        else:
            raise ValueError("mode must be 'extract_scalar', 'extract_vector', or 'limit'")

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])

        x = inputs['input_array']
        #print("Input Array: {}", x)
        if x.ndim == 1:
            x_flat = x
        else:
            x_flat = x.flatten()

        # Calculate mu automatically based on input magnitude
        # Use range of input values, with fallback to max absolute value
        x_range = np.max(x_flat) - np.min(x_flat)
        x_max_abs = np.max(np.abs(x_flat))
        # mu scales with input magnitude (1e-3 factor), with minimum to avoid numerical issues
        mu = 1e-3 * max(x_range, x_max_abs * 1e-6, 1e-10)

        if mode == 'extract_scalar':
            result = -mu * logsumexp(-x_flat / mu)
        elif mode == 'extract_vector':
            # min across num_nodes for each component: output[i] = smooth_min(x[i,:])
            # x has shape (n_comps, nn), we want min along axis 1
            # For extract_vector, calculate mu per component based on that component's range
            if n_comps > 1:
                # Calculate mu for each component separately
                mu_per_comp = np.zeros(n_comps)
                for i in range(n_comps):
                    comp_range = np.max(x[i, :]) - np.min(x[i, :])
                    comp_max_abs = np.max(np.abs(x[i, :]))
                    mu_per_comp[i] = 1e-3 * max(comp_range, comp_max_abs * 1e-6, 1e-10)
                # Use average mu for consistency (or could use per-component mu)
                mu = np.mean(mu_per_comp)
            result = -mu * logsumexp(-x / mu, axis=1)  # shape (n_comps,)
        else:  # mode == 'limit'
            # elementwise smooth min between x_i and limit_val:
            # out_i = -mu * logsumexp([-x_i/mu, -limit/mu])  => mu * lse of negatives with sign
            stacked = np.vstack((-x_flat / mu, np.full_like(x_flat / mu, -limit_val / mu)))
            lse_pair = logsumexp(stacked, axis=0)
            out_flat = -mu * lse_pair
            result = out_flat.reshape(self._in_shape)

        #print("Smooth Min Result: {}", result)
        outputs['output'] = result

    def compute_partials(self, inputs, partials):
        mode = self.options['mode']
        limit_val = float(self.options['limit_val'])
        nn = int(self.options['num_nodes'])
        n_comps = int(self.options['n_comps'])

        x = inputs['input_array']
        if x.ndim == 1:
            x_flat = x
        else:
            x_flat = x.flatten()
        in_size = x_flat.size

        # Calculate mu automatically based on input magnitude (same as in compute)
        x_range = np.max(x_flat) - np.min(x_flat)
        x_max_abs = np.max(np.abs(x_flat))
        mu = 1e-3 * max(x_range, x_max_abs * 1e-6, 1e-10)

        if mode == 'extract_scalar':
            lse = logsumexp(-x_flat / mu)
            grad_flat = np.exp(-x_flat / mu - lse)  # shape (in_size,)
            partials['output', 'input_array'] = grad_flat.reshape(1, -1)
        elif mode == 'extract_vector':
            # for each output[i], compute softmax over -x[i,:] / mu (nn elements)
            # output[i] = -mu * logsumexp(-x[i,:] / mu)
            # d output[i] / d x[i,j] = exp(-x[i,j]/mu) / sum_k(exp(-x[i,k]/mu))
            # Calculate mu per component for consistency
            if n_comps > 1:
                mu_per_comp = np.zeros(n_comps)
                for i in range(n_comps):
                    comp_range = np.max(x[i, :]) - np.min(x[i, :])
                    comp_max_abs = np.max(np.abs(x[i, :]))
                    mu_per_comp[i] = 1e-3 * max(comp_range, comp_max_abs * 1e-6, 1e-10)
                mu = np.mean(mu_per_comp)
            lse_per_comp = logsumexp(-x / mu, axis=1)  # shape (n_comps,)
            # grad[i,j] = exp(-x[i,j]/mu - lse[i])
            grad_matrix = np.exp(-x / mu - lse_per_comp[:, np.newaxis])  # shape (n_comps, nn)
            # flatten in row-major order to match declared sparsity
            # declared as: rows = repeat(arange(n_comps), nn), cols = repeat(arange(n_comps), nn) * nn + tile(arange(nn), n_comps)
            # so partials array should be ordered as: [d_out[0]/d_x[0,0], d_out[0]/d_x[0,1], ..., d_out[1]/d_x[1,0], ...]
            partials['output', 'input_array'] = grad_matrix.flatten()
        else:  # mode == 'limit'
            # elementwise diagonal derivatives:
            # for each element i: derivative of -mu*logsumexp([-x_i/mu, -limit/mu]) wrt x_i:
            # = -mu * ( ( -1/mu * exp(-x_i/mu) ) / (exp(-x_i/mu)+exp(-limit/mu)) )
            # simplifies to exp(-x_i/mu) / (exp(-x_i/mu) + exp(-limit/mu))
            x_scaled = -x_flat / mu
            limit_scaled = -limit_val / mu
            denom = 1.0 + np.exp(limit_scaled - x_scaled)  # = (exp(-x_i/mu)+exp(-limit/mu)) / exp(-x_i/mu)
            diag_vals = 1.0 / denom   # shape (in_size,)
            partials['output', 'input_array'] = diag_vals.reshape(-1)


def test_smooth_min_extract_scalar_vector():
    """Test SmoothMinComp in extract_scalar mode with vector input."""
    print("\n" + "="*60)
    print("Testing SmoothMinComp - Extract Scalar Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             SmoothMinComp(num_nodes=5, n_comps=1, mode='extract_scalar', units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_min = np.min(test_vector)
    print(f"Input vector: {test_vector}")
    print(f"Smooth min result: {result[0]:.6f}")
    print(f"Exact min: {exact_min:.6f}")
    print(f"Approximation error: {abs(result[0] - exact_min):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMinComp Extract Scalar Vector test passed")


def test_smooth_min_extract_vector_matrix():
    """Test SmoothMinComp in extract_vector mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMinComp - Extract Vector Mode (Matrix)")
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
                             SmoothMinComp(num_nodes=4, n_comps=3, mode='extract_vector', units='V'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_min = np.min(test_matrix, axis=1)  # min across nodes for each component
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Smooth min result (vector): {result}")
    print(f"Exact min (across axis 1 - nodes): {exact_min}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_min)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMinComp Extract Vector Matrix test passed")


def test_smooth_min_extract_matrix():
    """Test SmoothMinComp in extract mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMinComp - Extract Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, -5.0, 3.0],
        [8.0, 1.0, 6.0]
    ])
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             SmoothMinComp(num_nodes=3, n_comps=2, mode='extract_scalar', units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_min = np.min(test_matrix)
    print(f"Input matrix:\n{test_matrix}")
    print(f"Smooth min result: {result[0]:.6f}")
    print(f"Exact min: {exact_min:.6f}")
    print(f"Approximation error: {abs(result[0] - exact_min):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMinComp Extract Matrix test passed")


def test_smooth_min_limit_vector():
    """Test SmoothMinComp in limit mode with vector input."""
    print("\n" + "="*60)
    print("Testing SmoothMinComp - Limit Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    limit_value = 6.0
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             SmoothMinComp(num_nodes=5, n_comps=1, mode='limit', 
                                          limit_val=limit_value, units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.minimum(test_vector, limit_value)
    print(f"Input vector: {test_vector}")
    print(f"Limit value: {limit_value}")
    print(f"Smooth limited result: {result}")
    print(f"Exact limited: {exact_limit}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_limit)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMinComp Limit Vector test passed")


def test_smooth_min_limit_matrix():
    """Test SmoothMinComp in limit mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMinComp - Limit Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, -5.0, 3.0],
        [8.0, 1.0, 6.0]
    ])
    limit_value = 4.0
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             SmoothMinComp(num_nodes=3, n_comps=2, mode='limit', 
                                          limit_val=limit_value, units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.minimum(test_matrix, limit_value)
    print(f"Input matrix:\n{test_matrix}")
    print(f"Limit value: {limit_value}")
    print(f"Smooth limited result:\n{result}")
    print(f"Exact limited:\n{exact_limit}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_limit)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMinComp Limit Matrix test passed")


def test_smooth_max_extract_scalar_vector():
    """Test SmoothMaxComp in extract_scalar mode with vector input."""
    print("\n" + "="*60)
    print("Testing SmoothMaxComp - Extract Scalar Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             SmoothMaxComp(num_nodes=5, n_comps=1, mode='extract_scalar', units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_max = np.max(test_vector)
    print(f"Input vector: {test_vector}")
    print(f"Smooth max result: {result[0]:.6f}")
    print(f"Exact max: {exact_max:.6f}")
    print(f"Approximation error: {abs(result[0] - exact_max):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMaxComp Extract Scalar Vector test passed")


def test_smooth_max_extract_vector_matrix():
    """Test SmoothMaxComp in extract_vector mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMaxComp - Extract Vector Mode (Matrix)")
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
                             SmoothMaxComp(num_nodes=4, n_comps=3, mode='extract_vector', units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_max = np.max(test_matrix, axis=1)  # max across nodes for each component
    print(f"Input matrix (n_comps=3, num_nodes=4):\n{test_matrix}")
    print(f"Smooth max result (vector): {result}")
    print(f"Exact max (across axis 1 - nodes): {exact_max}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_max)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMaxComp Extract Vector Matrix test passed")


def test_smooth_max_extract_matrix():
    """Test SmoothMaxComp in extract mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMaxComp - Extract Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, -5.0, 3.0],
        [8.0, 15.0, 6.0]
    ])
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             SmoothMaxComp(num_nodes=3, n_comps=2, mode='extract_scalar', units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_max = np.max(test_matrix)
    print(f"Input matrix:\n{test_matrix}")
    print(f"Smooth max result: {result[0]:.6f}")
    print(f"Exact max: {exact_max:.6f}")
    print(f"Approximation error: {abs(result[0] - exact_max):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMaxComp Extract Matrix test passed")


def test_smooth_max_limit_vector():
    """Test SmoothMaxComp in limit mode with vector input."""
    print("\n" + "="*60)
    print("Testing SmoothMaxComp - Limit Mode (Vector)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_vector = np.array([10.0, 5.0, 3.0, 7.0, 8.0])
    limit_value = 6.0
    ivc.add_output('input_array', val=test_vector, units='m')
    
    prob.model.add_subsystem('comp', 
                             SmoothMaxComp(num_nodes=5, n_comps=1, mode='limit', 
                                          limit_val=limit_value, units='m'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.maximum(test_vector, limit_value)
    print(f"Input vector: {test_vector}")
    print(f"Limit value: {limit_value}")
    print(f"Smooth limited result: {result}")
    print(f"Exact limited: {exact_limit}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_limit)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMaxComp Limit Vector test passed")


def test_smooth_max_limit_matrix():
    """Test SmoothMaxComp in limit mode with matrix input."""
    print("\n" + "="*60)
    print("Testing SmoothMaxComp - Limit Mode (Matrix)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    test_matrix = np.array([
        [10.0, -5.0, 3.0],
        [8.0, 1.0, 6.0]
    ])
    limit_value = 4.0
    ivc.add_output('input_array', val=test_matrix, units='kg')
    
    prob.model.add_subsystem('comp', 
                             SmoothMaxComp(num_nodes=3, n_comps=2, mode='limit', 
                                          limit_val=limit_value, units='kg'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    result = prob.get_val('output')
    exact_limit = np.maximum(test_matrix, limit_value)
    print(f"Input matrix:\n{test_matrix}")
    print(f"Limit value: {limit_value}")
    print(f"Smooth limited result:\n{result}")
    print(f"Exact limited:\n{exact_limit}")
    print(f"Max approximation error: {np.max(np.abs(result - exact_limit)):.6e}")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ SmoothMaxComp Limit Matrix test passed")


def test_smooth_clamp_chain():
    """Test chaining SmoothMaxComp and SmoothMinComp to clamp values between bounds."""
    print("\n" + "="*60)
    print("Testing Chained SmoothMax → SmoothMin (Clamping)")
    print("="*60)
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # Test vector with values below lower bound, within bounds, and above upper bound
    test_vector = np.array([-5.0, 0.2, 0.5, 0.8, 1.5, 2.0])
    lower_bound = 0.1
    upper_bound = 1.0
    ivc.add_output('raw_input', val=test_vector, units=None)
    
    # Add SmoothMaxComp to enforce lower bound
    prob.model.add_subsystem('lower_limit', 
                             SmoothMaxComp(num_nodes=6, n_comps=1, mode='limit', 
                                          limit_val=lower_bound, units=None),
                             promotes_inputs=[('input_array', 'raw_input')],
                             promotes_outputs=[])
    
    # Add SmoothMinComp to enforce upper bound (connected to lower_limit output)
    prob.model.add_subsystem('upper_limit', 
                             SmoothMinComp(num_nodes=6, n_comps=1, mode='limit', 
                                          limit_val=upper_bound, units=None),
                             promotes_inputs=[],
                             promotes_outputs=[('output', 'clamped_output')])
    
    # Connect the chain: raw_input → lower_limit → upper_limit → clamped_output
    prob.model.connect('lower_limit.output', 'upper_limit.input_array')
    
    prob.setup()
    
    prob.run_model()
    
    raw_input = prob.get_val('raw_input')
    after_lower = prob.get_val('lower_limit.output')
    result = prob.get_val('clamped_output')
    
    # Exact clamping: np.clip(x, lower, upper)
    exact_clamp = np.clip(test_vector, lower_bound, upper_bound)
    
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print(f"\nRaw input:          {raw_input}")
    print(f"After lower limit:  {after_lower}")
    print(f"After upper limit:  {result}")
    print(f"Exact clamp:        {exact_clamp}")
    print(f"\nMax error vs exact: {np.max(np.abs(result - exact_clamp)):.6e}")
    
    # Verify each stage
    print(f"\nVerification:")
    for i, (raw, lower, upper, exact) in enumerate(zip(raw_input, after_lower, result, exact_clamp)):
        status = "✓" if abs(upper - exact) < 1e-3 else "✗"
        print(f"  [{i}] {raw:6.2f} → {lower:6.4f} → {upper:6.4f} (exact: {exact:6.4f}) {status}")
    
    # Check partials
    prob.check_partials(compact_print=True, method='cs')
    print("✓ Chained SmoothMax → SmoothMin clamping test passed")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# RUNNING SMOOTH MIN/MAX COMPONENT TESTS")
    print("#"*60)
    
    # Test SmoothMinComp - all modes
    #test_smooth_min_extract_scalar_vector()
    test_smooth_min_extract_vector_matrix()
    #test_smooth_min_extract_matrix() # extract scalar from a matrix
    #test_smooth_min_limit_vector()
    #test_smooth_min_limit_matrix()
    
    # Test SmoothMaxComp - all modes
    #test_smooth_max_extract_scalar_vector()
    test_smooth_max_extract_vector_matrix()
    #test_smooth_max_extract_matrix() # extract scalar from a matrix
    #test_smooth_max_limit_vector()
    #test_smooth_max_limit_matrix()
    
    # Test chained components
    #test_smooth_clamp_chain()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)