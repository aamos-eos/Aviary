import numpy as np
import openmdao.api as om


class SumAlongAxis(om.ExplicitComponent):
    """
    Generic component that sums along the num_comps axis to create a vector along num_nodes
    Input: input_data with shape (num_comps, num_nodes)
    Output: output_data with shape (num_nodes,)

    Parameters
    ----------
    num_nodes : int
        Number of analysis points (fixed)
    num_comps : int
        Number of cases to sum over (e.g., num_props, num_motors, etc.)
    input_name : str
        Name of the input variable
    output_name : str
        Name of the output variable
    input_units : str
        Units of the input variable
    output_units : str
        Units of the output variable
    input_desc : str
        Description of the input variable
    output_desc : str
        Description of the output variable
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_comps', default=4, desc='Number of cases to sum over')
        self.options.declare('input_name', default='input_data', desc='Name of input variable')
        self.options.declare('output_name', default='output_data', desc='Name of output variable')
        self.options.declare('input_units', default=None, desc='Units of input variable')
        self.options.declare('output_units', default=None, desc='Units of output variable')
        self.options.declare('input_desc', default='Input data', desc='Description of input variable')
        self.options.declare('output_desc', default='Output data', desc='Description of output variable')

    def setup(self):
        nn = self.options['num_nodes']
        nc = self.options['num_comps']
        input_name = self.options['input_name']
        output_name = self.options['output_name']
        input_units = self.options['input_units']
        output_units = self.options['output_units']
        input_desc = self.options['input_desc']
        output_desc = self.options['output_desc']

        # Input: data from all cases (num_comps, num_nodes)
        self.add_input(input_name, shape=(nc, nn), units=input_units, desc=input_desc)
        # Output: data summed over all cases (num_nodes,)
        self.add_output(output_name, shape=(nn,), units=output_units, desc=output_desc)

        # Declare partials
        # output[j] depends on input[i, j] for all i in range(nc)
        # Input is flattened in row-major order: input[i, j] -> flat index i * nn + j
        # Output is flattened: output[j] -> flat index j
        rows = np.tile(np.arange(nn), nc)  # [0,1,2,...,nn-1, 0,1,2,...,nn-1, ...] repeated nc times
        cols = np.arange(nc * nn)  # [0,1,2,...,nc*nn-1]
        self.declare_partials(output_name, input_name, rows=rows, cols=cols, val=1.0)

    def compute(self, inputs, outputs):
        input_name = self.options['input_name']
        output_name = self.options['output_name']
        input_data = inputs[input_name]

        # Sum along the num_comps axis (axis=0) to get total at each time point
        outputs[output_name] = np.sum(input_data, axis=0)

    def compute_partials(self, inputs, partials):
        # Partials are constant (all 1.0) and already declared in setup
        # No need to recompute them
        pass

if __name__ == "__main__":
    # Test the SumAlongAxis component
    nn = 10
    nc = 4
    
    # Create a problem to test the component
    prob = om.Problem(reports=False)
    
    # Add an IndepVarComp for the input
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
    ivc.add_output('input_data', val=np.random.rand(nc, nn), units='kW')
    
    # Add the SumAlongAxis component
    prob.model.add_subsystem('sum_comp', 
                             SumAlongAxis(num_nodes=nn, 
                                         num_comps=nc,
                                         input_name='input_data',
                                         output_name='output_data',
                                         input_units='kW',
                                         output_units='kW',
                                         input_desc='Power from each component',
                                         output_desc='Total power'),
                             promotes_inputs=['*'],
                             promotes_outputs=['*'])
    
    # Setup and run
    prob.setup(force_alloc_complex=True)
    prob.run_model()


    test_count = 5
    
    # Print results
    print("\n" + "="*60)
    print("SumAlongAxis Component Test")
    print("="*60)
    print(f"\nInput shape: {prob.get_val('input_data').shape}")
    print(f"Output shape: {prob.get_val('output_data').shape}")
    print(f"\nInput data, up to index {test_count-1}:")
    print(prob.get_val('input_data')[:, :test_count])
    print(f"\nOutput data, up to index {test_count-1}:")
    print(prob.get_val('output_data')[:test_count])
    print(f"\nExpected sum (manual calculation up to index{test_count-1}):")
    print(np.sum(prob.get_val('input_data')[:, :test_count], axis=0))
    
    # Check partials
    print("\n" + "="*60)
    print("Checking Partials")
    print("="*60)
    partials = prob.check_partials(compact_print=True, show_only_incorrect=True)
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
