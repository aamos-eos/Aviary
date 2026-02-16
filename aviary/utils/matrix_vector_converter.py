import numpy as np
import openmdao.api as om


class MatrixToVectorConverter(om.ExplicitComponent):
    """
    Component that converts matrix inputs to vector outputs for use with MetaModel components.
    Flattens (num_comps, num_nodes) matrices into vectors of size (num_comps * num_nodes,).
    
    This is particularly useful when interfacing with OpenMDAO MetaModel components that expect
    vectorized inputs rather than matrix inputs.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes to evaluate
    num_comps : int
        Number of props/motors
    input_names : list
        List of input variable names to convert
    output_names : list, optional
        List of output variable names (defaults to input_names)
    units : dict, optional
        Dictionary of units for each variable
    input_default : list, optional
        List of default values for input variables (defaults to 1.0 for each if None)
    output_default : list, optional
        List of default values for output variables (defaults to 1.0 for each if None)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_comps', default=4, desc='number of props')
        self.options.declare('input_names', default=['rpm', 'torque'], desc='list of input variable names to convert')
        self.options.declare('output_names', default=None, desc='list of output variable names (defaults to input_names with _vect suffix)')
        self.options.declare('units', default=None, desc='dictionary of units for each variable')
        self.options.declare('input_default', default=None, desc='list of default values for inputs (defaults to 1.0 for each if None)')
        self.options.declare('output_default', default=None, desc='list of default values for outputs (defaults to 1.0 for each if None)')
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        num_comps = self.options['num_comps']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_vect" for name in input_names]
        units = self.options['units'] or {}
        input_defaults = self.options['input_default']
        output_defaults = self.options['output_default']
        
        # Add matrix inputs
        for i, name in enumerate(input_names):
            unit = units.get(name, None)
            # Get default value for this input: use list element if available, otherwise 1.0
            if input_defaults is None:
                input_val = 1.0
            elif isinstance(input_defaults, list) and i < len(input_defaults):
                input_val = float(input_defaults[i]) if input_defaults[i] is not None else 1.0
            else:
                input_val = 1.0
            
            self.add_input(name, shape=(num_comps, num_nodes), units=unit, 
                          desc=f'Matrix input for {name}', val=input_val)
        
        # Add vector outputs
        for i, (input_name, output_name) in enumerate(zip(input_names, output_names)):
            unit = units.get(input_name, None)  # Get units from input name, not output name
            # Get default value for this output: use list element if available, otherwise 1.0
            if output_defaults is None:
                output_val = 1.0
            elif isinstance(output_defaults, list) and i < len(output_defaults):
                output_val = float(output_defaults[i]) if output_defaults[i] is not None else 1.0
            else:
                output_val = 1.0
            
            self.add_output(output_name, shape=(num_comps * num_nodes,), units=unit,
                           desc=f'Vector output for {output_name}', lower=1e-6, val=output_val)
        
        # Set up partial derivatives
        for i, (input_name, output_name) in enumerate(zip(input_names, output_names)):
            # Each output element depends only on the corresponding input element
            rows = np.arange(num_comps * num_nodes)
            cols = np.arange(num_comps * num_nodes)
            vals = np.ones(num_comps * num_nodes)
            
            self.declare_partials(
                of=output_name,
                wrt=input_name,
                val=vals,
                rows=rows,
                cols=cols,
                method='exact'
            )
    
    def compute(self, inputs, outputs):
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        units = self.options['units']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_vect" for name in input_names]
        
        for input_name, output_name in zip(input_names, output_names):
            # Flatten the matrix input to vector output

            matrix_data = inputs[input_name]
            #print(f"Units: {units[input_name]}")
            #print(f"Matrix data: {matrix_data}")

            outputs[output_name] = matrix_data.flatten()
            #print(f"Matrix Flattened: {matrix_data.flatten()}")

        
    
    
    def compute_partials(self, inputs, partials):
        # Partial derivatives are constant (1.0) - each output element depends on exactly one input element
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_vect" for name in input_names]
        
        for input_name, output_name in zip(input_names, output_names):
            # The Jacobian is an identity matrix when viewing flattened indices
            # Each output[i] = input[i], so d(output[i])/d(input[j]) = 1 if i==j, else 0
            # But we declared this sparsely, so we just need to set the values
            partials[output_name, input_name] = np.ones(num_comps * num_nodes)


class VectorToMatrixConverter(om.ExplicitComponent):
    """
    Component that converts vector inputs to matrix outputs.
    Reshapes vectors of size (num_comps * num_nodes,) into (num_comps, num_nodes) matrices.
    
    This is particularly useful when converting outputs from MetaModel components back
    to matrix format for downstream components that expect matrix inputs.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes to evaluate
    num_comps : int
        Number of props/motors
    input_names : list
        List of input variable names to convert
    output_names : list, optional
        List of output variable names (defaults to input_names)
    units : dict, optional
        Dictionary of units for each variable
    input_default : list, optional
        List of default values for input variables (defaults to 1.0 for each if None)
    output_default : list, optional
        List of default values for output variables (defaults to 1.0 for each if None)
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('num_comps', default=4, desc='number of props')
        self.options.declare('input_names', default=['eff'], desc='list of input variable names to convert')
        self.options.declare('output_names', default=None, desc='list of output variable names (defaults to input_names with _mat suffix)')
        self.options.declare('units', default=None, desc='dictionary of units for each variable')
        self.options.declare('input_default', default=None, desc='list of default values for inputs (defaults to 1.0 for each if None)')
        self.options.declare('output_default', default=None, desc='list of default values for outputs (defaults to 1.0 for each if None)')
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        num_comps = self.options['num_comps']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_mat" for name in input_names]
        units = self.options['units'] or {}
        input_defaults = self.options['input_default']
        output_defaults = self.options['output_default']
        
        # Add vector inputs
        for i, name in enumerate(input_names):
            unit = units.get(name, None)
            # Get default value for this input: use list element if available, otherwise 1.0
            if input_defaults is None:
                input_val = 1.0
            elif isinstance(input_defaults, list) and i < len(input_defaults):
                input_val = float(input_defaults[i]) if input_defaults[i] is not None else 1.0
            else:
                input_val = 1.0
            
            self.add_input(name, shape=(num_comps * num_nodes,), units=unit,
                          desc=f'Vector input for {name}', val=input_val)
        
        # Add matrix outputs
        for i, (input_name, output_name) in enumerate(zip(input_names, output_names)):
            unit = units.get(input_name, None)  # Get units from input name, not output name
            # Get default value for this output: use list element if available, otherwise 1.0
            if output_defaults is None:
                output_val = 1.0
            elif isinstance(output_defaults, list) and i < len(output_defaults):
                output_val = float(output_defaults[i]) if output_defaults[i] is not None else 1.0
            else:
                output_val = 1.0
            
            self.add_output(output_name, shape=(num_comps, num_nodes), units=unit,
                           desc=f'Matrix output for {output_name}', val=output_val)
        
        # Set up partial derivatives
        for i, (input_name, output_name) in enumerate(zip(input_names, output_names)):
            # Each output element depends only on the corresponding input element
            rows = np.arange(num_comps * num_nodes)
            cols = np.arange(num_comps * num_nodes)
            vals = np.ones(num_comps * num_nodes)
            
            self.declare_partials(
                of=output_name,
                wrt=input_name,
                val=vals,
                rows=rows,
                cols=cols,
                method='exact'
            )
    
    def compute(self, inputs, outputs):
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_mat" for name in input_names]
        
        for input_name, output_name in zip(input_names, output_names):
            # Reshape the vector input to matrix output
            vector_data = inputs[input_name]
            outputs[output_name] = vector_data.reshape((num_comps, num_nodes))
    
    def compute_partials(self, inputs, partials):
        # Partial derivatives are constant (1.0) - each output element depends on exactly one input element
        num_comps = self.options['num_comps']
        num_nodes = self.options['num_nodes']
        input_names = self.options['input_names']
        output_names = self.options['output_names'] or [f"{name}_mat" for name in input_names]
        
        for input_name, output_name in zip(input_names, output_names):
            # The Jacobian is an identity matrix when viewing flattened indices
            # Each output[i] = input[i], so d(output[i])/d(input[j]) = 1 if i==j, else 0
            # But we declared this sparsely, so we just need to set the values
            partials[output_name, input_name] = np.ones(num_comps * num_nodes)


if __name__ == "__main__":
    """
    Test function for MatrixToVectorConverter and VectorToMatrixConverter.
    Tests both components and runs check_partials to verify derivatives.
    """
    print("=" * 80)
    print("Testing MatrixToVectorConverter")
    print("=" * 80)
    
    # Test MatrixToVectorConverter
    num_nodes = 11
    num_comps = 4
    prob = om.Problem(reports=False)
    
    # Create test data
    ivc = om.IndepVarComp()
    test_matrix_rpm = np.random.rand(num_comps, num_nodes) * 2000 + 1000  # RPM between 1000-3000
    test_matrix_torque = np.random.rand(num_comps, num_nodes) * 100 + 50   # Torque between 50-150
    
    ivc.add_output('rpm', val=test_matrix_rpm, units='rpm')
    ivc.add_output('torque', val=test_matrix_torque, units='N*m')
    
    prob.model.add_subsystem('ivc', ivc)
    
    mat_to_vec = MatrixToVectorConverter(
        num_nodes=num_nodes,
        num_comps=num_comps,
        input_names=['rpm', 'torque'],
        units={'rpm': 'rpm', 'torque': 'N*m'}
    )
    prob.model.add_subsystem('mat_to_vec', mat_to_vec)
    
    prob.model.connect('ivc.rpm', 'mat_to_vec.rpm')
    prob.model.connect('ivc.torque', 'mat_to_vec.torque')
    
    prob.setup()
    prob.run_model()
    
    # Check outputs
    rpm_vect = prob.get_val('mat_to_vec.rpm_vect')
    torque_vect = prob.get_val('mat_to_vec.torque_vect')
    
    expected_rpm_vect = test_matrix_rpm.flatten()
    expected_torque_vect = test_matrix_torque.flatten()
    
    print(f"\nMatrix shape: {test_matrix_rpm.shape}")
    print(f"Vector shape: {rpm_vect.shape}")
    print(f"Expected vector shape: ({num_comps * num_nodes},)")
    
    if np.allclose(rpm_vect, expected_rpm_vect):
        print("✓ RPM conversion correct")
    else:
        print("✗ RPM conversion FAILED")
        print(f"  Max difference: {np.max(np.abs(rpm_vect - expected_rpm_vect))}")
    
    if np.allclose(torque_vect, expected_torque_vect):
        print("✓ Torque conversion correct")
    else:
        print("✗ Torque conversion FAILED")
        print(f"  Max difference: {np.max(np.abs(torque_vect - expected_torque_vect))}")
    
    # Check partials
    print("\nChecking partial derivatives for MatrixToVectorConverter...")
    prob.check_partials(compact_print=True)
    
    print("\n" + "=" * 80)
    print("Testing VectorToMatrixConverter")
    print("=" * 80)
    
    # Test VectorToMatrixConverter
    prob2 = om.Problem(reports=False)
    
    ivc2 = om.IndepVarComp()
    test_vector_eff = np.random.rand(num_comps * num_nodes) * 0.3 + 0.7  # Efficiency between 0.7-1.0
    
    ivc2.add_output('eff_vect', val=test_vector_eff, units=None)
    
    prob2.model.add_subsystem('ivc2', ivc2)
    
    vec_to_mat = VectorToMatrixConverter(
        num_nodes=num_nodes,
        num_comps=num_comps,
        input_names=['eff_vect'],
        output_names=['eff'],
        units={'eff_vect': None}
    )
    prob2.model.add_subsystem('vec_to_mat', vec_to_mat)
    
    prob2.model.connect('ivc2.eff_vect', 'vec_to_mat.eff_vect')
    
    prob2.setup()
    prob2.run_model()
    
    # Check outputs
    eff_mat = prob2.get_val('vec_to_mat.eff')
    expected_eff_mat = test_vector_eff.reshape((num_comps, num_nodes))
    
    print(f"\nVector shape: {test_vector_eff.shape}")
    print(f"Matrix shape: {eff_mat.shape}")
    print(f"Expected matrix shape: ({num_comps}, {num_nodes})")
    
    if np.allclose(eff_mat, expected_eff_mat):
        print("✓ Efficiency conversion correct")
    else:
        print("✗ Efficiency conversion FAILED")
        print(f"  Max difference: {np.max(np.abs(eff_mat - expected_eff_mat))}")
    
    # Check partials
    print("\nChecking partial derivatives for VectorToMatrixConverter...")
    prob2.check_partials(compact_print=True)
    
    print("\n" + "=" * 80)
    print("Testing Round-Trip Conversion")
    print("=" * 80)
    
    # Test round-trip: matrix -> vector -> matrix
    prob3 = om.Problem(reports=False)
    
    ivc3 = om.IndepVarComp()
    original_matrix = np.random.rand(num_comps, num_nodes) * 0.5 + 0.5  # Values between 0.5-1.0
    ivc3.add_output('original', val=original_matrix, units=None)
    
    prob3.model.add_subsystem('ivc3', ivc3)
    
    mat_to_vec3 = MatrixToVectorConverter(
        num_nodes=num_nodes,
        num_comps=num_comps,
        input_names=['original'],
        units={'original': None}
    )
    prob3.model.add_subsystem('mat_to_vec3', mat_to_vec3)
    
    vec_to_mat3 = VectorToMatrixConverter(
        num_nodes=num_nodes,
        num_comps=num_comps,
        input_names=['original_vect'],
        output_names=['reconstructed'],
        units={'original_vect': None}
    )
    prob3.model.add_subsystem('vec_to_mat3', vec_to_mat3)
    
    prob3.model.connect('ivc3.original', 'mat_to_vec3.original')
    prob3.model.connect('mat_to_vec3.original_vect', 'vec_to_mat3.original_vect')
    
    prob3.setup()
    prob3.run_model()
    
    reconstructed = prob3.get_val('vec_to_mat3.reconstructed')
    
    if np.allclose(original_matrix, reconstructed):
        print("✓ Round-trip conversion correct")
        print(f"  Max difference: {np.max(np.abs(original_matrix - reconstructed))}")
    else:
        print("✗ Round-trip conversion FAILED")
        print(f"  Max difference: {np.max(np.abs(original_matrix - reconstructed))}")
        print(f"  Original shape: {original_matrix.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
    
    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)