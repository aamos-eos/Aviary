import numpy as np
import openmdao.api as om


class ChainVars(om.ExplicitComponent):
    """
    Concatenate vectors or matrices along the time axis.
    
    Inputs
    ------
    Inputs are automatically generated as {phase}_{variable} for each phase.
    
    Outputs
    -------
    Outputs are specified via vars_list during initialization.
    
    Options
    -------
    phases : list of str
        List of phase names (e.g., ['climb', 'cruise', 'descent'])
    
    vars_list : list of tuples
        Each entry is (variable_name, output_name, units, nc)
        - variable_name: base name of the variable (e.g., 't', 'p_train_elec')
        - output_name: name of the output variable (e.g., 'mission_time')
        - units: units string (or None)
        - nc: number of cases/motors (first dimension). Use 1 for vectors.
    
    nn : int
        Number of nodes per phase
    """
    
    def initialize(self):
        self.options.declare('phases', default=None, desc='List of phase names')
        self.options.declare('nn', default=None, desc='Number of nodes per phase')
        self.options.declare('vars_list', default=None, desc='List of (variable_name, output_name, units, nc)')

    def setup(self):
        phases = self.options['phases']
        nn = self.options['nn']
        vars_list = self.options['vars_list']
        nv = len(phases)
        nnc = nn * nv  # total nodes across all phases
        
        for variable_name, output_name, units, nc in vars_list:
            # Generate input names: {phase}_{variable} for each phase
            input_names = [f"{phase}_{variable_name}" for phase in phases]
            
            # Add inputs - shape determined by connections
            for input_name in input_names:
                if units is None:
                    self.add_input(input_name, shape_by_conn=True)
                else:
                    self.add_input(input_name, units=units, shape_by_conn=True)
            
            # Output shape: (nc, nn * n_phases) or (nn * n_phases,) if nc=1
            if nc == 1:
                output_shape = (nnc,)
            else:
                output_shape = (nc, nnc)
            
            # Add output with explicit shape
            if units is None:
                self.add_output(output_name, shape=output_shape)
            else:
                self.add_output(output_name, shape=output_shape, units=units)
            
            # Declare partials - analytical derivatives for concatenation
            # For each input phase, we need to set up the sparsity pattern
            for i, input_name in enumerate(input_names):
                # Determine input shape
                if nc == 1:
                    # 1D input: shape (nn,)
                    # Output: shape (nnc,) = (nn * nv,)
                    # Partial: identity matrix for elements [i*nn : (i+1)*nn]
                    rows = np.arange(i * nn, (i + 1) * nn, dtype=int)
                    cols = np.arange(nn, dtype=int)
                    self.declare_partials(
                        of=output_name,
                        wrt=input_name,
                        rows=rows,
                        cols=cols
                    )
                else:
                    # 2D input: shape (nc, nn)
                    # Output: shape (nc, nnc) = (nc, nn * nv)
                    # Partial: block identity matrix for columns [i*nn : (i+1)*nn]
                    # For each output row r, columns [i*nn : (i+1)*nn] map to input row r, columns [0:nn]
                    rows_list = []
                    cols_list = []
                    for r in range(nc):
                        for c in range(nn):
                            # Output row r, column (i*nn + c)
                            rows_list.append(r * nnc + i * nn + c)
                            # Input row r, column c
                            cols_list.append(r * nn + c)
                    self.declare_partials(
                        of=output_name,
                        wrt=input_name,
                        rows=np.array(rows_list, dtype=int),
                        cols=np.array(cols_list, dtype=int)
                    )

    def compute(self, inputs, outputs):
        phases = self.options['phases']
        vars_list = self.options['vars_list']
        
        for variable_name, output_name, units, nc in vars_list:
            # Generate input names: {phase}_{variable} for each phase
            input_names = [f"{phase}_{variable_name}" for phase in phases]
            
            # Collect all inputs and reshape to (nc, nn) if needed
            input_arrays = []
            for input_name in input_names:
                data = inputs[input_name]
                
                # If 1D, reshape to (1, nn); if 2D, use as-is
                if len(data.shape) == 1:
                    input_arrays.append(data[np.newaxis, :])  # (1, nn)
                else:
                    input_arrays.append(data)  # (nc, nn)
            
            # Concatenate along axis 1 (time axis): (nc, nn * n_phases)
            concatenated = np.concatenate(input_arrays, axis=1)
            
            # Flatten if nc=1
            if nc == 1:
                outputs[output_name] = concatenated.flatten()
            else:
                outputs[output_name] = concatenated

    def compute_partials(self, inputs, partials):
        """
        Compute analytical partial derivatives for concatenation.
        
        For each input phase, the partial derivative is an identity matrix
        for the corresponding slice of the output.
        """
        phases = self.options['phases']
        vars_list = self.options['vars_list']
        nn = self.options['nn']
        
        for variable_name, output_name, units, nc in vars_list:
            input_names = [f"{phase}_{variable_name}" for phase in phases]
            
            for i, input_name in enumerate(input_names):
                if nc == 1:
                    # 1D case: identity matrix for slice [i*nn : (i+1)*nn]
                    # All derivatives are 1.0
                    partials[output_name, input_name] = np.ones(nn)
                else:
                    # 2D case: block identity matrix
                    # For each output row r, columns [i*nn : (i+1)*nn] map 1-to-1 to input row r, columns [0:nn]
                    # All derivatives are 1.0
                    partials[output_name, input_name] = np.ones(nc * nn)


if __name__ == "__main__":
    """Test the ChainVars component"""
    
    nn = 3  # nodes per phase
    phases = ['climb', 'cruise', 'descent']
    nc_time = 1  # time is 1D
    nc_power = 4  # power has 4 motors
    
    # Test data
    climb_t = np.array([0.0, 1.0, 2.0])
    cruise_t = np.array([2.0, 3.0, 4.0])
    descent_t = np.array([4.0, 5.0, 6.0])
    
    climb_p = np.array([
        [100.0, 200.0, 300.0],
        [110.0, 210.0, 310.0],
        [120.0, 220.0, 320.0],
        [130.0, 230.0, 330.0],
    ])
    
    cruise_p = np.array([
        [400.0, 500.0, 600.0],
        [410.0, 510.0, 610.0],
        [420.0, 520.0, 620.0],
        [430.0, 530.0, 630.0],
    ])
    
    descent_p = np.array([
        [700.0, 800.0, 900.0],
        [710.0, 810.0, 910.0],
        [720.0, 820.0, 920.0],
        [730.0, 830.0, 930.0],
    ])
    
    # Define variables to concatenate: (variable_name, output_name, units, nc)
    vars_list = [
        ('t', 'mission_time', 's', nc_time),
        ('p', 'mission_p_train_elec', 'kW', nc_power),
    ]
    
    # Create problem
    prob = om.Problem(reports=False)
    prob.model = om.Group()
    
    # Create inputs (names match {phase}_{variable} pattern)
    ivc = om.IndepVarComp()
    ivc.add_output('climb_t', val=climb_t, units='s')
    ivc.add_output('cruise_t', val=cruise_t, units='s')
    ivc.add_output('descent_t', val=descent_t, units='s')
    ivc.add_output('climb_p', val=climb_p, units='kW')
    ivc.add_output('cruise_p', val=cruise_p, units='kW')
    ivc.add_output('descent_p', val=descent_p, units='kW')
    
    prob.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])
    prob.model.add_subsystem('chain_vars', ChainVars(phases=phases, vars_list=vars_list, nn=nn), 
                             promotes_inputs=['*'], promotes_outputs=['*'])
    
    prob.setup()
    prob.run_model()
    
    # Get results
    result_t = prob['mission_time']
    result_p = prob['mission_p_train_elec']
    
    # Expected
    expected_t = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    expected_p = np.concatenate([climb_p, cruise_p, descent_p], axis=1)
    
    print("Time test:", np.allclose(result_t, expected_t))
    print("Power test:", np.allclose(result_p, expected_p))
    print("Result t:", result_t)
    print("Expected t:", expected_t)
    print("Result p:", result_p)
    print("Expected p:", expected_p)
    print(f"Time shape: {result_t.shape} (expected: {expected_t.shape})")
    print(f"Power shape: {result_p.shape} (expected: {expected_p.shape})")
    
    # Check partial derivatives - verify analytical derivatives match finite difference
    print("\n" + "="*60)
    print("Checking partial derivatives (analytical vs finite difference)")
    print("="*60)
    partials_data = prob.check_partials(
        method='fd',
        compact_print=False,
        show_only_incorrect=True
    )
    
