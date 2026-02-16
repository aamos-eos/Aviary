import openmdao.api as om
import numpy as np


class Tiler(om.ExplicitComponent):
    """
    A component that tiles multiple vectors to create either:
    1. Matrices with shape (n_comps, num_nodes) - tiling along the first axis
    2. Longer vectors with shape (num_nodes * n_comps) - flattening the tiled result
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input vectors
    n_comps : int
        Number of components to tile to
    tile_option : str
        Either 'matrix' or 'vector' - determines output shape
    input_names : list
        List of input variable names
    output_names : list
        List of output variable names
    input_units : dict
        Dictionary mapping input names to units
    output_units : dict
        Dictionary mapping output names to units
    input_desc : dict
        Dictionary mapping input names to descriptions
    output_desc : dict
        Dictionary mapping output names to descriptions
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of nodes in input vectors')
        self.options.declare('n_comps', default=4, desc='Number of components to tile to')
        self.options.declare('tile_option', default='matrix', desc='Tile option: "matrix" or "vector"')
        self.options.declare('input_names', default=['input_vector'], desc='List of input variable names')
        self.options.declare('output_names', default=['output_tiled'], desc='List of output variable names')
        self.options.declare('input_units', default=None, desc='Dictionary mapping input names to units')
        self.options.declare('output_units', default=None, desc='Dictionary mapping output names to units')
        self.options.declare('input_desc', default=None, desc='Dictionary mapping input names to descriptions')
        self.options.declare('output_desc', default=None, desc='Dictionary mapping output names to descriptions')
    
    def setup(self):
        nn = self.options['num_nodes']
        nc = self.options['n_comps']
        tile_option = self.options['tile_option']
        input_names = self.options['input_names']
        output_names = self.options['output_names']
        input_units = self.options['input_units'] or {}
        output_units = self.options['output_units'] or {}
        input_desc = self.options['input_desc'] or {}
        output_desc = self.options['output_desc'] or {}
        
        # Add inputs and outputs for each variable
        for i, (input_name, output_name) in enumerate(zip(input_names, output_names)):
            # Input: vector of length num_nodes
            self.add_input(input_name, shape=(nn,), 
                          units=input_units.get(input_name), 
                          desc=input_desc.get(input_name, f'Input vector {i+1}'))
            
            # Output: depends on tile option
            if tile_option == 'matrix':
                # Output: matrix of shape (n_comps, num_nodes)
                self.add_output(output_name, shape=(nc, nn), 
                               units=output_units.get(output_name), 
                               desc=output_desc.get(output_name, f'Tiled output {i+1}'))
            elif tile_option == 'vector':
                # Output: vector of length num_nodes * n_comps
                self.add_output(output_name, shape=(nn * nc,), 
                               units=output_units.get(output_name), 
                               desc=output_desc.get(output_name, f'Tiled output {i+1}'))
            else:
                raise ValueError(f"tile_option must be 'matrix' or 'vector', got '{tile_option}'")
            
            # Declare partial derivatives
            if tile_option == 'matrix':
                # Each output element (i,j) depends only on input element j
                rows = np.arange(nc * nn)
                cols = np.tile(np.arange(nn), nc)
                self.declare_partials(output_name, input_name, rows=rows, cols=cols)
            else:  # vector format
                # Each output element i depends on input element (i % nn)
                rows = np.arange(nn * nc)
                cols = np.arange(nn * nc) % nn
                self.declare_partials(output_name, input_name, rows=rows, cols=cols)
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nc = self.options['n_comps']
        tile_option = self.options['tile_option']
        input_names = self.options['input_names']
        output_names = self.options['output_names']
        
        for input_name, output_name in zip(input_names, output_names):
            input_vector = inputs[input_name]
            
            if tile_option == 'matrix':
                # Tile the vector to create a matrix (n_comps, num_nodes)
                # np.tile(x, (n_comps, 1))
                outputs[output_name] = np.tile(input_vector, (nc, 1))
            else:  # vector format
                # Tile the vector to create a longer vector (num_nodes * n_comps)
                # np.tile(x, (1, n_comps)) then flatten
                outputs[output_name] = np.tile(input_vector, (1, nc)).flatten()
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        nc = self.options['n_comps']
        tile_option = self.options['tile_option']
        input_names = self.options['input_names']
        output_names = self.options['output_names']
        
        for input_name, output_name in zip(input_names, output_names):
            if tile_option == 'matrix':
                # Each output element (i,j) has partial derivative 1.0 with respect to input element j
                partials[output_name, input_name] = np.ones(nc * nn)
            else:  # vector format
                # Each output element i has partial derivative 1.0 with respect to input element (i % nn)
                partials[output_name, input_name] = np.ones(nn * nc)
