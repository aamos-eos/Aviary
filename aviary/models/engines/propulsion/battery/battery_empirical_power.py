import numpy as np
import openmdao.api as om

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from aviary.utils.extract_last import ExtractLast
from aviary.utils.smooth_minmax import SmoothMaxComp, SmoothMinComp
from .cross_feed_logic import DemandFeed

import openmdao.api as om
from aviary.utils.math.integrals import Integrator
from .battery_mm_interpolation_group import BatteryMMInterpolationGroup
from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
from .battery_data import BatteryData


import openmdao.api as om
import jax

class BatteryCurrentVoltage(om.ExplicitComponent):
    """
    Explicit component for calculating the battery voltage, current, and power.
    Handles matrices with shape (n_str, nn) where each row is a battery string.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('n_str', default=1, desc='number of battery strings')

    def setup(self):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']

        self.add_input('vline_cell', shape=(n_str, nn), units='V', desc='line voltage of the cell matrix (one row per string)')
        self.add_input('i_cell', shape=(n_str, nn), units='A', desc='cell current matrix (one row per string)')
        self.add_input('n_series_per_str', shape=(1,), units=None, desc='number of cells in series per string')
        self.add_input('n_parallel_per_str', shape=(1,), units=None, desc='number of cells in parallel per string')
        self.add_input('eta_parc', shape=(nn,), units=None, desc='power architecture electrical efficiency')

        self.add_output('v_bat', shape=(n_str, nn), units='V', desc='Battery voltage matrix', upper = 850, val = 800)
        self.add_output('v_mot', shape=(n_str, nn), units='V', desc='Motor voltage matrix', upper = 1200, lower = -100, val = 800)
        self.add_output('i_bat', shape=(n_str, nn), units='A', desc='Battery current matrix', upper = 500 * 4 * 27, lower = -25 * 4 * 27)
        
        # Define sparsity patterns for partials
        total_elements = n_str * nn
        
        # v_bat = vline_cell * n_series_per_str
        # v_bat wrt vline_cell: diagonal (element-wise multiplication)
        diag_indices = np.arange(total_elements)
        self.declare_partials('v_bat', 'vline_cell', rows=diag_indices, cols=diag_indices)
        # v_bat wrt n_series_per_str: all elements depend on this scalar
        self.declare_partials('v_bat', 'n_series_per_str', rows=diag_indices, cols=np.zeros(total_elements, dtype=int))
        
        # v_mot = v_bat * eta_parc = vline_cell * n_series_per_str * eta_parc
        # v_mot wrt vline_cell: diagonal (element-wise)
        self.declare_partials('v_mot', 'vline_cell', rows=diag_indices, cols=diag_indices)
        # v_mot wrt n_series_per_str: all elements depend on this scalar
        self.declare_partials('v_mot', 'n_series_per_str', rows=diag_indices, cols=np.zeros(total_elements, dtype=int))
        # v_mot wrt eta_parc: v_mot[i,j] depends on eta_parc[j]
        rows_eta = np.arange(total_elements)
        cols_eta = np.tile(np.arange(nn), n_str)  # Each string repeats the pattern 0,1,2,...,nn-1
        self.declare_partials('v_mot', 'eta_parc', rows=rows_eta, cols=cols_eta)
        
        # i_bat = i_cell * n_parallel_per_str
        # i_bat wrt i_cell: diagonal (element-wise multiplication)
        self.declare_partials('i_bat', 'i_cell', rows=diag_indices, cols=diag_indices)
        # i_bat wrt n_parallel_per_str: all elements depend on this scalar
        self.declare_partials('i_bat', 'n_parallel_per_str', rows=diag_indices, cols=np.zeros(total_elements, dtype=int))

    def compute(self, inputs, outputs):
        vline_cell = inputs['vline_cell']
        i_cell = inputs['i_cell']
        n_series_per_str = inputs['n_series_per_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        eta_parc = inputs['eta_parc']
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        
        # Calculate battery voltage: v_bat = vline_cell * n_series_per_str (broadcasted)
        v_bat = vline_cell * n_series_per_str
        
        # Calculate motor voltage: v_mot = v_bat * eta_parc (broadcast eta_parc to each string)
        v_mot = v_bat * eta_parc
        
        # Calculate battery current: i_bat = i_cell * n_parallel_per_str (per string, no n_str multiplier)
        i_bat = i_cell * n_parallel_per_str

        outputs['v_bat'] = v_bat
        outputs['v_mot'] = v_mot
        outputs['i_bat'] = i_bat
    
    def compute_partials(self, inputs, partials):
        n_series_per_str = inputs['n_series_per_str'][0]
        n_parallel_per_str = inputs['n_parallel_per_str'][0]
        eta_parc = inputs['eta_parc']
        i_cell = inputs['i_cell']
        vline_cell = inputs['vline_cell']

        nn = self.options['num_nodes']
        n_str = self.options['n_str']

        total_elements = n_str * nn
        
        # v_bat = vline_cell * n_series_per_str
        # dv_bat/dvline_cell = n_series_per_str (diagonal)
        partials['v_bat', 'vline_cell'] = np.ones(total_elements) * n_series_per_str
        # dv_bat/dn_series_per_str = vline_cell (flattened)
        partials['v_bat', 'n_series_per_str'] = vline_cell.flatten()
        
        # v_mot = vline_cell * n_series_per_str * eta_parc
        # dv_mot/dvline_cell = n_series_per_str * eta_parc (diagonal, but eta_parc is per node)
        # Fix partial to match element-wise multiplication shape for (n_str, nn)
        partials['v_mot', 'vline_cell'] = np.tile(eta_parc * n_series_per_str, (1, n_str))
        # dv_mot/dn_series_per_str = vline_cell * eta_parc
        partials['v_mot', 'n_series_per_str'] = (vline_cell * eta_parc).flatten()
        # dv_mot/deta_parc = vline_cell * n_series_per_str (grouped by node)
        partials['v_mot', 'eta_parc'] = (vline_cell * n_series_per_str).flatten()
        
        # i_bat = i_cell * n_parallel_per_str
        # di_bat/di_cell = n_parallel_per_str (diagonal)
        partials['i_bat', 'i_cell'] = np.ones(total_elements) * n_parallel_per_str
        # di_bat/dn_parallel_per_str = i_cell (flattened)
        partials['i_bat', 'n_parallel_per_str'] = i_cell.flatten()

    #def self_get_statics(self):
    #    return ( self.options['eta_parc'], )




    
class BatteryPower(om.ExplicitComponent):
    """
    Explicit component for calculating the battery voltage, current, and power.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('n_str', default=1, desc='number of battery strings')
    
    def setup(self):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        
        self.add_input('p_aux_elec', shape=(nn,), units='W', desc='electrical power of the auxiliary system')
        self.add_input('p_train_elec_req', shape=(n_str, nn), units='W', desc='electrical power requirement per battery string')
        self.add_input('n_series_per_str', shape=(1,), units=None, desc='number of cells in series per string')
        self.add_input('n_parallel_per_str', shape=(1,), units=None, desc='number of cells in parallel per string')
        self.add_input('eta_parc', shape=(nn,), units=None, desc='power architecture electrical efficiency')

        self.add_output('p_cell', shape=(n_str, nn), units='W', desc='Cell power', upper = 1200, lower = -100)
        self.add_output('p_bat', shape=(n_str, nn), units='W', desc='Battery power', upper = 1200, lower = -100)
        
        total_elements = n_str * nn
        diag_indices = np.arange(total_elements)
        
        # p_bat = (p_train_elec_req + p_aux_elec/n_str) / eta_parc
        # p_bat wrt p_train_elec_req: diagonal (element-wise)
        self.declare_partials('p_bat', 'p_train_elec_req', rows=diag_indices, cols=diag_indices)
        # p_bat wrt p_aux_elec: p_bat[i,j] depends on p_aux_elec[j] (column-wise)
        rows_aux = np.arange(total_elements)
        cols_aux = np.tile(np.arange(nn), n_str)
        self.declare_partials('p_bat', 'p_aux_elec', rows=rows_aux, cols=cols_aux)
        # p_bat wrt eta_parc: p_bat[i,j] depends on eta_parc[j] (column-wise)
        rows_eta = np.arange(total_elements)
        cols_eta = np.tile(np.arange(nn), n_str)
        self.declare_partials('p_bat', 'eta_parc', rows=rows_eta, cols=cols_eta)
        
        # p_cell = p_bat / (n_series_per_str * n_parallel_per_str)
        # p_cell wrt p_bat: diagonal (element-wise)
        # p_cell wrt n_series_per_str: all elements depend on this scalar
        self.declare_partials('p_cell', 'n_series_per_str', rows=diag_indices, cols=np.zeros(total_elements, dtype=int))
        # p_cell wrt n_parallel_per_str: all elements depend on this scalar
        self.declare_partials('p_cell', 'n_parallel_per_str', rows=diag_indices, cols=np.zeros(total_elements, dtype=int))
        # p_cell wrt p_train_elec_req: indirect through p_bat (diagonal)
        self.declare_partials('p_cell', 'p_train_elec_req', rows=diag_indices, cols=diag_indices)
        # p_cell wrt p_aux_elec: indirect through p_bat (column-wise)
        self.declare_partials('p_cell', 'p_aux_elec', rows=rows_aux, cols=cols_aux)
        # p_cell wrt eta_parc: indirect through p_bat (column-wise)
        self.declare_partials('p_cell', 'eta_parc', rows=rows_eta, cols=cols_eta)


    def compute(self, inputs, outputs):
        p_aux_elec = inputs['p_aux_elec']
        p_train_elec_req = inputs['p_train_elec_req']
        n_series_per_str = inputs['n_series_per_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        eta_parc = inputs['eta_parc']

        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        
        # Broadcast p_aux_elec to each battery string and calculate p_bat
        # p_aux_elec is total auxiliary power, distribute equally to each string
        p_aux_per_str = p_aux_elec / n_str
        p_bat = (p_train_elec_req + p_aux_per_str) / eta_parc
        
        # Calculate cell power: p_cell = p_bat / (n_series_per_str * n_parallel_per_str)
        # Note: n_str is not divided here since p_bat is already per-string
        p_cell = p_bat / n_series_per_str / n_parallel_per_str

        #jax.debug.print("p_cell (W): {}", p_cell)  # Disabled for speed
        #jax.debug.print("p_bat (kW): {}", p_bat / 1000.)  # Disabled for speed

        outputs['p_cell'] = p_cell
        outputs['p_bat'] = p_bat
    
    def compute_partials(self, inputs, partials):
        p_aux_elec = inputs['p_aux_elec']
        p_train_elec_req = inputs['p_train_elec_req']
        n_series_per_str = inputs['n_series_per_str'][0]
        n_parallel_per_str = inputs['n_parallel_per_str'][0]
        eta_parc = inputs['eta_parc']
        
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        
        # Intermediate values
        p_aux_per_str = p_aux_elec / n_str
        denominator = n_series_per_str * n_parallel_per_str
        
        # p_bat = (p_train_elec_req + p_aux_per_str) / eta_parc
        # ∂p_bat/∂p_train_elec_req = 1/eta_parc (diagonal, element-wise)
        partials['p_bat', 'p_train_elec_req'] = np.tile(1.0 / eta_parc, n_str)
        
        # ∂p_bat/∂p_aux_elec = (1/n_str) / eta_parc (column-wise, repeated for each string)
        partials['p_bat', 'p_aux_elec'] = np.tile(1.0 / (n_str * eta_parc), n_str)
        
        # ∂p_bat/∂eta_parc = -(p_train_elec_req + p_aux_per_str) / eta_parc^2 (column-wise)
        numerator = p_train_elec_req + p_aux_per_str
        partials['p_bat', 'eta_parc'] = (-numerator / (eta_parc**2)).flatten()
        
        # p_cell = p_bat / (n_series_per_str * n_parallel_per_str)
        # ∂p_cell/∂p_bat = 1 / (n_series_per_str * n_parallel_per_str) (diagonal)
        
        # ∂p_cell/∂n_series_per_str = -p_bat / (n_series_per_str^2 * n_parallel_per_str)
        p_bat = (p_train_elec_req + p_aux_per_str) / eta_parc
        partials['p_cell', 'n_series_per_str'] = (-p_bat / (n_series_per_str**2 * n_parallel_per_str)).flatten()
        
        # ∂p_cell/∂n_parallel_per_str = -p_bat / (n_series_per_str * n_parallel_per_str^2)
        partials['p_cell', 'n_parallel_per_str'] = (-p_bat / (n_series_per_str * n_parallel_per_str**2)).flatten()
        
        # Chain rule for p_cell indirect dependencies
        # ∂p_cell/∂p_train_elec_req = (1/denominator) * (1/eta_parc)
        partials['p_cell', 'p_train_elec_req'] = np.tile(1.0 / (denominator * eta_parc), n_str)
        
        # ∂p_cell/∂p_aux_elec = (1/denominator) * (1/(n_str * eta_parc))
        partials['p_cell', 'p_aux_elec'] = np.tile(1.0 / (denominator * n_str * eta_parc), n_str)
        
        # ∂p_cell/∂eta_parc = (1/denominator) * (-numerator / eta_parc^2)
        partials['p_cell', 'eta_parc'] = (-numerator / (denominator * eta_parc**2)).flatten()


class UpdateCellState(om.ExplicitComponent):
    """
    Updates cell state variables for battery modeling.
    Handles matrices with shape (n_str, nn) where each row is a battery string.
    """
    def initialize(self):
        #self.options.declare('mode', types=str)
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('n_str', default=1, desc='number of battery strings')

    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        n_str = self.options['n_str']

        self.add_input('i_cell', units='A', desc='cell current matrix', shape=(n_str, num_nodes))

        self.add_input('cell_capacity', units='A*h', desc='cell capacity', shape=(1,))
        self.add_input('n_series_per_str', units=None, desc='number of cells in series per string', shape=(1,))
        self.add_input('n_parallel_per_str', units=None, desc='number of cells in parallel per string', shape=(1,))
        self.add_input('ocv_cell', units='V', desc='open circuit voltage of the cell matrix', shape=(n_str, num_nodes))
        self.add_input('vline_cell', units='V', desc='line voltage of the cell matrix', shape=(n_str, num_nodes))
        self.add_input('m_cell', units='kg', desc='mass of the cell', shape=(1,))
        self.add_input('cp_cell', units='J/kg/K', desc='specific heat capacity of the cell', shape=(1,))
        self.add_input('q_cool_bat', units='kW', desc='cooling power of the battery matrix', shape=(n_str, num_nodes))

        self.add_output('dsoc_dt', units='1/h', desc='state of charge rate of change of the battery matrix', shape=(n_str, num_nodes))
        self.add_output('eta_cell', units=None, desc='efficiency of the cell matrix', shape=(n_str, num_nodes))
        self.add_output('net_q_heat_ess', units='W', desc='net heat of the battery matrix', shape=(n_str, num_nodes))
        self.add_output('heat_ess', units='W', desc='heat of the battery matrix', shape=(n_str, num_nodes))
        self.add_output('c_rate', units='1/h', desc='c-rate of the cell matrix', shape=(n_str, num_nodes))
        self.add_output('dT_dt', units='K/s', desc='heat rate of the cell matrix', shape=(n_str, num_nodes))
        
        # Define sparsity patterns
        total_elements = n_str * num_nodes
        diag_indices = np.arange(total_elements)
        scalar_rows = diag_indices
        scalar_cols_zeros = np.zeros(total_elements, dtype=int)
        
        # dsoc_dt = -i_cell / cell_capacity
        self.declare_partials('dsoc_dt', 'i_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('dsoc_dt', 'cell_capacity', rows=scalar_rows, cols=scalar_cols_zeros)
        
        # c_rate = i_cell / cell_capacity
        self.declare_partials('c_rate', 'i_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('c_rate', 'cell_capacity', rows=scalar_rows, cols=scalar_cols_zeros)
        
        # heat_cell = abs(ocv_cell - vline_cell) * i_cell (diagonal dependencies)
        # eta_cell = 1 - heat_cell / (vline_cell * i_cell) (diagonal dependencies)
        self.declare_partials('eta_cell', 'ocv_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('eta_cell', 'vline_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('eta_cell', 'i_cell', rows=diag_indices, cols=diag_indices)
        
        # heat_ess = heat_cell * n_cells_per_str (diagonal + scalar dependencies)
        self.declare_partials('heat_ess', 'ocv_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('heat_ess', 'vline_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('heat_ess', 'i_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('heat_ess', 'n_series_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        self.declare_partials('heat_ess', 'n_parallel_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        
        # net_q_heat_ess = net_q_heat_cell * n_cells_per_str
        self.declare_partials('net_q_heat_ess', 'ocv_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('net_q_heat_ess', 'vline_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('net_q_heat_ess', 'i_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('net_q_heat_ess', 'q_cool_bat', rows=diag_indices, cols=diag_indices)
        self.declare_partials('net_q_heat_ess', 'n_series_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        self.declare_partials('net_q_heat_ess', 'n_parallel_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        
        # dT_dt = net_q_heat_cell / (m_cell * cp_cell)
        self.declare_partials('dT_dt', 'ocv_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('dT_dt', 'vline_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('dT_dt', 'i_cell', rows=diag_indices, cols=diag_indices)
        self.declare_partials('dT_dt', 'q_cool_bat', rows=diag_indices, cols=diag_indices)
        self.declare_partials('dT_dt', 'n_series_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        self.declare_partials('dT_dt', 'n_parallel_per_str', rows=scalar_rows, cols=scalar_cols_zeros)
        self.declare_partials('dT_dt', 'm_cell', rows=scalar_rows, cols=scalar_cols_zeros)
        self.declare_partials('dT_dt', 'cp_cell', rows=scalar_rows, cols=scalar_cols_zeros)

    def compute(self, inputs, outputs):
        i_cell = inputs['i_cell']
        cell_capacity = inputs['cell_capacity']
        n_series_per_str = inputs['n_series_per_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        ocv_cell = inputs['ocv_cell']
        vline_cell = inputs['vline_cell']
        m_cell = inputs['m_cell']
        cp_cell = inputs['cp_cell']
        q_cool_bat = inputs['q_cool_bat']

        nn = self.options['num_nodes']
        n_str = self.options['n_str']

        #cap_delta = jnp.cumsum(i_cell * dt / cell_capacity)

        # debug_print("i_cell: {}", i_cell)  # Disabled for speed
        # dsoc_dt = -i_cell / cell_capacity (broadcast cell_capacity to each string)
        dsoc_dt = -i_cell / cell_capacity
        #jax.debug.print("dsoc_dt: {}", dsoc_dt)
        
        # debug_print("dsoc_dt: {}", dsoc_dt)  # Disabled for speed
        # n_cells per string
        n_cells_per_str = n_series_per_str * n_parallel_per_str

        #elif self.options['mode'] == 'discharge':
        #soc = soc_init - cap_delta
        # net_q_heat_cell and heat_cell are per cell, so divide q_cool_bat by n_cells_per_str
        net_q_heat_cell = np.abs(ocv_cell - vline_cell) * i_cell - q_cool_bat * 1000. / n_cells_per_str
        heat_cell = np.abs(ocv_cell - vline_cell) * i_cell
        #else:
        #    raise ValueError(f"Invalid mode: {self.options['mode']}")

        # dT_dt is per cell (broadcast m_cell and cp_cell)
        dT_dt = net_q_heat_cell / (m_cell * cp_cell)
        
        #t_cell = t_cell_init + jnp.cumsum(net_q_heat_cell * dt / (m_cell * cp_cell))
        eta_cell = 1 - heat_cell / (vline_cell * i_cell)

        # Battery-level state variables (multiply cell-level by n_cells_per_str)
        # Note: Since we're working per string, we multiply by n_cells_per_str (not n_str * n_series_per_str * n_parallel_per_str)
        net_q_heat_ess = net_q_heat_cell * n_cells_per_str
        heat_ess = heat_cell * n_cells_per_str

        # Compute c-rate
        c_rate = i_cell / cell_capacity
        # debug_print("Cell current (A): {}", i_cell)  # Disabled for speed

        outputs['dsoc_dt'] = dsoc_dt
        outputs['eta_cell'] = eta_cell
        outputs['net_q_heat_ess'] = net_q_heat_ess
        outputs['heat_ess'] = heat_ess
        outputs['c_rate'] = c_rate
        outputs['dT_dt'] = dT_dt
    
    def compute_partials(self, inputs, partials):
        i_cell = inputs['i_cell']
        cell_capacity = inputs['cell_capacity'][0]
        n_series_per_str = inputs['n_series_per_str'][0]
        n_parallel_per_str = inputs['n_parallel_per_str'][0]
        ocv_cell = inputs['ocv_cell']
        vline_cell = inputs['vline_cell']
        m_cell = inputs['m_cell'][0]
        cp_cell = inputs['cp_cell'][0]
        q_cool_bat = inputs['q_cool_bat']
        
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        
        # Intermediate calculations
        n_cells_per_str = n_series_per_str * n_parallel_per_str
        diff_v = ocv_cell - vline_cell
        abs_diff_v = np.abs(diff_v)
        sign_diff_v = np.sign(diff_v)  # Derivative of abs()
        
        heat_cell = abs_diff_v * i_cell
        net_q_heat_cell = abs_diff_v * i_cell - q_cool_bat * 1000.0 / n_cells_per_str
        
        # 1. dsoc_dt = -i_cell / cell_capacity
        partials['dsoc_dt', 'i_cell'] = np.full(n_str * nn, -1.0 / cell_capacity)
        partials['dsoc_dt', 'cell_capacity'] = (i_cell / cell_capacity**2).flatten()
        
        # 2. c_rate = i_cell / cell_capacity
        partials['c_rate', 'i_cell'] = np.full(n_str * nn, 1.0 / cell_capacity)
        partials['c_rate', 'cell_capacity'] = (-i_cell / cell_capacity**2).flatten()
        
        # 3. eta_cell = 1 - heat_cell / (vline_cell * i_cell) = 1 - abs_diff_v / vline_cell
        # ∂eta_cell/∂ocv_cell = -sign(diff_v) / vline_cell
        partials['eta_cell', 'ocv_cell'] = (-sign_diff_v / vline_cell).flatten()
        # ∂eta_cell/∂vline_cell = sign(diff_v) / vline_cell + abs_diff_v / vline_cell^2
        partials['eta_cell', 'vline_cell'] = (sign_diff_v / vline_cell + abs_diff_v / vline_cell**2).flatten()
        # ∂eta_cell/∂i_cell = 0 (cancels out in the ratio)
        partials['eta_cell', 'i_cell'] = np.zeros(n_str * nn)
        
        # 4. heat_ess = heat_cell * n_cells_per_str
        # ∂heat_ess/∂ocv_cell = sign(diff_v) * i_cell * n_cells_per_str
        partials['heat_ess', 'ocv_cell'] = (sign_diff_v * i_cell * n_cells_per_str).flatten()
        # ∂heat_ess/∂vline_cell = -sign(diff_v) * i_cell * n_cells_per_str
        partials['heat_ess', 'vline_cell'] = (-sign_diff_v * i_cell * n_cells_per_str).flatten()
        # ∂heat_ess/∂i_cell = abs_diff_v * n_cells_per_str
        partials['heat_ess', 'i_cell'] = (abs_diff_v * n_cells_per_str).flatten()
        # ∂heat_ess/∂n_series_per_str = heat_cell * n_parallel_per_str
        partials['heat_ess', 'n_series_per_str'] = (heat_cell * n_parallel_per_str).flatten()
        # ∂heat_ess/∂n_parallel_per_str = heat_cell * n_series_per_str
        partials['heat_ess', 'n_parallel_per_str'] = (heat_cell * n_series_per_str).flatten()
        
        # 5. net_q_heat_ess = net_q_heat_cell * n_cells_per_str
        # ∂net_q_heat_ess/∂ocv_cell = sign(diff_v) * i_cell * n_cells_per_str
        partials['net_q_heat_ess', 'ocv_cell'] = (sign_diff_v * i_cell * n_cells_per_str).flatten()
        # ∂net_q_heat_ess/∂vline_cell = -sign(diff_v) * i_cell * n_cells_per_str
        partials['net_q_heat_ess', 'vline_cell'] = (-sign_diff_v * i_cell * n_cells_per_str).flatten()
        # ∂net_q_heat_ess/∂i_cell = abs_diff_v * n_cells_per_str
        partials['net_q_heat_ess', 'i_cell'] = (abs_diff_v * n_cells_per_str).flatten()
        # ∂net_q_heat_ess/∂q_cool_bat = -1000
        partials['net_q_heat_ess', 'q_cool_bat'] = np.full(n_str * nn, -1000.0)
        # ∂net_q_heat_ess/∂n_series_per_str (using product rule and chain rule)
        # = net_q_heat_cell * n_parallel_per_str + q_cool_bat * 1000 * n_parallel_per_str / n_cells_per_str
        partials['net_q_heat_ess', 'n_series_per_str'] = (
            net_q_heat_cell * n_parallel_per_str + q_cool_bat * 1000.0 * n_parallel_per_str / n_cells_per_str
        ).flatten()
        # ∂net_q_heat_ess/∂n_parallel_per_str (using product rule and chain rule)
        # = net_q_heat_cell * n_series_per_str + q_cool_bat * 1000 * n_series_per_str / n_cells_per_str
        partials['net_q_heat_ess', 'n_parallel_per_str'] = (
            net_q_heat_cell * n_series_per_str + q_cool_bat * 1000.0 * n_series_per_str / n_cells_per_str
        ).flatten()
        
        # 6. dT_dt = net_q_heat_cell / (m_cell * cp_cell)
        denominator = m_cell * cp_cell
        # ∂dT_dt/∂ocv_cell = sign(diff_v) * i_cell / denominator
        partials['dT_dt', 'ocv_cell'] = (sign_diff_v * i_cell / denominator).flatten()
        # ∂dT_dt/∂vline_cell = -sign(diff_v) * i_cell / denominator
        partials['dT_dt', 'vline_cell'] = (-sign_diff_v * i_cell / denominator).flatten()
        # ∂dT_dt/∂i_cell = abs_diff_v / denominator
        partials['dT_dt', 'i_cell'] = (abs_diff_v / denominator).flatten()
        # ∂dT_dt/∂q_cool_bat = -1000 / (n_cells_per_str * denominator)
        partials['dT_dt', 'q_cool_bat'] = np.full(n_str * nn, -1000.0 / (n_cells_per_str * denominator))
        # ∂dT_dt/∂n_series_per_str = q_cool_bat * 1000 / (n_cells_per_str^2 * n_parallel_per_str * denominator)
        partials['dT_dt', 'n_series_per_str'] = (
            q_cool_bat * 1000.0 / (n_series_per_str**2 * n_parallel_per_str * denominator)
        ).flatten()
        # ∂dT_dt/∂n_parallel_per_str = q_cool_bat * 1000 / (n_cells_per_str^2 * n_series_per_str * denominator)
        partials['dT_dt', 'n_parallel_per_str'] = (
            q_cool_bat * 1000.0 / (n_parallel_per_str**2 * n_series_per_str * denominator)
        ).flatten()
        # ∂dT_dt/∂m_cell = -net_q_heat_cell / (m_cell^2 * cp_cell)
        partials['dT_dt', 'm_cell'] = (-net_q_heat_cell / (m_cell**2 * cp_cell)).flatten()
        # ∂dT_dt/∂cp_cell = -net_q_heat_cell / (m_cell * cp_cell^2)
        partials['dT_dt', 'cp_cell'] = (-net_q_heat_cell / (m_cell * cp_cell**2)).flatten()



class UnchainSOC(om.ExplicitComponent):
    """
    Component to unchain (split) the concatenated dsoc_dt vector into individual phase vectors.
    """
    
    def initialize(self):
        self.options.declare('num_phases', default=1, desc='Number of phases')
        self.options.declare('num_nodes', default=11, desc='Number of nodes per phase')
    
    def setup(self):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        total_nodes = num_phases * num_nodes
        
        # Add input for concatenated dsoc_dt
        self.add_input('dsoc_dt', shape=(total_nodes,), units='1/h', 
                      desc='Concatenated SOC rate for all phases')
        
        # Add outputs for each phase's dsoc_dt
        for i in range(num_phases):
            self.add_output(f'phase_{i}_dsoc_dt', shape=(num_nodes,), units='1/h', 
                           desc=f'SOC rate for phase {i}')
        
        # Set up partial derivatives
        for i in range(num_phases):
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            rows = np.arange(num_nodes)
            cols = np.arange(start_idx, end_idx)
            vals = np.ones(num_nodes)
            
            self.declare_partials(
                of=f'phase_{i}_dsoc_dt',
                wrt='dsoc_dt',
                val=vals,
                rows=rows,
                cols=cols
            )
    
    def compute(self, inputs, outputs):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        
        dsoc_dt = inputs['dsoc_dt']
        
        # Split dsoc_dt into individual phase vectors
        for i in range(num_phases):
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            outputs[f'phase_{i}_dsoc_dt'] = dsoc_dt[start_idx:end_idx]
    
    def compute_partials(self, inputs, partials):
        # Partial derivatives are constant (1.0) and already set up in setup()
        pass

class StringToMotorVoltageConverter(om.ExplicitComponent):
    """
    Component that converts voltage from battery strings (n_str, nn) to motors (nm, nn).
    Handles cases where n_str != nm by broadcasting/repeating voltage values.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of nodes')
        self.options.declare('n_str', default=1, desc='Number of battery strings')
        self.options.declare('nm', default=1, desc='Number of motors')
        self.options.declare('units', default='V', desc='Units of the voltage')
    
    def setup(self):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        nm = self.options['nm']
        units = self.options['units']
        
        self.add_input('v_mot_str', shape=(n_str, nn), units=units,
                      desc='Motor voltage per battery string')
        self.add_output('v_mot_mot', shape=(nm, nn), units=units,
                       desc='Motor voltage per motor')
        
        # Declare partials - depends on mapping between strings and motors
        self.declare_partials('v_mot_mot', 'v_mot_str', method='exact')
    
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        nm = self.options['nm']
        v_mot_str = inputs['v_mot_str']
        
        outputs['v_mot_mot'] = np.zeros((nm, nn))
        
        if n_str == nm:
            # 1-to-1 mapping
            outputs['v_mot_mot'] = v_mot_str
        elif nm == 2 * n_str:
            # Each string feeds 2 motors
            for i in range(n_str):
                outputs['v_mot_mot'][2*i, :] = v_mot_str[i, :]
                outputs['v_mot_mot'][2*i+1, :] = v_mot_str[i, :]
        else:
            # Map first min(n_str, nm) strings to motors
            m = min(n_str, nm)
            outputs['v_mot_mot'][:m, :] = v_mot_str[:m, :]
            # If nm > n_str, repeat last string's voltage for remaining motors
            if nm > n_str:
                for j in range(n_str, nm):
                    outputs['v_mot_mot'][j, :] = v_mot_str[-1, :]
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        nm = self.options['nm']
        
        # Build Jacobian matrix
        jac = np.zeros((nm * nn, n_str * nn))
        
        if n_str == nm:
            # Identity mapping
            for i in range(nm):
                for j in range(nn):
                    jac[i * nn + j, i * nn + j] = 1.0
        elif nm == 2 * n_str:
            # Each string feeds 2 motors
            for i in range(n_str):
                for j in range(nn):
                    # Motor 2*i depends on string i
                    jac[(2*i) * nn + j, i * nn + j] = 1.0
                    # Motor 2*i+1 depends on string i
                    jac[(2*i+1) * nn + j, i * nn + j] = 1.0
        else:
            # Map first min(n_str, nm) strings
            m = min(n_str, nm)
            for i in range(m):
                for j in range(nn):
                    jac[i * nn + j, i * nn + j] = 1.0
            # Remaining motors depend on last string
            if nm > n_str:
                for j in range(n_str, nm):
                    for k in range(nn):
                        jac[j * nn + k, (n_str-1) * nn + k] = 1.0
        
        partials['v_mot_mot', 'v_mot_str'] = jac


class UnchainVoltage(om.ExplicitComponent):
    """
    Component to unchain (split) the concatenated battery voltage matrix into individual phase matrices.
    Takes the long voltage matrix of shape (nm, total_nodes) and creates matrices of shape (nm, num_nodes) for each phase.
    
    Input format: Matrix of shape (nm, total_nodes) where total_nodes = num_phases * num_nodes
    Output format: Matrices of shape (nm, num_nodes) for each phase
    """
    
    def initialize(self):
        self.options.declare('num_phases', default=1, desc='Number of phases')
        self.options.declare('num_nodes', default=11, desc='Number of nodes per phase')
        self.options.declare('nm', default=1, desc='Number of motors')
    
    def setup(self):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        nm = self.options['nm']
        total_nodes = num_phases * num_nodes
        
        # Add input for concatenated voltage matrix
        self.add_input('voltage', shape=(nm, total_nodes), units='V', 
                      desc='Concatenated battery voltage matrix for all phases')
        
        # Add outputs for each phase's voltage matrix
        for i in range(num_phases):
            self.add_output(f'phase_{i}_voltage', shape=(nm, num_nodes), units='V', val = 800,
                           desc=f'Battery voltage matrix for phase {i}')
        
        # Set up partial derivatives
        for i in range(num_phases):
            start_idx = i * num_nodes
            # Partial derivatives for matrix: each output element (m, n) depends on input element (m, start_idx+n)
            rows = np.arange(nm * num_nodes)
            # Column indices: for each motor m, columns are m*total_nodes + start_idx + n
            cols = (np.repeat(np.arange(nm), num_nodes) * total_nodes + 
                   np.tile(np.arange(start_idx, start_idx + num_nodes), nm))
            vals = np.ones(nm * num_nodes)
            
            self.declare_partials(
                of=f'phase_{i}_voltage',
                wrt='voltage',
                val=vals,
                rows=rows,
                cols=cols
            )
    
    def compute(self, inputs, outputs):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        
        voltage = inputs['voltage']
        
        # Split voltage matrix into individual phase matrices
        # For each phase, extract the corresponding columns from each row (motor)
        for i in range(num_phases):
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            outputs[f'phase_{i}_voltage'] = voltage[:, start_idx:end_idx]
    
    def compute_partials(self, inputs, partials):
        # Partial derivatives are constant (1.0) and already set up in setup()
        pass


class ExtractMatrixRows(om.ExplicitComponent):
    """
    Component that extracts all rows from a matrix at once.
    Extracts all rows from matrix of shape (num_comps, num_nodes) to separate vectors.
    Outputs one vector per row.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of nodes (columns in matrix)')
        self.options.declare('num_comps', default=1, desc='Number of components (rows in matrix)')
        self.options.declare('units', default=None, desc='Units of the input matrix and output vectors')
    
    def setup(self):
        num_nodes = self.options['num_nodes']
        num_comps = self.options['num_comps']
        units = self.options['units']
        
        self.add_input('matrix_in', shape=(num_comps, num_nodes), units=units, 
                      desc='Input matrix')
        
        # Add one output per row
        for i in range(num_comps):
            self.add_output(f'vector_out_{i}', shape=(num_nodes,), units=units, 
                           desc=f'Extracted row {i} vector')
        
        # Declare partials for all outputs
        for i in range(num_comps):
            self.declare_partials(f'vector_out_{i}', 'matrix_in', method='exact')
    
    def compute(self, inputs, outputs):
        matrix_in = inputs['matrix_in']
        # Extract all rows at once
        for i in range(self.options['num_comps']):
            outputs[f'vector_out_{i}'] = matrix_in[i, :]
    
    def compute_partials(self, inputs, partials):
        num_nodes = self.options['num_nodes']
        num_comps = self.options['num_comps']
        
        # Partial derivatives: row i of input matrix contributes to output vector_i
        # Shape: (num_nodes, num_comps*num_nodes) for each output
        for i in range(num_comps):
            # For output i, only columns [i*num_nodes : (i+1)*num_nodes] are non-zero
            # Create a matrix with identity block at columns [i*num_nodes : (i+1)*num_nodes]
            start_col = i * num_nodes
            end_col = (i + 1) * num_nodes
            
            # Create zero matrix and set identity block
            partial_matrix = np.zeros((num_nodes, num_comps * num_nodes))
            partial_matrix[:, start_col:end_col] = np.eye(num_nodes)
            
            partials[f'vector_out_{i}', 'matrix_in'] = partial_matrix



class ConcatenateVectors(om.ExplicitComponent):
    """
    Component that concatenates multiple input vectors into a single output vector.
    """
    
    def initialize(self):
        self.options.declare('num_vectors', default=1, desc='Number of vectors to concatenate')
        self.options.declare('vector_length', default=1, desc='Length of each input vector')
    
    def setup(self):
        num_vectors = self.options['num_vectors']
        vector_length = self.options['vector_length']
        
        # Add inputs - one vector per input
        for i in range(num_vectors):
            self.add_input(f'vec{i}', shape=(vector_length,), units=None, desc=f'Input vector {i}')
        
        # Add output - concatenated vector
        self.add_output('vector_out', shape=(num_vectors * vector_length,), units=None, 
                       desc='Concatenated output vector')
        
        # Declare partials
        self.declare_partials('vector_out', [f'vec{i}' for i in range(num_vectors)], method='exact')
    
    def compute(self, inputs, outputs):
        num_vectors = self.options['num_vectors']
        vector_length = self.options['vector_length']
        
        # Concatenate all input vectors - collect them into array first
        vec_array = np.stack([inputs[f'vec{i}'] for i in range(num_vectors)])
        outputs['vector_out'] = vec_array.flatten()
    
    def compute_partials(self, inputs, partials):
        num_vectors = self.options['num_vectors']
        vector_length = self.options['vector_length']
        
        # Create block diagonal matrix - each input vector contributes to a contiguous block
        # Use scipy.sparse.block_diag for vectorized construction
        from scipy.sparse import block_diag
        identity_blocks = [np.eye(vector_length)] * num_vectors
        partials_matrix = block_diag(identity_blocks).toarray()
        
        # Assign partials for each input - slice the block diagonal matrix
        # Each input contributes to columns [i*vector_length : (i+1)*vector_length]
        for i in range(num_vectors):
            start_idx = i * vector_length
            partials['vector_out', f'vec{i}'] = partials_matrix[:, start_idx:start_idx + vector_length]




class ChainSOC(om.ExplicitComponent):
    """
    Component to chain (concatenate) SOC results from multiple phases.
    """
    
    def initialize(self):
        self.options.declare('num_phases', default=1, desc='Number of phases')
        self.options.declare('num_nodes', default=11, desc='Number of nodes per phase')
    
    def setup(self):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        total_nodes = num_phases * num_nodes
        
        # Add inputs for each phase's SOC
        for i in range(num_phases):
            self.add_input(f'phase_{i}_soc', shape=(num_nodes,), units=None, 
                          desc=f'SOC for phase {i}')
        
        # Add output for concatenated SOC
        self.add_output('soc', shape=(total_nodes,), units=None, 
                       desc='Concatenated SOC for all phases')
        
        # Set up partial derivatives
        for i in range(num_phases):
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            rows = np.arange(start_idx, end_idx)
            cols = np.arange(num_nodes)
            vals = np.ones(num_nodes)
            
            self.declare_partials(
                of='soc',
                wrt=f'phase_{i}_soc',
                val=vals,
                rows=rows,
                cols=cols
            )
    
    def compute(self, inputs, outputs):
        num_phases = self.options['num_phases']
        num_nodes = self.options['num_nodes']
        
        # Concatenate SOC from all phases
        soc_list = []
        for i in range(num_phases):
            soc_list.append(inputs[f'phase_{i}_soc'])
        
        #soc_clipped = np.clip(np.concatenate(soc_list), 0, 1)
        soc_clipped = np.concatenate(soc_list)
        
        outputs['soc'] = soc_clipped
    
    def compute_partials(self, inputs, partials):
        # Partial derivatives are constant (1.0) and already set up in setup()
        pass

class IntegrateMissionSOC(om.Group):
    """
    Group that handles multi-phase SOC integration for mission analysis.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('phases', default=['climb', 'cruise', 'descent'], desc='phases as strings')
    
    def setup(self):
        # Unpack Options
        num_nodes = self.options['num_nodes']
        phases = self.options['phases']
        num_phases = len(phases)
        
        # Calculate nodes per phase (total nodes divided by number of phases)
        nodes_per_phase = num_nodes 
        
        # Add component to unchain (split) the concatenated dsoc_dt vector
        self.add_subsystem('unchain_soc', 
                          UnchainSOC(num_phases=num_phases, num_nodes=nodes_per_phase),
                          promotes=['dsoc_dt'])
        
        for idx, phase in enumerate(phases):
            integrator = self.add_subsystem(f'soc_integrator_{phase}', 
                                        Integrator(num_nodes=nodes_per_phase, 
                                                   time_setup="duration",
                                                   diff_units="h",
                                                   method="simpson"),
                                        promotes=[])
        
            # Set up integrator inputs/outputs
            integrator.add_integrand('soc', units=None, rate_name='dsoc_dt')
            
            
            # Connect unchained dsoc_dt to integrator
            self.connect(f'unchain_soc.phase_{idx}_dsoc_dt', f'soc_integrator_{phase}.dsoc_dt')
            
            # Connect initial SOC (only for first phase)
            if idx != 0:
                # Connect final SOC from previous phase to initial SOC of current phase
                prev_phase = phases[idx-1]
                self.connect(f'soc_integrator_{prev_phase}.soc_final', f'soc_integrator_{phase}.soc_initial')

        # Add component to chain (concatenate) SOC results from all phases
        self.add_subsystem('chain_soc', 
                          ChainSOC(num_phases=num_phases, num_nodes=nodes_per_phase),
                          promotes=['soc'])
        
        # Connect phase SOC outputs to chain component
        for idx, phase in enumerate(phases):
            self.connect(f'soc_integrator_{phase}.soc', f'chain_soc.phase_{idx}_soc')


class SOCLimiter(om.ExplicitComponent):
    """
    Clips the state of charge (soc) between 0 and 1 using smooth min and max.
    Handles matrices with shape (n_str, nn) where each row is a battery string.
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('n_str', default=1, desc='number of battery strings')
        self.options.declare('soc_min', default=1e-3, desc='minimum state of charge')
        self.options.declare('soc_max', default=1-1e-6, desc='maximum state of charge')
        self.options.declare('mu', default=0.0001, desc='smoothing parameter')


    def setup(self):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        self.add_input('soc', shape=(nn * n_str), val = 0.99, desc='Raw state of charge matrix')
        self.add_output('soc_clipped', shape=(nn * n_str), val = 0.99, desc='Clipped state of charge [0,1]', 
                       lower = 1e-3, upper = 1-1e-3)
        
        rows = np.arange(nn * n_str)
        cols = np.arange(nn * n_str)

        
        self.declare_partials('soc_clipped', 'soc', rows=rows, cols=cols, method='exact')

    def compute(self, inputs, outputs):
        soc = inputs['soc']
        soc_min = np.maximum(soc, self.options['soc_min'])
        soc_clipped = np.minimum(soc_min, self.options['soc_max'])

        outputs['soc_clipped'] = soc_clipped

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        n_str = self.options['n_str']
        soc_min = self.options['soc_min']
        soc_max = self.options['soc_max']
        soc = inputs['soc']
        # Partial derivatives are constant (1.0) and already set up in setup()
        partials['soc_clipped', 'soc'] = np.ones(nn * n_str)



class EmpiricalBatteryPower(om.Group):
    """
    Battery power group that operates with matrix inputs/outputs.
    
    - Inputs/outputs are matrices of shape (num_comps, num_nodes)
    - MatrixToVectorConverter components flatten matrices before battery interpolation and SOC integration
    - VectorToMatrixConverter components reshape outputs back to matrices
    - Voltage outputs remain as matrices and can be passed back to motor components
    """

    _data_loaded = False
    _ocv_data_loaded = False
    _loaded_datasheet_name = None

    def initialize(self):

        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('phases', default=['climb', 'cruise', 'descent'], desc='phases as strings')
        self.options.declare('nm', default=1, desc='Number of motors')
        self.options.declare('n_str', default=1, desc='Number of battery strings (for matrix dimensions)')
        self.options.declare('feeder_mode', default='independent', desc='Feeder mode: independent, full_crossfeed, or grouped')
        self.options.declare('active_strings', default=None, 
                           desc='Array of active string indices (None = all active)')
        self.options.declare('battery_datasheet_name', 
                            default='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105',
                            desc='Name of the battery datasheet to use for interpolation')

    @classmethod
    def _load_data(cls, battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105'):
        # Check if we already have this datasheet loaded
        if cls._ocv_data_loaded and cls._loaded_datasheet_name == battery_datasheet_name:
            return
        BatteryData.load_data(bat_filename='models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '.xlsx', 
                            cell_sheetname='BOL_cell_fct_CRate', config_sheetname='battery_config')

        cls._ocv_data_incr = BatteryData.ocv_data_incr
        cls._soc_data_incr = BatteryData.soc_data_incr
        cls._loaded_datasheet_name = battery_datasheet_name
        cls._ocv_data_loaded = True

    def setup(self):

        # Unpack Options
        num_nodes = self.options['num_nodes']
        phases = self.options['phases']
        num_phases = len(phases)
        nm = self.options['nm']
        n_str = self.options['n_str']  # Number of battery strings determines matrix dimensions
        
        # Get feeder mode from options (default to independent)
        feeder_mode = self.options['feeder_mode']
        active_strings = self.options['active_strings']
        battery_datasheet_name = self.options['battery_datasheet_name']
        
        # Load battery data using the option value (now available in setup)
        self._load_data(battery_datasheet_name=battery_datasheet_name)
        
        # Create auxiliary power interpolator based on altitude and disa
        # Training data from table: altitude_ft [0, 12000, 25000, 30000], disa [0, 35]
        # Total EL values in kW, converted to W for p_aux_elec
        altitude_training_ft = np.array([0, 12000, 25000, 30000], dtype=float)  # feet
        disa_training_degC = np.array([0, 35], dtype=float)  # disa values
        
        # Create 2D grid of total_EL values in W (converted from kW)
        # Rows: altitude, Columns: disa
        # disa=0: [42, 64.5, 107.4, 107.4] kW -> [42000, 64500, 107400, 107400] W
        # disa=35: [48.2, 74.8, 124.7, 124.7] kW -> [48200, 74800, 124700, 124700] W
        p_aux_training_W = np.array([
            [42000.0, 48200.0],      # altitude 0 ft
            [64500.0, 74800.0],      # altitude 12000 ft
            [107400.0, 124700.0],    # altitude 25000 ft
            [107400.0, 124700.0]     # altitude 30000 ft
        ])
        
        p_aux_interp = om.MetaModelStructuredComp(vec_size=num_nodes, method='slinear')
        p_aux_interp.add_input('altitude', 0.0, training_data=altitude_training_ft, units='ft', shape=(num_nodes,))
        p_aux_interp.add_input('disa', 0.0, training_data=disa_training_degC, units='degC', shape=(num_nodes,))
        p_aux_interp.add_output('p_aux_elec', 50000.0, training_data=p_aux_training_W, units='W', shape=(num_nodes,))
        p_aux_interp.options['extrapolate'] = True
        self.add_subsystem('p_aux_interp', p_aux_interp, promotes=['*'])
        
        # Add DemandFeed component to convert motor power matrix to battery power matrix
        self.add_subsystem('demand_feed',
                          DemandFeed(num_nodes=num_nodes, nm=nm, n_str=n_str, 
                                    feeder_mode=feeder_mode, active_strings=active_strings),
                          promotes_inputs=['p_train_elec'],
                          promotes_outputs=['p_train_elec_req'])

        self.add_subsystem('battery_power', BatteryPower(num_nodes=num_nodes, n_str=n_str), 
                          promotes_inputs=['p_aux_elec', 'p_train_elec_req', 'n_series_per_str', 
                                         'n_parallel_per_str', 'eta_parc'],
                          promotes_outputs=['p_cell', 'p_bat'])

        # Add matrix-to-vector converter before battery interpolation
        self.add_subsystem('mat_to_vec_interp',
                          MatrixToVectorConverter(
                              num_nodes=num_nodes,
                              num_comps=n_str,
                              input_names=['soc', 'p_cell'],
                              output_names=['soc_vect', 'p_cell_vect'],
                              units={'soc': None, 'p_cell': 'W'}
                          ),
                          promotes_inputs=['soc', 'p_cell'])

        

        self.add_subsystem('limit_min_soc', SmoothMaxComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1e-3, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('limit_max_soc', SmoothMinComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'soc_clipped')])
        #self.add_subsystem('limit_min_soc', SmoothMaxComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1e-3, n_comps=1, expected_min=1e-3, expected_max=1), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_soc', SmoothMinComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1, n_comps=1, expected_min=1e-3, expected_max=1), promotes_inputs=[], promotes_outputs=[('output', 'soc_clipped')])

        #self.add_subsystem('limit_min_soc', DiscontMaxComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1e-3, n_comps=1), promotes_inputs=[], promotes_outputs=[])
        #self.add_subsystem('limit_max_soc', DiscontMinComp(num_nodes=num_nodes * n_str, mode='limit', units=None, limit_val=1, n_comps=1), promotes_inputs=[], promotes_outputs=[('output', 'soc_clipped')])


        self.connect('mat_to_vec_interp.soc_vect', 'limit_min_soc.input_array')
        self.connect('limit_min_soc.output', 'limit_max_soc.input_array')
        

        # Create OCV interpolator - vec_size=1 with explicit shape
        ocv_interp = om.MetaModelStructuredComp(vec_size=int(round(num_nodes * n_str, 0)), method='akima')
        ocv_interp.add_input('soc_clipped', 0.5, training_data=self._soc_data_incr, units=None)
        ocv_interp.add_output('ocv_cell_vect', 3.5, training_data=self._ocv_data_incr, units='V', upper = np.max(self._ocv_data_incr), lower = np.min(self._ocv_data_incr))
        ocv_interp.options['extrapolate'] = True
        #ocv_interp.options['always_opt'] = True
        self.add_subsystem('ocv_interp', ocv_interp, promotes_inputs=['soc_clipped'], promotes_outputs=[])


        # self.add_subsystem('cell_power_calc', CellPowerCalculator(num_nodes = num_nodes), promotes=['*'])
        self.add_subsystem('battery_interp', BatteryMMInterpolationGroup(nn=int(round(num_nodes*n_str,0)), battery_datasheet_name=battery_datasheet_name), promotes_inputs=['soc_clipped'])
        
        # Connect matrix-to-vector converter to battery interpolation
        self.connect('mat_to_vec_interp.p_cell_vect', 'battery_interp.p_cell')
        
        # Add vector-to-matrix converter after battery interpolation
        self.add_subsystem('vec_to_mat_interp',
                          VectorToMatrixConverter(
                              num_nodes=num_nodes,
                              num_comps=n_str,
                              input_names=['vline_cell_vect', 'i_cell_vect','ocv_cell_vect'],
                              output_names=['vline_cell', 'i_cell', 'ocv_cell'],
                              units={'vline_cell_vect': 'V', 'i_cell_vect': 'A', 'ocv_cell_vect': 'V'}
                          ),
                          promotes_outputs=['vline_cell', 'i_cell', 'ocv_cell'])
        self.connect('battery_interp.vline_cell', 'vec_to_mat_interp.vline_cell_vect')
        self.connect('battery_interp.i_cell', 'vec_to_mat_interp.i_cell_vect')
        self.connect('ocv_interp.ocv_cell_vect', 'vec_to_mat_interp.ocv_cell_vect')


        self.add_subsystem('battery_current_voltage', BatteryCurrentVoltage(num_nodes=num_nodes, n_str=n_str), 
                          promotes_inputs=['vline_cell', 'i_cell', 'n_series_per_str', 'n_parallel_per_str', 'eta_parc'],
                          promotes_outputs=['v_bat', 'i_bat'])

        self.add_subsystem('extract_min_voltage', SmoothMinComp(num_nodes=num_nodes, n_comps=n_str, units='V', mode = 'extract_vector'), 
                          promotes=[])

        self.connect('battery_current_voltage.v_mot', 'lim_min_vmot.input_array')

        self.add_subsystem('lim_min_vmot', SmoothMaxComp(num_nodes=num_nodes, n_comps=n_str, units='V', 
        mode = 'limit', limit_val = 450, default_output = 700), promotes_outputs=[('output','v_mot')])
        #self.add_subsystem('lim_min_vmot', SmoothMaxComp(num_nodes=num_nodes, n_comps=n_str, units='V', 
        #mode = 'limit', limit_val = 450, default_output = 700, expected_min=450, expected_max=850), promotes_outputs=[('output','v_mot')])
        
        # Connect battery voltage to extract min component
        self.connect('v_bat', 'extract_min_voltage.input_array')

        

        self.add_subsystem('update_cell_state', UpdateCellState(num_nodes=num_nodes, n_str=n_str),
                          promotes_inputs=['i_cell', 'cell_capacity', 'n_series_per_str', 'n_parallel_per_str',
                                         'ocv_cell', 'vline_cell', 'm_cell', 'cp_cell', 'q_cool_bat'],
                          promotes_outputs=['dsoc_dt', 'eta_cell', 'net_q_heat_ess', 'heat_ess', 'c_rate', 'dT_dt'])
        self.connect('dsoc_dt', 'extract_dsoc_dt_rows.matrix_in')


        if num_phases > 1:
            # Multi-phase integration: one IntegrateMissionSOC per battery string
            # num_nodes is already the total across all phases
            total_nodes_per_str = num_nodes
            nodes_per_phase = num_nodes // num_phases
            
            # Extract all rows from dsoc_dt matrix at once
            self.add_subsystem('extract_dsoc_dt_rows',
                              ExtractMatrixRows(num_nodes=total_nodes_per_str, num_comps=n_str, units='1/h'),
                              promotes_inputs=[])
            
            # Add IntegrateMissionSOC components - one per battery string
            # No padding needed since each phase integrates separately and nodes_per_phase can be guaranteed odd
            for i in range(n_str):
                # Add IntegrateMissionSOC for this battery string
                self.add_subsystem(f'integrate_mission_soc_str{i}', 
                                  IntegrateMissionSOC(num_nodes=nodes_per_phase, phases=phases),
                                  promotes=[])
                self.connect(f'extract_dsoc_dt_rows.vector_out_{i}', f'integrate_mission_soc_str{i}.dsoc_dt')
            
            # Concatenate all SOC outputs from individual IntegrateMissionSOC components
            # Each IntegrateMissionSOC outputs a vector soc (length total_nodes_per_str), we need to concatenate them
            self.add_subsystem('concat_soc',
                              ConcatenateVectors(num_vectors=n_str, vector_length=total_nodes_per_str),
                              promotes=[])
            # Connect each IntegrateMissionSOC soc output to the concatenator
            for i in range(n_str):
                self.connect(f'integrate_mission_soc_str{i}.soc', f'concat_soc.vec{i}')
            
            # Convert concatenated vector back to matrix
            self.add_subsystem('collect_soc_matrix',
                              VectorToMatrixConverter(
                                  num_nodes=total_nodes_per_str,
                                  num_comps=n_str,
                                  input_names=['soc_vect'],
                                  output_names=['soc'],
                                  units={'soc_vect': None}
                              ),
                              promotes_outputs=['soc'])
            
            # Connect concatenated vector to matrix converter
            self.connect('concat_soc.vector_out', 'collect_soc_matrix.soc_vect')
            
            # Add voltage unchaining for multi-phase analysis
            nodes_per_phase = num_nodes // num_phases
            self.add_subsystem('unchain_voltage', 
                              UnchainVoltage(num_phases=num_phases, num_nodes=nodes_per_phase, nm=nm),
                              promotes=[])
            
            # Convert voltage from battery strings (n_str, nn) to motors (nm, nn) if needed
            if n_str != nm:
                self.add_subsystem('v_mot_str_to_mot',
                                  StringToMotorVoltageConverter(num_nodes=num_nodes, n_str=n_str, nm=nm, units='V'),
                                  promotes=[])
                self.connect('v_mot', 'v_mot_str_to_mot.v_mot_str')
                self.connect('v_mot_str_to_mot.v_mot_mot', 'unchain_voltage.voltage')
            else:
                self.connect('v_mot', 'unchain_voltage.voltage')
            # Note: voltage chaining output will be matrices of shape (nm, nodes_per_phase) that can be passed back to motors
            
        else:
            # Single-phase integration setup
            # Extract all rows from dsoc_dt matrix at once
            self.add_subsystem('extract_dsoc_dt_rows',
                              ExtractMatrixRows(num_nodes=num_nodes, num_comps=n_str, units='1/h'),
                              promotes_inputs=[])
            
            # Integrate each string separately - no padding needed since num_nodes is guaranteed odd
            # Add one integrator per battery string
            for i in range(n_str):
                integrator_name = f'soc_integrator_str{i}'
                integrator = Integrator(num_nodes=num_nodes, 
                                       time_setup="duration",
                                       diff_units="h",
                                       method="simpson")
                self.add_subsystem(integrator_name, integrator, promotes=[])
                
                # Set up integrator inputs/outputs
                integrator.add_integrand('soc', units=None, rate_name='dsoc_dt')
                
                # Connect extracted row to integrator
                self.connect(f'extract_dsoc_dt_rows.vector_out_{i}', f'{integrator_name}.dsoc_dt')
            
            # Concatenate all SOC outputs from individual integrators
            self.add_subsystem('concat_soc',
                              ConcatenateVectors(num_vectors=n_str, vector_length=num_nodes),
                              promotes=[])
            # Connect each integrator soc output to the concatenator
            for i in range(n_str):
                self.connect(f'soc_integrator_str{i}.soc', f'concat_soc.vec{i}')
            
            # Convert concatenated vector back to matrix
            self.add_subsystem('vec_to_mat_soc',
                              VectorToMatrixConverter(
                                  num_nodes=num_nodes,
                                  num_comps=n_str,
                                  input_names=['soc_vect'],
                                  output_names=['soc'],
                                  units={'soc_vect': None}
                              ),
                              promotes_outputs=['soc'])
            self.connect('concat_soc.vector_out', 'vec_to_mat_soc.soc_vect')
            
  

        
        

        # Use BroydenSolver for better performance with feedback loops
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        # Optimize BroydenSolver for speed
        
        #self.nonlinear_solver = om.BroydenSolver()

        #self.nonlinear_solver.options["compute_jacobian"] = True  # Skip Jacobian computation
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        #self.nonlinear_solver.linesearch.options['bound_enforcement'] = 'vector'
        self.nonlinear_solver.options['iprint'] = 2  # Reduce output
        self.nonlinear_solver.options['maxiter'] = 50  # Even lower for Broyden
        self.nonlinear_solver.options['atol'] = 5e-2  # More relaxed for Broyden
        self.nonlinear_solver.options['rtol'] = 5e-2  # More relaxed for Broyden
        # Use ScipyKrylov instead of DirectSolver to avoid memory issues
        self.linear_solver = om.DirectSolver()


def test_battery_component(battery_datasheet_name='MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent'):

    import time

    plot_results = True

    start_time = time.time()
    
    # Get the data for any other calculations you need
    bat_data = BatteryData.get_data(bat_filename='models/atlas/atlas/propulsion/empirical_data/' + battery_datasheet_name + '.xlsx', 
                                    cell_sheetname='BOL_cell_fct_CRate', 
                                    config_sheetname='battery_config')
    
    # Create a new instance of the BatteryPack component
    # Number of nodes to evaluate
    #phases = ['climb', 'cruise']
    phases = ["climb_e","climb_hy","cruise_1","descent"]
    num_phases = len(phases)
    num_nodes = 11*num_phases
    nm = 4  # Number of motors
    v_cutoff = 2.5
    v_ocv_max = 4.2
    feeder_mode = 'full_crossfeed'
    active_strings = None
    #bat_pack = BatteryPack(num_nodes = num_nodes)


    # Set up the OpenMDAO model
    model = om.Group()
    ivc = om.IndepVarComp()

    # p_train_elec as matrix: (nm, num_nodes) - power per motor at each time node
    ivc.add_output('p_train_elec', 1000 * np.ones((nm, num_nodes)), units='kW', desc='Propulsion electric power demand per motor')
    ivc.add_output('eta_parc', 1.0 * np.ones(num_nodes), units=None, desc='power architecture electrical efficiency')
    ivc.add_output('altitude', 10000.0 * np.ones(num_nodes), units='ft', desc='Altitude')
    ivc.add_output('disa', 0.0 * np.ones(num_nodes), units='degC', desc='Dry bulb temperature')
    #ivc.add_output('r_hv_cbl_ptrain', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to each motor')
    ivc.add_output('n_str', bat_data.n_str, desc='number of battery strings')
    duration_hours = 5 / 60
    mission_time = np.linspace(0, duration_hours, num_nodes)
    #ivc.add_output('duration', duration, units='min', desc='phase duration')
    for phase in phases:
        ivc.add_output(f'{phase}_duration', duration_hours, units='h', desc=f'{phase} duration')


    ivc.add_output('eta_converter', 1.0 * np.ones(num_nodes), units=None, desc='efficiency of the converter')

    #ivc.add_output('p_aux_elec', 100 * np.ones(num_nodes), units='W', desc='Auxiliary electric load on LV side')
    #ivc.add_output('r_hv_cbl_aux', 0.001 * np.ones(num_nodes), units='ohm', desc='DC loop resistance to LV converter')

    # Load up Battery Data
    ivc.add_output('cell_capacity', bat_data.cell_Ah_capacity, units='A*h', desc='Cell capacity')
    ivc.add_output('t_cell_initial', 303.15, units='K', desc='Initial temperature of the cell')
    ivc.add_output('m_cell', bat_data.m_cell, units='kg', desc='Mass of the cell')
    ivc.add_output('cp_cell', bat_data.cp_cell, units='J/kg/K', desc='Specific heat capacity of the cell')
    ivc.add_output('q_cool_bat', 25 * np.ones((bat_data.n_str, num_nodes)), units='kW', desc='Cooling power of the battery')
    ivc.add_output('soc_initial', 0.98, units=None, desc='Initial state of charge of the battery')
    ivc.add_output('n_series_per_str', bat_data.n_series_per_str, desc='Number of cells in series')
    ivc.add_output('n_parallel_per_str', bat_data.n_parallel_per_str, desc='Number of cells in parallel')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    #model.add_subsystem('charge_battery_group', ChargeEmpiricalBattery(charge_mode='known_power', num_nodes=num_nodes), promotes=['*'])

    model.add_subsystem('battery_group', EmpiricalBatteryPower(num_nodes=num_nodes, 
                                                                          nm=nm,
                                                                          n_str=bat_data.n_str,
                                                                          feeder_mode=feeder_mode,
                                                                          active_strings=active_strings,
                                                                          phases = phases,
                                                                          battery_datasheet_name=battery_datasheet_name), promotes=['*'])

    #model.options['default_surrogate'] = om.NearestNeighbor()
    prob = om.Problem(model, reports=False)
    

    for i in range(bat_data.n_str):
        for phase in phases:
            model.connect(f'{phase}_duration', f'integrate_mission_soc_str{i}.soc_integrator_{phase}.duration')
        model.connect(f'soc_initial', f'integrate_mission_soc_str{i}.soc_integrator_{phases[0]}.soc_initial')



    prob.setup()


    run_start_time = time.time()

    #prob.check_totals()

    om.n2(prob)

    prob.run_model()
    #prob.check_partials(compact_print=True, show_only_incorrect=True)

    end_time = time.time()
    print(f"Setup time: {run_start_time - start_time:.2f} seconds")
    print(f"Execution time: {end_time - run_start_time:.2f} seconds")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")

    #total_
    time_vec = []
    t0 = 0
    for phase in phases:
        time_vec.append(np.linspace(t0, t0 + prob.get_val(f'battery_group.integrate_mission_soc_str{i}.soc_integrator_{phase}.duration', units='h'), int(num_nodes / num_phases)))
        t0 += prob.get_val(f'battery_group.integrate_mission_soc_str{i}.soc_integrator_{phase}.duration', units='h')
    time_vec = np.concatenate(time_vec).flatten()



    # SOC Consistency Checker: Re-integrate current to verify SOC calculation (vectorized)
    i_cell = prob.get_val('i_cell', units='A')  # cell current array
    cell_capacity = prob.get_val('cell_capacity', units='A*h')[0]  # cell capacity (scalar)
    soc_initial = prob.get_val('soc_initial')  # initial SOC (scalar)
    soc_model = prob.get_val('soc_clipped')  # SOC from the model
    v_cell = prob.get_val('vline_cell')
    p_cell_approx = (prob.get_val('p_train_elec_req', units ='W')  + prob.get_val('p_aux_elec', units ='W') / bat_data.n_str) / bat_data.n_series_per_str / bat_data.n_parallel_per_str  / prob.get_val('eta_parc')
    i_cell_approx = p_cell_approx / v_cell


    # Re-integrate current to get SOC
    soc_reintegrated = soc_initial - cumulative_trapezoid(y = i_cell_approx  / cell_capacity, x=time_vec, initial=0)

    #soc_error = soc_model - soc_reintegrated
    # Plot comparison
    import matplotlib.pyplot as plt

    if plot_results:
        n_str = bat_data.n_str
        
        # Create time array for x-axis
        time_min = time_vec * 60
        
        # Handle soc_model shape - it might be 1D vector (nn * n_str,) or 2D matrix (n_str, nn)
        if len(soc_model.shape) == 1:
            # Reshape from (nn * n_str,) to (n_str, nn)
            nn_total = len(soc_model) // n_str
            soc_model = soc_model.reshape((n_str, nn_total))
        
        # SOC Consistency Check - plot each string
        fig_soc_check, ax_soc = plt.subplots(1, 1, figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_str))
        
        for i in range(n_str):
            soc_model_str = soc_model[i, :] if len(soc_model.shape) > 1 else soc_model
            # Handle soc_reintegrated shape - it might be 1D or 2D
            if len(soc_reintegrated.shape) > 1:
                soc_reintegrated_str = soc_reintegrated[i, :]
            else:
                soc_reintegrated_str = soc_reintegrated
            
            ax_soc.plot(time_min, soc_model_str, 'o-', linewidth=2, markersize=4, 
                       color=colors[i], label=f'Model SOC - String {i}', alpha=0.7)
            ax_soc.plot(time_min, soc_reintegrated_str, 'x--', linewidth=2, markersize=4, 
                       color=colors[i], label=f'Reintegrated SOC - String {i}', alpha=0.7)
        ax_soc.set_xlabel('Time (min)')
        ax_soc.set_ylabel('State of Charge (SOC)')
        ax_soc.set_title('SOC Consistency Check - Per String')
        ax_soc.legend(ncol=2, loc='best')
        ax_soc.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print max difference
        if len(soc_model.shape) > 1:
            soc_diff = np.abs(soc_model - soc_reintegrated) if len(soc_reintegrated.shape) > 1 else np.abs(soc_model - soc_reintegrated[:, np.newaxis])
            print(f'Max absolute difference between model and reintegrated SOC: {np.max(soc_diff):.6f}')
        else:
            print(f'Max absolute difference between model and reintegrated SOC: {np.max(np.abs(soc_model - soc_reintegrated)):.6f}')

        
        # Print results
        print("\n" + "="*50)
        print("BATTERY SYSTEM RESULTS")
        print("="*50)
        
        # Get SOC values
        soc_values = prob.get_val('soc', units=None)
        print(f"SOC values: {soc_values}")
        
        # Get line voltage values
        vline_cell_values = prob.get_val('vline_cell', units='V')
        print(f"Line voltage values (V): {vline_cell_values}")
        
        # Get OCV values for comparison
        ocv_cell_values = prob.get_val('ocv_cell', units='V')
        print(f"OCV values (V): {ocv_cell_values}")
        
        # Get C-rate values
        c_rate_values = prob.get_val('c_rate', units='1/h')
        print(f"C-rate values (1/h): {c_rate_values}")
        
        # Get cell current values
        i_cell_values = prob.get_val('i_cell', units='A')
        print(f"Cell current values (A): {i_cell_values}")

        print("="*50)
        
        # Create plots showing Evolution over time
        print("\nCreating Evolution plots...")
        
        # time_min already defined above

        
        # Create figure with subplots - now 3x2 to include cell power
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 18))
        
        # Colors for each string
        colors = plt.cm.tab10(np.linspace(0, 1, n_str))
        
        # Plot 1: SOC Evolution - per string (model and reintegrated)
        for i in range(n_str):
            soc_str = soc_values[i, :] if len(soc_values.shape) > 1 else soc_values
            # Handle soc_reintegrated shape
            if len(soc_reintegrated.shape) > 1:
                soc_reintegrated_str = soc_reintegrated[i, :]
            else:
                soc_reintegrated_str = soc_reintegrated
            
            ax1.plot(time_min, soc_str, 'o-', linewidth=2, markersize=4, 
                    color=colors[i], label=f'Model SOC - String {i}', alpha=0.7)
            ax1.plot(time_min, soc_reintegrated_str, 'x--', linewidth=2, markersize=3, 
                    color=colors[i], label=f'Reintegrated SOC - String {i}', alpha=0.5, linestyle='--')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('State of Charge')
        ax1.set_title('SOC Evolution Over Time - Per String (Model vs Reintegrated)')
        ax1.legend(ncol=min(2, n_str), loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Open Circuit Voltage Evolution - per string
        for i in range(n_str):
            ocv_str = ocv_cell_values[i, :] if len(ocv_cell_values.shape) > 1 else ocv_cell_values
            ax2.plot(time_min, ocv_str, 's-', linewidth=2, markersize=4, 
                    color=colors[i], label=f'String {i}', alpha=0.7)
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Open Circuit Voltage (V)')
        ax2.set_title('OCV Evolution Over Time - Per String')
        ax2.legend(ncol=min(3, n_str), loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Line Voltage Evolution - per string
        for i in range(n_str):
            vline_str = vline_cell_values[i, :] if len(vline_cell_values.shape) > 1 else vline_cell_values
            ax3.plot(time_min, vline_str, '^-', linewidth=2, markersize=4, 
                    color=colors[i], label=f'String {i}', alpha=0.7)
        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('Line Voltage (V)')
        ax3.set_title('Line Voltage Evolution Over Time - Per String')
        ax3.legend(ncol=min(3, n_str), loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: C-rate and current - per string (overlaid)
        ax4_twin = ax4.twinx()
        for i in range(n_str):
            c_rate_str = c_rate_values[i, :] if len(c_rate_values.shape) > 1 else c_rate_values
            i_cell_str = i_cell_values[i, :] if len(i_cell_values.shape) > 1 else i_cell_values
            if i == 0:
                ax4.plot(time_min, c_rate_str, '-', linewidth=2, color=colors[i], 
                        label='C-rate', alpha=0.7)
                ax4_twin.plot(time_min, i_cell_str, '--', linewidth=2, color=colors[i], 
                             label='Cell Current', alpha=0.7)
            else:
                ax4.plot(time_min, c_rate_str, '-', linewidth=2, color=colors[i], alpha=0.7)
                ax4_twin.plot(time_min, i_cell_str, '--', linewidth=2, color=colors[i], alpha=0.7)
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('C-rate (1/h)', color='purple')
        ax4_twin.set_ylabel('Cell Current (A)', color='orange')
        ax4.set_title('C-rate and Current Over Time - Per String')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        # Plot 5: Cell Power Evolution - per string
        p_cell_values = prob.get_val('p_cell', units='W')
        for i in range(n_str):
            p_cell_str = p_cell_values[i, :] if len(p_cell_values.shape) > 1 else p_cell_values
            ax5.plot(time_min, p_cell_str, 'd-', linewidth=2, markersize=4, 
                    color=colors[i], label=f'String {i}', alpha=0.7)
        ax5.set_xlabel('Time (min)')
        ax5.set_ylabel('Cell Power (W)')
        ax5.set_title('Cell Power Evolution Over Time - Per String')
        ax5.legend(ncol=min(3, n_str), loc='best')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Battery Power Evolution (aggregate, not per string)
        p_bat_values = prob.get_val('p_bat', units='W')
        if len(p_bat_values.shape) > 1:
            # Sum across strings if it's a matrix
            p_bat_total = np.sum(p_bat_values, axis=0)
            ax6.plot(time_min, p_bat_total, 's-', linewidth=2, markersize=6, color='c', label='Total Battery Power')
        else:
            ax6.plot(time_min, p_bat_values, 's-', linewidth=2, markersize=6, color='c', label='Battery Power')
        ax6.set_xlabel('Time (min)')
        ax6.set_ylabel('Battery Power (W)')
        ax6.set_title('Battery Power Evolution Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create a combined plot showing voltage comparison - per string
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for i in range(n_str):
            ocv_str = ocv_cell_values[i, :] if len(ocv_cell_values.shape) > 1 else ocv_cell_values
            vline_str = vline_cell_values[i, :] if len(vline_cell_values.shape) > 1 else vline_cell_values
            ax.plot(time_min, ocv_str, 's-', linewidth=2, markersize=4, 
                   color=colors[i], label=f'OCV - String {i}', alpha=0.7)
            ax.plot(time_min, vline_str, '^-', linewidth=2, markersize=4, 
                   color=colors[i], linestyle='--', label=f'Line Voltage - String {i}', alpha=0.7)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title('Voltage Comparison: OCV vs Line Voltage - Per String')
        ax.legend(ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        

        
        # Print summary statistics - per string
        print("\n" + "="*50)
        print("SUMMARY STATISTICS - PER STRING")
        print("="*50)
        
        for i in range(n_str):
            print(f"\n--- Battery String {i} ---")
            soc_str = soc_values[i, :] if len(soc_values.shape) > 1 else soc_values
            ocv_str = ocv_cell_values[i, :] if len(ocv_cell_values.shape) > 1 else ocv_cell_values
            vline_str = vline_cell_values[i, :] if len(vline_cell_values.shape) > 1 else vline_cell_values
            c_rate_str = c_rate_values[i, :] if len(c_rate_values.shape) > 1 else c_rate_values
            i_cell_str = i_cell_values[i, :] if len(i_cell_values.shape) > 1 else i_cell_values
            p_cell_str = p_cell_values[i, :] if len(p_cell_values.shape) > 1 else p_cell_values
            
            print(f"Initial SOC: {soc_str[0]:.4f}")
            print(f"Final SOC: {soc_str[-1]:.4f}")
            print(f"SOC Change: {soc_str[0] - soc_str[-1]:.4f}")
            print(f"Initial OCV: {ocv_str[0]:.2f} V")
            print(f"Final OCV: {ocv_str[-1]:.2f} V")
            print(f"OCV Change: {ocv_str[0] - ocv_str[-1]:.2f} V")
            print(f"Initial Line Voltage: {vline_str[0]:.2f} V")
            print(f"Final Line Voltage: {vline_str[-1]:.2f} V")
            print(f"Line Voltage Change: {vline_str[0] - vline_str[-1]:.2f} V")
            print(f"Average C-rate: {np.mean(c_rate_str):.2f} 1/h")
            print(f"Average Current: {np.mean(i_cell_str):.2f} A")
            print(f"Average Cell Power: {np.mean(p_cell_str):.2f} W")
        
        print(f"\n--- Aggregate Statistics ---")
        p_bat_values = prob.get_val('p_bat', units='W')
        if len(p_bat_values.shape) > 1:
            p_bat_total = np.sum(p_bat_values, axis=0)
            print(f"Average Battery Power: {np.mean(p_bat_total):.2f} W")
        else:
            print(f"Average Battery Power: {np.mean(p_bat_values):.2f} W")
        print(f"Motor Voltage: ")
        print(prob['v_bat'])
        print("="*50)
        
        # Save all battery data to CSV
        print("\nSaving battery data to CSV...")
        
        # Get all values
        p_cell_vals = prob.get_val('p_cell', units='W')
        p_bat_vals = prob.get_val('p_bat', units='W')
        v_bat_vals = prob.get_val('v_bat', units='V')
        v_mot_vals = prob.get_val('v_mot', units='V')
        i_bat_vals = prob.get_val('i_bat', units='A')
        eta_cell_vals = prob.get_val('eta_cell')
        net_q_heat_ess_vals = prob.get_val('net_q_heat_ess', units='W')
        heat_ess_vals = prob.get_val('heat_ess', units='W')
        dT_dt_vals = prob.get_val('dT_dt', units='K/s')
        p_train_elec_vals = prob.get_val('p_train_elec', units='W')
        p_aux_elec_vals = prob.get_val('p_aux_elec', units='W')
        eta_parc_vals = prob.get_val('eta_parc')
        
        # Reshape data for DataFrame: create rows for each (string, time) combination
        import pandas as pd
        
        # Create lists for each column
        time_min_list = []
        time_hours_list = []
        string_idx_list = []
        time_idx_list = []
        
        soc_clipped_list = []
        ocv_cell_list = []
        vline_cell_list = []
        c_rate_list = []
        i_cell_list = []
        p_cell_list = []
        soc_reintegrated_list = []
        
        # Aggregate values (not per-string)
        p_bat_list = []
        v_bat_list = []
        v_mot_list = []
        i_bat_list = []
        eta_cell_list = []
        net_q_heat_ess_list = []
        heat_ess_list = []
        dT_dt_list = []
        p_train_elec_list = []
        p_aux_elec_list = []
        eta_parc_list = []
        
        num_nodes = len(time_vec)
        
        for str_idx in range(n_str):
            for time_idx in range(num_nodes):
                time_min_list.append(time_min[time_idx])
                time_hours_list.append(time_vec[time_idx])
                string_idx_list.append(str_idx)
                time_idx_list.append(time_idx)
                
                # Per-string values
                if len(soc_values.shape) > 1:
                    soc_clipped_list.append(soc_values[str_idx, time_idx])
                else:
                    soc_clipped_list.append(soc_values[time_idx])
                
                if len(ocv_cell_values.shape) > 1:
                    ocv_cell_list.append(ocv_cell_values[str_idx, time_idx])
                else:
                    ocv_cell_list.append(ocv_cell_values[time_idx])
                
                if len(vline_cell_values.shape) > 1:
                    vline_cell_list.append(vline_cell_values[str_idx, time_idx])
                else:
                    vline_cell_list.append(vline_cell_values[time_idx])
                
                if len(c_rate_values.shape) > 1:
                    c_rate_list.append(c_rate_values[str_idx, time_idx])
                else:
                    c_rate_list.append(c_rate_values[time_idx])
                
                if len(i_cell_values.shape) > 1:
                    i_cell_list.append(i_cell_values[str_idx, time_idx])
                else:
                    i_cell_list.append(i_cell_values[time_idx])
                
                if len(p_cell_vals.shape) > 1:
                    p_cell_list.append(p_cell_vals[str_idx, time_idx])
                else:
                    p_cell_list.append(p_cell_vals[time_idx])
                
                # Aggregate values (repeat for each string)
                if len(p_bat_vals.shape) > 1:
                    # If matrix, sum across strings for this time point
                    p_bat_list.append(np.sum(p_bat_vals[:, time_idx]))
                else:
                    p_bat_list.append(p_bat_vals[time_idx])
                
                # Handle motor/power variables (may be matrices)
                if len(v_bat_vals.shape) > 1:
                    v_bat_list.append(v_bat_vals[0, time_idx] if v_bat_vals.shape[0] > 0 else v_bat_vals[time_idx])
                else:
                    v_bat_list.append(v_bat_vals[time_idx])
                
                if len(v_mot_vals.shape) > 1:
                    v_mot_list.append(v_mot_vals[0, time_idx] if v_mot_vals.shape[0] > 0 else v_mot_vals[time_idx])
                else:
                    v_mot_list.append(v_mot_vals[time_idx])
                
                if len(i_bat_vals.shape) > 1:
                    i_bat_list.append(i_bat_vals[0, time_idx] if i_bat_vals.shape[0] > 0 else i_bat_vals[time_idx])
                else:
                    i_bat_list.append(i_bat_vals[time_idx])
                
                if len(eta_cell_vals.shape) > 1:
                    eta_cell_list.append(eta_cell_vals[str_idx, time_idx])
                else:
                    eta_cell_list.append(eta_cell_vals[time_idx])
                
                if len(net_q_heat_ess_vals.shape) > 1:
                    net_q_heat_ess_list.append(net_q_heat_ess_vals[str_idx, time_idx])
                else:
                    net_q_heat_ess_list.append(net_q_heat_ess_vals[time_idx])
                
                if len(heat_ess_vals.shape) > 1:
                    heat_ess_list.append(heat_ess_vals[str_idx, time_idx])
                else:
                    heat_ess_list.append(heat_ess_vals[time_idx])
                
                if len(dT_dt_vals.shape) > 1:
                    dT_dt_list.append(dT_dt_vals[str_idx, time_idx])
                else:
                    dT_dt_list.append(dT_dt_vals[time_idx])
                
                if len(p_train_elec_vals.shape) > 1:
                    # Sum across motors
                    p_train_elec_list.append(np.sum(p_train_elec_vals[:, time_idx]))
                else:
                    p_train_elec_list.append(p_train_elec_vals[time_idx])
                
                if len(p_aux_elec_vals.shape) > 1:
                    p_aux_elec_list.append(p_aux_elec_vals[0, time_idx])
                else:
                    p_aux_elec_list.append(p_aux_elec_vals[time_idx])
                
                if len(eta_parc_vals.shape) > 1:
                    eta_parc_list.append(eta_parc_vals[0, time_idx])
                else:
                    eta_parc_list.append(eta_parc_vals[time_idx])
                
                # soc_reintegrated
                if len(soc_reintegrated.shape) > 1:
                    soc_reintegrated_list.append(soc_reintegrated[str_idx, time_idx])
                else:
                    soc_reintegrated_list.append(soc_reintegrated[time_idx])
        
        # Create DataFrame from lists
        battery_data = {
            'time_min': time_min_list,
            'time_hours': time_hours_list,
            'string_idx': string_idx_list,
            'time_idx': time_idx_list,
            'soc_clipped': soc_clipped_list,
            'ocv_cell': ocv_cell_list,
            'vline_cell': vline_cell_list,
            'c_rate': c_rate_list,
            'i_cell': i_cell_list,
            'p_cell': p_cell_list,
            'p_bat': p_bat_list,
            'v_bat': v_bat_list,
            'v_mot': v_mot_list,
            'i_bat': i_bat_list,
            'eta_cell': eta_cell_list,
            'net_q_heat_ess': net_q_heat_ess_list,
            'heat_ess': heat_ess_list,
            'dT_dt': dT_dt_list,
            'p_train_elec': p_train_elec_list,
            'p_aux_elec': p_aux_elec_list,
            'eta_parc': eta_parc_list,
            'soc_reintegrated': soc_reintegrated_list
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(battery_data)
        
        # Generate date-based results directory and timestamped filename
        from datetime import datetime

        repo_root = Path(__file__).resolve().parents[5]  # aircraft_performance
        results_base = repo_root.parent / "aircraft_results"
        today_folder = datetime.now().strftime("%d_%m_%Y")
        results_dir = results_base / today_folder
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filepath = results_dir / f"battery_data_{timestamp}.csv"

        # Save to CSV
        df.to_csv(csv_filepath, index=False)
        #print(f"Battery data saved to: {csv_filename}")
        #print(f"Data shape: {df.shape}")
        #print(f"Columns: {list(df.columns)}")

    


if __name__ == "__main__":

    test_battery_component(battery_datasheet_name='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105')
