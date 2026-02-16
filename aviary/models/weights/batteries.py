import openmdao.api as om
import numpy as np


class ESSWeight(om.ExplicitComponent):
    """
    OpenMDAO component for ESS (battery pack) weight calculation.
    
    Computes total ESS mass from battery cell configuration and packaging factor.
    Uses nomenclature consistent with battery_empirical_power.py.
    
    Inputs
    ------
    n_str : float
        Number of battery strings in the pack
    n_parallel_per_str : float
        Number of cells in parallel per string
    n_series_per_str : float
        Number of cells in series per string
    m_cell : float
        Mass of a single battery cell (kg)
    
    Outputs
    -------
    total_battery_weight : float
        Total ESS mass including packaging (kg)
    n_cells_total : float
        Total number of cells in the pack
    cell_mass_total : float
        Total mass of cells only, before packaging factor (kg)
    
    Options
    -------
    pack_scale_factor : float
        Scale factor to account for packaging, BMS, cooling, structure, etc.
        Typical values: 1.2 - 1.5 (default 1.3)
    """
    
    def initialize(self):
        self.options.declare('pack_scale_factor', default=1.247, 
                            desc='Scale factor from cell mass to pack mass (accounts for BMS, cooling, structure)')
        self.options.declare('cg_batteries_inboard_percentage', default=80.2, types=float,
                           desc='Batteries Inboard C.G location as percentage of MAC')
        self.options.declare('cg_batteries_outboard_percentage', default=91.4, types=float,
                           desc='Batteries Outboard C.G location as percentage of MAC')
        self.options.declare('cell_capacity', default=7.4*3.7, types=float,
                           desc='Energy of a single cell (W*h)')

    def setup(self):
        # Inputs - using nomenclature from battery_empirical_power.py
        self.add_input('n_str', val=1.0, desc='Number of battery strings')
        self.add_input('n_parallel_per_str', val=1.0, desc='Number of cells in parallel per string')
        self.add_input('n_series_per_str', val=1.0, desc='Number of cells in series per string')
        self.add_input('m_cell', val=0.075, units='kg', desc='Mass of single cell')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('cg_batteries', val=0.0, units='inch', desc='Batteries total C.G location (weighted average of inboard and outboard)')
        self.add_output('total_battery_weight', val=1.0, units='kg', desc='Total battery weight including packaging')
        self.add_output('n_cells_total', val=1.0, desc='Total number of cells')
        self.add_output('cell_mass_total', val=1.0, units='kg', desc='Total cell mass before packaging')
        self.add_output('batt_energy', val=0.0, units='W*h', desc='total Battery energy capacity')
        
        # Declare partials analytically
        self.declare_partials('n_cells_total', ['n_str', 'n_parallel_per_str', 'n_series_per_str'])
        self.declare_partials('cell_mass_total', ['n_str', 'n_parallel_per_str', 'n_series_per_str', 'm_cell'])
        self.declare_partials('total_battery_weight', ['n_str', 'n_parallel_per_str', 'n_series_per_str', 'm_cell'])
        self.declare_partials('cg_batteries', ['MAC', 'LEMAC'])
        self.declare_partials('batt_energy', ['n_str', 'n_parallel_per_str', 'n_series_per_str'])

    def compute(self, inputs, outputs):
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        cell_capacity = self.options['cell_capacity']
        
        # Total cell count = strings * (cells in parallel per string * cells in series per string)
        n_cells_total = n_str * n_parallel_per_str * n_series_per_str
        
        # Total cell mass
        cell_mass_total = n_cells_total * m_cell
        
        # ESS mass with packaging factor
        total_battery_weight = cell_mass_total * pack_scale_factor
        
        outputs['n_cells_total'] = n_cells_total
        outputs['cell_mass_total'] = cell_mass_total
        outputs['total_battery_weight'] = total_battery_weight

        # Split weight in half (inboard and outboard)
        weight_inboard = total_battery_weight / 2.0
        weight_outboard = total_battery_weight / 2.0
        
        # Calculate C.G for each half
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_batteries_inboard_percentage']
        cg_outboard_pct = self.options['cg_batteries_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)

        outputs['batt_energy'] = n_cells_total * cell_capacity
        
        # Calculate weighted average total C.G
        outputs['cg_batteries'] = (weight_inboard * cg_inboard + weight_outboard * cg_outboard) / total_battery_weight
    
    def compute_partials(self, inputs, partials):
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        cell_capacity = self.options['cell_capacity']
        
        # Partials for n_cells_total = n_str * n_parallel_per_str * n_series_per_str
        partials['n_cells_total', 'n_str'] = n_parallel_per_str * n_series_per_str
        partials['n_cells_total', 'n_parallel_per_str'] = n_str * n_series_per_str
        partials['n_cells_total', 'n_series_per_str'] = n_str * n_parallel_per_str
        
        # cell_mass_total = n_cells_total * m_cell
        # cell_mass_total = n_str * n_parallel_per_str * n_series_per_str * m_cell
        partials['cell_mass_total', 'n_str'] = n_parallel_per_str * n_series_per_str * m_cell
        partials['cell_mass_total', 'n_parallel_per_str'] = n_str * n_series_per_str * m_cell
        partials['cell_mass_total', 'n_series_per_str'] = n_str * n_parallel_per_str * m_cell
        partials['cell_mass_total', 'm_cell'] = n_str * n_parallel_per_str * n_series_per_str
        
        # total_battery_weight = cell_mass_total * pack_scale_factor
        # total_battery_weight = n_str * n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        partials['total_battery_weight', 'n_str'] = n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        partials['total_battery_weight', 'n_parallel_per_str'] = n_str * n_series_per_str * m_cell * pack_scale_factor
        partials['total_battery_weight', 'n_series_per_str'] = n_str * n_parallel_per_str * m_cell * pack_scale_factor
        partials['total_battery_weight', 'm_cell'] = n_str * n_parallel_per_str * n_series_per_str * pack_scale_factor

        # Partials for C.G
        # Since weight_inboard = weight_outboard = total_weight/2, cg_total = (cg_in + cg_out) / 2
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_batteries_inboard_percentage']
        cg_outboard_pct = self.options['cg_batteries_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_batteries', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 0.5 * (1 + 1) = 1.0
        partials['cg_batteries', 'LEMAC'] = 1.0

        # batt_energy = n_str * n_parallel_per_str * n_series_per_str * cell_capacity
        partials['batt_energy', 'n_str'] = n_parallel_per_str * n_series_per_str * cell_capacity
        partials['batt_energy', 'n_parallel_per_str'] = n_str * n_series_per_str * cell_capacity
        partials['batt_energy', 'n_series_per_str'] = n_str * n_parallel_per_str * cell_capacity


class BatteryWeight(om.ExplicitComponent):
    """
    Calculates battery weight based on energy and energy density.
    
    Weight = Energy / Energy Density
    
    where:
    - Energy is in Wh (Watt-hours)
    - Energy Density is in Wh/lbm (Watt-hours per pound mass)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('cg_batteries_inboard_percentage', default=80.2, types=float,
                           desc='Batteries Inboard C.G location as percentage of MAC')
        self.options.declare('cg_batteries_outboard_percentage', default=91.4, types=float,
                           desc='Batteries Outboard C.G location as percentage of MAC')

    def setup(self):
        self.add_input('batt_energy', val=3000000.0, units='W*h', desc='Battery energy capacity')
        self.add_input('batt_density', val=136.08, units='W*h/lbm', desc='Battery energy density')
        self.add_input('MAC', val=106.43, units='inch', desc='Mean Aerodynamic Chord')
        self.add_input('LEMAC', val=949.11, units='inch', desc='Leading Edge Mean Aerodynamic Chord')
        
        self.add_output('total_battery_weight', val=0.0, units='lbm', desc='Battery weight')
        self.add_output('cg_batteries', val=0.0, units='inch', desc='Batteries total C.G location (weighted average of inboard and outboard)')
        
        # Declare partials - only for weight and C.G outputs
        self.declare_partials('total_battery_weight', ['batt_energy', 'batt_density'])
        self.declare_partials('cg_batteries', ['MAC', 'LEMAC'])

    def compute(self, inputs, outputs):
        # Weight = Energy / Energy Density
        total_weight = inputs['batt_energy'] / inputs['batt_density']
        outputs['total_battery_weight'] = total_weight
        
        # Split weight in half (inboard and outboard)
        weight_inboard = total_weight / 2.0
        weight_outboard = total_weight / 2.0
        
        # Calculate C.G for each half
        # C.G = LEMAC + (MAC * percentage / 100)
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_batteries_inboard_percentage']
        cg_outboard_pct = self.options['cg_batteries_outboard_percentage']
        
        cg_inboard = LEMAC + (MAC * cg_inboard_pct / 100.0)
        cg_outboard = LEMAC + (MAC * cg_outboard_pct / 100.0)
        
        # Calculate weighted average total C.G
        outputs['cg_batteries'] = (weight_inboard * cg_inboard + weight_outboard * cg_outboard) / total_weight

    def compute_partials(self, inputs, partials):
        # Partial w.r.t. batt_energy
        partials['total_battery_weight', 'batt_energy'] = 1.0 / inputs['batt_density']
        
        # Partial w.r.t. batt_density
        partials['total_battery_weight', 'batt_density'] = -inputs['batt_energy'] / (inputs['batt_density'] ** 2)
        
        # Partials for C.G
        # Since weight_inboard = weight_outboard = total_weight/2, cg_total = (cg_in + cg_out) / 2
        MAC = inputs['MAC']
        LEMAC = inputs['LEMAC']
        cg_inboard_pct = self.options['cg_batteries_inboard_percentage']
        cg_outboard_pct = self.options['cg_batteries_outboard_percentage']
        
        # d(cg_total)/d(MAC) = 0.5 * (percentage_in + percentage_out) / 100
        partials['cg_batteries', 'MAC'] = 0.5 * (cg_inboard_pct + cg_outboard_pct) / 100.0
        
        # d(cg_total)/d(LEMAC) = 0.5 * (1 + 1) = 1.0
        partials['cg_batteries', 'LEMAC'] = 1.0


if __name__ == "__main__":

    # Battery Weight
    batt_prob = om.Problem(reports=False)
    model = batt_prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('batt_energy', val=3000000.0, units='W*h')
    ivc.add_output('batt_density', val=136.08, units='W*h/lbm')

    model.add_subsystem('battery', BatteryWeight(), promotes=['*'])

    batt_prob.setup()
    batt_prob.run_model()

    print('Battery Weight:', batt_prob.get_val('total_battery_weight', units='lbm'))
    batt_prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)

    # ESS Weight
    ess_prob = om.Problem(reports=False)
    model = ess_prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('n_str', val=4.0)
    ivc.add_output('n_parallel_per_str', val=112.0)
    ivc.add_output('n_series_per_str', val=210.0)
    ivc.add_output('m_cell', val=0.075, units='kg')
    
    model.add_subsystem('ess', ESSWeight(), promotes=['*'])
    ess_prob.setup()
    ess_prob.run_model()

    print('ESS Weight:', ess_prob.get_val('total_battery_weight', units='lbm'))
    #ess_prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)