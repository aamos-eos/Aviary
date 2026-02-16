"""
OpenMDAO ExplicitComponent for Energy Storage System (ESS) mass calculation.

This component computes the total ESS mass based on battery configuration
(strings, cells in parallel, cells in series) and applies a packaging scale factor.

Nomenclature matches battery_empirical_power.py:
- n_str: number of battery strings
- n_series_per_str: number of cells in series per string
- n_parallel_per_str: number of cells in parallel per string
- m_cell: mass of the cell (kg)
"""

import numpy as np
import openmdao.api as om


class ESSWeightComp(om.ExplicitComponent):
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
    ess_mass : float
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
    
    def setup(self):
        # Inputs - using nomenclature from battery_empirical_power.py
        self.add_input('n_str', val=1.0, desc='Number of battery strings')
        self.add_input('n_parallel_per_str', val=1.0, desc='Number of cells in parallel per string')
        self.add_input('n_series_per_str', val=1.0, desc='Number of cells in series per string')
        self.add_input('m_cell', val=0.07, units='kg', desc='Mass of single cell')
        
        # Outputs
        self.add_output('ess_mass', val=1.0, units='kg', desc='Total ESS mass with packaging')
        self.add_output('n_cells_total', val=1.0, desc='Total number of cells')
        self.add_output('cell_mass_total', val=1.0, units='kg', desc='Total cell mass before packaging')
        
        # Declare partials analytically
        self.declare_partials('n_cells_total', ['n_str', 'n_parallel_per_str', 'n_series_per_str'])
        self.declare_partials('cell_mass_total', ['n_str', 'n_parallel_per_str', 'n_series_per_str', 'm_cell'])
        self.declare_partials('ess_mass', ['n_str', 'n_parallel_per_str', 'n_series_per_str', 'm_cell'])
    
    def compute(self, inputs, outputs):
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        
        # Total cell count = strings * (cells in parallel per string * cells in series per string)
        n_cells_total = n_str * n_parallel_per_str * n_series_per_str
        
        # Total cell mass
        cell_mass_total = n_cells_total * m_cell
        
        # ESS mass with packaging factor
        ess_mass = cell_mass_total * pack_scale_factor
        
        outputs['n_cells_total'] = n_cells_total
        outputs['cell_mass_total'] = cell_mass_total
        outputs['ess_mass'] = ess_mass
    
    def compute_partials(self, inputs, partials):
        n_str = inputs['n_str']
        n_parallel_per_str = inputs['n_parallel_per_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        
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
        
        # ess_mass = cell_mass_total * pack_scale_factor
        # ess_mass = n_str * n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'n_str'] = n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'n_parallel_per_str'] = n_str * n_series_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'n_series_per_str'] = n_str * n_parallel_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'm_cell'] = n_str * n_parallel_per_str * n_series_per_str * pack_scale_factor


class ESSParallelFromMassComp(om.ExplicitComponent):
    """
    OpenMDAO component to determine the number of cells in parallel per string
    from a target ESS mass allocation.
    
    Given a target ESS mass, number of strings, cells in series, and cell mass,
    this component calculates the required number of cells in parallel per string.
    The result is ROUNDED DOWN (floor) to ensure we don't exceed the target mass allocation.
    
    Calculation:
        ess_mass = n_str * n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        
        Solving for n_parallel_per_str:
        n_parallel_per_str_raw = ess_mass_target / (n_str * n_series_per_str * m_cell * pack_scale_factor)
        n_parallel_per_str = floor(n_parallel_per_str_raw)
        
        The actual ess_mass is then recalculated from the rounded value.
    
    Inputs
    ------
    ess_mass_target : float
        Target ESS mass allocation (kg)
    n_str : float
        Number of battery strings in the pack
    n_series_per_str : float
        Number of cells in series per string
    m_cell : float
        Mass of a single battery cell (kg)
    
    Outputs
    -------
    n_parallel_per_str : float
        Number of cells in parallel per string (rounded to nearest integer)
    n_parallel_per_str_raw : float
        Unrounded number of cells in parallel (for optimization)
    n_cells_total : float
        Total number of cells in the pack (based on rounded n_parallel)
    cell_mass_total : float
        Total mass of cells only, before packaging factor (kg)
    ess_mass : float
        Actual ESS mass (based on rounded n_parallel, may differ from target)
    
    Options
    -------
    pack_scale_factor : float
        Scale factor to account for packaging, BMS, cooling, structure, etc.
        Typical values: 1.2 - 1.5 (default 1.3)
    
    Notes
    -----
    For gradient-based optimization, the partials are computed based on the 
    unrounded (continuous) value. The rounding introduces a discontinuity that
    cannot be captured by smooth derivatives. For optimization purposes, consider
    using n_parallel_per_str_raw and applying rounding as a post-processing step.
    """
    
    def initialize(self):
        self.options.declare('pack_scale_factor', default=1.247, 
                            desc='Scale factor from cell mass to pack mass (accounts for BMS, cooling, structure)')
    
    def setup(self):
        # Inputs
        self.add_input('ess_mass_target', val=100.0, units='kg', desc='Target ESS mass allocation')
        self.add_input('n_str', val=1.0, desc='Number of battery strings')
        self.add_input('n_series_per_str', val=1.0, desc='Number of cells in series per string')
        self.add_input('m_cell', val=0.07, units='kg', desc='Mass of single cell')
        
        # Outputs
        self.add_output('n_parallel_per_str', val=1.0, desc='Number of cells in parallel per string (rounded)', lower=1.0)
        self.add_output('n_cells_total', val=1.0, desc='Total number of cells (based on rounded n_parallel)')
        self.add_output('ess_mass', val=1.0, units='kg', desc='Actual ESS mass (based on rounded n_parallel)')
        
        # Declare partials analytically
        # n_parallel_per_str = floor(n_parallel_per_str_raw)
        # For optimization, use smooth derivative of unrounded value (approximation)
        self.declare_partials('n_parallel_per_str', ['ess_mass_target', 'n_str', 'n_series_per_str', 'm_cell'])
        # n_cells_total, cell_mass_total, ess_mass use the floored n_parallel_per_str
        self.declare_partials('n_cells_total', ['n_str', 'n_series_per_str', 'm_cell'])
        self.declare_partials('ess_mass', ['n_str', 'n_series_per_str', 'm_cell'])
    
    def compute(self, inputs, outputs):
        ess_mass_target = inputs['ess_mass_target']
        n_str = inputs['n_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        
        # Calculate n_parallel_per_str from target mass (unrounded)
        # ess_mass = n_str * n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        # n_parallel_per_str = ess_mass / (n_str * n_series_per_str * m_cell * pack_scale_factor)
        denominator = n_str * n_series_per_str * m_cell * pack_scale_factor
        n_parallel_per_str_raw = ess_mass_target / denominator
        
        # Round DOWN to integer (can't have fractional cells, and we don't want to exceed target mass)
        n_parallel_per_str = np.trunc(n_parallel_per_str_raw)
        # Ensure at least 1 cell in parallel
        n_parallel_per_str = np.maximum(n_parallel_per_str, 1.0)
        
        # Calculate derived quantities based on ROUNDED n_parallel
        n_cells_total = n_str * n_parallel_per_str * n_series_per_str
        cell_mass_total = n_cells_total * m_cell
        ess_mass = cell_mass_total * pack_scale_factor
        
        outputs['n_parallel_per_str'] = n_parallel_per_str
        outputs['n_cells_total'] = n_cells_total
        outputs['ess_mass'] = ess_mass
    
    def compute_partials(self, inputs, partials):
        ess_mass_target = inputs['ess_mass_target']
        n_str = inputs['n_str']
        n_series_per_str = inputs['n_series_per_str']
        m_cell = inputs['m_cell']
        pack_scale_factor = self.options['pack_scale_factor']
        
        # n_parallel_per_str_raw = ess_mass_target / (n_str * n_series_per_str * m_cell * pack_scale_factor)
        denominator = n_str * n_series_per_str * m_cell * pack_scale_factor
        
        # Compute floored n_parallel for use in other partials
        n_parallel_per_str_raw = ess_mass_target / denominator
        n_parallel_per_str = np.trunc(n_parallel_per_str_raw)
        n_parallel_per_str = np.maximum(n_parallel_per_str, 1.0)
        
        # Partials for n_parallel_per_str
        # Use smooth derivative of unrounded value for optimization (approximation of floor)
        # ∂n_parallel_per_str/∂ess_mass_target = 1 / denominator
        partials['n_parallel_per_str', 'ess_mass_target'] = 1.0 / denominator
        # ∂n_parallel_per_str/∂n_str = -ess_mass_target / (n_str^2 * n_series_per_str * m_cell * pack_scale_factor)
        partials['n_parallel_per_str', 'n_str'] = -ess_mass_target / (n_str**2 * n_series_per_str * m_cell * pack_scale_factor)
        # ∂n_parallel_per_str/∂n_series_per_str = -ess_mass_target / (n_str * n_series_per_str^2 * m_cell * pack_scale_factor)
        partials['n_parallel_per_str', 'n_series_per_str'] = -ess_mass_target / (n_str * n_series_per_str**2 * m_cell * pack_scale_factor)
        # ∂n_parallel_per_str/∂m_cell = -ess_mass_target / (n_str * n_series_per_str * m_cell^2 * pack_scale_factor)
        partials['n_parallel_per_str', 'm_cell'] = -ess_mass_target / (n_str * n_series_per_str * m_cell**2 * pack_scale_factor)
        
        # n_cells_total = n_str * n_parallel_per_str * n_series_per_str
        # where n_parallel_per_str is floored (treated as constant w.r.t. ess_mass_target)
        partials['n_cells_total', 'n_str'] = n_parallel_per_str * n_series_per_str
        partials['n_cells_total', 'n_series_per_str'] = n_str * n_parallel_per_str
        partials['n_cells_total', 'm_cell'] = 0.0  # n_cells_total doesn't depend on m_cell directly
        
        # cell_mass_total = n_cells_total * m_cell = n_str * n_parallel_per_str * n_series_per_str * m_cell
        n_cells_total = n_str * n_parallel_per_str * n_series_per_str

        # ess_mass = cell_mass_total * pack_scale_factor
        partials['ess_mass', 'n_str'] = n_parallel_per_str * n_series_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'n_series_per_str'] = n_str * n_parallel_per_str * m_cell * pack_scale_factor
        partials['ess_mass', 'm_cell'] = n_cells_total * pack_scale_factor


def test_ess_weight_component():
    """Test the ESSWeightComp with example values."""
    
    prob = om.Problem()
    
    ivc = om.IndepVarComp()
    # Example: 4 strings, 27 cells in parallel, 208 cells in series, 70g per cell
    # (matching typical battery config from battery_empirical_power.py)
    ivc.add_output('n_str', val=4.0)
    ivc.add_output('n_parallel_per_str', val=27.0)
    ivc.add_output('n_series_per_str', val=208.0)
    ivc.add_output('m_cell', val=0.070, units='kg')  # 70 grams per cell
    
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('ess_weight', ESSWeightComp(pack_scale_factor=1.247), promotes=['*'])
    
    prob.setup()
    prob.run_model()
    
    print("=== ESS Weight Component Test Results ===")
    print(f"Configuration:")
    print(f"  n_str:                 {prob.get_val('n_str')[0]:.0f}")
    print(f"  n_parallel_per_str:    {prob.get_val('n_parallel_per_str')[0]:.0f}")
    print(f"  n_series_per_str:      {prob.get_val('n_series_per_str')[0]:.0f}")
    print(f"  m_cell:                {prob.get_val('m_cell', units='kg')[0]*1000:.1f} g")
    print(f"  Pack Scale Factor:     1.3")
    print()
    print(f"Results:")
    print(f"  n_cells_total:         {prob.get_val('n_cells_total')[0]:.0f} cells")
    print(f"  cell_mass_total:       {prob.get_val('cell_mass_total', units='kg')[0]:.2f} kg")
    print(f"  ess_mass:              {prob.get_val('ess_mass', units='kg')[0]:.2f} kg")
    
    # Check partials
    print("\n=== Checking Partials ===")
    prob.check_partials(compact_print=True,show_only_incorrect=True)
    
    return prob


def test_ess_parallel_from_mass_component():
    """Test the ESSParallelFromMassComp with example values."""
    
    prob = om.Problem()
    
    ivc = om.IndepVarComp()
    # Target 2000 kg ESS mass, with 4 strings, 208 cells in series, 70g per cell
    ivc.add_output('ess_mass_target', val=2000.0, units='kg')
    ivc.add_output('n_str', val=4.0)
    ivc.add_output('n_series_per_str', val=208.0)
    ivc.add_output('m_cell', val=0.070, units='kg')  # 70 grams per cell
    
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('ess_sizing', ESSParallelFromMassComp(pack_scale_factor=1.247), promotes=['*'])
    
    prob.setup()
    prob.check_partials(compact_print=True,show_only_incorrect=True)
    prob.run_model()
    
    print("=== ESS Parallel From Mass Component Test Results ===")
    print(f"Inputs:")
    print(f"  ess_mass_target:       {prob.get_val('ess_mass_target', units='kg')[0]:.2f} kg")
    print(f"  n_str:                 {prob.get_val('n_str')[0]:.0f}")
    print(f"  n_series_per_str:      {prob.get_val('n_series_per_str')[0]:.0f}")
    print(f"  m_cell:                {prob.get_val('m_cell', units='kg')[0]*1000:.1f} g")
    print(f"  Pack Scale Factor:     1.3")
    print()
    print(f"Calculated Outputs:")
    print(f"  n_parallel_per_str:    {prob.get_val('n_parallel_per_str')[0]:.0f} (rounded)")
    print(f"  n_cells_total:         {prob.get_val('n_cells_total')[0]:.0f} cells")
    print(f"  ess_mass:              {prob.get_val('ess_mass', units='kg')[0]:.2f} kg")
    
    # Verify calculation
    n_parallel = prob.get_val('n_parallel_per_str')[0]
    n_str = prob.get_val('n_str')[0]
    n_series = prob.get_val('n_series_per_str')[0]
    m_cell = prob.get_val('m_cell')[0]
    pack_scale = 1.3
    ess_mass_target = prob.get_val('ess_mass_target')[0]
    
    # Show the difference due to rounding
    mass_with_rounding = n_str * n_parallel * n_series * m_cell * pack_scale
    
    print(f"\n  Verification:")
    print(f"    Target ESS mass:     {ess_mass_target:.2f} kg")
    print(f"    Mass with rounding:  {mass_with_rounding:.2f} kg")
    print(f"    Difference from target: {mass_with_rounding - ess_mass_target:.2f} kg ({(mass_with_rounding - ess_mass_target)/ess_mass_target*100:.2f}%)")
    
    # Check partials
    print("\n=== Checking Partials ===")
    prob.check_partials(compact_print=True,show_only_incorrect=True)
    
    return prob


if __name__ == "__main__":
    print("Testing ESSWeightComp...")
    test_ess_weight_component()
    
    print("\n" + "="*60 + "\n")
    
    print("Testing ESSParallelFromMassComp...")
    test_ess_parallel_from_mass_component()
