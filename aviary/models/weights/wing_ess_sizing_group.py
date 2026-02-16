"""
OpenMDAO Group that combines Wing Weight calculation with ESS sizing.

This group:
1. Calculates wing weight using WingWeightComp (NASA FLOPS methodology)
2. Computes ESS mass target as: ess_mass_target = total_propulsion_mass - wing_weight
3. Sizes the battery (n_parallel_per_str) based on the available ESS mass target

The idea is that the total available propulsion mass budget is fixed (e.g., 26549 kg),
and the ESS can use whatever mass is left after accounting for the wing weight.
"""

import numpy as np
import openmdao.api as om

from aviary.models.weights.ess_weight_component import ESSParallelFromMassComp
from aviary.utils.math_components.add_subtract_comp import AddSubtractComp
from aviary.models.engines.propulsion.battery.battery_data import BatteryData


class WingESSSizingGroup(om.Group):
    """
    OpenMDAO Group that combines wing weight calculation with ESS sizing.
    
    This group computes:
    1. Wing weight from aircraft parameters (using FLOPS methodology)
    2. ESS mass target = total_propulsion_mass_budget - wing_weight
    3. Number of cells in parallel based on the ESS mass target
    
    Inputs (promoted)
    -----------------
    Wing inputs (to WingWeightComp, names match load_ac_data.py):
        S_ref : float
            Wing planform area (m^2)
        mtow : float
            Maximum takeoff weight (kg)
        span : float
            Wingspan (m)
        sweep : float
            Wing sweep angle (degrees)
        tc : float
            Wing thickness-to-chord ratio
        taper : float
            Wing taper ratio (tip chord / root chord)
        AR : float
            Wing aspect ratio
        nult : float
            Design ultimate load factor (default 3.75)
        num_engines : float
            Number of wing-mounted engines
        outboard_nacelle_mass : float
            Mass of outboard nacelle (kg)
        inboard_nacelle_mass : float
            Mass of inboard nacelle (kg)
    
    ESS sizing inputs (to ESSParallelFromMassComp):
        total_propulsion_mass_budget : float
            Total mass budget for propulsion system (kg)
            ESS target = total_propulsion_mass_budget - wing_weight
        n_str : float
            Number of battery strings
        n_series_per_str : float
            Number of cells in series per string
        m_cell : float
            Mass of a single battery cell (kg)
    
    Outputs (promoted)
    ------------------
    From WingWeightComp:
        wing_weight : float
            Total wing weight (kg)
        wing_bending_mass : float
            Wing bending material mass (kg)
        wing_shear_control_mass : float
            Wing shear and control surface mass (kg)
        wing_misc_mass : float
            Wing miscellaneous mass (kg)
    
    From AddSubtractComp:
        ess_mass_target : float
            Target ESS mass allocation (kg)
    
    From ESSParallelFromMassComp:
        n_parallel_per_str : float
            Number of cells in parallel per string (rounded)
        n_parallel_per_str_raw : float
            Unrounded number of cells in parallel
        n_cells_total : float
            Total number of cells in the pack
        cell_mass_total : float
            Total mass of cells only (kg)
        ess_mass : float
            Actual ESS mass based on rounded n_parallel (kg)
    
    From OEW update calculation:
        ess_mass_delta : float
            Delta between ESS target and actual mass (kg)
            = ess_mass_target - ess_mass (positive when rounding down)
        oew_updated : float
            Updated OEW after accounting for unused ESS mass (kg)
            = oew_ref - ess_mass_delta
    
    Options
    -------
    total_propulsion_mass_budget : float
        Default total mass budget for propulsion system (kg). Default is 26549.0
    pack_scale_factor : float
        Scale factor for ESS packaging. Default is 1.23
    oew_ref : float
        Reference OEW value (kg). Default is 63500.0
    
    Wing weight options (passed to WingWeightComp):
        comp_frac : float
            Composite utilization factor (0 = none, 1 = full composite), default 0
        fstrt : float
            Strut bracing factor, default 0
        faert : float
            Aeroelastic tailoring factor, default 0
        num_fuse : int
            Number of fuselages, default 1
        load_distribution_factor : int
            Load distribution type (1=triangular, 2=elliptical, 3=rectangular), default 2
        num_integration_stations : int
            Number of integration stations along span, default 50
        total_mass_scaler : float
            Total mass scaling factor, default 1.25
    """
    
    def initialize(self):
        # ESS options
        self.options.declare('total_propulsion_mass_budget', default=26549.0, 
                            desc='Total mass budget for propulsion system (kg)')
        self.options.declare('pack_scale_factor', default=1.247, 
                            desc='Scale factor from cell mass to pack mass')
        self.options.declare('oew_ref', default=63500.0, 
                            desc='Reference OEW value (kg)')
        
        # Wing weight options (passed through to WingWeightComp)
        self.options.declare('comp_frac', default=0.0, desc='Composite utilization factor')
        self.options.declare('fstrt', default=0.0, desc='Strut bracing factor')
        self.options.declare('faert', default=0.0, desc='Aeroelastic tailoring factor')
        self.options.declare('num_fuse', default=1, desc='Number of fuselages')
        self.options.declare('load_distribution_factor', default=2, desc='Load distribution type')
        self.options.declare('num_integration_stations', default=50, desc='Number of integration stations')
        self.options.declare('total_mass_scaler', default=1.25, desc='Total mass scaling factor')
        self.options.declare('scaler_ShearCtrlMass', default=1.0, desc='Shear control mass scaling factor')
        self.options.declare('scaler_MiscMass', default=1.0, desc='Misc mass scaling factor')
        self.options.declare('scaler_BendingMass', default=1.0, desc='Bending mass scaling factor')
    
    def setup(self):
        # Get options
        total_propulsion_mass_budget = self.options['total_propulsion_mass_budget']
        pack_scale_factor = self.options['pack_scale_factor']
        
        # Wing weight options
        wing_options = {
            'comp_frac': self.options['comp_frac'],
            'fstrt': self.options['fstrt'],
            'faert': self.options['faert'],
            'num_fuse': self.options['num_fuse'],
            'load_distribution_factor': self.options['load_distribution_factor'],
            'num_integration_stations': self.options['num_integration_stations'],
            'total_mass_scaler': self.options['total_mass_scaler'],
            'scaler_ShearCtrlMass': self.options['scaler_ShearCtrlMass'],
            'scaler_MiscMass': self.options['scaler_MiscMass'],
            'scaler_BendingMass': self.options['scaler_BendingMass'],
        }
        
        # 1. Add WingWeightComp to calculate wing weight
        self.add_subsystem('wing_weight_comp', 
                          WingWeightComp(**wing_options),
                          promotes_inputs=['S_ref', 'mtow', 'span', 'sweep', 
                                          'tc', 'taper', 'AR', 'nult',
                                          'num_engines', 'outboard_nacelle_mass', 'inboard_nacelle_mass'],
                          promotes_outputs=['wing_weight', 'wing_bending_mass', 
                                           'wing_shear_control_mass', 'wing_misc_mass'])
        
        # 2. Add AddSubtractComp to compute: ess_mass_target = total_propulsion_mass_budget - wing_weight
        # Using scaling_factors [1, -1] for subtraction
        ess_target_calc = AddSubtractComp(
            output_name='ess_mass_target',
            input_names=['total_propulsion_mass_budget', 'wing_weight'],
            scaling_factors=[1.0, -1.0],
            units='kg',
            desc='Target ESS mass allocation (total budget minus wing weight)'
        )
        self.add_subsystem('ess_target_calc', ess_target_calc,
                          promotes_inputs=['total_propulsion_mass_budget'],
                          promotes_outputs=['ess_mass_target'])
        
        # Connect wing_weight output to the AddSubtractComp input
        self.connect('wing_weight', 'ess_target_calc.wing_weight')
        
        # 3. Add ESSParallelFromMassComp to size the battery
        self.add_subsystem('ess_sizing', 
                          ESSParallelFromMassComp(pack_scale_factor=pack_scale_factor),
                          promotes_inputs=['n_str', 'n_series_per_str', 'm_cell'],
                          promotes_outputs=['n_parallel_per_str',
                                           'n_cells_total', 'ess_mass'])
        
        # Connect ess_mass_target to ESS sizing component
        self.connect('ess_mass_target', 'ess_sizing.ess_mass_target')
        
        # 4. Calculate ESS mass delta: delta = ess_mass_target - ess_mass
        # (positive when rounding down, represents unused mass allocation)
        ess_delta_calc = AddSubtractComp(
            output_name='ess_mass_delta',
            input_names=['ess_mass_target', 'ess_mass'],
            scaling_factors=[1.0, -1.0],
            units='kg',
            desc='Delta between ESS target and actual mass (unused allocation)'
        )
        self.add_subsystem('ess_delta_calc', ess_delta_calc,
                          promotes_outputs=['ess_mass_delta'])
        
        # Connect ess_mass_target and ess_mass to delta calculation
        self.connect('ess_mass_target', 'ess_delta_calc.ess_mass_target')
        self.connect('ess_mass', 'ess_delta_calc.ess_mass')
        
        # 5. Calculate updated OEW: oew_updated = oew_ref - ess_mass_delta
        # Since ESS rounds down, we have unused mass that reduces the OEW
        oew_update_calc = AddSubtractComp(
            output_name='oew_updated',
            input_names=['oew_ref', 'ess_mass_delta'],
            scaling_factors=[1.0, -1.0],
            units='kg',
            desc='Updated OEW after accounting for unused ESS mass allocation'
        )
        self.add_subsystem('oew_update_calc', oew_update_calc,
                          promotes_inputs=['oew_ref'],
                          promotes_outputs=['oew_updated'])
        
        # Connect ess_mass_delta to OEW update calculation
        self.connect('ess_mass_delta', 'oew_update_calc.ess_mass_delta')


def test_wing_ess_sizing_group():
    """Test the WingESSSizingGroup with example values."""
    
    # Load battery data from Excel file
    bat_data = BatteryData.get_data(
        bat_filename='models/atlas/atlas/propulsion/empirical_data/MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent.xlsx', 
        cell_sheetname='BOL_cell_fct_CRate', 
        config_sheetname='battery_config'
    )
    
    print("=" * 70)
    print("LOADED BATTERY DATA")
    print("=" * 70)
    print(f"  n_str:              {bat_data.n_str}")
    print(f"  n_series_per_str:   {bat_data.n_series_per_str}")
    print(f"  n_parallel_per_str: {bat_data.n_parallel_per_str}")
    print(f"  m_cell:             {bat_data.m_cell * 1000:.1f} g")
    print(f"  cell_Ah_capacity:   {bat_data.cell_Ah_capacity:.2f} Ah")
    print(f"  cp_cell:            {bat_data.cp_cell:.1f} J/kg/K")
    print("=" * 70 + "\n")
    
    prob = om.Problem(reports=False)
    
    # Add independent variable component for all inputs
    ivc = om.IndepVarComp()
    
    # Wing inputs (names match load_ac_data.py convention)
    ivc.add_output('S_ref', val=84.8, units='m**2')
    ivc.add_output('mtow', val=86000.0, units='lbm')
    ivc.add_output('span', val=118, units='ft')
    ivc.add_output('sweep', val=0.0, units='deg')
    ivc.add_output('tc', val=0.17)
    ivc.add_output('taper', val=0.5)
    ivc.add_output('AR', val=13.5)
    ivc.add_output('nult', val=3.75)
    ivc.add_output('num_engines', val=4.0)
    ivc.add_output('outboard_nacelle_mass', val=2630.0, units='lbm')
    ivc.add_output('inboard_nacelle_mass', val=3380.0, units='lbm')
    
    # ESS inputs - loaded from battery data
    ivc.add_output('total_propulsion_mass_budget', val=26549.0, units='lbm')
    ivc.add_output('n_str', val=float(bat_data.n_str))
    ivc.add_output('n_series_per_str', val=float(bat_data.n_series_per_str))
    ivc.add_output('m_cell', val=bat_data.m_cell, units='kg')
    
    # OEW reference value
    ivc.add_output('oew_ref', val=63500.0, units='lbm')
    
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('wing_ess_sizing', 
                            WingESSSizingGroup(total_propulsion_mass_budget=26549.0),
                            promotes=['*'])
    
    prob.setup()
    prob.run_model()
    
    print("=" * 70)
    print("WING + ESS SIZING GROUP TEST RESULTS")
    print("=" * 70)
    
    print("\nWing Inputs:")
    print(f"  S_ref (Wing Area):      {prob.get_val('S_ref', units='m**2')[0]:.2f} m²")
    print(f"  MTOW:                   {prob.get_val('mtow', units='kg')[0]:.0f} kg")
    print(f"  span:                   {prob.get_val('span', units='m')[0]:.2f} m")
    print(f"  sweep:                  {prob.get_val('sweep', units='deg')[0]:.1f}°")
    print(f"  tc:                     {prob.get_val('tc')[0]:.3f}")
    print(f"  taper:                  {prob.get_val('taper')[0]:.2f}")
    print(f"  AR:                     {prob.get_val('AR')[0]:.2f}")
    
    print("\nWing Weight Results:")
    print(f"  Total Wing Weight:      {prob.get_val('wing_weight', units='kg')[0]:.2f} kg")
    print(f"    - Bending Mass:       {prob.get_val('wing_bending_mass', units='kg')[0]:.2f} kg")
    print(f"    - Shear/Ctrl Mass:    {prob.get_val('wing_shear_control_mass', units='kg')[0]:.2f} kg")
    print(f"    - Misc Mass:          {prob.get_val('wing_misc_mass', units='kg')[0]:.2f} kg")
    
    print("\nESS Mass Budget Calculation:")
    print(f"  Total Propulsion Budget:{prob.get_val('total_propulsion_mass_budget', units='kg')[0]:.2f} kg")
    print(f"  Wing Weight:            {prob.get_val('wing_weight', units='kg')[0]:.2f} kg")
    print(f"  ESS Mass Target:        {prob.get_val('ess_mass_target', units='kg')[0]:.2f} kg")
    
    print("\nESS Sizing Results:")
    print(f"  n_str:                  {prob.get_val('n_str')[0]:.0f}")
    print(f"  n_series_per_str:       {prob.get_val('n_series_per_str')[0]:.0f}")
    print(f"  m_cell:                 {prob.get_val('m_cell', units='kg')[0]*1000:.1f} g")
    print(f"  n_parallel_per_str:     {prob.get_val('n_parallel_per_str')[0]:.0f}")
    print(f"  n_cells_total:          {prob.get_val('n_cells_total')[0]:.0f} cells")
    print(f"  ess_mass (actual):      {prob.get_val('ess_mass', units='kg')[0]:.2f} kg")
    
    # ESS delta and OEW update
    print("\nOEW Update Calculation:")
    print(f"  OEW Reference:          {prob.get_val('oew_ref', units='lbm')[0]:.2f} lbm")
    print(f"  ESS Mass Delta:         {prob.get_val('ess_mass_delta', units='lbm')[0]:.2f} lbm (unused due to rounding)")
    print(f"  OEW Updated:            {prob.get_val('oew_updated', units='lbm')[0]:.2f} lbm")
    
    # Verification
    ess_target = prob.get_val('ess_mass_target', units='lbm')[0]
    ess_actual = prob.get_val('ess_mass', units='lbm')[0]
    ess_delta = prob.get_val('ess_mass_delta', units='lbm')[0]
    oew_ref = prob.get_val('oew_ref', units='lbm')[0]
    oew_updated = prob.get_val('oew_updated', units='lbm')[0]
    
    print(f"\nVerification:")
    print(f"  ESS Target - Actual:    {ess_target - ess_actual:.2f} lbm (should equal ess_mass_delta)")
    print(f"  OEW Ref - Delta:        {oew_ref - ess_delta:.2f} lbm (should equal oew_updated)")
    print(f"  OEW Updated:            {oew_updated:.2f} lbm")
    
    print("\n" + "=" * 70)
    
    # Generate N2 diagram
    om.n2(prob, outfile='wing_ess_sizing_n2.html', show_browser=False)
    print("N2 diagram saved to wing_ess_sizing_n2.html")
    
    return prob


def sweep_wing_area_ess_weight():
    """
    Sweep wing area from 80 to 100 m² and aspect ratio from 10 to 15,
    plotting how ESS weight changes.
    
    For each combination, span is calculated as: span = sqrt(AR * S_ref)
    
    As wing area increases, wing weight increases, leaving less mass budget
    for the ESS (battery), so ESS mass decreases.
    Higher aspect ratio generally increases wing weight for the same area.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Load battery data from Excel file
    bat_data = BatteryData.get_data(
        bat_filename='models/atlas/atlas/propulsion/empirical_data/MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent.xlsx', 
        cell_sheetname='BOL_cell_fct_CRate', 
        config_sheetname='battery_config'
    )
    
    print("=" * 70)
    print("LOADED BATTERY DATA")
    print("=" * 70)
    print(f"  n_str:              {bat_data.n_str}")
    print(f"  n_series_per_str:   {bat_data.n_series_per_str}")
    print(f"  n_parallel_per_str: {bat_data.n_parallel_per_str}")
    print(f"  m_cell:             {bat_data.m_cell * 1000:.1f} g")
    print("=" * 70 + "\n")
    
    # Wing areas to sweep (m²)
    s_ref_values = np.linspace(80, 100, 21)
    # Aspect ratios to sweep
    ar_values = np.linspace(10, 15, 6)  # 6 values: 10, 11, 12, 13, 14, 15
    
    # Storage for results - organized by AR
    results_by_ar = {}
    for ar in ar_values:
        results_by_ar[ar] = {
            'S_ref_m2': [],
            'span_m': [],
            'span_ft': [],
            'wing_weight_kg': [],
            'wing_weight_lbm': [],
            'wing_weight_kg_validation': [],  # From wingWeightCalc.py
            'wing_weight_lbm_validation': [],  # From wingWeightCalc.py
            'ess_mass_target_kg': [],
            'ess_mass_target_lbm': [],
            'ess_mass_actual_kg': [],
            'ess_mass_actual_lbm': [],
            'ess_mass_delta_lbm': [],  # Delta between target and actual
            'n_parallel_per_str': [],
        }
    
    # Simple class to hold aircraft config for wingWeightCalc
    class SimpleAircraftConfig:
        def __init__(self, wingArea_m2, mtow_kg, wingSpan_m, wingSweep_deg, 
                     wingTC, wingTapRa, wingAR, propNum, wgtOutboardNac_kg, wgtInboardNac_kg):
            self.wingArea = wingArea_m2
            self.mtow = mtow_kg
            self.wingSpan = wingSpan_m
            self.wingSweep = wingSweep_deg
            self.wingTC = wingTC
            self.wingTapRa = wingTapRa
            self.wingAR = wingAR
            self.propNum = propNum
            self.wgtOutboardNac = wgtOutboardNac_kg
            self.wgtInboardNac = wgtInboardNac_kg
    
    class SimpleAircraft:
        def __init__(self, config):
            self.config = config
    
    print("=" * 70)
    print("WING AREA vs ESS WEIGHT SWEEP (with AR variation)")
    print("=" * 70)
    print(f"Sweeping: S_ref = 80-100 m², AR = 10-15")
    print(f"Span calculated as: span = sqrt(AR * S_ref)")
    print("-" * 70)
    
    total_cases = len(s_ref_values) * len(ar_values)
    case_num = 0
    
    for ar in ar_values:
        print(f"\nAR = {ar:.1f}:")
        print(f"{'S_ref (m²)':<12} {'Span (ft)':<12} {'Wing Wt (lbm)':<15} {'ESS Actual (lbm)':<18} {'n_parallel':<12}")
        print("-" * 70)
        
        for s_ref in s_ref_values:
            case_num += 1
            # Calculate span from AR and area: AR = span² / area, so span = sqrt(AR * area)
            span_m = np.sqrt(ar * s_ref)
            span_ft = span_m * 3.28084  # Convert to feet
            
            prob = om.Problem(reports=False)
            
            # Add independent variable component for all inputs
            ivc = om.IndepVarComp()
            
            # Wing inputs - vary S_ref and AR, calculate span accordingly
            ivc.add_output('S_ref', val=s_ref, units='m**2')
            ivc.add_output('mtow', val=86000.0, units='lbm')
            ivc.add_output('span', val=span_ft, units='ft')
            ivc.add_output('sweep', val=4.0, units='deg')
            ivc.add_output('tc', val=0.15)
            ivc.add_output('taper', val=0.463)
            ivc.add_output('AR', val=ar)
            ivc.add_output('nult', val=3.75)
            ivc.add_output('num_engines', val=4.0)
            ivc.add_output('outboard_nacelle_mass', val=2630.0, units='lbm')
            ivc.add_output('inboard_nacelle_mass', val=3380.0, units='lbm')
            
            # ESS inputs - loaded from battery data
            ivc.add_output('total_propulsion_mass_budget', val=26549.0, units='lbm')
            ivc.add_output('n_str', val=float(bat_data.n_str))
            ivc.add_output('n_series_per_str', val=float(bat_data.n_series_per_str))
            ivc.add_output('m_cell', val=bat_data.m_cell, units='kg')
            
            # OEW reference value
            ivc.add_output('oew_ref', val=63500.0, units='lbm')
            
            prob.model.add_subsystem('ivc', ivc, promotes=['*'])
            prob.model.add_subsystem('wing_ess_sizing', 
                                    WingESSSizingGroup(total_propulsion_mass_budget=26549.0),
                                    promotes=['*'])
            
            prob.setup()
            prob.run_model()
            
            # Extract results from OpenMDAO component
            wing_wt_kg = prob.get_val('wing_weight', units='kg')[0]
            wing_wt_lbm = prob.get_val('wing_weight', units='lbm')[0]
            ess_target_kg = prob.get_val('ess_mass_target', units='kg')[0]
            ess_target_lbm = prob.get_val('ess_mass_target', units='lbm')[0]
            ess_actual_kg = prob.get_val('ess_mass', units='kg')[0]
            ess_actual_lbm = prob.get_val('ess_mass', units='lbm')[0]
            n_parallel = prob.get_val('n_parallel_per_str')[0]
            
            # Compute wing weight using wingWeightCalc.py for validation
            # Convert inputs to match what wingWeightCalc expects:
            # - mtow: from lbm to kg
            # - span: from ft to m
            # - nacelle masses: from lbm to kg
            mtow_kg = 86000.0 / 2.2046  # Convert lbm to kg
            outboard_nac_kg = 2630.0 / 2.2046  # Convert lbm to kg
            inboard_nac_kg = 3380.0 / 2.2046  # Convert lbm to kg
            
            # Create simple aircraft object for wingWeightCalc
            aircraft_config = SimpleAircraftConfig(
                wingArea_m2=s_ref,
                mtow_kg=mtow_kg,
                wingSpan_m=span_m,
                wingSweep_deg=4.0,
                wingTC=0.15,
                wingTapRa=0.463,
                wingAR=ar,
                propNum=4,
                wgtOutboardNac_kg=outboard_nac_kg,
                wgtInboardNac_kg=inboard_nac_kg
            )
            aircraft = SimpleAircraft(aircraft_config)
            
            # Compute wing weight using wingWeightCalc.py
            wing_wt_kg_validation = wingWeightCalc(aircraft, nult=3.75)
            wing_wt_lbm_validation = wing_wt_kg_validation * 2.2046
            
            # Store results
            results_by_ar[ar]['S_ref_m2'].append(s_ref)
            results_by_ar[ar]['span_m'].append(span_m)
            results_by_ar[ar]['span_ft'].append(span_ft)
            results_by_ar[ar]['wing_weight_kg'].append(wing_wt_kg)
            results_by_ar[ar]['wing_weight_lbm'].append(wing_wt_lbm)
            results_by_ar[ar]['wing_weight_kg_validation'].append(wing_wt_kg_validation)
            results_by_ar[ar]['wing_weight_lbm_validation'].append(wing_wt_lbm_validation)
            results_by_ar[ar]['ess_mass_target_kg'].append(ess_target_kg)
            results_by_ar[ar]['ess_mass_target_lbm'].append(ess_target_lbm)
            results_by_ar[ar]['ess_mass_actual_kg'].append(ess_actual_kg)
            results_by_ar[ar]['ess_mass_actual_lbm'].append(ess_actual_lbm)
            # Calculate delta: target - actual (positive when rounding down)
            ess_mass_delta_lbm = ess_target_lbm - ess_actual_lbm
            results_by_ar[ar]['ess_mass_delta_lbm'].append(ess_mass_delta_lbm)
            results_by_ar[ar]['n_parallel_per_str'].append(n_parallel)
            
            if case_num % 10 == 0 or s_ref == s_ref_values[0] or s_ref == s_ref_values[-1]:
                print(f"{s_ref:<12.1f} {span_ft:<12.1f} {wing_wt_lbm:<15.1f} {ess_actual_lbm:<18.1f} {n_parallel:<12.0f}")
    
    print("=" * 70)
    print(f"Completed {total_cases} cases")
    print("=" * 70)
    
    # Create plot with multiple AR curves (3 subplots in a row)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    # Color map for different AR values
    colors = plt.cm.viridis(np.linspace(0, 1, len(ar_values)))
    
    # Left plot: ESS Weight vs Wing Area (multiple AR curves)
    ax1.set_xlabel('Wing Reference Area (m²)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ESS Weight (lbm)', fontsize=14, fontweight='bold')
    ax1.set_title('ESS Weight vs Wing Area', 
                  fontsize=16, fontweight='bold', pad=15)
    
    for i, ar in enumerate(ar_values):
        results = results_by_ar[ar]
        ax1.plot(results['S_ref_m2'], results['ess_mass_actual_lbm'], 
                'o-', color=colors[i], linewidth=2.5, markersize=6,
                label=f'AR = {ar:.1f}', alpha=0.8)
    
    ax1.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='-')
    
    # Right plot: Wing Weight vs Wing Area (multiple AR curves)
    ax2.set_xlabel('Wing Reference Area (m²)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Wing Weight (lbm)', fontsize=14, fontweight='bold')
    ax2.set_title('Wing Weight vs Wing Area', 
                  fontsize=16, fontweight='bold', pad=15)
    
    # Plot OpenMDAO component results (solid lines with markers)
    for i, ar in enumerate(ar_values):
        results = results_by_ar[ar]
        ax2.plot(results['S_ref_m2'], results['wing_weight_lbm'], 
                's-', color=colors[i], linewidth=2.5, markersize=6,
                label=f'AR = {ar:.1f} (atlas)', alpha=0.8)
    
    # Plot wingWeightCalc.py validation results (dashed lines)
    for i, ar in enumerate(ar_values):
        results = results_by_ar[ar]
        ax2.plot(results['S_ref_m2'], results['wing_weight_lbm_validation'], 
                '--', color=colors[i], linewidth=2.0, alpha=0.6,
                label=f'AR = {ar:.1f} (apolo)')
    
    ax2.legend(loc='best', fontsize=8, framealpha=0.95, ncol=3)
    ax2.grid(True, alpha=0.3, linestyle='-')
    
    # Third plot: ESS Mass Delta (Target - Actual) vs Wing Area (scatter plot)
    ax3.set_xlabel('Wing Reference Area (m²)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mass Delta (lbm)', fontsize=14, fontweight='bold')
    ax3.set_title('ESS Mass\nTarget Allocated vs Actual Calculated', 
                  fontsize=16, fontweight='bold', pad=15)
    
    for i, ar in enumerate(ar_values):
        results = results_by_ar[ar]
        ax3.scatter(results['S_ref_m2'], results['ess_mass_delta_lbm'], 
                   color=colors[i], s=50, alpha=0.8)
    
    ax3.grid(True, alpha=0.3, linestyle='-')
    ax3.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)  # Zero reference line
    
    # Add overall title
    fig.suptitle('Fixed Total Propulsion Mass Budget = 26,549 lbm. P70X 211s450p BOL 1grp1.', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    # Save figure
    output_file = 'wing_area_vs_ess_weight.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()
    
    return results_by_ar


if __name__ == "__main__":
    # Run the original test
    test_wing_ess_sizing_group()
    
    # Run the wing area sweep
    print("\n\n")
    sweep_wing_area_ess_weight()

