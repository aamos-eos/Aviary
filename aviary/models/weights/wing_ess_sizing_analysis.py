"""
OpenMDAO Group that loads aircraft data and performs Wing Weight + ESS sizing analysis.

This group:
1. Loads aircraft data from Excel using load_ac_data_from_excel
2. Creates a DictIndepVarComp to parse the data into OpenMDAO variables
3. Connects the data to WingESSSizingGroup for wing weight and ESS sizing calculations

This provides a complete workflow from data file to analysis results.
"""

import numpy as np
import openmdao.api as om

from aviary.utils.dict_indepvarcomp import DictIndepVarComp
from aviary.models.aircraft.e180.load_ac_data import load_ac_data_from_excel
from wing_weight_component import WingWeightComp
from ess_weight_component import ESSParallelFromMassComp
from aviary.utils.math_components.add_subtract_comp import AddSubtractComp


def setup_wing_ess_dv(dv_comp):
    """
    Setup the design variable component with wing and ESS related outputs.
    
    Parameters
    ----------
    dv_comp : DictIndepVarComp
        The design variable component to add outputs to
        
    Returns
    -------
    dv_comp : DictIndepVarComp
        The updated design variable component
    """
    # Wing geometry inputs for wing weight calculation
    dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
    dv_comp.add_output_from_dict("ac|geom|wing|span")
    dv_comp.add_output_from_dict("ac|geom|wing|AR")
    dv_comp.add_output_from_dict("ac|geom|wing|taper")
    dv_comp.add_output_from_dict("ac|geom|wing|toverc")
    dv_comp.add_output_from_dict("ac|geom|wing|c4_sweep")
    
    # Weights
    dv_comp.add_output_from_dict("ac|weights|MTOW")
    dv_comp.add_output_from_dict("ac|weights|OEW")
    
    # Battery configuration for ESS sizing
    dv_comp.add_output_from_dict("ac|propulsion|battery|n_str")
    dv_comp.add_output_from_dict("ac|propulsion|battery|n_series_per_str")
    dv_comp.add_output_from_dict("ac|propulsion|battery|n_parallel_per_str")
    dv_comp.add_output_from_dict("ac|propulsion|battery|m_cell")
    
    # Number of propulsors (engines)
    dv_comp.add_output_from_dict("ac|propulsion|prop|num_props")
    
    return dv_comp


class WingESSSizingAnalysis(om.Group):
    """
    OpenMDAO Group that loads aircraft data and performs wing weight + ESS sizing.
    
    This group provides a complete analysis workflow:
    1. Loads aircraft configuration from Excel file
    2. Parses data into OpenMDAO variables using DictIndepVarComp
    3. Computes wing weight using NASA FLOPS methodology
    4. Sizes ESS based on available mass budget
    5. Updates OEW based on ESS rounding effects
    
    Options
    -------
    ac_data : dict
        Pre-loaded aircraft data dictionary. If None, will load from files.
    ac_filename : str
        Path to aircraft data Excel file (used if ac_data is None)
    bat_filename : str
        Path to battery data Excel file (used if ac_data is None)
    cell_sheetname : str
        Sheet name for cell data (used if ac_data is None)
    config_sheetname : str
        Sheet name for battery config (used if ac_data is None)
    total_propulsion_mass_budget : float
        Total mass budget for propulsion system (kg). Default is 26549.0
    pack_scale_factor : float
        ESS packaging scale factor. Default is 1.247
    oew_ref : float
        Reference OEW value (kg). Default is 63500.0
    
    Wing weight options:
        comp_frac, fstrt, faert, num_fuse, load_distribution_factor,
        num_integration_stations, total_mass_scaler, scaler_ShearCtrlMass,
        scaler_MiscMass, scaler_BendingMass
    
    Outputs (promoted)
    ------------------
    wing_weight : float
        Total wing weight (kg)
    wing_bending_mass, wing_shear_control_mass, wing_misc_mass : float
        Wing weight breakdown (kg)
    ess_mass_target : float
        Target ESS mass allocation (kg)
    n_parallel_per_str : float
        Number of cells in parallel per string (rounded)
    n_cells_total : float
        Total number of cells
    ess_mass : float
        Actual ESS mass (kg)
    ess_mass_delta : float
        Unused ESS mass allocation (kg)
    oew_updated : float
        Updated OEW after accounting for ESS rounding (kg)
    """
    
    def initialize(self):
        # Data loading options
        self.options.declare('ac_data', default=None, 
                            desc='Pre-loaded aircraft data dictionary')
        self.options.declare('ac_filename', default=None, 
                            desc='Path to aircraft data Excel file')
        self.options.declare('bat_filename', default=None, 
                            desc='Path to battery data Excel file')
        self.options.declare('cell_sheetname', default='BOL_cell_fct_CRate', 
                            desc='Sheet name for cell data')
        self.options.declare('config_sheetname', default='battery_config', 
                            desc='Sheet name for battery config')
        
        # ESS sizing options
        self.options.declare('total_propulsion_mass_budget', default=26549.0 / 2.20462, 
                            desc='Total mass budget for propulsion system (kg)')
        self.options.declare('pack_scale_factor', default=1.247, 
                            desc='Scale factor from cell mass to pack mass')
        self.options.declare('oew_ref', default=63500.0 / 2.20462, 
                            desc='Reference OEW value (kg)')
        
        # Wing weight options
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
        ac_data = self.options['ac_data']
        pack_scale_factor = self.options['pack_scale_factor']
        
        # Load data if not provided
        if ac_data is None:
            ac_filename = self.options['ac_filename']
            bat_filename = self.options['bat_filename']
            cell_sheetname = self.options['cell_sheetname']
            config_sheetname = self.options['config_sheetname']
            
            if ac_filename is None or bat_filename is None:
                raise ValueError("Either ac_data or (ac_filename and bat_filename) must be provided")
            
            ac_data = load_ac_data_from_excel(
                filename=ac_filename,
                bat_filename=bat_filename,
                cell_sheetname=cell_sheetname,
                config_sheetname=config_sheetname
            )
        
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
        
        # 1. Add DictIndepVarComp to parse aircraft data
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(ac_data), 
                                     promotes_outputs=["*"])
        dv_comp = setup_wing_ess_dv(dv_comp)
        
        # 2. Add additional inputs that may not be in the data file
        aux_inputs = self.add_subsystem("aux_inputs", om.IndepVarComp(), 
                                        promotes_outputs=["*"])
        aux_inputs.add_output('total_propulsion_mass_budget', 
                             val=self.options['total_propulsion_mass_budget'], units='kg')
        aux_inputs.add_output('oew_ref', val=self.options['oew_ref'], units='kg')
        aux_inputs.add_output('nult', val=3.75, desc='Ultimate load factor')
        
        # 3. Add WingWeightComp to calculate wing weight
        self.add_subsystem('wing_weight_comp', 
                          WingWeightComp(**wing_options),
                          promotes_outputs=['wing_weight', 'wing_bending_mass', 
                                           'wing_shear_control_mass', 'wing_misc_mass'])
        
        # Connect wing inputs from DictIndepVarComp to WingWeightComp
        # Map from ac_data naming to WingWeightComp input names
        self.connect('ac|geom|wing|S_ref', 'wing_weight_comp.S_ref')
        self.connect('ac|geom|wing|span', 'wing_weight_comp.span')
        self.connect('ac|geom|wing|c4_sweep', 'wing_weight_comp.sweep')
        self.connect('ac|geom|wing|toverc', 'wing_weight_comp.tc')
        self.connect('ac|geom|wing|taper', 'wing_weight_comp.taper')
        self.connect('ac|geom|wing|AR', 'wing_weight_comp.AR')
        self.connect('ac|weights|MTOW', 'wing_weight_comp.mtow')
        self.connect('nult', 'wing_weight_comp.nult')
        self.connect('ac|propulsion|prop|num_props', 'wing_weight_comp.num_engines')
        
        # For nacelle masses, we'll use a default or could add to ac_data
        # For now, add as auxiliary inputs
        aux_inputs.add_output('outboard_nacelle_mass', val=2630.0, units='kg')
        aux_inputs.add_output('inboard_nacelle_mass', val=3380.0, units='kg')
        self.connect('outboard_nacelle_mass', 'wing_weight_comp.outboard_nacelle_mass')
        self.connect('inboard_nacelle_mass', 'wing_weight_comp.inboard_nacelle_mass')
        
        # 4. Add AddSubtractComp to compute: ess_mass_target = total_propulsion_mass_budget - wing_weight
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
        self.connect('wing_weight', 'ess_target_calc.wing_weight')
        
        # 5. Add ESSParallelFromMassComp to size the battery
        self.add_subsystem('ess_sizing', 
                          ESSParallelFromMassComp(pack_scale_factor=pack_scale_factor),
                          promotes_outputs=['n_parallel_per_str', 'n_cells_total', 'ess_mass'])
        
        # Connect ESS inputs
        self.connect('ess_mass_target', 'ess_sizing.ess_mass_target')
        self.connect('ac|propulsion|battery|n_str', 'ess_sizing.n_str')
        self.connect('ac|propulsion|battery|n_series_per_str', 'ess_sizing.n_series_per_str')
        self.connect('ac|propulsion|battery|m_cell', 'ess_sizing.m_cell')
        
        # 6. Calculate ESS mass delta: delta = ess_mass_target - ess_mass
        ess_delta_calc = AddSubtractComp(
            output_name='ess_mass_delta',
            input_names=['ess_mass_target', 'ess_mass'],
            scaling_factors=[1.0, -1.0],
            units='kg',
            desc='Delta between ESS target and actual mass (unused allocation)'
        )
        self.add_subsystem('ess_delta_calc', ess_delta_calc,
                          promotes_outputs=['ess_mass_delta'])
        self.connect('ess_mass_target', 'ess_delta_calc.ess_mass_target')
        self.connect('ess_mass', 'ess_delta_calc.ess_mass')
        
        # 7. Calculate updated OEW: oew_updated = oew_ref - ess_mass_delta
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
        self.connect('ess_mass_delta', 'oew_update_calc.ess_mass_delta')


def test_wing_ess_sizing_analysis():
    """Test the WingESSSizingAnalysis with example aircraft data."""
    import os
    
    # Define file paths (adjust as needed for your setup)
    base_path = os.path.dirname(os.path.abspath(__file__))
    ac_filename = os.path.join(base_path, '..', 'scenarios', 'setup_mission', 'ac_data.xlsx')
    bat_filename = os.path.join(base_path, '..', 'propulsion', 'empirical_data', 
                                'MolicelP70X_module211s450p_BOL_1grp1.xlsx')
    
    prob = om.Problem(reports=False)
    
    # Add the analysis group
    prob.model.add_subsystem('analysis', 
                            WingESSSizingAnalysis(
                                ac_filename=ac_filename,
                                bat_filename=bat_filename,
                                cell_sheetname='BOL_cell_fct_CRate',
                                config_sheetname='battery_config',
                                total_propulsion_mass_budget=26549.0 / 2.20462,
                                oew_ref=63500.0 / 2.20462
                            ),
                            promotes=['*'])
    
    prob.setup()
    prob.run_model()
    
    print("=" * 70)
    print("WING + ESS SIZING ANALYSIS TEST RESULTS")
    print("=" * 70)
    
    print("\nWing Inputs (from ac_data):")
    print(f"  S_ref:                  {prob.get_val('ac|geom|wing|S_ref', units='m**2')[0]:.2f} m²")
    print(f"  span:                   {prob.get_val('ac|geom|wing|span', units='m')[0]:.2f} m")
    print(f"  AR:                     {prob.get_val('ac|geom|wing|AR')[0]:.2f}")
    print(f"  taper:                  {prob.get_val('ac|geom|wing|taper')[0]:.3f}")
    print(f"  tc (toverc):            {prob.get_val('ac|geom|wing|toverc')[0]:.3f}")
    print(f"  sweep (c4_sweep):        {prob.get_val('ac|geom|wing|c4_sweep', units='deg')[0]:.1f}°")
    print(f"  MTOW:                   {prob.get_val('ac|weights|MTOW', units='kg')[0]:.0f} kg")
    print(f"  num_props:              {prob.get_val('ac|propulsion|prop|num_props')[0]:.0f}")
    
    print("\nWing Weight Results:")
    print(f"  Total Wing Weight:      {prob.get_val('wing_weight', units='lbm')[0]:.2f} lbm")
    print(f"    - Bending Mass:       {prob.get_val('wing_bending_mass', units='lbm')[0]:.2f} lbm")
    print(f"    - Shear/Ctrl Mass:    {prob.get_val('wing_shear_control_mass', units='lbm')[0]:.2f} lbm")
    print(f"    - Misc Mass:          {prob.get_val('wing_misc_mass', units='lbm')[0]:.2f} lbm")
    
    print("\nESS Mass Budget Calculation:")
    print(f"  Total Propulsion Budget:{prob.get_val('total_propulsion_mass_budget', units='lbm')[0]:.2f} lbm")
    print(f"  Wing Weight:            {prob.get_val('wing_weight', units='lbm')[0]:.2f} lbm")
    print(f"  ESS Mass Target:        {prob.get_val('ess_mass_target', units='lbm')[0]:.2f} lbm")
    
    print("\nBattery Config (from ac_data):")
    print(f"  n_str:                  {prob.get_val('ac|propulsion|battery|n_str')[0]:.0f}")
    print(f"  n_series_per_str:       {prob.get_val('ac|propulsion|battery|n_series_per_str')[0]:.0f}")
    print(f"  m_cell:                 {prob.get_val('ac|propulsion|battery|m_cell', units='kg')[0]*1000:.1f} g")
    
    print("\nESS Sizing Results:")
    print(f"  n_parallel_per_str:     {prob.get_val('n_parallel_per_str')[0]:.0f} (computed)")
    print(f"  n_cells_total:          {prob.get_val('n_cells_total')[0]:.0f} cells")
    print(f"  ess_mass (actual):      {prob.get_val('ess_mass', units='kg')[0]:.2f} kg")
    
    print("\nOEW Update Calculation:")
    print(f"  OEW Reference:          {prob.get_val('oew_ref', units='kg')[0]:.2f} kg")
    print(f"  ESS Mass Delta:         {prob.get_val('ess_mass_delta', units='kg')[0]:.2f} kg")
    print(f"  OEW Updated:            {prob.get_val('oew_updated', units='kg')[0]:.2f} kg")
    
    print("\n" + "=" * 70)
    
    # Generate N2 diagram
    om.n2(prob, outfile='wing_ess_sizing_analysis_n2.html', show_browser=False)
    print("N2 diagram saved to wing_ess_sizing_analysis_n2.html")
    
    return prob


def test_wing_ess_sizing_with_preloaded_data():
    """Test using pre-loaded aircraft data (simulating integration with setup_mission_analysis.py)."""
    import os
    
    # Load data first (like in setup_mission_analysis.py)
    base_path = os.path.dirname(os.path.abspath(__file__))
    ac_filename = os.path.join(base_path, '..', 'scenarios', 'setup_mission', 'ac_data.xlsx')
    bat_filename = os.path.join(base_path, '..', 'propulsion', 'empirical_data', 
                                'inHouse_battery_1motorConfig_208s27p_4grp_to_each_nacelle.xlsx')
    
    # Pre-load data
    ac_data = load_ac_data_from_excel(
        filename=ac_filename,
        bat_filename=bat_filename,
        cell_sheetname='BOL_cell_fct_CRate',
        config_sheetname='battery_config'
    )
    
    prob = om.Problem(reports=False)
    
    # Add the analysis group with pre-loaded data
    prob.model.add_subsystem('analysis', 
                            WingESSSizingAnalysis(
                                ac_data=ac_data,
                                total_propulsion_mass_budget=26549.0 / 2.20462,
                                oew_ref=63500.0 / 2.20462
                            ),
                            promotes=['*'])
    
    prob.setup()
    prob.run_model()
    
    print("=" * 70)
    print("TEST WITH PRE-LOADED DATA")
    print("=" * 70)
    print(f"Wing Weight:      {prob.get_val('wing_weight', units='kg')[0]:.2f} kg")
    print(f"ESS Mass Target:  {prob.get_val('ess_mass_target', units='kg')[0]:.2f} kg")
    print(f"ESS Mass Actual:  {prob.get_val('ess_mass', units='kg')[0]:.2f} kg")
    print(f"OEW Updated:      {prob.get_val('oew_updated', units='kg')[0]:.2f} kg")
    print("=" * 70)
    
    return prob


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test with file paths...")
    print("="*70 + "\n")
    
    try:
        test_wing_ess_sizing_analysis()
    except Exception as e:
        print(f"Test with file paths failed: {e}")
        print("This may be due to missing data files. Trying with pre-loaded data...")
    
    print("\n" + "="*70)
    print("Running test with pre-loaded data...")
    print("="*70 + "\n")
    
    try:
        test_wing_ess_sizing_with_preloaded_data()
    except Exception as e:
        print(f"Test with pre-loaded data failed: {e}")

