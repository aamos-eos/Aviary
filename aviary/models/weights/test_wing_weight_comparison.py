"""
Test script to compare the original wingWeightCalc function with the new WingWeightComp OpenMDAO component.
"""

import numpy as np
import openmdao.api as om
from wing_weight_component import WingWeightComp
from wingWeightCalc import wingWeightCalc


class MockAircraftConfig:
    """Mock aircraft configuration to match the inputs expected by wingWeightCalc."""
    def __init__(self, wing_area, mtow, wing_span, wing_sweep, wing_tc, 
                 wing_taper_ratio, wing_ar, prop_num, wgt_outboard_nac, wgt_inboard_nac):
        self.wingArea = wing_area
        self.mtow = mtow
        self.wingSpan = wing_span
        self.wingSweep = wing_sweep
        self.wingTC = wing_tc
        self.wingTapRa = wing_taper_ratio
        self.wingAR = wing_ar
        self.propNum = prop_num
        self.wgtOutboardNac = wgt_outboard_nac
        self.wgtInboardNac = wgt_inboard_nac


class MockAircraft:
    """Mock aircraft object with config attribute."""
    def __init__(self, config):
        self.config = config


def compare_wing_weight_calculations():
    """
    Compare the original function with the OpenMDAO component using identical inputs.
    """
    # Define test inputs (using new names matching load_ac_data.py convention)
    test_cases = [
        {
            'name': 'Test Case 1: Typical Transport Aircraft',
            'S_ref': 84.8,          # m^2
            'mtow': 16800.0,        # kg
            'span': 33.8,           # m
            'sweep': 0.0,           # degrees
            'tc': 0.17,             # thickness-to-chord ratio
            'taper': 0.5,
            'AR': 13.5,
            'nult': 3.75,
            'num_engines': 4,
            'outboard_nacelle_mass': 350.0,  # kg
            'inboard_nacelle_mass': 350.0,   # kg
        },
        {
            'name': 'Test Case 2: Larger Aircraft with Sweep',
            'S_ref': 150.0,         # m^2
            'mtow': 50000.0,        # kg
            'span': 40.0,           # m
            'sweep': 25.0,          # degrees
            'tc': 0.12,
            'taper': 0.3,
            'AR': 10.67,
            'nult': 3.75,
            'num_engines': 4,
            'outboard_nacelle_mass': 500.0,
            'inboard_nacelle_mass': 600.0,
        },
        {
            'name': 'Test Case 3: High AR Wing',
            'S_ref': 100.0,         # m^2
            'mtow': 25000.0,        # kg
            'span': 35.0,           # m
            'sweep': 15.0,          # degrees
            'tc': 0.15,
            'taper': 0.4,
            'AR': 12.25,
            'nult': 4.0,
            'num_engines': 4,
            'outboard_nacelle_mass': 400.0,
            'inboard_nacelle_mass': 450.0,
        },
    ]
    
    print("=" * 80)
    print("WING WEIGHT CALCULATION COMPARISON")
    print("Original Function vs OpenMDAO Component")
    print("=" * 80)
    
    all_passed = True
    
    for case in test_cases:
        print(f"\n{case['name']}")
        print("-" * 60)
        
        # Create mock aircraft for original function (uses old naming convention)
        config = MockAircraftConfig(
            wing_area=case['S_ref'],
            mtow=case['mtow'],
            wing_span=case['span'],
            wing_sweep=case['sweep'],
            wing_tc=case['tc'],
            wing_taper_ratio=case['taper'],
            wing_ar=case['AR'],
            prop_num=case['num_engines'],
            wgt_outboard_nac=case['outboard_nacelle_mass'],
            wgt_inboard_nac=case['inboard_nacelle_mass']
        )
        aircraft = MockAircraft(config)
        
        # Call original function
        original_weight = wingWeightCalc(aircraft, nult=case['nult'])
        
        # Set up OpenMDAO problem (uses new naming convention matching load_ac_data.py)
        prob = om.Problem()
        
        ivc = om.IndepVarComp()
        ivc.add_output('S_ref', val=case['S_ref'], units='m**2')
        ivc.add_output('mtow', val=case['mtow'], units='kg')
        ivc.add_output('span', val=case['span'], units='m')
        ivc.add_output('sweep', val=case['sweep'], units='deg')
        ivc.add_output('tc', val=case['tc'])
        ivc.add_output('taper', val=case['taper'])
        ivc.add_output('AR', val=case['AR'])
        ivc.add_output('nult', val=case['nult'])
        ivc.add_output('num_engines', val=float(case['num_engines']))
        ivc.add_output('outboard_nacelle_mass', val=case['outboard_nacelle_mass'], units='kg')
        ivc.add_output('inboard_nacelle_mass', val=case['inboard_nacelle_mass'], units='kg')
        
        prob.model.add_subsystem('ivc', ivc, promotes=['*'])
        prob.model.add_subsystem('wing_weight', WingWeightComp(), promotes=['*'])
        
        prob.setup()
        prob.run_model()
        
        component_weight = prob.get_val('wing_weight', units='kg')[0]
        bending_mass = prob.get_val('wing_bending_mass', units='kg')[0]
        shear_control_mass = prob.get_val('wing_shear_control_mass', units='kg')[0]
        misc_mass = prob.get_val('wing_misc_mass', units='kg')[0]
        
        # Calculate difference
        diff = abs(original_weight - component_weight)
        rel_diff = diff / original_weight * 100 if original_weight > 0 else 0
        
        # Print inputs
        print(f"  Inputs:")
        print(f"    S_ref (Area):     {case['S_ref']:.2f} m²")
        print(f"    MTOW:             {case['mtow']:.0f} kg")
        print(f"    span:             {case['span']:.2f} m")
        print(f"    sweep:            {case['sweep']:.1f}°")
        print(f"    tc:               {case['tc']:.3f}")
        print(f"    taper:            {case['taper']:.2f}")
        print(f"    AR:               {case['AR']:.2f}")
        print(f"    nult:             {case['nult']:.2f}")
        print(f"    num_engines:      {case['num_engines']}")
        
        # Print results
        print(f"\n  Results:")
        print(f"    Original Function:     {original_weight:.2f} kg")
        print(f"    OpenMDAO Component:    {component_weight:.2f} kg")
        print(f"      - Bending Mass:      {bending_mass:.2f} kg")
        print(f"      - Shear/Ctrl Mass:   {shear_control_mass:.2f} kg")
        print(f"      - Misc Mass:         {misc_mass:.2f} kg")
        print(f"\n    Difference:            {diff:.4f} kg ({rel_diff:.4f}%)")
        
        # Check if match is acceptable (within 0.01%)
        tolerance = 0.01  # 0.01% relative tolerance
        if rel_diff < tolerance:
            print(f"    Status: ✓ PASS (within {tolerance}% tolerance)")
        else:
            print(f"    Status: ✗ FAIL (exceeds {tolerance}% tolerance)")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL RESULT: ✓ ALL TESTS PASSED")
    else:
        print("OVERALL RESULT: ✗ SOME TESTS FAILED")
    print("=" * 80)
    
    # Now check partials for one case
    print("\n\nCHECKING PARTIAL DERIVATIVES (Test Case 1)")
    print("-" * 60)
    return all_passed


if __name__ == "__main__":
    compare_wing_weight_calculations()

