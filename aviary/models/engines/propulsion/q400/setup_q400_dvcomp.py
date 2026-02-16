"""
Q400-specific design variable component setup.

This is a simplified version of setup_ac_dvcomp.py that excludes battery-related 
fields since the Q400 is a conventional turboprop without electric propulsion.
"""


def setup_q400_dv(dv_comp, point_analysis=False, converge_tow=False):
    """
    Setup design variables for Q400 aircraft (no battery/hybrid fields).
    
    Parameters
    ----------
    dv_comp : DictIndepVarComp
        The design variable component to add outputs to
    point_analysis : bool
        Whether this is a point analysis (fixes TOW)
        
    Returns
    -------
    dv_comp : DictIndepVarComp
        The updated design variable component
    """
    
    # Geometry Overview
    dv_comp.add_output_from_dict("ac|geom|k_rough")

    # Aero
    dv_comp.add_output_from_dict("ac|aero|polar|cd_gear")
    dv_comp.add_output_from_dict("ac|aero|polar|cd_sp")
    dv_comp.add_output_from_dict("ac|aero|polar|cd_oei")
    dv_comp.add_output_from_dict("ac|aero|polar|cd_baseline")
    dv_comp.add_output_from_dict("ac|aero|polar|roughness_drag")
    dv_comp.add_output_from_dict("ac|aero|polar|cd_other")

    # Wing
    dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
    dv_comp.add_output_from_dict("ac|geom|wing|span")
    dv_comp.add_output_from_dict("ac|geom|wing|AR")
    dv_comp.add_output_from_dict("ac|geom|wing|taper")
    dv_comp.add_output_from_dict("ac|geom|wing|toverc")
    dv_comp.add_output_from_dict("ac|geom|wing|MAC")
    dv_comp.add_output_from_dict("ac|geom|wing|S_wet")
    dv_comp.add_output_from_dict("ac|geom|wing|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|wing|c4_sweep")
    dv_comp.add_output_from_dict("ac|geom|wing|airfoil_camber")

    # Horizontal Stabilizer
    dv_comp.add_output_from_dict("ac|geom|hstab|S_ref")
    dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
    dv_comp.add_output_from_dict("ac|geom|hstab|MAC")
    dv_comp.add_output_from_dict("ac|geom|hstab|S_wet")
    dv_comp.add_output_from_dict("ac|geom|hstab|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|hstab|toverc")
    dv_comp.add_output_from_dict("ac|geom|hstab|hinge_factor")

    # Vertical Stabilizer
    dv_comp.add_output_from_dict("ac|geom|vstab|S_ref")
    dv_comp.add_output_from_dict("ac|geom|vstab|MAC")
    dv_comp.add_output_from_dict("ac|geom|vstab|S_wet")
    dv_comp.add_output_from_dict("ac|geom|vstab|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|vstab|toverc")
    dv_comp.add_output_from_dict("ac|geom|vstab|hinge_factor")

    # Fuselage
    dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
    dv_comp.add_output_from_dict("ac|geom|fuselage|width")
    dv_comp.add_output_from_dict("ac|geom|fuselage|length")
    dv_comp.add_output_from_dict("ac|geom|fuselage|height")
    dv_comp.add_output_from_dict("ac|geom|fuselage|MAC")
    dv_comp.add_output_from_dict("ac|geom|fuselage|fineness_ratio")

    # Nacelle
    dv_comp.add_output_from_dict("ac|geom|nacelle|S_wet")
    dv_comp.add_output_from_dict("ac|geom|nacelle|length")
    dv_comp.add_output_from_dict("ac|geom|nacelle|fineness_ratio")

    # Weights (NO battery for Q400)
    dv_comp.add_output_from_dict("ac|weights|MTOW")
    if point_analysis or not converge_tow:
        dv_comp.add_output_from_dict("ac|weights|TOW")
    dv_comp.add_output_from_dict("ac|weights|OEW")
    dv_comp.add_output_from_dict("ac|weights|payload")
    dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
    dv_comp.add_output_from_dict("ac|weights|MLW")

    return dv_comp

