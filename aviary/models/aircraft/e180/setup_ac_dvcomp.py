def setup_ac_dv(dv_comp, point_analysis=False, compute_oew=False, 
compute_payload=False, size_motor=False, size_battery=False, opt_wing=False, 
size_tow=False):
    """
    Set up aircraft design variables on a DictIndepVarComp.
    
    Parameters
    ----------
    dv_comp : DictIndepVarComp
        The component to add outputs to.
    point_analysis : bool
        If True, add TOW as a fixed output (for point analysis).
    compute_oew : bool
        If True, skip OEW output (it will be computed internally).
    compute_payload : bool
        If True, skip payload output (it will be computed internally).
    size_motor : bool
        If True, skip ac|propulsion|motor|rating output (it will be provided externally
        as an input for motor sizing).
    size_battery : bool
        If True, skip ac|propulsion|battery|n_parallel_per_str output (it will be provided
        externally as an input for battery sizing).
    size_tow : bool
        If True, skip ac|weights|TOW output for point_analysis (it will be provided
        externally via DVLabel passthrough).
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
    dv_comp.add_output_from_dict("ac|aero|polar|CLmax")

    # Mission
    #dv_comp.add_output_from_dict("ac|mission|nn")
    #dv_comp.add_output_from_dict("ac|mission|ni")

    # Wing
    # Always expose span_plan/span_trap because OEW links reference them
    # regardless of whether wing geometry is optimized.
    dv_comp.add_output_from_dict("ac|geom|wing|span_plan")  # Planform span (for weight calculations)
    dv_comp.add_output_from_dict("ac|geom|wing|span_trap")  # Trapezoidal span (for geometry calculations)
    if not opt_wing:
        # When opt_wing=False, load all geometry from ac_data
        # Load both planform (ref) and trapezoidal (trap) variables
        dv_comp.add_output_from_dict("ac|geom|wing|S_ref")  # Planform area (for weight calculations)
        dv_comp.add_output_from_dict("ac|geom|wing|S_trap")  # Trapezoidal area (for geometry calculations)
        dv_comp.add_output_from_dict("ac|geom|wing|AR") # Refers to planform area by default
        dv_comp.add_output_from_dict("ac|geom|wing|MAC")
        dv_comp.add_output_from_dict("ac|geom|wing|S_wet")
        dv_comp.add_output_from_dict("ac|geom|wing|c_root")
        dv_comp.add_output_from_dict("ac|geom|wing|c_tip")
        dv_comp.add_output_from_dict("ac|geom|wing|mac_buttline")
        dv_comp.add_output_from_dict("ac|geom|wing|c4_sweep")
        dv_comp.add_output_from_dict("ac|geom|wing|toverc")

    # When opt_wing=True, S_ref, AR, span_plan/span_trap, MAC, S_wet, c_root, c_tip, mac_buttline 
    # are computed by WingGeometryComp from design variables
    
    dv_comp.add_output_from_dict("ac|geom|wing|taper")
    dv_comp.add_output_from_dict("ac|geom|wing|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|wing|airfoil_camber")
    dv_comp.add_output_from_dict("ac|geom|wing|c0_sweep")

    dv_comp.add_output_from_dict("ac|geom|wing|apex_percentage")
    dv_comp.add_output_from_dict("ac|geom|wing|control_surface_area_ratio")
    dv_comp.add_output_from_dict("ac|geom|wing|strut_bracing_factor")
    dv_comp.add_output_from_dict("ac|geom|wing|aeroelastic_tailoring_factor")
    dv_comp.add_output_from_dict("ac|geom|wing|composite_fraction")
    dv_comp.add_output_from_dict("ac|geom|wing|n_ult")
    dv_comp.add_output_from_dict("ac|geom|wing|bending_material_weight_scaler")
    dv_comp.add_output_from_dict("ac|geom|wing|load_fraction")
    dv_comp.add_output_from_dict("ac|geom|wing|misc_weight_scaler")
    dv_comp.add_output_from_dict("ac|geom|wing|shear_control_weight_scaler")
    dv_comp.add_output_from_dict("ac|geom|wing|var_sweep_weight_penalty")
    dv_comp.add_output_from_dict("ac|geom|wing|weight_scaler")

    # Horizontal Stabilizer
    dv_comp.add_output_from_dict("ac|geom|hstab|S_ref")
    dv_comp.add_output_from_dict("ac|geom|hstab|tail_volume")
    dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
    dv_comp.add_output_from_dict("ac|geom|hstab|MAC")
    dv_comp.add_output_from_dict("ac|geom|hstab|S_wet")
    dv_comp.add_output_from_dict("ac|geom|hstab|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|hstab|toverc")
    dv_comp.add_output_from_dict("ac|geom|hstab|hinge_factor")
    dv_comp.add_output_from_dict("ac|geom|hstab|c2_sweep")

    # Vertical Stabilizer
    dv_comp.add_output_from_dict("ac|geom|vstab|S_ref")
    dv_comp.add_output_from_dict("ac|geom|vstab|tail_volume")
    dv_comp.add_output_from_dict("ac|geom|vstab|c4_to_wing_c4")
    dv_comp.add_output_from_dict("ac|geom|vstab|MAC")
    dv_comp.add_output_from_dict("ac|geom|vstab|S_wet")
    dv_comp.add_output_from_dict("ac|geom|vstab|maxtcsweep")
    dv_comp.add_output_from_dict("ac|geom|vstab|toverc")
    dv_comp.add_output_from_dict("ac|geom|vstab|hinge_factor")
    dv_comp.add_output_from_dict("ac|geom|vstab|c2_sweep")

    # Fuselage
    dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
    dv_comp.add_output_from_dict("ac|geom|fuselage|width")
    dv_comp.add_output_from_dict("ac|geom|fuselage|nose_tail_length")
    dv_comp.add_output_from_dict("ac|geom|fuselage|length")
    dv_comp.add_output_from_dict("ac|geom|fuselage|height")
    dv_comp.add_output_from_dict("ac|geom|fuselage|MAC")
    # fineness_ratio is optional (only needed for aerodynamics, not OEW)
    try:
        dv_comp.add_output_from_dict("ac|geom|fuselage|fineness_ratio")
    except KeyError:
        pass  # fineness_ratio not in Excel, skip it
    dv_comp.add_output_from_dict("ac|geom|fuselage|dive_speed")
    dv_comp.add_output_from_dict("ac|geom|fuselage|pressure_diff")
    dv_comp.add_output_from_dict("ac|geom|fuselage|n_ult")


    # Nacelle Inboard
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|width")
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|height")
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|length")
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|S_wet")
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|y_loc")
    dv_comp.add_output_from_dict("ac|geom|nacelle_inboard|fineness_ratio")


    # Nacelle Outboard
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|width")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|height")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|length")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|S_wet")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|y_loc")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|dive_speed")
    dv_comp.add_output_from_dict("ac|geom|nacelle_outboard|fineness_ratio")


    # Landing Gear
    dv_comp.add_output_from_dict("ac|geom|landing_gear|mlg_height")
    dv_comp.add_output_from_dict("ac|geom|landing_gear|nlg_height")

    # Flight Controls
    dv_comp.add_output_from_dict("ac|geom|flight_controls|max_mach")

    # Windows
    dv_comp.add_output_from_dict("ac|geom|windows|a_eff")
    dv_comp.add_output_from_dict("ac|geom|windows|base_pressure")
    dv_comp.add_output_from_dict("ac|geom|windows|area")
    dv_comp.add_output_from_dict("ac|geom|windows|windshield_thickness")
    dv_comp.add_output_from_dict("ac|geom|windows|windshield_surface_front")
    dv_comp.add_output_from_dict("ac|geom|windows|windshield_surface_side")

    # Cabin
    dv_comp.add_output_from_dict("ac|geom|cabin|num_passengers")
    dv_comp.add_output_from_dict("ac|geom|cabin|cabin_length")
    dv_comp.add_output_from_dict("ac|geom|cabin|max_passengers")
    dv_comp.add_output_from_dict("ac|geom|cabin|business_class_pax")
    dv_comp.add_output_from_dict("ac|geom|cabin|num_rows_fwd")
    dv_comp.add_output_from_dict("ac|geom|cabin|pitch")
    dv_comp.add_output_from_dict("ac|geom|cabin|percent_cross_section")

    # Electrical (geom)
    # dv_comp.add_output_from_dict("ac|geom|electrical|engine_power")
    dv_comp.add_output_from_dict("ac|geom|electrical|nacelle_distance")
    dv_comp.add_output_from_dict("ac|geom|electrical|max_voltage")

    # Weights
    dv_comp.add_output_from_dict("ac|weights|MTOW")
    # Add TOW for point_analysis, unless size_tow=True (then it's passed through via DVLabel)
    if point_analysis and not size_tow:
        dv_comp.add_output_from_dict("ac|weights|TOW")
    if not compute_oew:
        dv_comp.add_output_from_dict("ac|weights|OEW")
    if not compute_payload:
        dv_comp.add_output_from_dict("ac|weights|payload")
    dv_comp.add_output_from_dict("ac|weights|W_fuel_max")
    dv_comp.add_output_from_dict("ac|weights|MLW")

    # Propulsion
    dv_comp.add_output_from_dict("ac|propulsion|turbine|rating")
    # Only add motor rating from Excel data if NOT sizing the motor
    # When size_motor=True, motor rating will be provided externally as an input
    if not size_motor:
        dv_comp.add_output_from_dict("ac|propulsion|motor|rating")
    dv_comp.add_output_from_dict("ac|propulsion|prop|num_props")

    # Battery inputs
    dv_comp.add_output_from_dict("ac|propulsion|battery|n_str")


    dv_comp.add_output_from_dict("ac|propulsion|battery|cell_capacity")
    dv_comp.add_output_from_dict("ac|propulsion|battery|t_cell_init")
    dv_comp.add_output_from_dict("ac|propulsion|battery|m_cell")
    dv_comp.add_output_from_dict("ac|propulsion|battery|cp_cell")

    #dv_comp.add_output_from_dict("ac|propulsion|battery|i_sys_charge_lim")
    #dv_comp.add_output_from_dict("ac|propulsion|battery|v_sys_charge_lim")
    dv_comp.add_output_from_dict("ac|propulsion|battery|n_series_per_str")
    if not size_battery:
        dv_comp.add_output_from_dict("ac|propulsion|battery|n_parallel_per_str")  # Use battery Excel value
    
    # Battery energy and density (for BatteryWeight and NacelleParameterLinks)
    if compute_oew:
        dv_comp.add_output_from_dict("ac|propulsion|battery|batt_energy")
        dv_comp.add_output_from_dict("ac|propulsion|battery|batt_density")

    # Nacelle
    #dv_comp.add_output_from_dict("ac|propulsion|nacelle|max_rated_power")
    dv_comp.add_output_from_dict("ac|propulsion|nacelle|num_em_per_nac")
    dv_comp.add_output_from_dict("ac|propulsion|nacelle|num_turb_per_nac")

    # Propeller
    dv_comp.add_output_from_dict("ac|propulsion|prop|diameter")
    dv_comp.add_output_from_dict("ac|propulsion|prop|num_blades")

    # Motor (gearbox/weight calculation)
    dv_comp.add_output_from_dict("ac|propulsion|motor|power_density")
    dv_comp.add_output_from_dict("ac|propulsion|motor|n_engines")
    #dv_comp.add_output_from_dict("ac|propulsion|motor|HP")
    dv_comp.add_output_from_dict("ac|propulsion|motor|k")
    dv_comp.add_output_from_dict("ac|propulsion|motor|RPMin")
    dv_comp.add_output_from_dict("ac|propulsion|motor|RPMout")

    # Turbine
    #dv_comp.add_output_from_dict("ac|propulsion|turbine|HP")
    dv_comp.add_output_from_dict("ac|propulsion|turbine|k")
    dv_comp.add_output_from_dict("ac|propulsion|turbine|RPMin")
    dv_comp.add_output_from_dict("ac|propulsion|turbine|RPMout")

    # Gearbox
    dv_comp.add_output_from_dict("ac|propulsion|gearbox|counter_rotating")

    # Thermal Management System
    dv_comp.add_output_from_dict("ac|propulsion|tms|motor_efficiency")
    dv_comp.add_output_from_dict("ac|propulsion|tms|gearbox_heat")
    dv_comp.add_output_from_dict("ac|propulsion|tms|turbine_heat_per_nac")
    dv_comp.add_output_from_dict("ac|propulsion|battery|battery_heat_per_nacelle")
    dv_comp.add_output_from_dict("ac|propulsion|tms|specific_cooling_capacity_em_tms")
    dv_comp.add_output_from_dict("ac|propulsion|tms|specific_cooling_capacity_turbine_tms")
    dv_comp.add_output_from_dict("ac|propulsion|tms|specific_cooling_capacity_battery_tms")
    # Note: delta_T_em, delta_T_turb, delta_T_batt no longer needed (using kW/kg method)

    return dv_comp
