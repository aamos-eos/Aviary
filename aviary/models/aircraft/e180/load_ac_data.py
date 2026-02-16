from aviary.models.weights.wing_geometry_comp import compute_airfoil_pc

"""
Aircraft data loading and geometric computation utilities.

This module handles loading aircraft configuration data from Excel files and
computing derived geometric parameters with automatic unit handling.

Unit Handling
-------------
All geometric computation functions use OpenMDAO's unit conversion system to
ensure consistency:
- Input values are automatically converted to metric units (m, m^2, etc.)
- All calculations are performed in metric units
- Output values are returned in metric units with proper unit tags
- The get_value_in_units() helper function extracts values from data dictionaries
  and handles unit conversion transparently

This approach prevents unit conversion errors and allows input data to use
any supported unit system.
"""
import pandas as pd
import numpy as np
from aviary.models.engines.propulsion.battery.battery_data import BatteryData
import openmdao.api as om


def get_value_in_units(data_dict, target_units):
    """
    Extract value from data dictionary and convert to target units.
    
    Uses OpenMDAO's built-in unit conversion to ensure consistency across
    all geometric calculations. Handles dimensionless quantities and
    ensures safe conversion between unit systems.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with 'value' and 'units' keys
    target_units : str or None
        Target units for conversion (None for dimensionless)
        
    Returns
    -------
    float or str
        Value converted to target units (numeric), or original string value
        for non-numeric parameters (e.g., sizing_method)
    """
    value = data_dict["value"]
    source_units = data_dict.get("units", None)
    
    # If no units specified (dimensionless) or units match, return as-is
    # This also handles string values (e.g., sizing_method) which have no units
    if source_units is None or target_units is None or source_units == target_units:
        return value
    
    # Convert using OpenMDAO's unit conversion (for numeric values only)
    try:
        return om.convert_units(value, source_units, target_units)
    except Exception as e:
        raise ValueError(
            f"Failed to convert {value} from '{source_units}' to '{target_units}': {str(e)}"
        )



def chord_at_span(root_chord, taper, wing_half_span, span_y_pos):
    """Chord length at spanwise station y (ft from root), linear taper."""
    return root_chord * (1 + (taper - 1) * span_y_pos / wing_half_span)
    


def nacelle_wing_planform_interference_area( nacelle_width, root_chord, taper, wing_half_span, y_loc_centre):
    """
    Compute wing-nacelle planform interference area.

    Parameters
    ----------
    width_ft : float
        Nacelle width (ft).
    y_loc_in : float
        Spanwise centreline location of nacelle (in from root).

    Returns
    -------
    dict with all intermediate values and the interference area (ft²).
    """

    inner_span = y_loc_centre - nacelle_width / 2
    outer_span = y_loc_centre + nacelle_width / 2

    chord_inner = chord_at_span(root_chord, taper, wing_half_span, inner_span)
    chord_outer = chord_at_span(root_chord, taper, wing_half_span, outer_span)

    area = (chord_inner + chord_outer) / 2 * (outer_span - inner_span)  

    return area


def compute_wing_geometry(wing_data, nacelle_inbd_data, nacelle_outbd_data):
    """
    Compute derived wing geometric parameters from basic wing data.
    
    Handles both planform and trapezoidal area/span definitions.
    Uses trapezoidal definitions for chord and MAC calculations.
    
    All calculations are performed in meters and square meters, with automatic
    unit conversion for inputs and outputs.
    
    Parameters
    ----------
    wing_data : dict
        Dictionary containing wing geometric parameters with units.
        Required keys: S_plan, S_trap, span_plan, span_trap, taper, c0_sweep, toverc
        
    Returns
    -------
    dict
        Dictionary with computed wing parameters added including:
        AR_plan, AR_trap, c_root, c_tip, MAC, mac_buttline, S_wet, c4_sweep
    """
    # Extract basic parameters with unit conversion to metric
    span_plan = get_value_in_units(wing_data["span_plan"], "m")
    span_trap = get_value_in_units(wing_data["span_trap"], "m")
    area_plan = get_value_in_units(wing_data["S_plan"], "m**2")
    area_trap = get_value_in_units(wing_data["S_trap"], "m**2")
    taper_ratio = wing_data["taper"]["value"]  # dimensionless
    c0_sweep = get_value_in_units(wing_data["c0_sweep"], "deg")
    toverc = get_value_in_units(wing_data["toverc"], None)
    b_fairing_strt_y_loc = get_value_in_units(wing_data["b_fairing_strt_y_loc"], "m")
    inbd_nacelle_width = get_value_in_units(nacelle_inbd_data["width"], "m")
    inbd_nacelle_y_loc = get_value_in_units(nacelle_inbd_data["y_loc"], "m")
    outbd_nacelle_width = get_value_in_units(nacelle_outbd_data["width"], "m")
    outbd_nacelle_y_loc = get_value_in_units(nacelle_outbd_data["y_loc"], "m")

    airfoil_pc = compute_airfoil_pc(toverc)
    
    # Compute aspect ratios - both planform and trapezoidal
    ar_plan = span_plan**2 / area_plan
    ar_trap = span_trap**2 / area_trap
    
    # Use trapezoidal definitions for chord lengths and MAC (in meters)
    c_root = 2 * (span_trap / ar_trap) / (1 + taper_ratio)
    c_tip = c_root * taper_ratio
    
    # Compute Mean Aerodynamic Chord (MAC) using trapezoidal span (in meters)
    mac = 2/3 * c_root * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    
    # Compute MAC buttline position using trapezoidal span (in meters)
    mac_buttline = span_trap * (1 + 2 * taper_ratio) / (6 * (1 + taper_ratio))
    
    # Compute wing efficiency coefficient using planform span
    wing_eff = 1.0922 * 2 / span_plan # NOTE This formula only works for span in metric units...
    
    # Compute wetted area using planform definitions (in m^2)
    independent_wetted_area = (span_trap - wing_eff * span_trap) * (
        np.sqrt(area_trap / ar_trap) - (1 - taper_ratio) / (1 + taper_ratio) 
        * wing_eff * span_trap / ar_trap
    ) * airfoil_pc

    # Compute Interference Wetted Area for subtraction

    fuse_junction_chord = chord_at_span(c_root, taper_ratio, span_trap / 2, b_fairing_strt_y_loc)

    # Trapezoid: average of root chord & junction chord × fuselage radius
    wetted_area_wingtip_ft2 = 39.33
    wetted_area_wingtip_m2 = wetted_area_wingtip_ft2 * 0.3048**2
    fuse_unit_interf_area = (fuse_junction_chord + c_root) * b_fairing_strt_y_loc / 2   # ft² (one side pair = LS+RS)
    inbd_unit_interf_area = nacelle_wing_planform_interference_area(inbd_nacelle_width,  c_root, taper_ratio, span_trap / 2,inbd_nacelle_y_loc)
    outbd_unit_interf_area = nacelle_wing_planform_interference_area(outbd_nacelle_width, c_root, taper_ratio, span_trap / 2, outbd_nacelle_y_loc)

    wing_total_interf_area = (fuse_unit_interf_area + inbd_unit_interf_area + outbd_unit_interf_area/2) * 2
    wetted_area = independent_wetted_area - wing_total_interf_area + wetted_area_wingtip_m2


    # Compute quarter-chord sweep from leading edge sweep using trapezoidal AR
    # tan(c4_sweep) = tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper)
    c0_sweep_rad = c0_sweep * np.pi / 180.0
    c4_sweep_rad = np.arctan(np.tan(c0_sweep_rad) - (1/ar_trap) * (1 - taper_ratio) / (1 + taper_ratio))
    c4_sweep_deg = c4_sweep_rad * 180.0 / np.pi
    wing_data["c4_sweep"] = {"value": c4_sweep_deg, "units": "deg"}

    # Add computed values to wing_data (all in metric units)
    wing_data["AR_plan"] = {"value": ar_plan, "units": None}
    wing_data["AR_trap"] = {"value": ar_trap, "units": None}
    wing_data["c_root"] = {"value": c_root, "units": "m"}
    wing_data["c_tip"] = {"value": c_tip, "units": "m"}
    wing_data["MAC"] = {"value": mac, "units": "m"}
    wing_data["mac_buttline"] = {"value": mac_buttline, "units": "m"}
    wing_data["S_wet"] = {"value": wetted_area, "units": "m**2"}
    wing_data["AR"] = {"value": ar_plan, "units": None}
    wing_data["S_ref"] = {"value": area_plan, "units": "m**2"}
    wing_data["span"] = {"value": span_plan, "units": "m"}
    
    return wing_data


def compute_tail_geometry(tail_data, tail_type="horizontal", wing_data=None):
    """
    Compute derived tail geometric parameters from basic tail data.

    Computes tail area using one of two methods selected by 'sizing_method':
    - "volume_coeff": Uses dimensionless volume coefficient (V_ht or V_vt)
    - "tail_volume": Uses direct tail volume (m^3)
    
    Handles both planform and trapezoidal area/span definitions. 
    Uses trapezoidal definitions for chord and MAC calculations.
    
    All calculations are performed in meters and square meters, with automatic
    unit conversion for inputs and outputs.

    Parameters
    ----------
    tail_data : dict
        Dictionary containing tail geometric parameters with units:
        - sizing_method : str, either "volume_coeff" or "tail_volume"
        - If sizing_method="volume_coeff": must include "volume_coeff" (dimensionless)
        - If sizing_method="tail_volume": must include "tail_volume" (m^3)
        - Also requires: c4_to_wing_c4, span_plan, span_trap, taper, toverc
    tail_type : str
        Type of tail ("horizontal" or "vertical")
    wing_data : dict
        Dictionary containing wing geometric parameters (S_plan, MAC, span_trap, toverc) with units
        Only required if sizing_method="volume_coeff"

    Returns
    -------
    dict
        Dictionary with computed tail parameters added including:
        S_plan, S_trap, AR_plan, AR_trap, c_root, c_tip, MAC, S_wet
    """
    # Extract inputs with unit conversion to metric
    tail_lever_arm = get_value_in_units(tail_data["c4_to_wing_c4"], "m")
    span_plan = get_value_in_units(tail_data["span_plan"], "m")
    span_trap = get_value_in_units(tail_data["span_trap"], "m")
    taper_ratio = tail_data["taper"]["value"]  # dimensionless

    # Get tail t/c for airfoil perimeter coefficient
    toverc = get_value_in_units(tail_data["toverc"], None)
    airfoil_pc = compute_airfoil_pc(toverc)
    
    # Determine which method to use for computing tail area based on sizing_method
    # Must specify either "volume_coeff" or "tail_volume" as the sizing method
    if "sizing_method" not in tail_data:
        raise ValueError(
            "tail_data must contain 'sizing_method' key with value 'volume_coeff' or 'tail_volume'"
        )
    
    sizing_method = tail_data["sizing_method"]["value"] if isinstance(tail_data["sizing_method"], dict) else tail_data["sizing_method"]
    
    if sizing_method == "volume_coeff":
        # Method 1: Volume Coefficient (dimensionless)
        # Typical for preliminary design - V_ht or V_vt relates tail sizing to wing reference geometry
        volume_coeff = tail_data["volume_coeff"]["value"]  # dimensionless
        
        # Get wing reference values (use planform for area, trapezoidal for span)
        wing_area = get_value_in_units(wing_data["S_plan"], "m**2")
        wing_mac = get_value_in_units(wing_data["MAC"], "m")
        wing_span = get_value_in_units(wing_data["span_trap"], "m")
        
        # Compute tail planform area from volume coefficient
        # Horizontal: V_ht = (S_ht * l_ht) / (S_wing * MAC_wing)  =>  S_ht = V_ht * S_wing * MAC_wing / l_ht
        # Vertical:   V_vt = (S_vt * l_vt) / (S_wing * b_wing)    =>  S_vt = V_vt * S_wing * b_wing / l_vt
        if tail_type == "horizontal":
            area_trap = volume_coeff * wing_area * wing_mac / tail_lever_arm
        else:
            area_trap = volume_coeff * wing_area * wing_span / tail_lever_arm
            
    elif sizing_method == "tail_volume":
        # Method 2: Direct Volume (m^3 or m^2*m)
        # Used when tail size is already determined from detailed design
        # tail_volume represents the product of tail area and tail moment arm
        tail_volume = get_value_in_units(tail_data["tail_volume"], "m**3")
        
        # Compute tail planform area from direct volume
        # S_ht = V_ht / l_ht
        area_trap = tail_volume / tail_lever_arm
        
    else:
        raise ValueError(
            f"Invalid sizing_method '{sizing_method}'. Must be either 'volume_coeff' or 'tail_volume'"
        )

    # For trapezoidal area, scale by the ratio of spans squared
    # This assumes similar geometry between planform and trapezoidal
    # Scale Trapezoidal Area to Planform Area

    # Use trapezoidal definitions for chord lengths and MAC (in meters)
    ar_trap = span_trap**2 / area_trap
    c_root = 2 * (span_trap / ar_trap) / (1 + taper_ratio)
    c_tip = c_root * taper_ratio

    if tail_type == "horizontal":
        area_plan = area_trap * 0.95966 # Cutout Area Correction Factor based on GA Plan I 
    elif tail_type == "vertical":

        dorsal_fin_chord_fraction = get_value_in_units(tail_data["dorsal_fin_chord_fraction"], None)
        dorsal_fin_span_fraction = get_value_in_units(tail_data["dorsal_fin_span_fraction"], None)
        dorsal_fin_chord = c_root * dorsal_fin_chord_fraction
        dorsal_fin_span = span_trap * dorsal_fin_span_fraction
        dorsal_fin_area = dorsal_fin_chord * dorsal_fin_span/2
        area_plan = area_trap + dorsal_fin_area # Dorsal Fin Correction Factor: Eq 14 for Jets: https://www.fzt.haw-hamburg.de/pers/Scholz/Aero/AERO_PUB_INCAS_TailVolume_Vol13No3_2021.pdf

    # Compute aspect ratios - both planform and trapezoidal
    ar_plan = span_plan**2 / area_plan



    # Compute Mean Aerodynamic Chord (MAC) using trapezoidal span (in meters)
    mac = 2/3 * c_root * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)

    # Compute MAC buttline/height position using trapezoidal span (in meters)
    if tail_type == "horizontal":
        mac_position = span_trap * (1 + 2 * taper_ratio) / (6 * (1 + taper_ratio))
    else:  # vertical tail
        mac_position = span_trap * (1 + 2 * taper_ratio) / (3 * (1 + taper_ratio))

    # Compute wetted area using planform area (in m^2)
    wetted_area = area_plan * airfoil_pc



    # Add computed values to tail_data (all in metric units)
    tail_data["S_plan"] = {"value": area_plan, "units": "m**2"}
    tail_data["S_trap"] = {"value": area_trap, "units": "m**2"}
    tail_data["S_ref"] = {"value": area_plan, "units": "m**2"}  # Use planform for backward compatibility
    tail_data["AR_plan"] = {"value": ar_plan, "units": None}
    tail_data["AR_trap"] = {"value": ar_trap, "units": None}
    tail_data["c_root"] = {"value": c_root, "units": "m"}
    tail_data["c_tip"] = {"value": c_tip, "units": "m"}
    tail_data["MAC"] = {"value": mac, "units": "m"}
    if tail_type == "horizontal":
        tail_data["mac_buttline"] = {"value": mac_position, "units": "m"}
    else:
        tail_data["mac_height"] = {"value": mac_position, "units": "m"}
    tail_data["S_wet"] = {"value": wetted_area, "units": "m**2"}
    tail_data["AR"] = {"value": ar_plan, "units": None}
    tail_data["S_ref"] = {"value": area_plan, "units": "m**2"}
    tail_data["span"] = {"value": span_plan, "units": "m"}

    return tail_data


def compute_fuselage_geometry(fuse_data, cabin_data):
    """
    Compute derived fuselage geometric parameters from basic fuselage data.
    
    All calculations are performed in meters and square meters, with automatic
    unit conversion for inputs and outputs.
    
    Parameters
    ----------
    fuse_data : dict
        Dictionary containing fuselage geometric parameters with units
        
    Returns
    -------
    dict
        Dictionary with computed fuselage parameters added
    """
    # Extract basic parameters with unit conversion to metric
    cabin_length = get_value_in_units(cabin_data["cabin_length"], "m")
    nose_tail_length = get_value_in_units(fuse_data["nose_tail_length"], "m")
    constant_sec_length = get_value_in_units(fuse_data["const_sec_length"], "m")
    height = get_value_in_units(fuse_data["height"], "m")
    width = get_value_in_units(fuse_data["width"], "m")
    aft_fuse_area = get_value_in_units(fuse_data["aft_fuse_area"], "m**2")
    cockpit_area = get_value_in_units(fuse_data["cockpit_area"], "m**2")
    belly_fairing_corr = get_value_in_units(fuse_data["belly_fairing_corr"], None)
    


    constant_sec_eff_diam = (height + width) / 2
    constant_sec_circ = constant_sec_eff_diam * np.pi

    constant_sec_area = constant_sec_length * constant_sec_circ

    wetted_area = (constant_sec_area + aft_fuse_area + cockpit_area) * belly_fairing_corr

    total_length = cabin_length + nose_tail_length

    cross_sec = height * width
    
    # Compute fineness ratio (dimensionless)
    # diameter = 2 * sqrt(cross_section / pi)
    fineness_ratio = total_length / (2 * np.sqrt(cross_sec / np.pi))
    
    # Add computed values to fuse_data (all in metric units)
    fuse_data["length"] = {"value": total_length, "units": "m"}
    fuse_data["fineness_ratio"] = {"value": fineness_ratio, "units": None}
    fuse_data["S_wet"] = {"value": wetted_area, "units": "m**2"}
    
    return fuse_data


def compute_nacelle_geometry(nacelle_data, prop_num=4):
    """
    Compute derived nacelle geometric parameters from basic nacelle data.
    
    All calculations are performed in meters and square meters, with automatic
    unit conversion for inputs and outputs.
    
    Parameters
    ----------
    nacelle_data : dict
        Dictionary containing nacelle geometric parameters with units
    prop_num : int, optional
        Number of propulsors (default: 4). Reserved for future use.
        
    Returns
    -------
    dict
        Dictionary with computed nacelle parameters added
    """
    # Extract basic parameters with unit conversion to metric
    length = get_value_in_units(nacelle_data["length"], "m")
    height = get_value_in_units(nacelle_data["height"], "m")
    width = get_value_in_units(nacelle_data["width"], "m")

    
    # Compute effective diameter (in meters)
    cross_sec = height * width
    diameff = np.sqrt(4/np.pi * cross_sec )
    
    # Compute fineness eratio (dimensionless)
    fineness_ratio = length / diameff
    
    # Extract wetted area with unit conversion
    wetted_area = get_value_in_units(nacelle_data["S_wet"], "m**2")
    
    # Add computed values to nacelle_data (all in metric units)
    nacelle_data["diameter_eff"] = {"value": diameff, "units": "m"}
    nacelle_data["fineness_ratio"] = {"value": fineness_ratio, "units": None}
    nacelle_data["S_wet"] = {"value": wetted_area, "units": "m**2"}
    
    return nacelle_data

def _collect_ac_data_lines(ac_data, indent=0):
    """Recursively collect formatted aircraft data lines into a list."""
    lines = []
    indent_str = "  " * indent
    for key, value in ac_data.items():
        if isinstance(value, dict):
            if 'value' in value and 'units' in value:
                units_str = f" [{value['units']}]" if value['units'] else ""
                lines.append(f"{indent_str}{key}: {value['value']}{units_str}")
            else:
                lines.append(f"{indent_str}{key}:")
                lines.extend(_collect_ac_data_lines(value, indent + 1))
        else:
            lines.append(f"{indent_str}{key}: {value}")
    return lines


def print_ac_data(ac_data, indent=0, output_file=None):
    """
    Print aircraft data dictionary in a neat, hierarchical format.
    
    Parameters
    ----------
    ac_data : dict
        Aircraft data dictionary to print
    indent : int
        Current indentation level for nested printing
    output_file : str or None, optional
        If provided, saves the output to a text file at this path
        in addition to printing to the console.
    """
    indent_str = "  " * indent
    
    for key, value in ac_data.items():
        if isinstance(value, dict):
            # Check if this is a leaf node (has 'value' and 'units' keys)
            if 'value' in value and 'units' in value:
                units_str = f" [{value['units']}]" if value['units'] else ""
                print(f"{indent_str}{key}: {value['value']}{units_str}")
            else:
                # This is a nested dictionary, print the key and recurse
                print(f"{indent_str}{key}:")
                print_ac_data(value, indent + 1)
        else:
            # Direct value
            print(f"{indent_str}{key}: {value}")
    
    # Save to file when at top level and output_file is specified
    if output_file is not None and indent == 0:
        lines = _collect_ac_data_lines(ac_data)
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nOutput saved to: {output_file}")


def load_ac_data_from_excel(filename, bat_filename, cell_sheetname, config_sheetname, validation_mode=False):
    data = dict()
    ac = dict()

    # Sheets correspond to top-level categories
    for sheet in ["aero", "geom", "weights", "propulsion"]:
        df = pd.read_excel(filename, sheet_name=sheet)

        section = dict()
        for _, row in df.iterrows():
            key = row["key"]
            subkey = row["subkey"]
            
            # Try to convert to float, but keep as string if conversion fails
            # String-valued parameters include: sizing_method (for tail geometry)
            # This allows flexible data types in the Excel configuration
            raw_value = row["value"]
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                value = raw_value  # Keep as string for non-numeric parameters
            
            units = row.get("units", None)

            if pd.isna(subkey):
                # direct value
                section[key] = {"value": value}
                if pd.notna(units):
                    section[key]["units"] = units
                else:
                    section[key]["units"] = None
            else:
                # nested dictionary
                if key not in section:
                    section[key] = dict()
                section[key][subkey] = {"value": value}
                if pd.notna(units):
                    section[key][subkey]["units"] = units
                else:
                    section[key][subkey]["units"] = None

        ac[sheet] = section

    # Compute derived geometric parameters
    if "geom" in ac:
        # Compute wing geometry
        if "wing" in ac["geom"]:
            ac["geom"]["wing"] = compute_wing_geometry(ac["geom"]["wing"], ac["geom"]["nacelle_inboard"], ac["geom"]["nacelle_outboard"])
        
        # Compute fuselage geometry first (needed for tail lever arms)
        if "fuselage" in ac["geom"]:
            ac["geom"]["fuselage"] = compute_fuselage_geometry(ac["geom"]["fuselage"], ac["geom"]["cabin"])
        
        # Compute tail lever arms from fuselage length, then tail geometry
        if "fuselage" in ac["geom"]:
            fuse_length = ac["geom"]["fuselage"]["length"]["value"]
            
            if "hstab" in ac["geom"]:
                hstab_ratio = ac["geom"]["hstab"]["lever_arm_fuse_length_ratio"]["value"]
                ac["geom"]["hstab"]["c4_to_wing_c4"] = {"value": hstab_ratio * fuse_length, "units": "m"}
            if "vstab" in ac["geom"]:
                vstab_ratio = ac["geom"]["vstab"]["lever_arm_fuse_length_ratio"]["value"]
                ac["geom"]["vstab"]["c4_to_wing_c4"] = {"value": vstab_ratio * fuse_length, "units": "m"}
        
        # Compute horizontal tail geometry
        if "hstab" in ac["geom"]:
            ac["geom"]["hstab"] = compute_tail_geometry(ac["geom"]["hstab"], "horizontal", ac["geom"]["wing"])
        
        # Compute vertical tail geometry
        if "vstab" in ac["geom"]:
            ac["geom"]["vstab"] = compute_tail_geometry(ac["geom"]["vstab"], "vertical", ac["geom"]["wing"])
        
        # Compute nacelle geometry
        # Prefer explicit inboard/outboard nacelle keys when available.
        prop_num = 4  # default
        if "propulsion" in ac and "num_propulsors" in ac["propulsion"]:
            prop_num = ac["propulsion"]["num_propulsors"]["value"]

        if "nacelle_inboard" in ac["geom"]:
            ac["geom"]["nacelle_inboard"] = compute_nacelle_geometry(ac["geom"]["nacelle_inboard"], prop_num)
        if "nacelle_outboard" in ac["geom"]:
            ac["geom"]["nacelle_outboard"] = compute_nacelle_geometry(ac["geom"]["nacelle_outboard"], prop_num)

    # Validation mode: override computed wetted areas with hardcoded values (in ft², converted to m²)
    if validation_mode: # PLAN I Data:
        # Conversion factor: 1 ft² = 0.092903 m²
        FT2_TO_M2 = 0.092903
        
        # Hardcoded wetted areas in ft² (validation case)
        validation_wetted_areas_ft2 = {
            "wing": 1486 + 39,  # 1525 ft²
            "vstab": 352.2,
            "hstab": 424.25,
            "nacelle_inboard": 405.49,
            "nacelle_outboard": 307.39,
            "fuselage": 2728.0
        }
        
        # Override computed wetted areas with validation values
        if "wing" in ac["geom"]:
            ac["geom"]["wing"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["wing"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        if "vstab" in ac["geom"]:
            ac["geom"]["vstab"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["vstab"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        if "hstab" in ac["geom"]:
            ac["geom"]["hstab"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["hstab"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        if "nacelle_inboard" in ac["geom"]:
            ac["geom"]["nacelle_inboard"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["nacelle_inboard"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        if "nacelle_outboard" in ac["geom"]:
            ac["geom"]["nacelle_outboard"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["nacelle_outboard"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        if "fuselage" in ac["geom"]:
            ac["geom"]["fuselage"]["S_wet"] = {
                "value": validation_wetted_areas_ft2["fuselage"] * FT2_TO_M2,
                "units": "m**2"
            }
        
        print("\n" + "=" * 80)
        print("PLAN I Wetted Areas Validation Mode: Using hardcoded wetted areas")
        print("=" * 80)
        for component, area_ft2 in validation_wetted_areas_ft2.items():
            area_m2 = area_ft2 * FT2_TO_M2
            print(f"{component:20s}: {area_ft2:>10.2f} ft² = {area_m2:>10.4f} m²")
        print("=" * 80 + "\n")

    if bat_filename is not None:
        bat_data = BatteryData.get_data(bat_filename=bat_filename, 
                                                cell_sheetname=cell_sheetname, 
                                                config_sheetname=config_sheetname)


        ac["propulsion"]["battery"]["cell_capacity"] = {"value": bat_data.cell_Ah_capacity, "units": "A*h"}
        ac["propulsion"]["battery"]["m_cell"] = {"value": bat_data.m_cell, "units": "kg"}
        ac["propulsion"]["battery"]["cp_cell"] = {"value": bat_data.cp_cell, "units": "J/kg/K"}
        ac["propulsion"]["battery"]["n_series_per_str"] = {"value": bat_data.n_series_per_str}
        ac["propulsion"]["battery"]["n_parallel_per_str"] = {"value": bat_data.n_parallel_per_str}
        ac["propulsion"]["battery"]["n_str"] = {"value": bat_data.n_str}

    motor_rating_val = ac["propulsion"]["motor"]["cont_rating_per_nac"]["value"]
    motor_rating_units = ac["propulsion"]["motor"]["cont_rating_per_nac"].get("units", None)
    
    em_eff = 0.95
    batt_eff = 0.95

    systems_max_pow = 130


    battery_heat = (motor_rating_val / em_eff + systems_max_pow) / batt_eff * (1 - batt_eff)
    

    ac["propulsion"]["battery"]["battery_heat_per_nacelle"] = {
        "value": battery_heat,
        "units": motor_rating_units
    }

    data["ac"] = ac

    return data


def print_wetted_areas(ac_data):
    """
    Print wetted areas of all aircraft components in ft².
    
    Parameters
    ----------
    ac_data : dict
        Aircraft data dictionary containing geometry information
    """
    # Conversion factor: 1 m² = 10.7639 ft²
    M2_TO_FT2 = 10.7639
    
    ac = ac_data.get("ac", {})
    geom = ac.get("geom", {})
    
    print("\n" + "=" * 80)
    print("AIRCRAFT WETTED AREAS")
    print("=" * 80)
    
    total_wetted_area_ft2 = 0.0
    
    # Wing
    if "wing" in geom and "S_wet" in geom["wing"]:
        wing_wet_m2 = geom["wing"]["S_wet"]["value"]
        wing_wet_ft2 = wing_wet_m2 * M2_TO_FT2
        total_wetted_area_ft2 += wing_wet_ft2
        print(f"Wing:              {wing_wet_ft2:>12.2f} ft²")
    
    # Horizontal tail
    if "hstab" in geom and "S_wet" in geom["hstab"]:
        hstab_wet_m2 = geom["hstab"]["S_wet"]["value"]
        hstab_wet_ft2 = hstab_wet_m2 * M2_TO_FT2
        total_wetted_area_ft2 += hstab_wet_ft2
        print(f"Horizontal Tail:   {hstab_wet_ft2:>12.2f} ft²")
    
    # Vertical tail
    if "vstab" in geom and "S_wet" in geom["vstab"]:
        vstab_wet_m2 = geom["vstab"]["S_wet"]["value"]
        vstab_wet_ft2 = vstab_wet_m2 * M2_TO_FT2
        total_wetted_area_ft2 += vstab_wet_ft2
        print(f"Vertical Tail:    {vstab_wet_ft2:>12.2f} ft²")
    
    # Fuselage
    if "fuselage" in geom and "S_wet" in geom["fuselage"]:
        fuse_wet_m2 = geom["fuselage"]["S_wet"]["value"]
        fuse_wet_ft2 = fuse_wet_m2 * M2_TO_FT2
        total_wetted_area_ft2 += fuse_wet_ft2
        print(f"Fuselage:          {fuse_wet_ft2:>12.2f} ft²")
    
    # Inboard nacelles (per nacelle, then total for 2 nacelles)
    if "nacelle_inboard" in geom and "S_wet" in geom["nacelle_inboard"]:
        nac_inbd_wet_m2 = geom["nacelle_inboard"]["S_wet"]["value"]
        nac_inbd_wet_ft2 = nac_inbd_wet_m2 * M2_TO_FT2
        nac_inbd_total_ft2 = nac_inbd_wet_ft2 * 2  # 2 inboard nacelles
        total_wetted_area_ft2 += nac_inbd_total_ft2
        print(f"Inboard Nacelle:   {nac_inbd_wet_ft2:>12.2f} ft² (per nacelle)")
        print(f"Inboard Nacelles:  {nac_inbd_total_ft2:>12.2f} ft² (2 × {nac_inbd_wet_ft2:.2f})")
    
    # Outboard nacelles (per nacelle, then total for 2 nacelles)
    if "nacelle_outboard" in geom and "S_wet" in geom["nacelle_outboard"]:
        nac_outbd_wet_m2 = geom["nacelle_outboard"]["S_wet"]["value"]
        nac_outbd_wet_ft2 = nac_outbd_wet_m2 * M2_TO_FT2
        nac_outbd_total_ft2 = nac_outbd_wet_ft2 * 2  # 2 outboard nacelles
        total_wetted_area_ft2 += nac_outbd_total_ft2
        print(f"Outboard Nacelle:  {nac_outbd_wet_ft2:>12.2f} ft² (per nacelle)")
        print(f"Outboard Nacelles:  {nac_outbd_total_ft2:>12.2f} ft² (2 × {nac_outbd_wet_ft2:.2f})")
    
    print("-" * 80)
    print(f"Total Wetted Area: {total_wetted_area_ft2:>12.2f} ft²")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    filename = "models/atlas/atlas/scenarios/setup_mission/ac_data.xlsx"
    bat_filename = 'models/atlas/atlas/propulsion/empirical_data/MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent.xlsx'
    cell_sheetname = 'BOL_cell_fct_CRate'
    config_sheetname = 'battery_config'

    ac_data = load_ac_data_from_excel(filename=filename,
    bat_filename=bat_filename,
    cell_sheetname=cell_sheetname,
    config_sheetname=config_sheetname)

    # Print wetted areas
    print_wetted_areas(ac_data)

    # Print aircraft data in a neat format
    print("=" * 80)
    print("AIRCRAFT DATA SUMMARY")
    print("=" * 80)
    print_ac_data(ac_data, output_file="ac_data_summary.txt")
    print("=" * 80)
