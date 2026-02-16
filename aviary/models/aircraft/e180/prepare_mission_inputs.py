# DATA FOR King Air C90GT
# Collected from AOPA Pilot article
# and rough photogrammetry
import numpy as np
from openmdao.api import (
    IndepVarComp,
)


def prepare_mission_inputs(num_nodes=11, ac_data=None, mission_config=None, turb_op_array = None, opt_mission=False, opt_objective=None):

    ivc = IndepVarComp()

    npp = int(round(ac_data["propulsion"]["prop"]["num_props"]["value"], 0)) # per aircraft
    nm = int(round(ac_data["propulsion"]["nacelle"]["num_em_per_nac"]["value"], 0))
    nt = int(round(ac_data["propulsion"]["nacelle"]["num_turb_per_nac"]["value"], 0))
    nac = int(round(ac_data["propulsion"]["nacelle"]["num_nac"]["value"], 0))  
    n_str = int(round(ac_data["propulsion"]["battery"]["n_str"]["value"], 0))
    keys = [k for k in mission_config['mission']['phase_names'] if k not in ["mission", "reserve"]]

    num_turb_op = np.sum(turb_op_array)

    #key = next(k for k, v in mission_config.items() if v == 'taxi')
    #first_phase_index = next(i for i, name in enumerate(mission_config['mission']['phase_names']) 
    #                    if name not in ["mission", "reserve"])


    
    if any("reserve" in k for k in keys):
        rsv_mission = True
    else:
        rsv_mission = False
    


    ivc.add_output(f"nacelle_max_rated_power", ac_data["propulsion"]["nacelle"]["max_rated_power"]["value"] * np.ones(num_nodes), units= ac_data["propulsion"]["nacelle"]["max_rated_power"]["units"])
    # end 
    if not mission_config[f"{keys[0]}"]["point_analysis"]:
        ivc.add_output(f"{keys[0]}|soc_initial", mission_config[f"{keys[0]}"]["trajectory"]["soc_initial"]["value"], units=mission_config[f"{keys[0]}"]["trajectory"]["soc_initial"]["units"])
        ivc.add_output(f"{keys[0]}|range_initial", mission_config[f"{keys[0]}"]["trajectory"]["range_initial"]["value"], units=mission_config[f"{keys[0]}"]["trajectory"]["range_initial"]["units"])
        ivc.add_output(f"{keys[0]}|fuel_used_initial", mission_config[f"{keys[0]}"]["trajectory"]["fuel_used_initial"]["value"], units=mission_config[f"{keys[0]}"]["trajectory"]["fuel_used_initial"]["units"])
        ivc.add_output(f"{keys[0]}|h0", mission_config[f"{keys[0]}"]["trajectory"]["h0"]["value"], units=mission_config[f"{keys[0]}"]["trajectory"]["h0"]["units"])

    ivc.add_output(f"{keys[0]}|t0", 0, units='s')

    if "holding_cruise" in keys:
        if not mission_config[f"holding_cruise"]["point_analysis"]:
            ivc.add_output(f"holding_cruise|ode_integ_phase|fltcond|h_initial", mission_config[f"holding_cruise"]["trajectory"]["h0"]["value"], units=mission_config[f"holding_cruise"]["trajectory"]["h0"]["units"])

    if mission_config[f"{keys[0]}"]["point_analysis"]:
        ivc.add_output(f"{keys[0]}|fltcond|h", mission_config[f"{keys[0]}"]["fltcond"]["h"]["value"], units=mission_config[f"{keys[0]}"]["fltcond"]["h"]["units"])
    #ivc.add_output(f"climbdt|cruise|h0", mission_config["cruise_1"]["trajectory"]["h0"]["value"], units=mission_config["cruise_1"]["trajectory"]["h0"]["units"])
    #ivc.add_output("descentdt|h1", mission_config["descent"]["trajectory"]["h1"]["value"], units=mission_config["descent"]["trajectory"]["h1"]["units"])
    #ivc.add_output("climbdt|h0", mission_config["climb"]["trajectory"]["h0"]["value"], units=mission_config["climb"]["trajectory"]["h0"]["units"])
    #ivc.add_output("phased|mission_range", mission_config["mission"]["range"]["value"], units=mission_config["mission"]["range"]["units"])

    #if rsv_mission:
    #    ivc.add_output("reserve_cruisedt|reserve_range", mission_config["reserve"]["range"]["value"], units=mission_config["reserve"]["range"]["units"])
    # end 

    if opt_mission:
        if opt_objective == "range":
            ivc.add_output("cruise_duration", val=mission_config["mission"]["cruise_duration"]["value"], units=mission_config["mission"]["cruise_duration"]["units"])
        ivc.add_output("hy_cruise_t_ratio", val=mission_config["mission"]["hy_cruise_t_ratio"]["value"], units=mission_config["mission"]["hy_cruise_t_ratio"]["units"])
        # GT cruise power derate factor (1.0 = MCP, 0.5 = 50% of MCP) - scalar, applies to all GTs equally
        # Load from mission config if specified, otherwise default to 1.0
        gt_derate_config = mission_config.get("cruise_1", {}).get("propulsion", {}).get("nacelle", {}).get("gt_derate")
        gt_derate_val = gt_derate_config["value"] if gt_derate_config is not None else 1.0
        ivc.add_output("cruise_1|gt_derate", val=gt_derate_val, units=None)
        #elif opt_objective == "fuel":
        #    ivc.add_output("e_cruise_d_ratio", val=mission_config["mission"]["e_cruise_d_ratio"]["value"], units=mission_config["mission"]["e_cruise_d_ratio"]["units"])

    ivc.add_output("ac|propulsion|battery|q_cool_bat", ac_data["propulsion"]["battery"]["q_cool_bat"]["value"] * np.ones((n_str, num_nodes)), units=ac_data["propulsion"]["battery"]["q_cool_bat"]["units"])
    # Propeller diameter needs to be a matrix (num_props, num_nodes) for the parallel hybrid system

    ivc.add_output("ac|propulsion|propeller|diameter", ac_data["propulsion"]["prop"]["diameter"]["value"] * np.ones((npp, num_nodes)), units=ac_data["propulsion"]["prop"]["diameter"]["units"])

    #ivc.add_output("descent|descentdt|approach|h", 35, units="ft")

    phase_names = mission_config['mission']['phase_names']

    # Add cruise hybridization ratio IVCs for optimization
    if "cruise_1" in phase_names:  
        if mission_config["cruise_1"]["propulsion"]["s_curve_hy_profile"]:
            const_hy_ratio = mission_config["cruise_1"]["propulsion"]["const_hy_ratio"]

            if mission_config["cruise_1"]["propulsion"]["eq_hy_nacelles_op"]:
                if const_hy_ratio:  # constant
                    ivc.add_output(f"cruise_1|hy_ratio", val=mission_config["cruise_1"]["propulsion"]["nacelle"]["power_split_fraction_gt"]["value"], units=None, 
                                        desc="Cruise hybridization ratio (power split fraction GT) for all nacelles")
                else:  # transient - use smootherstep function
                    ivc.add_output("cruise_1|hy_ratio_start", val=mission_config["cruise_1"]["propulsion"]["nacelle"]["hy_ratio_start"]["value"], units=None, 
                                        desc="Cruise hybridization ratio start for all nacelles")
                    ivc.add_output("cruise_1|hy_ratio_end", val=mission_config["cruise_1"]["propulsion"]["nacelle"]["hy_ratio_end"]["value"], units=None, 
                                        desc="Cruise hybridization ratio end for all nacelles")
                    ivc.add_output("cruise_1|hy_ratio_loc_trans", val=mission_config["cruise_1"]["propulsion"]["nacelle"]["hy_ratio_loc_trans"]["value"], units=None, 
                                        desc="Cruise hybridization ratio tau for all nacelles")
                    ivc.add_output("cruise_1|hy_ratio_w_trans", val=mission_config["cruise_1"]["propulsion"]["nacelle"]["hy_ratio_w_trans"]["value"], units=None, 
                                        desc="Cruise hybridization ratio transition width for all nacelles")
            else:
                for i in range(npp):
                    if turb_op_array[i] == 1:
                        # Add IVC for this nacelle's hybridization ratio
                        hy_ratio = mission_config["cruise_1"]["propulsion"]["nacelle"][f"power_split_fraction_gt_nac{i+1}"]["value"]
                        if const_hy_ratio:  # constant
                            # Scalar input for constant hybridization
                            ivc.add_output(f"cruise_1|hy_ratio_nac{i+1}", val=hy_ratio, units=None, 
                                        desc=f"Cruise hybridization ratio for nacelle {i+1}")
                        else:  # transient
                            ivc.add_output(f"cruise_1|hy_ratio_nac{i+1}_start", val=0.5, units=None, 
                                        desc=f"Cruise hybridization ratio start for nacelle {i+1}")
                            ivc.add_output(f"cruise_1|hy_ratio_nac{i+1}_end", val=0.5, units=None, 
                                                desc=f"Cruise hybridization ratio end for nacelle {i+1}")
                            ivc.add_output(f"cruise_1|hy_ratio_nac{i+1}_loc_trans", val=0.5, units=None, 
                                                desc=f"Cruise hybridization ratio loc_trans for nacelle {i+1}")
                            ivc.add_output(f"cruise_1|hy_ratio_nac{i+1}_w_trans", val=0.05, units=None, 
                                                desc=f"Cruise hybridization ratio transition width for nacelle {i+1}")

    # Add climb_hy power control IVCs for optimization
    if "climb_hy" in phase_names:  
        if mission_config["climb_hy"]["propulsion"]["s_curve_power_profile"]:

            if mission_config["climb_hy"]["propulsion"]["eq_power_nacelles_op"]:

                ivc.add_output("climb_hy|total_mech_power_out_start", val=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_start"]["value"], units=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_start"]["units"], 
                                    desc="Climb_hy total mechanical power out start for all nacelles")
                ivc.add_output("climb_hy|total_mech_power_out_end", val=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_end"]["value"], units=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_end"]["units"], 
                                    desc="Climb_hy total mechanical power out end for all nacelles")
                ivc.add_output("climb_hy|total_mech_power_out_loc_trans", val=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_loc_trans"]["value"], units=None, 
                                    desc="Climb_hy total mechanical power out loc_trans for all nacelles")
                ivc.add_output("climb_hy|total_mech_power_out_w_trans", val=mission_config["climb_hy"]["propulsion"]["nacelle"]["total_mech_power_out_w_trans"]["value"], units=None, 
                                    desc="Climb_hy total mechanical power out transition width for all nacelles")
        else:
                for i in range(npp):
                    if turb_op_array[i] == 1:
                        # Add IVC for this nacelle's power control
                        ivc.add_output(f"climb_hy|total_mech_power_out_nac{i+1}_start", val=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_start"]["value"], units=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_start"]["units"], 
                                    desc=f"Climb_hy total mechanical power out start for nacelle {i+1}")
                        ivc.add_output(f"climb_hy|total_mech_power_out_nac{i+1}_end", val=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_end"]["value"], units=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_end"]["units"], 
                                            desc=f"Climb_hy total mechanical power out end for nacelle {i+1}")
                        ivc.add_output(f"climb_hy|total_mech_power_out_nac{i+1}_loc_trans", val=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_loc_trans"]["value"], units=None, 
                                            desc=f"Climb_hy total mechanical power out loc_trans for nacelle {i+1}")
                        ivc.add_output(f"climb_hy|total_mech_power_out_nac{i+1}_w_trans", val=mission_config["climb_hy"]["propulsion"]["nacelle"][f"total_mech_power_out_nac{i+1}_w_trans"]["value"], units=None, 
                                            desc=f"Climb_hy total mechanical power out transition width for nacelle {i+1}")


    # Phase Specific Variables
    for idx, phase_name in enumerate(phase_names):
        #print(f"Setting up inputs for {phase_name}")
        if phase_name != "mission":

            # Get configuration parameters
            velocity_in = mission_config[phase_name]["velocity_in"]
            vs_in = mission_config[phase_name]["vs_in"]
            gamma_in = mission_config[phase_name]["gamma_in"]
            power_in = mission_config[phase_name]["power_in"]
            duration_set = mission_config[phase_name]["duration_set"]
            match_alt = mission_config[phase_name]["match_alt"]
            match_speed = mission_config[phase_name]["match_speed"]
            match_range = mission_config[phase_name]["match_range"]
            phase_type = mission_config[phase_name]["phase_type"]
            true_airspeed_in = mission_config[phase_name]["true_airspeed_in"]


            if duration_set:
                ivc.add_output(f"{phase_name}|duration", mission_config[phase_name]["trajectory"]["duration"]["value"], units=mission_config[phase_name]["trajectory"]["duration"]["units"])

            if not mission_config[phase_name]["point_analysis"]:
                if match_alt:
                    if idx == 0:
                        ivc.add_output(f"{keys[idx]}|altitude_duration|y0", mission_config[keys[idx]]["trajectory"]["h0"]["value"], units=mission_config[keys[idx]]["trajectory"]["h0"]["units"])
                    elif mission_config[phase_names[idx-1]]["phase_type"] == "power_only":
                        ivc.add_output(f"{keys[idx]}|altitude_duration|y0", mission_config[keys[idx]]["trajectory"]["h0"]["value"], units=mission_config[keys[idx]]["trajectory"]["h0"]["units"])

                    ivc.add_output(f"{keys[idx]}|altitude_duration|y_target", mission_config[keys[idx]]["trajectory"]["h1"]["value"], units=mission_config[keys[idx]]["trajectory"]["h1"]["units"])

                    #if "descent" not in phase_name and "approach" not in phase_name:
                    #    ivc.add_output(f"{keys[idx]}|{phase_name}dt|{keys[idx+1]}|h0", mission_config[keys[idx]]["trajectory"]["h1"]["value"], units=mission_config[keys[idx]]["trajectory"]["h1"]["units"])
                    #else:
                    #    ivc.add_output(f"{keys[idx]}|{phase_name}dt|{keys[idx]}|h1", mission_config[keys[idx]]["trajectory"]["h1"]["value"], units=mission_config[keys[idx]]["trajectory"]["h1"]["units"])
                elif match_speed:
                    ivc.add_output(f"{keys[idx]}|speed_duration|y0", mission_config[keys[idx]]["trajectory"]["Utrue_0"]["value"], units=mission_config[keys[idx]]["trajectory"]["Utrue_0"]["units"])
                    ivc.add_output(f"{keys[idx]}|speed_duration|y_target", mission_config[keys[idx]]["trajectory"]["Utrue_1"]["value"], units=mission_config[keys[idx]]["trajectory"]["Utrue_1"]["units"])
                elif match_range:   

                    # Cruise 1 and Reserve Cruise use special duration functions because their range is dependant on the ranges of other segments
                    if phase_name == "cruise_1":
                        if not duration_set:
                            ivc.add_output(f"{keys[idx]}dt|mission_range", mission_config["mission"]["range"]["value"], units=mission_config["mission"]["range"]["units"])
                    elif phase_name == "reserve_cruise":
                        ivc.add_output(f"{keys[idx]}dt|reserve_range", mission_config["reserve"]["range"]["value"], units=mission_config["reserve"]["range"]["units"])
                    elif (phase_name != "cruise_2" or not opt_mission):
                        ivc.add_output(f"{keys[idx]}|range_duration|y_target", mission_config[keys[idx]]["trajectory"]["range_target"]["value"], units=mission_config[keys[idx]]["trajectory"]["range_target"]["units"])


            if phase_type != "power_only":
                ivc.add_output(f"{phase_name}|flap_angle", mission_config[phase_name]["fltcond"]["flap_angle"]["value"], units=mission_config[phase_name]["fltcond"]["flap_angle"]["units"])
            ivc.add_output(f"{phase_name}|fltcond|TempIncrement", mission_config[phase_name]["fltcond"]["disa"]["value"], units=mission_config[phase_name]["fltcond"]["disa"]["units"])
            ivc.add_output(f"{phase_name}|fltcond|disa", val=mission_config[phase_name]["fltcond"]["disa"]["value"], units=mission_config[phase_name]["fltcond"]["disa"]["units"])
            ivc.add_output(f"{phase_name}|eta_parc", mission_config[phase_name]["propulsion"]["eta_parc"]["value"], units=mission_config[phase_name]["propulsion"]["eta_parc"]["units"])

            if vs_in or phase_type == "steady_eas":
                ivc.add_output(f"{phase_name}|fltcond|vs", val=mission_config[phase_name]["fltcond"]["vs"]["value"], units=mission_config[phase_name]["fltcond"]["vs"]["units"])
            if gamma_in or phase_type == "steady_eas":
                ivc.add_output(f"{phase_name}|fltcond|gamma", val=mission_config[phase_name]["fltcond"]["gamma"]["value"], units=mission_config[phase_name]["fltcond"]["gamma"]["units"])
            if (velocity_in or phase_type == "steady_eas") and not true_airspeed_in:
                ivc.add_output(f"{phase_name}|fltcond|Ueas", val=mission_config[phase_name]["fltcond"]["Ueas"]["value"], units=mission_config[phase_name]["fltcond"]["Ueas"]["units"])

            if true_airspeed_in:
                ivc.add_output(f"{phase_name}|fltcond|Utrue", val=mission_config[phase_name]["fltcond"]["Utrue"]["value"], units=mission_config[phase_name]["fltcond"]["Utrue"]["units"])

            #if vs_in and power_in and phase_type == "unsteady_eas":
            #    ivc.add_output(f"{phase_name}|fltcond|Utrue_initial", val=mission_config[phase_name]["fltcond"]["Utrue"], units=None)

            #if gamma_in and power_in and phase_type == "unsteady_eas":
            #    ivc.add_output(f"{phase_name}|fltcond|Utrue_initial", val=mission_config[phase_name]["fltcond"]["gamma"], units=None)

            if mission_config[phase_name]["propulsion"]["eq_power_nacelles_op"] or num_turb_op == 4 or num_turb_op == 0:
                ivc.add_output(f"{phase_name}|thrust_share", mission_config[phase_name]["propulsion"]["prop"]["thrust_share"], units=None)
            else:
                if mission_config[phase_name]["propulsion"]["const_power_profile"]:
                    ivc.add_output(f"{phase_name}|thrust_share_delta", mission_config[phase_name]["propulsion"]["prop"]["thrust_share_delta"]["value"], units=mission_config[phase_name]["propulsion"]["prop"]["thrust_share_delta"]["units"])
                else:
                    ivc.add_output(f"{phase_name}|thrust_share_delta", np.ones(num_nodes) * mission_config[phase_name]["propulsion"]["prop"]["thrust_share_delta"]["value"], units=mission_config[phase_name]["propulsion"]["prop"]["thrust_share_delta"]["units"])

            #npp = mission_config[phase_name]["propulsion"]["prop"]["num_props_oper"] # per aircraft

            if phase_type == "unsteady_eas":# or phase_type == "power_only":
                ivc.add_output(f"{phase_name}|num_props", mission_config[phase_name]["num_props"], units=None)

            #ivc.add_output(f"{phase_name}|num_turbs", mission_config[phase_name]["num_turbs"], units=None)


            ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|rpm", mission_config[phase_name]["propulsion"]["motor"]["rpm"]["value"], units=mission_config[phase_name]["propulsion"]["motor"]["rpm"]["units"])

            if mission_config[phase_name]["propulsion"]["s_curve_hy_profile"] == False:
                if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "relative":
                    if not mission_config[phase_name]["propulsion"]["const_hy_ratio"]:  # transient
                        if mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fraction":
                            if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                                # Power split controls
                                ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_fraction_gt", mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_gt"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_gt"]["units"])
                            elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                                # Power split controls
                                ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_fraction_em", mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_em"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_em"]["units"])
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fixed":
                            if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                                ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_amount_gt", mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_gt"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_gt"]["units"])
                            elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                                ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_amount_em", mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_em"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_em"]["units"])
                    else:  # constant
                        if mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fraction":
                            if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                                ivc.add_output(f"{phase_name}|power_split_vect_to_mat|vector", np.ones(mission_config[phase_name]["num_props"]) * mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_gt"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_gt"]["units"])
                            elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                                ivc.add_output(f"{phase_name}|power_split_vect_to_mat|vector", np.ones(mission_config[phase_name]["num_props"]) * mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_em"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_fraction_em"]["units"])
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fixed":
                            if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                                ivc.add_output(f"{phase_name}|power_split_vect_to_mat|vector", np.ones(mission_config[phase_name]["num_props"]) * mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_gt"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_gt"]["units"])
                            elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                                ivc.add_output(f"{phase_name}|power_split_vect_to_mat|vector", np.ones(mission_config[phase_name]["num_props"]) * mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_em"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["power_split_amount_em"]["units"])

            if mission_config[phase_name]["propulsion"]["s_curve_power_profile"] == False:
                if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "relative":
                    if mission_config[phase_name]["propulsion"]["nacelle"]["command"] == "throttle" and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"] and phase_type != "steady_eas":
                        if mission_config[phase_name]["propulsion"]["const_power_profile"]:  # constant
                            ivc.add_output(f"{phase_name}|throttle_vect_to_mat|vector", np.ones(mission_config[phase_name]["num_props"]) * mission_config[phase_name]["propulsion"]["nacelle"]["throttle"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["throttle"]["units"])
                        else:  # transient
                            ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|throttle", mission_config[phase_name]["propulsion"]["nacelle"]["throttle"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["throttle"]["units"])

                    if mission_config[phase_name]["propulsion"]["nacelle"]["command"] == "power" and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"]:
                        # Max Nacelle power
                        if mission_config[phase_name]["propulsion"]["const_power_profile"] and mission_config[phase_name]["propulsion"]["eq_power_nacelles_op"]:
                            try:
                                ivc.add_output(f"{phase_name}|total_mech_power_out_scalar", mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out_scalar"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out_scalar"]["units"])
                            except:
                                pass
                        #elif mission_config[phase_name]["propulsion"]["const_power_profile"] or mission_config[phase_name]["propulsion"]["eq_power_nacelles_op"]:
                        #        ivc.add_output(f"{phase_name}|total_mech_power_out_vector",  mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out_vector"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out_vector"]["units"])
                        else:  # fully transient
                            ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|total_mech_power_out", mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out"]["value"], units=mission_config[phase_name]["propulsion"]["nacelle"]["total_mech_power_out"]["units"])


            if mission_config[phase_name]["propulsion"]["prop"]["prop_rpm_set"]:
                ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|prop|rpm", mission_config[phase_name]["propulsion"]["prop"]["rpm"]["value"], units=mission_config[phase_name]["propulsion"]["prop"]["rpm"]["units"])



            if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "independent" and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"]:
                
                for k in range(nt):
                    if mission_config[phase_name]["propulsion"]["turbine"]["control"] == "throttle":
                            ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|turb|throttle", mission_config[phase_name]["propulsion"]["turbine"]["throttle"]["value"], units=mission_config[phase_name]["propulsion"]["turbine"]["throttle"]["units"])
                    
                    elif mission_config[phase_name]["propulsion"]["turbine"]["control"] == "power":
                        ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|turb|power", mission_config[phase_name]["propulsion"]["turbine"]["power"]["value"], units=mission_config[phase_name]["propulsion"]["turbine"]["power"]["units"])


                for k in range(nm):
                    if mission_config[phase_name]["propulsion"]["motor"]["control"] == "torque_speed":
                        ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|torque", mission_config[phase_name]["propulsion"]["motor"]["torque"]["value"], units=mission_config[phase_name]["propulsion"]["motor"]["torque"]["units"])

                    elif mission_config[phase_name]["propulsion"]["motor"]["control"] == "power":
                        ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|power", mission_config[phase_name]["propulsion"]["motor"]["power"]["value"], units=mission_config[phase_name]["propulsion"]["motor"]["power"]["units"])
                    
                    elif mission_config[phase_name]["propulsion"]["motor"]["control"] == "throttle":
                        ivc.add_output(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|throttle", mission_config[phase_name]["propulsion"]["motor"]["throttle"]["value"], units=None)


            # end

    return ivc


def prepare_mission_connections(prob, mission_config=None, num_props = 4, num_motors_per_nac = 1, n_str = 1, opt_mission=False, drag_model = "empirical", turb_op_array = None, skip_velocity_connections=False):  
    """
    Set up connections for mission analysis.
    
    Parameters
    ----------
    skip_velocity_connections : bool
        If True, skip connections for fltcond|Ueas and fltcond|Utrue.
        Use this when velocity inputs will be provided by an external dynamic component.
    """
    npp = num_props # per aircraft
    nm = num_motors_per_nac # per nacelle
    n_str = n_str # number of battery strings
    num_turb_op = np.sum(np.array(turb_op_array))

    keys = [k for k in mission_config['mission']['phase_names'] if k not in ["mission", "reserve"]]
    # Find first phase that is not "mission" or "reserve"
    #first_phase_index = next(i for i, phase_name in enumerate(mission_config['mission']['phase_names']) 
    #                    if phase_name not in ["mission", "reserve"])



    if not mission_config[f"{keys[0]}"]["point_analysis"]:
        for j in range(0,n_str):
            if len(keys) > 1:
                prob.model.connect(f"{keys[0]}|soc_initial", f"batt1.integrate_mission_soc_str{j}.soc_integrator_{keys[0]}.soc_initial")
            else:
                prob.model.connect(f"{keys[0]}|soc_initial", f"batt1.soc_integrator_str{j}.soc_initial")
        #prob.model.connect(f"{keys[0]}|soc_initial", f"batt1.soc_initial")
        prob.model.connect(f"{keys[0]}|range_initial", f"{keys[0]}.ode_integ_phase.range_initial")
        prob.model.connect(f"{keys[0]}|fuel_used_initial", f"{keys[0]}.fuel_used_initial")
        prob.model.connect(f"{keys[0]}|h0", [f"{keys[0]}.ode_integ_phase.fltcond|h_initial"])
        prob.model.connect(f"{keys[0]}|t0", f"{keys[0]}.t0")
        
        
    
       #prob.model.connect("climb|ode_integ_phase|fltcond|h_initial", "climb.ode_integ_phase.fltcond|h_initial")
    #prob.model.connect("descent|descentdt|approach|h", "descent.descentdt.approach|h")

    #return prob
    # Phase Specific Variables

    if "cruise_1" in keys:
        if not mission_config["cruise_1"]["duration_set"]:
            prob.model.connect("cruise_1dt|mission_range", "cruise_1.cruise_1dt.mission_range")

    if "reserve_cruise" in keys:
        if not mission_config["reserve_cruise"]["duration_set"]:
            prob.model.connect("reserve_cruisedt|reserve_range", "reserve_cruise.reserve_cruisedt.reserve_range")

    if mission_config[f"{keys[0]}"]["point_analysis"]:
        prob.model.connect(f"{keys[0]}|fltcond|h", f"{keys[0]}.fltcond|h")
    
    if "holding_cruise" in keys:
        prob.model.connect("holding_cruise|ode_integ_phase|fltcond|h_initial", ["holding_cruise.ode_integ_phase.fltcond|h_initial", "holding_cruise.bcast_height.scalar"])

    phase_names = mission_config['mission']['phase_names']
    for idx, phase_name in enumerate(mission_config['mission']['phase_names']):
        if phase_name != "mission":

            # Get configuration parameters
            velocity_in = mission_config[phase_name]["velocity_in"]
            vs_in = mission_config[phase_name]["vs_in"]
            power_in = mission_config[phase_name]["power_in"]
            duration_set = mission_config[phase_name]["duration_set"]
            match_alt = mission_config[phase_name]["match_alt"]
            match_speed = mission_config[phase_name]["match_speed"]
            match_range = mission_config[phase_name]["match_range"]
            phase_type = mission_config[phase_name]["phase_type"]
            gamma_in = mission_config[phase_name]["gamma_in"]


            #if idx != len(keys) - 1:
            if not mission_config[phase_name]["point_analysis"]:
                if match_alt:
                    if idx == 0:
                        prob.model.connect(f"{keys[idx]}|altitude_duration|y0", f"{keys[idx]}.altitude_duration.y0")
                    elif mission_config[phase_names[idx-1]]["phase_type"] == "power_only":
                        prob.model.connect(f"{keys[idx]}|altitude_duration|y0", f"{keys[idx]}.altitude_duration.y0")
                    prob.model.connect(f"{keys[idx]}|altitude_duration|y_target", f"{keys[idx]}.altitude_duration.y_target")
                    #if "descent" not in phase_name and "approach" not in phase_name:   
                    #    prob.model.connect(f"{keys[idx]}dt|{keys[idx+1]}|h0", f"{keys[idx]}.{keys[idx]}dt.{keys[idx+1]}|h0")
                    #else:
                        #prob.model.connect(f"{keys[idx]}dt|{keys[idx]}|h1", f"{keys[idx]}.{keys[idx]}dt.{keys[idx]}|h1")
                    #    prob.model.connect(f"{keys[idx]}dt|{keys[idx+1]}|h1", f"{keys[idx]}.altitude_duration.y_target")
                elif match_speed:
                    prob.model.connect(f"{keys[idx]}|speed_duration|y0", f"{keys[idx]}.speed_duration.y0")
                    prob.model.connect(f"{keys[idx]}|speed_duration|y_target", f"{keys[idx]}.speed_duration.y_target")
                    #if "descent" not in phase_name and "approach" not in phase_name:
                    #    prob.model.connect(f"{keys[idx]}dt|{keys[idx+1]}|Utrue_0", f"{keys[idx]}.{keys[idx]}dt.{keys[idx+1]}|Utrue_0")
                    #else:
                    #    prob.model.connect(f"{keys[idx]}dt|{keys[idx+1]}|Utrue_1", f"{keys[idx]}.{keys[idx]}dt.{keys[idx+1]}|Utrue_1")
                #else:
                #    if match_alt:
                #        prob.model.connect(f"{keys[idx]}dt|{keys[idx]}|h1", f"{keys[idx]}.{keys[idx]}dt.{keys[idx]}|h1")
                #    elif match_speed:
                #        prob.model.connect(f"{keys[idx]}dt|{keys[idx]}|Utrue_final_tgt", f"{keys[idx]}.fltcond|Utrue_final_tgt")
    
                # When opt_objective is fuel. Range is fixed. Cruise 2 duration is a design variable
                # Cruise 1 and Reserve cruise take their range targets at the mission level...
                if match_range and "cruise_1" not in phase_name and "reserve_cruise" not in phase_name and not opt_mission:
                    prob.model.connect(f"{keys[idx]}|range_duration|y_target", f"{keys[idx]}.range_duration.y_target")

                    if not mission_config[phase_name]["trajectory"]["range_target"]["relative"]:
                        prob.model.connect(f"{keys[idx]}|range_duration|y0", f"{keys[idx]}.range_duration.y0")
                    #elif match_range:
                    #    prob.model.connect(f"{keys[idx]}dt|{keys[idx+1]}|range", f"{keys[idx]}.ode_integ_phase.range_initial")

            if drag_model == "empirical":
                if phase_type == "unsteady_eas":# or phase_type == "power_only":
                    prob.model.connect(f"{phase_name}|num_props", f"{phase_name}.num_props")

            if phase_type != "power_only":  
                prob.model.connect(f"{phase_name}|flap_angle", f"{phase_name}.flap_angle")
            prob.model.connect(f"{phase_name}|fltcond|disa", [f"{phase_name}.fltcond|disa", f"{phase_name}.fltcond|TempIncrement"])

            if vs_in or phase_type == "steady_eas":
                prob.model.connect(f"{phase_name}|fltcond|vs", f"{phase_name}.fltcond|vs")
                if match_alt and not mission_config[phase_name]["point_analysis"]:
                    prob.model.connect(f"{phase_name}|fltcond|vs", f"{phase_name}.altitude_duration.dy_dt")
            if gamma_in or phase_type == "steady_eas":
                prob.model.connect(f"{phase_name}|fltcond|gamma", f"{phase_name}.fltcond|gamma")
                #if match_alt:

            if (velocity_in or phase_type == "steady_eas") and not skip_velocity_connections:
                if not mission_config[phase_name]["true_airspeed_in"]:
                    if 'constant' in mission_config[phase_name]["speed_law"]:
                        prob.model.connect(f"{phase_name}|fltcond|Ueas", f"{phase_name}.bcast_speed.scalar")
                    else:
                        prob.model.connect(f"{phase_name}|fltcond|Ueas", f"{phase_name}.fltcond|Ueas")
                if mission_config[phase_name]["true_airspeed_in"] and (phase_name != "cruise_2" or not opt_mission):
                    if 'constant' in mission_config[phase_name]["speed_law"]:
                        prob.model.connect(f"{phase_name}|fltcond|Utrue", f"{phase_name}.bcast_speed.scalar")
                    else:
                        prob.model.connect(f"{phase_name}|fltcond|Utrue", f"{phase_name}.fltcond|Utrue")

                    
            if vs_in and power_in and phase_type == "unsteady_eas":
                prob.model.connect(f"{phase_name}|fltcond|Utrue_initial", f"{phase_name}.fltcond|Utrue_initial")
            if gamma_in and power_in and phase_type == "unsteady_eas":
                prob.model.connect(f"{phase_name}|fltcond|gamma_initial", f"{phase_name}.fltcond|gamma_initial")
            if not power_in and mission_config[phase_name]["phase_type"] == "unsteady_eas":
                if mission_config[phase_name]["propulsion"]["eq_power_nacelles_op"] or num_turb_op == 4 or num_turb_op == 0:
                    prob.model.connect(f"{phase_name}|thrust_share", f"{phase_name}.hy_parallel_ptrain.thrust_share")
                else:
                    prob.model.connect(f"{phase_name}|thrust_share_delta", f"{phase_name}.thrust_share_delta")

            #prob.model.connect('ac|propulsion|battery|r_hv_cbl_ptrain', [f'{phase_name}.hy_parallel_ptrain.hv_current_solver.r_cable', f'{phase_name}.hy_parallel_ptrain.hv_voltage_calc.r_cable'])
            #prob.model.connect('ac|propulsion|battery|r_hv_cbl_aux', [f'{phase_name}.hy_parallel_ptrain.lv_current_solver.r_cable', f'{phase_name}.hy_parallel_ptrain.lv_voltage_calc.r_cable'])

            if duration_set:
                if not opt_mission or "cruise_" not in phase_name:
                    prob.model.connect(f"{phase_name}|duration", f"{phase_name}.duration_in")

            #prob.model.connect(f"{phase_name}|hy_parallel_ptrain|p_sys_lv", f"{phase_name}.hy_parallel_ptrain.p_sys_lv")
            #prob.model.connect(f"{phase_name}|p_aux_elec", f"{phase_name}.p_aux_elec")
            #prob.model.connect(f"{phase_name}|eta_parc", f"{phase_name}.eta_parc")
            #if phase_type != power_only:
            prob.model.connect(f"nacelle_max_rated_power", f"{phase_name}.hy_parallel_ptrain.nacelle_max_rated_power")

            #prob.model.connect(f"{phase_name}|num_turbs", f"{phase_name}.num_turbs")

            #if not mission_config[phase_name]["power_in"]:

            if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "relative":
                if mission_config[phase_name]["propulsion"]["nacelle"]["command"] == 'throttle' and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"]:
                    prob.model.connect(f"nacelle_max_rated_power", f"{phase_name}.hy_parallel_ptrain.nacelles.max_rated_power")

                    if mission_config[phase_name]["phase_type"] == "unsteady_eas" or mission_config[phase_name]["phase_type"] == "power_only":
                        prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|throttle", f"{phase_name}.hy_parallel_ptrain.nacelles.throttle")

            #if mission_config[phase_name]["phase_type"] == "unsteady_eas" or mission_config[phase_name]["phase_type"] == "power_only": # for nacelle throttle tracking
            #    prob.model.connect(f"hy_parallel_ptrain|nacelles|max_rated_power", f"{phase_name}.hy_parallel_ptrain|nacelles|max_rated_power")
            # Always connect RPM to nacelles.rpm for hybrid_split (needed for motor power limit interpolation)
            # This is required regardless of motor_command because hybrid_split.motor_rpm_volt_pow_interp always needs RPM
            
            if mission_config[phase_name]["propulsion"]["motor"]["control"] == "torque_speed":
                prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|rpm", f"{phase_name}.hy_parallel_ptrain.nacelles.motor.rpm")
            elif mission_config[phase_name]["propulsion"]["motor"]["control"] == "power" or mission_config[phase_name]["propulsion"]["motor"]["control"] == "throttle":
                prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|rpm", f"{phase_name}.hy_parallel_ptrain.nacelles.rpm")


            if mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"] == True and mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "independent":

                if mission_config[phase_name]["propulsion"]["motor"]["control"] == "torque_speed" and mission_config[phase_name]["propulsion"]["motor"]["control"] == "power":
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|torque", f"{phase_name}.hy_parallel_ptrain.nacelles.motor.torque_cmd")

                elif mission_config[phase_name]["propulsion"]["motor"]["control"] == "power":
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|power", [f"{phase_name}.hy_parallel_ptrain.nacelles.motor.mech_power", 
                    f"{phase_name}.hy_parallel_ptrain.nacelles.em_shaft_power_matrix"])#, f"{phase_name}.hy_parallel_ptrain.nacelles.motor_power_to_throttle.mech_power"])
                
                elif mission_config[phase_name]["propulsion"]["motor"]["control"] == "throttle":
                    # Connect motor throttle for independent power_spec with throttle command
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|motor|throttle", [f"{phase_name}.hy_parallel_ptrain.nacelles.motor_throttle",
                    f"{phase_name}.hy_parallel_ptrain.nacelles.motor.throttle"])

            # Architecture specific inputs
            #if mission_config[phase_name]["prop"]["prop_thrust_set"]: 
            #    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelle{j+1}|thrust", f"{phase_name}.hy_parallel_ptrain.nacelle{j+1}.thrust")



            # Connect hybridization ratio controls (when not optimized)
            if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "relative" and mission_config[phase_name]["propulsion"]["s_curve_hy_profile"] == False:

                if not mission_config[phase_name]["propulsion"]["const_hy_ratio"]:  # transient
                    if mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fraction":
                        if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                            # Power split controls
                            prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_fraction_gt", f"{phase_name}.hy_parallel_ptrain.nacelles.power_split_fraction_gt")
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                            # Power split controls
                            prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_fraction_em", f"{phase_name}.hy_parallel_ptrain.nacelles.power_split_fraction_em")
                    elif mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fixed":
                        if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                            prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_amount_gt", f"{phase_name}.hy_parallel_ptrain.nacelles.power_split_amount_gt")
                            # Connect GT derate for cruise_1 phase during optimization
                            if opt_mission and phase_name == "cruise_1":
                                prob.model.connect(f"{phase_name}|gt_derate", f"{phase_name}.hy_parallel_ptrain.nacelles.gt_derate")
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                            prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|power_split_amount_em", f"{phase_name}.hy_parallel_ptrain.nacelles.power_split_amount_em")
                else:  # constant
                    if mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fraction":
                        if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                            prob.model.connect(f"{phase_name}|power_split_vect_to_mat|vector", f"{phase_name}.power_split_vect_to_mat.vector")
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                            prob.model.connect(f"{phase_name}|power_split_vect_to_mat|vector", f"{phase_name}.power_split_vect_to_mat.vector")
                    elif mission_config[phase_name]["propulsion"]["nacelle"]["rule"] == "fixed":
                        if mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "gt":
                            prob.model.connect(f"{phase_name}|power_split_vect_to_mat|vector", f"{phase_name}.power_split_vect_to_mat.vector")
                        elif mission_config[phase_name]["propulsion"]["nacelle"]["bias"] == "em":
                            prob.model.connect(f"{phase_name}|power_split_vect_to_mat|vector", f"{phase_name}.power_split_vect_to_mat.vector")

            # Connect power controls (when not optimized)
            if mission_config[phase_name]["propulsion"]["s_curve_power_profile"] == False:
                if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "relative":
                    if mission_config[phase_name]["propulsion"]["nacelle"]["command"] == "power" and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"]:
                        # Max Nacelle power
                        if mission_config[phase_name]["propulsion"]["const_power_profile"]:  # constant
                            # For broadcast nacelles, connect to vector input
                            pass
                            #prob.model.connect(f"{phase_name}|total_mech_power_vect_to_mat|vector", f"{phase_name}.total_mech_power_vect_to_mat.vector")
                        else:  # transient
                            # For broadcast nacelles, connect to broadcast
                            prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|total_mech_power_out", [f"{phase_name}.hy_parallel_ptrain.nacelles.total_mech_power_out",f"{phase_name}.hy_parallel_ptrain.nacelles.gb_comp.power_in"])


            if mission_config[phase_name]["propulsion"]["prop"]["prop_rpm_set"]:
                # Propeller RPM
                if mission_config[phase_name]["propulsion"]["prop"]["prop_rpm_set"]:
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|prop|rpm", f"{phase_name}.hy_parallel_ptrain.nacelles.prop.rpm")



            if mission_config[phase_name]["propulsion"]["nacelle"]["power_spec"] == "independent" and mission_config[phase_name]["propulsion"]["prop"]["nacelle_power_set"]:
                # Gas turbine and motor controls
                if mission_config[phase_name]["propulsion"]["turbine"]["control"] == "throttle":
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|turb|throttle", f"{phase_name}.hy_parallel_ptrain.nacelles.turb.throttle")
                
                elif mission_config[phase_name]["propulsion"]["turbine"]["control"] == "power":
                    prob.model.connect(f"{phase_name}|hy_parallel_ptrain|nacelles|turb|power", [f"{phase_name}.hy_parallel_ptrain.nacelles.turb.power", f"{phase_name}.hy_parallel_ptrain.nacelles.gt_shaft_power_matrix"])

    # If Opt miossion is true, broadcast cruise1 speed to cruise2
    if opt_mission and "cruise_2" in keys:
        if not mission_config["cruise_2"]["true_airspeed_in"]:
            if 'constant' in mission_config["cruise_2"]["speed_law"]:
                prob.model.connect(f"cruise_1|fltcond|Ueas", f"cruise_2.bcast_speed.scalar")
            else:
                prob.model.connect(f"cruise_1|fltcond|Ueas", f"cruise_2.fltcond|Ueas")
        if mission_config["cruise_2"]["true_airspeed_in"]:
            if 'constant' in mission_config["cruise_2"]["speed_law"]:
                prob.model.connect(f"cruise_1|fltcond|Utrue", f"cruise_2.bcast_speed.scalar")
            else:
                prob.model.connect(f"cruise_1|fltcond|Utrue", f"cruise_2.fltcond|Utrue")

def airspeed_conversion(h, vel, out="tas"):

    # Constants from the ISA model
    g = 9.80665  # gravitational acceleration (m/s^2)
    R_spec = 287.05  # specific gas constant for dry air (J/(kg·K))

    # Sea-level values
    T0 = 288.15  # standard sea-level temperature (K)
    p0 = 101325.0  # standard sea-level pressure (Pa)
    rho0 = 1.225  # standard sea-level density (kg/m^3)

    # Troposphere (0 to 11,000 m)
    L = -0.0065  # temperature lapse rate (K/m)
    T = T0 + L * h
    p = p0 * (T / T0)**(-g / (L * R_spec))
    rho = p / (R_spec * T)

    if out == "tas":
        return vel * np.sqrt(rho0 / rho)
    elif out == "eas":
        return vel / np.sqrt(rho0 / rho)
    else:
        raise ValueError("Invalid output type")
    
