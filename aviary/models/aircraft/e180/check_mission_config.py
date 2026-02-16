def check_mission_config(mission_config,  num_props=4):

    for phase in mission_config:
        if phase not in ["mission", "reserve"]:
            #for i in range(num_props):

            if mission_config[phase]["phase_type"] == "steady_eas" and mission_config[phase]["propulsion"]["nacelle"]["command"] == "power":
                    raise ValueError(f"Phase `{phase}` incorrectly configured. If [phase_type] is 'steady_eas', [nacelle][command] must be `throttle`")
                
            if mission_config[phase]["propulsion"]["prop"]["prop_thrust_set"] and mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"] and mission_config[phase]["propulsion"]["prop"]["prop_rpm_set"]:
                raise ValueError(f"Phase `{phase}` incorrectly configured. Only two of [prop][prop_thrust_set], [prop][nacelle_power_set], and [prop][prop_rpm_set] can be True, not all three")

            if mission_config[phase]["phase_type"] == "unsteady_eas":

                
                if mission_config[phase]["propulsion"]["prop"]["prop_thrust_set"] and mission_config[phase]["power_in"] and mission_config[phase]["phase_type"] == "unsteady_eas":
                    raise ValueError(f"Phase `{phase}` incorrectly configured. If [prop][prop_thrust_set] is True, [power_in] cannot be True")
                if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"] and not mission_config[phase]["power_in"] and mission_config[phase]["phase_type"] == "unsteady_eas":
                    raise ValueError(f"Phase `{phase}` incorrectly configured. If [prop][nacelle_power_set] is True, [power_in] cannot be False")
            # end
            if mission_config[phase]["power_in"] and mission_config[phase]["velocity_in"] and mission_config[phase]["vs_in"] and mission_config[phase]["gamma_in"]  and mission_config[phase]["phase_type"] == "unsteady_eas":
                raise ValueError(f"Phase `{phase}` overconstrained.  `power_in`, `velocity_in`, and `vs_in` cannot all be True")
            if mission_config[phase]["power_in"] and not mission_config[phase]["velocity_in"] and ((not mission_config[phase]["vs_in"]) and (not mission_config[phase]["gamma_in"])):
                raise ValueError(f"Phase `{phase}` underconstrained. Two of `power_in`, `velocity_in`, and `vs_in` must be True")
            if mission_config[phase]["velocity_in"] and not mission_config[phase]["power_in"] and ((not mission_config[phase]["vs_in"]) and (not mission_config[phase]["gamma_in"])):
                raise ValueError(f"Phase `{phase}` underconstrained. Two of `power_in`, `velocity_in`, and `vs_in` must be True")
            if (mission_config[phase]["vs_in"] or mission_config[phase]["gamma_in"]) and not mission_config[phase]["power_in"] and not mission_config[phase]["velocity_in"]:
                raise ValueError(f"Phase `{phase}` underconstrained. Two of `power_in`, `velocity_in`, and `vs_in` must be True")
            if mission_config[phase]["duration_set"] and (mission_config[phase]["match_alt"] or mission_config[phase]["match_speed"] or mission_config[phase]["match_range"]):
                Warning(f"`{phase}` duration_set is True while match_alt is {mission_config[phase]['match_alt']}, match_speed is {mission_config[phase]['match_speed']}, or match_range is {mission_config[phase]['match_range']}. Matching constraint will be ignored unless duration_set is False.")
            
            # Check propulsion/nacelle key matching based on power_spec, rule, and bias
            nacelle_config = mission_config[phase]["propulsion"]["nacelle"]
            power_spec = nacelle_config["power_spec"]
            rule = nacelle_config["rule"]
            bias = nacelle_config["bias"]

            if mission_config[phase]["propulsion"]["s_curve_hy_profile"] == False:
                if power_spec == "relative":
                    if rule == "fraction" and bias == "gt":
                        try:
                            if nacelle_config["power_split_fraction_gt"] is None:
                                raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fraction', and [bias] is 'gt', then [power_split_fraction_gt] must not be None")
                        except KeyError:
                            raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fraction', and [bias] is 'gt', then [power_split_fraction_gt] must not be None")
                    elif rule == "fraction" and bias == "em":
                        try:
                            if nacelle_config["power_split_fraction_em"] is None:
                                raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fraction', and [bias] is 'em', then [power_split_fraction_em] must not be None")
                        except KeyError:
                            raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fraction', and [bias] is 'em', then [power_split_fraction_em] must not be None")
                    elif rule == "fixed" and bias == "gt":
                        try:
                            if nacelle_config["power_split_amount_gt"] is None:
                                raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fixed', and [bias] is 'gt', then [power_split_amount_gt] must not be None")
                        except KeyError:
                            raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fixed', and [bias] is 'gt', then [power_split_amount_gt] must not be None")
                    elif rule == "fixed" and bias == "em":
                        try:
                            if nacelle_config["power_split_amount_em"] is None:
                                raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fixed', and [bias] is 'em', then [power_split_amount_em] must not be None")
                        except KeyError:
                            raise ValueError(f"Phase `{phase}` incorrectly configured. If [power_spec] is 'relative', [rule] is 'fixed', and [bias] is 'em', then [power_split_amount_em] must not be None")