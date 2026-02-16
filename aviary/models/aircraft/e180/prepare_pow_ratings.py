def prepare_pow_ratings(ac_data):
    # Prepare pow_ratings_dictionary
    # TODO: Convert throttle setting to PLA then revet back to [0.1]
    pow_ratings = {
        "nacelle": {
            "MCP": {"throttle": ac_data["ac"]["propulsion"]["nacelle"]["max_continuous_power"]["value"] / ac_data["ac"]["propulsion"]["nacelle"]["max_rated_power"]["value"], 
                    "value": ac_data["ac"]["propulsion"]["nacelle"]["max_continuous_power"]["value"],
                    "units": ac_data["ac"]["propulsion"]["nacelle"]["max_continuous_power"]["units"]
                    },
            "MTOP": {"throttle": 1.0 ,
                    "value": ac_data["ac"]["propulsion"]["nacelle"]["max_rated_power"]["value"],
                    "units": ac_data["ac"]["propulsion"]["nacelle"]["max_rated_power"]["units"]
                    },
                },
        "turbine": {
            "MCP": {"throttle": 1.0, 
            "value": ac_data["ac"]["propulsion"]["turbine"]["rating"]["value"],
            "units": ac_data["ac"]["propulsion"]["turbine"]["rating"]["units"]},
        },
        "motor": {
            "MCP": {"throttle": ac_data["ac"]["propulsion"]["motor"]["cont_rating_per_nac"]["value"] / ac_data["ac"]["propulsion"]["motor"]["rating"]["value"], 
            "value": ac_data["ac"]["propulsion"]["motor"]["cont_rating_per_nac"]["value"],
            "units": ac_data["ac"]["propulsion"]["motor"]["cont_rating_per_nac"]["units"]},
            "MTOP": {"throttle": 1.0, 
            "value": ac_data["ac"]["propulsion"]["motor"]["rating"]["value"],
            "units": ac_data["ac"]["propulsion"]["motor"]["rating"]["units"]},
        },
    }
    # end
    return pow_ratings
