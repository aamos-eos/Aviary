from aviary.visualization.plot_funcs import plot_trajectory


def show_outputs(nn, prob, mission_config,  save_plots=False, output_filename="mission_analysis"):
    """
    Show and optionally save mission analysis plots.
    
    Parameters
    ----------
    prob : OpenMDAO Problem object
        The problem containing the data to plot
    mission_config : dict
        Configuration dictionary for the propulsion system
    save_plots : bool, optional
        Whether to save the plots to PDF files instead of displaying them
    output_filename : str, optional
        Base name for the output files (without extension)
    """

    npp = mission_config["seg"]["num_props"]
    nm = mission_config["seg"]["num_em_per_nac"]
    nt = mission_config["seg"]["num_turb_per_nac"]
    nac = mission_config["seg"]["num_nac"]
    #phases = ac_data["ac"]["mission"]["phase_names"]
    phases = ["seg"]

    # Plot settings
    x_var = "range"
    x_unit = "NM"

    if mission_config["seg"]["propulsion"]["prop"]["nacelle_power_set"]:
        thrust_var = "hy_parallel_ptrain.total_thrust"
    else:
        thrust_var = "thrust"
    
    if mission_config["seg"]["phase_type"] == "unsteady_eas":
        y_vars = ["fltcond|h", "fltcond|Ueas", "fuel_used",
                "fltcond|vs", "hy_parallel_ptrain.batt1.soc", 
                thrust_var]
        y_units = ["ft", "kn", "lbm", "ft/min", None, "N"]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Vertical speed (ft/min)",
            "Battery SOC",
            "Total thrust (N)",
        ]
    else:
        y_vars = ["fltcond|h", "fltcond|Ueas", "fuel_used",
                "fltcond|vs", "hy_parallel_ptrain.batt1.soc", 
                thrust_var]
        y_units = ["ft", "kn", "lbm",  "ft/min", None, "N"]
        x_label = "Range (nmi)"
        y_labels = [
            "Altitude (ft)",
            "Veas airspeed (knots)",
            "Fuel used (lb)",
            "Vertical speed (ft/min)",
            "Battery SOC",
            "Total thrust (N)",
        ]



    # Create mission profile plots
    plot_trajectory(
        prob,
        x_var,
        x_unit,
        y_vars,
        y_units,
        phases,
        x_label=x_label,
        y_labels=y_labels,
        marker="-",
        plot_title="Mission Profile",
        save_plots=save_plots,
        output_filename=output_filename
    )

    # Create nacelle performance plots
    plot_nacelle_perfo(
        prob,
        phases,
        nn,
        mission_config,
        num_nacelles=npp,
        num_motors=nm,
        x_label=None,
        plot_title="Nacelle Performance",
        save_plots=save_plots,
        output_filename=output_filename
    )