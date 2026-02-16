from aviary.visualization.plot_funcs import plot_trajectory, plot_unsteady


def show_outputs(nn, prob, mission_config, ac_data, save_plots=False, output_filename="mission_analysis"):
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
    nn = 11

    npp = ac_data["propulsion"]["prop"]["num_props"]["value"]
    nm = ac_data["propulsion"]["nacelle"]["num_em_per_nac"]["value"]
    phases = mission_config["mission"]["phase_names"]

    plot_unsteady(prob, phases, mission_config, save_plots=save_plots, output_filename=output_filename)
    """

    # Plot settings
    x_var = "range"
    x_unit = "NM"


    y_vars = ["fltcond|h", "fltcond|Ueas", "fuel_used",
              "fltcond|vs", "hy_parallel_ptrain.batt1.soc",["cd", "cl"] ]
    y_units = ["ft", "kn", "lbm", "ft/min", None, None]
    x_label = "Range (nmi)"
    y_labels = [
        "Altitude (ft)",
        "Veas airspeed (knots)",
        "Fuel used (lb)",
        "Vertical speed (ft/min)",
        "Battery SOC",
        "Lift coefficient",
    ]

    # Add individual nacelle throttle plots

    


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
    """