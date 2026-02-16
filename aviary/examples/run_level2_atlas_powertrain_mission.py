"""
Run a true Aviary level-2 mission with the Atlas-derived parallel hybrid powertrain
inserted into the mission architecture as the propulsion subsystem.

This script replaces the core propulsion mission builder with a custom builder that:
1) keeps Aviary's pre-mission propulsion path,
2) uses ParallelHybridElectricPropulsionSystem during mission phases, and
3) maps outputs to Aviary mission variables consumed by EnergyODE.
"""

import copy

import numpy as np
import openmdao.api as om

import aviary.api as av
from aviary.models.engines.propulsion.systems.parallel_hybrid import (
    ParallelHybridElectricPropulsionSystem,
)
from aviary.models.missions.height_energy_default import phase_info as default_phase_info
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
from aviary.utils.matrix_vector_converter import VectorToMatrixConverter
from aviary.utils.sum_axis import SumAlongAxis
from aviary.visualization.plot_funcs import plot_atlas_aviary_mission
from aviary.variable_info.variables import Aircraft, Dynamic


class AtlasMissionPropulsionGroup(om.Group):
    """Adapter group that maps Aviary mission variables to the Atlas powertrain."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("num_props", types=int)
        self.options.declare("num_batt_strings", default=4, types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        n_str = self.options["num_batt_strings"]

        self.add_subsystem(
            "aviary_inputs_passthrough",
            om.ExecComp(
                [
                    "mach_in = mach",
                    "altitude_in = altitude",
                    "density_in = density",
                    "sos_in = speed_of_sound",
                    "velocity_in = velocity",
                    "tempK_in = temperature",
                    "throttle_in = throttle",
                ],
                mach={"val": np.ones(nn), "units": "unitless"},
                mach_in={"val": np.ones(nn), "units": "unitless"},
                altitude={"val": np.ones(nn), "units": "m"},
                altitude_in={"val": np.ones(nn), "units": "m"},
                density={"val": np.ones(nn), "units": "kg/m**3"},
                density_in={"val": np.ones(nn), "units": "kg/m**3"},
                speed_of_sound={"val": np.ones(nn), "units": "m/s"},
                sos_in={"val": np.ones(nn), "units": "m/s"},
                velocity={"val": np.ones(nn), "units": "m/s"},
                velocity_in={"val": np.ones(nn), "units": "m/s"},
                temperature={"val": np.ones(nn), "units": "K"},
                tempK_in={"val": np.ones(nn), "units": "K"},
                throttle={"val": np.ones(nn), "units": "unitless"},
                throttle_in={"val": np.ones(nn), "units": "unitless"},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ("mach", Dynamic.Atmosphere.MACH),
                ("altitude", Dynamic.Mission.ALTITUDE),
                ("density", Dynamic.Atmosphere.DENSITY),
                ("speed_of_sound", Dynamic.Atmosphere.SPEED_OF_SOUND),
                ("velocity", Dynamic.Mission.VELOCITY),
                ("temperature", Dynamic.Atmosphere.TEMPERATURE),
                ("throttle", Dynamic.Vehicle.Propulsion.THROTTLE),
            ],
            promotes_outputs=[
                "mach_in",
                "altitude_in",
                "density_in",
                "sos_in",
                "velocity_in",
                "tempK_in",
                "throttle_in",
            ],
        )

        const = om.IndepVarComp()
        const.add_output("nacelle_max_rated_power", val=2.0e4 * np.ones(nn), units="kW")
        const.add_output("motor_rpm_cmd", val=1750.0 * np.ones((npp, nn)), units="rpm")
        const.add_output("power_split_fraction_em", val=0.5 * np.ones((npp, nn)))
        const.add_output("battery_q_cool", val=25.0 * np.ones((n_str, nn)), units="kW")
        const.add_output("prop_diameter", val=4.2 * np.ones((npp, nn)), units="m")
        const.add_output("motor_rating", val=4.0e3, units="kW")
        const.add_output("disa_degC", val=np.zeros(nn), units="degC")
        self.add_subsystem("const", const, promotes_outputs=["*"])

        self.add_subsystem(
            "temp_conv",
            om.ExecComp(
                "temp_degC = temp_K - 273.15",
                temp_K={"val": np.ones(nn), "units": "K"},
                temp_degC={"val": np.ones(nn), "units": "degC"},
                has_diag_partials=True,
            ),
            promotes_inputs=[],
            promotes_outputs=["temp_degC"],
        )
        self.connect("tempK_in", "temp_conv.temp_K")

        self.add_subsystem(
            "throttle_to_nacelles",
            VectorToMatrixConverter(
                num_nodes=nn,
                num_comps=npp,
                input_names=["throttle_flat"],
                output_names=["throttle_nac"],
                units={"throttle_flat": "unitless"},
            ),
            promotes_inputs=[],
            promotes_outputs=[],
        )
        repeat_mat = np.vstack([np.eye(nn) for _ in range(npp)])
        self.add_subsystem(
            "repeat_throttle",
            om.ExecComp(
                "throttle_flat = repeat_mat @ throttle_in",
                throttle_in={"val": np.ones(nn), "units": "unitless"},
                repeat_mat={"val": repeat_mat, "units": "unitless"},
                throttle_flat={"val": np.ones(npp * nn), "units": "unitless"},
            ),
            promotes_inputs=[],
            promotes_outputs=[],
        )
        self.connect("throttle_in", "repeat_throttle.throttle_in")
        self.connect("repeat_throttle.throttle_flat", "throttle_to_nacelles.throttle_flat")

        self.add_subsystem(
            "atlas_ptrain",
            ParallelHybridElectricPropulsionSystem(
                num_nodes=nn,
                num_props=npp,
                num_em_per_nac=1,
                num_turb_per_nac=1,
                n_str=n_str,
                rule="fraction",
                bias="em",
                nacelle_power_set=True,
                prop_rpm_set=False,
                prop_thrust_set=False,
                cnvg_throttle=False,
                gt_command="power",
                motor_command="power",
                nacelle_command="throttle",
                power_spec="relative",
                gt_idle_allowed=True,
                size_motor=False,
                turb_type="PT6",
            ),
            promotes_outputs=[],
        )

        # Map Aviary mission conditions to Atlas inputs.
        self.connect("mach_in", "atlas_ptrain.fltcond|M")
        self.connect("altitude_in", "atlas_ptrain.fltcond|h")
        self.connect("density_in", "atlas_ptrain.fltcond|rho")
        self.connect("sos_in", "atlas_ptrain.fltcond|a")
        self.connect("velocity_in", "atlas_ptrain.fltcond|Utrue")
        self.connect("temp_degC", "atlas_ptrain.fltcond|T")
        self.connect("disa_degC", "atlas_ptrain.fltcond|disa")
        self.connect("throttle_to_nacelles.throttle_nac", "atlas_ptrain.nacelles.throttle_nac")

        # Atlas static configuration feeds.
        self.connect("nacelle_max_rated_power", "atlas_ptrain.nacelle_max_rated_power")
        self.connect("motor_rpm_cmd", "atlas_ptrain.nacelles.rpm")
        self.connect("power_split_fraction_em", "atlas_ptrain.nacelles.power_split_fraction_em")
        self.connect("prop_diameter", "atlas_ptrain.ac|propulsion|propeller|diameter")
        self.connect("motor_rating", "atlas_ptrain.ac|propulsion|motor|rating")
        self.connect("battery_q_cool", "atlas_ptrain.ac|propulsion|battery|q_cool_bat")

        self.add_subsystem(
            "thrust_convert",
            om.ExecComp(
                "thrust_lbf = thrust_N * 0.22480894387096263",
                thrust_N={"val": np.ones(nn), "units": "N"},
                thrust_lbf={"val": np.ones(nn), "units": "lbf"},
                has_diag_partials=True,
            ),
            promotes_outputs=[("thrust_lbf", Dynamic.Vehicle.Propulsion.THRUST_TOTAL)],
        )
        self.connect("atlas_ptrain.total_thrust", "thrust_convert.thrust_N")

        self.add_subsystem(
            "thrust_max_proxy",
            om.ExecComp(
                "thrust_max_lbf = 1.25 * thrust_lbf",
                thrust_lbf={"val": np.ones(nn), "units": "lbf"},
                thrust_max_lbf={"val": np.ones(nn), "units": "lbf"},
                has_diag_partials=True,
            ),
            promotes_inputs=[("thrust_lbf", Dynamic.Vehicle.Propulsion.THRUST_TOTAL)],
            promotes_outputs=[("thrust_max_lbf", Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL)],
        )

        self.add_subsystem(
            "sum_elec",
            SumAlongAxis(
                num_nodes=nn,
                num_comps=npp,
                input_name="p_train_elec",
                output_name="p_elec_w_total",
                input_units="W",
                output_units="W",
            ),
            promotes_outputs=[],
        )
        self.connect("atlas_ptrain.p_train_elec", "sum_elec.p_train_elec")

        self.add_subsystem(
            "elec_convert",
            om.ExecComp(
                "electric_power_in_total = p_elec_w_total / 1000.0",
                p_elec_w_total={"val": np.ones(nn), "units": "W"},
                electric_power_in_total={"val": np.ones(nn), "units": "kW"},
                has_diag_partials=True,
            ),
            promotes_outputs=[Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL],
        )
        self.connect("sum_elec.p_elec_w_total", "elec_convert.p_elec_w_total")

        self.add_subsystem(
            "sum_fuel",
            SumAlongAxis(
                num_nodes=nn,
                num_comps=npp,
                input_name="fuel_flow_matrix",
                output_name="fuel_flow_kgph_total",
                input_units="kg/h",
                output_units="kg/h",
            ),
            promotes_outputs=[],
        )
        self.connect("atlas_ptrain.nacelles.turb.fuel_flow", "sum_fuel.fuel_flow_matrix")

        self.add_subsystem(
            "fuel_convert",
            om.ExecComp(
                "fuel_flow_rate_negative_total = -fuel_flow_kgph_total * 2.20462262185",
                fuel_flow_kgph_total={"val": np.ones(nn), "units": "kg/h"},
                fuel_flow_rate_negative_total={"val": -np.ones(nn), "units": "lbm/h"},
                has_diag_partials=True,
            ),
            promotes_outputs=[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL],
        )
        self.connect("sum_fuel.fuel_flow_kgph_total", "fuel_convert.fuel_flow_kgph_total")


class AtlasMissionPropulsionBuilder(PropulsionBuilder):
    """PropulsionBuilder that uses stock pre-mission and Atlas mission propulsion."""

    def __init__(self, core_propulsion_builder, num_props):
        super().__init__(name="propulsion", meta_data=core_propulsion_builder.meta_data)
        self._core_propulsion_builder = core_propulsion_builder
        self._num_props = num_props

    def build_pre_mission(self, aviary_inputs, **kwargs):
        return self._core_propulsion_builder.build_pre_mission(aviary_inputs, **kwargs)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return AtlasMissionPropulsionGroup(num_nodes=num_nodes, num_props=self._num_props)

    def mission_inputs(self, **kwargs):
        return [
            Dynamic.Atmosphere.MACH,
            Dynamic.Mission.ALTITUDE,
            Dynamic.Atmosphere.DENSITY,
            Dynamic.Atmosphere.SPEED_OF_SOUND,
            Dynamic.Atmosphere.TEMPERATURE,
            Dynamic.Mission.VELOCITY,
            Dynamic.Vehicle.Propulsion.THROTTLE,
        ]

    def mission_outputs(self, **kwargs):
        return [
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
        ]

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        return {}

    def get_controls(self, phase_name=None):
        return {}

    def get_states(self):
        return {}

    def get_pre_mission_bus_variables(self, aviary_inputs=None):
        return {}


def _build_phase_info_with_throttle_control():
    phase_info = copy.deepcopy(default_phase_info)

    # Use a propeller-compatible mission envelope and let Dymos control throttle.
    phase_info["climb"]["user_options"]["mach_final"] = (0.35, "unitless")
    phase_info["climb"]["user_options"]["mach_bounds"] = ((0.18, 0.40), "unitless")
    phase_info["climb"]["user_options"]["altitude_final"] = (12000.0, "ft")
    phase_info["climb"]["user_options"]["altitude_bounds"] = ((0.0, 15000.0), "ft")
    phase_info["climb"]["user_options"]["throttle_enforcement"] = "control"

    phase_info["cruise"]["user_options"]["mach_initial"] = (0.35, "unitless")
    phase_info["cruise"]["user_options"]["mach_final"] = (0.35, "unitless")
    phase_info["cruise"]["user_options"]["mach_bounds"] = ((0.30, 0.40), "unitless")
    phase_info["cruise"]["user_options"]["altitude_initial"] = (12000.0, "ft")
    phase_info["cruise"]["user_options"]["altitude_final"] = (12000.0, "ft")
    phase_info["cruise"]["user_options"]["altitude_bounds"] = ((9000.0, 15000.0), "ft")
    phase_info["cruise"]["user_options"]["throttle_enforcement"] = "control"

    phase_info["descent"]["user_options"]["mach_initial"] = (0.35, "unitless")
    phase_info["descent"]["user_options"]["mach_final"] = (0.20, "unitless")
    phase_info["descent"]["user_options"]["mach_bounds"] = ((0.18, 0.40), "unitless")
    phase_info["descent"]["user_options"]["altitude_initial"] = (12000.0, "ft")
    phase_info["descent"]["user_options"]["altitude_final"] = (500.0, "ft")
    phase_info["descent"]["user_options"]["altitude_bounds"] = ((0.0, 15000.0), "ft")
    phase_info["descent"]["user_options"]["throttle_enforcement"] = "control"

    phase_info["post_mission"]["target_range"] = (250.0, "nmi")

    return phase_info


if __name__ == "__main__":
    phase_info = _build_phase_info_with_throttle_control()

    prob = av.AviaryProblem()
    prob.load_inputs(
        "models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv",
        phase_info,
    )
    prob.check_and_preprocess_inputs()

    core_prop = None
    for subsystem in prob.model.subsystems:
        if subsystem.name == "propulsion":
            core_prop = subsystem
            break
    if core_prop is None:
        raise RuntimeError("Unable to find core propulsion subsystem in Aviary model.")

    # NOTE: the imported Atlas powertrain currently assumes 4 nacelle rows internally.
    # Keep this fixed at 4 to avoid shape conflicts in its current architecture.
    num_props = 4
    atlas_prop = AtlasMissionPropulsionBuilder(core_propulsion_builder=core_prop, num_props=num_props)

    for i, subsystem in enumerate(prob.model.subsystems):
        if subsystem.name == "propulsion":
            prob.model.subsystems[i] = atlas_prop
            break

    # Keep ode args aligned with modified subsystem list.
    prob.model.ode_args["subsystems"] = prob.model.subsystems

    prob.build_model()
    prob.add_driver("SLSQP", max_iter=25)
    prob.add_design_variables()
    prob.add_objective()
    prob.setup()
    prob.run_aviary_problem(
        suppress_solver_print=True,
        run_driver=False,
        make_plots=False,
    )

    plot_atlas_aviary_mission(
        prob,
        phases=("climb", "cruise", "descent"),
        save_plot=True,
        output_filename="atlas_aviary_mission_profile.png",
        show_plot=False,
    )
