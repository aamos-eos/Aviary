"""
Run a true Aviary level-2 mission with the Atlas-derived parallel hybrid powertrain
inserted into the mission architecture as the propulsion subsystem.

This script replaces the core propulsion mission builder with a custom builder that:
1) keeps Aviary's pre-mission propulsion path,
2) uses ParallelHybridElectricPropulsionSystem during mission phases, and
3) maps outputs to Aviary mission variables consumed by EnergyODE.
"""

import copy
import time

import numpy as np
import openmdao.api as om

import aviary.api as av
from aviary.models.engines.propulsion.systems.parallel_hybrid import (
    ParallelHybridElectricPropulsionSystem,
)
from aviary.models.engines.propulsion.battery.battery_data import BatteryData
from aviary.models.engines.propulsion.battery.battery_empirical_power import (
    EmpiricalBatteryPower,
)
from aviary.models.missions.height_energy_default import phase_info as default_phase_info
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.matrix_vector_converter import VectorToMatrixConverter
from aviary.utils.sum_axis import SumAlongAxis
from aviary.visualization.plot_funcs import plot_atlas_aviary_mission
from aviary.variable_info.variables import Aircraft, Dynamic


class AtlasPreMissionPropulsionGroup(om.Group):
    """Atlas-backed pre-mission propulsion sizing surrogate for Aviary pre-mission flow."""

    def initialize(self):
        self.options.declare("aviary_inputs", types=AviaryValues)
        self.options.declare("num_props", types=int, default=4)
        self.options.declare("num_batt_strings", default=4, types=int)

    def setup(self):
        aviary_inputs = self.options["aviary_inputs"]
        npp = self.options["num_props"]
        n_str = self.options["num_batt_strings"]
        nn = 1

        ivc = om.IndepVarComp()
        ivc.add_output("fltcond|M", val=np.array([0.0]), units="unitless")
        ivc.add_output("fltcond|h", val=np.array([0.0]), units="ft")
        ivc.add_output("fltcond|rho", val=np.array([1.225]), units="kg/m**3")
        ivc.add_output("fltcond|a", val=np.array([340.0]), units="m/s")
        ivc.add_output("fltcond|Utrue", val=np.array([75.0]), units="m/s")
        ivc.add_output("fltcond|T", val=np.array([15.0]), units="degC")
        ivc.add_output("fltcond|disa", val=np.array([0.0]), units="degC")
        ivc.add_output("nacelle_max_rated_power", val=np.array([2.0e4]), units="kW")
        ivc.add_output("nacelle_throttle", val=np.ones((npp, nn)))
        ivc.add_output("motor_rpm_cmd", val=1750.0 * np.ones((npp, nn)), units="rpm")
        ivc.add_output("power_split_fraction_em", val=0.5 * np.ones((npp, nn)))
        ivc.add_output("battery_q_cool", val=25.0 * np.ones((n_str, nn)), units="kW")
        ivc.add_output("prop_diameter", val=4.2 * np.ones((npp, nn)), units="m")
        ivc.add_output("motor_rating", val=4.0e3, units="kW")
        self.add_subsystem("ivc", ivc, promotes_outputs=["*"])

        self.add_subsystem(
            "atlas_ptrain_premission",
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

        self.connect("fltcond|M", "atlas_ptrain_premission.fltcond|M")
        self.connect("fltcond|h", "atlas_ptrain_premission.fltcond|h")
        self.connect("fltcond|rho", "atlas_ptrain_premission.fltcond|rho")
        self.connect("fltcond|a", "atlas_ptrain_premission.fltcond|a")
        self.connect("fltcond|Utrue", "atlas_ptrain_premission.fltcond|Utrue")
        self.connect("fltcond|T", "atlas_ptrain_premission.fltcond|T")
        self.connect("fltcond|disa", "atlas_ptrain_premission.fltcond|disa")
        self.connect("nacelle_max_rated_power", "atlas_ptrain_premission.nacelle_max_rated_power")
        self.connect("nacelle_throttle", "atlas_ptrain_premission.nacelles.throttle_nac")
        self.connect("motor_rpm_cmd", "atlas_ptrain_premission.nacelles.rpm")
        self.connect(
            "power_split_fraction_em", "atlas_ptrain_premission.nacelles.power_split_fraction_em"
        )
        self.connect("battery_q_cool", "atlas_ptrain_premission.ac|propulsion|battery|q_cool_bat")
        self.connect("prop_diameter", "atlas_ptrain_premission.ac|propulsion|propeller|diameter")
        self.connect("motor_rating", "atlas_ptrain_premission.ac|propulsion|motor|rating")

        self.add_subsystem(
            "sls_thrust_convert",
            om.ExecComp(
                "total_scaled_sls_thrust = total_thrust_N * 0.22480894387096263",
                total_thrust_N={"val": np.ones(nn), "units": "N"},
                total_scaled_sls_thrust={"val": np.ones(nn), "units": "lbf"},
                has_diag_partials=True,
            ),
            promotes_outputs=[("total_scaled_sls_thrust", Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST)],
        )
        self.connect("atlas_ptrain_premission.total_thrust", "sls_thrust_convert.total_thrust_N")

        # Provide per-engine scaled SLS thrust for components expecting this pre-mission vector.
        num_engine_types = len(np.atleast_1d(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)))
        try:
            per_engine_sls = np.atleast_1d(
                aviary_inputs.get_val(Aircraft.Engine.SCALED_SLS_THRUST, units="lbf")
            )
        except Exception:
            per_engine_sls = np.ones(num_engine_types) * 20000.0
        if per_engine_sls.size != num_engine_types:
            per_engine_sls = np.ones(num_engine_types) * float(per_engine_sls.ravel()[0])

        engine_sls_ivc = om.IndepVarComp()
        engine_sls_ivc.add_output(Aircraft.Engine.SCALED_SLS_THRUST, val=per_engine_sls, units="lbf")
        self.add_subsystem("engine_sls_ivc", engine_sls_ivc, promotes_outputs=[Aircraft.Engine.SCALED_SLS_THRUST])


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

    def __init__(self, core_propulsion_builder, num_props, use_core_premission=True):
        super().__init__(name="propulsion", meta_data=core_propulsion_builder.meta_data)
        self._core_propulsion_builder = core_propulsion_builder
        self._num_props = num_props
        self._use_core_premission = use_core_premission

    def build_pre_mission(self, aviary_inputs, **kwargs):
        # Keeping core pre-mission propulsion preserves stock sizing/weight plumbing.
        # Set use_core_premission=False to skip building the default turbofan pre-mission model.
        if self._use_core_premission:
            return self._core_propulsion_builder.build_pre_mission(aviary_inputs, **kwargs)
        return AtlasPreMissionPropulsionGroup(
            aviary_inputs=aviary_inputs,
            num_props=self._num_props,
        )

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


def _knots_to_mach(knots):
    """Approximate TAS in knots to Mach for phase initialization."""
    return float(np.clip(knots / 661.0, 0.12, 0.45))


def _build_atlas_14_phase_info():
    """
    Build a 14-phase payload/range-style mission profile using E180 altitude/speed intent.

    Excludes taxi, takeoff, and reserve_taxi_in as requested.
    """
    # altitude/speed intent from aviary/models/aircraft/e180/pr_mission.py
    phase_defs = {
        "climb_e": {"h0_ft": 35.0, "h1_ft": 3000.0, "tas_kn": 180.0, "dur_min": 12.0},
        "climb_hy": {"h0_ft": 3000.0, "h1_ft": 10000.0, "tas_kn": 180.0, "dur_min": 18.0},
        "cruise_1": {"h0_ft": 10000.0, "h1_ft": 10000.0, "tas_kn": 235.0, "dur_min": 20.0},
        "cruise_2": {"h0_ft": 10000.0, "h1_ft": 10000.0, "tas_kn": 235.0, "dur_min": 20.0},
        "descent": {"h0_ft": 10000.0, "h1_ft": 1500.0, "tas_kn": 180.0, "dur_min": 15.0},
        "circuit": {"h0_ft": 1500.0, "h1_ft": 1500.0, "tas_kn": 155.0, "dur_min": 3.0},
        "approach": {"h0_ft": 1500.0, "h1_ft": 50.0, "tas_kn": 130.0, "dur_min": 6.0},
        "reserve_go_around": {"h0_ft": 0.0, "h1_ft": 3500.0, "tas_kn": 130.0, "dur_min": 8.0},
        "reserve_climb": {"h0_ft": 3500.0, "h1_ft": 8000.0, "tas_kn": 180.0, "dur_min": 10.0},
        "reserve_cruise": {"h0_ft": 8000.0, "h1_ft": 8000.0, "tas_kn": 170.0, "dur_min": 20.0},
        "reserve_descent": {"h0_ft": 8000.0, "h1_ft": 1500.0, "tas_kn": 180.0, "dur_min": 10.0},
        "reserve_circuit": {"h0_ft": 1500.0, "h1_ft": 1500.0, "tas_kn": 155.0, "dur_min": 3.0},
        "reserve_approach": {"h0_ft": 1500.0, "h1_ft": 50.0, "tas_kn": 120.0, "dur_min": 6.0},
        "holding_cruise": {"h0_ft": 1500.0, "h1_ft": 1500.0, "tas_kn": 165.0, "dur_min": 30.0},
    }
    phase_sequence = list(phase_defs.keys())

    phase_info = {"post_mission": copy.deepcopy(default_phase_info["post_mission"])}
    phase_plan = {}

    throttle_input_phases = {
        "climb_e",
        "climb_hy",
        "reserve_go_around",
        "reserve_climb",
    }

    for phase_name in phase_sequence:
        cfg = phase_defs[phase_name]
        h0_ft = cfg["h0_ft"]
        h1_ft = cfg["h1_ft"]
        mach = _knots_to_mach(cfg["tas_kn"])
        mach_bounds = (max(0.12, mach - 0.03), min(0.45, mach + 0.03))
        alt_lo = max(0.0, min(h0_ft, h1_ft) - 500.0)
        alt_hi = max(h0_ft, h1_ft) + 500.0

        if h1_ft > h0_ft + 1.0:
            template = copy.deepcopy(default_phase_info["climb"])
        elif h1_ft < h0_ft - 1.0:
            template = copy.deepcopy(default_phase_info["descent"])
        else:
            template = copy.deepcopy(default_phase_info["cruise"])

        uopt = template["user_options"]
        uopt["mach_initial"] = (mach, "unitless")
        uopt["mach_final"] = (mach, "unitless")
        uopt["mach_bounds"] = (mach_bounds, "unitless")
        uopt["altitude_initial"] = (h0_ft, "ft")
        uopt["altitude_final"] = (h1_ft, "ft")
        uopt["altitude_bounds"] = ((alt_lo, alt_hi), "ft")
        if phase_name in throttle_input_phases:
            # Atlas convention: throttle in climb phases is provided as a control input.
            uopt["throttle_enforcement"] = "control"
            uopt["throttle_optimize"] = False
            uopt["throttle_initial"] = (0.65, "unitless")
            uopt["throttle_final"] = (0.65, "unitless")
        else:
            uopt["throttle_enforcement"] = "path_constraint"

        phase_info[phase_name] = template
        phase_plan[phase_name] = {
            "mach_initial": mach,
            "mach_final": mach,
            "altitude_initial_ft": h0_ft,
            "altitude_final_ft": h1_ft,
            "duration_min": cfg["dur_min"],
        }

    phase_info["post_mission"]["target_range"] = (275.0, "nmi")
    return phase_info, phase_sequence, phase_plan


def _set_reasonable_initial_guesses(prob, phase_sequence, phase_plan):
    """Set mission guesses from the 14-phase altitude/speed definitions."""
    t0_s = 0.0
    mass_start = 170000.0
    total_mass_drop = 12000.0
    dm = total_mass_drop / max(1, len(phase_sequence))

    throttle_input_phases = {
        "climb_e",
        "climb_hy",
        "reserve_go_around",
        "reserve_climb",
    }

    for i, phase_name in enumerate(phase_sequence):
        phase = prob.model.traj._phases[phase_name]
        plan = phase_plan[phase_name]
        duration_s = float(plan["duration_min"] * 60.0)

        phase.set_time_val(initial=t0_s, duration=duration_s, units="s")
        t0_s += duration_s

        try:
            phase.set_control_val(
                "mach",
                vals=[plan["mach_initial"], plan["mach_final"]],
                time_vals=[-1, 1],
                units="unitless",
            )
        except Exception:
            pass

        try:
            phase.set_control_val(
                "altitude",
                vals=[plan["altitude_initial_ft"], plan["altitude_final_ft"]],
                time_vals=[-1, 1],
                units="ft",
            )
        except Exception:
            pass

        if phase_name in throttle_input_phases:
            try:
                phase.set_control_val(
                    "throttle",
                    vals=[0.65, 0.65],
                    time_vals=[-1, 1],
                    units="unitless",
                )
            except Exception:
                pass

        m0 = mass_start - i * dm
        m1 = m0 - dm
        try:
            phase.set_state_val("mass", vals=[m0, m1], units="lbm")
        except Exception:
            pass


def _confirm_atlas_powertrain_active(prob, phase_sequence):
    """
    Verify the mission ODE is using the Atlas powertrain by reading a variable that only
    exists in the custom Atlas propulsion mission group.
    """
    candidate_paths = []
    for phase_name in phase_sequence:
        candidate_paths.extend(
            [
                f"traj.{phase_name}.rhs_all.solver_sub.propulsion.atlas_ptrain.total_thrust",
                f"traj.{phase_name}.rhs_all.propulsion.atlas_ptrain.total_thrust",
            ]
        )

    last_exc = None
    for path in candidate_paths:
        try:
            sample = prob.get_val(path)
            print(
                "Atlas powertrain active "
                f"(found '{path}', sample total_thrust shape: {np.shape(sample)})"
            )
            return
        except Exception as exc:  # pragma: no cover - defensive path probing
            last_exc = exc

    raise RuntimeError(
        "Atlas powertrain does not appear to be active in mission ODE wiring."
    ) from last_exc


def _report_battery_soc(prob):
    """Report final SOC from mission timeseries if battery subsystem is present."""
    candidate_paths = [
        f"traj.descent.timeseries.{Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}",
        f"traj.cruise.timeseries.{Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}",
        f"traj.climb.timeseries.{Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE}",
    ]

    for path in candidate_paths:
        try:
            soc = np.atleast_1d(prob.get_val(path))
            print(f"Battery final SOC ({path}): {float(soc[-1]):.4f}")
            return
        except Exception:
            continue

    print("Battery SOC not found in mission timeseries.")


def _run_empirical_battery_model(
    prob,
    phases=("climb", "cruise", "descent"),
    num_props=4,
    num_batt_strings=4,
    battery_datasheet_name="MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105",
):
    """
    Run Atlas empirical battery model using mission electric power output from Aviary.

    This intentionally calls the same battery model used in Atlas workflows
    (aviary/models/engines/propulsion/battery/battery_empirical_power.py).
    """
    phase_names = list(phases)
    phase_durations_h = []
    p_total_kw = []
    altitude_ft = []

    for phase in phase_names:
        time_h = np.atleast_1d(prob.get_val(f"traj.{phase}.timeseries.time", units="h")).ravel()
        p_kw = np.atleast_1d(
            prob.get_val(
                f"traj.{phase}.timeseries.{Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL}",
                units="kW",
            )
        ).ravel()
        alt_ft = np.atleast_1d(
            prob.get_val(f"traj.{phase}.timeseries.{Dynamic.Mission.ALTITUDE}", units="ft")
        ).ravel()

        phase_durations_h.append(float(time_h[-1] - time_h[0]))
        p_total_kw.append(p_kw)
        altitude_ft.append(alt_ft)

    p_total_kw = np.concatenate(p_total_kw)
    altitude_ft = np.concatenate(altitude_ft)
    mission_duration_h = float(np.sum(phase_durations_h))

    # EmpiricalBatteryPower integrator uses Simpson and requires odd num_nodes.
    if p_total_kw.size % 2 == 0:
        p_total_kw = np.concatenate([p_total_kw, p_total_kw[-1:]])
        altitude_ft = np.concatenate([altitude_ft, altitude_ft[-1:]])

    total_nodes = int(p_total_kw.size)

    # Atlas battery model expects per-motor electric demand (nm, num_nodes). Split total evenly.
    p_train_elec_kw = np.tile(p_total_kw / float(num_props), (num_props, 1))

    bat_data = BatteryData.get_data(
        bat_filename=(
            "aviary/models/engines/propulsion/empirical_data/"
            + battery_datasheet_name
            + ".xlsx"
        ),
        cell_sheetname="BOL_cell_fct_CRate",
        config_sheetname="battery_config",
    )

    n_str = int(num_batt_strings)
    if hasattr(bat_data, "n_str"):
        n_str = int(bat_data.n_str)

    batt_prob = om.Problem(reports=False)
    ivc = om.IndepVarComp()
    ivc.add_output("p_train_elec", val=p_train_elec_kw, units="kW")
    ivc.add_output("eta_parc", val=np.ones(total_nodes), units=None)
    ivc.add_output("altitude", val=altitude_ft, units="ft")
    ivc.add_output("disa", val=np.zeros(total_nodes), units="degC")
    ivc.add_output("cell_capacity", val=bat_data.cell_Ah_capacity, units="A*h")
    ivc.add_output("m_cell", val=bat_data.m_cell, units="kg")
    ivc.add_output("cp_cell", val=bat_data.cp_cell, units="J/kg/K")
    ivc.add_output("q_cool_bat", val=25.0 * np.ones((n_str, total_nodes)), units="kW")
    ivc.add_output("soc_initial", val=0.98, units=None)
    ivc.add_output("n_series_per_str", val=bat_data.n_series_per_str, units=None)
    ivc.add_output("n_parallel_per_str", val=bat_data.n_parallel_per_str, units=None)
    ivc.add_output("mission_duration", val=mission_duration_h, units="h")

    batt_prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    batt_prob.model.add_subsystem(
        "batt_emp",
        EmpiricalBatteryPower(
            num_nodes=total_nodes,
            nm=num_props,
            n_str=n_str,
            phases=["mission"],
            feeder_mode="independent",
            active_strings=None,
            battery_datasheet_name=battery_datasheet_name,
        ),
        promotes=["*"],
    )

    for i in range(n_str):
        batt_prob.model.connect("mission_duration", f"soc_integrator_str{i}.duration")
        batt_prob.model.connect(
            "soc_initial",
            f"soc_integrator_str{i}.soc_initial",
        )

    batt_prob.setup()
    batt_prob.run_model()

    soc = np.asarray(batt_prob.get_val("soc"))
    v_bat = np.asarray(batt_prob.get_val("v_bat", units="V"))
    final_soc = soc[:, -1] if soc.ndim == 2 else np.array([float(np.ravel(soc)[-1])])
    print("EmpiricalBatteryPower final SOC per string:", np.array2string(final_soc, precision=4))
    print(
        "EmpiricalBatteryPower min battery voltage (V):",
        float(np.min(v_bat)),
    )

    return {"soc": soc, "v_bat": v_bat}


if __name__ == "__main__":
    PLOT_MISSION = True
    PLOT_N2 = True
    USE_CORE_PREMISSION_PROPULSION = False
    RUN_EMPIRICAL_BATTERY_MODEL = True

    phase_info, mission_phases, phase_plan = _build_atlas_14_phase_info()

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
    atlas_prop = AtlasMissionPropulsionBuilder(
        core_propulsion_builder=core_prop,
        num_props=num_props,
        use_core_premission=USE_CORE_PREMISSION_PROPULSION,
    )

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
    _set_reasonable_initial_guesses(prob, mission_phases, phase_plan)
    run_start = time.perf_counter()
    prob.run_aviary_problem(
        suppress_solver_print=True,
        run_driver=False,
        make_plots=False,
    )
    run_elapsed = time.perf_counter() - run_start
    print(f"Mission simulation runtime: {run_elapsed:.2f} s")
    _confirm_atlas_powertrain_active(prob, mission_phases)
    if RUN_EMPIRICAL_BATTERY_MODEL:
        _run_empirical_battery_model(prob, phases=tuple(mission_phases), num_props=num_props)

    if PLOT_MISSION:
        plot_atlas_aviary_mission(
            prob,
            phases=tuple(mission_phases),
            save_plot=True,
            output_filename="atlas_aviary_mission_profile.png",
            show_plot=True,
        )

    if PLOT_N2:
        om.n2(prob, outfile="atlas_aviary_n2.html", show_browser=True)
