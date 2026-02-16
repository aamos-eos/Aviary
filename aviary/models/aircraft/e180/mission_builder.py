import openmdao.api as om
from atlas.mission.profiles import TrajectoryGroup, UnsteadyFlightPhase, PTrainOnlyPhase 
from atlas.utilities import AddSubtractComp
from openmdao.api import ExplicitComponent, JaxExplicitComponent, BalanceComp
import numpy as np
#from .jax_integrator import DurationGroup
from .duration_converge import DurationGroup
import jax.numpy as jnp
from atlas.mission.split_cruise import SplitCruiseTime
from atlas.utilities.extract_last import ExtractLast
from atlas.utilities.tracktime import TrackTime

class GeneralMission(TrajectoryGroup):
    """
    A general mission builder that iteratively sets up mission segments based on the provided mission_config.
    
    This class loops over the phase names in mission_config["mission"]["phase_names"] and adds subsystems
    for each phase (steady or unsteady), sets up duration balance components or translators, and daisy-chains
    the phases by linking them sequentially.
    
    Assumptions:
    - Takeoff is handled separately if needed (not included in the loop).
    - Connections for balance components (e.g., target altitudes, ranges) need to be added manually after setup
      or via subclassing for specific missions.
    - Promotes all ac|* inputs for aircraft parameters.
    
    Options:
    - mission_config: Dictionary containing mission configuration with phase settings.
    - num_nodes: Number of nodes per phase.
    - aircraft_model: OpenConcept-compliant aircraft model class.
    """
    
    def initialize(self):
        self.options.declare("mission_config", default=None, desc="Dictionary with mission configuration")
        self.options.declare("num_nodes", default=9, desc="Number of points per phase (should be 2N + 1 for Simpson's rule)")
        self.options.declare("aircraft_model", default=None, desc="OpenConcept-compliant airplane model class")
        self.options.declare("point_analysis", default=False, desc="Whether to analyze as phase or static points")
        self.options.declare("turb_op_array", default=[1, 1, 1, 1], desc="Binary array: 1=turbine operating, 0=electric only")
        self.options.declare("drag_model", default="empirical", desc="Whether to  use empirical or simulation based drag model")
        self.options.declare("opt_mission", default=False, desc="Whether to optimize mission parameters")
        self.options.declare("opt_objective", default="range", desc="Whether to optimize for fuel or range")
        self.options.declare("ptrain_model", default="hy_parallel_ptrain", desc="Powertrain model to use")

    def setup(self):
        mission_config = self.options["mission_config"]
        nn = self.options["num_nodes"]
        turb_op_array = np.array(self.options["turb_op_array"])
        acmodelclass = self.options["aircraft_model"]
        drag_model = self.options["drag_model"]
        opt_mission = self.options["opt_mission"]
        opt_objective = self.options["opt_objective"]
        ptrain_model = self.options["ptrain_model"]

        if mission_config is None:
            raise ValueError("mission_config must be provided")
        
        phase_names = mission_config["mission"]["phase_names"]

        # Vectorized: create a boolean array indicating which phase_names contain 'cruise'
        cruise_mask = [("cruise_1" in name) for name in phase_names]
        num_cruise_phases = sum(cruise_mask)
        previous_phase = None
        self.phases = {}  # Dictionary to store phase subsystems for easy access

        
        
        # Create list of phases up to and including first reserve segment
        main_mission_phases = []
        reserve_mission_phases = []
        holding_cruise_phases = []
        found_reserve = False
        
        for phase_name in phase_names:
            if phase_name != "mission" and phase_name != "reserve":
                if not found_reserve and "reserve" not in phase_name:
                    main_mission_phases.append(phase_name)
                elif phase_name != 'holding_cruise':
                    reserve_mission_phases.append(phase_name)
                    found_reserve = True
                else:
                    holding_cruise_phases.append(phase_name)
        
        main_mission_string = main_mission_phases  # Now a list of strings
        reserve_mission_string = reserve_mission_phases  # List of reserve phases
        holding_string = holding_cruise_phases  # List of holding cruise phases


        
        


        # Create main mission group
        if main_mission_string:
            main_mission_group = self.add_subsystem(
                "main_mission_group",
                om.Group(),
                promotes=["*"]
            )
            
            #main_mission_group.linear_solver = om.DirectSolver()
            ##main_mission_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            #main_mission_group.nonlinear_solver.options['maxiter'] = 40
            #main_mission_group.nonlinear_solver.options['iprint'] = 2
            #main_mission_group.nonlinear_solver.options['atol'] = 1e-3
            #main_mission_group.nonlinear_solver.options['rtol'] = 1e-3
            
        
        # Create reserve mission group
        if reserve_mission_string:
            reserve_mission_group = self.add_subsystem(
                "reserve_mission_group",
                om.Group(),
                promotes=["*"]
            )
            
            #reserve_mission_group.linear_solver = om.DirectSolver()
            ##reserve_mission_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            #reserve_mission_group.nonlinear_solver.options['maxiter'] = 40
            # reserve_mission_group.nonlinear_solver.options['iprint'] = 2
            # reserve_mission_group.nonlinear_solver.options['atol'] = 1e-2
            # reserve_mission_group.nonlinear_solver.options['rtol'] = 1e-2
            
        
        # Create holding cruise group
        if holding_string:
            holding_group = self.add_subsystem(
                "holding_group",
                om.Group(),
                promotes=["*"]
            )
            
            #holding_group.linear_solver = om.DirectSolver()
            #holding_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            #holding_group.nonlinear_solver.options['maxiter'] = 40
            #holding_group.nonlinear_solver.options['iprint'] = 2
            #holding_group.nonlinear_solver.options['atol'] = 1e-2
            #holding_group.nonlinear_solver.options['rtol'] = 1e-2
            
        
        # Track which group each phase belongs to
        phase_to_group = {}
        main_phase_count = 0
        reserve_phase_count = 0
        holding_phase_count = 0
        
        for i, phase_name in enumerate(phase_names):



            phase_config = mission_config[phase_name]
            if phase_name != "mission":


                #print("\n\n--------------------------------")
                #print(f"Phase {phase_name}")
                #print("--------------------------------\n\n")


                # Determine phase class based on phase_type
                if phase_config["phase_type"] == "unsteady_eas":
                    phase_class = UnsteadyFlightPhase
                elif phase_config["phase_type"] == "power_only":
                    phase_class = PTrainOnlyPhase
                else:
                    raise ValueError(f"Invalid phase_type: {phase_config['phase_type']}. Must be 'steady_eas', 'unsteady_eas', or 'power_only'")

                
                # Determine which group this phase belongs to
                if phase_name in main_mission_string:
                    parent_group = main_mission_group
                    phase_to_group[phase_name] = "main_mission"
                    main_phase_count += 1
                elif phase_name in reserve_mission_string:
                    parent_group = reserve_mission_group
                    phase_to_group[phase_name] = "reserve_mission"
                    reserve_phase_count += 1
                elif phase_name in holding_string:
                    parent_group = holding_group
                    phase_to_group[phase_name] = "holding"
                    holding_phase_count += 1
                else:
                    # Fallback to main group if not explicitly categorized
                    parent_group = main_mission_group if main_mission_string else self
                    phase_to_group[phase_name] = "main_mission"
                    main_phase_count += 1
                
                # Add the phase subsystem to the appropriate group
                phase = parent_group.add_subsystem(
                    phase_name,
                    phase_class(
                        num_nodes=nn,
                        aircraft_model=acmodelclass,
                        flight_phase=phase_name,
                        mission_config=phase_config,
                        ptrain_model = ptrain_model,
                        # Pass additional unsteady params if needed
                        **({"velocity_in": phase_config["velocity_in"],
                        "power_in": phase_config["power_in"],
                        "true_airspeed_in": phase_config["true_airspeed_in"],
                        "gamma_in": phase_config["gamma_in"],
                        "turb_op_array": turb_op_array,
                        "point_analysis": phase_config["point_analysis"],
                        "drag_model": drag_model,
                        "vs_in": phase_config["vs_in"]} if phase_config["phase_type"] == "unsteady_eas" else {})
                    ),
                    promotes_inputs=["ac|*"],
                )
                
                self.phases[phase_name] = phase
                match_alt = phase_config["match_alt"]
                match_speed = phase_config["match_speed"]
                match_range = phase_config["match_range"]
                vs_in = phase_config["vs_in"]
                gamma_in = phase_config["gamma_in"]
                velocity_in = phase_config["velocity_in"]
                duration_set = phase_config["duration_set"]
                if not phase_config["duration_set"]:

                    y0_set = True

                    if match_alt:
                        dur_cond = "alt"
                    elif match_speed:
                        dur_cond = "speed"
                    elif match_range:
                        dur_cond = "range"
                        if mission_config[phase_names[i]]["trajectory"]["range_target"]["relative"]:
                            y0_set = False
                    else:
                        dur_cond = "duration_set"

                    if "descent" in phase_name or "approach" in phase_name:
                        sign = "neg"
                        if gamma_in and match_alt:
                            self.set_input_defaults(f"{phase_name}.fltcond|vs", np.ones((nn,)) * -500, units="ft/min")
                    elif "climb" in phase_name or "go_around" in phase_name:
                        sign = "pos"
                        if gamma_in and match_alt:
                            self.set_input_defaults(f"{phase_name}.fltcond|vs", np.ones((nn,)) * 500, units="ft/min")
                    else:
                        sign = "pos"

                    # Cruise 1 and reserve cruise duration are based on the distance of other segments combined. 
                    # Integration can't be done within the component itself
                    if phase_name != "cruise_1" and phase_name != "reserve_cruise":
                        phase.add_subsystem(
                            "phasedt",
                            DurationGroup(num_nodes=nn, dur_cond=dur_cond, sign=sign, y0_set=y0_set),
                            promotes_inputs=["*"],
                            promotes_outputs=["*"],
                        )
                    
                        if dur_cond == "alt":
                            # Only connect from previous phase if i > 0 (avoid negative index wraparound)
                            if i > 0 and mission_config[phase_names[i-1]]["phase_type"] != "power_only":
                                if len(phase_names) > 1:
                                    self.connect(f'{phase_names[i-1]}.ode_integ_phase.fltcond|h_final', f'{phase_names[i]}.altitude_duration.y0')
                            #self.connect('seg|h1', 'altitude_duration.y_target')
                            if vs_in == False:
                                self.connect(f'{phase_name}.fltcond|vs', f'{phase_name}.altitude_duration.dy_dt')
                            #if i != 0:
                            #    self.connect(f"{phase_names[i-1]}.t_final", f"{phase_names[i]}.t0")
                        if dur_cond == "speed":
                            self.connect(f'{phase_name}.fltcond|accel_horiz', f'{phase_name}.speed_duration.dy_dt')
                            #if i != 0:
                            #    self.connect(f"{phase_names[i-1]}.t_final", f"{phase_names[i]}.t0")

                        if dur_cond == "range":
                            # Only connect from previous phase if i > 0 (avoid negative index wraparound)
                            if i > 0 and mission_config[phase_names[i-1]]["phase_type"] != "power_only":
                                if len(phase_names) > 1:

                                    if mission_config[phase_names[i]]["trajectory"]["range_target"]["relative"]: # Relative range
                                        pass # y0 defaults to 0 

                                    else:
                                        self.connect(f'{phase_names[i-1]}.ode_integ_phase.range_final', f'{phase_names[i]}.range_duration.y0')


                            self.connect(f"{phase_name}.fltcond|groundspeed", f"{phase_name}.range_duration.dy_dt")


                    # Use a balance component here because cruise and reserve cruise duration are based on the distance of other segments combined. Integration can't be done within the component intself
                    elif dur_cond == "range" and phase_name == "cruise_1" or phase_name == "reserve_cruise":
                        bal_params = {
                            "name": "duration",
                            "units": "s",
                            "eq_units": "km",
                            "val": 1200,
                            "upper": 3e4,
                            "lower": 1e-4,
                        }

                        bal_params["eq_units"] = "km"
                        if phase_name == "cruise_1":
                            bal_params["rhs_name"] = "mission_range"
                        elif phase_name == "reserve_cruise":
                            bal_params["rhs_name"] = "reserve_range"
                        else:
                            bal_params["rhs_name"] = phase_names[i] + "|range_target"
                        bal_params["lhs_name"] = "range_final"
                        phase.add_subsystem(
                            phase_name + "dt",
                            om.BalanceComp(**bal_params),
                            promotes_outputs=["duration"],
                        )
                        if phase_name != "cruise_1" and phase_name != "reserve_cruise":
                            phase.connect("ode_integ_phase.range_final", phase_name + "dt.range_final")

                    
                else:
                    # Duration is set directly
                    phase.add_subsystem(
                        "duration_translator",
                        om.ExecComp(
                            "duration = duration_in",
                            duration={"units": "s", "shape": (1,)},
                            duration_in={"units": "s", "shape": (1,)},
                        ),
                        promotes_inputs=["duration_in"],
                        promotes_outputs=["duration"],
                    )
                # end 

                #if mission_config[phase_name]["propulsion"]["prop"]["prop_thrust_set"]:
                #    phase.non_linear_solver = om.NewtonSolver(solve_subsystems=True)
                #    phase.non_linear_solver.options['maxiter'] = 100
                #    phase.non_linear_solver.options['iprint'] = 2
                #    phase.non_linear_solver.options['atol'] = 1e-6
                #    phase.non_linear_solver.options['rtol'] = 1e-6
                #    phase.linear_solver = om.DirectSolver()


                if previous_phase and phase_name != "holding_cruise": # Don't link holding cruise phase to rest of mission
                    # Link phases - handle cross-group connections
                    self.link_phases(previous_phase, phase)
                    # Connect range and altitude continuity if applicable
                    #self.connect(f"{previous_phase.name}.ode_integ_phase.range_final", f"{phase_name}.ode_integ_phase.range_initial")
                    #self.connect(f"{previous_phase.name}.ode_integ_phase.fltcond|h_final", f"{phase_name}.ode_integ_phase.fltcond|h_initial")
                
                # Add net_reserve_range component after reserve_descent (if it exists) or after reserve_cruise
                if "reserve_cruise" in phase_names:
                    # Add component after reserve_descent if it exists, otherwise after reserve_cruise
                    if phase_name == "reserve_descent":
                        # Add to reserve mission group if it exists, otherwise to main group
                        parent_group = reserve_mission_group if reserve_mission_string else self
                        net_reserve_range = parent_group.add_subsystem(
                            "net_reserve_range",
                            AddSubtractComp(
                                input_names=["reserve_range_initial", "reserve_range_final"],
                                output_name="net_reserve_range",
                                vec_size=1,
                                length=1,
                                scaling_factors=[-1, 1],  # Subtract initial, add final
                                units="NM"
                            ),
                            promotes_outputs=["net_reserve_range"]
                        )
                        
                        # Connect inputs - use full path for cross-group connections
                        if "approach" in main_mission_string and reserve_mission_string:
                            self.connect("approach.ode_integ_phase.range_final", 
                                       "net_reserve_range.reserve_range_initial")
                        else:
                            self.connect("approach.ode_integ_phase.range_final", "net_reserve_range.reserve_range_initial")
                        
                        if reserve_mission_string:
                            self.connect("reserve_approach.ode_integ_phase.range_final", 
                                       "net_reserve_range.reserve_range_final")
                            self.connect("net_reserve_range", 
                                       "reserve_cruise.reserve_cruisedt.range_final")
                        else:
                            self.connect("reserve_approach.ode_integ_phase.range_final", "net_reserve_range.reserve_range_final")
                            self.connect("net_reserve_range", "reserve_cruise.reserve_cruisedt.range_final")
            
                        
                    elif phase_name == "reserve_cruise" and "reserve_descent" not in phase_names:
                        # Only add after reserve_cruise if reserve_descent doesn't exist
                        parent_group = reserve_mission_group if reserve_mission_string else self
                        net_reserve_range = parent_group.add_subsystem(
                            "net_reserve_range",
                            AddSubtractComp(
                                input_names=["reserve_range_initial", "reserve_range_final"],
                                output_name="net_reserve_range",
                                vec_size=1,
                                length=1,
                                scaling_factors=[-1, 1],  # Subtract initial, add final
                                units="NM"
                            ),
                            promotes_outputs=["net_reserve_range"]
                        )
                        
                        # Connect inputs - use full path for cross-group connections
                        if "approach" in main_mission_string and reserve_mission_string:
                            self.connect("approach.ode_integ_phase.range_final", 
                                       "net_reserve_range.reserve_range_initial")
                        else:
                            self.connect("approach.ode_integ_phase.range_final", "net_reserve_range.reserve_range_initial")
                        
                        if reserve_mission_string:
                            self.connect("reserve_cruise.ode_integ_phase.range_final", 
                                       "net_reserve_range.reserve_range_final")
                            self.connect("net_reserve_range", 
                                       "net_reserve_range", 
                                       "reserve_cruise.reserve_cruisedt.range_final")
                        else:
                            self.connect("reserve_cruise.ode_integ_phase.range_final", "net_reserve_range.reserve_range_final")
                            self.connect("net_reserve_range", "reserve_cruise.reserve_cruisedt.range_final")
                
                # Only add if vs_in and match_alt, or
                phase.add_subsystem("track_time", TrackTime(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
                if i < len(phase_names) - 1:
                    self.connect(f"{phase_names[i]}.t_final", f"{phase_names[i+1]}.t0")

                if "descent" in phase_name or "approach" in phase_name:
                    self.set_input_defaults(f"{phase_name}.altitude_duration.dy_dt", -10 * np.ones(nn), units = 'm/s')
                elif "climb" in phase_name or "go_around" in phase_name:
                    self.set_input_defaults(f"{phase_name}.altitude_duration.dy_dt", 10 * np.ones(nn), units = 'm/s')

                if "cruise" in phase_name and phase_name != "holding_cruise":
                    self.connect(f"{phase_names[i-1]}.ode_integ_phase.fltcond|h_final", f"{phase_name}.bcast_height.scalar")


                if phase_config["phase_type"] == "unsteady_eas":
                    self.set_input_defaults(f"{phase_names[i]}.weight", 36000 * np.ones(nn), units="kg")

                previous_phase = phase

                if opt_mission and "cruise_1" in phase_names and "cruise_2" in phase_names and opt_objective == "fuel":
                    if phase_name == "cruise_1":
                        if mission_config["cruise_1"]["match_range"]:
                            main_mission_group.add_subsystem("split_cruise_time", 
                                            SplitCruiseTime(split_wrt="cruise_1"),
                                            promotes_inputs=["hy_cruise_t_ratio"])
                    if phase_name == "cruise_2":
                        self.connect("cruise_1.duration", "split_cruise_time.cruise_1|duration")
                        self.connect("split_cruise_time.cruise_2|duration", "cruise_2.duration_in")
                        #self.connect("cruise_1.hy_cruise_t_ratio", "split_cruise_time.hy_cruise_t_ratio")

            # end 
        # Additional mission parameters (can be extended in subclasses)
        
        # Handle cruise range connections with new group structure
        # Determine the end phase for mission range calculation
        # Priority: approach > descent > cruise_1 (self-reference)
        if "cruise_1" in phase_names:
            has_approach = "approach" in phase_names
            has_descent = "descent" in phase_names
            
            # Determine which phase marks the end of the mission for range calculation
            if has_approach:
                end_phase = "approach"
            elif has_descent:
                end_phase = "descent"
            else:
                end_phase = None  # Will use cruise_1's own range_final
            
            if end_phase is not None and "takeoff" in phase_names:
                # Add net cruise range component to calculate actual cruise range
                # (end_phase.range_final - takeoff.range_final) to exclude taxi/takeoff
                if "cruise_1" in main_mission_string:
                    parent_group = main_mission_group
                else:
                    parent_group = self
                    
                net_cruise_range = parent_group.add_subsystem(
                    "net_cruise_range",
                    AddSubtractComp(
                        input_names=["end_phase_range_final", "takeoff_range_final"],
                        output_name="net_cruise_range",
                        vec_size=1,
                        length=1,
                        scaling_factors=[1, -1],  # Add end phase final, subtract takeoff final
                        units="NM"
                    ),
                    promotes_outputs=["net_cruise_range"]
                )
                
                # Connect inputs
                self.connect(f"{end_phase}.ode_integ_phase.range_final", 
                           "net_cruise_range.end_phase_range_final")
                self.connect("takeoff.ode_integ_phase.range_final", 
                           "net_cruise_range.takeoff_range_final")
                
                # Connect to cruise duration component
                if not mission_config["cruise_1"]["duration_set"]:
                    self.connect("net_cruise_range", "cruise_1.cruise_1dt.range_final")
                    
            elif end_phase is not None:
                # No takeoff phase - connect end phase range directly
                if not mission_config["cruise_1"]["duration_set"]:
                    self.connect(f"{end_phase}.ode_integ_phase.range_final", 
                                "cruise_1.cruise_1dt.range_final")
            else:
                # No approach or descent - use cruise_1's own range_final (feedback)
                if not mission_config["cruise_1"]["duration_set"]:
                    self.connect("cruise_1.ode_integ_phase.range_final", 
                                "cruise_1.cruise_1dt.range_final")

        if "holding_cruise" in phase_names: # Assumes holding cruise is second last phase
            self.connect(f"{phase_names[-2]}.fuel_used_final", f"holding_cruise.fuel_used_initial")
            self.connect(f"{phase_names[-2]}.ode_integ_phase.range_final", f"holding_cruise.ode_integ_phase.range_initial")
              

        self.add_subsystem("extract_last_fuel", ExtractLast(num_nodes=nn, units = 'kg', mode = 'extract_scalar'), promotes_outputs=[])
        self.connect(f"{phase_names[-1]}.fuel_used", "extract_last_fuel.input_vector")

        
        self.linear_solver = om.DirectSolver()
        # Keep a mission-level Newton solve for global closures (e.g., fuel <-> TOW),
        # but avoid re-solving all mission subsystems every Newton iteration.
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['atol'] = 1e-5  # global closure tolerance
        self.nonlinear_solver.options['rtol'] = 1e-5

        #self.connect(f"{main_mission_string[-1]}.t_final", f"{reserve_mission_string[0]}.t0")

 