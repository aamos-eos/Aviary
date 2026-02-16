import os
import logging
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om
import openmdao.jax as omj
import copy
from openmdao.api import (
    Group,
    IndepVarComp,
)
import pdb
from atlas.utilities import AddSubtractComp, DictIndepVarComp
from atlas.mission import GeneralMission
from atlas.scenarios.setup_mission.setup_ac import ParallelHybridAircraft
from atlas.scenarios.setup_mission.setup_ac_dvcomp import setup_ac_dv
from atlas.propulsion.battery.battery_empirical_power import EmpiricalBatteryPower
from atlas.utilities.chain_vars import ChainVars
from atlas.utilities.extract_last import ExtractLast
from atlas.utilities.broadcast_scalars import ScalarToVectorBroadcast, ScalarToMatrixBroadcast
from atlas.utilities.matrix_vector_converter import MatrixToVectorConverter
from atlas.mission.phases import UnsteadyFlightPhase
from atlas.mission.split_cruise import SplitCruiseTime
from atlas.scenarios.setup_mission.objective_func import ObjectiveFuncRange, ObjectiveFuncFuel
from atlas.scenarios.setup_mission.smooth_step_cntrl import (
    SmoothstepTransition, 
    SmoothstepTransitionGroup, 
    VectorToNacelleMatrix,
)
from atlas.scenarios.setup_mission.fuel_lim_comp import FuelLimitComp
from atlas.weights.compute_oew import OEWGroup, PayloadWeight, FuelWeight
# ObjectiveFunc class moved here to avoid circular import
from atlas.weights.oew_analysis import connect_oew_inputs
from atlas.weights.parameter_links import FuselageParameterLinks, WingParameterLinks
from atlas.utilities.dvlabel import DVLabel
from atlas.weights.wing_geometry_comp import WingGeometryComp

class TOWMarginComponent(om.ExplicitComponent):
    """
    Component that computes TOW margin using absolute value.
    """
    def initialize(self):
        pass
    
    def setup(self):
        self.add_input("TOW", units="kg")
        self.add_input("OEW", units="kg")
        self.add_input("total_fuel", units="kg")
        self.add_input("payload", units="kg")
        
        self.add_output("TOW_margin", units="kg")
        
        # Declare partials
        self.declare_partials("TOW_margin", "TOW")
        self.declare_partials("TOW_margin", "OEW")
        self.declare_partials("TOW_margin", "total_fuel")
        self.declare_partials("TOW_margin", "payload")
    
    def compute(self, inputs, outputs):
        # Compute the margin using absolute value
        margin = inputs["TOW"] - inputs["OEW"] - inputs["total_fuel"] - inputs["payload"]
        outputs["TOW_margin"] = np.abs(margin)
        print(f"TOW Margin: {outputs['TOW_margin']}")
        print(f"TOW: {inputs['TOW']}")
    
    def compute_partials(self, inputs, partials):
        # Compute the margin
        margin = inputs["TOW"] - inputs["OEW"] - inputs["total_fuel"] - inputs["payload"]
        
        # d(|margin|)/d(margin) = sign(margin) = margin / |margin| for margin != 0
        # For margin = 0, we use 0 as the derivative (subgradient)
        if margin != 0:
            sign = np.sign(margin)
        else:
            sign = 0.0
        
        # d(TOW_margin)/d(TOW) = sign(margin) * d(margin)/d(TOW) = sign(margin) * 1
        partials["TOW_margin", "TOW"] = sign
        
        # d(TOW_margin)/d(OEW) = sign(margin) * d(margin)/d(OEW) = sign(margin) * (-1)
        partials["TOW_margin", "OEW"] = -sign
        
        # d(TOW_margin)/d(total_fuel) = sign(margin) * d(margin)/d(total_fuel) = sign(margin) * (-1)
        partials["TOW_margin", "total_fuel"] = -sign
        
        # d(TOW_margin)/d(payload) = sign(margin) * d(margin)/d(payload) = sign(margin) * (-1)
        partials["TOW_margin", "payload"] = -sign


class LinkBatteryInputs(om.Group):


    def initialize(self):
        self.options.declare("mission_config", default=None)
        self.options.declare("ac_data", default=None)

    def setup(self):
        mission_config = self.options["mission_config"]
        ac_data = self.options["ac_data"]

        self.add_subsystem("n_series_per_str", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("n_parallel_per_str", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("n_str", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("cell_capacity", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("cp_cell", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("m_cell", IndepVarComp(), promotes_outputs=["*"])
        self.add_subsystem("q_cool_bat", IndepVarComp(), promotes_outputs=["*"])

class AnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis."""
    
    def initialize(self):
        self.options.declare("mission_config", default=None)
        self.options.declare("ac_data", default=None)
        self.options.declare("nn", default=11)
        self.options.declare("soc_min", default=0.13)
        self.options.declare("min_voltage", default=570)
        self.options.declare("fuel_lim", default=3500)
        self.options.declare("connect_battery_voltage", default=False)
        self.options.declare("converge_tow", default=False)
        self.options.declare("point_analysis", default=False)
        self.options.declare("size_motor", default=False)
        self.options.declare("size_battery", default=False)
        self.options.declare("drag_model", default="empirical", desc="Whether to  use empirical or simulation based drag model")
        self.options.declare("turb_op_array", default=None)
        self.options.declare("compute_oew", default=False)
        self.options.declare("compute_payload", default=False)
        self.options.declare("opt_objective", default="range", desc="Optimization objective: range or fuel")
        self.options.declare('feeder_mode', default='independent', desc='Feeder mode: independent, full_crossfeed, or grouped')
        self.options.declare('opt_mission', default=False, desc='If True, optimize mission parameters like cruise time split')
        self.options.declare("opt_wing", default=False, desc='If True, optimize wing geometry')
        self.options.declare("size_tow", default=False, desc='If True, TOW is provided externally for point_analysis')
        # Wing + ESS sizing options
        self.options.declare("total_propulsion_mass_budget", default=26549.0, desc='Total mass budget for propulsion (kg)')
        self.options.declare("pack_scale_factor", default=1.247, desc='ESS packaging scale factor')
        self.options.declare("battery_datasheet_name", default='MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105', desc='Name of the battery datasheet to use for interpolation')


    def setup(self):
        mission_config = self.options["mission_config"]
        ac_data = self.options["ac_data"]
        nn = self.options["nn"]
        turb_op_array = self.options["turb_op_array"]
        converge_tow = self.options["converge_tow"]
        connect_battery_voltage = self.options["connect_battery_voltage"]
        drag_model = self.options["drag_model"]
        point_analysis = self.options["point_analysis"]
        compute_oew = self.options["compute_oew"]
        compute_payload = self.options["compute_payload"]
        opt_objective = self.options["opt_objective"]
        n_str = ac_data["ac"]["propulsion"]["battery"]["n_str"]["value"]
        feeder_mode = self.options["feeder_mode"]
        phases = mission_config["mission"]["phase_names"]
        size_motor = self.options["size_motor"]
        size_battery = self.options["size_battery"]
        opt_wing = self.options["opt_wing"]
        size_tow = self.options["size_tow"]
        battery_datasheet_name = self.options["battery_datasheet_name"]

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem("dv_comp", DictIndepVarComp(ac_data), promotes_outputs=["*"])
        dv_comp = setup_ac_dv(dv_comp, point_analysis=point_analysis, compute_oew=compute_oew, 
        compute_payload=compute_payload, size_motor=size_motor, size_battery=size_battery, opt_wing=opt_wing, size_tow=size_tow)
        
        # When size_motor=True, motor rating is provided externally as an input.
        # Use DVLabel passthrough: input 'rated_power_em_scalar' -> output 'ac|propulsion|motor|rating'
        if size_motor:
            default_rating = ac_data["ac"]["propulsion"]["motor"]["rating"]["value"]
            rating_units = ac_data["ac"]["propulsion"]["motor"]["rating"]["units"]
            self.add_subsystem(
                'motor_rating_passthru',
                DVLabel([['rated_power_em_scalar', 'ac|propulsion|motor|rating', default_rating, rating_units]]),
                promotes_inputs=['rated_power_em_scalar'],
                promotes_outputs=['ac|propulsion|motor|rating']
            )
        
        # When size_battery=True, n_parallel_per_str is provided externally as an input.
        # Use DVLabel passthrough: input 'n_parallel_per_str_dv' -> output 'ac|propulsion|battery|n_parallel_per_str'
        if size_battery:
            default_n_parallel = ac_data["ac"]["propulsion"]["battery"]["n_parallel_per_str"]["value"]
            self.add_subsystem(
                'battery_n_parallel_passthru',
                DVLabel([['n_parallel_per_str_dv', 'ac|propulsion|battery|n_parallel_per_str', default_n_parallel, None]]),
                promotes_inputs=['n_parallel_per_str_dv'],
                promotes_outputs=['ac|propulsion|battery|n_parallel_per_str']
            )
        
        # When opt_wing=True, S_ref and AR are provided externally as inputs.
        # Use DVLabel passthrough: input 'S_ref_dv' -> output 'ac|geom|wing|S_ref'
        if opt_wing:
            default_S_ref = ac_data["ac"]["geom"]["wing"]["S_ref"]["value"]
            S_ref_units = ac_data["ac"]["geom"]["wing"]["S_ref"]["units"]
            self.add_subsystem(
                'wing_S_ref_passthru',
                DVLabel([['S_ref_dv', 'ac|geom|wing|S_ref', default_S_ref, S_ref_units]]),
                promotes_inputs=['S_ref_dv'],
                promotes_outputs=['ac|geom|wing|S_ref']
            )
            
            default_AR = ac_data["ac"]["geom"]["wing"]["AR"]["value"]
            self.add_subsystem(
                'wing_AR_passthru',
                DVLabel([['AR_dv', 'ac|geom|wing|AR', default_AR, None]]),
                promotes_inputs=['AR_dv'],
                promotes_outputs=['ac|geom|wing|AR']
            )

            default_toverc = ac_data["ac"]["geom"]["wing"]["toverc"]["value"]
            self.add_subsystem(
                'wing_toverc_passthru',
                DVLabel([['wing_toverc_dv', 'ac|geom|wing|toverc', default_toverc, None]]),
                promotes_inputs=['wing_toverc_dv'],
                promotes_outputs=['ac|geom|wing|toverc']
            )

            # Compute all wing geometry from S_ref and AR
            # This ensures span, MAC, chords, wetted area are consistent with design variables
            default_taper = ac_data["ac"]["geom"]["wing"]["taper"]["value"]
            default_c0_sweep = ac_data["ac"]["geom"]["wing"]["c0_sweep"]["value"]
            default_toverc = ac_data["ac"]["geom"]["wing"]["toverc"]["value"]
            self.add_subsystem(
                'wing_geometry',
                WingGeometryComp(
                    default_S_ref=default_S_ref,
                    default_AR=default_AR,
                    default_taper=default_taper,
                    default_c0_sweep=default_c0_sweep,
                    default_toverc=default_toverc,
                ),
                promotes_inputs=[
                    ('S_ref', 'ac|geom|wing|S_ref'),
                    ('AR', 'ac|geom|wing|AR'),
                    ('taper', 'ac|geom|wing|taper'),
                    ('c0_sweep', 'ac|geom|wing|c0_sweep'),
                    ('toverc', 'ac|geom|wing|toverc'),
                ],
                promotes_outputs=[
                    ('span', 'ac|geom|wing|span'),
                    ('c_root', 'ac|geom|wing|c_root'),
                    ('c_tip', 'ac|geom|wing|c_tip'),
                    ('MAC', 'ac|geom|wing|MAC'),
                    ('mac_buttline', 'ac|geom|wing|mac_buttline'),
                    ('S_wet', 'ac|geom|wing|S_wet'),
                    ('airfoil_pc', 'ac|geom|wing|airfoil_pc'),
                    ('c4_sweep', 'ac|geom|wing|c4_sweep'),
                ]
            )



        # Add SplitCruiseTime component when opt_mission=True and both cruise phases exist
        opt_mission = self.options["opt_mission"]
        if opt_mission and "cruise_1" in phases and "cruise_2" in phases:
            if mission_config["cruise_1"]["duration_set"]:
                self.add_subsystem("split_cruise_time", 
                                SplitCruiseTime(),
                                promotes_inputs=["cruise_duration", "hy_cruise_t_ratio"])


        npp = len(turb_op_array)



        if compute_oew:
            # Add the OEW Group
            self.add_subsystem('oew', OEWGroup(), promotes_inputs=['*'], promotes_outputs=['*'])

            if not self.options["point_analysis"]:
                self.connect("OEW", "ac|weights|OEW")


            


        if compute_payload:

            if not compute_oew:
                self.add_subsystem('fuse_parameter_links', FuselageParameterLinks(), 
                promotes_inputs=['num_passengers'],
                promotes_outputs=['length_pax_cabin', 'fuselage_length', 'lt', 'Sg'])

                # Link wing geometry parameters to compute LEMAC and front_spar_location
                # Don't promote taper_ratio, span, c0_sweep, MAC - connect them explicitly
                self.add_subsystem('wing_parameter_links', WingParameterLinks(), 
                                promotes_inputs=['wing_apex_percentage', 'fuselage_length', 'MAC'],
                                promotes_outputs=['LEMAC', 'front_spar_location'])

            # Add Payload
            self.add_subsystem('payload', PayloadWeight(), 
                                promotes_inputs=['*'],
                                promotes_outputs=['payload_weight', 'cg_passenger', 'passenger_weight',
                                                'cargo_weight', 'cg_cargo'])
            if not self.options["point_analysis"]:
                self.connect("payload_weight", "ac|weights|payload")


        if compute_oew or compute_payload:
            connect_oew_inputs(self, compute_payload=compute_payload, compute_oew=compute_oew)


        if not self.options["point_analysis"]:



            self.add_subsystem("calc_fuel_limit", FuelLimitComp(), 
                            promotes_inputs=["*"],
                            promotes_outputs=["fuel_limit"])

                

            # ================================================================
            # Redefine TOW using the updated OEW from wing/ESS sizing
            # TOW = OEW_updated + payload + fuel_used
            # ================================================================




            #promotes_inputs=["ac|weights|OEW", "ac|weights|payload", "total_fuel_used"],
            #promotes_outputs=["ac|weights|TOW"],
            
            # Add cruise_1 hybridization ratio control if optimization is enabled
            if "cruise_1" in phases and mission_config["cruise_1"]["propulsion"]["s_curve_hy_profile"]:
                npp = ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"]
                const_hy_ratio = mission_config["cruise_1"]["propulsion"]["const_hy_ratio"]
                eq_hy_nacelles_op = mission_config["cruise_1"]["propulsion"]["eq_hy_nacelles_op"]
                var_name = "cruise_1|hy_ratio"
                promotes_inputs_hy = []
                
                if eq_hy_nacelles_op:
                    # Single control for all operating nacelles
                    if not const_hy_ratio:  # transient
                        promotes_inputs_hy = [
                            f"{var_name}_start", f"{var_name}_end",
                            f"{var_name}_loc_trans", f"{var_name}_w_trans"
                        ]
                        self.add_subsystem("cruise_hy_ratio_comp", 
                            SmoothstepTransitionGroup(
                                num_rows=npp, num_nodes=nn, 
                                op_array=turb_op_array,
                                var_name=var_name,
                                eq_control=True),
                            promotes_inputs=promotes_inputs_hy,
                            promotes_outputs=[])
                    else:  # constant
                        promotes_inputs_hy = [var_name]
                        self.add_subsystem("cruise_hy_ratio_comp", 
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn, 
                                op_array=turb_op_array,
                                var_name=var_name,
                                control_law="constant",
                                eq_control=True),
                            promotes_inputs=promotes_inputs_hy,
                            promotes_outputs=[])
                else:
                    # Independent control for each operating nacelle
                    for i in range(npp):
                        if turb_op_array[i] == 1:
                            nac_var = f"{var_name}_nac{i+1}"
                            if not const_hy_ratio:  # transient
                                promotes_inputs_hy.extend([
                                    f"{nac_var}_start", f"{nac_var}_end",
                                    f"{nac_var}_loc_trans", f"{nac_var}_w_trans"
                                ])
                            else:  # constant
                                promotes_inputs_hy.append(nac_var)
                    
                    if not const_hy_ratio:  # transient
                        self.add_subsystem("cruise_hy_ratio_comp", 
                            SmoothstepTransitionGroup(
                                num_rows=npp, num_nodes=nn, 
                                op_array=turb_op_array,
                                var_name=var_name,
                                eq_control=False),
                            promotes_inputs=promotes_inputs_hy,
                            promotes_outputs=[])
                    else:  # constant
                        self.add_subsystem("cruise_hy_ratio_comp", 
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn, 
                                op_array=turb_op_array,
                                var_name=var_name,
                                control_law="constant",
                                eq_control=False),
                            promotes_inputs=promotes_inputs_hy,
                            promotes_outputs=[])

            # Add climb_hy power control if optimization is enabled
            if "climb_hy" in phases and mission_config["climb_hy"]["propulsion"]["s_curve_power_profile"]:
                npp = int(round(ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"], 0))
                const_power_profile = mission_config["climb_hy"]["propulsion"]["const_power_profile"]
                eq_power_nacelles_op = mission_config["climb_hy"]["propulsion"]["eq_power_nacelles_op"]
                var_name = "climb_hy|total_mech_power_out"
                power_units = "kW"
                promotes_inputs_power = []
                
                if eq_power_nacelles_op:
                    # Single smoothstep for all nacelles
                    if not const_power_profile:  # transient
                        promotes_inputs_power = [
                            f"{var_name}_start", f"{var_name}_end",
                            f"{var_name}_loc_trans", f"{var_name}_w_trans"
                        ]
                        self.add_subsystem("climb_hy_power_comp",
                            SmoothstepTransitionGroup(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,
                                units=power_units,
                                eq_control=True),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])
                    else:  # constant
                        promotes_inputs_power = [var_name]
                        self.add_subsystem("climb_hy_power_comp",
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,
                                units=power_units,
                                control_law="constant",
                                eq_control=True),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])
                else:
                    # Independent control for each active nacelle
                    for i in range(npp):
                        if turb_op_array[i] == 1:
                            nac_var = f"{var_name}_nac{i+1}"
                            if not const_power_profile:  # transient
                                promotes_inputs_power.extend([
                                    f"{nac_var}_start", f"{nac_var}_end",
                                    f"{nac_var}_loc_trans", f"{nac_var}_w_trans"
                                ])
                            else:  # constant
                                promotes_inputs_power.append(nac_var)
                    
                    if not const_power_profile:  # transient
                        self.add_subsystem("climb_hy_power_comp",
                            SmoothstepTransitionGroup(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,
                                units=power_units,
                                eq_control=False),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])
                    else:  # constant
                        self.add_subsystem("climb_hy_power_comp",
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,  
                                units=power_units,
                                control_law="constant",
                                eq_control=False),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])

            # Handle non-s_curve power profile cases (direct scalar/vector to matrix broadcast)
            for phase in phases:
                if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"] and not mission_config[phase]["propulsion"]["s_curve_power_profile"]:
            #elif "climb_hy" in phases and not mission_config["climb_hy"]["propulsion"]["s_curve_power_profile"]:
                    npp = int(round(ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"], 0))
                    const_power_profile = mission_config[phase]["propulsion"]["const_power_profile"]
                    eq_power_nacelles_op = mission_config[phase]["propulsion"]["eq_power_nacelles_op"]
                    power_units = "kW"
                    promotes_inputs_power = []
                    
                    # Get nacelle operating mask from config (shape: npp x nn)
                    nac_op_array = mission_config[phase]["propulsion"]["nac_op_array"]
                    
                    if eq_power_nacelles_op and const_power_profile:
                        var_name = f"{phase}|total_mech_power_out_scalar"

                        # Case 1: Single scalar value for all nacelles, constant over time
                        # Input: scalar -> Output: matrix (npp, nn)
                        promotes_inputs_power = [("scalar", var_name)]
                        self.add_subsystem(f"{phase}_power_comp",
                            ScalarToMatrixBroadcast(
                                num_nodes=nn,
                                num_comps=npp,
                                units=power_units),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[("matrix", f"{var_name}_matrix")])
                        
                        # Apply nacelle operating mask to the broadcasted power
                        # This zeros out power for inoperative nacelles (e.g., OEI scenarios)
                        masked_var_name = f"{var_name}_masked"
                        self.add_subsystem(f"{phase}_power_mask",
                            om.ExecComp(
                                'masked_power = shaft_power * nac_op_mask',
                                shaft_power={'val': np.ones((npp, nn)), 'units': power_units, 'shape': (npp, nn)},
                                nac_op_mask={'val': nac_op_array, 'shape': (npp, nn)},
                                masked_power={'val': np.ones((npp, nn)), 'units': power_units, 'shape': (npp, nn)}),
                            promotes_outputs=[("masked_power", masked_var_name)])
                        
                        # Connect broadcast output to mask component
                        self.connect(f"{var_name}_matrix", f"{phase}_power_mask.shaft_power")
                        
                        # Connect masked power to downstream propulsion components
                        self.connect(masked_var_name, 
                                [f"{phase}.hy_parallel_ptrain.nacelles.total_mech_power_out", f"{phase}.hy_parallel_ptrain.nacelles.gb_comp.power_in"])
                    
                    """
                    elif const_power_profile and not eq_power_nacelles_op:
                        var_name = f"{phase}|total_mech_power_out_vector"
                        # Case 2: Each nacelle has its own constant power (different per nacelle)
                        # Input: vector (npp,) -> Output: matrix (npp, nn) where each row is constant
                        for i in range(npp):
                            promotes_inputs_power.append(f"{var_name}_nac{i+1}")
                        self.add_subsystem(f"{phase}_power_comp",
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,
                                units=power_units,
                                control_law="constant",
                                eq_control=False),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])
                    
                    elif not const_power_profile and eq_power_nacelles_op:
                        var_name = f"{phase}|total_mech_power_out_vector"
                        # Case 3: Same power profile for all nacelles, varying over time
                        # Input: vector (nn,) -> Output: matrix (npp, nn) where each row is the same
                        promotes_inputs_power = [var_name]
                        self.add_subsystem(f"{phase}_power_comp",
                            VectorToNacelleMatrix(
                                num_rows=npp, num_nodes=nn,
                                op_array=np.ones(npp),
                                var_name=var_name,
                                units=power_units,
                                control_law="transient",
                                eq_control=True),
                            promotes_inputs=promotes_inputs_power,
                            promotes_outputs=[])
                    """

                                # Connect climb_hy power output if enabled (for both s_curve and non-s_curve cases)
                    #if const_power_profile or eq_power_nacelles_op:
                    #    var_name = f"{phase}|total_mech_power_out"
                    #    
                    #    self.connect(f"{phase}_power_comp.{var_name}_matrix", 
                    #            [f"{phase}.hy_parallel_ptrain.nacelles.total_mech_power_out", f"{phase}.hy_parallel_ptrain.nacelles.gb_comp.power_in"])

            analysis = self.add_subsystem("analysis", GeneralMission(num_nodes=nn, aircraft_model=ParallelHybridAircraft, ptrain_model="hy_parallel_ptrain",
            mission_config=mission_config, drag_model=drag_model, opt_mission=opt_mission,turb_op_array=turb_op_array, opt_objective=opt_objective),promotes_inputs=["*"],promotes_outputs=["*"])
            
            # Connect cruise_1 hybridization ratio component output if enabled
            if "cruise_1" in phases and mission_config["cruise_1"]["propulsion"]["s_curve_hy_profile"]:
                self.connect("cruise_hy_ratio_comp.cruise_1|hy_ratio_matrix", 
                           "cruise_1.hy_parallel_ptrain.nacelles.power_split_fraction_gt")
            
            # Connect climb_hy power output if enabled
            if "climb_hy" in phases and mission_config["climb_hy"]["propulsion"]["s_curve_power_profile"]:
                var_name = "climb_hy|total_mech_power_out"
                self.connect(f"climb_hy_power_comp.{var_name}_matrix", 
                           ["climb_hy.hy_parallel_ptrain.nacelles.total_mech_power_out", "climb_hy.hy_parallel_ptrain.nacelles.gb_comp.power_in"])

            

            # Connect SplitCruiseTime outputs to cruise phase duration inputs
            if opt_mission and "cruise_1" in phases and "cruise_2" in phases:
                if opt_objective == "range":
                    self.connect("split_cruise_time.cruise_1|duration", "cruise_1.duration_in")
                    self.connect("split_cruise_time.cruise_2|duration", "cruise_2.duration_in")
                #elif opt_objective == "fuel":
                #    self.connect("split_cruise_distance.cruise_2|range_target", "cruise_2.range_duration.y_target")

            analysis.add_subsystem("redefine_tow", RedefineTOWComp(), promotes_inputs=["*"], promotes_outputs=["*"])
            #self.add_subsystem(
            #        "analysis",
            #        GeneralMission(num_nodes=nn, aircraft_model=ParallelHybridAircraft, mission_config=mission_config),
            #        promotes_inputs=["*"],
            #        promotes_outputs=["*"],
            #    )


            #if compute_oew or compute_payload:

            # Compute fuel limit = MTOW - payload - OEW
            fuel_constraint_comp = AddSubtractComp(
                output_name="fuel_margin",
                input_names=["fuel_limit", "total_fuel_used"],
                vec_size=1,
                units="kg",
                scaling_factors=[-1.0, 1.0]  # total_fuel_used - fuel_limit
            )
            self.add_subsystem("fuel_margin_comp", fuel_constraint_comp, 
                            promotes_inputs=["total_fuel_used", "fuel_limit"],
                            promotes_outputs=["fuel_margin"])

            phases_bat_chain = phases

            # make a deep copy. keep battery in holding cruise phase. 
            phases_bat_chain = copy.deepcopy(phases)


            #sduration_inputs = []
            # Make Connections for Chain Vars
            npp = ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"]
            for phase in phases_bat_chain:
                phase_config = mission_config[phase]

                self.connect(f"{phase}.hy_parallel_ptrain.q_cool_bat", f"{phase}_q_cool_bat")
                self.connect(f"{phase}.p_train_elec", f"{phase}_p_train_elec")
                self.connect(f"{phase}.t", f"{phase}_t")
                # eta_parc, altitude, and disa are inputs, not outputs - connect from promoted inputs
                self.connect(f"{phase}|eta_parc", f"{phase}_eta_parc")
                if "cruise" not in phase:
                    self.connect(f"{phase}.fltcond|h", f"{phase}_altitude")
                else:
                    self.connect(f"{phase}.bcast_height.vector", f"{phase}_altitude")
                self.connect(f"{phase}|fltcond|disa", f"{phase}_disa")
                #duration_inputs.append(f"{phase}_duration")

                #if len(phases_bat_chain) > 1:
                #    self.connect(f"{phase}.duration", f"batt1.{phase}_duration")
                #else:
                #    self.connect(f"{phase}.duration", f"batt1.duration")

            

                for j in range(0,n_str):
                    if len(phases_bat_chain) > 1:
                        self.connect(f"{phase}.duration", f"batt1.integrate_mission_soc_str{j}.soc_integrator_{phase}.duration")
                    else:
                        self.connect(f"{phase}.duration", f"batt1.soc_integrator_str{j}.duration")
            # end 



            #self.add_subsystem("extract_last_fuel", ExtractLast(num_nodes=nn, units = 'kg'), promotes_outputs=[])
            #self.connect(f"{phases[-1]}.fuel_used", "extract_last_fuel.input_vector")
            #self.connect("extract_last_fuel.output_scalar", ["obj_func.fuel_used", "total_fuel_used"])

            npp = ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"]
            vars_list = [
                ('t', 'mission_time', 's', 1),  # Time: 1D vector
                ('p_train_elec', 'mission_p_train_elec', 'kW', npp),  # p_train_elec: 2D matrix with n_motors rows
                ('q_cool_bat', 'mission_q_cool_bat', 'W', n_str),  # q_cool_bat: 1D vector
                ('eta_parc', 'mission_eta_parc', None, 1),  # eta_parc: 1D vector
                ('altitude', 'mission_altitude', 'm', 1),  # altitude: 1D vector
                ('disa', 'mission_disa', 'degC', 1),  # heading: 1D vector
            ]

            last_phase = phases[-1]
            if opt_objective == "range":
                self.connect(f"{last_phase}.ode_integ_phase.range_final", "obj_func.total_range")
            #elif opt_objective == "fuel":
            #    self.connect(f"{last_phase}.ode_integ_phase.fuel_used_final", "obj_func.total_fuel_used")


            
            # Redefine TOW based on fuel used after holding_cruise

            #self.connect("redefine_tow.ac|weights|TOW", "bcast_tow.scalar")
            #self.connect("bcast_tow.vector", "ac|weights|TOW") 

            #self.set_input_defaults("ac|weights|TOW", val=86000 * np.ones(nn), units="lbm")

            #self.connect("redefine_tow.ac|weights|TOW", "ac|weights|TOW")
            
                
            # Setup Battery 
            self.add_subsystem("chain_vars", ChainVars(phases=phases_bat_chain, vars_list=vars_list, nn=nn), promotes_inputs=['*'], promotes_outputs=['*'])
            ncv = nn * len(phases_bat_chain)
            
            # Connect mission_time to establish connection so shape_by_conn works
            #self.add_subsystem("extract_last_time", ExtractLast(num_nodes=ncv, units = 's'), promotes_outputs=[])
            #self.connect("mission_time", "extract_last_time.input_vector")
            

            #if not connect_battery_voltage:
            self.add_subsystem(
                "batt1", EmpiricalBatteryPower(num_nodes=ncv, nm=npp, n_str=n_str, phases=phases_bat_chain, 
                feeder_mode = feeder_mode, battery_datasheet_name=battery_datasheet_name), promotes_inputs=[]
            )

            #self.connect("mission_p_aux_elec", "batt1.p_aux_elec")
            self.connect("mission_p_train_elec", "batt1.p_train_elec")
            self.connect("mission_eta_parc", "batt1.eta_parc")
            self.connect("mission_q_cool_bat", "batt1.q_cool_bat")
            self.connect("mission_altitude", "batt1.altitude")
            self.connect("mission_disa", "batt1.disa")


            self.connect("ac|propulsion|battery|n_series_per_str", "batt1.n_series_per_str")
            # Use computed n_parallel_per_str from WingESSSizingGroup instead of ac_data value
            #self.connect("ac|propulsion|battery|n_parallel_per_str", "batt1.n_parallel_per_str")
            #self.connect("ac|propulsion|battery|n_str", "batt1.n_str")
            #self.connect("ac|propulsion|battery|n_parallel_per_str_computed", "batt1.n_parallel_per_str")
            self.connect("ac|propulsion|battery|n_parallel_per_str", "batt1.n_parallel_per_str")

            self.connect("ac|propulsion|battery|cell_capacity", "batt1.cell_capacity")
            self.connect("ac|propulsion|battery|cp_cell", "batt1.cp_cell")
            self.connect("ac|propulsion|battery|m_cell", "batt1.m_cell")


            self.add_subsystem("extract_last_soc", ExtractLast(num_nodes=nn * len(phases_bat_chain), n_comps=n_str, units = None, mode = 'extract_vector'), promotes_outputs=[])




                

            
            if sum(turb_op_array) != 0 and sum(turb_op_array) != 4:

                e_motor_crz_pts = npp * nn
                self.add_subsystem("flatten_emotor_crz_power",
                    MatrixToVectorConverter(
                        num_nodes=nn,
                        num_comps=npp,
                        input_names=['g_pow'],
                        output_names=['g_pow_flat'],
                        units={'g_pow': 'kW'}
                    ),
                    promotes_outputs=[])
                self.connect("cruise_1.hy_parallel_ptrain.nacelles.unit_mech_power_in_em", "flatten_emotor_crz_power.g_pow")

                self.add_subsystem("ks_emotor_crz_power",
                    om.KSComp(width=int(e_motor_crz_pts), rho=50.0, 
                    add_constraint=True, units='kW', ref = 800.0, lower_flag=True, upper = 0.0),
                    promotes_outputs=[])
                self.connect("flatten_emotor_crz_power.g_pow_flat", "ks_emotor_crz_power.g")
            
                # --- SOC CONSTRAINT: SOC_final >= 0.11 ---
                # Step 1: Create violation vector g_SOC = SOC_min - SOC_final (shape: n_str,)
                #SOC_min = 0.11
                
            if n_str > 1:

                # =====================================================================
                # KS CONSTRAINT AGGREGATION
                # For lower bound constraints (value >= min), we create violation vectors:
                #   g = min - value
                # Then g <= 0 means constraint satisfied (value >= min)
                # Ref: https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/ks_comp.html
                
                # =====================================================================



                
                # --- VOLTAGE CONSTRAINT: V_bat >= 568V ---
                # Step 1: Create violation vector g_V = V_min - V_bat (shape: n_str, ncv)
                ks_out = self.add_subsystem("ks_out", IndepVarComp(), promotes_outputs=["*"])
                ks_out.add_output('V_min', val=self.options["min_voltage"], desc='Voltage violation')
                ks_out.add_output('soc_min', val=self.options["soc_min"], desc='SOC violation')

                
                #V_min = 568.0
                self.add_subsystem("voltage_violation",
                    om.ExecComp(
                        'g_V = (V_min - v_bat)',
                        v_bat={'shape': (n_str, ncv), 'units': 'V'},
                        g_V={'shape': (n_str, ncv), 'units': 'V'},
                        #V_min={'val': V_min, 'units': 'V'}
                    ),
                    promotes_inputs=["V_min"],
                    promotes_outputs=[])
                self.connect("batt1.v_bat", "voltage_violation.v_bat")
                #self.connect("ks_out.V_min", "V_min")
                

                # Step 2: Flatten violation matrix to vector using MatrixToVectorConverter
                total_voltage_points = n_str * ncv
                self.add_subsystem("flatten_voltage",
                    MatrixToVectorConverter(
                        num_nodes=ncv,
                        num_comps=n_str,
                        input_names=['g_V'],
                        output_names=['g_V_flat'],
                        units={'g_V': 'V'}
                    ),
                    promotes_outputs=[])
                self.connect("voltage_violation.g_V", "flatten_voltage.g_V")
                #self.connect("batt1.v_bat", "flatten_voltage.g_V")


                # Step 3: KS aggregation - KS ≈ max(g_V_flat), constraint: KS <= 0
                self.add_subsystem("ks_voltage",
                    om.KSComp(width=int(total_voltage_points), rho=100.0, 
                    add_constraint=True, units='V', ref = 100.0),
                    promotes_outputs=[])
                self.connect("flatten_voltage.g_V_flat", "ks_voltage.g")
                
                
                # --- SOC CONSTRAINT: SOC_final >= SOC_min ---
                # Build violation vector g_soc = SOC_min - SOC_final.
                # Driver constrains this vector directly (g_soc <= 0), so no KS smoothing bias.
                self.add_subsystem(
                    "soc_violation",
                    om.ExecComp(
                        "g_soc = (soc_min - soc_end)",
                        soc_end={"shape": (n_str,)},
                        g_soc={"shape": (n_str,)},
                    ),
                    promotes_inputs=["soc_min"],
                    promotes_outputs=[],
                )
                self.connect("extract_last_soc.output_vector", "soc_violation.soc_end")

                # self.add_subsystem(
                #     "ks_soc",
                #     om.KSComp(width=int(n_str), rho=500.0, add_constraint=True, ref=0.01, units=None),
                #     promotes_outputs=[],
                # )
                # self.connect("soc_violation.g_soc", "ks_soc.g")


            self.connect("extract_last_fuel.output_scalar", "total_fuel_used")

            if opt_objective == "fuel":

                # Compute block fuel = approach fuel - takeoff fuel
                block_fuel_comp = AddSubtractComp(
                    output_name="block_fuel",
                    input_names=["approach_fuel_used_final", "takeoff_fuel_used_final"],
                    vec_size=1,
                    units="lbm",
                    scaling_factors=[1.0, -1.0]  # approach - takeoff
                )

                self.add_subsystem("calc_block_fuel", block_fuel_comp, promotes_outputs=["block_fuel"])
                self.connect("approach.fuel_used_final", "calc_block_fuel.approach_fuel_used_final")
                self.connect("takeoff.fuel_used_final", "calc_block_fuel.takeoff_fuel_used_final")

                self.add_subsystem("obj_func", ObjectiveFuncFuel(soc_min=self.options["soc_min"], min_voltage=self.options["min_voltage"], 
                n_str=n_str, n_cases=1), promotes_inputs=["block_fuel"], promotes_outputs=['mixed_objective'])

            
            elif opt_objective == "range":
                self.add_subsystem("obj_func", ObjectiveFuncRange(soc_min=self.options["soc_min"], min_voltage=self.options["min_voltage"],
                n_str=n_str, n_cases=1), promotes_inputs=["total_fuel_used", "fuel_limit", "TOW", "OEW", "payload"], promotes_outputs=['mixed_objective'])
                self.connect("ac|weights|TOW", "TOW")
                self.connect("ac|weights|payload", "payload")

                
                # Compute mission range = approach range - takeoff range
                mission_range_comp = AddSubtractComp(
                    output_name="mission_range",
                    input_names=["approach_range_final", "takeoff_range_final"],
                    vec_size=1,
                    units="NM",
                    scaling_factors=[1.0, -1.0]  # approach - takeoff
                )

                # Add to model
                self.add_subsystem("calc_mission_range", mission_range_comp, promotes_outputs=["mission_range"])

                # Connect the inputs
                self.connect("approach.ode_integ_phase.range_final", "calc_mission_range.approach_range_final")
                self.connect("takeoff.ode_integ_phase.range_final", "calc_mission_range.takeoff_range_final")


                if not compute_oew:
                    self.connect("ac|weights|OEW", "OEW")


            self.connect(f"batt1.soc", "extract_last_soc.input_matrix")

            if opt_objective in ("fuel", "range"):
                self.connect("extract_last_soc.output_vector", "obj_func.soc_end")
                self.connect(f"batt1.extract_min_voltage.output", ["obj_func.min_voltage"])



        else:

                    # When point_analysis=True and size_tow=True, TOW is provided externally as an input.
        # Use DVLabel passthrough: input 'TOW_dv' -> output 'ac|weights|TOW'
            if size_tow:
                default_TOW = ac_data["ac"]["weights"]["TOW"]["value"]
                TOW_units = ac_data["ac"]["weights"]["TOW"]["units"]
                self.add_subsystem(
                    'tow_passthru',
                    DVLabel([['TOW_dv', 'ac|weights|TOW', default_TOW, TOW_units]]),
                    promotes_inputs=['TOW_dv'],
                    promotes_outputs=['ac|weights|TOW']
                )

            point_phase = mission_config["mission"]["phase_names"][0]

            self.add_subsystem(
                "bcast_weight",
                ScalarToVectorBroadcast(num_nodes=nn, units="lbm"),
                promotes_inputs=[("scalar", "ac|weights|TOW")],
                promotes_outputs=[]
            )
            self.connect("bcast_weight.vector", f"{point_phase}.weight")


            if mission_config[point_phase]["propulsion"]["prop"]["nacelle_power_set"] and not mission_config[point_phase]["propulsion"]["s_curve_power_profile"]:
        #elif "climb_hy" in phases and not mission_config["climb_hy"]["propulsion"]["s_curve_power_profile"]:
                npp = int(round(ac_data["ac"]["propulsion"]["prop"]["num_props"]["value"], 0))
                const_power_profile = mission_config[point_phase]["propulsion"]["const_power_profile"]
                eq_power_nacelles_op = mission_config[point_phase]["propulsion"]["eq_power_nacelles_op"]
                power_units = "kW"
                promotes_inputs_power = []
                
                # Get nacelle operating mask from config (shape: npp x nn)
                # This indicates which nacelles are active at each analysis point
                nac_op_array = mission_config[point_phase]["propulsion"]["nac_op_array"]
                
                if eq_power_nacelles_op and const_power_profile:
                    var_name = f"{point_phase}|total_mech_power_out_scalar"

                    # Case 1: Single scalar value for all nacelles, constant over time
                    # Input: scalar -> Output: matrix (npp, nn)
                    promotes_inputs_power = [("scalar", var_name)]
                    self.add_subsystem(f"{point_phase}_power_comp",
                        ScalarToMatrixBroadcast(
                            num_nodes=nn,
                            num_comps=npp,
                            units=power_units),
                        promotes_inputs=promotes_inputs_power,
                        promotes_outputs=[("matrix", f"{var_name}_matrix")])
                    
                    # Apply nacelle operating mask to the broadcasted power
                    # This zeros out power for inoperative nacelles (e.g., OEI scenarios in WATLIM)
                    masked_var_name = f"{var_name}_masked"
                    self.add_subsystem(f"{point_phase}_power_mask",
                        om.ExecComp(
                            'masked_power = shaft_power * nac_op_mask',
                            shaft_power={'val': np.ones((npp, nn)), 'units': power_units, 'shape': (npp, nn)},
                            nac_op_mask={'val': nac_op_array, 'shape': (npp, nn)},
                            masked_power={'val': np.ones((npp, nn)), 'units': power_units, 'shape': (npp, nn)}),
                        promotes_outputs=[("masked_power", masked_var_name)])
                    
                    # Connect broadcast output to mask component
                    self.connect(f"{var_name}_matrix", f"{point_phase}_power_mask.shaft_power")
                    
                    # Connect masked power to downstream propulsion components
                    self.connect(masked_var_name, 
                            [f"{point_phase}.hy_parallel_ptrain.nacelles.total_mech_power_out", f"{point_phase}.hy_parallel_ptrain.nacelles.gb_comp.power_in"])


            self.add_subsystem(
                "seg",
                UnsteadyFlightPhase(num_nodes=nn, aircraft_model=ParallelHybridAircraft, mission_config=mission_config[point_phase], 
                turb_op_array=turb_op_array,
                point_analysis=True, 
                flight_phase=point_phase,
                drag_model=drag_model,
                velocity_in=mission_config[point_phase]["velocity_in"],
                power_in=mission_config[point_phase]["power_in"],
                true_airspeed_in=mission_config[point_phase]["true_airspeed_in"],
                vs_in=mission_config[point_phase]["vs_in"],
                prop_thrust_set=mission_config[point_phase]["propulsion"]["prop"]["prop_thrust_set"],
                nacelle_power_set=mission_config[point_phase]["propulsion"]["prop"]["nacelle_power_set"]),
                promotes_inputs=["ac|*"]
            )
                #self.connect('seg.altitude_duration.duration', 'seg.ode_integ_phase.duration')

                # Connect Battery Inptus
            #self.connect(f"ac|weights|TOW", f"{point_phase}.weight")
        # end


        for idx, phase in enumerate(phases):
            if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"] and mission_config[phase]["phase_type"] != "power_only":
                self.connect(f"{phase}.hy_parallel_ptrain.total_thrust", f"{phase}.thrust")
            
            if self.options["connect_battery_voltage"]:
                if len(phases) > 1:
                    self.connect(f"batt1.unchain_voltage.phase_{idx}_voltage", f"{phase}.hy_parallel_ptrain.nacelles.voltage")
                else:
                    self.connect(f"batt1.v_mot", f"{phase}.hy_parallel_ptrain.nacelles.voltage")
                #self.connect("batt1.v_bat", f"{phase}.hy_parallel_ptrain.nacelles.voltage")
            #if mission_config[phase]["propulsion"]["prop"]["nacelle_power_set"] and mission_config[phase]["phase_type"] != "power_only":
            #    self.connect(f"{phase}.hy_parallel_ptrain.total_thrust", f"{phase}.thrust")
        # end

        for phase in phases:
            if mission_config[phase]["phase_type"] == "steady_eas":
                if mission_config[phase]["indep_nacelles"]:
                    for j in range(mission_config[phase]["num_nac"]):
                        self.connect(f"{phase}.throttle", f"{phase}.hy_parallel_ptrain.nacelle{j+1}.throttle")
                else:
                    self.connect(f"{phase}.throttle", f"{phase}.hy_parallel_ptrain.nacelles.throttle")


class RedefineTOWComp(om.ExplicitComponent):
    """
    Compute the takeoff weight (TOW) as the sum of OEW, payload, and total fuel used.
    TOW = OEW + payload + total_fuel_used
    """
    def setup(self):
        self.add_input("ac|weights|OEW", shape=(1,), units="kg")
        self.add_input("ac|weights|payload", shape=(1,), units="kg")
        self.add_input("total_fuel_used", shape=(1,), units="kg")
        self.add_output("ac|weights|TOW", shape=(1,), units="kg")

        # Partial derivatives: dTOW/dOEW = 1, dTOW/dpayload = 1, dTOW/dfuel = 1
        self.declare_partials(of="ac|weights|TOW", wrt="ac|weights|OEW", val=1.0)
        self.declare_partials(of="ac|weights|TOW", wrt="ac|weights|payload", val=1.0)
        self.declare_partials(of="ac|weights|TOW", wrt="total_fuel_used", val=1.0)

    def compute(self, inputs, outputs):

        OEW = inputs["ac|weights|OEW"]
        payload = inputs["ac|weights|payload"]
        total_fuel_used = inputs["total_fuel_used"]
        outputs["ac|weights|TOW"] = OEW + payload + total_fuel_used
    
    