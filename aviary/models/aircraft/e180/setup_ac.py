from openmdao.api import Group

from aviary.models.engines.propulsion import ParallelHybridElectricPropulsionSystem
from aviary.utils.math_components.add_subtract_comp import AddSubtractComp
from aviary.utils.math_components.integrals import Integrator
from aviary.utils.math_components.multiply_divide_comp import ElementMultiplyDivideComp
from aviary.models.engines.propulsion.systems.parallel_hybrid import SumAlongAxis



class ParallelHybridAircraft(Group):
    """
    A custom model specific to a series hybrid twin turboprop-class airplane
    This class will be passed in to the mission analysis code.

    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)
        self.options.declare("mission_config", default=None)
        self.options.declare("point_analysis", default=False)


    def setup(self):
        nn = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]
        mission_config = self.options["mission_config"]


        propulsion_promotes_outputs =['p_train_elec']# ["fuel_flow", "thrust"]
        propulsion_promotes_inputs = [
            "fltcond|*",
            "ac|propulsion|*", 
            #"throttle",
            #"propulsor_active",
            #"ac|weights|*",
        ]

        # Unpack the ptrain config, and convert into individual options
        rule = mission_config["propulsion"]["nacelle"]["rule"]
        bias = mission_config["propulsion"]["nacelle"]["bias"]
        prop_rpm_set = mission_config["propulsion"]["prop"]["prop_rpm_set"]
        prop_thrust_set = mission_config["propulsion"]["prop"]["prop_thrust_set"]
        nacelle_power_set = mission_config["propulsion"]["prop"]["nacelle_power_set"]
        
        gt_command = mission_config["propulsion"]["turbine"]["control"]
        motor_command = mission_config["propulsion"]["motor"]["control"]
        #cnvg_throttle = mission_config["propulsion"]["cnvg_throttle"]

        nacelle_command = mission_config["propulsion"]["nacelle"]["command"]
        power_spec = mission_config["propulsion"]["nacelle"]["power_spec"]

        num_props = mission_config["num_props"]
        num_em_per_nac = mission_config["num_em_per_nac"]
        num_turb_per_nac = mission_config["num_turb_per_nac"]
        size_motor = mission_config["size_motor"]
        n_str = mission_config["n_str"]

        gt_idle_allowed = mission_config["propulsion"]["turbine"]["gt_idle_allowed"]
        turb_type = mission_config["propulsion"]["turbine"]["turb_type"]

        self.add_subsystem(
            "hy_parallel_ptrain",
            ParallelHybridElectricPropulsionSystem(num_nodes=nn,
                rule=rule,
                bias=bias,
                prop_rpm_set=prop_rpm_set,
                size_motor=size_motor,
                prop_thrust_set=prop_thrust_set,
                nacelle_power_set=nacelle_power_set,
                gt_command=gt_command,
                motor_command=motor_command,
                nacelle_command=nacelle_command,
                power_spec=power_spec,
                num_props=num_props,
                num_em_per_nac=num_em_per_nac,
                num_turb_per_nac=num_turb_per_nac,
                n_str=n_str,
                gt_idle_allowed=gt_idle_allowed,
                cnvg_throttle=True,
                phase_name=flight_phase,
                turb_type=turb_type,
                ),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )


        # Sum fuel flow from all nacelles using SumAlongAxis
        self.add_subsystem("add_fuel", SumAlongAxis(
            num_nodes=nn,
            num_comps=num_props,  # Number of nacelles
            input_name="nacelles_fuel_flow",
            output_name="total_fuel_flow",
            input_units="kg/h",
            output_units="kg/h",
            input_desc="Fuel flow from all nacelles",
            output_desc="Total fuel flow from all nacelles"
        ), promotes_outputs=["total_fuel_flow"])

        # Connect individual nacelle fuel flows to the matrix input
        # Connect each nacelle's turbine fuel flow to the corresponding row in the matrix
        self.connect("hy_parallel_ptrain.nacelles.turb.fuel_flow", "add_fuel.nacelles_fuel_flow")


        #for j in range(num_em_per_nac):
        #    self.connect(f"hy_parallel_ptrain.motor_rating", f"hy_parallel_ptrain.nacelles.motor_power_to_throttle.rated_power")

        if nacelle_power_set and nacelle_command == 'throttle':
            self.connect(f"hy_parallel_ptrain.nacelles.total_mech_power_out", f"hy_parallel_ptrain.nacelles.gb_comp.power_in")

        intfuel = self.add_subsystem(
            "intfuel",
            Integrator(num_nodes=nn, method="simpson", diff_units="h", time_setup="duration"),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        intfuel.add_integrand("fuel_used", rate_name="total_fuel_flow", val=1.0, units="kg")
        
        # end 

        #self.connect("proprpm", ["propmodel.prop1.rpm", "propmodel.prop2.rpm"])
        #self.connect("hybrid_factor.vec", "propmodel.hybrid_split.power_split_fraction")
        # use a different drag coefficient for takeoff versus cruise
        if not self.options["point_analysis"]:
            self.add_subsystem(
                "weight",
                AddSubtractComp(
                    output_name="weight",
                    input_names=["ac|weights|TOW", "fuel_used"],
                    units="kg",
                    vec_size=[1, nn],
                    scaling_factors=[1, -1],
                    val=86000/2.204 -1,
                    #upper = 86000 / 2.204,
                    #lower = (86000 - 7500) / 2.204
                ),
                promotes_inputs=["*"],
                promotes_outputs=["weight"],
            )

