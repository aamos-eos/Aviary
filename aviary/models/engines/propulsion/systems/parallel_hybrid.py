
#from aviary.models.engines.propulsion import  PowerSplitNacelle
from aviary.utils.dvlabel import DVLabel
import jax
from ..prop import EmpiricalPropellerCoeffMM, EmpiricalPropellerCoeffMMMtip
from ..turbine.turbo_empirical_mm_structured import TurboMission
from ..nacelle_splitter import PowerSplitNacelle
from ..battery.battery_data import BatteryData
from ..motor import RubberMotor
from ..motor import EmpiricalMotor
from ..gearbox.gearbox_efficiency import GearboxEfficiencyGroup
from openmdao.api import Group, IndepVarComp
from aviary.utils.sum_axis import SumAlongAxis
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import jax.numpy as jnp

class GearboxComponent(om.ExplicitComponent):
    """
    Bidirectional gearbox component that can work forward or backward based on nacelle_power_set.

    Forward mode (nacelle_power_set=True): 
        - Input: prop_power_in (npp, nn), carrier_efficiency
        - Output: total_mech_power_out (npp, nn)
        - Calculation: total_mech_power_out = prop_power_in / carrier_efficiency

    Backward mode (nacelle_power_set=False):
        - Input: total_mech_power_out (npp, nn), carrier_efficiency  
        - Output: prop_power_in (npp, nn)
        - Calculation: prop_power_in = total_mech_power_out * carrier_efficiency
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers')
        self.options.declare('nacelle_power_set', default=True, desc='Whether nacelle power is set (forward mode)')

    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        nacelle_power_set = self.options['nacelle_power_set']

        # Common inputs
        self.add_input('carrier_efficiency', shape=(nn,), units=None, desc='Carrier gear efficiency')

        if nacelle_power_set:
            # Forward mode: prop power in -> total mech power out
            self.add_input('prop_power_in', shape=(npp, nn), units='W', desc='Propeller power input')
            self.add_output('total_mech_power_out', shape=(npp, nn), units='W', desc='Total mechanical power output')
        else:

            # Backward mode: total mech power out -> prop power in
            self.add_input('total_mech_power_out', shape=(npp, nn), units='W', desc='Total mechanical power output')
            self.add_output('prop_power_in', shape=(npp, nn), units='W', desc='Propeller power input')


        # Declare partials
        if nacelle_power_set:
            self.declare_partials('total_mech_power_out', ['prop_power_in', 'carrier_efficiency'], method='exact')
        else:

            self.declare_partials('prop_power_in', ['total_mech_power_out', 'carrier_efficiency'], method='exact')


    def compute(self, inputs, outputs):
        carrier_efficiency = inputs['carrier_efficiency']  # Vector of shape (nn,)
        if self.options['nacelle_power_set']:
            # Forward mode: total_mech_power_out = prop_power_in / carrier_efficiency
            outputs['total_mech_power_out'] = inputs['prop_power_in'] / carrier_efficiency
        else:
            # Backward mode: prop_power_in = total_mech_power_out * carrier_efficiency
            outputs['prop_power_in'] = inputs['total_mech_power_out'] * carrier_efficiency

    def compute_partials(self, inputs, partials):
        carrier_efficiency = inputs['carrier_efficiency']
        nn = self.options['num_nodes']
        npp = self.options['num_props']

        if self.options['nacelle_power_set']:
            # Forward mode partials
            partials['total_mech_power_out', 'prop_power_in'] = np.ones(npp * nn) / carrier_efficiency
            partials['total_mech_power_out', 'carrier_efficiency'] = -inputs['prop_power_in'].flatten() / (carrier_efficiency**2)
        else:
            # Backward mode partials
            partials['prop_power_in', 'total_mech_power_out'] = np.ones(npp * nn) * carrier_efficiency
            partials['prop_power_in', 'carrier_efficiency'] = inputs['total_mech_power_out'].flatten()



class JetThrustDeltaComponent(om.ExplicitComponent):
    """
    Generic component that handles jet thrust addition or subtraction based on nacelle_power_set.
    
    When nacelle_power_set=True (power_set mode):
        - Adds jet thrust to propeller thrust to get total thrust output
        - Inputs: prop_thrust_calc, total_jet_thrust (both shape npp, nn)
        - Output: thrust_out_calc = prop_thrust_calc + total_jet_thrust
    
    When nacelle_power_set=False (thrust_set mode):
        - Subtracts jet thrust from total thrust to get propeller thrust required
        - Inputs: thrust, total_jet_thrust (both shape npp, nn)
        - Output: prop_thrust_required = thrust - total_jet_thrust
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('nacelle_power_set', default=False, desc='Whether nacelle power is set (determines operation mode)')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        nacelle_power_set = self.options['nacelle_power_set']
        
        # Common input
        self.add_input('total_jet_thrust', shape=(npp, nn), units='N', desc='Total jet thrust from all turbines per nacelle')
        
        if nacelle_power_set:
            # Power set mode: add jet thrust to propeller thrust
            self.add_input('prop_thrust_calc', shape=(npp, nn), units='N', desc='Propeller thrust calculated per nacelle')
            self.add_output('thrust_out_calc', shape=(npp, nn), units='N', desc='Total thrust output per nacelle')
            
            # Declare partials
            self.declare_partials('thrust_out_calc', ['prop_thrust_calc', 'total_jet_thrust'], method='exact')
        else:
            # Thrust set mode: subtract jet thrust from total thrust
            self.add_input('thrust', shape=(npp, nn), units='N', desc='Total thrust required per nacelle')
            self.add_output('prop_thrust_required', shape=(npp, nn), units='N', desc='Propeller thrust required per nacelle')
            
            # Declare partials
            self.declare_partials('prop_thrust_required', ['thrust', 'total_jet_thrust'], method='exact')
    
    def compute(self, inputs, outputs):
        if self.options['nacelle_power_set']:
            # Power set mode: thrust_out_calc = prop_thrust_calc + total_jet_thrust
            outputs['thrust_out_calc'] = inputs['prop_thrust_calc'] + inputs['total_jet_thrust']
        else:
            # Thrust set mode: prop_thrust_required = thrust - total_jet_thrust
            outputs['prop_thrust_required'] = inputs['thrust'] - inputs['total_jet_thrust']
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        if self.options['nacelle_power_set']:
            # Power set mode partials
            partials['thrust_out_calc', 'prop_thrust_calc'] = np.eye(npp * nn)
            partials['thrust_out_calc', 'total_jet_thrust'] = np.eye(npp * nn)
        else:
            # Thrust set mode partials
            partials['prop_thrust_required', 'thrust'] = np.eye(npp * nn)
            partials['prop_thrust_required', 'total_jet_thrust'] = -np.eye(npp * nn)

class NacellePowerSumComponent(om.JaxExplicitComponent):
    """
    Simple component that sums motor and gas turbine shaft powers for each nacelle.
    Uses matrix inputs and sums along the axis with efficiency scaling.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('num_em_per_nac', default=2, desc='Number of motors per nacelle')
        self.options.declare('num_turb_per_nac', default=1, desc='Number of turbines per nacelle')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        nm = self.options['num_em_per_nac']
        nt = self.options['num_turb_per_nac']
        
        # Matrix inputs
        self.add_input('em_shaft_power_matrix', shape=(npp * nm, nn), units='W', 
                      desc='Motor shaft power matrix (npp * nm, nn)')
        self.add_input('gt_shaft_power_matrix', shape=(npp * nt, nn), units='W', 
                      desc='Gas turbine shaft power matrix (npp * nt, nn)')
        self.add_input('motor_gb_efficiency', shape=(nn,),val= np.ones(nn), units=None, desc='Motor gearbox efficiency')
        
        # Output
        self.add_output('total_mech_power_out', shape=(npp, nn), units='W', desc='Total mechanical power per nacelle')
        
        # Declare partials
        self.declare_partials('total_mech_power_out', ['em_shaft_power_matrix', 'gt_shaft_power_matrix', 'motor_gb_efficiency'], method='exact')
    
    def compute_primal(self, em_shaft_power_matrix, gt_shaft_power_matrix, motor_gb_efficiency):
        # Both matrices are already (npp, nn), so just apply efficiency scaling
        motor_gb_eta = motor_gb_efficiency
        total_mech_power_out = em_shaft_power_matrix * motor_gb_eta + gt_shaft_power_matrix
        return total_mech_power_out
    """"
    def compute_primal(self, em_shaft_power_matrix, gt_shaft_power_matrix, motor_gb_efficiency):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        nm = self.options['num_em_per_nac']
        nt = self.options['num_turb_per_nac']
        
        motor_gb_eta = inputs['motor_gb_efficiency']
        
        # Motor power partials: d(total)/d(motor_matrix) = efficiency
        # Shape: (npp*nn, npp*nm*nn) - each output element depends on corresponding motor elements
        partials['total_mech_power_out', 'em_shaft_power_matrix'] = np.eye(npp * nn) * np.tile(motor_gb_eta, npp)
        
        # GT power partials: d(total)/d(gt_matrix) = 1.0  
        # Shape: (npp*nn, npp*nt*nn) - each output element depends on corresponding GT elements
        partials['total_mech_power_out', 'gt_shaft_power_matrix'] = np.eye(npp * nn)
        
        # Efficiency partials: d(total)/d(efficiency) = motor power matrix
        # Shape: (npp*nn, nn) - each output element depends on efficiency at that time point
        partials['total_mech_power_out', 'motor_gb_efficiency'] = inputs['em_shaft_power_matrix']
    """

 # Create bidirectional throttle/power calculation component
class ThrottlePowerCalculator(om.ExplicitComponent):
    """
    Bidirectional component that can calculate throttle from power or power from throttle.
    
    Forward mode (power_to_throttle=True):
        - Inputs: nacelle_max_rated_power, total_mech_power_out
        - Output: throttle_nac = total_mech_power_out / nacelle_max_rated_power
    
    Backward mode (power_to_throttle=False):
        - Inputs: nacelle_max_rated_power, throttle_nac
        - Output: total_mech_power_out = nacelle_max_rated_power * throttle_nac
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('power_to_throttle', default=False, desc='Direction: True=power->throttle, False=throttle->power')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        power_to_throttle = self.options['power_to_throttle']
        
        # Common input
        self.add_input('nacelle_max_rated_power', shape=(nn,), units='kW', 
                        desc='Maximum rated power per time point')
        
        if power_to_throttle:
            # Forward mode: power -> throttle
            self.add_input('total_mech_power_out', shape=(npp, nn), units='kW', 
                            desc='Total mechanical power output per nacelle per time point')
            self.add_output('throttle_nac', shape=(npp, nn), units=None, 
                            desc='Nacelle throttle per nacelle per time point', lower = 1e-6, upper = 1-1e-6)
            
            # Declare partials
            self.declare_partials('throttle_nac', ['nacelle_max_rated_power', 'total_mech_power_out'], method='exact')
        else:
            # Backward mode: throttle -> power
            self.add_input('throttle_nac', shape=(npp, nn), units=None, 
                            desc='Nacelle throttle per nacelle per time point')
            self.add_output('total_mech_power_out', shape=(npp, nn), units='kW', 
                            desc='Total mechanical power output per nacelle per time point', lower = 1e-6)
            
            # Declare partials
            self.declare_partials('total_mech_power_out', ['nacelle_max_rated_power', 'throttle_nac'], method='exact')
    
    def compute(self, inputs, outputs):
        if self.options['power_to_throttle']:
            # Forward mode: throttle_nac = total_mech_power_out / nacelle_max_rated_power
            outputs['throttle_nac'] = inputs['total_mech_power_out'] / inputs['nacelle_max_rated_power']
        else:
            # Backward mode: total_mech_power_out = nacelle_max_rated_power * throttle_nac
            # Smooth out nacelle throttle 
            throttle_nac =inputs['throttle_nac']
            #throttle_nac = np.minimum(throttle_nac, 1-1e-6)
            #print(f"Throttle nac: {throttle_nac}")
            #print(f"Nacelle max rated power: {inputs['nacelle_max_rated_power']}")


            total_mech_power_out = inputs['nacelle_max_rated_power'] * throttle_nac

            #jax.debug.print("Nacelle max rated power: {}", inputs['nacelle_max_rated_power'])
            

            #print(f"Throttle nac: {throttle_nac}")
            #print(f"Total mech power out: {total_mech_power_out}")
            outputs['total_mech_power_out'] = total_mech_power_out
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        if self.options['power_to_throttle']:
            # Forward mode partials
            # d(throttle)/d(total_mech_power_out) = 1/nacelle_max_rated_power
            partials['throttle_nac', 'total_mech_power_out'] = np.eye(npp * nn) * np.tile(1.0 / inputs['nacelle_max_rated_power'], npp).flatten()
            
            # d(throttle)/d(nacelle_max_rated_power) = -total_mech_power_out/nacelle_max_rated_power^2
            # Shape: (npp*nn, nn) = (20, 5)
            nacelle_power_squared = inputs['nacelle_max_rated_power']**2  # (5,)
            power_out_div_power_sq = inputs['total_mech_power_out'] / nacelle_power_squared  # (4, 5)
            # Create (20, 5) matrix using block diagonal structure
            partial_matrix = np.zeros((npp * nn, nn))
            row_indices = np.arange(npp * nn)
            # Output flattening is row-major: row = i*nn + j, so column index
            # must cycle with j for each nacelle block.
            col_indices = np.tile(np.arange(nn), npp)
            partial_matrix[row_indices, col_indices] = -power_out_div_power_sq.flatten()
            partials['throttle_nac', 'nacelle_max_rated_power'] = partial_matrix
        else:
            # Backward mode partials
            # Apply the same smoothing operations as in compute()
            throttle_nac = inputs['throttle_nac']
            #throttle_nac_smoothed = np.minimum(throttle_nac_smoothed, 1-1e-6)
            #throttle_nac_smoothed = np.array(throttle_nac_smoothed)  # Convert to numpy
            
            # d(total_mech_power_out)/d(throttle_nac) = nacelle_max_rated_power
            # (Note: smoothing derivative is close to 1 in most regions, so this is approximate)
            partials['total_mech_power_out', 'throttle_nac'] = np.eye(npp * nn) * np.tile(inputs['nacelle_max_rated_power'], npp).flatten()
            
            # d(total_mech_power_out)/d(nacelle_max_rated_power) = smoothed_throttle_nac
            # Shape: (npp*nn, nn) = (20, 5)
            partial_matrix = np.zeros((npp * nn, nn))
            row_indices = np.arange(npp * nn)
            # Output flattening is row-major: row = i*nn + j, so column index
            # must cycle with j for each nacelle block.
            col_indices = np.tile(np.arange(nn), npp)
            partial_matrix[row_indices, col_indices] = throttle_nac.flatten()
            partials['total_mech_power_out', 'nacelle_max_rated_power'] = partial_matrix
        

class ParallelHybridNacelle(Group):

    def initialize(self):

        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("num_props", default=4, desc="Number of props per aircraft")
        self.options.declare("cnvg_throttle", default=True, desc="Setting to solve for thrust-> power with throttle convergence or feedback loops")
        self.options.declare("num_em_per_nac", default=2, desc="Number of motors per nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines per nacelle")
        self.options.declare("rule", default="fraction", desc="Rule for power split between turbo and electric motor.\n" +
                             "fraction' means that the power split is a fraction of the total power.\n" +
                             "fixed' means that the power split is derived from a fixed amount of power.")
        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")
        self.options.declare("prop_thrust_set", default=False, desc="Solve for prop thrust from thrust input")
        self.options.declare("nacelle_power_set", default=False, desc="Solve for power from power input")
        self.options.declare("prop_rpm_set", default=False, desc="Solve for torque from RPM input")
        self.options.declare("nacelle_command", default=False, desc="Solve for throttle from throttle input")
        self.options.declare("motor_gb_efficiency", default=1, desc="Efficiency of motor reduction gearbox")
        self.options.declare("turbine_gb_efficiency", default=1, desc="Efficiency of turbine reduction gearbox")
        self.options.declare("motor_prop_gb_efficiency", default=1, desc="Efficiency of motor to propeller planetary gearbox")
        self.options.declare("turbine_prop_gb_efficiency", default=1, desc="Efficiency of turbine to propeller planetary gearbox")
        self.options.declare("gt_command", default=False, desc="gas turbine takes throttle as input. Otherwise, power is input")
        self.options.declare("motor_command", default='power', desc="motor takes torque and rpm as input. Otherwise, power is input")
        self.options.declare("power_spec", default='relative', desc="Gas Turbine / Motor Hybrdization Ratio Known")
        self.options.declare("gt_idle_allowed", default=True, desc="allow idle mode (and thus non zero fuel consumption) at 0 power")
        self.options.declare("size_motor", default=False, desc="Whether to use a rubber motor for sizing or not")
        self.options.declare("turb_type", default='PT6', desc="Turbine type: PT6 or ACCE")



    def setup(self):
        nn = self.options["num_nodes"]
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        npp = self.options["num_props"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        prop_thrust_set = self.options["prop_thrust_set"]
        nacelle_power_set = self.options["nacelle_power_set"]
        prop_rpm_set = self.options["prop_rpm_set"]
        nacelle_command = self.options["nacelle_command"]
        motor_gb_eta = self.options["motor_gb_efficiency"]
        turbine_gb_eta = self.options["turbine_gb_efficiency"]
        motor_prop_gb_eta = self.options["motor_prop_gb_efficiency"]
        turbine_prop_gb_eta = self.options["turbine_prop_gb_efficiency"]
        size_motor = self.options["size_motor"]
        power_spec = self.options["power_spec"]
        gt_idle_allowed = self.options["gt_idle_allowed"]
        gt_command = self.options["gt_command"]
        motor_command = self.options["motor_command"]
        turb_type = self.options["turb_type"]

        dvlist = [
            ["fltcond|T", "T", 15* np.ones(nn), "degC"],
            ["fltcond|h", "h", 0* np.ones(nn), "m"],
            ["fltcond|disa", "disa", 0* np.ones(nn), "degC"],
            ["fltcond|M", "M", 0.2* np.ones(nn), None],
            ["fltcond|rho", "rho", 1.225 * np.ones(nn), "kg/m**3"],
            ["fltcond|Utrue", "Utrue", 90* np.ones(nn), "m/s"],
            ["fltcond|a", "a", 300* np.ones(nn), "m/s"],
        ]

        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        nac_calc_throttle = ThrottlePowerCalculator(num_nodes=nn, num_props=npp, power_to_throttle=True)

        if prop_thrust_set:
            # Thrust is set: subtract jet thrust from required thrust to get propeller thrust
            self.add_subsystem("subtract_jet_thrust", JetThrustDeltaComponent(num_nodes=nn, num_props=npp, nacelle_power_set=False), 
                              promotes_inputs=["total_jet_thrust"], promotes_outputs=[])
            self.connect("subtract_jet_thrust.prop_thrust_required", "prop.windmilling_drag.thrust_in")

            self.add_subsystem("prop", EmpiricalPropellerCoeffMMMtip(num_nodes=nn,
                                                        num_props=npp,
                                                        thrust_set = prop_thrust_set,
                                                        power_set = nacelle_power_set,
                                                        rpm_set = prop_rpm_set),
                                                        promotes_inputs=['rho','Utrue','a']
                                                        )

            # Add gearbox efficiency group (power_set mode for prop_thrust_set case)
            self.add_subsystem("gb_comp", GearboxEfficiencyGroup(num_nodes=nn,num_props=npp, mode='thrust_set'), promotes_inputs=['throttle_nac'], promotes_outputs=[])
            self.add_subsystem("nac_calc_throttle", nac_calc_throttle, promotes_inputs=["*"], promotes_outputs=["*"])

            # Connect prop power to gearbox power calculator
            self.connect("prop.power_calc", "gb_comp.power_out")
            self.connect("gb_comp.power_in", "total_mech_power_out")

        if nacelle_command=='throttle' and nacelle_power_set and power_spec=='relative':
            # Use bidirectional component in throttle->power mode
            self.connect("total_mech_power_out","gb_comp.power_in")
            nacelle_throttle_to_power = ThrottlePowerCalculator(num_nodes=nn, num_props=npp, power_to_throttle=False)
            self.add_subsystem("nacelle_throttle_to_power", nacelle_throttle_to_power, promotes_inputs=["*"], promotes_outputs=["*"])
        # define design variables that are independent of flight condition or control states
        #if power_spec == 'relative':
        self.add_subsystem("hybrid_split", PowerSplitNacelle(rule=rule, num_nodes=nn, num_props=npp, bias=bias, num_em_per_nac=nm, num_turb_per_nac=nt, power_spec=power_spec, motor_command=motor_command, size_motor=size_motor, turb_type=turb_type), promotes_inputs=["*"], promotes_outputs=["*"])

        if gt_command=='throttle' and power_spec=='independent':
            gt_throttle_set = True
        else:
            gt_throttle_set=False
        self.connect("h", "altitude")
        self.connect("M", "mach")

        self.add_subsystem(f"turb", TurboMission(num_nodes=nn, num_turbs = npp, throttle_set=gt_throttle_set, idle_allowed=gt_idle_allowed, turb_type=turb_type), promotes_inputs=[ "mach", "disa"], promotes_outputs=[])
        self.connect(f'lim_min_alt.output', [ f'turb.altitude'])
        if not gt_throttle_set:
            self.connect(f'turb_compute_max_power.max_power', [f'turb.max_power'])
        # end
        
        for i in range(nm):
            if motor_command == 'torque_rpm':
                torque_rpm_set = True
            elif motor_command == 'power':
                torque_rpm_set = False
            if size_motor:
                self.add_subsystem(f"motor", RubberMotor(num_nodes=nn, num_motors=npp * nt), promotes_inputs=["voltage"], promotes_outputs=["p_train_elec"])
            else:
                self.add_subsystem(f"motor", EmpiricalMotor(num_nodes=nn, num_motors=npp * nt), promotes_inputs=["voltage"], promotes_outputs=["p_train_elec"])
        # end

        self.connect("turb.jet_thrust","total_jet_thrust")

        if power_spec=='independent':





            
            self.add_subsystem("add_shaft_powers", NacellePowerSumComponent(
                num_nodes=nn,
                num_props=npp,
                num_em_per_nac=nm,
                num_turb_per_nac=nt
            ), promotes_inputs=["motor_gb_efficiency", "gt_shaft_power_matrix"],
              promotes_outputs=["total_mech_power_out"])
            
            # Connect motor power to add_shaft_powers.em_shaft_power_matrix based on power_spec/motor_command
            # This must be done inside ParallelHybridNacelle to avoid "same system" error
            if power_spec == 'relative':
                self.connect("unit_mech_power_in_em", "add_shaft_powers.em_shaft_power_matrix")
            elif power_spec == 'independent' and motor_command == 'throttle':
                self.connect("unit_mech_power_calc_em", "add_shaft_powers.em_shaft_power_matrix")
            # For power_spec == 'independent' with motor_command == 'power':
            # em_shaft_power_matrix is set externally via IVC, so promote it as input
            elif power_spec == 'independent' and motor_command == 'power':
                self.promotes('add_shaft_powers', inputs=['em_shaft_power_matrix'])

            if gt_command=='throttle':
                self.connect("turb.power", "gt_shaft_power_matrix")
            self.connect("total_mech_power_out", "gb_comp.power_in")


        if nacelle_power_set:
            if nacelle_command!='throttle':
                self.add_subsystem("nac_calc_throttle", nac_calc_throttle, promotes_inputs=["*"], promotes_outputs=["*"])
            # Add gearbox efficiency group (thrust_set mode for nacelle_power_set case)
            self.add_subsystem("gb_comp", GearboxEfficiencyGroup(num_nodes=nn, mode='power_set'), promotes_inputs=['throttle_nac'], promotes_outputs=[])
            self.add_subsystem("prop", EmpiricalPropellerCoeffMMMtip(num_nodes=nn,
                                                        num_props=npp,
                                                        thrust_set = prop_thrust_set,
                                                        power_set = nacelle_power_set,
                                                        rpm_set = prop_rpm_set),
                                                        promotes_inputs=['rho','Utrue','a']

                                                        )

            #self.connect("nac_calc_throttle.throttle_nac", "gb_comp.throttle_nac")
            if prop_rpm_set:
                self.connect("gb_comp.power_out", ["prop.power"])
            else:
                self.connect("gb_comp.power_out", ["prop.power","prop.matrix_to_vector_rpm_power.power"])


            self.add_subsystem("add_jet_thrust", JetThrustDeltaComponent(num_nodes=nn, num_props=npp, nacelle_power_set=True), 
                              promotes_inputs=["total_jet_thrust"], promotes_outputs=["thrust_out_calc"])
            self.connect("prop.thrust_calc", "add_jet_thrust.prop_thrust_calc")
        
        if motor_command == 'throttle' and power_spec == 'relative':
            self.connect("motor_power_to_throttle.throttle", "motor.throttle")

import numpy as np
import openmdao.api as om

def smooth_zeroed_abs(x, mu):
    """
    Smooth mapping with f(0)=0 and f(x) ~ x for |x| >> mu.
    f(x,mu) = x * tanh(x/mu)
    Vectorized: x can be scalar or numpy array. mu is scalar > 0.
    """
    x = np.asarray(x)
    return x * np.tanh(x / mu)


def d_smooth_zeroed_abs_dx(x, mu):
    """
    df/dx = tanh(x/mu) + x * (1/mu) * sech^2(x/mu)
    where sech^2(u) = 1 / cosh(u)^2
    """
    x = np.asarray(x)
    u = x / mu
    t = np.tanh(u)
    sech2 = 1.0 / (np.cosh(u) ** 2)
    return t + x * (1.0 / mu) * sech2


def d_smooth_zeroed_abs_dmu(x, mu):
    """
    df/dmu = x * d/dmu[tanh(x/mu)]
           = x * sech^2(x/mu) * d/dmu(x/mu)
           = x * sech^2(u) * (-x / mu^2)
           = - x^2 / mu^2 * sech^2(x/mu)
    (not used in this component because mu is an option, not an input)
    """
    x = np.asarray(x)
    u = x / mu
    sech2 = 1.0 / (np.cosh(u) ** 2)
    return - (x * x) / (mu * mu) * sech2


class SmoothThrustBalanceImplicit(om.ImplicitComponent):
    """
    Implicit component enforcing:
        R = f(thrust_calculated - thrust_required, mu) = 0
    Unknown / output solved for by the group: throttle_nac  (kept as the component output)
    Inputs (known in this component): thrust_required, thrust_calculated
    Notes:
      - The residual depends on inputs only; throttle_nac is the unknown the solver
        will vary elsewhere in the group (thrust_calculated should depend on throttle_nac
        through other components).
      - Locally, dR/d(throttle_nac) = 0; OpenMDAO assembles the global Jacobian to include
        the chain rule from components that compute thrust_calculated from throttle_nac.
    """

    def initialize(self):
        self.options.declare("mu", default=1e-3, types=float,
                             desc="Smoothing parameter for x*tanh(x/mu). Must be > 0.")
        # allow vectorization: user can supply shape via add_input later if desired
        self.options.declare("num_nodes", default=1, desc='Number of analysis points')
        self.options.declare("num_props", default=4, desc='Number of props/nacelles')
        self.options.declare("phase_name", default='cruise', desc='Phase type: cruise, approach, climb, descent, loiter')
        self.options.declare("mu", default=1e-3, desc='Smoothing parameter for smooth_abs')

    def setup(self):

        nn = self.options['num_nodes']
        npp = self.options['num_props']

        phase_name = self.options['phase_name']


        if "cruise_1" in phase_name:
            throttle_initial_guess = 0.4
        elif "circuit" in phase_name or "approach" in phase_name:
            throttle_initial_guess = 0.2
        else:
            throttle_initial_guess = 0.5


        shape = (npp, nn)
        self.add_input("thrust_required", shape=(npp, nn), units="N")
        self.add_input("thrust_calculated", shape=(npp, nn), units="N")
        self.add_output("throttle_nac", val=throttle_initial_guess, shape=(npp, nn), units=None, lower = 0, upper = 1)
        # declare partials with matching shapes (diagonal)
        rows = np.arange(npp * nn)
        cols = np.arange(npp * nn)
        
        self.declare_partials(of="throttle_nac", wrt="thrust_calculated",
                                rows=rows, cols=cols)
        self.declare_partials(of="throttle_nac", wrt="thrust_required",
                                rows=rows, cols=cols)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Residual: R = f(x, mu) where x = thrust_calculated - thrust_required
        and f(x,mu) = x * tanh(x/mu).  This is zero iff x==0.
        """
        mu = self.options["mu"]
        x = inputs["thrust_calculated"] - inputs["thrust_required"]
        residuals["throttle_nac"] = smooth_zeroed_abs(x, mu)

    def linearize(self, inputs, outputs, partials):
        """
        Local Jacobian:
          R = f(x,mu) with x = thrust_calculated - thrust_required

        dR/d(thrust_calculated) = df/dx * (dx/dthrust_calculated) = df/dx * 1
        dR/d(thrust_required)   = df/dx * (dx/dthrust_required)   = df/dx * (-1)
        dR/d(throttle_nac)      = 0 (no explicit dependence)

        df/dx provided by d_smooth_zeroed_abs_dx
        """
        mu = self.options["mu"]
        x = inputs["thrust_calculated"] - inputs["thrust_required"]
        dfdx = d_smooth_zeroed_abs_dx(x, mu)

        partials["throttle_nac", "thrust_calculated"] = dfdx.flatten()
        partials["throttle_nac", "thrust_required"] = -dfdx.flatten()


class DetermineNominalThrottle(om.ImplicitComponent):
    """
    Implicit component that determines a SINGLE nominal throttle value to match
    total thrust required across all nacelles.
    
    This solves: sum(thrust_calculated) - total_thrust_required = 0
    by varying nominal_throttle.
    
    The nominal_throttle is then multiplied by thrust_share to get per-nacelle throttle.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
        self.options.declare('phase_name', default='cruise', desc='Phase name for initial guess')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        phase_name = self.options['phase_name']

        # Initial guess based on phase
        if "cruise" in phase_name:
            throttle_initial_guess = 0.4
        elif "circuit" in phase_name or "approach" in phase_name:
            throttle_initial_guess = 0.2
        else:
            throttle_initial_guess = 0.5
        
        # Inputs
        self.add_input('total_thrust_required', shape=(nn,), units='N', 
                      desc='Total thrust required from all nacelles')
        self.add_input('total_thrust_calculated', shape=(nn,), units='N',
                      desc='Sum of thrust from all nacelles')
        
        # Output: single nominal throttle per time point
        self.add_output('nominal_throttle', shape=(nn,), units=None, val=throttle_initial_guess,
                       desc='Nominal throttle value (same for all active nacelles)', 
                       lower=1e-6, upper=1-1e-6)
        
        # Declare partials (diagonal since each time point is independent)
        self.declare_partials('nominal_throttle', 'total_thrust_required', 
                             rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('nominal_throttle', 'total_thrust_calculated', 
                             rows=np.arange(nn), cols=np.arange(nn))

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Residual: total_thrust_calculated - total_thrust_required = 0"""
        residuals['nominal_throttle'] = (
            inputs['total_thrust_calculated'] - inputs['total_thrust_required']
        ) / 1000.0  # Scale for numerical stability

    def linearize(self, inputs, outputs, partials):
        nn = self.options['num_nodes']
        # d(residual)/d(total_thrust_calculated) = 1/1000
        partials['nominal_throttle', 'total_thrust_calculated'] = np.ones(nn) / 1000.0
        # d(residual)/d(total_thrust_required) = -1/1000
        partials['nominal_throttle', 'total_thrust_required'] = -np.ones(nn) / 1000.0


class ApplyThrustShare(om.ExplicitComponent):
    """
    Broadcasts nominal throttle to per-nacelle throttle using thrust_share.
    
    throttle_nac[i, j] = nominal_throttle[j] * thrust_share[i, j]
    
    thrust_share is normally 1.0 for all nacelles, but can be 0 for disabled nacelles
    or varied for asymmetric thrust distribution.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/nacelles')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # Inputs
        self.add_input('nominal_throttle', shape=(nn,), units=None,
                      desc='Nominal throttle value')
        self.add_input('thrust_share', shape=(npp, nn), units=None, val=1.0,
                      desc='Thrust share per nacelle (1.0 = full, 0 = disabled)')
        
        # Output
        self.add_output('throttle_nac', shape=(npp, nn), units=None, val=0.5,
                       desc='Per-nacelle throttle', lower=0.0, upper=1.0)
        
        # Declare partials
        # d(throttle_nac[i,j])/d(nominal_throttle[j]) = thrust_share[i,j]
        rows = np.arange(npp * nn)
        cols_nom = np.tile(np.arange(nn), npp)
        self.declare_partials('throttle_nac', 'nominal_throttle', rows=rows, cols=cols_nom)
        
        # d(throttle_nac[i,j])/d(thrust_share[i,j]) = nominal_throttle[j] (diagonal)
        self.declare_partials('throttle_nac', 'thrust_share', rows=rows, cols=rows)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # Broadcast: throttle_nac[i,j] = nominal_throttle[j] * thrust_share[i,j]
        outputs['throttle_nac'] = inputs['nominal_throttle'] * inputs['thrust_share']

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # d(throttle_nac)/d(nominal_throttle) = thrust_share (flattened, repeated for each nacelle)
        partials['throttle_nac', 'nominal_throttle'] = inputs['thrust_share'].flatten()
        
        # d(throttle_nac)/d(thrust_share) = nominal_throttle (tiled for each nacelle)
        partials['throttle_nac', 'thrust_share'] = np.tile(inputs['nominal_throttle'], npp)


class ParallelHybridElectricPropulsionSystem(Group):

    """

    This is an example model of a parallel-hybrid propulsion system.
    """



    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("num_props", default=4, desc="Number of props")
        self.options.declare("num_em_per_nac", default=2, desc="Number of motors per nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines per nacelle")
        self.options.declare("num_nac", default=1, desc="Number of nacelles")

        self.options.declare("rule", default="fraction", desc="Rule for power split between turbo and electric motor.\n" +

                             "fraction' means that the power split is a fraction of the total power.\n" +

                             "fixed' means that the power split is derived from a fixed amount of power.")

        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")
        self.options.declare("phase_name", default="cruise_1", desc="Phase type: cruise, approach, climb, descent, loiter")
        self.options.declare("nacelle_power_set", default=False, desc="propeller power in (nacelle power out) is specified")
        self.options.declare("prop_rpm_set", default=False, desc="propeller rpm is an specified")
        self.options.declare("prop_thrust_set", default=False, desc="propelelr thrust is specified")
        self.options.declare("cnvg_throttle", default=True, desc="Setting to solve for thrust-> power with throttle convergence or feedback loops")
        self.options.declare("gt_command", default=False, desc="Command type: `throttle` or `power`")
        self.options.declare("motor_command", default=False, desc="Command type: `torque_rpm` or `power`")
        self.options.declare("nacelle_command", default=False, desc="Command type: `throttle` or `power`")
        self.options.declare("power_spec", default='relative', desc="hybridization ratio specified")
        self.options.declare("gt_idle_allowed", default=True, desc="allow idle mode (and thus non zero fuel consumption) at 0 power")
        self.options.declare("n_str", default=4, desc="Number of battery strings in aircraft")
        self.options.declare("size_motor", default=False, desc="Whether to use a rubber motor for sizing or not")
        self.options.declare("turb_type", default='PT6', desc="Turbine type: PT6 or ACCE")

    def setup(self):

        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        nacelle_command = self.options["nacelle_command"]
        nacelle_power_set = self.options["nacelle_power_set"]
        prop_rpm_set = self.options["prop_rpm_set"]
        prop_thrust_set = self.options["prop_thrust_set"]
        gt_command = self.options["gt_command"]
        motor_command = self.options["motor_command"]
        phase_name = self.options["phase_name"]
        n_str = self.options["n_str"]
        power_spec =self.options["power_spec"]
        gt_idle_allowed = self.options["gt_idle_allowed"]
        cnvg_throttle = self.options["cnvg_throttle"]
        size_motor = self.options["size_motor"]
        turb_type = self.options["turb_type"]

        dvlist = [
            ["ac|propulsion|propeller|diameter", "prop_diameter", 13*0.3048 * np.ones((npp, nn)), "m"],
            ["ac|propulsion|motor|rating", "rated_power_em", 1000.0 , "kW"],
            
            ["ac|propulsion|battery|q_cool_bat", "q_cool_bat", 25 * np.ones((n_str, nn)), "kW"],
        ]

        if size_motor:
            self.connect("rated_power_em", "nacelles.rated_power_em")



        self.add_subsystem("dvs", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])


        if cnvg_throttle:
            prop_thrust_set_nac = False
            nacelle_power_set_nac = True

            if prop_thrust_set:
                # NEW ARCHITECTURE: 
                # 1. Solve for nominal_throttle to match TOTAL thrust
                # 2. Apply thrust_share to get per-nacelle throttle
                # 3. Sum nacelle thrusts to get total_thrust_calculated
                
                # Implicit solver for nominal throttle (single value per time point)
                self.add_subsystem("throttle_balance", 
                    DetermineNominalThrottle(num_nodes=nn, num_props=npp, phase_name=phase_name), 
                    promotes_inputs=['total_thrust_required'], 
                    promotes_outputs=['nominal_throttle'])
                
                # Apply thrust share to get per-nacelle throttle
                self.add_subsystem("apply_thrust_share",
                    ApplyThrustShare(num_nodes=nn, num_props=npp),
                    promotes_inputs=['nominal_throttle', 'thrust_share'],
                    promotes_outputs=['throttle_nac'])
                
                # Sum thrust across nacelles to get total
                #self.add_subsystem("add_thrust",
                #     SumAlongAxis(num_nodes=nn, num_comps=npp, 
                #             input_name='thrust_per_nacelle', 
                #             output_name='total_thrust_calculated',
                #             input_units='N', output_units='N'),
                #     promotes_outputs=['total_thrust_calculated'])
                
                # Connect throttle to nacelles
                self.connect("throttle_nac", "nacelles.throttle_nac")
 
                nacelle_command = 'throttle'
        else:
            nacelle_power_set_nac = nacelle_power_set
            prop_thrust_set_nac = prop_thrust_set
            #prop_rpm_set_nac = prop_rpm_set



        # introduce model components
        self.add_subsystem(f"nacelles", ParallelHybridNacelle(num_nodes=nn,
                                                                                num_em_per_nac=nm,
                                                                                size_motor=size_motor,
                                                                                rule=rule,
                                                                                bias=bias,
                                                                                nacelle_command=nacelle_command,
                                                                                nacelle_power_set=nacelle_power_set_nac,
                                                                                prop_rpm_set=prop_rpm_set,
                                                                                prop_thrust_set=prop_thrust_set_nac,
                                                                                gt_command=gt_command,
                                                                                motor_command=motor_command,
                                                                                power_spec=power_spec,
                                                                                gt_idle_allowed=gt_idle_allowed,
                                                                                cnvg_throttle=cnvg_throttle,
                                                                                turb_type=turb_type),
                                                                                promotes_inputs=["fltcond|*", "nacelle_max_rated_power"],
                                                                                promotes_outputs=["p_train_elec"])

        self.add_subsystem("add_thrust", SumAlongAxis(
                        num_nodes=nn, 
                        num_comps=npp,
                        input_name="thrust_out_calc",
                        output_name="total_thrust",
                        input_units="N",
                        output_units="N",
                        input_desc="Thrust from all propellers/nacelles",
                        output_desc="Total thrust summed over all propellers",
                    ), promotes_inputs=[], promotes_outputs=["total_thrust"])

        if prop_thrust_set and cnvg_throttle:
            # Connect nacelle thrust output to sum component
            #self.connect("nacelles.thrust_out_calc", "add_thrust.thrust_per_nacelle")
            # Connect summed thrust back to implicit solver
            self.connect("total_thrust", "throttle_balance.total_thrust_calculated")


        #input_prop_thrust_names_str = []
        self.connect(f"nacelles.thrust_out_calc", f"add_thrust.thrust_out_calc")
        self.set_input_defaults(f"nacelles.voltage", 800 * np.ones((npp, nn)), units = 'V')
        #self.connect(f"motor_rating", f"nacelles.motor_power_to_throttle.rated_power")
        # Connect battery voltage to motor votlage
        #self.connect("batt1.v_mot", [f"nacelles.motor.voltage", f"nacelles.voltage"])
        # Assign unit mechanical power to each motor in nacelle if needed
        # Define motor command
        if power_spec == 'relative':
            # When power spec is relative, we never set the throttle of the EM. 
            self.connect(f"nacelles.unit_mech_power_in_em", f"nacelles.motor.mech_power")
            #self.connect(f"nacelles.unit_mech_power_in_em", f"nacelles.em_shaft_power_matrix")
        elif power_spec == 'independent' and motor_command == 'throttle':
            # When motor_command is throttle, unit_mech_power_calc_em is computed from throttle * max_power
            # Connect to motor (em_shaft_power_matrix connection is made inside ParallelHybridNacelle)
            self.connect(f"nacelles.unit_mech_power_calc_em", f"nacelles.motor.mech_power")
        # Otherwise, when power spec is independent with power command, component settings are defined outside the group by IVC
        # and connect directly to motor.mech_power + em_shaft_power_matrix (see prepare_mission_inputs.py)
        
        # Connect individual motor electrical powers to matrix input for SumAlongAxis
        # end
        #input_prop_thrust_names_str.append(f"nacelles.prop.thrust_calc")
        if power_spec == 'relative':
            # When power spec is relative, we never set the throttle of the GT. A
            for i in range (nt):
                self.connect(f"nacelles.unit_mech_power_in_gt", f"nacelles.turb.power")
            # self.connect(f"nacelles.compute_max_power_gt.max_power", f"nacelles.max_power_gt")
            # end
        # Otherwise, when power spec is independent, component settings are defined outside the group, by the IVC
        self.connect("prop_diameter", f"nacelles.prop.diameter")
        # Otherwise, when power spec is independent, component settings are defined outside the group, by the IVC

        # end
        # Set up the linear solver
        #self.linear_solver = om.DirectSolver()
#!/usr/bin/env python3

def plot_hybrid_propulsion_results(prob, npp=4, nm=2, nt=1, dt=30, prop_thrust_set=False, nacelle_power_set=False):
    """
    Plot comprehensive results for all components in the hybrid propulsion system.
    Uses matrix format where each nacelle is represented as a row in the matrix.

    Parameters
    ----------
    prob : om.Problem
        OpenMDAO problem after running the model
    npp : int
        Number of propellers/nacelles
    nm : int
        Number of motors per nacelle
    nt : int
        Number of turbines per nacelle
    dt : float
        Time step in seconds
    prop_thrust_set : bool
        Whether propeller thrust is set
    nacelle_power_set : bool
        Whether nacelle power is set
    """

    # Get number of nodes
    num_nodes = len(prob.get_val('fltcond|Utrue'))
    dt_ = dt * np.ones(num_nodes)
    time_min = np.cumsum(dt_) / 60

    # Create separate plots for each component type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nacelle Performance', fontsize=16, fontweight='bold')

    # Plot 1: Motors - Power and Efficiency
    for j in range(nm):
        # Get motor data as matrix (npp, nn) and plot each row (nacelle)
        motor_power_matrix = prob.get_val(f'nacelles.motor.mech_power', units='W') / 1000  # Convert to kW
        motor_eff_matrix = prob.get_val(f'nacelles.motor.eff') * 100  # Convert to percentage

        for i in range(npp):
            axes[0, 0].plot(time_min, motor_power_matrix[i, :], marker='o', linewidth=2, 
                           label=f'Nacelle {i+1} Motor {j+1} Power')
            axes[0, 1].plot(time_min, motor_eff_matrix[i, :], marker='s', linewidth=2, 
                           label=f'Nacelle {i+1} Motor {j+1} Efficiency')

    axes[0, 0].set_title('Motor Electrical Power')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Motor Efficiency')
    axes[0, 1].set_ylabel('Efficiency (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 2: Turboshafts - Fuel Flow and Power
    fuel_flow_matrix = prob.get_val(f'nacelles.turb.fuel_flow')
    power_matrix = prob.get_val(f'nacelles.turb.power', units='kW')  # Convert to kW
        
    for i in range(npp):
        axes[1, 0].plot(time_min, fuel_flow_matrix[i, :], marker='o', linewidth=2, 
                    label=f'Nacelle {i+1} Turbine Fuel Flow')
        axes[1, 1].plot(time_min, power_matrix[i, :], marker='s', linewidth=2, 
                    label=f'Nacelle {i+1} Turbine Power')
    axes[1, 0].set_title('Turboshaft Fuel Flow')
    axes[1, 0].set_ylabel('Fuel Flow (kg/h)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Turboshaft Power')
    axes[1, 1].set_ylabel('Power (kW)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Set x-axis labels
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel('Time (min)')

    plt.tight_layout()
    plt.show()

    # Create second figure for propellers
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 12))
    fig2.suptitle('Propeller Performance', fontsize=16, fontweight='bold')

    # Plot 3: Propellers - Thrust, Efficiency, and Power
    # Get propeller data as matrix (npp, nn) and plot each row (nacelle)
    if prop_thrust_set:
        try:
            thrust_matrix = prob.get_val(f'nacelles.prop.windmilling_drag.thrust_calc')
        except:
            thrust_matrix = prob.get_val(f'nacelles.thrust_out_calc')
    else:
        thrust_matrix = prob.get_val(f'nacelles.thrust_out_calc')
    efficiency_matrix = prob.get_val(f'nacelles.prop.eta') * 100


    if nacelle_power_set:
        power_matrix = prob.get_val(f'nacelles.prop.power', units='kW')
    else:
        try:
            power_matrix = prob.get_val(f'nacelles.prop.power', units='kW')
        except:
            power_matrix = prob.get_val(f'nacelles.gb_comp.power_out', units='kW')


    total_thrust = np.zeros(num_nodes)
    for i in range(npp):
        axes2[0].plot(time_min, thrust_matrix[i, :]/1000, marker='o', linewidth=2, 
                     label=f'Nacelle {i+1} Propeller Thrust')
        axes2[1].plot(time_min, efficiency_matrix[i, :], marker='s', linewidth=2, 
                     label=f'Nacelle {i+1} Propeller Efficiency')
        total_thrust += thrust_matrix[i, :]

    axes2[0].set_title('Propeller Thrust')
    axes2[0].set_ylabel('Thrust (kN)')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_title('Propeller Efficiency')
    axes2[1].set_ylabel('Efficiency (%)')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    # Set x-axis labels
    for i in range(2):
        axes2[i].set_xlabel('Time (min)')

    plt.tight_layout()
    plt.show()
    

    # Create third figure for system summary
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle('System Summary', fontsize=16, fontweight='bold')    

    # Total system thrust
    axes3[0, 0].plot(time_min, total_thrust, marker='o', linewidth=2, color='red', label='Total Thrust')
    axes3[0, 0].set_title('Total System Thrust')
    axes3[0, 0].set_ylabel('Thrust (N)')
    axes3[0, 0].legend()
    axes3[0, 0].grid(True, alpha=0.3)

    # Total electrical power
    total_elec_power = np.sum(prob.get_val('p_train_elec') / 1000, axis=0)
    axes3[0, 1].plot(time_min, total_elec_power, marker='o', linewidth=2, color='blue', label='Total Electrical Power')
    axes3[0, 1].set_title('Total Electrical Power')
    axes3[0, 1].set_ylabel('Power (kW)')
    axes3[0, 1].legend()
    axes3[0, 1].grid(True, alpha=0.3)

    # Total fuel flow
    total_fuel_flow = np.zeros(num_nodes)
    for k in range(nt):
        fuel_flow_matrix = prob.get_val(f'nacelles.turb.fuel_flow')
        total_fuel_flow += np.sum(fuel_flow_matrix, axis=0)  # Sum across all nacelles
    axes3[1, 0].plot(time_min, total_fuel_flow, marker='o', linewidth=2, color='orange', label='Total Fuel Flow')
    axes3[1, 0].set_title('Total Fuel Flow')
    axes3[1, 0].set_ylabel('Fuel Flow (kg/h)')
    axes3[1, 0].legend()
    axes3[1, 0].grid(True, alpha=0.3)

    # Set x-axis labels
    for i in range(2):
        for j in range(2):
            axes3[i, j].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.show()

    # Create fourth figure for power breakdown by nacelle
    fig4, axes4 = plt.subplots(2, 2, figsize=(15, 15))
    fig4.suptitle('Power Breakdown by Nacelle', fontsize=16, fontweight='bold')

    # Define color schemes for each nacelle
    nacelle_colors = {
        1: {'gt': '#d62728', 'motor': '#ff6b6b', 'generating': '#ffb3b3'},      # Red scheme
        2: {'gt': '#1f77b4', 'motor': '#74c0fc', 'generating': '#b3d9ff'},      # Blue scheme
        3: {'gt': '#2ca02c', 'motor': '#51cf66', 'generating': '#b3f0b3'},      # Green scheme
        4: {'gt': '#ff7f0e', 'motor': '#ffa726', 'generating': '#ffcc80'}       # Orange scheme
    }

    # Collect power data for all power sources
    gt_powers = []
    motor_powers = []

    # Gas turbine power for each nacelle
    for k in range(nt):
        gt_power_matrix = prob.get_val(f'nacelles.turb.power', units='kW')
        for i in range(npp):

            gt_powers.append(gt_power_matrix[i, :])


    # Motor power for each nacelle
    for j in range(nm):
        motor_power_matrix = prob.get_val(f'nacelles.motor.mech_power', units='kW')
        for i in range(npp):
            motor_powers.append(motor_power_matrix[i, :])

    # Separate motor power into positive and negative arrays by nacelle
    motor_powers_array = np.array(motor_powers)
    motor_positive = np.where(motor_powers_array >= 0, motor_powers_array, 0)
    motor_negative = np.where(motor_powers_array < 0, motor_powers_array, 0)

    # Create subplots for each nacelle
    for nacelle_idx in range(npp):
        row = nacelle_idx // 2
        col = nacelle_idx % 2
        ax = axes4[row, col]

        # Get nacelle power requirement for this specific nacelle
        nacelle_power_req_matrix = prob.get_val('nacelles.total_mech_power_out', units='kW')
        nacelle_power_req = nacelle_power_req_matrix[nacelle_idx, :]  # Get power for this specific nacelle

        # Get GT power for this nacelle
        gt = gt_powers[nacelle_idx]

        # Get motor powers for this nacelle
        em_drive_sum = np.zeros_like(gt)
        em_gen_sum = np.zeros_like(gt)
        for motor_idx in range(nm):
            array_idx = nacelle_idx * nm + motor_idx
            em_power = motor_powers_array[array_idx]
            em_drive_sum += np.where(em_power > 0, em_power, 0)
            em_gen_sum += np.where(em_power < 0, -em_power, 0)  # Flip sign for negative (generating) power
        # Stackplot for positive value
        ax.stackplot(
            time_min,
            gt,
            em_drive_sum,
            labels=['GT Power', 'EM Drive Power'],
            colors=[
                nacelle_colors[nacelle_idx+1]['gt'],
                nacelle_colors[nacelle_idx+1]['motor'],
            ],
            alpha=0.7
        )

    
        # Fill betwen for negative (generating) power
        if np.any(em_gen_sum > 0):
            ax.fill_between(
                time_min,
                0,
                -em_gen_sum,  # negative, so it fills below x-axis
                color=nacelle_colors[nacelle_idx+1]['generating'],
                alpha=0.7,
                label='EM Generating Power'
            )

        # Plot required power (dashed, black)
        ax.plot(
            time_min,
            nacelle_power_req,
            linestyle='--',
            linewidth=2,
            color='black',
            label='Required Power'
        )
        ax.set_title(f'Nacelle {nacelle_idx+1} Required & Delivered Power')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Mechanical Power (kW)')
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict()
        for h, l in zip(handles, labels):
            if l and l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def test_parallel_hybrid_electric_propulsion_system(plot_results=False):
    print("=" * 60)
    print("Testing ParallelHybridElectricPropulsionSystem")
    print("=" * 60)
    import time 
    # Create the problem
    prob = om.Problem(reports=False)
    start_time = time.time()
    # Number of analysis points
    num_nodes = 5
    npp = 4 # number of nacelles on aircraft
    nm = 1 # num motors per nacelle
    n_str = 4 # Number of battery strings in aircraft
    nt = 1 # num turbines per nacelle
    # Hybrid Powertrain Architecture Configuration Settings
    rule = "fraction" # Hybridization rule. 'fixed' or 'fraction'. Fraction = (component power out / total nacelle power out). Fixed = (fixed component power out). Other component takes the difference with respect to power requested by propeller
    bias = "em" # Component to apply the hybridization rule to. Bias = Gas Turbine (GT) or Electric Motor (EM). 
    # if bias = gt, and rule = fraction, power_split_fraction is an input. For eg, if power_split_fraciton = 0.2, 20% of the nacelle power is provided by the gas turbine.
    # if bias = em, and rule = fixed, power_split_amount is an input. For eg, if power_split_amount = 200kW, 200kW of the nacelle power is provided by the electric motor, the gas turbine provides the rest.
    nacelle_power_set = False # Whether or not the nacelle power is an input. If true, nacelle power is an input. If false, nacelle power is an output.
    prop_rpm_set = False # Whether or not the propeller RPM is an input. If true, propeller RPM is an input. If false, propeller RPM is an output.
    prop_thrust_set = True # Whether or not the thrust is an input. If true, thrust is an input. If false, thrust is an output.
    power_spec = 'relative' # Whether or not the power of the gas turbine and electric motor are specified independently. 
    # If 'independent', the power is specified independently. If 'relative', the power is specified relative to each other according to the hybridization rule and bias
    nacelle_command = 'power' # 'throttle' or 'power'. Whether to operate the nacelle power as a function of throttle or directly as power
    gt_command = 'power' # 'throttle' or 'power'. Whether to operate the gas turbine power as a function of throttle or directly as power.
    motor_command = 'power' # 'torque_rpm' or 'power'. Whether to operate the electric motor according to torque & rpm or power.
    gt_idle_allowed = True
    cnvg_throttle = True
    size_motor = False
    turb_type = 'PT6'
    # Incompatible Inconfiguration checker
    if power_spec=='independent' and prop_thrust_set == True:
        raise Warning("WARNING\n" + "="*50 + "System will not solve to meet thrust requirements if power_spec is `independent`")
    if power_spec=='independent' and nacelle_power_set == False:
        raise TypeError("Architecture is underconstrained. If power_spec is `independent`, `nacelle_power_set` is True")
    if power_spec=='independent' and nacelle_power_set== True and nacelle_command=='throttle':
        raise TypeError("Architecture is overconstrained. `power_spec` must be `relative` or `nacelle_command` must be `power`.")

    if nacelle_power_set==True and prop_thrust_set == True:
        raise TypeError("Architecture is overconstrained. `nacelle_power_set` and `prop_thrust_set` cannot both be True")
    if prop_rpm_set==False and prop_thrust_set == False and nacelle_power_set == False:
        raise TypeError("Architecture is underconstrained. `prop_thrust_set` or `nacelle_power_set` must be True")
    if (gt_command == 'throttle' or nacelle_command == 'throttle') and prop_thrust_set == True:
        raise TypeError("Architecture is overconstrained. `prop_thrust_set` must be False if `gt_command` or `nacelle_command` is `throttle`")
    # Add IndepVarComp that promotes all its outputs
    ivc = IndepVarComp()
    # Setup Mission Conditions
    velocity_profile = np.linspace(230, 230, num_nodes) * 0.514
    altitude_profile = np.linspace(20000,20000,num_nodes)
    temp_degC = 15.04 - 0.00649 * altitude_profile * 0.3048
    press_kPa = 101.29 * ((temp_degC + 273.15) / 288.08) ** 5.256
    rho_kgpm3 = press_kPa / (0.2869 * (temp_degC + 273.15))
    a = np.sqrt(1.4*287*(temp_degC+273.15))


    # Calculate thrust required
    cd0 = 0.03
    sref_m2 = 913 * 0.3048**2
    q_inf = 0.5 * 1.225 * velocity_profile**2
    weight_N = 83000 / 2.204 * 9.81
    CL = 0.5 * weight_N / (q_inf * sref_m2)
    wingspan_m = 111 * 0.3048
    ar = wingspan_m**2 / sref_m2
    e = 0.8
    cd = cd0 + CL**2 / (np.pi * ar * e)
    drag_N = cd * q_inf * sref_m2
    num_engines = 4
    thrust_profile_unit = drag_N / num_engines * np.ones((npp,num_nodes)) * np.array([1.0, 1.0, 1e-3, 1.0])[:,np.newaxis]
    dt = 5  # time step
    # Flight conditions
    ivc.add_output('fltcond|Utrue', val=velocity_profile, units='m/s')
    ivc.add_output('fltcond|h', val=altitude_profile, units='ft')
    ivc.add_output('fltcond|M', val=velocity_profile/330 , units=None)
    ivc.add_output('fltcond|disa', np.zeros((num_nodes)), desc='DISA in degrees Celsius', units='degC')
    ivc.add_output('fltcond|T', temp_degC, units='degC',desc='Temperature in degrees Celsius')
    ivc.add_output('fltcond|rho', rho_kgpm3,units='kg/m**3',desc='Air density')
    ivc.add_output('fltcond|a', a, units='m/s',desc='Sound speed')

    # Design variables
    ivc.add_output('ac|propulsion|turbine|rating', val=892.0 * np.ones((npp, num_nodes)), units='kW')
    ivc.add_output('ac|propulsion|propeller|diameter', val=14.3 *0.308 * np.ones((npp, num_nodes)), units='m')
    ivc.add_output('ac|propulsion|motor|rating', val=2300, units='kW')
    ivc.add_output('ac|propulsion|battery|q_cool_bat', val=25 * np.ones((n_str, num_nodes)), units='kW')
    #ivc.add_output('ac|propulsion|generator|rating', val=250.0 * np.ones(num_nodes), units='kW')
    # Max Rated Nacelle Power. THe same on each nacelle. 
    ivc.add_output('nacelle_max_rated_power', val=2500.0 * np.ones(num_nodes), units='hp')

    ivc.add_output('nacelles|voltage', val=800.0 * np.ones((npp,num_nodes)), units='V')
    #ivc.add_output('nacelles|rated_power_gt', val=872.0 , units='kW')
    #ivc.add_output('nacelles|rated_power_em', val=2300.0 , units='kW')
    # Battery Inputs
    bat_data = BatteryData.get_data(bat_filename='models/atlas/atlas/propulsion/empirical_data/MolicelP70Xplus_module210s8p_BOL_4grp1s14p_independent.xlsx', 
                                    cell_sheetname='BOL_cell_fct_CRate', 
                                    config_sheetname='battery_config')
    #ivc.add_output('ac|propulsion|battery|n_str', 4, desc='number of battery strings')
    #ivc.add_output('ac|propulsion|battery|n_motor', 4, desc='Number of electric motors')
    
    for i in range (nm):
        ivc.add_output(f'nacelles|motor|rpm', val=1790*np.ones((npp, num_nodes)) , units='rpm')
    # end 
    # Architecture Specific Inputs
    if prop_thrust_set:

        total_thrust_required = drag_N  # Total thrust = total drag (nn,)
        thrust_share_array = np.array([1.0, 1.0, 0.0, 1.0])  # Nacelle 2 disabled
        thrust_share = np.tile(thrust_share_array[:, np.newaxis], (1, num_nodes))  # (npp, nn)
        
        ivc.add_output('total_thrust_required', total_thrust_required, units="N")
        ivc.add_output('thrust_share', thrust_share, units=None)
    # end 
    if power_spec == 'relative':
        if rule == 'fraction':
            # Power split fraction
            ivc.add_output('nacelles|power_split_fraction', val=np.linspace(0.3, 1,num_nodes) * np.ones((npp, num_nodes)), units=None)
        elif rule == 'fixed':
            # Power split amount
            ivc.add_output('nacelles|power_split_amount', val=np.linspace(500, 600,num_nodes) * np.ones((npp, num_nodes)), units='kW')
        # end
    # end
    # Propeller RPM
    if prop_rpm_set:
        ivc.add_output('nacelles|prop_rpm', val=1000.0 * np.ones((npp, num_nodes)), units='rpm', desc='Propeller RPM')
    # end 
    nac_throttle = np.linspace(1e-3, 0.3, num_nodes) 
    p_max_nac_kW = 2500.0 * 0.7457 * np.ones(num_nodes) # 2500 SHP limit for propeller
    nac_mech_power__kW = p_max_nac_kW * nac_throttle
    # Max Nacelle power
    if nacelle_power_set and power_spec == 'relative':
        if nacelle_command == 'throttle':
            ivc.add_output('nacelles|max_rated_power', val=p_max_nac_kW , units='kW')
            ivc.add_output('nacelles|throttle', np.tile(nac_throttle, (npp, 1)), units=None)
        if nacelle_command == 'power':
            ivc.add_output('nacelles|total_mech_power_out', val=np.tile(nac_mech_power__kW, (npp, 1)), units='kW')
        # end
    gt_throttle = np.ones((num_nodes,)) * 0.5
    gt_power_kW = np.ones((num_nodes,)) * 1000
    if power_spec == 'independent':
        if gt_command=='throttle':
            for i in range (nt):
                ivc.add_output(f'nacelles|turb|throttle', gt_throttle * np.ones((npp, num_nodes)), units=None)
            # end
        elif gt_command=='power':
            for i in range (nt):
                ivc.add_output(f'nacelles|turb|power', val=gt_power_kW * np.ones((npp, num_nodes)), units='kW')
        # end
        if not size_motor:
            if motor_command=='torque_rpm':
                for i in range (nm):
                    ivc.add_output(f'nacelles|motor|torque', val=1000.0 * np.ones((npp, num_nodes)), units='N*m')
                # end
            elif motor_command=='power':
                for i in range (nm):
                    ivc.add_output(f'nacelles|motor|power', val=1000.0 * np.ones((npp, num_nodes)), units='kW')
                # end
            # end
        else:
            ivc.add_output('nacelles|motor|throttle', val=0.5* np.ones((npp, num_nodes)), units=None)
    # end
    prob.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])
    # Add the propulsion system
    prob.model.add_subsystem(
        'propulsion',
        ParallelHybridElectricPropulsionSystem(num_nodes=num_nodes, 
                                                   n_str=n_str,
                                                   num_props=npp, 
                                                   num_em_per_nac=nm, 
                                                   num_turb_per_nac=nt,
                                                   rule=rule, 
                                                   bias=bias, 
                                                   nacelle_power_set = nacelle_power_set,
                                                   prop_thrust_set = prop_thrust_set,
                                                   nacelle_command = nacelle_command,
                                                   prop_rpm_set = prop_rpm_set,
                                                   gt_command = gt_command,
                                                   motor_command = motor_command,
                                                   power_spec=power_spec,
                                                   gt_idle_allowed=gt_idle_allowed,
                                                   size_motor=size_motor,
                                                   cnvg_throttle=cnvg_throttle,
                                                   turb_type=turb_type,
                                                   ),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )

    #prob.model.connect('p_train_elec', 'hv_current_solver.p_load_required')
    #prob.model.connect('r_hv_cbl_ptrain', ['hv_current_solver.r_cable', 'hv_voltage_calc.r_cable'])
    #prob.model.connect('r_hv_cbl_aux', ['lv_current_solver.r_cable', 'lv_voltage_calc.r_cable'])
    # Set up connections for the problem
    # Always connect RPM to hybrid_split for motor power limit interpolation
    # This is needed regardless of motor_command because hybrid_split.motor_rpm_volt_pow_interp always requires RPM
    
    if motor_command=='torque_rpm':
        prob.model.connect(f"nacelles|motor|rpm", f"nacelles.motor.rpm")
    if motor_command=='power' or motor_command=='throttle':
        prob.model.connect(f"nacelles|motor|rpm", f"nacelles.rpm")

    if power_spec == 'independent':
        if motor_command=='torque_rpm':
            prob.model.connect(f"nacelles|motor|torque", f"nacelles.motor.torque")
        elif motor_command=='power':
            prob.model.connect(f"nacelles|motor|power", f"nacelles.motor.mech_power")
        if size_motor:
            prob.model.connect(f"nacelles|motor|throttle", f"nacelles.motor.throttle")
            # end
    if power_spec == 'relative':
        if rule == 'fraction':
            if bias == 'gt':
                prob.model.connect(f"nacelles|power_split_fraction", f"nacelles.power_split_fraction_gt")
            elif bias == 'em':
                prob.model.connect(f"nacelles|power_split_fraction", f"nacelles.power_split_fraction_em")
        elif rule == 'fixed':
            if bias == 'gt':
                prob.model.connect(f"nacelles|power_split_amount", f"nacelles.power_split_amount_gt")
            elif bias == 'em':
                prob.model.connect(f"nacelles|power_split_amount", f"nacelles.power_split_amount_em")
        # end

            # end
    # end
    if nacelle_power_set and power_spec == 'relative':
        if nacelle_command=='throttle':
            prob.model.connect(f"nacelles|throttle", f"nacelles.throttle")
            prob.model.connect(f"nacelles|max_rated_power", f"nacelles.max_rated_power")
        elif nacelle_command=='power':
            prob.model.connect(f"nacelles|total_mech_power_out", [f"nacelles.gb_comp.power_in", "nacelles.total_mech_power_out"])
        # end
    if prop_rpm_set:
        prob.model.connect(f"nacelles|prop_rpm", f"nacelles.prop.rpm")
    if power_spec == 'independent':
        if gt_command=='throttle':
            for i in range (nt):
                prob.model.connect(f"nacelles|turb|throttle", f"nacelles.turb.throttle")
        elif gt_command=='power':
            for i in range (nt):
                prob.model.connect(f"nacelles|turb|power", [f"nacelles.turb.power", f"nacelles.turb_power"])

    if prop_thrust_set:
        if cnvg_throttle:
            # NEW: total_thrust_required and thrust_share are promoted, no explicit connection needed
            pass
        else:
            prob.model.connect(f"nacelles|thrust", f"nacelles.subtract_jet_thrust.thrust")
            

    # end

    #prob.model.connect(f"nacelles|rated_power_gt", f"nacelles.rated_power_gt")
    #prob.model.connect(f"nacelles|rated_power_em", f"nacelles.rated_power_em")

    prob.model.connect(f"nacelles|voltage", f"nacelles.voltage")
    import time 
    prob.setup()

    run_start_time = time.time()

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 50
    prob.model.nonlinear_solver.options['atol'] = 1e-3
    prob.model.nonlinear_solver.options['rtol'] = 1e-3
    # Use ScipyKrylov instead of DirectSolver to avoid memory issues
    prob.model.linear_solver = om.DirectSolver()


    om.n2(prob)
    prob.check_partials(compact_print=True, show_only_incorrect=False, method='fd')
    print("Problem setup complete!")
    # Run the model
    print("\nRunning the model...")

    prob.run_model()
    print("Model executed successfully!")
    end_time = time.time()
    print(f"Setup time: {run_start_time - start_time:.2f} seconds")
    print(f"Execution time: {end_time - run_start_time:.2f} seconds")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")


    # Test the throttle balance component
    if cnvg_throttle and prop_thrust_set:
        total_thrust_calc = prob.get_val('throttle_balance.total_thrust_calculated', units='N')
        total_thrust_req = prob.get_val('total_thrust_required', units='N')
        nominal_throttle = prob.get_val('nominal_throttle')
        throttle_nac = prob.get_val('throttle_nac')
        thrust_share = prob.get_val('thrust_share')
        
        abs_error = np.abs(total_thrust_calc - total_thrust_req)

        print(f"\n---- Thrust Convergence Check (NEW ARCHITECTURE): ----")
        print(f"Total thrust required: {total_thrust_req}")
        print(f"Total thrust calculated: {total_thrust_calc}")
        print(f"Absolute error: {abs_error}")
        print(f"Max error: {np.max(abs_error):.2f} N")
        print(f"\nNominal throttle: {nominal_throttle}")
        print(f"Thrust share:\n{thrust_share}")
        print(f"Throttle per nacelle:\n{throttle_nac}")

    if plot_results:
    # Generate plots
        plot_hybrid_propulsion_results(prob, npp=npp, nm=nm, nt=nt, dt=dt, 
                                    prop_thrust_set=prop_thrust_set, 
                                    nacelle_power_set=nacelle_power_set)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)



if __name__ == "__main__":
    # Run the test

    test_parallel_hybrid_electric_propulsion_system(plot_results=True)
