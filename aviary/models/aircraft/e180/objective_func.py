import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import openmdao.jax as omj
import jax.numpy as jnp
import jax

class ObjectiveFuncFuel(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare("soc_min", default=0.13)
        self.options.declare("min_voltage", default=570)
        self.options.declare("n_str", default=1)
        self.options.declare("n_cases", default=1)
        self.options.declare("mu_volts", default=1e-3)
        self.options.declare("mu_soc", default=1e-4)
    
    def setup(self):
        n_str = self.options["n_str"]
        n_cases = self.options["n_cases"]

        self.add_input("soc_end", units=None, shape=(n_str, n_cases))
        self.add_input("min_voltage", units="V", shape=(n_str, n_cases))
        self.add_input("block_fuel", units="lbm", shape=(n_cases,))

        self.add_output("mixed_objective", units=None, shape=(n_cases,))

        # Diagonal partials for scalar-to-scalar derivatives
        rows = np.arange(n_cases, dtype=int)
        cols = np.arange(n_cases, dtype=int)
        
        # For soc_end and min_voltage: output[i] depends on input[j, i] for all j
        # rows: which output element [0,0,...,0, 1,1,...,1, ..., n_cases-1,...]
        # cols: which input element in flattened array
        rows_str = np.repeat(np.arange(n_cases, dtype=int), n_str)
        cols_str = np.tile(np.arange(n_str, dtype=int), n_cases) * n_cases + np.repeat(np.arange(n_cases, dtype=int), n_str)
        
        self.declare_partials("mixed_objective", "soc_end", rows=rows_str, cols=cols_str)
        self.declare_partials("mixed_objective", "min_voltage", rows=rows_str, cols=cols_str)
        self.declare_partials("mixed_objective", "block_fuel", rows=rows, cols=cols)

    def compute_primal(self, soc_end, min_voltage, block_fuel):
        n_str = self.options["n_str"]
        mu_volts = self.options["mu_volts"]
        mu_soc = self.options["mu_soc"]

        n_cases = self.options["n_cases"]
        voltage_penalty = np.zeros(n_cases)
        soc_penalty = np.zeros(n_cases)

        for i in range(n_str):
            diff_volt = self.options["min_voltage"] - min_voltage[i, :]
            #voltage_penalty += omj.smooth_max(0, diff_volt, mu=mu_volts) /30
            voltage_penalty += jnp.maximum(0, diff_volt) / 30
            
            diff_soc = self.options["soc_min"] - soc_end[i, :]
            # Smooth asymmetric penalty: different slopes on each side of target
            # When soc < target (diff_soc > 0): higher penalty (alpha)
            # When soc > target (diff_soc < 0): lower penalty (beta)
            # Using smooth_max ensures C∞ continuity
            alpha = 20.0  # Penalty coefficient when soc < target (undershoot - worse)
            beta = 5.0    # Penalty coefficient when soc > target (overshoot - less bad)
            soc_penalty += alpha * omj.smooth_max(0, diff_soc, mu=mu_soc)# + 
                           #beta * omj.smooth_max(0, -diff_soc, mu=mu_soc))
            #soc_penalty += alpha * jnp.maximum(0, diff_soc) 

        mixed_objective =  block_fuel / 100  + voltage_penalty + soc_penalty # Can also divide by 250...


        jax.debug.print("Mixed Objective: {}", mixed_objective)

        jax.debug.print("Fuel Used (lbm): {}", block_fuel)
        jax.debug.print("Voltage Penalty: {}", voltage_penalty)
        jax.debug.print("Min Voltage: {}", min_voltage)
        jax.debug.print("SOC Penalty: {}", soc_penalty)
        jax.debug.print("SOC Min: {}", soc_end)

        return mixed_objective

    def self_get_statics(self):
        return (self.options["mu_volts"], self.options["mu_soc"])

class ObjectiveFuncRange(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare("soc_min", default=0.13)
        #self.options.declare("fuel_limit", default=3500)
        self.options.declare("min_voltage", default=570)
        self.options.declare("n_str", default=1)
        self.options.declare("n_cases", default=1)
        self.options.declare("mu_volts", default=1e-3)
        self.options.declare("mu_soc", default=1e-4)
    
    def setup(self):
        n_str = self.options["n_str"]
        n_cases = self.options["n_cases"]

        self.add_input("total_range", units="NM", shape=(n_cases,))
        self.add_input("soc_end", units=None, shape=(n_str, n_cases))
        self.add_input("min_voltage", units="V", shape=(n_str, n_cases))
        self.add_input("total_fuel_used", units="kg", shape=(n_cases,))
        self.add_input("fuel_limit", units="kg", shape=(1))
        self.add_input("TOW", units="lbm", shape=(1))
        self.add_input("OEW", units="lbm", shape=(1))
        self.add_input("payload", units="lbm", shape=(1))

        self.add_output("mixed_objective", units=None, shape=(n_cases,))

        # Diagonal partials for scalar-to-scalar derivatives
        rows = np.arange(n_cases, dtype=int)
        cols = np.arange(n_cases, dtype=int)
        
        # For soc_end and min_voltage: output[i] depends on input[j, i] for all j
        # rows: which output element [0,0,...,0, 1,1,...,1, ..., n_cases-1,...]
        # cols: which input element in flattened array
        rows_str = np.repeat(np.arange(n_cases, dtype=int), n_str)
        cols_str = np.tile(np.arange(n_str, dtype=int), n_cases) * n_cases + np.repeat(np.arange(n_cases, dtype=int), n_str)
        
        self.declare_partials("mixed_objective", "total_range", rows=rows, cols=cols)
        self.declare_partials("mixed_objective", "soc_end", rows=rows_str, cols=cols_str)
        self.declare_partials("mixed_objective", "min_voltage", rows=rows_str, cols=cols_str)
        self.declare_partials("mixed_objective", "total_fuel_used", rows=rows, cols=cols)

    def compute_primal(self, total_range, soc_end, min_voltage, total_fuel_used, fuel_limit, TOW, OEW, payload):
        n_str = self.options["n_str"]
        mu_volts = self.options["mu_volts"]
        mu_soc = self.options["mu_soc"]

        diff_fuel = total_fuel_used - fuel_limit
        #alpha_fuel = 200.0 # Overshoot 
        alpha_fuel = 100
        #fuel_penalty = jnp.maximum(0, diff_fuel) / alpha_fuel

        fuel_penalty = omj.smooth_max(0, diff_fuel, mu=0.1) / alpha_fuel #+ 
        #                #omj.smooth_max(0, -diff_fuel, mu=mu_volts) / beta_fuel)

        n_cases = self.options["n_cases"]
        voltage_penalty = np.zeros(n_cases)
        soc_penalty = np.zeros(n_cases)

        for i in range(n_str):
            diff_volt = self.options["min_voltage"] - min_voltage[i, :]
            #voltage_penalty += omj.smooth_max(0, diff_volt, mu=mu_volts) /30
            voltage_penalty += jnp.maximum(0, diff_volt) / 30
            
            diff_soc = self.options["soc_min"] - soc_end[i, :]
            # Smooth asymmetric penalty: different slopes on each side of target
            # When soc < target (diff_soc > 0): higher penalty (alpha)
            # When soc > target (diff_soc < 0): lower penalty (beta)
            # Using smooth_max ensures C∞ continuity
            alpha = 30.0  # Penalty coefficient when soc < target (undershoot - worse)
            beta = 5.0    # Penalty coefficient when soc > target (overshoot - less bad)
            soc_penalty += alpha * omj.smooth_max(0, diff_soc, mu=mu_soc)# + 
                           #beta * omj.smooth_max(0, -diff_soc, mu=mu_soc))
            #soc_penalty += alpha * jnp.maximum(0, diff_soc) 

        mixed_objective = -total_range / 100 + soc_penalty + voltage_penalty#+ fuel_penalty # Can also divide by 250...
        mission_range_est = total_range - 250



        jax.debug.print("Mixed Objective: {}", mixed_objective)

        jax.debug.print("Total Range (NM): {}", total_range)
        jax.debug.print("Mission Range Estimate (NM): {}", mission_range_est)
        jax.debug.print("Fuel Used (kg): {}", total_fuel_used)
        jax.debug.print("Fuel Penalty (kg): {}", fuel_penalty)
        jax.debug.print("Fuel Limit (kg): {}", fuel_limit)
        jax.debug.print("Voltage Penalty: {}", voltage_penalty)
        jax.debug.print("Min Voltage (V): {}", min_voltage)
        jax.debug.print("SOC Penalty: {}", soc_penalty)
        jax.debug.print("SOC Min (): {}", soc_end)
        jax.debug.print("TOW (lbm): {}", TOW)
        jax.debug.print("OEW (lbm): {}", OEW)
        jax.debug.print("Payload (lbm): {}", payload)

        return mixed_objective

    def self_get_statics(self):
        return (self.options["mu_volts"], self.options["mu_soc"])
    
def test_jax_objective_func(soc_end_min, min_voltage_limit, fuel_limit):

    # =========================================================================
    # TEST ObjectiveFunc (JAX) - check partials
    # =========================================================================
    print("\n" + "="*80)
    print("TESTING ObjectiveFunc (JAX) - Automatic Differentiation")
    print("="*80)
    
    prob_check = Problem(reports=False)
    ivc_check = IndepVarComp()
    ivc_check.add_output("total_range", val=np.array([500.0]), units="NM")
    ivc_check.add_output("soc_end", val=np.array([[0.13]]), units=None)
    ivc_check.add_output("min_voltage", val=np.array([[570.0]]), units="V")
    ivc_check.add_output("total_fuel_used", val=np.array([3000.0]), units="kg")
    prob_check.model.add_subsystem("ivc", ivc_check, promotes=["*"])
    prob_check.model.add_subsystem("obj", ObjectiveFuncRange(n_str=1, n_cases=1, soc_min=soc_end_min, 
                                                         min_voltage=min_voltage_limit, fuel_lim=fuel_limit), 
                                    promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
    prob_check.setup()
    prob_check.check_partials(compact_print=True, show_only_incorrect=True)

    # Default values for fixed variables
    defaults = {
        "total_range": 500.0,
        "soc_end": 0.2,
        "min_voltage": 600.0,
        "total_fuel_used": 3000.0,
    }

    # Sweep ranges - 5 points each
    n_points = 11
    sweeps = {
        "total_range": np.linspace(0, 1000, n_points),
        "soc_end": np.linspace(0, 1, n_points),
        "min_voltage": np.linspace(500, 800, n_points),
        "total_fuel_used": np.linspace(1000, 5000, n_points),
    }

    # Create meshgrid for 2D combinations
    # Pair 1: range vs fuel
    range_vals, fuel_vals = np.meshgrid(sweeps["total_range"], sweeps["total_fuel_used"], indexing='ij')
    n_cases = range_vals.size
    
    # Pair 2: soc vs voltage
    soc_vals, volt_vals = np.meshgrid(sweeps["soc_end"], sweeps["min_voltage"], indexing='ij')
    
    # Pair 3: range vs soc
    range_vals2, soc_vals2 = np.meshgrid(sweeps["total_range"], sweeps["soc_end"], indexing='ij')
    
    # Pair 4: fuel vs voltage
    fuel_vals2, volt_vals2 = np.meshgrid(sweeps["total_fuel_used"], sweeps["min_voltage"], indexing='ij')

    # Compute range vs fuel combinations (vectorized)
    print("Computing range vs fuel meshgrid...")
    prob1 = Problem(reports=False)
    ivc1 = IndepVarComp()
    ivc1.add_output("total_range", val=np.zeros(n_cases), units="NM")
    ivc1.add_output("soc_end", val=np.zeros((1, n_cases)), units=None)
    ivc1.add_output("min_voltage", val=np.zeros((1, n_cases)), units="V")
    ivc1.add_output("total_fuel_used", val=np.zeros(n_cases), units="kg")
    prob1.model.add_subsystem("ivc", ivc1, promotes=["*"])
    prob1.model.add_subsystem("obj", ObjectiveFunc(n_str=1, n_cases=n_cases, soc_min=soc_end_min, 
                                                   min_voltage=min_voltage_limit, fuel_lim=fuel_limit), 
                             promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
    prob1.setup()
    prob1.set_val("total_range", range_vals.flatten())
    prob1.set_val("total_fuel_used", fuel_vals.flatten())
    prob1.set_val("soc_end", np.tile(defaults["soc_end"], (1, n_cases)))
    prob1.set_val("min_voltage", np.tile(defaults["min_voltage"], (1, n_cases)))
    prob1.run_model()
    obj_range_fuel = prob1.get_val("mixed_objective").reshape(range_vals.shape)
    print(f"Range vs Fuel - Min: {obj_range_fuel.min():.4f}, Max: {obj_range_fuel.max():.4f}")

    # Compute soc vs voltage combinations (vectorized)
    print("Computing soc vs voltage meshgrid...")
    prob2 = Problem(reports=False)
    ivc2 = IndepVarComp()
    ivc2.add_output("total_range", val=np.zeros(n_cases), units="NM")
    ivc2.add_output("soc_end", val=np.zeros((1, n_cases)), units=None)
    ivc2.add_output("min_voltage", val=np.zeros((1, n_cases)), units="V")
    ivc2.add_output("total_fuel_used", val=np.zeros(n_cases), units="kg")
    prob2.model.add_subsystem("ivc", ivc2, promotes=["*"])
    prob2.model.add_subsystem("obj", ObjectiveFunc(n_str=1, n_cases=n_cases, soc_min=soc_end_min, 
                                                   min_voltage=min_voltage_limit, fuel_lim=fuel_limit), 
                             promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
    prob2.setup()
    prob2.set_val("soc_end", soc_vals.flatten().reshape(1, -1))
    prob2.set_val("min_voltage", volt_vals.flatten().reshape(1, -1))
    prob2.set_val("total_range", np.full(n_cases, defaults["total_range"]))
    prob2.set_val("total_fuel_used", np.full(n_cases, defaults["total_fuel_used"]))
    prob2.run_model()
    obj_soc_volt = prob2.get_val("mixed_objective").reshape(soc_vals.shape)
    print(f"SOC vs Voltage - Min: {obj_soc_volt.min():.4f}, Max: {obj_soc_volt.max():.4f}")

    # Compute range vs soc combinations (vectorized)
    print("Computing range vs soc meshgrid...")
    prob3 = Problem(reports=False)
    ivc3 = IndepVarComp()
    ivc3.add_output("total_range", val=np.zeros(n_cases), units="NM")
    ivc3.add_output("soc_end", val=np.zeros((1, n_cases)), units=None)
    ivc3.add_output("min_voltage", val=np.zeros((1, n_cases)), units="V")
    ivc3.add_output("total_fuel_used", val=np.zeros(n_cases), units="kg")
    prob3.model.add_subsystem("ivc", ivc3, promotes=["*"])
    prob3.model.add_subsystem("obj", ObjectiveFunc(n_str=1, n_cases=n_cases, soc_min=soc_end_min, 
                                                   min_voltage=min_voltage_limit, fuel_lim=fuel_limit), 
                             promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
    prob3.setup()
    prob3.set_val("total_range", range_vals2.flatten())
    prob3.set_val("soc_end", soc_vals2.flatten().reshape(1, -1))
    prob3.set_val("min_voltage", np.tile(defaults["min_voltage"], (1, n_cases)))
    prob3.set_val("total_fuel_used", np.full(n_cases, defaults["total_fuel_used"]))
    prob3.run_model()
    obj_range_soc = prob3.get_val("mixed_objective").reshape(range_vals2.shape)
    print(f"Range vs SOC - Min: {obj_range_soc.min():.4f}, Max: {obj_range_soc.max():.4f}")

    # Compute fuel vs voltage combinations (vectorized)
    print("Computing fuel vs voltage meshgrid...")
    prob4 = Problem(reports=False)
    ivc4 = IndepVarComp()
    ivc4.add_output("total_range", val=np.zeros(n_cases), units="NM")
    ivc4.add_output("soc_end", val=np.zeros((1, n_cases)), units=None)
    ivc4.add_output("min_voltage", val=np.zeros((1, n_cases)), units="V")
    ivc4.add_output("total_fuel_used", val=np.zeros(n_cases), units="kg")
    prob4.model.add_subsystem("ivc", ivc4, promotes=["*"])
    prob4.model.add_subsystem("obj", ObjectiveFunc(n_str=1, n_cases=n_cases, soc_min=soc_end_min, 
                                                   min_voltage=min_voltage_limit, fuel_lim=fuel_limit), 
                             promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
    prob4.setup()
    prob4.set_val("total_fuel_used", fuel_vals2.flatten())
    prob4.set_val("min_voltage", volt_vals2.flatten().reshape(1, -1))
    prob4.set_val("total_range", np.full(n_cases, defaults["total_range"]))
    prob4.set_val("soc_end", np.tile(defaults["soc_end"], (1, n_cases)))
    prob4.run_model()
    obj_fuel_volt = prob4.get_val("mixed_objective").reshape(fuel_vals2.shape)
    print(f"Fuel vs Voltage - Min: {obj_fuel_volt.min():.4f}, Max: {obj_fuel_volt.max():.4f}")
    
    # Print breakdown for a typical case
    print("\n--- Sample breakdown (range=500, fuel=3000, soc=0.2, voltage=600) ---")
    print(f"Range term: {-500/250:.4f}")
    print(f"Fuel penalty (3000 < 3500): {max(0, 3000-3500)/100:.4f}")
    print(f"Voltage penalty (600 > 570): {max(0, 570-600)/100:.4f}")
    print(f"SOC penalty (0.2 > 0.13): {max(0, 0.13-0.2)*10:.4f}")
    print(f"Total: {-500/250 + max(0, 3000-3500)/100 + max(0, 570-600)/100 + max(0, 0.13-0.2)*10:.4f}")

    # Plotting
    fig = plt.figure(figsize=(16, 14))
    
    # Range vs Fuel
    ax1 = fig.add_subplot(2, 2, 1)
    contour1 = ax1.contourf(range_vals, fuel_vals, obj_range_fuel, levels=20, cmap='viridis')
    ax1.axhline(y=fuel_limit, color='r', linestyle='--', linewidth=2, label=f'Fuel Limit = {fuel_limit} kg')
    ax1.set_xlabel("Total Range (NM)", fontsize=11)
    ax1.set_ylabel("Fuel Used (kg)", fontsize=11)
    ax1.set_title("Objective: Range vs Fuel", fontsize=12, pad=10)
    ax1.legend(loc='best', fontsize=9)
    plt.colorbar(contour1, ax=ax1)
    
    # SOC vs Voltage
    ax2 = fig.add_subplot(2, 2, 2)
    contour2 = ax2.contourf(soc_vals, volt_vals, obj_soc_volt, levels=20, cmap='viridis')
    ax2.axhline(y=min_voltage_limit, color='r', linestyle='--', linewidth=2, label=f'Min Voltage = {min_voltage_limit} V')
    ax2.axvline(x=soc_end_min, color='r', linestyle='--', linewidth=2, label=f'Min SOC = {soc_end_min}')
    ax2.set_xlabel("SOC End Value", fontsize=11)
    ax2.set_ylabel("Min Voltage (V)", fontsize=11)
    ax2.set_title("Objective: SOC vs Voltage", fontsize=12, pad=10)
    ax2.legend(loc='best', fontsize=9)
    plt.colorbar(contour2, ax=ax2)
    
    # Range vs SOC
    ax3 = fig.add_subplot(2, 2, 3)
    contour3 = ax3.contourf(range_vals2, soc_vals2, obj_range_soc, levels=20, cmap='viridis')
    ax3.axhline(y=soc_end_min, color='r', linestyle='--', linewidth=2, label=f'Min SOC = {soc_end_min}')
    ax3.set_xlabel("Total Range (NM)", fontsize=11)
    ax3.set_ylabel("SOC End Value", fontsize=11)
    ax3.set_title("Objective: Range vs SOC", fontsize=12, pad=10)
    ax3.legend(loc='best', fontsize=9)
    plt.colorbar(contour3, ax=ax3)
    
    # Fuel vs Voltage
    ax4 = fig.add_subplot(2, 2, 4)
    contour4 = ax4.contourf(fuel_vals2, volt_vals2, obj_fuel_volt, levels=20, cmap='viridis')
    ax4.axhline(y=min_voltage_limit, color='r', linestyle='--', linewidth=2, label=f'Min Voltage = {min_voltage_limit} V')
    ax4.axvline(x=fuel_limit, color='r', linestyle='--', linewidth=2, label=f'Fuel Limit = {fuel_limit} kg')
    ax4.set_xlabel("Fuel Used (kg)", fontsize=11)
    ax4.set_ylabel("Min Voltage (V)", fontsize=11)
    ax4.set_title("Objective: Fuel vs Voltage", fontsize=12, pad=10)
    ax4.legend(loc='best', fontsize=9)
    plt.colorbar(contour4, ax=ax4)
    
    plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=2.0)
    plt.show()



class SmoothMaxTestComponent(om.JaxExplicitComponent):
    """
    Simple test component to visualize smooth_max behavior with different mu values.
    
    output = smooth_max(0, input - threshold, mu)
    """
    
    def initialize(self):
        self.options.declare("threshold", default=3000.0)
        self.options.declare("mu", default=1.0)
        self.options.declare("n_cases", default=1)
    
    def setup(self):
        n_cases = self.options["n_cases"]
        self.add_input("input_vector", shape=(n_cases,))
        self.add_output("output_vector", shape=(n_cases,))
        
        rows = np.arange(n_cases, dtype=int)
        cols = np.arange(n_cases, dtype=int)
        self.declare_partials("output_vector", "input_vector", rows=rows, cols=cols)
    
    def compute_primal(self, input_vector):
        threshold = self.options["threshold"]
        mu = self.options["mu"]
        
        delta = input_vector - threshold
        output = omj.smooth_max(0, delta, mu=mu)
        
        return output


def test_smooth_max_mu_sweep():
    """
    Test and visualize smooth_max behavior with different mu values.
    Plots input vs output for several orders of magnitude of mu.
    """
    import matplotlib.pyplot as plt
    from openmdao.api import Problem, IndepVarComp
    
    print("\n" + "="*80)
    print("TESTING smooth_max with different mu values")
    print("="*80)
    
    threshold = 3000.0
    n_points = 1000
    input_vals = np.linspace(2900, 3100, n_points)
    
    # Different orders of magnitude for mu
    mu_values = [1e-4,1e-3,1e-2, 1e-1, 1.0, 10.0, 100.0]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Also compute the hard max for reference
    hard_max = np.maximum(0, input_vals - threshold)
    raw_delta = input_vals - threshold
    ax1.plot(input_vals, hard_max, 'k--', linewidth=2, label='max(0, x-threshold)', zorder=10)
    ax1.plot(input_vals, raw_delta, 'k--', linewidth=2, label='x-threshold', zorder=10)
    for mu, color in zip(mu_values, colors):
        print(f"\nTesting mu = {mu}")
        
        prob = Problem(reports=False)
        ivc = IndepVarComp()
        ivc.add_output("input_vector", val=input_vals)
        prob.model.add_subsystem("ivc", ivc, promotes=["*"])
        prob.model.add_subsystem("smooth_max", SmoothMaxTestComponent(
            threshold=threshold,
            mu=mu,
            n_cases=n_points
        ), promotes_inputs=["*"], promotes_outputs=["*"])
        prob.setup()
        prob.run_model()
        
        output_vals = prob.get_val("output_vector")
        
        ax1.plot(input_vals, output_vals, color=color, linewidth=1.5, label=f'mu = {mu}')
        
        # Also plot the derivative (sigmoid)
        # d/dx smooth_max(0, x-threshold) ≈ sigmoid((x-threshold)/mu)
        delta = input_vals - threshold
        sigmoid = 1.0 / (1.0 + np.exp(-delta / mu))
        ax2.plot(input_vals, sigmoid, color=color, linewidth=1.5, label=f'mu = {mu}')
    
    # Hard step function for reference
    hard_step = np.where(input_vals > threshold, 1.0, 0.0)
    ax2.plot(input_vals, hard_step, 'k--', linewidth=2, label='Heaviside step', zorder=10)
    
    # Configure first plot (smooth_max output)
    ax1.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel("Input Value", fontsize=11)
    ax1.set_ylabel("smooth_max(0, input - threshold)", fontsize=11)
    ax1.set_title(f"smooth_max Output (threshold = {threshold})", fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([2900, 3100])
    ax1.set_ylim([-100, 100])
    
    # Configure second plot (derivative/sigmoid)
    ax2.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel("Input Value", fontsize=11)
    ax2.set_ylabel("Derivative (sigmoid)", fontsize=11)
    ax2.set_title(f"smooth_max Derivative (threshold = {threshold})", fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([2900, 3100])
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig("models/atlas/atlas/scenarios/setup_mission/smooth_max_mu_comparison.png", dpi=150)
    print(f"\nPlot saved to: smooth_max_mu_comparison.png")
    plt.show()
    
    print("\n" + "="*80)
    print("Observations:")
    print("  - Smaller mu (e.g., 1e-2): Sharper transition, closer to hard max")
    print("  - Larger mu (e.g., 100): Smoother transition, gradients extend further")
    print("  - For optimization: larger mu = easier gradients but less accurate constraint")
    print("="*80)


def test_smooth_objective_func(soc_end_min, min_voltage_limit, fuel_limit):

    # Define constraint values
    # =========================================================================
    # TEST ObjectiveFuncSoftplus - check partials at various operating points
    # =========================================================================
    print("\n" + "="*80)
    print("TESTING ObjectiveFuncSoftplus - Analytical Partials")
    print("="*80)
    
    test_cases = [
        # (range, soc, voltage, fuel, description)
        (500.0, 0.5, 700.0, 3000.0, "Nominal - all constraints satisfied"),
        (500.0, 0.10, 700.0, 3000.0, "SOC below minimum (0.10 < 0.13)"),
        (500.0, 0.5, 550.0, 3000.0, "Voltage below minimum (550 < 570)"),
        (500.0, 0.5, 700.0, 3800.0, "Fuel above limit (3800 > 3500)"),
        (500.0, 0.13, 570.0, 3500.0, "At constraint boundaries"),
        (500.0, 0.05, 540.0, 4000.0, "All constraints violated"),
    ]
    
    for range_val, soc_val, volt_val, fuel_val, desc in test_cases:
        print(f"\n--- Test: {desc} ---")
        print(f"    Range={range_val}, SOC={soc_val}, Voltage={volt_val}, Fuel={fuel_val}")
        
        prob_sp = Problem(reports=False)
        ivc_sp = IndepVarComp()
        ivc_sp.add_output("total_range", val=np.array([range_val]), units="NM")
        ivc_sp.add_output("soc_end", val=np.array([[soc_val]]), units=None)
        ivc_sp.add_output("min_voltage", val=np.array([[volt_val]]), units="V")
        ivc_sp.add_output("total_fuel_used", val=np.array([fuel_val]), units="kg")
        prob_sp.model.add_subsystem("ivc", ivc_sp, promotes=["*"])
        prob_sp.model.add_subsystem("obj", ObjectiveFuncSoftplus(
            n_str=1, n_cases=1, 
            soc_min=soc_end_min, 
            min_voltage=min_voltage_limit, 
            fuel_lim=fuel_limit
        ), promotes_inputs=["*"], promotes_outputs=["mixed_objective"])
        prob_sp.setup()
        prob_sp.run_model()
        
        # Check partials
        prob_sp.check_partials(compact_print=True, show_only_incorrect=True, out_stream=None)
   


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from openmdao.api import Problem, IndepVarComp

    soc_end_min = 0.13
    min_voltage_limit = 570.0
    fuel_limit = 3500.0

    # Test smooth_max with different mu values first (quick visual check)
    test_smooth_max_mu_sweep()
    
    # Then run the objective function tests
    # test_jax_objective_func(soc_end_min, min_voltage_limit, fuel_limit)
    # test_smooth_objective_func(soc_end_min, min_voltage_limit, fuel_limit)
    