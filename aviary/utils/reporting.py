import openmdao.api as om

class PrintDurationComp(om.ExplicitComponent):
    """
    Explicit component that takes in 'duration' and prints it.
    """
    def initialize(self):
        self.options.declare("phase_name", default="text")

    def setup(self):
        self.add_input("duration", shape=(1,), units="s")

    def compute(self, inputs, outputs):
        pass
        #print("\n--------------------------------------------------")
        #print(f"Duration for {self.options['phase_name']}: {inputs['duration']/60} minutes")
        #print("--------------------------------------------------\n")

# Example usage (add as a subsystem if desired):
# self.add_subsystem("print_duration", PrintDurationComp(num_nodes=nn), promotes_inputs=["duration"])

