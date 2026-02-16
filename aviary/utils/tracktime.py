import openmdao.api as om
import numpy as np

class TrackTime(om.ExplicitComponent):
    """
    component that tracks the time of each phase.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("t0", units="s", shape=(1,))
        self.add_input("duration", units="s", shape=(1,))

        self.add_output("t_final", units="s", shape=(1,))
        self.add_output("t", units="s", shape=(nn,))

        rows_t = np.arange(nn)
        cols_t = np.zeros(nn)
        rows_t_final = np.arange(1)
        cols_t_final = np.arange(1)
        self.declare_partials('t_final', 't0', rows=rows_t_final, cols=cols_t_final)
        self.declare_partials('t_final', 'duration', rows=rows_t_final, cols=cols_t_final)
        self.declare_partials('t', 't0', rows=rows_t, cols=cols_t)
        self.declare_partials('t', 'duration', rows=rows_t, cols=cols_t)

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]

        t0 = inputs['t0']
        duration = inputs['duration']

        t_final = t0 + duration
        t = np.linspace(t0, t_final, nn).flatten()

        outputs['t_final'] = t_final
        outputs['t'] = t
    
    def compute_partials(self, inputs, partials):
        nn = self.options["num_nodes"]
        idx = np.arange(nn)
        frac = idx / (nn - 1)

        # t_final partials
        partials['t_final', 't0'] = 1.0
        partials['t_final', 'duration'] = 1.0

        # t partials
        partials['t', 't0'] = np.ones(nn)
        partials['t', 'duration'] = frac


if __name__ == "__main__":
    import openmdao.api as om
    import numpy as np
    
    # Test TrackTime component with check_partials
    print("Testing TrackTime component with check_partials...")
    
    # Create a problem with the component
    prob = om.Problem()
    
    # Add an independent variable component for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('t0', val=0.0, units='s')
    ivc.add_output('duration', val=100.0, units='s')
    
    # Add the TrackTime component
    prob.model.add_subsystem('track_time', TrackTime(num_nodes=11), promotes=['*'])
    
    # Setup and run
    prob.setup()
    prob.run_model()
    
    # Print outputs
    print(f"\nt0: {prob.get_val('t0', units='s')}")
    print(f"duration: {prob.get_val('duration', units='s')}")
    print(f"t_final: {prob.get_val('t_final', units='s')}")
    print(f"t: {prob.get_val('t', units='s')}")
    
    # Check partials
    print("\nChecking partials...")
    check_results = prob.check_partials(compact_print=True, method='fd', step=1e-6, show_only_incorrect=True)
    
    print("\nPartial derivatives check complete!")
