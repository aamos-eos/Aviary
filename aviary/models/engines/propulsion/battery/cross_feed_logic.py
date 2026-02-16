import numpy as np
import openmdao.api as om

class DemandFeed(om.ExplicitComponent):
    """
    Component that converts motor battery power matrix to battery pack power matrix
    based on feeder logic. Maps motor powers to battery string powers.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('nm', default=4, desc='number of motors')
        self.options.declare('n_str', default=4, desc='number of battery strings')
        self.options.declare('feeder_mode', default='independent',
                             desc='Feeder mode: independent, full_crossfeed, or grouped')
        self.options.declare('active_strings', default=None,
                             desc='Array of active string indices (None = all active)')

    def setup(self):
        nn = self.options['num_nodes']
        nm = self.options['nm']
        n_str = self.options['n_str']

        # Input: motor power matrix (nm, nn)
        self.add_input('p_train_elec', shape=(nm, nn), units='kW',
                       desc='Motor battery power matrix (one row per motor)')

        # Output: battery pack power matrix (n_str, nn)
        self.add_output('p_train_elec_req', shape=(n_str, nn), units='kW',
                        desc='Battery pack power matrix (one row per battery string)')

        # We'll provide exact analytic partials
        self.declare_partials('p_train_elec_req', 'p_train_elec', method='exact')

    def compute(self, inputs, outputs):
        p_train_elec = inputs['p_train_elec']
        nn = self.options['num_nodes']
        nm = self.options['nm']
        n_str = self.options['n_str']
        feeder_mode = self.options['feeder_mode']
        active_strings = self.options['active_strings']

        # Convert active_strings to numpy array for easier indexing
        if active_strings is None:
            active_strings_np = np.ones(n_str, dtype=float)
        else:
            active_strings_np = np.array(active_strings, dtype=float)

        num_active = np.sum(active_strings_np)
        if num_active == 0:
            raise ValueError("No active battery strings (num_active == 0).")

        if feeder_mode == 'independent':
            # ---- NEW: support case nm > n_str where each string feeds two motors (nm == 2*n_str)
            active_mask = active_strings_np
            if n_str == nm:
                # 1-to-1 mapping as before
                p_train_elec_req = p_train_elec * active_mask[:, np.newaxis]
            elif nm == 2 * n_str:
                # Each string i feeds motors [2*i, 2*i+1]
                p_train_elec_req = np.zeros((n_str, nn))
                for i in range(n_str):
                    motors_slice = slice(2 * i, 2 * i + 2)
                    # sum motor powers for the pair and apply active mask for that string
                    p_train_elec_req[i, :] = np.sum(p_train_elec[motors_slice, :], axis=0) * active_mask[i]
            else:
                # fallback (unchanged): map first min(n_str,nm) motors to strings 0..m-1
                m = min(n_str, nm)
                p_train_elec_req = np.zeros((n_str, nn))
                p_train_elec_req[:m, :] = p_train_elec[:m, :] * active_mask[:m, np.newaxis]

        elif feeder_mode == 'full_crossfeed':
            # total power per time column divided across active strings equally
            p_train_total = np.sum(p_train_elec, axis=0)   # shape (nn,)
            # each active string gets p_train_total / num_active
            active_mask = active_strings_np[:, np.newaxis]  # (n_str,1)
            p_train_elec_req = active_mask * (p_train_total[np.newaxis, :] / num_active)

        elif feeder_mode == 'grouped':
            # Two independent groups (first half strings handle first half motors, etc.)
            group_size = int(n_str // 2)
            if group_size == 0:
                raise ValueError("n_str must be >= 2 for grouped mode.")

            # safe guards - identify motor groups consistent with original code
            # group1 motors: 0:group_size, group2 motors: group_size:n_str (assumes nm >= n_str)
            # If nm < n_str this will still slice appropriately.
            grp1_m_inds = np.arange(0, min(group_size, nm))
            grp2_m_inds = np.arange(group_size, min(n_str, nm))

            grp1_pow_mot_total = np.sum(p_train_elec[grp1_m_inds, :], axis=0) if grp1_m_inds.size > 0 else np.zeros(nn)
            grp2_pow_mot_total = np.sum(p_train_elec[grp2_m_inds, :], axis=0) if grp2_m_inds.size > 0 else np.zeros(nn)

            num_active_grp1 = np.sum(active_strings_np[0:group_size])
            num_active_grp2 = np.sum(active_strings_np[group_size:n_str])

            if num_active_grp1 == 0 and grp1_m_inds.size > 0:
                raise ValueError("No active strings in group 1.")
            if num_active_grp2 == 0 and grp2_m_inds.size > 0:
                raise ValueError("No active strings in group 2.")

            grp1_pow_bat = grp1_pow_mot_total / (num_active_grp1 if num_active_grp1 > 0 else 1.0)
            grp2_pow_bat = grp2_pow_mot_total / (num_active_grp2 if num_active_grp2 > 0 else 1.0)

            active_mask_grp1 = active_strings_np[0:group_size][:, np.newaxis] if group_size > 0 else np.zeros((0, 1))
            active_mask_grp2 = active_strings_np[group_size:n_str][:, np.newaxis] if (n_str - group_size) > 0 else np.zeros((0, 1))

            p_train_elec_req_grp1 = active_mask_grp1 * grp1_pow_bat[np.newaxis, :] if grp1_m_inds.size > 0 else np.zeros((group_size, nn))
            p_train_elec_req_grp2 = active_mask_grp2 * grp2_pow_bat[np.newaxis, :] if grp2_m_inds.size > 0 else np.zeros((n_str - group_size, nn))

            p_train_elec_req = np.vstack((p_train_elec_req_grp1, p_train_elec_req_grp2))

        else:
            raise ValueError(f"Unknown feeder_mode: {feeder_mode}")

        outputs['p_train_elec_req'] = p_train_elec_req

    def compute_partials(self, inputs, partials):
        """
        Build dense Jacobian of shape (n_str * nn, nm * nn) using row-major flattening:
            out_idx = str_idx * nn + time_idx
            in_idx  = mot_idx * nn + time_idx
        Nonzero pattern mirrors compute().
        """
        nn = self.options['num_nodes']
        nm = self.options['nm']
        n_str = self.options['n_str']
        feeder_mode = self.options['feeder_mode']
        active_strings = self.options['active_strings']

        if active_strings is None:
            active_strings_np = np.ones(n_str, dtype=float)
        else:
            active_strings_np = np.array(active_strings, dtype=float)

        num_active = np.sum(active_strings_np)
        if num_active == 0:
            raise ValueError("No active battery strings (num_active == 0).")

        # Pre-allocate dense jacobian
        n_out = n_str * nn
        n_in = nm * nn
        J = np.zeros((n_out, n_in), dtype=float)

        if feeder_mode == 'independent':
            # ---- UPDATED to support motor-pairs when nm == 2*n_str
            if n_str == nm:
                # 1-to-1 mapping
                m = n_str
                for i in range(m):
                    a = active_strings_np[i]
                    if a == 0.0:
                        continue
                    for t in range(nn):
                        out_idx = i * nn + t
                        in_idx = i * nn + t
                        J[out_idx, in_idx] = a

            elif nm == 2 * n_str:
                # each string i depends on motor 2*i and 2*i+1 (both at same time t)
                for i in range(n_str):
                    a = active_strings_np[i]
                    if a == 0.0:
                        continue
                    motor0 = 2 * i
                    motor1 = 2 * i + 1
                    for t in range(nn):
                        out_idx = i * nn + t
                        if motor0 < nm:
                            in_idx0 = motor0 * nn + t
                            J[out_idx, in_idx0] = a
                        if motor1 < nm:
                            in_idx1 = motor1 * nn + t
                            J[out_idx, in_idx1] = a

            else:
                # fallback mapping first min(n_str,nm) motors to strings
                m = min(n_str, nm)
                for i in range(m):
                    a = active_strings_np[i]
                    if a == 0.0:
                        continue
                    for t in range(nn):
                        out_idx = i * nn + t
                        in_idx = i * nn + t
                        J[out_idx, in_idx] = a


        elif feeder_mode == 'full_crossfeed':
            # For each output (string i, time t), derivative wrt any motor k at same time t:
            # d(out_{i,t})/d(in_{k,t}) = active_mask[i] / num_active
            for i in range(n_str):
                a = active_strings_np[i]
                if a == 0.0:
                    continue
                coef = a / num_active
                for k in range(nm):
                    for t in range(nn):
                        out_idx = i * nn + t
                        in_idx = k * nn + t
                        J[out_idx, in_idx] = coef

        elif feeder_mode == 'grouped':
            group_size = int(n_str // 2)
            if group_size == 0:
                raise ValueError("n_str must be >= 2 for grouped mode.")

            # Group 1: strings 0:group_size map to motors 0:group_size
            # Group 2: strings group_size:n_str map to motors group_size:n_str
            grp1_m_end = min(group_size, nm)
            grp2_m_start = group_size
            grp2_m_end = min(n_str, nm)

            num_active_grp1 = np.sum(active_strings_np[0:group_size])
            num_active_grp2 = np.sum(active_strings_np[group_size:n_str])

            # grp1
            if grp1_m_end > 0 and num_active_grp1 > 0:
                coef1 = 1.0 / num_active_grp1
                for i in range(0, group_size):
                    a = active_strings_np[i]
                    if a == 0.0:
                        continue
                    for k in range(0, grp1_m_end):   # motors considered in grp1
                        for t in range(nn):
                            out_idx = i * nn + t
                            in_idx = k * nn + t
                            J[out_idx, in_idx] = a * coef1

            # grp2
            if (grp2_m_end - grp2_m_start) > 0 and num_active_grp2 > 0:
                coef2 = 1.0 / num_active_grp2
                for i_rel, i in enumerate(range(group_size, n_str)):
                    a = active_strings_np[i]
                    if a == 0.0:
                        continue
                    for k in range(grp2_m_start, grp2_m_end):   # motors considered in grp2
                        for t in range(nn):
                            out_idx = i * nn + t
                            in_idx = k * nn + t
                            J[out_idx, in_idx] = a * coef2

        else:
            raise ValueError(f"Unknown feeder_mode: {feeder_mode}")

        # assign dense Jacobian to partials dict
        partials['p_train_elec_req', 'p_train_elec'] = J


def test_demand_feed_independent():
    """Test DemandFeed component in independent mode."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Independent Mode")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],  # Motor 0
        [200, 210, 220, 230, 240],  # Motor 1
        [150, 160, 170, 180, 190],  # Motor 2
        [250, 260, 270, 280, 290],  # Motor 3
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, feeder_mode='independent'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: Each string matches its corresponding motor (1-to-1 mapping)")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Independent Mode test passed")


def test_demand_feed_independent_with_failure():
    """Test DemandFeed component in independent mode with one string failed."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Independent Mode with String Failure")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    # String 2 is failed (inactive)
    active_strings = [1, 1, 0, 1]
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [200, 210, 220, 230, 240],
        [150, 160, 170, 180, 190],
        [250, 260, 270, 280, 290],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='independent',
                                       active_strings=active_strings),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    print(f"Active strings: {active_strings}")
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: String 2 power is zero (failed)")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Independent Mode with Failure test passed")


def test_demand_feed_full_crossfeed():
    """Test DemandFeed component in full crossfeed mode."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Full Crossfeed Mode")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [200, 210, 220, 230, 240],
        [150, 160, 170, 180, 190],
        [250, 260, 270, 280, 290],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='full_crossfeed'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    total_power = np.sum(motor_powers, axis=0)
    expected_per_string = total_power / n_str
    
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Total motor power per time: {total_power}")
    print(f"Expected per string: {expected_per_string}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: All strings share total power equally")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Full Crossfeed Mode test passed")


def test_demand_feed_full_crossfeed_with_failure():
    """Test DemandFeed component in full crossfeed mode with string failure."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Full Crossfeed Mode with String Failure")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    # String 1 is failed
    active_strings = [1, 0, 1, 1]
    num_active = 3
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [200, 210, 220, 230, 240],
        [150, 160, 170, 180, 190],
        [250, 260, 270, 280, 290],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='full_crossfeed',
                                       active_strings=active_strings),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    total_power = np.sum(motor_powers, axis=0)
    expected_per_active_string = total_power / num_active
    
    print(f"Active strings: {active_strings}")
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Total motor power per time: {total_power}")
    print(f"Expected per active string: {expected_per_active_string}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: Active strings share equally, string 1 is zero")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Full Crossfeed Mode with Failure test passed")


def test_demand_feed_grouped():
    """Test DemandFeed component in grouped mode."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Grouped Mode")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],  # Motor 0 - Group 1
        [200, 210, 220, 230, 240],  # Motor 1 - Group 1
        [150, 160, 170, 180, 190],  # Motor 2 - Group 2
        [250, 260, 270, 280, 290],  # Motor 3 - Group 2
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='grouped'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    group1_total = np.sum(motor_powers[0:2, :], axis=0)
    group2_total = np.sum(motor_powers[2:4, :], axis=0)
    expected_grp1_per_string = group1_total / 2
    expected_grp2_per_string = group2_total / 2
    
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Group 1 total (motors 0-1): {group1_total}")
    print(f"Group 2 total (motors 2-3): {group2_total}")
    print(f"Expected Group 1 per string: {expected_grp1_per_string}")
    print(f"Expected Group 2 per string: {expected_grp2_per_string}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: Strings 0-1 share Group 1 power, Strings 2-3 share Group 2 power")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Grouped Mode test passed")


def test_demand_feed_grouped_with_failure():
    """Test DemandFeed component in grouped mode with string failure."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Grouped Mode with String Failure")
    print("="*60)
    
    nn = 5
    nm = 4
    n_str = 4
    
    # String 0 in group 1 failed, string 3 in group 2 failed
    active_strings = [0, 1, 1, 0]
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [200, 210, 220, 230, 240],
        [150, 160, 170, 180, 190],
        [250, 260, 270, 280, 290],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='grouped',
                                       active_strings=active_strings),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    print(f"Active strings: {active_strings}")
    print(f"Motor powers:\n{prob.get_val('p_train_elec')}")
    print(f"Battery string powers:\n{prob.get_val('p_train_elec_req')}")
    print("Expected: String 0 = 0, String 1 gets all Group 1 power")
    print("          String 2 gets all Group 2 power, String 3 = 0")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Grouped Mode with Failure test passed")


def test_demand_feed_independent_motor_pairs():
    """Test DemandFeed component in independent mode with motor pairs (nm = 2*n_str)."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Independent Mode with Motor Pairs")
    print("="*60)
    
    nn = 5
    nm = 8  # 8 motors
    n_str = 4  # 4 battery strings
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],  # Motor 0 → String 0
        [105, 115, 125, 135, 145],  # Motor 1 → String 0
        [200, 210, 220, 230, 240],  # Motor 2 → String 1
        [205, 215, 225, 235, 245],  # Motor 3 → String 1
        [150, 160, 170, 180, 190],  # Motor 4 → String 2
        [155, 165, 175, 185, 195],  # Motor 5 → String 2
        [250, 260, 270, 280, 290],  # Motor 6 → String 3
        [255, 265, 275, 285, 295],  # Motor 7 → String 3
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, feeder_mode='independent'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    print(f"Motor powers (8 motors):\n{prob.get_val('p_train_elec')}")
    print(f"\nBattery string powers (4 strings):\n{prob.get_val('p_train_elec_req')}")
    
    # Calculate expected values
    expected = np.array([
        motor_powers[0, :] + motor_powers[1, :],  # String 0 = Motor 0 + Motor 1
        motor_powers[2, :] + motor_powers[3, :],  # String 1 = Motor 2 + Motor 3
        motor_powers[4, :] + motor_powers[5, :],  # String 2 = Motor 4 + Motor 5
        motor_powers[6, :] + motor_powers[7, :],  # String 3 = Motor 6 + Motor 7
    ])
    print(f"\nExpected string powers:\n{expected}")
    print("Expected: Each string gets sum of its two motors")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Independent Mode with Motor Pairs test passed")


def test_demand_feed_full_crossfeed_motor_pairs():
    """Test DemandFeed component in full crossfeed mode with motor pairs (nm = 2*n_str)."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Full Crossfeed Mode with Motor Pairs")
    print("="*60)
    
    nn = 5
    nm = 8  # 8 motors
    n_str = 4  # 4 battery strings
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [105, 115, 125, 135, 145],
        [200, 210, 220, 230, 240],
        [205, 215, 225, 235, 245],
        [150, 160, 170, 180, 190],
        [155, 165, 175, 185, 195],
        [250, 260, 270, 280, 290],
        [255, 265, 275, 285, 295],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='full_crossfeed'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    total_power = np.sum(motor_powers, axis=0)
    expected_per_string = total_power / n_str
    
    print(f"Motor powers (8 motors):\n{prob.get_val('p_train_elec')}")
    print(f"\nTotal motor power per time: {total_power}")
    print(f"Expected per string: {expected_per_string}")
    print(f"\nBattery string powers (4 strings):\n{prob.get_val('p_train_elec_req')}")
    print("Expected: All 8 motors' power shared equally across 4 strings")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Full Crossfeed Mode with Motor Pairs test passed")


def test_demand_feed_grouped_motor_pairs():
    """Test DemandFeed component in grouped mode with motor pairs (nm = 2*n_str)."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Grouped Mode with Motor Pairs")
    print("="*60)
    
    nn = 5
    nm = 8  # 8 motors
    n_str = 4  # 4 battery strings
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],  # Motor 0 - Group 1
        [105, 115, 125, 135, 145],  # Motor 1 - Group 1
        [200, 210, 220, 230, 240],  # Motor 2 - Group 1
        [205, 215, 225, 235, 245],  # Motor 3 - Group 1
        [150, 160, 170, 180, 190],  # Motor 4 - Group 2
        [155, 165, 175, 185, 195],  # Motor 5 - Group 2
        [250, 260, 270, 280, 290],  # Motor 6 - Group 2
        [255, 265, 275, 285, 295],  # Motor 7 - Group 2
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='grouped'),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    # Group 1: motors 0-1 (first group_size motors) → strings 0-1
    # Group 2: motors 2-3 (next group_size motors) → strings 2-3
    # Note: grouped mode uses first group_size motors, not all motors
    group_size = n_str // 2  # = 2
    group1_total = np.sum(motor_powers[0:group_size, :], axis=0)  # motors 0-1
    group2_total = np.sum(motor_powers[group_size:n_str, :], axis=0)  # motors 2-3
    expected_grp1_per_string = group1_total / 2
    expected_grp2_per_string = group2_total / 2
    
    print(f"Motor powers (8 motors):\n{prob.get_val('p_train_elec')}")
    print(f"\nGroup 1 total (motors 0-{group_size-1}): {group1_total}")
    print(f"Group 2 total (motors {group_size}-{n_str-1}): {group2_total}")
    print(f"Expected Group 1 per string: {expected_grp1_per_string}")
    print(f"Expected Group 2 per string: {expected_grp2_per_string}")
    print(f"\nBattery string powers (4 strings):\n{prob.get_val('p_train_elec_req')}")
    print(f"Expected: Strings 0-1 share Group 1 power (motors 0-{group_size-1})")
    print(f"          Strings 2-3 share Group 2 power (motors {group_size}-{n_str-1})")
    print(f"Note: Motors {n_str}-{nm-1} are not used in grouped mode")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Grouped Mode with Motor Pairs test passed")


def test_demand_feed_independent_motor_pairs_with_failure():
    """Test DemandFeed independent mode with motor pairs and string failure."""
    print("\n" + "="*60)
    print("Testing DemandFeed - Independent Mode Motor Pairs with Failure")
    print("="*60)
    
    nn = 5
    nm = 8
    n_str = 4
    
    # String 2 is failed (inactive)
    active_strings = [1, 1, 0, 1]
    
    prob = om.Problem(reports=False)
    
    # Add IVC for inputs
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    motor_powers = np.array([
        [100, 110, 120, 130, 140],
        [105, 115, 125, 135, 145],
        [200, 210, 220, 230, 240],
        [205, 215, 225, 235, 245],
        [150, 160, 170, 180, 190],  # These feed string 2 (failed)
        [155, 165, 175, 185, 195],  # These feed string 2 (failed)
        [250, 260, 270, 280, 290],
        [255, 265, 275, 285, 295],
    ])
    ivc.add_output('p_train_elec', val=motor_powers, units='kW')
    
    prob.model.add_subsystem('comp', 
                             DemandFeed(num_nodes=nn, nm=nm, n_str=n_str, 
                                       feeder_mode='independent',
                                       active_strings=active_strings),
                             promotes=['*'])
    prob.setup()
    
    prob.run_model()
    
    print(f"Active strings: {active_strings}")
    print(f"Motor powers (8 motors):\n{prob.get_val('p_train_elec')}")
    print(f"\nBattery string powers (4 strings):\n{prob.get_val('p_train_elec_req')}")
    print("Expected: String 2 power is zero (failed), motors 4-5 power is lost")
    
    # Check partials
    prob.check_partials(compact_print=True)
    print("✓ DemandFeed Independent Mode Motor Pairs with Failure test passed")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# RUNNING CROSS_FEED_LOGIC COMPONENT TESTS")
    print("#"*60)
    
    # Test standard configurations (nm = n_str)
    #test_demand_feed_independent()
    #test_demand_feed_independent_with_failure()
    #test_demand_feed_full_crossfeed()
    #test_demand_feed_full_crossfeed_with_failure()
    #test_demand_feed_grouped()
    #test_demand_feed_grouped_with_failure()
    
    # Test motor pair configurations (nm = 2*n_str)
    test_demand_feed_independent_motor_pairs()
    test_demand_feed_full_crossfeed_motor_pairs()
    test_demand_feed_grouped_motor_pairs()
    test_demand_feed_independent_motor_pairs_with_failure()
    
    print("\n" + "#"*60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*60)