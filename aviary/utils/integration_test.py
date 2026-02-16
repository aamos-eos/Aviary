import numpy as np
import openmdao.api as om


class SimpsonMatrixIntegral(om.ExplicitComponent):
    """
    Integrate each row of a matrix F with respect to time using Simpson's rule.
    
    For each row m of F(M, N), computes:
        Y[m, k] = integral from t[0] to t[k] of F[m, :]
    
    Uses Simpson's rule for even number of intervals, with trapezoidal rule
    for the last interval if there are an odd number of intervals.
    
    Assumes uniform (constant) time steps.

    Inputs
    ------
    F : (M, N) array
        Function values to integrate (M rows, N time points)
    time : (N,) array
        Time vector (must be uniformly spaced)
    
    Outputs
    -------
    Y : (M, N) array
        Cumulative integral of F with respect to time
        Y[m, 0] = 0 for all m
        Y[m, k] = integral from t[0] to t[k] of F[m, :]

    Options
    -------
    M : int
        Number of rows (trajectories) to integrate
    N : int
        Number of time points
    """

    def initialize(self):
        self.options.declare('M', types=int, desc='Number of rows (trajectories)')
        self.options.declare('N', types=int, desc='Number of time points')

    def setup(self):
        M = self.options['M']
        N = self.options['N']
        
        self.add_input('F', shape=(M, N), units='1/h')
        self.add_input('time', shape=(N,), units='h')
        self.add_output('Y', shape=(M, N), units=None)
        
        # Declare dense partials
        self.declare_partials('Y', 'F')
        self.declare_partials('Y', 'time')

    def compute(self, inputs, outputs):
        F = inputs['F']  # (M, N)
        time = inputs['time']  # (N,)
        M, N = F.shape
        
        Y = np.zeros((M, N))
        
        # For each row (trajectory)
        for m in range(M):
            # Integrate F[m, :] with respect to time
            Y[m, :] = self._simpson_integrate(F[m, :], time)
        
        outputs['Y'] = Y
    
    def _simpson_integrate(self, f, time):
        """
        Compute cumulative integral of f with respect to time using Simpson's rule.
        
        Assumes uniform (constant) time steps.
        - Even number of intervals: full Simpson's rule
        - Odd number of intervals: Simpson's on all but last, trapezoidal on last
        
        Parameters
        ----------
        f : (N,) array
            Function values
        time : (N,) array
            Time points (must be uniformly spaced)
        
        Returns
        -------
        integral : (N,) array
            Cumulative integral, with integral[0] = 0
        """
        N = len(f)
        integral = np.zeros(N)
        
        if N < 2:
            return integral
        
        # Track the start of the current segment for incremental integration
        # This resets at segment boundaries (duplicate time points)
        segment_start_idx = 0
        
        for k in range(1, N):
            # Check if time hasn't advanced (duplicate time point = segment boundary)
            dt_k = time[k] - time[k-1]
            if abs(dt_k) < 1e-12:  # Zero or near-zero time step
                # No time has elapsed, so integral doesn't change
                integral[k] = integral[k-1]
                # Reset segment start to this boundary point (start of new segment)
                segment_start_idx = k
                continue
            
            # Compute time step for the current segment (uniform within segment)
            # Use the first two points of the current segment to get h
            if segment_start_idx + 1 < N:
                h = time[segment_start_idx + 1] - time[segment_start_idx]
            else:
                # Fallback: use previous step size
                h = time[k] - time[k-1]
            
            # Use incremental integration: compute integral from segment_start_idx to k
            # This handles segments correctly by starting from the segment boundary
            n_points_in_range = k - segment_start_idx + 1  # Number of points from segment_start to k (inclusive)
            
            if n_points_in_range == 2:
                # Single interval: trapezoidal rule
                integral[k] = integral[segment_start_idx] + 0.5 * h * (f[segment_start_idx] + f[k])
            elif n_points_in_range >= 3:
                # Multiple intervals: use Simpson's rule on the range [segment_start_idx, k]
                # Number of intervals in this range
                n_intervals = n_points_in_range - 1
                
                if n_intervals % 2 == 0:
                    # Even number of intervals: full Simpson's rule
                    result = f[segment_start_idx]
                    for i in range(segment_start_idx + 1, k):
                        rel_idx = i - segment_start_idx
                        if rel_idx % 2 == 1:
                            result += 4.0 * f[i]
                        else:
                            result += 2.0 * f[i]
                    result += f[k]
                    increment = (h / 3.0) * result
                    integral[k] = integral[segment_start_idx] + increment
                else:
                    # Odd number of intervals (>1): Simpson's on first (n-1), trapezoidal on last
                    if n_intervals >= 3:
                        # Simpson's on [segment_start_idx, k-1]
                        simpson_result = f[segment_start_idx]
                        for i in range(segment_start_idx + 1, k-1):
                            rel_idx = i - segment_start_idx
                            if rel_idx % 2 == 1:
                                simpson_result += 4.0 * f[i]
                            else:
                                simpson_result += 2.0 * f[i]
                        simpson_result += f[k-1]
                        simpson_part = (h / 3.0) * simpson_result
                    else:
                        simpson_part = 0.0
                    
                    # Trapezoidal on last interval [k-1, k]
                    trap_part = 0.5 * h * (f[k-1] + f[k])
                    integral[k] = integral[segment_start_idx] + simpson_part + trap_part
            else:
                # Shouldn't happen, but handle gracefully
                integral[k] = integral[k-1]
        
        return integral

    def compute_partials(self, inputs, partials):
        """
        Compute partial derivatives dY/dF and dY/dtime.
        
        Matches the segment-based incremental integration logic:
        - Detects segment boundaries (duplicate time points)
        - Uses segment-specific step sizes
        - Computes incremental derivatives per segment
        """
        F = inputs['F']
        time = inputs['time']
        M, N = F.shape
        
        # Initialize partial derivative arrays
        dY_dF = np.zeros((M * N, M * N))
        dY_dtime = np.zeros((M * N, N))

        for m in range(M):
            # Track segment boundaries (matching the integration logic)
            segment_start_idx = 0
            
            for k in range(1, N):
                out_idx = m * N + k
                
                # Check if time hasn't advanced (duplicate time point = segment boundary)
                dt_k = time[k] - time[k-1]
                if abs(dt_k) < 1e-12:
                    # No time has elapsed, so integral doesn't change
                    # Y[k] = Y[k-1], so derivatives are the same
                    if k > 0:
                        prev_out_idx = m * N + (k-1)
                        # Copy derivatives from previous point
                        dY_dF[out_idx, :] = dY_dF[prev_out_idx, :]
                        dY_dtime[out_idx, :] = dY_dtime[prev_out_idx, :]
                    segment_start_idx = k
                    continue
                
                # Compute time step for the current segment
                if segment_start_idx + 1 < N:
                    h = time[segment_start_idx + 1] - time[segment_start_idx]
                else:
                    h = time[k] - time[k-1]
                
                # Number of points in current segment range
                n_points_in_range = k - segment_start_idx + 1
                
                # Start with derivatives from segment_start (incremental integration)
                # Y[k] = Y[segment_start] + increment, so derivatives include previous contributions
                if segment_start_idx > 0:
                    prev_out_idx = m * N + segment_start_idx
                    dY_dF[out_idx, :] = dY_dF[prev_out_idx, :].copy()
                    dY_dtime[out_idx, :] = dY_dtime[prev_out_idx, :].copy()
                else:
                    # First segment: start with zeros (no previous contributions)
                    dY_dF[out_idx, :] = 0.0
                    dY_dtime[out_idx, :] = 0.0
                
                if n_points_in_range == 2:
                    # Single interval: trapezoidal rule
                    # Y[k] = Y[segment_start] + 0.5 * h * (F[segment_start] + F[k])
                    in_idx_start = m * N + segment_start_idx
                    in_idx_k = m * N + k
                    # Only add derivative for the new point k (F[segment_start] already accounted for in copy)
                    dY_dF[out_idx, in_idx_k] = 0.5 * h
                    # Also need to add the contribution from F[segment_start] in this increment
                    if segment_start_idx == 0:
                        # First segment: no previous copy, so add both
                        dY_dF[out_idx, in_idx_start] = 0.5 * h
                    else:
                        # Subsequent segments: F[segment_start] contributes to increment, add it
                        dY_dF[out_idx, in_idx_start] += 0.5 * h
                    
                    # dY/dh = 0.5 * (F[segment_start] + F[k])
                    # For trapezoidal: Y = 0.5 * h * (F[segment_start] + F[k])
                    # dY/dh = 0.5 * (F[segment_start] + F[k])
                    dY_dh = 0.5 * (F[m, segment_start_idx] + F[m, k])
                    # h = time[segment_start_idx + 1] - time[segment_start_idx]
                    # dh/dtime[segment_start_idx] = -1
                    # dh/dtime[segment_start_idx + 1] = +1
                    # dY/dtime = dY/dh * dh/dtime
                    # Add contribution from this segment's h (copied derivatives already include previous segments)
                    if segment_start_idx < N:
                        dY_dtime[out_idx, segment_start_idx] += -dY_dh
                    if segment_start_idx + 1 < N:
                        dY_dtime[out_idx, segment_start_idx + 1] += dY_dh
                    
                elif n_points_in_range >= 3:
                    n_intervals = n_points_in_range - 1
                    
                    if n_intervals % 2 == 0:
                        # Even number of intervals: full Simpson's rule
                        # Weights for Simpson's rule from segment_start to k
                        for i in range(segment_start_idx, k + 1):
                            in_idx = m * N + i
                            rel_idx = i - segment_start_idx
                            if rel_idx == 0 or rel_idx == n_intervals:
                                weight = h / 3.0
                            elif rel_idx % 2 == 1:
                                weight = 4.0 * h / 3.0
                            else:
                                weight = 2.0 * h / 3.0
                            dY_dF[out_idx, in_idx] += weight
                        
                        # dY/dh: derivative with respect to segment step size
                        # For Simpson's: Y = (h/3) * [F[0] + 4*F[1] + 2*F[2] + ... + F[n]]
                        # dY/dh = (1/3) * [F[0] + 4*F[1] + 2*F[2] + ... + F[n]]
                        simpson_sum = F[m, segment_start_idx]
                        for i in range(segment_start_idx + 1, k):
                            rel_idx = i - segment_start_idx
                            if rel_idx % 2 == 1:
                                simpson_sum += 4.0 * F[m, i]
                            else:
                                simpson_sum += 2.0 * F[m, i]
                        simpson_sum += F[m, k]
                        dY_dh = simpson_sum / 3.0
                        
                        # h = time[segment_start_idx + 1] - time[segment_start_idx]
                        # dh/dtime[segment_start_idx] = -1
                        # dh/dtime[segment_start_idx + 1] = +1
                        # dY/dtime = dY/dh * dh/dtime
                        # Add contribution from this segment's h (copied derivatives already include previous segments)
                        if segment_start_idx < N:
                            dY_dtime[out_idx, segment_start_idx] += -dY_dh
                        if segment_start_idx + 1 < N:
                            dY_dtime[out_idx, segment_start_idx + 1] += dY_dh
                        
                    else:
                        # Odd number of intervals: Simpson's on first (n-1), trapezoidal on last
                        # Simpson's part
                        if n_intervals >= 3:
                            for i in range(segment_start_idx, k):
                                in_idx = m * N + i
                                rel_idx = i - segment_start_idx
                                if rel_idx == 0 or rel_idx == n_intervals - 1:
                                    weight = h / 3.0
                                elif rel_idx % 2 == 1:
                                    weight = 4.0 * h / 3.0
                                else:
                                    weight = 2.0 * h / 3.0
                                dY_dF[out_idx, in_idx] += weight
                        
                        # Trapezoidal part on last interval
                        in_idx_km1 = m * N + (k-1)
                        in_idx_k = m * N + k
                        dY_dF[out_idx, in_idx_km1] += 0.5 * h
                        dY_dF[out_idx, in_idx_k] += 0.5 * h
                        
                        # dY/dtime: combine Simpson and trapezoidal parts
                        # Simpson part derivative
                        if n_intervals >= 3:
                            simpson_sum = F[m, segment_start_idx]
                            for i in range(segment_start_idx + 1, k-1):
                                rel_idx = i - segment_start_idx
                                if rel_idx % 2 == 1:
                                    simpson_sum += 4.0 * F[m, i]
                                else:
                                    simpson_sum += 2.0 * F[m, i]
                            simpson_sum += F[m, k-1]
                            dY_dh_simpson = simpson_sum / 3.0
                        else:
                            dY_dh_simpson = 0.0
                        
                        # Trapezoidal part derivative
                        dY_dh_trap = 0.5 * (F[m, k-1] + F[m, k])
                        
                        # Total dY/dh
                        dY_dh = dY_dh_simpson + dY_dh_trap
                        
                        # h = time[segment_start_idx + 1] - time[segment_start_idx]
                        # dh/dtime[segment_start_idx] = -1
                        # dh/dtime[segment_start_idx + 1] = +1
                        # dY/dtime = dY/dh * dh/dtime
                        # Add contribution from this segment's h (copied derivatives already include previous segments)
                        if segment_start_idx < N:
                            dY_dtime[out_idx, segment_start_idx] += -dY_dh
                        if segment_start_idx + 1 < N:
                            dY_dtime[out_idx, segment_start_idx + 1] += dY_dh
        
        partials['Y', 'F'] = dY_dF
        partials['Y', 'time'] = dY_dtime


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("Simpson's Rule Integration Test")
    print("=" * 70)
    
    # Example: Integrate velocity with respect to time
    # Scenario: Constant acceleration a = 2 m/s^2, initial velocity v0 = 10 m/s
    # Velocity: v(t) = v0 + a*t = 10 + 2*t
    # Position (analytical): x(t) = x0 + v0*t + 0.5*a*t^2 = 0 + 10*t + t^2
    
    # Time vector
    t_total = 10.0  # seconds
    N_points = 101
    time = np.linspace(0, t_total, N_points)
    
    # Velocity profile (M=1 for single trajectory)
    v0 = 10.0  # m/s
    a = 2.0    # m/s^2
    velocity = v0 + a * time  # (N_points,)
    
    # Analytical position solution
    x0 = 0.0
    position_analytical = x0 + v0 * time + 0.5 * a * time**2
    
    # Setup integration component
    comp = SimpsonMatrixIntegral(M=1, N=N_points)
    prob = om.Problem(reports=False)
    prob.model.add_subsystem('int_vel', comp, promotes=['*'])
    prob.setup()

    # Set inputs: velocity as (M=1, N=N_points) matrix
    prob.set_val('F', velocity.reshape(1, -1))
    prob.set_val('time', time)
    
    # Run integration
    prob.run_model()
    position_numerical = prob.get_val('Y')[0, :]  # Extract first (and only) row
    
    # Compute error
    error = np.abs(position_numerical - position_analytical)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"Time range: 0 to {t_total} seconds")
    print(f"Number of points: {N_points}")
    print(f"Time step: {time[1] - time[0]:.4f} seconds")
    print(f"Initial velocity: {v0} m/s")
    print(f"Acceleration: {a} m/s^2")
    print(f"\nResults:")
    print(f"  Final position (numerical): {position_numerical[-1]:.6f} m")
    print(f"  Final position (analytical): {position_analytical[-1]:.6f} m")
    print(f"\nError analysis:")
    print(f"  Maximum error: {max_error:.2e} m")
    print(f"  RMS error: {rms_error:.2e} m")
    
    # Check partials
    print("\nChecking partial derivatives...")
    prob.check_partials(compact_print=True)

    # Plot results
    try:
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Velocity vs Time
        plt.subplot(2, 3, 1)
        plt.plot(time, velocity, 'b-', linewidth=2, label='Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Position comparison
        plt.subplot(2, 3, 2)
        plt.plot(time, position_analytical, 'r-', linewidth=2, label='Analytical', linestyle='--')
        plt.plot(time, position_numerical, 'b-', linewidth=2, label='Simpson Integration', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position: Analytical vs Numerical')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Error
        plt.subplot(2, 3, 3)
        plt.semilogy(time, error, 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title('Integration Error')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative integral visualization
        plt.subplot(2, 3, 4)
        plt.plot(time, position_numerical, 'b-', linewidth=2, label='Position (integrated)')
        plt.fill_between(time, 0, position_numerical, alpha=0.3, color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Cumulative Integral of Velocity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 5: Error distribution
        plt.subplot(2, 3, 5)
        plt.plot(time[1:], error[1:], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.title('Error vs Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Relative error
        plt.subplot(2, 3, 6)
        relative_error = error / (np.abs(position_analytical) + 1e-10) * 100
        plt.semilogy(time[1:], relative_error[1:], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('velocity_integration_example.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: velocity_integration_example.png")
        plt.show()
        plt.close()
    except Exception as e:
        print(f"\nNote: Could not generate plot ({e})")
    
    print("\n" + "=" * 70)
    print("Example 2: Multiple trajectories")
    print("=" * 70)
    
    # Integrate multiple velocity profiles
    M = 3
    N = 51
    time2 = np.linspace(0, 5.0, N)
    
    # Different accelerations for each trajectory
    velocities = np.zeros((M, N))
    velocities[0, :] = 10.0 + 2.0 * time2  # a = 2 m/s^2
    velocities[1, :] = 15.0 + 1.0 * time2  # a = 1 m/s^2
    velocities[2, :] = 5.0 + 3.0 * time2   # a = 3 m/s^2
    
    comp2 = SimpsonMatrixIntegral(M=M, N=N)
    prob2 = om.Problem(reports=False)
    prob2.model.add_subsystem('int_vel2', comp2, promotes=['*'])
    prob2.setup()
    
    prob2.set_val('F', velocities)
    prob2.set_val('time', time2)
    
    prob2.run_model()
    positions = prob2.get_val('Y')
    
    print(f"Integrated {M} velocity profiles")
    print(f"Final positions:")
    for m in range(M):
        print(f"  Trajectory {m+1}: {positions[m, -1]:.6f} m")
    
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for m in range(M):
            plt.plot(time2, velocities[m, :], linewidth=2, label=f'Trajectory {m+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Multiple Velocity Profiles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for m in range(M):
            plt.plot(time2, positions[m, :], linewidth=2, label=f'Trajectory {m+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Integrated Positions')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('velocity_integration_multiple.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: velocity_integration_multiple.png")
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Note: Could not generate plot ({e})")
    
    print("\n" + "=" * 70)
    print("Example 3: Non-monotonic time vector (concatenated segments)")
    print("=" * 70)
    
    # Create time vector from concatenated segments
    # Segment 1: [1, 2, 3]
    # Segment 2: [3, 4, 5] (starts where segment 1 ends)
    # Concatenated: [1, 2, 3, 3, 4, 5] - non-monotonic due to duplicate at boundary
    
    time_seg1 = np.array([1.0, 2.0, 3.0])
    time_seg2 = np.array([3.0, 4.0, 5.0])  # Starts at 3, same as end of seg1
    time_concat = np.concatenate([time_seg1, time_seg2])
    
    # Check monotonicity
    is_monotonic = np.all(np.diff(time_concat) >= 0)
    has_duplicates = len(time_concat) != len(np.unique(time_concat))
    
    print(f"Time segments:")
    print(f"  Segment 1: {time_seg1}")
    print(f"  Segment 2: {time_seg2}")
    print(f"  Concatenated: {time_concat}")
    print(f"  Is monotonic: {is_monotonic}")
    print(f"  Has duplicates: {has_duplicates}")
    print(f"  Time differences: {np.diff(time_concat)}")
    
    # Create velocity profile (same length as time)
    # Use a simple linear function: v(t) = 10 + 2*t
    v0 = 10.0
    a = 2.0
    velocity_concat = v0 + a * time_concat
    
    # Analytical position (integral of velocity)
    # x(t) = x0 + v0*t + 0.5*a*t^2
    x0 = 0.0
    position_analytical_concat = x0 + v0 * time_concat + 0.5 * a * time_concat**2
    
    # Setup integration component
    N_concat = len(time_concat)
    M_concat = 1
    comp3 = SimpsonMatrixIntegral(M=M_concat, N=N_concat)
    prob3 = om.Problem(reports=False)
    prob3.model.add_subsystem('int_concat', comp3, promotes=['*'])
    prob3.setup()
    
    # Set inputs
    prob3.set_val('F', velocity_concat.reshape(1, -1))
    prob3.set_val('time', time_concat)
    
    # Run integration
    print("\nRunning integration with non-monotonic time vector...")
    try:
        prob3.run_model()
        position_numerical_concat = prob3.get_val('Y')[0, :]
        
        # Compute error
        error_concat = np.abs(position_numerical_concat - position_analytical_concat)
        max_error_concat = np.max(error_concat)
        rms_error_concat = np.sqrt(np.mean(error_concat**2))
        
        print(f"\nResults:")
        print(f"  Final position (numerical): {position_numerical_concat[-1]:.6f} m")
        print(f"  Final position (analytical): {position_analytical_concat[-1]:.6f} m")
        print(f"\nError analysis:")
        print(f"  Maximum error: {max_error_concat:.2e} m")
        print(f"  RMS error: {rms_error_concat:.2e} m")
        
        # Check what happened at the duplicate time point
        duplicate_idx = 2  # Index where time = 3 appears twice
        print(f"\nAt duplicate time point (t={time_concat[duplicate_idx]:.1f}):")
        print(f"  Position at first occurrence: {position_numerical_concat[duplicate_idx]:.6f} m")
        print(f"  Position at second occurrence: {position_numerical_concat[duplicate_idx+1]:.6f} m")
        print(f"  Position difference: {position_numerical_concat[duplicate_idx+1] - position_numerical_concat[duplicate_idx]:.6f} m")
        
        # Check partials (may fail due to non-uniform time steps)
        print("\nChecking partial derivatives (may have issues with non-uniform time steps)...")
        try:
            prob3.check_partials(compact_print=True)
        except Exception as e:
            print(f"  Partial check failed: {e}")
            print("  This is expected because the component assumes uniform time steps.")
        
        # Plot results
        try:
            plt.figure(figsize=(14, 10))
            
            # Plot 1: Time vector showing duplicates
            plt.subplot(2, 3, 1)
            plt.plot(time_concat, 'o-', linewidth=2, markersize=8, label='Time vector')
            plt.axvline(duplicate_idx, color='r', linestyle='--', alpha=0.5, label='Duplicate at boundary')
            plt.axvline(duplicate_idx+1, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Index')
            plt.ylabel('Time (s)')
            plt.title('Non-monotonic Time Vector')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Time differences
            plt.subplot(2, 3, 2)
            dt_concat = np.diff(time_concat)
            plt.plot(dt_concat, 'o-', linewidth=2, markersize=8)
            plt.axhline(0, color='k', linestyle='-', alpha=0.3)
            plt.xlabel('Index')
            plt.ylabel('Time Step (s)')
            plt.title('Time Step Differences')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Velocity vs Time
            plt.subplot(2, 3, 3)
            plt.plot(time_concat, velocity_concat, 'b-o', linewidth=2, markersize=6, label='Velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (m/s)')
            plt.title('Velocity Profile')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 4: Position comparison
            plt.subplot(2, 3, 4)
            plt.plot(time_concat, position_analytical_concat, 'r-o', linewidth=2, 
                    markersize=6, label='Analytical', linestyle='--', alpha=0.7)
            plt.plot(time_concat, position_numerical_concat, 'b-o', linewidth=2, 
                    markersize=6, label='Simpson Integration', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Position: Analytical vs Numerical')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 5: Error
            plt.subplot(2, 3, 5)
            plt.plot(time_concat[1:], error_concat[1:], 'g-o', linewidth=2, markersize=6)
            plt.xlabel('Time (s)')
            plt.ylabel('Absolute Error (m)')
            plt.title('Integration Error')
            plt.grid(True, alpha=0.3)
            
            # Plot 6: Position at duplicate time point
            plt.subplot(2, 3, 6)
            duplicate_range = range(max(0, duplicate_idx-1), min(N_concat, duplicate_idx+3))
            plt.plot(time_concat[duplicate_range], position_numerical_concat[duplicate_range], 
                    'b-o', linewidth=2, markersize=8, label='Numerical')
            plt.plot(time_concat[duplicate_range], position_analytical_concat[duplicate_range], 
                    'r--o', linewidth=2, markersize=8, label='Analytical', alpha=0.7)
            plt.axvline(time_concat[duplicate_idx], color='r', linestyle='--', alpha=0.5, label='Duplicate time')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Behavior at Duplicate Time Point')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('velocity_integration_nonmonotonic.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: velocity_integration_nonmonotonic.png")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"\nNote: Could not generate plot ({e})")
            
    except Exception as e:
        print(f"\nIntegration failed: {e}")
        print("This is expected because the component assumes uniform time steps,")
        print("but the concatenated time vector has non-uniform steps.")
    
    print("\n" + "=" * 70)
    print("Example 4: 4 segments, each length 11 (concatenated integration accuracy test)")
    print("=" * 70)
    
    # Create 4 segments, each with 11 time points (same number of nodes)
    # Each segment will be concatenated with shared boundaries
    # Each segment has different durations (different time steps)
    n_segments = 4
    n_points_per_segment = 11
    
    # Create segments with different durations (different time step sizes)
    segments = []
    t_start = 0.0
    durations = [5.0, 3.0, 7.0, 4.0]  # Different durations for each segment
    
    for i in range(n_segments):
        t_end = t_start + durations[i]
        time_seg = np.linspace(t_start, t_end, n_points_per_segment)
        segments.append(time_seg)
        t_start = t_end  # Next segment starts where this one ends (duplicate boundary)
    
    # Concatenate all segments
    time_4seg = np.concatenate(segments)
    
    # Check monotonicity and duplicates
    is_monotonic = np.all(np.diff(time_4seg) >= 0)
    has_duplicates = len(time_4seg) != len(np.unique(time_4seg))
    n_duplicates = len(time_4seg) - len(np.unique(time_4seg))
    
    print(f"Segments:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: {seg[0]:.2f} to {seg[-1]:.2f} s ({len(seg)} points)")
    print(f"\nConcatenated time vector:")
    print(f"  Total length: {len(time_4seg)} points")
    print(f"  Time range: {time_4seg[0]:.2f} to {time_4seg[-1]:.2f} s")
    print(f"  Is monotonic: {is_monotonic}")
    print(f"  Has duplicates: {has_duplicates} ({n_duplicates} duplicate points)")
    print(f"  Time step sizes by segment:")
    for i, seg in enumerate(segments):
        dt_seg = np.diff(seg)
        duration = seg[-1] - seg[0]
        print(f"    Segment {i+1}: duration = {duration:.2f} s, dt = {dt_seg[0]:.4f} s (uniform within segment)")
    
    # Create velocity profile (same length as time)
    # Use a simple linear function: v(t) = 10 + 2*t
    v0 = 10.0
    a = 2.0
    velocity_4seg = v0 + a * time_4seg
    
    # Analytical position (integral of velocity)
    # x(t) = x0 + v0*t + 0.5*a*t^2
    x0 = 0.0
    position_analytical_4seg = x0 + v0 * time_4seg + 0.5 * a * time_4seg**2
    
    # Setup integration component
    N_4seg = len(time_4seg)
    M_4seg = 1
    comp4 = SimpsonMatrixIntegral(M=M_4seg, N=N_4seg)
    prob4 = om.Problem(reports=False)
    prob4.model.add_subsystem('int_4seg', comp4, promotes=['*'])
    prob4.setup()
    
    # Set inputs
    prob4.set_val('F', velocity_4seg.reshape(1, -1))
    prob4.set_val('time', time_4seg)
    
    # Run integration
    print("\nRunning integration with 4 concatenated segments...")
    try:
        prob4.run_model()
        position_numerical_4seg = prob4.get_val('Y')[0, :]
        
        # Compute error
        error_4seg = np.abs(position_numerical_4seg - position_analytical_4seg)
        max_error_4seg = np.max(error_4seg)
        rms_error_4seg = np.sqrt(np.mean(error_4seg**2))
        
        # Compute error at segment boundaries (duplicate points)
        boundary_indices = []
        for i in range(1, len(time_4seg)):
            if abs(time_4seg[i] - time_4seg[i-1]) < 1e-12:
                boundary_indices.append(i)
        
        print(f"\nResults:")
        print(f"  Final position (numerical): {position_numerical_4seg[-1]:.6f} m")
        print(f"  Final position (analytical): {position_analytical_4seg[-1]:.6f} m")
        print(f"\nError analysis:")
        print(f"  Maximum error: {max_error_4seg:.2e} m")
        print(f"  RMS error: {rms_error_4seg:.2e} m")
        
        if boundary_indices:
            print(f"\nAt segment boundaries (duplicate time points):")
            for idx in boundary_indices:
                if idx > 0:
                    pos_diff = position_numerical_4seg[idx] - position_numerical_4seg[idx-1]
                    print(f"  Boundary at index {idx} (t={time_4seg[idx]:.2f} s):")
                    print(f"    Position difference: {pos_diff:.6e} m (should be ~0)")
        
        # Check accuracy at end of each segment
        print(f"\nPosition at end of each segment:")
        segment_ends = [n_points_per_segment - 1]
        for i in range(1, n_segments):
            segment_ends.append(segment_ends[-1] + n_points_per_segment)
        
        for i, end_idx in enumerate(segment_ends):
            if end_idx < len(time_4seg):
                t_end = time_4seg[end_idx]
                pos_num = position_numerical_4seg[end_idx]
                pos_ana = position_analytical_4seg[end_idx]
                err = abs(pos_num - pos_ana)
                print(f"  Segment {i+1} end (t={t_end:.2f} s):")
                print(f"    Numerical: {pos_num:.6f} m")
                print(f"    Analytical: {pos_ana:.6f} m")
                print(f"    Error: {err:.2e} m")
        
        # Check partials (may have issues with non-uniform time steps)
        print("\nChecking partial derivatives...")
        try:
            prob4.check_partials(compact_print=True)
        except Exception as e:
            print(f"  Partial check failed: {e}")
            print("  This is expected because the component assumes uniform time steps.")
        
        # Plot results
        try:
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Time vector showing segments
            plt.subplot(3, 3, 1)
            colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
            for i, seg in enumerate(segments):
                seg_indices = np.arange(len(seg)) + i * (n_points_per_segment - (1 if i > 0 else 0))
                if i > 0:
                    # Account for duplicate boundary
                    seg_indices = np.arange(len(seg)) + (i * n_points_per_segment - i)
                    actual_indices = seg_indices[seg_indices < len(time_4seg)]
                    actual_seg = time_4seg[actual_indices]
                else:
                    actual_indices = seg_indices
                    actual_seg = seg
                plt.plot(actual_indices, actual_seg, 'o-', linewidth=2, 
                        markersize=4, color=colors[i], label=f'Segment {i+1}', alpha=0.7)
            plt.xlabel('Index')
            plt.ylabel('Time (s)')
            plt.title('Time Vector with 4 Segments')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Time differences
            plt.subplot(3, 3, 2)
            dt_4seg = np.diff(time_4seg)
            plt.plot(dt_4seg, 'o-', linewidth=1, markersize=3)
            plt.axhline(0, color='r', linestyle='--', alpha=0.5, label='Zero step')
            plt.xlabel('Index')
            plt.ylabel('Time Step (s)')
            plt.title('Time Step Differences')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Velocity vs Time
            plt.subplot(3, 3, 3)
            plt.plot(time_4seg, velocity_4seg, 'b-', linewidth=2, label='Velocity')
            # Mark segment boundaries
            for idx in boundary_indices:
                if idx < len(time_4seg):
                    plt.axvline(time_4seg[idx], color='r', linestyle='--', alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (m/s)')
            plt.title('Velocity Profile')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 4: Position comparison
            plt.subplot(3, 3, 4)
            plt.plot(time_4seg, position_analytical_4seg, 'r-', linewidth=2, 
                    label='Analytical', linestyle='--', alpha=0.7)
            plt.plot(time_4seg, position_numerical_4seg, 'b-', linewidth=2, 
                    label='Simpson Integration', alpha=0.7)
            # Mark segment boundaries
            for idx in boundary_indices:
                if idx < len(time_4seg):
                    plt.axvline(time_4seg[idx], color='g', linestyle='--', alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Position: Analytical vs Numerical')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 5: Error
            plt.subplot(3, 3, 5)
            plt.semilogy(time_4seg[1:], error_4seg[1:], 'g-', linewidth=2)
            # Mark segment boundaries
            for idx in boundary_indices:
                if idx < len(time_4seg):
                    plt.axvline(time_4seg[idx], color='r', linestyle='--', alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('Absolute Error (m)')
            plt.title('Integration Error')
            plt.grid(True, alpha=0.3)
            
            # Plot 6: Error by segment
            plt.subplot(3, 3, 6)
            for i in range(n_segments):
                start_idx = i * n_points_per_segment - (i if i > 0 else 0)
                end_idx = start_idx + n_points_per_segment
                if end_idx > len(time_4seg):
                    end_idx = len(time_4seg)
                if start_idx < len(time_4seg):
                    seg_time = time_4seg[start_idx:end_idx]
                    seg_error = error_4seg[start_idx:end_idx]
                    plt.plot(seg_time, seg_error, 'o-', linewidth=2, 
                            markersize=4, label=f'Segment {i+1}', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Absolute Error (m)')
            plt.title('Error by Segment')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 7: Position at boundaries (zoom)
            plt.subplot(3, 3, 7)
            if boundary_indices:
                for idx in boundary_indices[:2]:  # Show first 2 boundaries
                    if idx > 0 and idx < len(time_4seg):
                        range_start = max(0, idx - 2)
                        range_end = min(len(time_4seg), idx + 2)
                        range_idx = np.arange(range_start, range_end)
                        plt.plot(time_4seg[range_idx], position_numerical_4seg[range_idx], 
                                'b-o', linewidth=2, markersize=8, label=f'Boundary {idx}')
                        plt.plot(time_4seg[range_idx], position_analytical_4seg[range_idx], 
                                'r--o', linewidth=2, markersize=8, alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Position at Boundaries (Zoom)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 8: Cumulative error
            plt.subplot(3, 3, 8)
            plt.plot(time_4seg, np.cumsum(error_4seg), 'g-', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Cumulative Error (m)')
            plt.title('Cumulative Error')
            plt.grid(True, alpha=0.3)
            
            # Plot 9: Relative error
            plt.subplot(3, 3, 9)
            relative_error_4seg = error_4seg / (np.abs(position_analytical_4seg) + 1e-10) * 100
            plt.semilogy(time_4seg[1:], relative_error_4seg[1:], 'g-', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Error (%)')
            plt.title('Relative Error')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('velocity_integration_4segments.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: velocity_integration_4segments.png")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"\nNote: Could not generate plot ({e})")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"\nIntegration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nIntegration examples complete!")

