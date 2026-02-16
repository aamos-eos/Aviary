"""
OpenMDAO component for computing wing geometric parameters.

This component computes derived wing geometry from basic parameters,
allowing S_ref and AR to be design variables while keeping all 
dependent geometry parameters consistent.

Based on compute_wing_geometry() from load_ac_data.py
"""

import numpy as np
import openmdao.api as om


def compute_airfoil_pc(toverc, k=0.3, toverc_shift=0.05):
    """
    Compute airfoil perimeter coefficient from thickness-to-chord ratio.
    
    Based on the approximation:
    airfoil_pc = 0.5 * [sqrt((t/c)² + 16k²) + 2*sqrt((t/c)² + 4(1-k)²)] + shift
    
    where k is the chordwise location of maximum thickness (typically 0.3).
    The shift is added at the end to account for practical effects (e.g., 
    leading/trailing edge thickness, surface roughness).

    Reference: https://booksite.elsevier.com/9780123973085/content/APP-D-GEOMETRY_OF_LIFTING_SURFACES.pdf
    Equation (D-5)
    
    Parameters
    ----------
    toverc : float
        Thickness-to-chord ratio
    k : float
        Chordwise location of max thickness (default 0.3 = 30% chord)
    toverc_shift : float
        Additive shift applied to the final airfoil_pc (default 0.05)
        
    Returns
    -------
    float
        Airfoil perimeter coefficient
    """
    term1 = np.sqrt(toverc**2 + 16 * k**2)
    term2 = 2 * np.sqrt(toverc**2 + 4 * (1 - k)**2)
    return 0.5 * (term1 + term2) + toverc_shift


class WingGeometryComp(om.ExplicitComponent):
    """
    Compute derived wing geometric parameters from S_ref, AR, taper, toverc.
    
    This component should be used when S_ref and AR are design variables to ensure
    all dependent wing geometry (span, chords, MAC, wetted area) are recomputed.
    
    The airfoil perimeter coefficient is computed from toverc using:
    airfoil_pc = 0.5 * [sqrt((t/c)² + 16k²) + 2*sqrt((t/c)² + 4(1-k)²)] + shift
    where k = 0.3 (max thickness at 30% chord), shift = 0.05 (default).
    
    Inputs
    ------
    S_ref : float
        Wing reference area (m²)
    AR : float
        Wing aspect ratio
    taper : float
        Wing taper ratio (tip chord / root chord)
    toverc : float
        Wing thickness-to-chord ratio
    c0_sweep : float
        Leading edge sweep angle (rad)
        
    Outputs
    -------
    span : float
        Wing span (m)
    c_root : float
        Root chord (m)
    c_tip : float
        Tip chord (m)
    MAC : float
        Mean Aerodynamic Chord (m)
    mac_buttline : float
        MAC buttline position (m)
    S_wet : float
        Wing wetted area (m²)
    airfoil_pc : float
        Airfoil perimeter coefficient (computed from toverc)
    c4_sweep : float
        Quarter-chord sweep angle (rad)
        Computed as: tan(c4_sweep) = tan(c0_sweep) - (1/AR) * (1-taper)/(1+taper)
    """
    
    def initialize(self):
        self.options.declare('default_S_ref', default=90.0, types=float, 
                            desc='Default wing reference area (m²)')
        self.options.declare('default_AR', default=12.0, types=float,
                            desc='Default aspect ratio')
        self.options.declare('default_taper', default=0.35, types=float,
                            desc='Default taper ratio')
        self.options.declare('default_toverc', default=0.16, types=float,
                            desc='Default thickness-to-chord ratio')
        self.options.declare('default_c0_sweep', default=0.0, types=float,
                            desc='Default leading edge sweep angle (rad)')
        self.options.declare('k_max_thickness', default=0.3, types=float,
                            desc='Chordwise location of max thickness (fraction)')
        self.options.declare('toverc_shift', default=0.025, types=float,
                            desc='Shift applied to t/c for airfoil_pc calculation')
    
    def setup(self):
        default_S_ref = self.options['default_S_ref']
        default_AR = self.options['default_AR']
        default_taper = self.options['default_taper']
        default_toverc = self.options['default_toverc']
        default_c0_sweep = self.options['default_c0_sweep']
        k = self.options['k_max_thickness']
        toverc_shift = self.options['toverc_shift']
        
        # Inputs
        self.add_input('S_ref', val=default_S_ref, units='m**2', desc='Wing reference area')
        self.add_input('AR', val=default_AR, units=None, desc='Wing aspect ratio')
        self.add_input('taper', val=default_taper, units=None, desc='Wing taper ratio')
        self.add_input('toverc', val=default_toverc, units=None, 
                      desc='Wing thickness-to-chord ratio')
        self.add_input('c0_sweep', val=default_c0_sweep, units='rad', 
                      desc='Leading edge sweep angle')
        
        # Compute default airfoil_pc from toverc
        default_airfoil_pc = compute_airfoil_pc(default_toverc, k=k, toverc_shift=toverc_shift)
        
        # Compute default outputs
        default_span = np.sqrt(default_S_ref * default_AR)
        default_c_root = 2 * (default_span / default_AR) / (1 + default_taper)
        default_c_tip = default_c_root * default_taper
        default_MAC = 2/3 * default_c_root * (1 + default_taper + default_taper**2) / (1 + default_taper)
        default_mac_buttline = default_span * (1 + 2*default_taper) / (6 * (1 + default_taper))
        wing_eff = 1.0922 * 2 / default_span
        default_S_wet = (default_span - wing_eff * default_span) * (
            np.sqrt(default_S_ref / default_AR) - (1 - default_taper) / (1 + default_taper) 
            * wing_eff * default_span / default_AR
        ) * default_airfoil_pc
        
        # Compute default quarter-chord sweep: tan(c4) = tan(c0) - (1/AR) * (1-taper)/(1+taper)
        default_c4_sweep = np.arctan(np.tan(default_c0_sweep) - (1/default_AR) * (1 - default_taper) / (1 + default_taper))
        
        # Outputs
        self.add_output('span', val=default_span, units='m', desc='Wing span')
        self.add_output('c_root', val=default_c_root, units='m', desc='Root chord')
        self.add_output('c_tip', val=default_c_tip, units='m', desc='Tip chord')
        self.add_output('MAC', val=default_MAC, units='m', desc='Mean Aerodynamic Chord')
        self.add_output('mac_buttline', val=default_mac_buttline, units='m', 
                       desc='MAC buttline position')
        self.add_output('S_wet', val=default_S_wet, units='m**2', desc='Wing wetted area')
        self.add_output('airfoil_pc', val=default_airfoil_pc, units=None,
                       desc='Airfoil perimeter coefficient (computed from toverc)')
        self.add_output('c4_sweep', val=default_c4_sweep, units='rad', 
                       desc='Quarter-chord sweep angle')
        
        # Declare partials
        self.declare_partials('span', ['S_ref', 'AR'])
        self.declare_partials('c_root', ['S_ref', 'AR', 'taper'])
        self.declare_partials('c_tip', ['S_ref', 'AR', 'taper'])
        self.declare_partials('MAC', ['S_ref', 'AR', 'taper'])
        self.declare_partials('mac_buttline', ['S_ref', 'AR', 'taper'])
        self.declare_partials('airfoil_pc', ['toverc'])
        self.declare_partials('S_wet', ['S_ref', 'AR', 'taper', 'toverc'])
        self.declare_partials('c4_sweep', ['c0_sweep', 'AR', 'taper'])
        
    def compute(self, inputs, outputs):
        S_ref = inputs['S_ref']
        AR = inputs['AR']
        taper = inputs['taper']
        toverc = inputs['toverc']
        c0_sweep = inputs['c0_sweep']
        
        k = self.options['k_max_thickness']
        toverc_shift = self.options['toverc_shift']
        
        # Compute airfoil perimeter coefficient from toverc
        airfoil_pc = compute_airfoil_pc(toverc, k=k, toverc_shift=toverc_shift)
        
        # Compute span from S_ref and AR
        span = np.sqrt(S_ref * AR)
        
        # Compute chord lengths
        c_root = 2 * (span / AR) / (1 + taper)
        c_tip = c_root * taper
        
        # Compute Mean Aerodynamic Chord (MAC)
        MAC = 2/3 * c_root * (1 + taper + taper**2) / (1 + taper)
        
        # Compute MAC buttline position
        mac_buttline = span * (1 + 2 * taper) / (6 * (1 + taper))
        
        # Compute wing efficiency coefficient
        wing_eff = 1.0922 * 2 / span
        
        # Compute wetted area
        S_wet = (span - wing_eff * span) * (
            np.sqrt(S_ref / AR) - (1 - taper) / (1 + taper) 
            * wing_eff * span / AR
        ) * airfoil_pc
        
        # Compute quarter-chord sweep from leading edge sweep
        # tan(c4_sweep) = tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper)
        c4_sweep = np.arctan(np.tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper))
        
        outputs['span'] = span
        outputs['c_root'] = c_root
        outputs['c_tip'] = c_tip
        outputs['MAC'] = MAC
        outputs['mac_buttline'] = mac_buttline
        outputs['airfoil_pc'] = airfoil_pc
        outputs['S_wet'] = S_wet
        outputs['c4_sweep'] = c4_sweep
        
    def compute_partials(self, inputs, partials):
        S_ref = float(inputs['S_ref'])
        AR = float(inputs['AR'])
        taper = float(inputs['taper'])
        toverc = float(inputs['toverc'])
        
        k = self.options['k_max_thickness']
        toverc_shift = self.options['toverc_shift']
        
        # Compute airfoil_pc from toverc
        airfoil_pc = compute_airfoil_pc(toverc, k=k, toverc_shift=toverc_shift)
        
        span = np.sqrt(S_ref * AR)
        
        # Partials for span = sqrt(S_ref * AR)
        partials['span', 'S_ref'] = 0.5 * np.sqrt(AR / S_ref)
        partials['span', 'AR'] = 0.5 * np.sqrt(S_ref / AR)
        
        # c_root = 2 * (span / AR) / (1 + taper) = 2 * sqrt(S_ref/AR) / (1 + taper)
        d_c_root_d_S_ref = (1 / (1 + taper)) * np.sqrt(1 / (S_ref * AR))
        d_c_root_d_AR = -(1 / (1 + taper)) * np.sqrt(S_ref) / (AR ** 1.5)
        d_c_root_d_taper = -2 * np.sqrt(S_ref / AR) / ((1 + taper) ** 2)
        
        partials['c_root', 'S_ref'] = d_c_root_d_S_ref
        partials['c_root', 'AR'] = d_c_root_d_AR
        partials['c_root', 'taper'] = d_c_root_d_taper
        
        # c_tip = c_root * taper
        c_root = 2 * np.sqrt(S_ref / AR) / (1 + taper)
        partials['c_tip', 'S_ref'] = taper * d_c_root_d_S_ref
        partials['c_tip', 'AR'] = taper * d_c_root_d_AR
        partials['c_tip', 'taper'] = c_root + taper * d_c_root_d_taper
        
        # MAC = 2/3 * c_root * (1 + taper + taper^2) / (1 + taper)
        # Let f = (1 + taper + taper^2) / (1 + taper)
        f = (1 + taper + taper**2) / (1 + taper)
        df_dtaper = ((1 + 2*taper) * (1 + taper) - (1 + taper + taper**2)) / ((1 + taper)**2)
        
        partials['MAC', 'S_ref'] = (2/3) * f * d_c_root_d_S_ref
        partials['MAC', 'AR'] = (2/3) * f * d_c_root_d_AR
        partials['MAC', 'taper'] = (2/3) * (f * d_c_root_d_taper + c_root * df_dtaper)
        
        # mac_buttline = span * (1 + 2*taper) / (6 * (1 + taper))
        g = (1 + 2*taper) / (6 * (1 + taper))
        dg_dtaper = (2 * 6 * (1 + taper) - (1 + 2*taper) * 6) / (36 * (1 + taper)**2)
        dg_dtaper = (2 * (1 + taper) - (1 + 2*taper)) / (6 * (1 + taper)**2)
        dg_dtaper = (2 + 2*taper - 1 - 2*taper) / (6 * (1 + taper)**2)
        dg_dtaper = 1 / (6 * (1 + taper)**2)
        
        partials['mac_buttline', 'S_ref'] = g * 0.5 * np.sqrt(AR / S_ref)
        partials['mac_buttline', 'AR'] = g * 0.5 * np.sqrt(S_ref / AR)
        partials['mac_buttline', 'taper'] = span * dg_dtaper
        
        # Partial of airfoil_pc with respect to toverc
        # airfoil_pc = 0.5 * [sqrt(toverc² + 16k²) + 2*sqrt(toverc² + 4(1-k)²)] + shift
        # The shift is a constant added at the end, so it doesn't affect the derivative
        term1 = np.sqrt(toverc**2 + 16 * k**2)
        term2 = np.sqrt(toverc**2 + 4 * (1 - k)**2)
        
        # d(airfoil_pc)/d(toverc) = 0.5 * [toverc/term1 + 2*toverc/term2]
        d_airfoil_pc_d_toverc = 0.5 * (toverc / term1 + 2 * toverc / term2)
        partials['airfoil_pc', 'toverc'] = d_airfoil_pc_d_toverc
        
        # S_wet is complex, use finite difference for now
        # Could derive analytically if needed for performance
        eps = 1e-6
        
        def compute_S_wet(S, A, t, pc):
            s = np.sqrt(S * A)
            we = 1.0922 * 2 / s
            return (s - we * s) * (np.sqrt(S / A) - (1 - t) / (1 + t) * we * s / A) * pc
        
        S_wet_0 = compute_S_wet(S_ref, AR, taper, airfoil_pc)
        
        partials['S_wet', 'S_ref'] = (compute_S_wet(S_ref + eps, AR, taper, airfoil_pc) - S_wet_0) / eps
        partials['S_wet', 'AR'] = (compute_S_wet(S_ref, AR + eps, taper, airfoil_pc) - S_wet_0) / eps
        partials['S_wet', 'taper'] = (compute_S_wet(S_ref, AR, taper + eps, airfoil_pc) - S_wet_0) / eps
        
        # d(S_wet)/d(toverc) = d(S_wet)/d(airfoil_pc) * d(airfoil_pc)/d(toverc)
        d_S_wet_d_airfoil_pc = (compute_S_wet(S_ref, AR, taper, airfoil_pc + eps) - S_wet_0) / eps
        partials['S_wet', 'toverc'] = d_S_wet_d_airfoil_pc * d_airfoil_pc_d_toverc
        
        # c4_sweep = arctan(tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper))
        # Let u = tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper)
        c0_sweep = float(inputs['c0_sweep'])
        u = np.tan(c0_sweep) - (1/AR) * (1 - taper) / (1 + taper)
        du_dc4 = 1 / (1 + u**2)  # d(arctan(u))/du
        
        # d(c4_sweep)/d(c0_sweep) = du_dc4 * sec^2(c0_sweep)
        partials['c4_sweep', 'c0_sweep'] = du_dc4 / (np.cos(c0_sweep)**2)
        
        # d(c4_sweep)/d(AR) = du_dc4 * (1/AR^2) * (1 - taper) / (1 + taper)
        partials['c4_sweep', 'AR'] = du_dc4 * (1 / AR**2) * (1 - taper) / (1 + taper)
        
        # d(c4_sweep)/d(taper) = du_dc4 * (2/AR) / (1 + taper)^2
        # u = tan(c0) - (1/AR) * (1-taper)/(1+taper)
        # d/d(taper)[(1-taper)/(1+taper)] = -2/(1+taper)^2
        # du/d(taper) = -(1/AR) * (-2/(1+taper)^2) = 2/(AR*(1+taper)^2)
        partials['c4_sweep', 'taper'] = du_dc4 * (2 / AR) / ((1 + taper)**2)


if __name__ == "__main__":
    # Test the WingGeometryComp and verify partials
    
    print("=" * 60)
    print("Testing WingGeometryComp")
    print("=" * 60)
    
    # Create problem
    prob = om.Problem()
    
    # Add independent variable component for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('S_ref', val=95.0, units='m**2')
    ivc.add_output('AR', val=11.5, units=None)
    ivc.add_output('taper', val=0.35, units=None)
    ivc.add_output('toverc', val=0.16, units=None)
    ivc.add_output('c0_sweep', val=0.0, units='rad')
    
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('wing_geom', WingGeometryComp(), promotes=['*'])
    
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    
    # Print outputs
    print("\n" + "-" * 40)
    print("Inputs:")
    print("-" * 40)
    print(f"  S_ref     = {prob.get_val('S_ref')[0]:.4f} m²")
    print(f"  AR        = {prob.get_val('AR')[0]:.4f}")
    print(f"  taper     = {prob.get_val('taper')[0]:.4f}")
    print(f"  toverc    = {prob.get_val('toverc')[0]:.4f}")
    print(f"  c0_sweep  = {prob.get_val('c0_sweep')[0]:.4f} rad")
    
    print("\n" + "-" * 40)
    print("Outputs:")
    print("-" * 40)
    print(f"  span          = {prob.get_val('span')[0]:.4f} m")
    print(f"  c_root        = {prob.get_val('c_root')[0]:.4f} m")
    print(f"  c_tip         = {prob.get_val('c_tip')[0]:.4f} m")
    print(f"  MAC           = {prob.get_val('MAC')[0]:.4f} m")
    print(f"  mac_buttline  = {prob.get_val('mac_buttline')[0]:.4f} m")
    print(f"  airfoil_pc    = {prob.get_val('airfoil_pc')[0]:.6f}")
    print(f"  S_wet         = {prob.get_val('S_wet')[0]:.4f} m²")
    print(f"  c4_sweep      = {prob.get_val('c4_sweep')[0]:.6f} rad ({np.degrees(prob.get_val('c4_sweep')[0]):.4f} deg)")
    
    # Test airfoil_pc calculation standalone
    print("\n" + "-" * 40)
    print("Airfoil PC calculation test:")
    print("-" * 40)
    for tc in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
        pc = compute_airfoil_pc(tc, k=0.3, toverc_shift=0.025)
        print(f"  toverc = {tc:.2f} -> airfoil_pc = {pc:.6f}")
    
    # Check partials
    print("\n" + "=" * 60)
    print("Checking Partials (Analytical vs Finite Difference)")
    print("=" * 60)
    
    data = prob.check_partials(compact_print=True, method='fd', step=1e-6)
    
    # Also check with complex step for higher accuracy
    print("\n" + "=" * 60)
    print("Checking Partials (Analytical vs Complex Step)")
    print("=" * 60)
    
    data_cs = prob.check_partials(compact_print=True, method='cs', show_only_incorrect=True)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    # Generate curve showing airfoil_pc vs t/c
    print("\n" + "=" * 60)
    print("Generating airfoil_pc vs t/c curve")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # Range of t/c values (typical for transport aircraft: 0.08 to 0.20)
    toverc_range = np.linspace(0.06, 0.24, 100)
    
    # Compute airfoil_pc for different k values (max thickness location)
    k_values = [0.25, 0.30, 0.35, 0.40]
    toverc_shift = 0.025
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: airfoil_pc vs t/c for different k values
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    for k, color in zip(k_values, colors):
        airfoil_pc_values = [compute_airfoil_pc(tc, k=k, toverc_shift=toverc_shift) for tc in toverc_range]
        ax1.plot(toverc_range * 100, airfoil_pc_values, label=f'k = {k:.2f}', 
                color=color, linewidth=2)
    
    # Mark typical values
    typical_toverc = [0.10, 0.12, 0.14, 0.16, 0.18]
    for tc in typical_toverc:
        pc = compute_airfoil_pc(tc, k=0.30, toverc_shift=toverc_shift)
        ax1.scatter([tc * 100], [pc], color='#e74c3c', s=50, zorder=5)
        ax1.annotate(f'{tc*100:.0f}%', (tc * 100, pc), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    ax1.set_xlabel('Thickness-to-Chord Ratio (t/c) [%]', fontsize=12)
    ax1.set_ylabel('Airfoil Perimeter Coefficient', fontsize=12)
    ax1.set_title('Airfoil Perimeter Coefficient vs t/c\n(with 0.025 shift applied)', fontsize=13)
    ax1.legend(title='Max thickness\nlocation (k)', loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([6, 24])
    
    # Plot 2: Effect of shift on airfoil_pc
    shifts = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    k_fixed = 0.30
    
    for shift in shifts:
        airfoil_pc_values = [compute_airfoil_pc(tc, k=k_fixed, toverc_shift=shift) for tc in toverc_range]
        ax2.plot(toverc_range * 100, airfoil_pc_values, label=f'shift = {shift:.2f}', linewidth=2)
    
    ax2.set_xlabel('Thickness-to-Chord Ratio (t/c) [%]', fontsize=12)
    ax2.set_ylabel('Airfoil Perimeter Coefficient', fontsize=12)
    ax2.set_title(f'Effect of t/c Shift on Airfoil PC\n(k = {k_fixed})', fontsize=13)
    ax2.legend(title='t/c shift', loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([6, 24])
    
    plt.tight_layout()
    plt.savefig('airfoil_pc_vs_toverc.png', dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: airfoil_pc_vs_toverc.png")
    plt.show()
    
    # Generate curve showing S_wet vs t/c
    print("\n" + "=" * 60)
    print("Generating S_wet vs t/c curve")
    print("=" * 60)
    
    # Compute S_wet for different toverc values using the OpenMDAO component
    toverc_sweep = np.linspace(0.08, 0.22, 50)
    S_wet_values = []
    airfoil_pc_sweep = []
    
    # Fixed wing parameters
    S_ref_fixed = 84.81  # m²
    AR_fixed = 14.4
    taper_fixed = 0.356
    
    for tc in toverc_sweep:
        prob.set_val('toverc', tc)
        prob.run_model()
        S_wet_values.append(prob.get_val('S_wet')[0])
        airfoil_pc_sweep.append(prob.get_val('airfoil_pc')[0])
    
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 3: S_wet vs t/c
    ax3.plot(toverc_sweep * 100, S_wet_values, 'b-', linewidth=2.5, label='Wing Wetted Area')
    ax3.set_xlabel('Thickness-to-Chord Ratio (t/c) [%]', fontsize=12)
    ax3.set_ylabel('Wing Wetted Area S_wet [m²]', fontsize=12)
    ax3.set_title(f'Wing Wetted Area vs t/c\n(S_ref={S_ref_fixed} m², AR={AR_fixed}, taper={taper_fixed})', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([8, 22])
    
    # Mark baseline value
    baseline_tc = 0.155
    prob.set_val('toverc', baseline_tc)
    prob.run_model()
    baseline_S_wet = prob.get_val('S_wet')[0]
    ax3.scatter([baseline_tc * 100], [baseline_S_wet], color='red', s=100, zorder=5, marker='o')
    ax3.annotate(f'Baseline\nt/c={baseline_tc*100:.2f}%\nS_wet={baseline_S_wet:.1f} m²', 
                (baseline_tc * 100, baseline_S_wet), textcoords="offset points", 
                xytext=(10, -30), fontsize=10, 
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 4: S_wet vs airfoil_pc (parametric in t/c)
    ax4.plot(airfoil_pc_sweep, S_wet_values, 'g-', linewidth=2.5)
    
    # Add t/c markers
    marker_tcs = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    for tc in marker_tcs:
        pc = compute_airfoil_pc(tc, k=0.30, toverc_shift=0.025)
        # Find closest S_wet value
        idx = np.argmin(np.abs(toverc_sweep - tc))
        s_wet = S_wet_values[idx]
        ax4.scatter([pc], [s_wet], color='#e74c3c', s=60, zorder=5)
        ax4.annotate(f'{tc*100:.0f}%', (pc, s_wet), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    ax4.set_xlabel('Airfoil Perimeter Coefficient', fontsize=12)
    ax4.set_ylabel('Wing Wetted Area S_wet [m²]', fontsize=12)
    ax4.set_title('Wing Wetted Area vs Airfoil PC\n(markers show t/c %)', fontsize=13)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('S_wet_vs_toverc.png', dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: S_wet_vs_toverc.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("All plots generated")
    print("=" * 60)
