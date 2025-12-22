#!/usr/bin/env python3
"""
Oscillation Demo: Shows the effect of ESKF reset on switching behavior.

Compares:
1. With ESKF reset (old behavior) - causes oscillation
2. Without ESKF reset (new behavior) - stable switching
"""

import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

# Lazy import to avoid pyIGRF issues
def get_generator():
    from data.generator_enhanced import EnhancedAttitudeDataGenerator
    return EnhancedAttitudeDataGenerator


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_with_switching(sim_data, config_path, fault_config, reset_eskf_on_switch=True):
    """
    Run redundant estimator with configurable ESKF reset behavior.

    Args:
        reset_eskf_on_switch: If True, reset ESKF to smoother state when switching (causes oscillation)
                             If False, keep ESKF running independently (stable)
    """
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize ESKF
    eskf = ESKF(P0=P0, config_path=config_path)

    # Initialize smoother
    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=True,
    )

    # Initial state with perturbation
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_est = EskfState(nom=nom0, err=err0)

    # Initialize smoother
    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    # Switching parameters
    disagreement_threshold = 2.0  # degrees
    agreement_threshold = 0.5  # degrees
    consecutive_to_switch = 5
    consecutive_to_recover = 10

    # State tracking
    primary = 'ESKF'
    consecutive_disagreements = 0
    consecutive_agreements = 0

    # Results
    times = []
    eskf_errors = []
    smoother_errors = []
    selected_errors = []
    disagreements = []
    primaries = []
    switch_events = []

    fault_active = False
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()

        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        # Inject fault
        if fault_config and t >= fault_config.get('onset_time', float('inf')):
            if not fault_active:
                fault_active = True
                injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
            omega_k = omega_k + injected_bias

        # === ESKF Update ===
        x_est = eskf.predict(x_est, omega_k, dt_k)

        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # === Smoother Update ===
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        has_measurement = z_mag is not None or z_sun is not None or z_st is not None

        if has_measurement:
            smoother_state = smoother.add_measurement(
                t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=B_n, s_eci=s_n,
            )
        else:
            smoother_state = smoother.get_state()

        # === Compute disagreement ===
        if smoother_state is not None:
            dq = x_est.nom.ori.conjugate() @ smoother_state.ori
            disagreement_deg = np.rad2deg(2 * np.arccos(np.clip(abs(dq.mu), 0, 1)))
        else:
            disagreement_deg = 0.0

        # === Switching logic ===
        if has_measurement and smoother_state is not None:
            exceeds_threshold = disagreement_deg > disagreement_threshold
            is_agreement = disagreement_deg < agreement_threshold

            if exceeds_threshold:
                consecutive_disagreements += 1
                consecutive_agreements = 0
            elif is_agreement:
                consecutive_agreements += 1
                consecutive_disagreements = 0
            else:
                consecutive_disagreements = 0
                consecutive_agreements = 0

            # Check for switch
            if primary == 'ESKF' and consecutive_disagreements >= consecutive_to_switch:
                primary = 'SMOOTHER'
                switch_events.append((t, 'ESKF->SMOOTHER'))
                consecutive_disagreements = 0
                consecutive_agreements = 0

                # THIS IS THE KEY DIFFERENCE:
                if reset_eskf_on_switch:
                    # OLD BEHAVIOR: Reset ESKF to smoother state
                    x_est.nom.ori = smoother_state.ori.copy()
                    x_est.nom.gyro_bias = smoother_state.gyro_bias.copy()
                    x_est.err.mean[:] = 0.0
                    x_est.err.cov = 0.5 * x_est.err.cov
                # NEW BEHAVIOR: Don't reset, let ESKF continue diverging

            elif primary == 'SMOOTHER' and consecutive_agreements >= consecutive_to_recover:
                primary = 'ESKF'
                switch_events.append((t, 'SMOOTHER->ESKF'))
                consecutive_disagreements = 0
                consecutive_agreements = 0

        # === Record results ===
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true) if smoother_state else eskf_err
        selected_err = smoother_err if primary == 'SMOOTHER' else eskf_err

        times.append(t)
        eskf_errors.append(eskf_err)
        smoother_errors.append(smoother_err)
        selected_errors.append(selected_err)
        disagreements.append(disagreement_deg)
        primaries.append(primary)

    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'smoother_errors': np.array(smoother_errors),
        'selected_errors': np.array(selected_errors),
        'disagreements': np.array(disagreements),
        'primaries': primaries,
        'switch_events': switch_events,
    }


def main():
    config_path = "configs/config_baseline_short.yaml"

    # Load existing data from database
    print("Loading simulation data from database...")
    import sqlite3
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()
    print(f"Using simulation ID: {sim_id}")
    sim_data = db.load_run(sim_id)

    fault_config = {
        'type': 'gyro_bias_step',
        'onset_time': 100.0,  # Earlier onset to have more post-fault time
        'magnitude': 0.05,    # 50 mrad/s - larger to cause faster divergence
    }

    print("\nRunning WITH ESKF reset (old behavior - causes oscillation)...")
    result_with_reset = run_with_switching(sim_data, config_path, fault_config, reset_eskf_on_switch=True)

    print("Running WITHOUT ESKF reset (new behavior - stable)...")
    result_no_reset = run_with_switching(sim_data, config_path, fault_config, reset_eskf_on_switch=False)

    # Create comparison plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    t_start = 50.0  # Start earlier to see pre-fault behavior
    fault_onset = fault_config['onset_time']

    mask_with = result_with_reset['times'] >= t_start
    mask_no = result_no_reset['times'] >= t_start

    # === Plot 1: With ESKF reset (oscillating) ===
    ax1 = axes[0]
    ax1.semilogy(result_with_reset['times'][mask_with], result_with_reset['eskf_errors'][mask_with],
                 'b-', alpha=0.5, linewidth=1, label='ESKF')
    ax1.semilogy(result_with_reset['times'][mask_with], result_with_reset['smoother_errors'][mask_with],
                 'g-', alpha=0.5, linewidth=1, label='Smoother')
    ax1.semilogy(result_with_reset['times'][mask_with], result_with_reset['selected_errors'][mask_with],
                 'r-', linewidth=2, label='Selected output')
    ax1.axvline(fault_onset, color='black', linestyle='--', alpha=0.5)

    # Mark switches
    for t_sw, direction in result_with_reset['switch_events']:
        if t_sw >= t_start:
            color = 'purple' if 'SMOOTHER' in direction else 'orange'
            ax1.axvline(t_sw, color=color, linestyle=':', alpha=0.7)

    ax1.set_ylabel('Error [deg]')
    ax1.set_title(f'WITH ESKF Reset: {len(result_with_reset["switch_events"])} switches (OSCILLATING)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-2, 100])

    # === Plot 2: Disagreement with reset ===
    ax2 = axes[1]
    ax2.plot(result_with_reset['times'][mask_with], result_with_reset['disagreements'][mask_with],
             'purple', linewidth=1)
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
    ax2.axhline(0.5, color='g', linestyle='--', alpha=0.5, label='Recovery threshold')
    ax2.axvline(fault_onset, color='black', linestyle='--', alpha=0.5)

    for t_sw, direction in result_with_reset['switch_events']:
        if t_sw >= t_start:
            ax2.axvline(t_sw, color='purple' if 'SMOOTHER' in direction else 'orange', linestyle=':', alpha=0.7)

    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('Disagreement WITH reset: drops to ~0 after each switch, triggers switch-back')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 20])

    # === Plot 3: Without ESKF reset (stable) ===
    ax3 = axes[2]
    ax3.semilogy(result_no_reset['times'][mask_no], result_no_reset['eskf_errors'][mask_no],
                 'b-', alpha=0.5, linewidth=1, label='ESKF')
    ax3.semilogy(result_no_reset['times'][mask_no], result_no_reset['smoother_errors'][mask_no],
                 'g-', alpha=0.5, linewidth=1, label='Smoother')
    ax3.semilogy(result_no_reset['times'][mask_no], result_no_reset['selected_errors'][mask_no],
                 'r-', linewidth=2, label='Selected output')
    ax3.axvline(fault_onset, color='black', linestyle='--', alpha=0.5)

    for t_sw, direction in result_no_reset['switch_events']:
        if t_sw >= t_start:
            color = 'purple' if 'SMOOTHER' in direction else 'orange'
            ax3.axvline(t_sw, color=color, linestyle=':', alpha=0.7)

    ax3.set_ylabel('Error [deg]')
    ax3.set_title(f'WITHOUT ESKF Reset: {len(result_no_reset["switch_events"])} switches (STABLE)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([1e-2, 100])

    # === Plot 4: Disagreement without reset ===
    ax4 = axes[3]
    ax4.plot(result_no_reset['times'][mask_no], result_no_reset['disagreements'][mask_no],
             'purple', linewidth=1)
    ax4.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
    ax4.axhline(0.5, color='g', linestyle='--', alpha=0.5, label='Recovery threshold')
    ax4.axvline(fault_onset, color='black', linestyle='--', alpha=0.5)

    for t_sw, direction in result_no_reset['switch_events']:
        if t_sw >= t_start:
            ax4.axvline(t_sw, color='purple' if 'SMOOTHER' in direction else 'orange', linestyle=':', alpha=0.7)

    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Disagreement [deg]')
    ax4.set_title('Disagreement WITHOUT reset: stays high, no premature switch-back')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 60])

    plt.tight_layout()
    plt.savefig('oscillation_demo.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved to oscillation_demo.pdf")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("OSCILLATION ANALYSIS")
    print("="*70)

    print(f"\nWITH ESKF reset (old behavior):")
    print(f"  Total switches: {len(result_with_reset['switch_events'])}")
    for t_sw, direction in result_with_reset['switch_events']:
        print(f"    t={t_sw:.1f}s: {direction}")

    print(f"\nWITHOUT ESKF reset (new behavior):")
    print(f"  Total switches: {len(result_no_reset['switch_events'])}")
    for t_sw, direction in result_no_reset['switch_events']:
        print(f"    t={t_sw:.1f}s: {direction}")

    print("\n" + "-"*70)
    print("MECHANISM OF OSCILLATION (with reset):")
    print("-"*70)
    print("""
    1. Fault at t=150s causes ESKF to diverge
    2. Disagreement exceeds 2° → switch to SMOOTHER
    3. ESKF is RESET to smoother state → disagreement drops to ~0°
    4. After 10 consecutive agreements (< 0.5°) → switch back to ESKF
    5. But fault is still present! ESKF immediately diverges again
    6. Disagreement exceeds 2° again → switch back to SMOOTHER
    7. REPEAT steps 3-6 → OSCILLATION

    The "selected output" oscillates between:
    - Smoother (low error) when primary=SMOOTHER
    - ESKF (high error) when primary=ESKF after reset
    """)

    print("-"*70)
    print("WHY NO RESET FIXES IT:")
    print("-"*70)
    print("""
    1. Fault at t=150s causes ESKF to diverge
    2. Disagreement exceeds 2° → switch to SMOOTHER
    3. ESKF is NOT reset → continues diverging → disagreement STAYS HIGH
    4. Disagreement never drops below 0.5° → no switch back to ESKF
    5. System stays on smoother output → STABLE

    The smoother output is used continuously after the switch.
    """)
    print("="*70)


if __name__ == "__main__":
    main()
