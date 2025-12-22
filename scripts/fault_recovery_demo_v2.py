#!/usr/bin/env python3
"""
Fault Recovery Demonstration v2.

Key insight: The value isn't "recovering" ESKF - it's SWITCHING to the smoother output.

This shows:
1. ESKF only → diverges after fault
2. Redundant with ESKF output → still diverges (reset doesn't help if fault persists)
3. Redundant with SMOOTHER output → stays bounded (THIS is the value)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_eskf_only(sim_data, config_path, fault_config=None):
    """Run standalone ESKF (no smoother, no switching)."""
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)
    eskf = ESKF(P0=P0, config_path=config_path)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_est = EskfState(nom=nom0, err=err0)

    times, errors = [], []
    fault_active = False
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        if fault_config and t >= fault_config.get('onset_time', float('inf')):
            if not fault_active:
                fault_active = True
                injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
            omega_k = omega_k + injected_bias

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

        q_true = Quaternion.from_array(sim_data.q_true[k])
        err = compute_attitude_error(x_est.nom.ori, q_true)
        times.append(t)
        errors.append(err)

    return np.array(times), np.array(errors)


def run_redundant(sim_data, config_path, fault_config=None):
    """Run redundant estimator - tracks both ESKF and smoother errors."""
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=60.0,
        use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_est = EskfState(nom=nom0, err=err0)

    times, eskf_errors, smoother_errors, disagreements, primaries = [], [], [], [], []
    selected_errors = []  # Error from whichever estimator is primary
    fault_active = False
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        if fault_config and t >= fault_config.get('onset_time', float('inf')):
            if not fault_active:
                fault_active = True
                injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
            omega_k = omega_k + injected_bias

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        x_est, smoother_state, disagreement_deg, primary_str = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true)

        # The "selected" output based on which estimator is primary
        selected_err = smoother_err if primary_str == 'SMOOTHER' else eskf_err

        times.append(t)
        eskf_errors.append(eskf_err)
        smoother_errors.append(smoother_err)
        disagreements.append(disagreement_deg)
        primaries.append(primary_str)
        selected_errors.append(selected_err)

    stats = redundant.get_statistics()
    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'smoother_errors': np.array(smoother_errors),
        'selected_errors': np.array(selected_errors),
        'disagreements': np.array(disagreements),
        'primaries': primaries,
        'switch_events': stats.get('switch_events', []),
    }


def main():
    config_path = "configs/config_baseline_short.yaml"
    db = SimulationDatabase("simulations.db")

    import sqlite3
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()

    print(f"Loading simulation ID: {sim_id}")
    sim_data = db.load_run(sim_id)

    fault_onset = 150.0
    fault_magnitude = 0.02

    fault_config = {
        'type': 'gyro_bias_step',
        'onset_time': fault_onset,
        'magnitude': fault_magnitude
    }

    print("\nRunning ESKF only...")
    eskf_times, eskf_errors = run_eskf_only(sim_data, config_path, fault_config)

    print("Running Redundant...")
    redundant = run_redundant(sim_data, config_path, fault_config)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    t_start = 100.0
    mask_eskf = eskf_times >= t_start
    mask_red = redundant['times'] >= t_start

    # === Plot 1: Compare all outputs ===
    ax1 = axes[0]
    ax1.semilogy(eskf_times[mask_eskf], eskf_errors[mask_eskf], 'r-',
                 linewidth=1.5, alpha=0.9, label='ESKF only (standalone)')
    ax1.semilogy(redundant['times'][mask_red], redundant['smoother_errors'][mask_red], 'g-',
                 linewidth=1.5, alpha=0.9, label='Fixed-lag Smoother')
    ax1.semilogy(redundant['times'][mask_red], redundant['selected_errors'][mask_red], 'b--',
                 linewidth=2, alpha=0.9, label='Redundant output (selected)')

    ax1.axvline(fault_onset, color='black', linestyle='-', linewidth=2, alpha=0.7, label='Fault onset')

    # Shade smoother-primary regions
    times_red = redundant['times'][mask_red]
    primaries_red = [redundant['primaries'][i] for i in np.where(mask_red)[0]]
    in_smoother = False
    smoother_start = None
    for i, (t, p) in enumerate(zip(times_red, primaries_red)):
        if p == 'SMOOTHER' and not in_smoother:
            in_smoother = True
            smoother_start = t
        elif p == 'ESKF' and in_smoother:
            in_smoother = False
            ax1.axvspan(smoother_start, t, color='green', alpha=0.15)

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_title(f'Fault Recovery: Gyro Bias Step ({fault_magnitude*1000:.0f} mrad/s) at t={fault_onset}s\n(Green shading = smoother is primary output)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-2, 50])

    # === Plot 2: Disagreement ===
    ax2 = axes[1]
    ax2.plot(redundant['times'][mask_red], redundant['disagreements'][mask_red], 'purple', linewidth=1)
    ax2.fill_between(redundant['times'][mask_red], redundant['disagreements'][mask_red],
                     alpha=0.3, color='purple')
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold (2°)')
    ax2.axvline(fault_onset, color='black', linestyle='-', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('ESKF-Smoother Disagreement (triggers switch when > 2° for 5 consecutive measurements)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fault_recovery_v2.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved to fault_recovery_v2.pdf")
    plt.close()

    # === Statistics ===
    print("\n" + "="*70)
    print("FAULT RECOVERY VALUE ANALYSIS")
    print("="*70)

    post_mask_eskf = eskf_times >= fault_onset
    post_mask_red = redundant['times'] >= fault_onset

    eskf_only = eskf_errors[post_mask_eskf]
    smoother = redundant['smoother_errors'][post_mask_red]
    selected = redundant['selected_errors'][post_mask_red]

    print(f"\nPost-fault statistics (t >= {fault_onset}s):")
    print("-"*70)
    print(f"{'Output':<30} {'Mean [deg]':<12} {'Max [deg]':<12} {'RMS [deg]':<12}")
    print("-"*70)
    print(f"{'ESKF only (no redundancy)':<30} {np.mean(eskf_only):>10.2f} {np.max(eskf_only):>10.2f} {np.sqrt(np.mean(eskf_only**2)):>10.2f}")
    print(f"{'Smoother (always)':<30} {np.mean(smoother):>10.2f} {np.max(smoother):>10.2f} {np.sqrt(np.mean(smoother**2)):>10.2f}")
    print(f"{'Redundant (switched output)':<30} {np.mean(selected):>10.2f} {np.max(selected):>10.2f} {np.sqrt(np.mean(selected**2)):>10.2f}")

    print("\n" + "-"*70)
    improvement_vs_eskf = (np.mean(eskf_only) - np.mean(selected)) / np.mean(eskf_only) * 100
    print(f"Redundant output vs ESKF-only: {improvement_vs_eskf:+.1f}% error reduction")

    # Count how much time spent using smoother
    smoother_time = sum(1 for p in redundant['primaries'][post_mask_red.sum():] if p == 'SMOOTHER')
    total_time = post_mask_red.sum()
    print(f"Time using smoother as primary: {smoother_time}/{total_time} samples ({smoother_time/total_time*100:.1f}%)")

    print(f"\nSwitch events after fault: {len([e for e in redundant['switch_events'] if e[0] >= fault_onset])}")
    for t_sw, direction in redundant['switch_events']:
        if t_sw >= fault_onset:
            print(f"  t={t_sw:.1f}s: {direction}")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("""
The value of redundant architecture during persistent faults is:

1. FAULT DETECTION: Disagreement reliably indicates something is wrong

2. SWITCHING OUTPUT: When ESKF diverges, use smoother estimate instead
   - Smoother is more robust because it uses more measurements to correct
   - Even with faulty gyro, vector measurements keep smoother bounded

3. NOT "recovery": Resetting ESKF doesn't help if fault persists
   - ESKF will just diverge again using the same faulty gyro
   - Real recovery requires fixing the root cause (which we can't simulate)
""")
    print("="*70)


if __name__ == "__main__":
    main()
