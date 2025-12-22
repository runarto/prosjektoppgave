#!/usr/bin/env python3
"""
Fault Recovery Demonstration.

Shows what happens AFTER fault detection:
1. Does switching to smoother actually help?
2. Does the ESKF error decrease after reset?
3. Comparison: with vs without the switching mechanism
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

        # Fault injection
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
    """Run redundant estimator with switching."""
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

        times.append(t)
        eskf_errors.append(eskf_err)
        smoother_errors.append(smoother_err)
        disagreements.append(disagreement_deg)
        primaries.append(primary_str)

    stats = redundant.get_statistics()
    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'smoother_errors': np.array(smoother_errors),
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
    fault_magnitude = 0.02  # 20 mrad/s

    fault_config = {
        'type': 'gyro_bias_step',
        'onset_time': fault_onset,
        'magnitude': fault_magnitude
    }

    print("\nRunning ESKF only (no switching)...")
    eskf_times, eskf_errors = run_eskf_only(sim_data, config_path, fault_config)

    print("Running Redundant (with switching)...")
    redundant = run_redundant(sim_data, config_path, fault_config)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    t_start = 50.0  # After convergence
    mask_eskf = eskf_times >= t_start
    mask_red = redundant['times'] >= t_start

    # === Plot 1: ESKF-only vs Redundant ESKF error ===
    ax1 = axes[0]
    ax1.semilogy(eskf_times[mask_eskf], eskf_errors[mask_eskf], 'r-',
                 linewidth=1, alpha=0.8, label='ESKF only (no recovery)')
    ax1.semilogy(redundant['times'][mask_red], redundant['eskf_errors'][mask_red], 'b-',
                 linewidth=1, alpha=0.8, label='Redundant ESKF (with recovery)')
    ax1.axvline(fault_onset, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Fault onset')

    # Mark switch events
    for t_switch, direction in redundant['switch_events']:
        if t_switch >= t_start:
            color = 'purple' if 'SMOOTHER' in direction else 'orange'
            ax1.axvline(t_switch, color=color, linestyle=':', alpha=0.5)

    ax1.set_ylabel('ESKF Attitude Error [deg]')
    ax1.set_title(f'Recovery from Gyro Bias Fault ({fault_magnitude*1000:.0f} mrad/s at t={fault_onset}s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-3, 100])

    # === Plot 2: ESKF vs Smoother error (redundant system) ===
    ax2 = axes[1]
    ax2.semilogy(redundant['times'][mask_red], redundant['eskf_errors'][mask_red], 'b-',
                 linewidth=1, alpha=0.8, label='ESKF estimate')
    ax2.semilogy(redundant['times'][mask_red], redundant['smoother_errors'][mask_red], 'g-',
                 linewidth=1, alpha=0.8, label='Smoother estimate')
    ax2.axvline(fault_onset, color='green', linestyle='-', linewidth=2, alpha=0.7)

    # Shade regions by primary estimator
    times_red = redundant['times'][mask_red]
    primaries_red = [redundant['primaries'][i] for i in np.where(mask_red)[0]]

    # Find contiguous regions where smoother is primary
    in_smoother = False
    smoother_start = None
    for i, (t, p) in enumerate(zip(times_red, primaries_red)):
        if p == 'SMOOTHER' and not in_smoother:
            in_smoother = True
            smoother_start = t
        elif p == 'ESKF' and in_smoother:
            in_smoother = False
            ax2.axvspan(smoother_start, t, color='green', alpha=0.1)

    ax2.set_ylabel('Attitude Error [deg]')
    ax2.set_title('ESKF vs Smoother During Fault (green shading = smoother is primary)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1e-3, 100])

    # === Plot 3: Disagreement with switch markers ===
    ax3 = axes[2]
    ax3.plot(redundant['times'][mask_red], redundant['disagreements'][mask_red], 'purple',
             linewidth=1, alpha=0.8)
    ax3.fill_between(redundant['times'][mask_red], redundant['disagreements'][mask_red],
                     alpha=0.3, color='purple')
    ax3.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
    ax3.axvline(fault_onset, color='green', linestyle='-', linewidth=2, alpha=0.7)

    # Mark switch events with arrows
    for t_switch, direction in redundant['switch_events']:
        if t_switch >= t_start:
            if 'SMOOTHER' in direction:
                ax3.annotate('→Smoother', (t_switch, 2.5), fontsize=8, rotation=90, ha='center')
            else:
                ax3.annotate('→ESKF', (t_switch, 2.5), fontsize=8, rotation=90, ha='center')

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Disagreement [deg]')
    ax3.set_title('ESKF-Smoother Disagreement')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fault_recovery_demo.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved to fault_recovery_demo.pdf")
    plt.close()

    # === Print statistics ===
    print("\n" + "="*70)
    print("FAULT RECOVERY ANALYSIS")
    print("="*70)

    # Post-fault statistics
    post_fault_mask_eskf = eskf_times >= fault_onset
    post_fault_mask_red = redundant['times'] >= fault_onset

    eskf_only_post = eskf_errors[post_fault_mask_eskf]
    redundant_eskf_post = redundant['eskf_errors'][post_fault_mask_red]
    smoother_post = redundant['smoother_errors'][post_fault_mask_red]

    print(f"\nPost-fault error statistics (t >= {fault_onset}s):")
    print("-"*70)
    print(f"{'Estimator':<25} {'Mean Error':<15} {'Max Error':<15} {'RMS Error':<15}")
    print("-"*70)
    print(f"{'ESKF only (no recovery)':<25} {np.mean(eskf_only_post):>12.2f}° {np.max(eskf_only_post):>12.2f}° {np.sqrt(np.mean(eskf_only_post**2)):>12.2f}°")
    print(f"{'Redundant ESKF':<25} {np.mean(redundant_eskf_post):>12.2f}° {np.max(redundant_eskf_post):>12.2f}° {np.sqrt(np.mean(redundant_eskf_post**2)):>12.2f}°")
    print(f"{'Smoother':<25} {np.mean(smoother_post):>12.2f}° {np.max(smoother_post):>12.2f}° {np.sqrt(np.mean(smoother_post**2)):>12.2f}°")

    print(f"\nSwitch events: {len(redundant['switch_events'])}")
    for t_switch, direction in redundant['switch_events']:
        if t_switch >= fault_onset:
            print(f"  t={t_switch:.1f}s: {direction}")

    # Compute improvement
    improvement = (np.mean(eskf_only_post) - np.mean(redundant_eskf_post)) / np.mean(eskf_only_post) * 100
    print(f"\nRedundant architecture reduces mean error by {improvement:.1f}%")

    # Is smoother better than ESKF during fault?
    smoother_better_count = np.sum(smoother_post < redundant_eskf_post)
    total_count = len(smoother_post)
    print(f"Smoother has lower error than ESKF in {smoother_better_count}/{total_count} ({smoother_better_count/total_count*100:.1f}%) of post-fault samples")

    print("="*70)


if __name__ == "__main__":
    main()
