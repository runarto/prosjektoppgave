#!/usr/bin/env python3
"""
Transient Fault Recovery Demo.

Shows the value of redundancy for TRANSIENT faults:
- Fault occurs for limited duration, then clears
- ESKF alone may not recover (bad state, wrong bias estimate)
- Redundant architecture: smoother keeps good estimate, resets ESKF after fault clears

This is where redundancy truly shines.
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


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_eskf_only(sim_data, config_path, fault_config=None):
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
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        # TRANSIENT fault: only active for limited duration
        if fault_config:
            onset = fault_config.get('onset_time', float('inf'))
            duration = fault_config.get('duration', float('inf'))
            if onset <= t < onset + duration:
                if np.linalg.norm(injected_bias) < 1e-10:
                    injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                    injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
                omega_k = omega_k + injected_bias
            else:
                injected_bias = np.zeros(3)  # Fault cleared

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
    selected_errors = []
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        # TRANSIENT fault
        if fault_config:
            onset = fault_config.get('onset_time', float('inf'))
            duration = fault_config.get('duration', float('inf'))
            if onset <= t < onset + duration:
                if np.linalg.norm(injected_bias) < 1e-10:
                    injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                    injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
                omega_k = omega_k + injected_bias
            else:
                injected_bias = np.zeros(3)

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

    # TRANSIENT fault: 30 seconds of bad gyro, then it clears
    fault_onset = 120.0
    fault_duration = 30.0
    fault_magnitude = 0.05  # Larger fault for visibility

    fault_config = {
        'type': 'gyro_bias_step',
        'onset_time': fault_onset,
        'duration': fault_duration,
        'magnitude': fault_magnitude
    }

    print(f"\nTransient fault: {fault_magnitude*1000:.0f} mrad/s from t={fault_onset}s to t={fault_onset+fault_duration}s")

    print("\nRunning ESKF only...")
    eskf_times, eskf_errors = run_eskf_only(sim_data, config_path, fault_config)

    print("Running Redundant...")
    redundant = run_redundant(sim_data, config_path, fault_config)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    t_start = 80.0
    mask_eskf = eskf_times >= t_start
    mask_red = redundant['times'] >= t_start

    # === Plot 1: Error comparison ===
    ax1 = axes[0]
    ax1.semilogy(eskf_times[mask_eskf], eskf_errors[mask_eskf], 'r-',
                 linewidth=1.5, alpha=0.9, label='ESKF only')
    ax1.semilogy(redundant['times'][mask_red], redundant['smoother_errors'][mask_red], 'g-',
                 linewidth=1.5, alpha=0.9, label='Smoother')
    ax1.semilogy(redundant['times'][mask_red], redundant['selected_errors'][mask_red], 'b--',
                 linewidth=2, alpha=0.9, label='Redundant output')

    # Mark fault period
    ax1.axvspan(fault_onset, fault_onset + fault_duration, color='red', alpha=0.2, label='Fault active')
    ax1.axvline(fault_onset, color='black', linestyle='-', linewidth=1)
    ax1.axvline(fault_onset + fault_duration, color='black', linestyle='-', linewidth=1)

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
    ax1.set_title(f'TRANSIENT Fault Recovery: {fault_magnitude*1000:.0f} mrad/s bias for {fault_duration:.0f}s (t={fault_onset:.0f}s-{fault_onset+fault_duration:.0f}s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-2, 100])

    # === Plot 2: Disagreement ===
    ax2 = axes[1]
    ax2.plot(redundant['times'][mask_red], redundant['disagreements'][mask_red], 'purple', linewidth=1)
    ax2.fill_between(redundant['times'][mask_red], redundant['disagreements'][mask_red],
                     alpha=0.3, color='purple')
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
    ax2.axvspan(fault_onset, fault_onset + fault_duration, color='red', alpha=0.2)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('ESKF-Smoother Disagreement')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transient_fault_demo.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved to transient_fault_demo.pdf")
    plt.close()

    # === Statistics ===
    print("\n" + "="*70)
    print("TRANSIENT FAULT RECOVERY ANALYSIS")
    print("="*70)

    # Three phases: pre-fault, during-fault, post-fault
    pre_mask_eskf = (eskf_times >= t_start) & (eskf_times < fault_onset)
    during_mask_eskf = (eskf_times >= fault_onset) & (eskf_times < fault_onset + fault_duration)
    post_mask_eskf = eskf_times >= fault_onset + fault_duration

    pre_mask_red = (redundant['times'] >= t_start) & (redundant['times'] < fault_onset)
    during_mask_red = (redundant['times'] >= fault_onset) & (redundant['times'] < fault_onset + fault_duration)
    post_mask_red = redundant['times'] >= fault_onset + fault_duration

    print(f"\n{'Phase':<20} {'ESKF only':<15} {'Smoother':<15} {'Redundant':<15}")
    print("-"*70)

    # Pre-fault
    print(f"{'Pre-fault (mean)':<20} {np.mean(eskf_errors[pre_mask_eskf]):>12.3f}° {np.mean(redundant['smoother_errors'][pre_mask_red]):>12.3f}° {np.mean(redundant['selected_errors'][pre_mask_red]):>12.3f}°")

    # During fault
    print(f"{'During fault (mean)':<20} {np.mean(eskf_errors[during_mask_eskf]):>12.2f}° {np.mean(redundant['smoother_errors'][during_mask_red]):>12.2f}° {np.mean(redundant['selected_errors'][during_mask_red]):>12.2f}°")

    # Post-fault (RECOVERY phase)
    print(f"{'Post-fault (mean)':<20} {np.mean(eskf_errors[post_mask_eskf]):>12.3f}° {np.mean(redundant['smoother_errors'][post_mask_red]):>12.3f}° {np.mean(redundant['selected_errors'][post_mask_red]):>12.3f}°")

    print("-"*70)

    # Recovery analysis
    eskf_post = eskf_errors[post_mask_eskf]
    red_post = redundant['selected_errors'][post_mask_red]
    smoother_post = redundant['smoother_errors'][post_mask_red]

    print(f"\nPost-fault recovery (after fault clears at t={fault_onset+fault_duration}s):")
    print(f"  ESKF only mean error:    {np.mean(eskf_post):.3f}°")
    print(f"  Redundant mean error:    {np.mean(red_post):.3f}°")
    print(f"  Improvement:             {(np.mean(eskf_post) - np.mean(red_post)) / np.mean(eskf_post) * 100:.1f}%")

    print(f"\nSwitch events:")
    for t_sw, direction in redundant['switch_events']:
        if t_sw >= t_start:
            print(f"  t={t_sw:.1f}s: {direction}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
