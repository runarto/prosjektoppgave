#!/usr/bin/env python3
"""
Magnetometer Fault Demonstration.

Injects a bias into magnetometer measurements and compares:
1. ESKF alone
2. iSAM2/Smoother alone
3. Redundant architecture (with switching)

Shows disagreement evolution and recovery capability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sqlite3

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

# Better plot styling
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_estimators(sim_data, config_path, fault_config):
    """
    Run ESKF and Smoother in parallel, injecting magnetometer fault.

    Returns detailed results for analysis.
    """
    att_err_rad = np.deg2rad(5.0)  # Smaller initial error
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize both estimators
    eskf = ESKF(P0=P0, config_path=config_path)
    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=True,
    )

    # Initial state
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_eskf = EskfState(nom=nom0, err=err0)

    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    # Results storage
    results = {
        'times': [],
        'eskf_errors': [],
        'smoother_errors': [],
        'disagreements': [],
        'fault_active': [],
        'mag_measurement_times': [],
        'mag_residuals_eskf': [],
    }

    fault_onset = fault_config.get('onset_time', float('inf'))
    fault_end = fault_config.get('end_time', float('inf'))
    fault_sensor = fault_config.get('sensor', 'magnetometer')
    fault_bias = fault_config.get('bias', np.zeros(3))
    fault_attitude_error_deg = fault_config.get('attitude_error_deg', 0.0)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()

        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        # Check fault status
        fault_active = fault_onset <= t < fault_end

        # === ESKF Prediction ===
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)

        # === Smoother gyro integration ===
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        # Get reference vectors
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # Prepare measurements (with fault injection)
        z_mag = None
        z_sun = None
        z_st = None

        if not np.any(np.isnan(sim_data.mag_meas[k])):
            z_mag = sim_data.mag_meas[k].copy()
            if fault_active and fault_sensor == 'magnetometer':
                z_mag = z_mag + fault_bias  # Inject bias
            results['mag_measurement_times'].append(t)

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            z_sun = sim_data.sun_meas[k]

        if not np.any(np.isnan(sim_data.st_meas[k])):
            z_st = Quaternion.from_array(sim_data.st_meas[k])
            if fault_active and fault_sensor == 'star_tracker':
                # Inject attitude error into star tracker measurement
                error_rad = np.deg2rad(fault_attitude_error_deg)
                error_axis = np.array([1, 0.5, 0.3])
                error_axis = error_axis / np.linalg.norm(error_axis)
                error_avec = error_rad * error_axis
                q_error = Quaternion.from_avec(error_avec)
                z_st = z_st @ q_error  # Add error to measurement

        # === ESKF Updates ===
        if z_mag is not None:
            try:
                x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        if z_sun is not None:
            try:
                x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        if z_st is not None:
            try:
                x_eskf = eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # === Smoother Updates ===
        has_measurement = z_mag is not None or z_sun is not None or z_st is not None
        if has_measurement:
            smoother_state = smoother.add_measurement(
                t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=B_n, s_eci=s_n,
            )
        else:
            smoother_state = smoother.get_state()

        # === Compute metrics ===
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_eskf.nom.ori, q_true)

        if smoother_state is not None:
            smoother_err = compute_attitude_error(smoother_state.ori, q_true)
            # Disagreement
            dq = x_eskf.nom.ori.conjugate() @ smoother_state.ori
            disagreement = np.rad2deg(2 * np.arccos(np.clip(abs(dq.mu), 0, 1)))
        else:
            smoother_err = eskf_err
            disagreement = 0.0

        # Store results
        results['times'].append(t)
        results['eskf_errors'].append(eskf_err)
        results['smoother_errors'].append(smoother_err)
        results['disagreements'].append(disagreement)
        results['fault_active'].append(fault_active)

    # Convert to arrays
    for key in ['times', 'eskf_errors', 'smoother_errors', 'disagreements', 'fault_active']:
        results[key] = np.array(results[key])

    return results


def main():
    config_path = "configs/config_baseline_short.yaml"

    # Load simulation data
    print("Loading simulation data...")
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()
    print(f"Using simulation ID: {sim_id}")
    sim_data = db.load_run(sim_id)

    # Star tracker fault configuration
    # Transient error: starts at t=100s, ends at t=160s
    fault_config = {
        'onset_time': 100.0,
        'end_time': 160.0,  # Transient fault
        'sensor': 'star_tracker',  # Which sensor to corrupt
        'attitude_error_deg': 5.0,  # 5 degree attitude error in star tracker
    }

    print(f"\nFault configuration:")
    print(f"  Type: Star tracker attitude error")
    print(f"  Onset: t={fault_config['onset_time']}s")
    print(f"  End: t={fault_config['end_time']}s (transient)")
    print(f"  Error magnitude: {fault_config['attitude_error_deg']}°")

    print("\nRunning estimators...")
    results = run_estimators(sim_data, config_path, fault_config)

    # === Create figure ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    t = results['times']
    fault_onset = fault_config['onset_time']
    fault_end = fault_config['end_time']

    # Only show from t=50s onwards (after initial convergence)
    t_start = 50.0
    mask = t >= t_start

    # === Plot 1: Attitude errors ===
    ax1 = axes[0]
    ax1.plot(t[mask], results['eskf_errors'][mask], 'b-', linewidth=1.5, label='ESKF')
    ax1.plot(t[mask], results['smoother_errors'][mask], 'g-', linewidth=1.5, label='iSAM2 Smoother')

    # Shade fault period
    ax1.axvspan(fault_onset, fault_end, color='red', alpha=0.15, label='Fault active')
    ax1.axvline(fault_onset, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(fault_end, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax1.set_ylabel('Attitude Error [deg]')
    fault_type = fault_config.get('sensor', 'magnetometer').replace('_', ' ').title()
    ax1.set_title(f'Attitude Estimation Error: ESKF vs iSAM2 Smoother ({fault_type} Fault)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([t_start, t[-1]])

    # Set reasonable y-limits
    max_err = max(np.max(results['eskf_errors'][mask]), np.max(results['smoother_errors'][mask]))
    ax1.set_ylim([0, min(max_err * 1.2, 20)])

    # === Plot 2: Disagreement ===
    ax2 = axes[1]
    ax2.plot(t[mask], results['disagreements'][mask], 'purple', linewidth=1.5)
    ax2.fill_between(t[mask], results['disagreements'][mask], alpha=0.3, color='purple')

    # Shade fault period
    ax2.axvspan(fault_onset, fault_end, color='red', alpha=0.15)
    ax2.axvline(fault_onset, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(fault_end, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Threshold line
    ax2.axhline(2.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Detection threshold (2°)')

    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('ESKF - iSAM2 Disagreement (Fault Detection Signal)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([t_start, t[-1]])
    ax2.set_ylim([0, max(np.max(results['disagreements'][mask]) * 1.2, 5)])

    # === Plot 3: Error comparison (zoomed on fault period) ===
    ax3 = axes[2]

    # Focus on fault period +/- 30s
    fault_mask = (t >= fault_onset - 30) & (t <= fault_end + 50)

    ax3.plot(t[fault_mask], results['eskf_errors'][fault_mask], 'b-', linewidth=2, label='ESKF')
    ax3.plot(t[fault_mask], results['smoother_errors'][fault_mask], 'g-', linewidth=2, label='iSAM2 Smoother')

    # Shade fault period
    ax3.axvspan(fault_onset, fault_end, color='red', alpha=0.15, label='Fault active')
    ax3.axvline(fault_onset, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax3.axvline(fault_end, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Attitude Error [deg]')
    ax3.set_title('Zoomed View: Fault Period and Recovery')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([fault_onset - 30, fault_end + 50])

    plt.tight_layout()
    plt.savefig('magnetometer_fault_demo.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('magnetometer_fault_demo.png', dpi=150, bbox_inches='tight')
    print("\nSaved to magnetometer_fault_demo.pdf and .png")
    plt.close()

    # === Statistics ===
    print("\n" + "="*60)
    print("MAGNETOMETER FAULT ANALYSIS")
    print("="*60)

    # Pre-fault (steady state)
    pre_mask = (t >= 70) & (t < fault_onset)
    during_mask = (t >= fault_onset) & (t < fault_end)
    post_mask = (t >= fault_end + 10) & (t <= fault_end + 60)  # Recovery period

    print(f"\n{'Phase':<20} {'ESKF [deg]':<15} {'Smoother [deg]':<15} {'Disagreement [deg]':<20}")
    print("-"*70)

    print(f"{'Pre-fault (mean)':<20} {np.mean(results['eskf_errors'][pre_mask]):>12.3f} {np.mean(results['smoother_errors'][pre_mask]):>14.3f} {np.mean(results['disagreements'][pre_mask]):>18.3f}")
    print(f"{'During fault (mean)':<20} {np.mean(results['eskf_errors'][during_mask]):>12.3f} {np.mean(results['smoother_errors'][during_mask]):>14.3f} {np.mean(results['disagreements'][during_mask]):>18.3f}")
    print(f"{'Post-fault (mean)':<20} {np.mean(results['eskf_errors'][post_mask]):>12.3f} {np.mean(results['smoother_errors'][post_mask]):>14.3f} {np.mean(results['disagreements'][post_mask]):>18.3f}")

    print(f"\n{'During fault (max)':<20} {np.max(results['eskf_errors'][during_mask]):>12.3f} {np.max(results['smoother_errors'][during_mask]):>14.3f} {np.max(results['disagreements'][during_mask]):>18.3f}")

    # Detection analysis
    disagreement_during = results['disagreements'][during_mask]
    detection_threshold = 2.0
    detected = np.any(disagreement_during > detection_threshold)
    if detected:
        first_detection_idx = np.argmax(disagreement_during > detection_threshold)
        detection_delay = first_detection_idx * 0.02  # Assuming 50Hz
        print(f"\nFault detected: Yes (disagreement > {detection_threshold}°)")
        print(f"Detection delay: ~{detection_delay:.1f}s after fault onset")
    else:
        print(f"\nFault detected: No (disagreement never exceeded {detection_threshold}°)")

    # Recovery analysis
    if len(results['eskf_errors'][post_mask]) > 0:
        eskf_recovered = np.mean(results['eskf_errors'][post_mask]) < 1.0
        smoother_recovered = np.mean(results['smoother_errors'][post_mask]) < 1.0
        print(f"\nRecovery (error < 1°):")
        print(f"  ESKF: {'Yes' if eskf_recovered else 'No'} (mean={np.mean(results['eskf_errors'][post_mask]):.3f}°)")
        print(f"  Smoother: {'Yes' if smoother_recovered else 'No'} (mean={np.mean(results['smoother_errors'][post_mask]):.3f}°)")

    print("="*60)


if __name__ == "__main__":
    main()
