#!/usr/bin/env python3
"""
Sensor Fault Analysis: ESKF vs iSAM2 Smoother.

Demonstrates disagreement evolution and recovery capability under sensor faults.
Clean, publication-quality plots.
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

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['lines.linewidth'] = 1.5
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_estimators(sim_data, config_path, fault_config):
    """Run ESKF and Smoother in parallel with fault injection."""
    att_err_rad = np.deg2rad(5.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    eskf = ESKF(P0=P0, config_path=config_path)
    smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=60.0, use_robust=True)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_eskf = EskfState(nom=nom0, err=err0)
    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    results = {
        'times': [], 'eskf_errors': [], 'smoother_errors': [],
        'disagreements': [], 'fault_active': [],
    }

    fault_onset = fault_config.get('onset_time', float('inf'))
    fault_end = fault_config.get('end_time', float('inf'))
    fault_sensor = fault_config.get('sensor', 'star_tracker')
    fault_error_deg = fault_config.get('error_deg', 5.0)
    fault_bias = fault_config.get('bias', np.zeros(3))

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        fault_active = fault_onset <= t < fault_end

        # ESKF prediction
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Prepare measurements
        z_mag = None
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            z_mag = sim_data.mag_meas[k].copy()
            if fault_active and fault_sensor == 'magnetometer':
                z_mag = z_mag + fault_bias

        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = None

        if not np.any(np.isnan(sim_data.st_meas[k])):
            z_st = Quaternion.from_array(sim_data.st_meas[k])
            if fault_active and fault_sensor == 'star_tracker':
                error_rad = np.deg2rad(fault_error_deg)
                error_axis = np.array([1, 0.5, 0.3]) / np.linalg.norm([1, 0.5, 0.3])
                z_st = z_st @ Quaternion.from_avec(error_rad * error_axis)

        # ESKF updates
        if z_mag is not None:
            try: x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError: pass
        if z_sun is not None:
            try: x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError: pass
        if z_st is not None:
            try: x_eskf = eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except ValueError: pass

        # Smoother updates
        has_meas = z_mag is not None or z_sun is not None or z_st is not None
        if has_meas:
            smoother_state = smoother.add_measurement(t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_eci=B_n, s_eci=s_n)
        else:
            smoother_state = smoother.get_state()

        # Compute metrics
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_eskf.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true) if smoother_state else eskf_err

        if smoother_state:
            dq = x_eskf.nom.ori.conjugate() @ smoother_state.ori
            disagreement = np.rad2deg(2 * np.arccos(np.clip(abs(dq.mu), 0, 1)))
        else:
            disagreement = 0.0

        results['times'].append(t)
        results['eskf_errors'].append(eskf_err)
        results['smoother_errors'].append(smoother_err)
        results['disagreements'].append(disagreement)
        results['fault_active'].append(fault_active)

    for key in results:
        results[key] = np.array(results[key])
    return results


def main():
    import sys
    config_path = "configs/config_baseline_short.yaml"

    print("Loading simulation data...")
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()
    print(f"Using simulation ID: {sim_id}")
    sim_data = db.load_run(sim_id)

    # Select fault type from command line or default to star_tracker
    fault_type = sys.argv[1] if len(sys.argv) > 1 else 'star_tracker'

    if fault_type == 'magnetometer':
        fault_config = {
            'onset_time': 100.0,
            'end_time': 160.0,
            'sensor': 'magnetometer',
            'bias': np.array([0.5, 0.3, 0.2]),  # Large bias
        }
        print(f"\nFault: Magnetometer bias (|b|={np.linalg.norm(fault_config['bias']):.2f}), t=[{fault_config['onset_time']}, {fault_config['end_time']}]s")
    else:
        fault_config = {
            'onset_time': 100.0,
            'end_time': 160.0,
            'sensor': 'star_tracker',
            'error_deg': 5.0,
        }
        print(f"\nFault: {fault_config['error_deg']}° star tracker error, t=[{fault_config['onset_time']}, {fault_config['end_time']}]s")
    print("Running estimators...")
    results = run_estimators(sim_data, config_path, fault_config)

    t = results['times']
    fault_onset, fault_end = fault_config['onset_time'], fault_config['end_time']

    # === Create figure: 2 plots ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    t_start = 60.0
    mask = t >= t_start

    # --- Plot 1: Attitude Errors ---
    ax1.plot(t[mask], results['eskf_errors'][mask], 'b-', label='ESKF', zorder=3)
    ax1.plot(t[mask], results['smoother_errors'][mask], 'g-', label='iSAM2 Smoother', zorder=3)

    # Fault region
    ax1.axvspan(fault_onset, fault_end, color='red', alpha=0.12, label='Fault period', zorder=1)
    ax1.axvline(fault_onset, color='r', linestyle='--', alpha=0.6, linewidth=1)
    ax1.axvline(fault_end, color='r', linestyle='--', alpha=0.6, linewidth=1)

    ax1.set_ylabel('Attitude Error [deg]')
    if fault_config['sensor'] == 'star_tracker':
        ax1.set_title(f'Transient Star Tracker Fault ({fault_config["error_deg"]}° error)')
    else:
        ax1.set_title(f'Transient Magnetometer Fault (bias |b|={np.linalg.norm(fault_config["bias"]):.2f})')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # --- Plot 2: Disagreement ---
    ax2.plot(t[mask], results['disagreements'][mask], color='purple', label='ESKF−Smoother disagreement')
    ax2.fill_between(t[mask], results['disagreements'][mask], alpha=0.25, color='purple')

    ax2.axhline(2.0, color='orange', linestyle='--', linewidth=1.5, label='Detection threshold')
    ax2.axvspan(fault_onset, fault_end, color='red', alpha=0.12, zorder=1)
    ax2.axvline(fault_onset, color='r', linestyle='--', alpha=0.6, linewidth=1)
    ax2.axvline(fault_end, color='r', linestyle='--', alpha=0.6, linewidth=1)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('Estimator Disagreement (Fault Detection Signal)')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim([t_start, t[-1]])

    plt.tight_layout()
    plt.savefig('sensor_fault_analysis.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('sensor_fault_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved to sensor_fault_analysis.pdf/png")
    plt.close()

    # === Statistics table ===
    pre_mask = (t >= 70) & (t < fault_onset)
    during_mask = (t >= fault_onset) & (t < fault_end)
    post_mask = (t >= fault_end + 10) & (t <= fault_end + 60)

    print("\n" + "="*65)
    print("RESULTS SUMMARY")
    print("="*65)
    print(f"\n{'Phase':<18} {'ESKF':<12} {'Smoother':<12} {'Disagreement':<12}")
    print("-"*54)
    print(f"{'Pre-fault':<18} {np.mean(results['eskf_errors'][pre_mask]):>8.3f}°   {np.mean(results['smoother_errors'][pre_mask]):>8.3f}°   {np.mean(results['disagreements'][pre_mask]):>8.3f}°")
    print(f"{'During fault':<18} {np.mean(results['eskf_errors'][during_mask]):>8.3f}°   {np.mean(results['smoother_errors'][during_mask]):>8.3f}°   {np.mean(results['disagreements'][during_mask]):>8.3f}°")
    print(f"{'Post-fault':<18} {np.mean(results['eskf_errors'][post_mask]):>8.3f}°   {np.mean(results['smoother_errors'][post_mask]):>8.3f}°   {np.mean(results['disagreements'][post_mask]):>8.3f}°")
    print("-"*54)

    eskf_recovered = np.mean(results['eskf_errors'][post_mask]) < 0.5
    smoother_recovered = np.mean(results['smoother_errors'][post_mask]) < 0.5

    print(f"\nRecovery assessment (error < 0.5°):")
    print(f"  ESKF:     {'RECOVERED' if eskf_recovered else 'NOT RECOVERED'} ({np.mean(results['eskf_errors'][post_mask]):.3f}°)")
    print(f"  Smoother: {'RECOVERED' if smoother_recovered else 'NOT RECOVERED'} ({np.mean(results['smoother_errors'][post_mask]):.3f}°)")

    max_disagreement = np.max(results['disagreements'][during_mask | post_mask])
    print(f"\nMax disagreement: {max_disagreement:.2f}° (threshold: 2.0°)")
    print(f"Fault detectable: {'YES' if max_disagreement > 2.0 else 'NO'}")
    print("="*65)


if __name__ == "__main__":
    main()
