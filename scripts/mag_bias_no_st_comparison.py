#!/usr/bin/env python3
"""
Magnetometer Bias Without Star Tracker: ESKF vs iSAM2 vs Redundant.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sqlite3

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 9


def compute_attitude_error(q_est, q_true):
    q_err = q_true @ q_est.conjugate()
    return np.rad2deg(2 * np.arccos(np.clip(abs(q_err.mu), 0, 1)))


def run_all_estimators(sim_data, config_path, fault_config):
    """Run ESKF, Smoother, and Redundant estimator without star tracker."""
    att_err_rad = np.deg2rad(5.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize all three estimators
    eskf = ESKF(P0=P0, config_path=config_path)
    smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=60.0, use_robust=True)
    redundant = RedundantEstimator(
        P0=P0, config_path=config_path, smoother_lag=60.0, use_robust=True,
        disagreement_threshold_deg=2.0, consecutive_disagreements_to_switch=5,
    )

    # Initial state
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad]*3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    # ESKF state (standalone)
    x_eskf = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    # Redundant ESKF state
    x_redundant = EskfState(
        nom=NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    results = {
        'times': [],
        'eskf_errors': [],
        'smoother_errors': [],
        'redundant_errors': [],
        'disagreements': [],
        'primaries': [],
        'sensor_scores_mag': [],
        'sensor_scores_sun': [],
        'sensor_scores_st': [],
    }

    fault_onset = fault_config.get('onset_time', float('inf'))
    fault_end = fault_config.get('end_time', float('inf'))
    fault_bias = fault_config.get('bias', np.zeros(3))

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        fault_active = fault_onset <= t < fault_end

        # === Standalone ESKF ===
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)

        # === Standalone Smoother ===
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Prepare measurements - NO STAR TRACKER
        z_mag = None
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            z_mag = sim_data.mag_meas[k].copy()
            if fault_active:
                z_mag = z_mag + fault_bias

        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = None  # Star tracker disabled

        # === Standalone ESKF updates ===
        if z_mag is not None:
            try: x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except: pass
        if z_sun is not None:
            try: x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except: pass

        # === Standalone Smoother updates ===
        has_meas = z_mag is not None or z_sun is not None
        if has_meas:
            smoother_state = smoother.add_measurement(t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=None, B_eci=B_n, s_eci=s_n)
        else:
            smoother_state = smoother.get_propagated_state()

        # === Redundant estimator ===
        x_redundant, redundant_smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=None, B_n=B_n, s_n=s_n,
        )

        # Get redundant output (ESKF for both ESKF and CONSERVATIVE modes)
        if primary == 'SMOOTHER':
            redundant_state = redundant_smoother_state
        else:  # ESKF or CONSERVATIVE
            redundant_state = x_redundant.nom

        # Compute errors
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_eskf.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true) if smoother_state else eskf_err
        redundant_err = compute_attitude_error(redundant_state.ori, q_true)

        # Get sensor health scores
        sensor_scores = redundant._get_sensor_health_scores()

        results['times'].append(t)
        results['eskf_errors'].append(eskf_err)
        results['smoother_errors'].append(smoother_err)
        results['redundant_errors'].append(redundant_err)
        results['disagreements'].append(disagreement)
        results['primaries'].append(primary)
        results['sensor_scores_mag'].append(sensor_scores['mag'])
        results['sensor_scores_sun'].append(sensor_scores['sun'])
        results['sensor_scores_st'].append(sensor_scores['st'])

    for k in ['times', 'eskf_errors', 'smoother_errors', 'redundant_errors', 'disagreements',
              'sensor_scores_mag', 'sensor_scores_sun', 'sensor_scores_st']:
        results[k] = np.array(results[k])
    return results


def main():
    config_path = "configs/config_baseline_short.yaml"

    print("Loading simulation data...")
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()
    sim_data = db.load_run(sim_id)

    # Magnetometer fault
    fault_config = {
        'onset_time': 100.0,
        'end_time': 160.0,
        'bias': np.array([0.5, 0.3, 0.2]),
    }

    print(f"\nScenario: Magnetometer Bias WITHOUT Star Tracker")
    print(f"  Bias: {fault_config['bias']} (|b|={np.linalg.norm(fault_config['bias']):.3f})")
    print(f"  Fault period: t=[{fault_config['onset_time']}, {fault_config['end_time']}]s")
    print(f"  Star tracker: DISABLED\n")

    print("Running estimators...")
    results = run_all_estimators(sim_data, config_path, fault_config)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Magnetometer Bias Fault (No Star Tracker)', fontsize=12, fontweight='bold')

    t = results['times']
    t_start = 60.0
    fault_onset, fault_end = fault_config['onset_time'], fault_config['end_time']
    mask = t >= t_start

    # === Top left: Attitude errors ===
    ax = axes[0, 0]
    ax.plot(t[mask], results['smoother_errors'][mask], 'g-', lw=1.2, label='iSAM2', alpha=0.9)
    ax.plot(t[mask], results['eskf_errors'][mask], 'b-', lw=1.2, label='ESKF', alpha=0.9)
    ax.plot(t[mask], results['redundant_errors'][mask], 'r--', lw=2.5, label='Redundant', alpha=0.9, dashes=(5, 3))

    ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
    ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
    ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Attitude Error Comparison')
    ax.legend(loc='upper right')
    ax.set_xlim([t_start, t[-1]])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # === Top right: Disagreement ===
    ax = axes[0, 1]
    ax.plot(t[mask], results['disagreements'][mask], 'purple', lw=1.2, alpha=0.8)
    ax.fill_between(t[mask], results['disagreements'][mask], alpha=0.15, color='purple')

    ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Detection threshold')
    ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
    ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
    ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

    # Mark switches
    primaries = results['primaries']
    in_smoother = False
    for i, (ti, p) in enumerate(zip(t[mask], [primaries[j] for j in range(len(primaries)) if t[j] >= t_start])):
        if p == 'SMOOTHER' and not in_smoother:
            in_smoother = True
            ax.axvline(ti, color='red', ls=':', alpha=0.7, lw=1.5, label='Switch to Smoother' if i == 0 else '')
        elif p == 'ESKF' and in_smoother:
            in_smoother = False
            ax.axvline(ti, color='blue', ls=':', alpha=0.7, lw=1.5)

    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('ESKF − iSAM2 Disagreement')
    ax.legend(loc='upper right')
    ax.set_xlim([t_start, t[-1]])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # === Bottom left: Zoomed attitude errors ===
    ax = axes[1, 0]
    ax.plot(t[mask], results['smoother_errors'][mask], 'g-', lw=1.2, label='iSAM2', alpha=0.9)
    ax.plot(t[mask], results['eskf_errors'][mask], 'b-', lw=1.2, label='ESKF', alpha=0.9)
    ax.plot(t[mask], results['redundant_errors'][mask], 'r--', lw=2.5, label='Redundant', alpha=0.9, dashes=(5, 3))

    ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
    ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
    ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Attitude Error (Zoomed)')
    ax.legend(loc='upper right')
    ax.set_xlim([t_start, t[-1]])
    ax.set_ylim([0, 5])  # Zoom to see ESKF detail
    ax.grid(True, alpha=0.3)

    # === Bottom right: Sensor Health Scores ===
    ax = axes[1, 1]
    ax.plot(t[mask], results['sensor_scores_mag'][mask], 'b-', lw=1.5, label='Magnetometer', alpha=0.9)
    ax.plot(t[mask], results['sensor_scores_sun'][mask], 'orange', lw=1.5, label='Sun sensor', alpha=0.9)
    # Star tracker disabled in this scenario, but show for completeness
    ax.plot(t[mask], results['sensor_scores_st'][mask], 'g--', lw=1.5, label='Star tracker', alpha=0.5)

    ax.axhline(0.8, color='red', ls='--', lw=1.5, alpha=0.7, label='Health threshold')
    ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
    ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
    ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Health Score')
    ax.set_title('Per-Sensor NIS Health Scores')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([t_start, t[-1]])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mag_bias_no_st_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('mag_bias_no_st_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved to mag_bias_no_st_comparison.pdf/png")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    during = (t >= fault_onset) & (t < fault_end)
    post = (t >= fault_end + 10) & (t <= fault_end + 60)

    print(f"\nDuring fault (t=[{fault_onset}, {fault_end}]s):")
    print(f"  ESKF:      {np.mean(results['eskf_errors'][during]):.3f}°")
    print(f"  iSAM2:     {np.mean(results['smoother_errors'][during]):.3f}°")
    print(f"  Redundant: {np.mean(results['redundant_errors'][during]):.3f}°")

    print(f"\nPost-fault (t=[{fault_end+10}, {fault_end+60}]s):")
    print(f"  ESKF:      {np.mean(results['eskf_errors'][post]):.3f}°")
    print(f"  iSAM2:     {np.mean(results['smoother_errors'][post]):.3f}°")
    print(f"  Redundant: {np.mean(results['redundant_errors'][post]):.3f}°")

    print(f"\nMax disagreement: {np.max(results['disagreements']):.2f}°")

    # Count switches
    switches = sum(1 for i in range(1, len(primaries)) if primaries[i] != primaries[i-1])
    print(f"Number of switches: {switches}")

    print("="*70)


if __name__ == "__main__":
    main()
