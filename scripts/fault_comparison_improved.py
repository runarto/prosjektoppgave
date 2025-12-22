#!/usr/bin/env python3
"""
Fault Comparison with Improved Switching Logic: ESKF vs iSAM2 vs Redundant.

Tests both star tracker fault and magnetometer fault with the innovation-based
switching logic.
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
    """Run ESKF, Smoother, and Redundant estimator."""
    att_err_rad = np.deg2rad(5.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize estimators
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

    x_eskf = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )
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

        # === Standalone ESKF ===
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)

        # === Standalone Smoother ===
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Prepare measurements with fault injection
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
                err_axis = np.array([1, 0.5, 0.3]) / np.linalg.norm([1, 0.5, 0.3])
                z_st = z_st @ Quaternion.from_avec(np.deg2rad(fault_error_deg) * err_axis)

        # === Standalone ESKF updates ===
        if z_mag is not None:
            try: x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except: pass
        if z_sun is not None:
            try: x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except: pass
        if z_st is not None:
            try: x_eskf = eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except: pass

        # === Standalone Smoother updates ===
        has_meas = z_mag is not None or z_sun is not None or z_st is not None
        if has_meas:
            smoother_state = smoother.add_measurement(t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_eci=B_n, s_eci=s_n)
        else:
            smoother_state = smoother.get_propagated_state()

        # === Redundant estimator ===
        x_redundant, redundant_smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        # Get redundant output
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

    # Define fault scenarios
    fault_st = {'onset_time': 100.0, 'end_time': 160.0, 'sensor': 'star_tracker', 'error_deg': 5.0}
    fault_mag = {'onset_time': 100.0, 'end_time': 160.0, 'sensor': 'magnetometer', 'bias': np.array([0.5, 0.3, 0.2])}

    print("\nRunning star tracker fault scenario...")
    res_st = run_all_estimators(sim_data, config_path, fault_st)

    print("Running magnetometer fault scenario...")
    res_mag = run_all_estimators(sim_data, config_path, fault_mag)

    # Create figure with 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Fault Comparison with Per-Sensor NIS Tracking', fontsize=12, fontweight='bold')

    t_start = 60.0
    fault_onset, fault_end = 100.0, 160.0

    scenarios = [
        (axes[0], res_st, 'Star Tracker Fault (5° error)'),
        (axes[1], res_mag, 'Magnetometer Fault (|b|=0.62)'),
    ]

    for ax_row, res, title in scenarios:
        t = res['times']
        mask = t >= t_start

        # Left: Attitude errors
        ax = ax_row[0]
        ax.plot(t[mask], res['smoother_errors'][mask], 'g-', lw=1.2, label='iSAM2', alpha=0.8)
        ax.plot(t[mask], res['eskf_errors'][mask], 'b-', lw=1.2, label='ESKF', alpha=0.8)
        ax.plot(t[mask], res['redundant_errors'][mask], 'r--', lw=2.0, label='Redundant', alpha=0.9)

        ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
        ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
        ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

        ax.set_ylabel('Attitude Error [deg]')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([t_start, t[-1]])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Middle: Disagreement + mode indicator
        ax = ax_row[1]

        # Add mode regions as background colors
        primaries = res['primaries']
        mode_colors = {'ESKF': 'C0', 'CONSERVATIVE': 'C2', 'SMOOTHER': 'C1'}

        # Find mode transitions and draw regions
        current_mode = primaries[0]
        region_start = t[0]
        for i in range(1, len(primaries)):
            if primaries[i] != current_mode or i == len(primaries) - 1:
                region_end = t[i] if primaries[i] != current_mode else t[-1]
                if region_start >= t_start or region_end >= t_start:
                    ax.axvspan(max(region_start, t_start), region_end,
                              color=mode_colors[current_mode], alpha=0.15)
                region_start = t[i]
                current_mode = primaries[i]

        ax.plot(t[mask], res['disagreements'][mask], 'purple', lw=1.2, alpha=0.8)

        ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Threshold')
        ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.2, hatch='//')
        ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
        ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

        ax.set_ylabel('Disagreement [deg]')
        ax.set_title('Disagreement & Mode')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([t_start, t[-1]])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Right: Sensor Health Scores
        ax = ax_row[2]
        ax.plot(t[mask], res['sensor_scores_mag'][mask], 'b-', lw=1.5, label='Mag', alpha=0.9)
        ax.plot(t[mask], res['sensor_scores_sun'][mask], 'orange', lw=1.5, label='Sun', alpha=0.9)
        ax.plot(t[mask], res['sensor_scores_st'][mask], 'g-', lw=1.5, label='ST', alpha=0.9)

        ax.axhline(0.8, color='red', ls='--', lw=1.5, alpha=0.7, label='Threshold')
        ax.axvspan(fault_onset, fault_end, color='gray', alpha=0.15)
        ax.axvline(fault_onset, color='k', ls='--', alpha=0.4, lw=1)
        ax.axvline(fault_end, color='k', ls='--', alpha=0.4, lw=1)

        ax.set_ylabel('Health Score')
        ax.set_title('Sensor NIS Health')
        ax.set_ylim([0, 1.1])
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([t_start, t[-1]])
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 2].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig('fault_comparison_improved.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fault_comparison_improved.png', dpi=150, bbox_inches='tight')
    print("\nSaved to fault_comparison_improved.pdf/png")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("FAULT COMPARISON WITH IMPROVED SWITCHING LOGIC")
    print("="*70)

    for name, res, fault in [('Star Tracker Fault', res_st, fault_st),
                              ('Magnetometer Fault', res_mag, fault_mag)]:
        t = res['times']
        during = (t >= fault_onset) & (t < fault_end)
        post = (t >= fault_end + 10) & (t <= fault_end + 60)

        max_dis = np.max(res['disagreements'][during | post])
        eskf_during = np.mean(res['eskf_errors'][during])
        smoother_during = np.mean(res['smoother_errors'][during])
        redundant_during = np.mean(res['redundant_errors'][during])
        eskf_post = np.mean(res['eskf_errors'][post])
        smoother_post = np.mean(res['smoother_errors'][post])
        redundant_post = np.mean(res['redundant_errors'][post])

        # Count modes
        mode_counts = {'ESKF': 0, 'SMOOTHER': 0, 'CONSERVATIVE': 0}
        for p in res['primaries']:
            mode_counts[p] = mode_counts.get(p, 0) + 1

        print(f"\n{name}:")
        print(f"  Max disagreement: {max_dis:.2f}°")
        print(f"  During fault:")
        print(f"    ESKF:      {eskf_during:.3f}°")
        print(f"    iSAM2:     {smoother_during:.3f}°")
        print(f"    Redundant: {redundant_during:.3f}°")
        print(f"  Post-fault:")
        print(f"    ESKF:      {eskf_post:.3f}°")
        print(f"    iSAM2:     {smoother_post:.3f}°")
        print(f"    Redundant: {redundant_post:.3f}°")
        print(f"  Mode distribution: ESKF={mode_counts['ESKF']}, CONSERVATIVE={mode_counts['CONSERVATIVE']}, SMOOTHER={mode_counts['SMOOTHER']}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
