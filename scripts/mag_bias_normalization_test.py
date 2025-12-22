#!/usr/bin/env python3
"""
Magnetometer Bias Test: With vs Without Normalization.

Tests whether removing normalization makes the M-estimator more effective
at detecting/rejecting biased magnetometer measurements.
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

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 9


def compute_attitude_error(q_est, q_true):
    q_err = q_true @ q_est.conjugate()
    return np.rad2deg(2 * np.arccos(np.clip(abs(q_err.mu), 0, 1)))


def run_estimators(sim_data, config_path, fault_config, normalize_mag=True):
    """Run ESKF and Smoother with specified normalization setting."""
    att_err_rad = np.deg2rad(5.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    eskf = ESKF(P0=P0, config_path=config_path)
    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=True,
        normalize_mag=normalize_mag,
    )

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad]*3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    x_eskf = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )
    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    results = {
        'times': [], 'eskf_errors': [], 'smoother_errors': [], 'disagreements': [],
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
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Prepare measurements with fault injection
        z_mag = None
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            z_mag = sim_data.mag_meas[k].copy()
            if fault_active:
                z_mag = z_mag + fault_bias

        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        # ESKF updates
        if z_mag is not None:
            try: x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except: pass
        if z_sun is not None:
            try: x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except: pass
        if z_st is not None:
            try: x_eskf = eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except: pass

        # Smoother updates
        has_meas = z_mag is not None or z_sun is not None or z_st is not None
        if has_meas:
            smoother_state = smoother.add_measurement(t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_eci=B_n, s_eci=s_n)
        else:
            smoother_state = smoother.get_propagated_state()

        # Compute errors
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

    for k in results:
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

    # Magnetometer fault configuration
    fault_config = {
        'onset_time': 100.0,
        'end_time': 160.0,
        'bias': np.array([0.5, 0.3, 0.2]),
    }
    bias_mag = np.linalg.norm(fault_config['bias'])

    print(f"\nMagnetometer bias: {fault_config['bias']} (|b|={bias_mag:.3f})")
    print(f"Fault period: t=[{fault_config['onset_time']}, {fault_config['end_time']}]s\n")

    print("Running with normalization (default)...")
    res_norm = run_estimators(sim_data, config_path, fault_config, normalize_mag=True)

    print("Running WITHOUT normalization...")
    res_raw = run_estimators(sim_data, config_path, fault_config, normalize_mag=False)

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Magnetometer Bias Test: Normalized vs Raw (|bias|={bias_mag:.2f})', fontsize=12)

    t_start = 60.0
    fault_onset, fault_end = fault_config['onset_time'], fault_config['end_time']

    scenarios = [
        (axes[0], res_norm, 'With Normalization (default)'),
        (axes[1], res_raw, 'Without Normalization'),
    ]

    for ax_row, res, title in scenarios:
        t = res['times']
        mask = t >= t_start

        # Left: Attitude errors
        ax = ax_row[0]
        ax.plot(t[mask], res['eskf_errors'][mask], 'b-', lw=1.2, label='ESKF', alpha=0.8)
        ax.plot(t[mask], res['smoother_errors'][mask], 'g-', lw=1.2, label='iSAM2', alpha=0.8)

        ax.axvspan(fault_onset, fault_end, color='red', alpha=0.1)
        ax.axvline(fault_onset, color='r', ls='--', alpha=0.5, lw=1)
        ax.axvline(fault_end, color='r', ls='--', alpha=0.5, lw=1)

        ax.set_ylabel('Attitude Error [deg]')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_xlim([t_start, t[-1]])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Right: Disagreement
        ax = ax_row[1]
        ax.plot(t[mask], res['disagreements'][mask], 'purple', lw=1.2, alpha=0.8)
        ax.fill_between(t[mask], res['disagreements'][mask], alpha=0.15, color='purple')

        ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Detection threshold')
        ax.axvspan(fault_onset, fault_end, color='red', alpha=0.1)
        ax.axvline(fault_onset, color='r', ls='--', alpha=0.5, lw=1)
        ax.axvline(fault_end, color='r', ls='--', alpha=0.5, lw=1)

        ax.set_ylabel('Disagreement [deg]')
        ax.set_title('ESKF − iSAM2 Disagreement')
        ax.legend(loc='upper right')
        ax.set_xlim([t_start, t[-1]])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 1].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig('mag_bias_normalization_test.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('mag_bias_normalization_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved to mag_bias_normalization_test.pdf/png")
    plt.close()

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    t = res_norm['times']
    during = (t >= fault_onset) & (t < fault_end)
    post = (t >= fault_end + 10) & (t <= fault_end + 60)

    for name, res in [('Normalized', res_norm), ('Raw (no norm)', res_raw)]:
        max_dis = np.max(res['disagreements'][during | post])
        eskf_during = np.mean(res['eskf_errors'][during])
        smoother_during = np.mean(res['smoother_errors'][during])
        eskf_post = np.mean(res['eskf_errors'][post])
        smoother_post = np.mean(res['smoother_errors'][post])

        print(f"\n{name}:")
        print(f"  Max disagreement: {max_dis:.3f}° {'(DETECTABLE)' if max_dis > 2 else '(not detectable)'}")
        print(f"  During fault:")
        print(f"    ESKF:     {eskf_during:.4f}°")
        print(f"    iSAM2:    {smoother_during:.4f}°")
        print(f"  Post-fault:")
        print(f"    ESKF:     {eskf_post:.4f}°")
        print(f"    iSAM2:    {smoother_post:.4f}°")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
