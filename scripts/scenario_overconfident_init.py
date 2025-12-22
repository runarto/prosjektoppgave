#!/usr/bin/env python3
"""
Scenario: Overconfident Initialization (45° error, tiny P0).

Demonstrates how the redundant estimator handles poor initial conditions
where the ESKF is initialized with a large attitude error but small covariance.
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

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['lines.linewidth'] = 1.2


def compute_attitude_error(q_est, q_true):
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    return np.rad2deg(2 * np.arccos(np.clip(abs(q_err.mu), 0, 1)))


def run_scenario(sim_data, config_path, init_error_deg, P0_att_deg):
    """Run all three estimators with specified initial conditions."""

    # Convert to radians
    init_error_rad = np.deg2rad(init_error_deg)
    P0_att_rad = np.deg2rad(P0_att_deg)

    # Initial covariance (overconfident: small P0 despite large error)
    P0 = np.diag([P0_att_rad**2] * 3 + [1e-8] * 3)

    # Initialize estimators
    eskf = ESKF(P0=P0, config_path=config_path)
    smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=30.0, use_robust=True)
    redundant = RedundantEstimator(
        P0=P0, config_path=config_path, smoother_lag=30.0, use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=3,
        consecutive_agreements_to_recover=5,
        agreement_threshold_deg=0.5,
    )

    # True initial attitude
    q0_true = Quaternion.from_array(sim_data.q_true[0])

    # Create large initial error (rotation about arbitrary axis)
    error_axis = np.array([1.0, 0.5, 0.3])
    error_axis = error_axis / np.linalg.norm(error_axis)
    q0_est = q0_true @ Quaternion.from_avec(init_error_rad * error_axis)
    q0_est = q0_est.normalize()

    # Initialize states
    x_eskf = EskfState(
        nom=NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3)),
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
        'switch_times': [],
        'switch_types': [],
    }

    prev_primary = 'ESKF'

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        # === Standalone ESKF ===
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)

        # === Standalone Smoother ===
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
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

        # Redundant estimator
        x_redundant, redundant_smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        # Track switches
        if primary != prev_primary:
            results['switch_times'].append(t)
            results['switch_types'].append(f"{prev_primary}→{primary}")
            prev_primary = primary

        # Get redundant output
        redundant_state = redundant_smoother_state if primary == 'SMOOTHER' else x_redundant.nom

        # Compute errors
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_eskf.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true) if smoother_state else eskf_err
        redundant_err = compute_attitude_error(redundant_state.ori, q_true)

        results['times'].append(t)
        results['eskf_errors'].append(eskf_err)
        results['smoother_errors'].append(smoother_err)
        results['redundant_errors'].append(redundant_err)
        results['disagreements'].append(disagreement)
        results['primaries'].append(primary)

    for k in ['times', 'eskf_errors', 'smoother_errors', 'redundant_errors', 'disagreements']:
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

    # Scenario parameters
    init_error_deg = 45.0  # Large initial error
    P0_att_deg = 1.0       # Small (overconfident) initial covariance

    print(f"\nScenario: Overconfident Initialization")
    print(f"  Initial attitude error: {init_error_deg}°")
    print(f"  Initial covariance (1σ): {P0_att_deg}°")
    print(f"  Ratio: {init_error_deg/P0_att_deg:.0f}σ error\n")

    print("Running estimators...")
    results = run_scenario(sim_data, config_path, init_error_deg, P0_att_deg)

    # Limit to first 60 seconds for clearer visualization
    t_max = 60.0
    mask = results['times'] <= t_max

    t = results['times'][mask]
    eskf_err = results['eskf_errors'][mask]
    smoother_err = results['smoother_errors'][mask]
    redundant_err = results['redundant_errors'][mask]
    primaries = [results['primaries'][i] for i in range(len(results['primaries'])) if results['times'][i] <= t_max]

    # Find event window (when redundant uses smoother)
    smoother_active = np.array([p == 'SMOOTHER' for p in primaries])

    # Compute steady-state errors (last 20 seconds of window)
    ss_mask = t > 40
    eskf_ss = np.mean(eskf_err[ss_mask])
    smoother_ss = np.mean(smoother_err[ss_mask])
    redundant_ss = np.mean(redundant_err[ss_mask])

    # Create clean 2-panel figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])

    # === Top plot: Attitude error (log scale) ===
    ax1 = axes[0]

    ax1.semilogy(t, smoother_err, 'g-', lw=1.2, label='iSAM2', alpha=0.9)
    ax1.semilogy(t, eskf_err, 'b-', lw=1.2, label='ESKF', alpha=0.9)
    ax1.semilogy(t, redundant_err, 'r--', lw=2.0, label='Redundant', alpha=0.9)

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_ylim([1e-2, 100])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([0, t_max])
    ax1.set_title(f'Overconfident Initialization ({init_error_deg}° error, {P0_att_deg}° covariance)',
                  fontsize=11, fontweight='bold')

    # === Bottom plot: Mode indicator ===
    ax2 = axes[1]

    # Create mode regions
    mode_colors = {'ESKF': 'C0', 'CONSERVATIVE': 'C2', 'SMOOTHER': 'C1'}
    current_mode = primaries[0]
    region_start = t[0]

    for i in range(1, len(primaries)):
        if primaries[i] != current_mode or i == len(primaries) - 1:
            region_end = t[i] if primaries[i] != current_mode else t[-1]
            ax2.axvspan(region_start, region_end, color=mode_colors[current_mode], alpha=0.4)
            region_start = t[i]
            current_mode = primaries[i]

    # Add legend entries
    ax2.fill_between([], [], color='C0', alpha=0.4, label='ESKF')
    ax2.fill_between([], [], color='C2', alpha=0.4, label='CONSERVATIVE')
    ax2.fill_between([], [], color='C1', alpha=0.4, label='SMOOTHER')

    ax2.set_ylabel('Mode')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylim([0, 1])
    ax2.set_yticks([])
    ax2.legend(loc='upper right', ncol=3, fontsize=8)
    ax2.set_xlim([0, t_max])

    plt.tight_layout()
    plt.savefig('scenario_overconfident_init.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('scenario_overconfident_init.png', dpi=150, bbox_inches='tight')
    print("\nSaved to scenario_overconfident_init.pdf/png")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nInitial error: {init_error_deg}° ({init_error_deg/P0_att_deg:.0f}σ)")
    print(f"\nSteady-state errors (t > 40s):")
    print(f"  ESKF:      {eskf_ss:.4f}°")
    print(f"  iSAM2:     {smoother_ss:.4f}°")
    print(f"  Redundant: {redundant_ss:.4f}°")
    print(f"\nSwitches: {len(results['switch_times'])}")
    for st, stype in zip(results['switch_times'], results['switch_types']):
        print(f"  t={st:.2f}s: {stype}")
    print("="*60)


if __name__ == "__main__":
    main()
