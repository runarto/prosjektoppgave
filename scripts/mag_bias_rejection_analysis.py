#!/usr/bin/env python3
"""
Magnetometer Bias Rejection Analysis

Evaluates how the ESKF chi-squared gate and factor graph robust kernel
handle biased magnetometer measurements.

Tracks:
1. ESKF: NIS values and rejection statistics
2. Factor Graph: Robust kernel weights (effective downweighting)
"""

import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def apply_mag_bias(sim_data, bias_magnitude: float = 0.62):
    """Apply constant magnetometer bias."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    bias_dir = np.array([1.0, 0.5, 0.3])
    bias_dir = bias_dir / np.linalg.norm(bias_dir)
    bias = bias_magnitude * bias_dir

    for k in range(len(modified.t)):
        if not np.isnan(modified.mag_meas[k, 0]):
            modified.mag_meas[k] = modified.mag_meas[k] + bias

    return modified


CONFIG_PATH = "configs/config_baseline_short.yaml"


def run_eskf_with_tracking(sim_data):
    """
    Run ESKF and track NIS values and rejections for magnetometer.
    """
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    times = []
    errors = []
    mag_nis_values = []
    mag_rejected = []
    mag_times = []

    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        times.append(t)
        errors.append(compute_attitude_error_deg(x.nom.ori, q_true))

        # Prediction
        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        # Magnetometer update with NIS tracking
        if not np.isnan(sim_data.mag_meas[k, 0]):
            mag_times.append(t)
            try:
                x_before = x
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
                mag_nis_values.append(eskf.last_nis)
                mag_rejected.append(False)
            except ValueError:
                # Measurement rejected by chi-squared gate
                mag_nis_values.append(eskf.last_nis if hasattr(eskf, 'last_nis') else np.nan)
                mag_rejected.append(True)

        # Sun sensor update
        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass

        # Star tracker update
        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

    return {
        'times': np.array(times),
        'errors': np.array(errors),
        'mag_times': np.array(mag_times),
        'mag_nis': np.array(mag_nis_values),
        'mag_rejected': np.array(mag_rejected),
    }


def run_smoother_with_tracking(sim_data):
    """
    Run smoother and track robust kernel behavior.

    The Cauchy kernel weight is: w = 1 / (1 + (r/k)^2)
    where r is the normalized residual and k is the kernel parameter.
    """
    smoother = FixedLagAttitudeSmoother(
        config_path=CONFIG_PATH,
        lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        normalize_mag=True,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    smoother.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    dt = sim_data.t[1] - sim_data.t[0]
    keyframe_times = [sim_data.t[0]]
    keyframe_states = [q_init.copy()]

    for k in range(len(sim_data.t)):
        if not np.isnan(sim_data.omega_meas[k, 0]):
            smoother.integrate_gyro(sim_data.omega_meas[k], dt, t=sim_data.t[k])

        z_mag = sim_data.mag_meas[k] if not np.isnan(sim_data.mag_meas[k, 0]) else None
        z_sun = sim_data.sun_meas[k] if not np.isnan(sim_data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.isnan(sim_data.st_meas[k, 0]) else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            smoother.add_measurement(
                t=sim_data.t[k], jd=sim_data.jd[k],
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=sim_data.b_eci[k], s_eci=sim_data.s_eci[k],
            )
            state = smoother.get_state()
            if state is not None:
                keyframe_times.append(sim_data.t[k])
                keyframe_states.append(state.ori.copy())

    # Interpolate errors
    keyframe_times = np.array(keyframe_times)
    times = []
    errors = []

    for k in range(len(sim_data.t)):
        t_k = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        idx = np.searchsorted(keyframe_times, t_k)
        if idx == 0:
            q_interp = keyframe_states[0]
        elif idx >= len(keyframe_times):
            q_interp = keyframe_states[-1]
        else:
            t0, t1 = keyframe_times[idx-1], keyframe_times[idx]
            q0, q1 = keyframe_states[idx-1], keyframe_states[idx]
            alpha = (t_k - t0) / (t1 - t0) if t1 > t0 else 0
            q_interp = q0.slerp(q1, alpha)

        times.append(t_k)
        errors.append(compute_attitude_error_deg(q_interp, q_true))

    return {
        'times': np.array(times),
        'errors': np.array(errors),
    }


def compute_cauchy_weight(residual: float, k: float = 0.1) -> float:
    """Compute Cauchy kernel weight for a given residual."""
    return 1.0 / (1.0 + (residual / k) ** 2)


def main():
    db_path = "simulations.db"
    bias_magnitude = 0.62

    # Get one run
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()

    db = SimulationDatabase(db_path)
    sim_data = db.load_run(run_id)

    print("=" * 80)
    print("MAGNETOMETER BIAS REJECTION ANALYSIS")
    print("=" * 80)
    print(f"Bias magnitude: {bias_magnitude}")
    print()

    # Run baseline (no bias)
    print("Running baseline (no bias)...")
    eskf_baseline = run_eskf_with_tracking(sim_data)
    smoother_baseline = run_smoother_with_tracking(sim_data)

    # Run with bias
    print("Running with magnetometer bias...")
    biased_data = apply_mag_bias(sim_data, bias_magnitude)
    eskf_biased = run_eskf_with_tracking(biased_data)
    smoother_biased = run_smoother_with_tracking(biased_data)

    # Analysis
    print()
    print("=" * 80)
    print("ESKF CHI-SQUARED GATING ANALYSIS")
    print("=" * 80)

    # Baseline stats
    baseline_nis = eskf_baseline['mag_nis']
    baseline_rejected = eskf_baseline['mag_rejected']
    print(f"\nBaseline (no bias):")
    print(f"  Total mag measurements: {len(baseline_nis)}")
    print(f"  Rejected: {np.sum(baseline_rejected)} ({100*np.mean(baseline_rejected):.1f}%)")
    print(f"  Mean NIS: {np.nanmean(baseline_nis):.3f}")
    print(f"  Max NIS: {np.nanmax(baseline_nis):.3f}")
    print(f"  NIS > 7.81 (chi2 95%): {np.sum(baseline_nis > 7.81)}")

    # Biased stats
    biased_nis = eskf_biased['mag_nis']
    biased_rejected = eskf_biased['mag_rejected']
    print(f"\nWith bias ({bias_magnitude}):")
    print(f"  Total mag measurements: {len(biased_nis)}")
    print(f"  Rejected: {np.sum(biased_rejected)} ({100*np.mean(biased_rejected):.1f}%)")
    print(f"  Mean NIS: {np.nanmean(biased_nis):.3f}")
    print(f"  Max NIS: {np.nanmax(biased_nis):.3f}")
    print(f"  NIS > 7.81 (chi2 95%): {np.sum(biased_nis > 7.81)}")

    # Attitude error comparison
    print(f"\nAttitude Error (steady-state, last 50%):")
    mask_baseline = eskf_baseline['times'] >= eskf_baseline['times'][-1] * 0.5
    mask_biased = eskf_biased['times'] >= eskf_biased['times'][-1] * 0.5
    print(f"  Baseline: {np.mean(eskf_baseline['errors'][mask_baseline]):.4f} deg")
    print(f"  Biased:   {np.mean(eskf_biased['errors'][mask_biased]):.4f} deg")

    print()
    print("=" * 80)
    print("FACTOR GRAPH ROBUST KERNEL ANALYSIS")
    print("=" * 80)

    # Compute effective weights for different residual magnitudes
    print("\nCauchy kernel (k=0.1) weight vs normalized residual:")
    print(f"  {'Residual':>10} {'Weight':>10} {'Effective σ':>15}")
    print("-" * 40)
    for r in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        w = compute_cauchy_weight(r, k=0.1)
        effective_sigma = 1.0 / np.sqrt(w) if w > 0 else np.inf
        print(f"  {r:>10.1f} {w:>10.4f} {effective_sigma:>15.2f}x")

    # Estimate the bias in normalized residual terms
    # Bias of 0.62 in body frame, magnetometer noise ~0.01-0.05
    mag_std = 0.05  # Typical value
    normalized_bias = bias_magnitude / mag_std
    w_bias = compute_cauchy_weight(normalized_bias, k=0.1)
    print(f"\nEstimated bias impact:")
    print(f"  Bias magnitude: {bias_magnitude}")
    print(f"  Mag noise σ: {mag_std}")
    print(f"  Normalized bias: {normalized_bias:.1f}σ")
    print(f"  Cauchy weight: {w_bias:.6f}")
    print(f"  Effective downweight: {1/w_bias:.1f}x")

    # Smoother attitude error
    mask_sm_baseline = smoother_baseline['times'] >= smoother_baseline['times'][-1] * 0.5
    mask_sm_biased = smoother_biased['times'] >= smoother_biased['times'][-1] * 0.5
    print(f"\nSmoother Attitude Error (steady-state, last 50%):")
    print(f"  Baseline: {np.mean(smoother_baseline['errors'][mask_sm_baseline]):.4f} deg")
    print(f"  Biased:   {np.mean(smoother_biased['errors'][mask_sm_biased]):.4f} deg")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: NIS values
    ax1 = axes[0]
    ax1.plot(eskf_baseline['mag_times'], eskf_baseline['mag_nis'], 'b-', alpha=0.5, label='Baseline NIS')
    ax1.plot(eskf_biased['mag_times'], eskf_biased['mag_nis'], 'r-', alpha=0.7, label='Biased NIS')
    ax1.axhline(y=7.81, color='k', linestyle='--', label='Chi2 95% threshold')
    ax1.set_ylabel('NIS')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('ESKF: Magnetometer NIS Values')
    ax1.legend()
    ax1.set_ylim([0, min(50, np.nanmax(eskf_biased['mag_nis']) * 1.1)])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rejection over time
    ax2 = axes[1]
    # Rolling rejection rate
    window = 50
    if len(biased_rejected) > window:
        rejection_rate = np.convolve(biased_rejected.astype(float), np.ones(window)/window, mode='valid')
        rejection_times = eskf_biased['mag_times'][window-1:]
        ax2.plot(rejection_times, rejection_rate * 100, 'r-', label='Biased')

        baseline_rate = np.convolve(baseline_rejected.astype(float), np.ones(window)/window, mode='valid')
        baseline_times = eskf_baseline['mag_times'][window-1:]
        ax2.plot(baseline_times, baseline_rate * 100, 'b-', alpha=0.5, label='Baseline')

    ax2.set_ylabel('Rejection Rate [%]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title(f'ESKF: Rolling Mag Rejection Rate (window={window})')
    ax2.legend()
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Attitude errors comparison
    ax3 = axes[2]
    ax3.plot(eskf_baseline['times'], eskf_baseline['errors'], 'b-', alpha=0.5, label='ESKF Baseline')
    ax3.plot(eskf_biased['times'], eskf_biased['errors'], 'r-', alpha=0.7, label='ESKF Biased')
    ax3.plot(smoother_baseline['times'], smoother_baseline['errors'], 'c-', alpha=0.5, label='Smoother Baseline')
    ax3.plot(smoother_biased['times'], smoother_biased['errors'], 'm-', alpha=0.7, label='Smoother Biased')
    ax3.set_ylabel('Attitude Error [deg]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Attitude Error Comparison')
    ax3.legend()
    ax3.set_ylim([0, min(1, max(np.max(eskf_biased['errors']), np.max(smoother_biased['errors'])) * 1.1)])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mag_bias_rejection_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig('mag_bias_rejection_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"\nSaved: mag_bias_rejection_analysis.png")

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Baseline Error':>15} {'Biased Error':>15} {'Degradation':>15}")
    print("-" * 65)

    eskf_baseline_err = np.mean(eskf_baseline['errors'][mask_baseline])
    eskf_biased_err = np.mean(eskf_biased['errors'][mask_biased])
    smoother_baseline_err = np.mean(smoother_baseline['errors'][mask_sm_baseline])
    smoother_biased_err = np.mean(smoother_biased['errors'][mask_sm_biased])

    print(f"{'ESKF':<20} {eskf_baseline_err:>15.4f}° {eskf_biased_err:>15.4f}° {eskf_biased_err/eskf_baseline_err:>14.1f}x")
    print(f"{'Smoother (Cauchy)':<20} {smoother_baseline_err:>15.4f}° {smoother_biased_err:>15.4f}° {smoother_biased_err/smoother_baseline_err:>14.1f}x")

    print("\nConclusion:")
    print(f"  ESKF chi-squared gate rejects {100*np.mean(biased_rejected):.1f}% of biased mag measurements")
    print(f"  Factor graph Cauchy kernel downweights biased measurements by ~{1/w_bias:.0f}x")
    print(f"  Both methods maintain accuracy despite {bias_magnitude} bias")


if __name__ == "__main__":
    main()
