#!/usr/bin/env python3
"""
Analyze the true value propositions of redundant estimation.

This script focuses on what redundancy ACTUALLY provides:
1. Fault detection via estimator disagreement
2. Graceful degradation (backup when primary fails)
3. Estimator health monitoring

NOT: accuracy improvement (which batch FGO already provides better)

Usage:
    python scripts/analyze_redundancy_value.py --config configs/config_eclipse.yaml --sim-id 12
"""

import argparse
import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO
from estimation.hybrid_estimator import HybridEstimator
from environment.environment import OrbitEnvironmentModel
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.multiply(q_est.conjugate())
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def simulate_eskf_failure(
    sim_data,
    config_path: str,
    failure_start: float = 100.0,
    failure_duration: float = 30.0,
    failure_magnitude_deg: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate ESKF with an injected failure (measurement fault).

    Returns times, eskf_errors, failure_mask
    """
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2, att_err_rad**2, att_err_rad**2, 1e-6, 1e-6, 1e-6])
    eskf = ESKF(P0=P0, config_path=config_path)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    times = []
    errors = []
    failure_mask = []

    failure_end = failure_start + failure_duration

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        omega_k = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt_k = sim_data.t[k] - sim_data.t[k-1]

        x_est = eskf.predict(x_est, omega_k, dt_k)

        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # Check if we're in failure window
        is_failure = failure_start <= t <= failure_end

        # During failure, inject biased measurements
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            mag_meas = sim_data.mag_meas[k].copy()
            if is_failure:
                # Inject systematic bias during failure
                bias_vec = np.array([failure_magnitude_deg / 180 * np.pi, 0, 0])
                mag_meas = mag_meas + bias_vec
                mag_meas = mag_meas / np.linalg.norm(mag_meas)
            try:
                x_est = eskf.update(x_est, mag_meas, SensorType.MAGNETOMETER, B_n=B_n)
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
        failure_mask.append(is_failure)

    return np.array(times), np.array(errors), np.array(failure_mask)


def run_redundant_with_selection(
    sim_data,
    config_path: str,
    failure_start: float = 100.0,
    failure_duration: float = 30.0,
    failure_magnitude_deg: float = 5.0,
    disagreement_threshold: float = 1.0,
) -> Dict:
    """
    Run redundant estimator with smart output selection.

    When estimators disagree significantly, this simulates selecting the "better" one.
    In practice, this would be based on additional diagnostics.
    """
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2, att_err_rad**2, att_err_rad**2, 1e-6, 1e-6, 1e-6])

    hybrid = HybridEstimator(
        P0=P0,
        config_path=config_path,
        fgo_window_duration=120.0,
        fgo_optimize_interval=60.0,
        use_robust=True,
        enable_isam2_fallback=True,
    )

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    failure_end = failure_start + failure_duration

    times = []
    eskf_errors = []
    isam2_errors = []
    selected_errors = []  # Error using smart selection
    disagreements = []
    selections = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        omega_k = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt_k = sim_data.t[k] - sim_data.t[k-1]

        is_failure = failure_start <= t <= failure_end

        # Get measurements - inject failure into magnetometer
        z_mag = None
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            z_mag = sim_data.mag_meas[k].copy()
            if is_failure:
                bias_vec = np.array([failure_magnitude_deg / 180 * np.pi, 0, 0])
                z_mag = z_mag + bias_vec
                z_mag = z_mag / np.linalg.norm(z_mag)

        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # Run redundant step
        x_est, isam2_state, disagreement_deg, _ = hybrid.step_redundant(
            x_eskf=x_est,
            t=t,
            jd=jd,
            omega_meas=omega_k,
            dt=dt_k,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=B_n,
            s_n=s_n,
        )

        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
        isam2_err = compute_attitude_error(isam2_state.ori, q_true) if isam2_state else np.nan

        # Smart selection: if large disagreement, trust iSAM2 (it's more robust)
        if disagreement_deg > disagreement_threshold and isam2_state is not None:
            selected_err = isam2_err
            selection = "iSAM2"
        else:
            selected_err = eskf_err
            selection = "ESKF"

        times.append(t)
        eskf_errors.append(eskf_err)
        isam2_errors.append(isam2_err)
        selected_errors.append(selected_err)
        disagreements.append(disagreement_deg)
        selections.append(selection)

    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'isam2_errors': np.array(isam2_errors),
        'selected_errors': np.array(selected_errors),
        'disagreements': np.array(disagreements),
        'selections': selections,
        'failure_start': failure_start,
        'failure_end': failure_end,
    }


def analyze_fault_detection_value(result: Dict, standalone_errors: np.ndarray) -> Dict:
    """
    Analyze the value of fault detection capability.

    Metrics:
    - Detection rate: % of failure period where disagreement exceeded threshold
    - False positive rate: % of normal operation with disagreement above threshold
    - Recovery improvement: How much better is smart selection during failure
    """
    times = result['times']
    disagreements = result['disagreements']
    failure_start = result['failure_start']
    failure_end = result['failure_end']

    threshold = 1.0  # degrees

    # Failure period mask
    failure_mask = (times >= failure_start) & (times <= failure_end)
    normal_mask = ~failure_mask

    # Detection rate during failure
    failure_detections = disagreements[failure_mask] > threshold
    detection_rate = np.mean(failure_detections) * 100 if np.any(failure_mask) else 0

    # False positive rate during normal operation
    normal_detections = disagreements[normal_mask] > threshold
    false_positive_rate = np.mean(normal_detections) * 100 if np.any(normal_mask) else 0

    # Error comparison during failure
    if np.any(failure_mask):
        eskf_error_during_failure = np.mean(result['eskf_errors'][failure_mask])
        selected_error_during_failure = np.mean(result['selected_errors'][failure_mask])
        standalone_error_during_failure = np.mean(standalone_errors[failure_mask[1:]])
    else:
        eskf_error_during_failure = 0
        selected_error_during_failure = 0
        standalone_error_during_failure = 0

    return {
        'detection_rate_pct': detection_rate,
        'false_positive_rate_pct': false_positive_rate,
        'eskf_error_during_failure_deg': eskf_error_during_failure,
        'selected_error_during_failure_deg': selected_error_during_failure,
        'standalone_error_during_failure_deg': standalone_error_during_failure,
        'improvement_from_selection_pct':
            (eskf_error_during_failure - selected_error_during_failure) / eskf_error_during_failure * 100
            if eskf_error_during_failure > 0 else 0,
    }


def plot_fault_detection_analysis(
    result: Dict,
    standalone_times: np.ndarray,
    standalone_errors: np.ndarray,
    metrics: Dict,
    save_path: Optional[str] = None,
):
    """Create visualization of fault detection value."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    times = result['times']
    failure_start = result['failure_start']
    failure_end = result['failure_end']

    # Highlight failure region
    for ax in axes:
        ax.axvspan(failure_start, failure_end, alpha=0.2, color='red', label='Failure injected')

    # Plot 1: Attitude errors
    ax1 = axes[0]
    ax1.semilogy(standalone_times, standalone_errors, 'b-', alpha=0.5, label='Standalone ESKF', linewidth=1)
    ax1.semilogy(times, result['eskf_errors'], 'g-', alpha=0.7, label='Redundant ESKF', linewidth=1)
    ax1.semilogy(times, result['selected_errors'], 'k--', alpha=0.9, label='Smart Selection', linewidth=2)

    valid_isam2 = ~np.isnan(result['isam2_errors'])
    ax1.semilogy(times[valid_isam2], result['isam2_errors'][valid_isam2], 'r:', alpha=0.5, label='iSAM2', linewidth=1)

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.legend(loc='upper right')
    ax1.set_title('Value of Redundant Architecture: Fault Detection & Recovery')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-3, 20])

    # Plot 2: Disagreement (fault detection signal)
    ax2 = axes[1]
    ax2.plot(times, result['disagreements'], 'purple', linewidth=1)
    ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='Detection threshold')
    ax2.fill_between(times, result['disagreements'], alpha=0.3, color='purple')

    ax2.set_ylabel('ESKF-iSAM2 Disagreement [deg]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Fault Detection Signal (Detection Rate: {metrics["detection_rate_pct"]:.0f}%)')

    # Plot 3: Selection decisions
    ax3 = axes[2]
    selection_numeric = [1 if s == "iSAM2" else 0 for s in result['selections']]
    ax3.fill_between(times, selection_numeric, alpha=0.5, step='pre', color='red', label='iSAM2 selected')
    ax3.fill_between(times, selection_numeric, 1, alpha=0.5, step='pre', color='green', label='ESKF selected')
    ax3.set_ylabel('Selected Estimator')
    ax3.set_xlabel('Time [s]')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['ESKF', 'iSAM2'])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")

    plt.close()


def print_fault_detection_summary(metrics: Dict, scenario: str):
    """Print summary of fault detection analysis."""
    print("\n" + "=" * 70)
    print(f"REDUNDANT ARCHITECTURE VALUE: FAULT DETECTION")
    print(f"Scenario: {scenario}")
    print("=" * 70)

    print("\n1. FAULT DETECTION PERFORMANCE")
    print("-" * 50)
    print(f"   Detection rate during failure:    {metrics['detection_rate_pct']:.1f}%")
    print(f"   False positive rate (normal):     {metrics['false_positive_rate_pct']:.1f}%")

    print("\n2. ERROR DURING FAILURE PERIOD")
    print("-" * 50)
    print(f"   Standalone ESKF:                  {metrics['standalone_error_during_failure_deg']:.3f}°")
    print(f"   Redundant ESKF (corrupted):       {metrics['eskf_error_during_failure_deg']:.3f}°")
    print(f"   Smart Selection (ESKF+iSAM2):     {metrics['selected_error_during_failure_deg']:.3f}°")
    print(f"   Improvement from selection:       {metrics['improvement_from_selection_pct']:.1f}%")

    print("\n3. VALUE PROPOSITION SUMMARY")
    print("-" * 50)
    if metrics['detection_rate_pct'] > 50:
        print("   ✓ Fault detection capability demonstrated")
    else:
        print("   ✗ Fault detection needs improvement")

    if metrics['improvement_from_selection_pct'] > 10:
        print("   ✓ Smart selection improves accuracy during faults")
    else:
        print("   ~ Marginal improvement from selection")

    if metrics['false_positive_rate_pct'] < 10:
        print("   ✓ Low false positive rate")
    else:
        print("   ✗ High false positive rate - threshold tuning needed")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze redundancy value for fault detection")
    parser.add_argument("--config", type=str, default="configs/config_baseline_short.yaml")
    parser.add_argument("--db", type=str, default="simulations.db")
    parser.add_argument("--sim-id", type=int, help="Use existing simulation")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--failure-start", type=float, default=100.0)
    parser.add_argument("--failure-duration", type=float, default=30.0)
    parser.add_argument("--failure-magnitude", type=float, default=10.0, help="Failure magnitude in degrees")
    parser.add_argument("--save-plots", action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config)
    scenario = config.get('simulation', {}).get('run_name', Path(args.config).stem)

    print("=" * 70)
    print("REDUNDANT ARCHITECTURE: FAULT DETECTION VALUE ANALYSIS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Injected failure: t={args.failure_start}s to {args.failure_start + args.failure_duration}s")
    print(f"Failure magnitude: {args.failure_magnitude}°")

    # Get simulation
    if args.generate:
        generator = EnhancedAttitudeDataGenerator(db_path=args.db, config_path=args.config)
        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd']
        )
        sim_id = generator.run(sim_cfg)
        print(f"Generated simulation ID: {sim_id}")
    elif args.sim_id:
        sim_id = args.sim_id
    else:
        print("Error: Must specify --generate or --sim-id")
        return 1

    db = SimulationDatabase(args.db)
    sim_data = db.load_run(sim_id)
    print(f"Loaded simulation: {sim_data.t[-1]:.1f}s, {len(sim_data.t)} samples")

    print("\n" + "-" * 50)
    print("Running fault injection analysis...")

    # Run standalone ESKF with failure (baseline - no fault detection)
    print("\n  Running standalone ESKF with injected failure...")
    standalone_times, standalone_errors, _ = simulate_eskf_failure(
        sim_data, args.config,
        args.failure_start, args.failure_duration, args.failure_magnitude
    )

    # Run redundant estimator with smart selection
    print("  Running redundant estimator with smart selection...")
    result = run_redundant_with_selection(
        sim_data, args.config,
        args.failure_start, args.failure_duration, args.failure_magnitude
    )

    # Analyze fault detection value
    metrics = analyze_fault_detection_value(result, standalone_errors)

    # Print summary
    print_fault_detection_summary(metrics, scenario)

    # Create plot
    if args.save_plots:
        plot_path = f"fault_detection_analysis_{scenario}.pdf"
        plot_fault_detection_analysis(
            result, standalone_times, standalone_errors, metrics, plot_path
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
