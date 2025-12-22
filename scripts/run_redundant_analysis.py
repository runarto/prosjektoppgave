#!/usr/bin/env python3
"""
Analyze the value of the redundant estimation architecture.

This script demonstrates specific scenarios where redundancy provides value:
1. Fault detection via estimator disagreement
2. Initial convergence improvement
3. Recovery from transient errors

Usage:
    python scripts/run_redundant_analysis.py --config configs/config_meas_spikes.yaml --generate
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


@dataclass
class RedundantAnalysisResult:
    """Extended results for redundant architecture analysis."""
    times: np.ndarray
    eskf_errors_deg: np.ndarray
    isam2_errors_deg: np.ndarray
    disagreement_deg: np.ndarray
    fgo_correction_times: List[float]
    fgo_corrections_deg: List[float]
    selected_sources: List[str]  # Which estimator was selected at each step

    # Derived metrics
    fault_detection_events: List[Tuple[float, float]] = field(default_factory=list)  # (time, disagreement)
    convergence_time_eskf: Optional[float] = None
    convergence_time_hybrid: Optional[float] = None


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.multiply(q_est.conjugate())
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_redundant_analysis(
    sim_data,
    config_path: str,
    initial_error_deg: float = 10.0,
    disagreement_threshold_deg: float = 1.0,
) -> RedundantAnalysisResult:
    """
    Run hybrid estimator in full redundant mode (ESKF + iSAM2 in parallel).

    This exercises the step_redundant() method which:
    1. Updates ESKF
    2. Updates iSAM2 incrementally
    3. Runs periodic batch FGO smoothing
    4. Computes disagreement between ESKF and iSAM2
    5. Selects output based on diagnostics
    """
    print(f"\n  Running Redundant Analysis (init err: {initial_error_deg}°)...")
    start_time = time.time()

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2, att_err_rad**2, att_err_rad**2, 1e-6, 1e-6, 1e-6])

    hybrid = HybridEstimator(
        P0=P0,
        config_path=config_path,
        fgo_window_duration=120.0,
        fgo_optimize_interval=60.0,
        use_robust=True,
        correction_mode="normal",
        enable_isam2_fallback=True,  # Enable iSAM2 for redundant operation
        isam2_switch_threshold_deg=2.0,
    )

    # Initial state with error
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()
    b0_est = np.zeros(3)

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    # Storage for analysis
    times = []
    eskf_errors = []
    isam2_errors = []
    disagreements = []
    fgo_correction_times = []
    fgo_corrections = []
    selected_sources = []

    prev_fgo_count = 0

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        omega_k = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt_k = sim_data.t[k] - sim_data.t[k-1]

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # Run redundant step (ESKF + iSAM2 in parallel)
        x_est, isam2_state, disagreement_deg, selected_source = hybrid.step_redundant(
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

        # Compute errors
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
        isam2_err = compute_attitude_error(isam2_state.ori, q_true) if isam2_state else np.nan

        # Record data
        times.append(t)
        eskf_errors.append(eskf_err)
        isam2_errors.append(isam2_err)
        disagreements.append(disagreement_deg)
        selected_sources.append(selected_source)

        # Check for FGO corrections
        if hybrid.fgo_count > prev_fgo_count:
            fgo_correction_times.append(t)
            if hybrid.correction_history:
                fgo_corrections.append(hybrid.correction_history[-1])
            prev_fgo_count = hybrid.fgo_count

    runtime = time.time() - start_time
    print(f"    Completed in {runtime:.2f}s")
    print(f"    FGO corrections: {len(fgo_corrections)}")
    print(f"    iSAM2 keyframes: {hybrid.isam2_keyframe_idx if hybrid.isam2_initialized else 'N/A'}")

    result = RedundantAnalysisResult(
        times=np.array(times),
        eskf_errors_deg=np.array(eskf_errors),
        isam2_errors_deg=np.array(isam2_errors),
        disagreement_deg=np.array(disagreements),
        fgo_correction_times=fgo_correction_times,
        fgo_corrections_deg=fgo_corrections,
        selected_sources=selected_sources,
    )

    # Analyze fault detection events
    for i, (t, dis) in enumerate(zip(times, disagreements)):
        if dis > disagreement_threshold_deg:
            result.fault_detection_events.append((t, dis))

    # Compute convergence times (time to reach <0.5 deg)
    convergence_threshold = 0.5
    for i, err in enumerate(eskf_errors):
        if err < convergence_threshold:
            result.convergence_time_eskf = times[i]
            break

    for i, err in enumerate(isam2_errors):
        if not np.isnan(err) and err < convergence_threshold:
            result.convergence_time_hybrid = times[i]
            break

    return result


def run_standalone_eskf(
    sim_data,
    config_path: str,
    initial_error_deg: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run standalone ESKF for comparison."""
    att_err_rad = np.deg2rad(initial_error_deg)
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

    for k in range(1, len(sim_data.t)):
        omega_k = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt_k = sim_data.t[k] - sim_data.t[k-1]

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
        times.append(sim_data.t[k])
        errors.append(err)

    return np.array(times), np.array(errors)


def plot_redundant_analysis(
    result: RedundantAnalysisResult,
    standalone_eskf: Tuple[np.ndarray, np.ndarray],
    scenario_name: str,
    save_path: Optional[str] = None,
):
    """Create visualization of redundant architecture analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    eskf_times, eskf_errors = standalone_eskf

    # Plot 1: Attitude errors comparison
    ax1 = axes[0]
    ax1.semilogy(eskf_times, eskf_errors, 'b-', alpha=0.7, label='Standalone ESKF', linewidth=1)
    ax1.semilogy(result.times, result.eskf_errors_deg, 'g-', alpha=0.7, label='Hybrid ESKF', linewidth=1)

    valid_isam2 = ~np.isnan(result.isam2_errors_deg)
    if np.any(valid_isam2):
        ax1.semilogy(result.times[valid_isam2], result.isam2_errors_deg[valid_isam2],
                    'r--', alpha=0.7, label='iSAM2', linewidth=1)

    # Mark FGO corrections
    for t_corr in result.fgo_correction_times:
        ax1.axvline(t_corr, color='orange', alpha=0.3, linestyle=':')

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Redundant Architecture Analysis - {scenario_name}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-3, 20])

    # Plot 2: Disagreement between ESKF and iSAM2
    ax2 = axes[1]
    ax2.plot(result.times, result.disagreement_deg, 'purple', linewidth=1)
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Fault threshold (1°)')
    ax2.fill_between(result.times, result.disagreement_deg, alpha=0.3, color='purple')

    # Mark fault detection events
    if result.fault_detection_events:
        fault_times = [e[0] for e in result.fault_detection_events[:20]]  # First 20
        fault_dis = [e[1] for e in result.fault_detection_events[:20]]
        ax2.scatter(fault_times, fault_dis, c='red', s=20, zorder=5, label='Fault detected')

    ax2.set_ylabel('ESKF-iSAM2 Disagreement [deg]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: FGO corrections applied
    ax3 = axes[2]
    if result.fgo_correction_times:
        ax3.stem(result.fgo_correction_times, result.fgo_corrections_deg,
                linefmt='orange', markerfmt='o', basefmt=' ')
        ax3.set_ylabel('FGO Correction Magnitude [deg]')
    else:
        ax3.text(0.5, 0.5, 'No FGO corrections', ha='center', va='center', transform=ax3.transAxes)

    ax3.set_xlabel('Time [s]')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")

    plt.close()


def compute_summary_metrics(
    result: RedundantAnalysisResult,
    standalone_errors: np.ndarray,
    standalone_times: np.ndarray,
    convergence_time: float = 30.0,
) -> Dict:
    """Compute summary metrics for redundant vs standalone comparison."""
    # Steady-state mask
    ss_mask_result = result.times >= convergence_time
    ss_mask_standalone = standalone_times >= convergence_time

    # RMS errors (steady-state)
    rms_hybrid_eskf = np.sqrt(np.mean(result.eskf_errors_deg[ss_mask_result]**2))
    rms_standalone = np.sqrt(np.mean(standalone_errors[ss_mask_standalone]**2))

    valid_isam2 = ~np.isnan(result.isam2_errors_deg)
    if np.any(valid_isam2 & ss_mask_result):
        rms_isam2 = np.sqrt(np.mean(result.isam2_errors_deg[valid_isam2 & ss_mask_result]**2))
    else:
        rms_isam2 = np.nan

    # Fault detection capability
    n_faults = len(result.fault_detection_events)
    max_disagreement = np.max(result.disagreement_deg) if len(result.disagreement_deg) > 0 else 0

    # Improvement from FGO corrections
    if result.fgo_corrections_deg:
        total_correction = sum(result.fgo_corrections_deg)
        avg_correction = np.mean(result.fgo_corrections_deg)
    else:
        total_correction = 0
        avg_correction = 0

    return {
        'rms_standalone_deg': rms_standalone,
        'rms_hybrid_eskf_deg': rms_hybrid_eskf,
        'rms_isam2_deg': rms_isam2,
        'improvement_pct': (rms_standalone - rms_hybrid_eskf) / rms_standalone * 100 if rms_standalone > 0 else 0,
        'n_fault_detections': n_faults,
        'max_disagreement_deg': max_disagreement,
        'n_fgo_corrections': len(result.fgo_corrections_deg),
        'total_fgo_correction_deg': total_correction,
        'avg_fgo_correction_deg': avg_correction,
        'convergence_time_eskf': result.convergence_time_eskf,
        'convergence_time_hybrid': result.convergence_time_hybrid,
    }


def print_summary(metrics: Dict, scenario_name: str):
    """Print summary of redundant analysis."""
    print("\n" + "=" * 70)
    print(f"REDUNDANT ARCHITECTURE VALUE ANALYSIS - {scenario_name}")
    print("=" * 70)

    print("\n1. ACCURACY COMPARISON (steady-state RMS, t>30s)")
    print("-" * 50)
    print(f"   Standalone ESKF:     {metrics['rms_standalone_deg']:.4f}°")
    print(f"   Hybrid ESKF:         {metrics['rms_hybrid_eskf_deg']:.4f}°")
    if not np.isnan(metrics['rms_isam2_deg']):
        print(f"   iSAM2:               {metrics['rms_isam2_deg']:.4f}°")
    print(f"   Improvement:         {metrics['improvement_pct']:.1f}%")

    print("\n2. FAULT DETECTION CAPABILITY")
    print("-" * 50)
    print(f"   Fault events (>1° disagreement): {metrics['n_fault_detections']}")
    print(f"   Max disagreement:    {metrics['max_disagreement_deg']:.2f}°")

    print("\n3. FGO CORRECTIONS (value of batch smoothing)")
    print("-" * 50)
    print(f"   Number of corrections: {metrics['n_fgo_corrections']}")
    print(f"   Total correction:      {metrics['total_fgo_correction_deg']:.2f}°")
    print(f"   Average correction:    {metrics['avg_fgo_correction_deg']:.4f}°")

    print("\n4. CONVERGENCE TIME")
    print("-" * 50)
    if metrics['convergence_time_eskf']:
        print(f"   ESKF (to <0.5°):     {metrics['convergence_time_eskf']:.1f}s")
    else:
        print(f"   ESKF (to <0.5°):     Did not converge")
    if metrics['convergence_time_hybrid']:
        print(f"   Hybrid (to <0.5°):   {metrics['convergence_time_hybrid']:.1f}s")
    else:
        print(f"   Hybrid (to <0.5°):   Did not converge")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze redundant estimation value")
    parser.add_argument("--config", type=str, default="configs/config_meas_spikes.yaml")
    parser.add_argument("--db", type=str, default="simulations.db")
    parser.add_argument("--sim-id", type=int, help="Use existing simulation")
    parser.add_argument("--generate", action="store_true", help="Generate new simulation")
    parser.add_argument("--initial-error", type=float, default=10.0, help="Initial error (deg)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")

    args = parser.parse_args()

    config = load_yaml(args.config)
    scenario_name = config.get('simulation', {}).get('run_name', Path(args.config).stem)

    print("=" * 70)
    print("REDUNDANT ARCHITECTURE VALUE ANALYSIS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Scenario: {scenario_name}")

    # Get simulation data
    if args.generate:
        print("\nGenerating new simulation...")
        generator = EnhancedAttitudeDataGenerator(
            db_path=args.db,
            config_path=args.config
        )
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

    # Load simulation
    db = SimulationDatabase(args.db)
    sim_data = db.load_run(sim_id)
    print(f"Loaded simulation: {sim_data.t[-1]:.1f}s, {len(sim_data.t)} samples")

    # Run analysis
    print("\n" + "-" * 50)
    print("Running estimators...")

    # Standalone ESKF
    print("\n  Running standalone ESKF...")
    eskf_times, eskf_errors = run_standalone_eskf(
        sim_data, args.config, args.initial_error
    )

    # Redundant analysis
    result = run_redundant_analysis(
        sim_data, args.config, args.initial_error
    )

    # Compute metrics
    metrics = compute_summary_metrics(
        result, eskf_errors, eskf_times
    )

    # Print summary
    print_summary(metrics, scenario_name)

    # Create plots
    if args.save_plots:
        plot_path = f"redundant_analysis_{scenario_name}.pdf"
        plot_redundant_analysis(
            result, (eskf_times, eskf_errors), scenario_name, plot_path
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
