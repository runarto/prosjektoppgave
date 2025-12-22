#!/usr/bin/env python3
"""
Compare PRACTICAL real-time estimators only.

This script excludes full-batch FGO (300s) which is computationally infeasible
for real-time spacecraft operations.

PRACTICAL ESTIMATORS:
1. ESKF - Real-time recursive Kalman filter
2. Hybrid - ESKF + sliding window batch FGO (120s window)
3. iSAM2 - Incremental graph optimizer

EXCLUDED (impractical for real-time):
- Full Batch FGO (300s) - Post-processing only, used as oracle reference

Usage:
    python scripts/compare_practical_estimators.py --config configs/config_meas_spikes.yaml --sim-id 11
"""

import argparse
import copy
import sys
import time
from dataclasses import dataclass
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
class PracticalResult:
    """Results from a practical real-time estimator."""
    name: str
    times: np.ndarray
    errors_deg: np.ndarray
    runtime_s: float
    is_realtime: bool = True
    description: str = ""


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.multiply(q_est.conjugate())
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_eskf(sim_data, config_path: str, initial_error_deg: float = 10.0) -> PracticalResult:
    """Run ESKF (real-time recursive filter)."""
    print("  Running ESKF...")
    start = time.time()

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=config_path)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()

    x = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0)
    )

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        omega = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt = sim_data.t[k] - sim_data.t[k-1]
        x = eskf.predict(x, omega, dt)

        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except ValueError:
                pass
        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass
        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                x = eskf.update(x, Quaternion.from_array(sim_data.st_meas[k]), SensorType.STAR_TRACKER)
            except ValueError:
                pass

        err = compute_attitude_error(x.nom.ori, Quaternion.from_array(sim_data.q_true[k]))
        times.append(sim_data.t[k])
        errors.append(err)

    runtime = time.time() - start
    print(f"    Completed in {runtime:.2f}s")

    return PracticalResult(
        name="ESKF",
        times=np.array(times),
        errors_deg=np.array(errors),
        runtime_s=runtime,
        is_realtime=True,
        description="Real-time recursive Kalman filter"
    )


def run_hybrid(sim_data, config_path: str, initial_error_deg: float = 10.0) -> PracticalResult:
    """Run Hybrid ESKF + sliding window batch FGO."""
    print("  Running Hybrid (ESKF + 120s sliding window FGO)...")
    start = time.time()

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2]*3 + [1e-6]*3)

    hybrid = HybridEstimator(
        P0=P0,
        config_path=config_path,
        fgo_window_duration=120.0,
        fgo_optimize_interval=60.0,
        use_robust=True,
        enable_isam2_fallback=False,
    )

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()

    x = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0)
    )

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        omega = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt = sim_data.t[k] - sim_data.t[k-1]

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        x, _ = hybrid.step(
            x_eskf=x, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st,
            B_n=sim_data.b_eci[k], s_n=sim_data.s_eci[k]
        )

        err = compute_attitude_error(x.nom.ori, Quaternion.from_array(sim_data.q_true[k]))
        times.append(t)
        errors.append(err)

    runtime = time.time() - start
    n_corrections = hybrid.fgo_count
    print(f"    Completed in {runtime:.2f}s ({n_corrections} FGO corrections)")

    return PracticalResult(
        name="Hybrid (ESKF+FGO)",
        times=np.array(times),
        errors_deg=np.array(errors),
        runtime_s=runtime,
        is_realtime=True,
        description=f"ESKF with 120s sliding window batch FGO ({n_corrections} corrections)"
    )


def run_isam2(sim_data, config_path: str, use_robust: bool = True) -> PracticalResult:
    """Run iSAM2 incremental optimizer."""
    mode = "+Huber" if use_robust else ""
    print(f"  Running iSAM2{mode}...")
    start = time.time()

    env = OrbitEnvironmentModel()
    fgo = KeyframeFGO(
        config_path=config_path,
        use_robust=use_robust,
        use_isam2=True,  # Incremental!
        use_rk4=True,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)

    # Interpolate to full rate
    times, errors = [], []
    kf_idx = 0

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        while kf_idx < len(kf_times) - 1 and t >= kf_times[kf_idx + 1]:
            kf_idx += 1
        if kf_idx >= len(kf_states):
            kf_idx = len(kf_states) - 1

        kf_state = kf_states[kf_idx]
        kf_time = kf_times[kf_idx]

        if abs(t - kf_time) < 1e-6:
            q_est = kf_state.ori
        else:
            # Propagate from keyframe
            kf_sim_idx = np.argmin(np.abs(sim_data.t - kf_time))
            q_prop = kf_state.ori.copy()
            b_est = kf_state.gyro_bias.copy()

            for i in range(kf_sim_idx, k):
                if i + 1 < len(sim_data.t):
                    dt = sim_data.t[i + 1] - sim_data.t[i]
                    omega = sim_data.omega_meas[i + 1]
                    if not np.any(np.isnan(omega)):
                        q_prop = q_prop.propagate(omega - b_est, dt)
            q_est = q_prop

        err = compute_attitude_error(q_est, Quaternion.from_array(sim_data.q_true[k]))
        times.append(t)
        errors.append(err)

    runtime = time.time() - start
    print(f"    Completed in {runtime:.2f}s ({len(kf_states)} keyframes)")

    return PracticalResult(
        name=f"iSAM2{mode}",
        times=np.array(times),
        errors_deg=np.array(errors),
        runtime_s=runtime,
        is_realtime=True,
        description="Incremental graph optimizer"
    )


def run_batch_oracle(sim_data, config_path: str, use_robust: bool = True) -> PracticalResult:
    """Run full batch FGO (ORACLE - not practical for real-time)."""
    mode = "+Huber" if use_robust else ""
    print(f"  Running Batch FGO{mode} (ORACLE - post-processing only)...")
    start = time.time()

    env = OrbitEnvironmentModel()
    fgo = KeyframeFGO(
        config_path=config_path,
        use_robust=use_robust,
        use_isam2=False,
        use_rk4=True,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)

    # Interpolate to full rate
    times, errors = [], []
    kf_idx = 0

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        while kf_idx < len(kf_times) - 1 and t >= kf_times[kf_idx + 1]:
            kf_idx += 1
        if kf_idx >= len(kf_states):
            kf_idx = len(kf_states) - 1

        kf_state = kf_states[kf_idx]
        kf_time = kf_times[kf_idx]

        if abs(t - kf_time) < 1e-6:
            q_est = kf_state.ori
        else:
            kf_sim_idx = np.argmin(np.abs(sim_data.t - kf_time))
            q_prop = kf_state.ori.copy()
            b_est = kf_state.gyro_bias.copy()

            for i in range(kf_sim_idx, k):
                if i + 1 < len(sim_data.t):
                    dt = sim_data.t[i + 1] - sim_data.t[i]
                    omega = sim_data.omega_meas[i + 1]
                    if not np.any(np.isnan(omega)):
                        q_prop = q_prop.propagate(omega - b_est, dt)
            q_est = q_prop

        err = compute_attitude_error(q_est, Quaternion.from_array(sim_data.q_true[k]))
        times.append(t)
        errors.append(err)

    runtime = time.time() - start
    print(f"    Completed in {runtime:.2f}s ({len(kf_states)} keyframes)")

    return PracticalResult(
        name=f"Batch FGO{mode} (Oracle)",
        times=np.array(times),
        errors_deg=np.array(errors),
        runtime_s=runtime,
        is_realtime=False,  # NOT practical!
        description="Full batch optimization - POST-PROCESSING ONLY"
    )


def compute_metrics(result: PracticalResult, convergence_time: float = 30.0) -> Dict:
    """Compute RMS metrics."""
    ss_mask = result.times >= convergence_time
    return {
        'rms_all': np.sqrt(np.mean(result.errors_deg**2)),
        'rms_steady_state': np.sqrt(np.mean(result.errors_deg[ss_mask]**2)) if np.any(ss_mask) else np.nan,
        'final_error': result.errors_deg[-1] if len(result.errors_deg) > 0 else np.nan,
        'max_error': np.max(result.errors_deg),
    }


def plot_comparison(results: List[PracticalResult], scenario: str, save_path: Optional[str] = None):
    """Create comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'ESKF': 'blue', 'Hybrid (ESKF+FGO)': 'green', 'iSAM2': 'orange', 'iSAM2+Huber': 'red'}

    for r in results:
        style = '--' if not r.is_realtime else '-'
        alpha = 0.5 if not r.is_realtime else 0.8
        color = colors.get(r.name, 'gray')
        label = f"{r.name}" + (" [Oracle]" if not r.is_realtime else "")
        ax.semilogy(r.times, r.errors_deg, style, color=color, alpha=alpha, label=label, linewidth=1.5)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(f'Practical Real-Time Estimator Comparison - {scenario}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-3, 20])

    # Add annotation
    ax.text(0.02, 0.02, "Solid lines: Real-time practical\nDashed: Post-processing oracle",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def print_table(results: List[PracticalResult], scenario: str):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print(f"PRACTICAL ESTIMATOR COMPARISON - {scenario}")
    print("=" * 90)

    print("\nNOTE: Full Batch FGO is included only as an ORACLE reference.")
    print("      It requires post-processing and is NOT suitable for real-time operation.")
    print()

    print(f"{'Estimator':<25} {'Type':<12} {'RMS (t>30s)':<12} {'Final Err':<12} {'Runtime':<10}")
    print("-" * 90)

    for r in results:
        metrics = compute_metrics(r)
        type_str = "Real-time" if r.is_realtime else "Oracle"
        rms = f"{metrics['rms_steady_state']:.4f}°"
        final = f"{metrics['final_error']:.4f}°"
        runtime = f"{r.runtime_s:.2f}s"
        print(f"{r.name:<25} {type_str:<12} {rms:<12} {final:<12} {runtime:<10}")

    print("-" * 90)

    # Compute relative performance
    practical_results = [r for r in results if r.is_realtime]
    if len(practical_results) >= 2:
        eskf = next((r for r in practical_results if "ESKF" in r.name and "Hybrid" not in r.name), None)
        hybrid = next((r for r in practical_results if "Hybrid" in r.name), None)
        isam2 = next((r for r in practical_results if "iSAM2" in r.name), None)

        if eskf and hybrid:
            eskf_rms = compute_metrics(eskf)['rms_steady_state']
            hybrid_rms = compute_metrics(hybrid)['rms_steady_state']
            diff = (hybrid_rms - eskf_rms) / eskf_rms * 100
            print(f"\nHybrid vs ESKF: {diff:+.1f}% {'worse' if diff > 0 else 'better'}")

        if eskf and isam2:
            isam2_rms = compute_metrics(isam2)['rms_steady_state']
            diff = (isam2_rms - eskf_rms) / eskf_rms * 100
            print(f"iSAM2 vs ESKF: {diff:+.1f}% {'worse' if diff > 0 else 'better'}")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare practical real-time estimators")
    parser.add_argument("--config", type=str, default="configs/config_meas_spikes.yaml")
    parser.add_argument("--db", type=str, default="simulations.db")
    parser.add_argument("--sim-id", type=int, help="Use existing simulation")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--include-oracle", action="store_true", help="Include batch FGO oracle")
    parser.add_argument("--save-plots", action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config)
    scenario = config.get('simulation', {}).get('run_name', Path(args.config).stem)

    print("=" * 70)
    print("PRACTICAL REAL-TIME ESTIMATOR COMPARISON")
    print("=" * 70)
    print(f"Config: {args.config}")

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
    print("Running estimators...")

    results = []

    # Practical real-time estimators
    results.append(run_eskf(sim_data, args.config))
    results.append(run_hybrid(sim_data, args.config))
    results.append(run_isam2(sim_data, args.config, use_robust=True))

    # Oracle (optional)
    if args.include_oracle:
        results.append(run_batch_oracle(sim_data, args.config, use_robust=True))

    # Print results
    print_table(results, scenario)

    # Plot
    if args.save_plots:
        plot_path = f"practical_comparison_{scenario}.pdf"
        plot_comparison(results, scenario, plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
