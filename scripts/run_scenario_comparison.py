#!/usr/bin/env python3
"""
Run estimator comparison on simulation scenarios.

Usage:
    # Generate new simulation and run comparison
    python scripts/run_scenario_comparison.py --generate --config configs/config_baseline_short.yaml

    # Run on existing simulation
    python scripts/run_scenario_comparison.py --sim-id 1

    # Run specific estimators only
    python scripts/run_scenario_comparison.py --sim-id 1 --estimators eskf,fgo_batch,fgo_isam2

    # Compare multiple simulations
    python scripts/run_scenario_comparison.py --sim-ids 1,2,3

Estimators:
    eskf          - Error-State Kalman Filter
    fgo_batch     - Factor Graph (Batch LM, no M-estimator)
    fgo_batch_m   - Factor Graph (Batch LM, with M-estimator)
    fgo_isam2     - Factor Graph (iSAM2, no M-estimator)
    fgo_isam2_m   - Factor Graph (iSAM2, with M-estimator)
    hybrid        - Hybrid ESKF + FGO (ESKF frontend, periodic FGO backend)
"""

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

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
class EstimationResult:
    """Results from a single estimator run."""
    name: str
    times: np.ndarray
    states: List[NominalState]
    runtime_s: float

    # Computed metrics
    attitude_errors_deg: Optional[np.ndarray] = None
    bias_errors_deg_h: Optional[np.ndarray] = None
    final_attitude_error_deg: Optional[float] = None
    rms_attitude_error_deg: Optional[float] = None
    rms_steady_state_deg: Optional[float] = None  # RMS after convergence


def interpolate_to_full_rate(
    keyframe_times: np.ndarray,
    keyframe_states: List[NominalState],
    sim_data,
) -> Tuple[np.ndarray, List[NominalState]]:
    """
    Interpolate keyframe estimates to full simulation rate using gyro propagation.

    Between keyframes, propagate the attitude using gyro measurements (corrected for
    the estimated bias). This gives a fair comparison with ESKF which is evaluated
    at every timestep.

    Args:
        keyframe_times: Times of keyframe estimates
        keyframe_states: States at keyframes
        sim_data: Full simulation data with gyro measurements

    Returns:
        (full_times, full_states): Interpolated results at simulation rate
    """
    full_times = []
    full_states = []

    kf_idx = 0  # Current keyframe index

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]

        # Find the keyframe interval we're in
        # Move to next keyframe if we've passed current one
        while kf_idx < len(keyframe_times) - 1 and t >= keyframe_times[kf_idx + 1]:
            kf_idx += 1

        if kf_idx >= len(keyframe_states):
            kf_idx = len(keyframe_states) - 1

        # Get the keyframe state
        kf_state = keyframe_states[kf_idx]
        kf_time = keyframe_times[kf_idx]

        if abs(t - kf_time) < 1e-6:
            # Exactly at keyframe - use keyframe state directly
            full_states.append(copy.deepcopy(kf_state))
        else:
            # Between keyframes - propagate from keyframe using gyro
            # Find simulation indices from keyframe time to current time
            kf_sim_idx = np.argmin(np.abs(sim_data.t - kf_time))

            # Start from keyframe state
            q_prop = kf_state.ori.copy()
            b_est = kf_state.gyro_bias.copy()

            # Propagate through each gyro measurement
            for i in range(kf_sim_idx, k):
                if i + 1 < len(sim_data.t):
                    dt = sim_data.t[i + 1] - sim_data.t[i]
                    omega = sim_data.omega_meas[i + 1]
                    if not np.any(np.isnan(omega)):
                        omega_corrected = omega - b_est
                        q_prop = q_prop.propagate(omega_corrected, dt)

            # Create interpolated state
            interp_state = NominalState(ori=q_prop, gyro_bias=b_est.copy())
            full_states.append(interp_state)

        full_times.append(t)

    return np.array(full_times), full_states


def run_eskf(
    sim_data,
    config_path: str,
    initial_error_deg: float = 10.0,
) -> EstimationResult:
    """Run ESKF estimator."""
    print(f"\n  Running ESKF (init err: {initial_error_deg}°)...")
    start_time = time.time()

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2, att_err_rad**2, att_err_rad**2, 1e-6, 1e-6, 1e-6])
    eskf = ESKF(P0=P0, config_path=config_path)

    # Initial state with error
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    # Create perturbation of specified magnitude
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)  # ~10 deg total
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()
    b0_est = np.zeros(3)

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    states = [copy.deepcopy(x_est.nom)]
    times = [sim_data.t[0]]

    for k in range(1, len(sim_data.t)):
        omega_k = sim_data.omega_meas[k] if not np.any(np.isnan(sim_data.omega_meas[k])) else sim_data.omega_meas[k-1]
        dt_k = sim_data.t[k] - sim_data.t[k-1]

        # Predict
        x_est = eskf.predict(x_est, omega_k, dt_k)

        # Environment
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # Magnetometer update
        mag_meas = sim_data.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(x_est=x_est, y=mag_meas, sensor_type=SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        # Sun sensor update
        sun_meas = sim_data.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(x_est=x_est, y=sun_meas, sensor_type=SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        # Star tracker update
        st_meas = sim_data.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(x_est=x_est, y=q_meas, sensor_type=SensorType.STAR_TRACKER)
            except ValueError:
                pass

        states.append(copy.deepcopy(x_est.nom))
        times.append(sim_data.t[k])

    runtime = time.time() - start_time
    print(f"    Completed in {runtime:.2f}s")

    return EstimationResult(
        name="ESKF",
        times=np.array(times),
        states=states,
        runtime_s=runtime,
    )


def run_hybrid(
    sim_data,
    config_path: str,
    initial_error_deg: float = 10.0,
) -> EstimationResult:
    """Run Hybrid ESKF+FGO estimator."""
    print(f"\n  Running Hybrid (init err: {initial_error_deg}°)...")
    start_time = time.time()

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2, att_err_rad**2, att_err_rad**2, 1e-6, 1e-6, 1e-6])
    hybrid = HybridEstimator(
        P0=P0,
        config_path=config_path,
        fgo_window_duration=120.0,  # 120 second sliding window (more ST measurements)
        fgo_optimize_interval=60.0,  # Batch optimize every 60 seconds
        use_robust=True,
        correction_mode="normal",
    )

    # Initial state with error (same as ESKF for fair comparison)
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true.multiply(Quaternion.from_avec(perturb)).normalize()
    b0_est = np.zeros(3)

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    states = [copy.deepcopy(x_est.nom)]
    times = [sim_data.t[0]]

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

        # Hybrid step
        x_est, _ = hybrid.step(
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

        states.append(copy.deepcopy(x_est.nom))
        times.append(t)

    runtime = time.time() - start_time
    print(f"    Completed in {runtime:.2f}s")

    return EstimationResult(
        name="Hybrid",
        times=np.array(times),
        states=states,
        runtime_s=runtime,
    )


def run_fgo(
    sim_data,
    config_path: str,
    use_robust: bool = False,
    use_isam2: bool = False,
) -> EstimationResult:
    """Run FGO estimator."""
    mode = "iSAM2" if use_isam2 else "Batch"
    robust = "+M-est" if use_robust else ""
    name = f"FGO-{mode}{robust}"

    print(f"\n  Running {name}...")
    start_time = time.time()

    env = OrbitEnvironmentModel()
    fgo = KeyframeFGO(
        config_path=config_path,
        use_robust=use_robust,
        use_isam2=use_isam2,
        use_rk4=True,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)

    # Interpolate to full rate for fair comparison
    times, states = interpolate_to_full_rate(
        np.array(kf_times), kf_states, sim_data
    )

    runtime = time.time() - start_time
    print(f"    Completed in {runtime:.2f}s ({len(kf_states)} keyframes -> {len(states)} samples)")

    return EstimationResult(
        name=name,
        times=times,
        states=states,
        runtime_s=runtime,
    )


def compute_metrics(
    result: EstimationResult,
    sim_data,
    convergence_time: float = 30.0,
) -> EstimationResult:
    """Compute error metrics for an estimation result.

    Args:
        result: Estimation result to compute metrics for
        sim_data: Simulation data with ground truth
        convergence_time: Time in seconds to exclude for steady-state RMS
    """
    # Find matching indices
    attitude_errors = []
    bias_errors = []

    for i, t in enumerate(result.times):
        # Find closest simulation time
        idx = np.argmin(np.abs(sim_data.t - t))

        q_true = Quaternion.from_array(sim_data.q_true[idx])
        q_est = result.states[i].ori

        # Attitude error (right-multiply: q_err = q_true * q_est^{-1})
        q_err = q_true.multiply(q_est.conjugate())
        angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
        attitude_errors.append(np.rad2deg(angle_err))

        # Bias error
        b_true = sim_data.b_g_true[idx]
        b_est = result.states[i].gyro_bias
        b_err = b_est - b_true
        # Convert rad/s to deg/h
        bias_errors.append(np.rad2deg(b_err) * 3600)

    result.attitude_errors_deg = np.array(attitude_errors)
    result.bias_errors_deg_h = np.array(bias_errors)
    result.final_attitude_error_deg = attitude_errors[-1] if attitude_errors else None
    result.rms_attitude_error_deg = np.sqrt(np.mean(np.array(attitude_errors)**2)) if attitude_errors else None

    # Compute steady-state RMS (excluding convergence period)
    steady_state_mask = result.times >= convergence_time
    if np.any(steady_state_mask):
        steady_errors = result.attitude_errors_deg[steady_state_mask]
        result.rms_steady_state_deg = np.sqrt(np.mean(steady_errors**2))
    else:
        result.rms_steady_state_deg = result.rms_attitude_error_deg

    return result


def print_comparison_table(results: List[EstimationResult]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("ESTIMATOR COMPARISON RESULTS")
    print("=" * 90)

    print(f"\n{'Estimator':<20} {'Runtime':>8} {'Final Err':>12} {'RMS (all)':>12} {'RMS (t>30s)':>12} {'Samples':>8}")
    print("-" * 90)

    for r in results:
        runtime = f"{r.runtime_s:.2f}s"
        final_err = f"{r.final_attitude_error_deg:.4f}°" if r.final_attitude_error_deg is not None else "N/A"
        rms_err = f"{r.rms_attitude_error_deg:.4f}°" if r.rms_attitude_error_deg is not None else "N/A"
        rms_ss = f"{r.rms_steady_state_deg:.4f}°" if r.rms_steady_state_deg is not None else "N/A"
        samples = f"{len(r.times)}"
        print(f"{r.name:<20} {runtime:>8} {final_err:>12} {rms_err:>12} {rms_ss:>12} {samples:>8}")

    print("-" * 90)
    print("Note: RMS (t>30s) excludes first 30s convergence period")


def run_comparison(
    sim_run_id: int,
    db_path: str,
    config_path: str,
    estimators: List[str],
) -> List[EstimationResult]:
    """Run comparison on a single simulation."""
    print("=" * 80)
    print(f"RUNNING ESTIMATOR COMPARISON")
    print("=" * 80)
    print(f"Simulation ID: {sim_run_id}")
    print(f"Config: {config_path}")
    print(f"Estimators: {', '.join(estimators)}")

    # Load simulation data
    db = SimulationDatabase(db_path)
    sim_data = db.load_run(sim_run_id)

    print(f"\nSimulation: {sim_data.t[-1]:.1f}s duration, {len(sim_data.t)} samples")

    results = []

    # Run selected estimators
    if "eskf" in estimators:
        result = run_eskf(sim_data, config_path)
        result = compute_metrics(result, sim_data)
        results.append(result)

    if "fgo_batch" in estimators:
        result = run_fgo(sim_data, config_path, use_robust=False, use_isam2=False)
        result = compute_metrics(result, sim_data)
        results.append(result)

    if "fgo_batch_m" in estimators:
        result = run_fgo(sim_data, config_path, use_robust=True, use_isam2=False)
        result = compute_metrics(result, sim_data)
        results.append(result)

    if "fgo_isam2" in estimators:
        result = run_fgo(sim_data, config_path, use_robust=False, use_isam2=True)
        result = compute_metrics(result, sim_data)
        results.append(result)

    if "fgo_isam2_m" in estimators:
        result = run_fgo(sim_data, config_path, use_robust=True, use_isam2=True)
        result = compute_metrics(result, sim_data)
        results.append(result)

    if "hybrid" in estimators:
        result = run_hybrid(sim_data, config_path)
        result = compute_metrics(result, sim_data)
        results.append(result)

    # Print comparison
    print_comparison_table(results)

    return results


def generate_simulation(
    config_path: str,
    db_path: str,
) -> int:
    """Generate a new simulation."""
    print("=" * 80)
    print("GENERATING SIMULATION DATA")
    print("=" * 80)
    print(f"Config: {config_path}")

    config = load_yaml(config_path)

    generator = EnhancedAttitudeDataGenerator(
        db_path=db_path,
        config_path=config_path
    )

    sim_cfg = SimulationConfig(
        T=config['time']['sim_T'],
        dt=config['time']['sim_dt'],
        start_jd=config['time']['start_jd']
    )

    print(f"Duration: {sim_cfg.T:.0f}s, Timestep: {sim_cfg.dt}s")

    run_id = generator.run(sim_cfg)
    print(f"\nGenerated simulation with ID: {run_id}")

    return run_id


def main():
    parser = argparse.ArgumentParser(
        description="Run estimator comparison on simulation scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new simulation data"
    )
    parser.add_argument(
        "--sim-id",
        type=int,
        help="Simulation run ID to use"
    )
    parser.add_argument(
        "--sim-ids",
        type=str,
        help="Comma-separated list of simulation run IDs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_baseline_short.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="simulations.db",
        help="Path to database"
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default="eskf,fgo_batch,fgo_batch_m,fgo_isam2,fgo_isam2_m,hybrid",
        help="Comma-separated list of estimators to run"
    )

    args = parser.parse_args()

    # Parse estimators
    estimators = [e.strip() for e in args.estimators.split(",")]
    valid_estimators = {"eskf", "fgo_batch", "fgo_batch_m", "fgo_isam2", "fgo_isam2_m", "hybrid"}
    for e in estimators:
        if e not in valid_estimators:
            print(f"Error: Unknown estimator '{e}'")
            print(f"Valid estimators: {', '.join(valid_estimators)}")
            return 1

    # Generate or use existing simulation
    sim_ids = []

    if args.generate:
        sim_id = generate_simulation(args.config, args.db)
        sim_ids.append(sim_id)
    elif args.sim_id:
        sim_ids.append(args.sim_id)
    elif args.sim_ids:
        sim_ids = [int(x.strip()) for x in args.sim_ids.split(",")]
    else:
        print("Error: Must specify --generate, --sim-id, or --sim-ids")
        return 1

    # Run comparison on each simulation
    all_results = {}
    for sim_id in sim_ids:
        try:
            results = run_comparison(
                sim_run_id=sim_id,
                db_path=args.db,
                config_path=args.config,
                estimators=estimators,
            )
            all_results[sim_id] = results
        except Exception as e:
            print(f"\nError processing simulation {sim_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SIMULATIONS")
        print("=" * 80)

        for sim_id, results in all_results.items():
            print(f"\nSimulation {sim_id}:")
            for r in results:
                print(f"  {r.name:<20}: RMS={r.rms_attitude_error_deg:.4f} deg, Final={r.final_attitude_error_deg:.4f} deg")

    print("\nComparison complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
