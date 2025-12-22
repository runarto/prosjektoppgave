"""
Baseline Performance Comparison: ESKF vs Fixed-Lag Smoother vs Redundant Architecture

This script compares three estimation approaches:
1. ESKF (Error-State Kalman Filter) - standalone
2. IncrementalFixedLagSmoother - standalone
3. Redundant Architecture (ESKF + Smoother with fault detection)

Tests are run with:
- Zero-noise measurements (implementation verification)
- Normal noise measurements (baseline performance)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.utils import load_yaml
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import EskfState, NominalState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel


@dataclass
class EstimatorResult:
    """Results from running an estimator."""
    name: str
    times: np.ndarray
    errors_deg: np.ndarray
    bias_errors: np.ndarray  # deg/h

    # Statistics
    mean_error_deg: float = 0.0
    std_error_deg: float = 0.0
    max_error_deg: float = 0.0
    final_error_deg: float = 0.0

    # Steady-state (t > 30s)
    ss_mean_error_deg: float = 0.0
    ss_std_error_deg: float = 0.0

    def compute_statistics(self, ss_start: float = 30.0):
        """Compute error statistics."""
        self.mean_error_deg = np.mean(self.errors_deg)
        self.std_error_deg = np.std(self.errors_deg)
        self.max_error_deg = np.max(self.errors_deg)
        self.final_error_deg = self.errors_deg[-1]

        # Steady-state
        ss_mask = self.times > ss_start
        if np.any(ss_mask):
            self.ss_mean_error_deg = np.mean(self.errors_deg[ss_mask])
            self.ss_std_error_deg = np.std(self.errors_deg[ss_mask])


def compute_attitude_error(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    dq = q_true @ q_est.conjugate()
    angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))
    return np.rad2deg(angle)


def compute_bias_error(b_true: np.ndarray, b_est: np.ndarray) -> float:
    """Compute bias error magnitude in deg/h."""
    error_rad_s = np.linalg.norm(b_true - b_est)
    return np.rad2deg(error_rad_s) * 3600  # Convert to deg/h


def run_eskf(
    sim_data,
    env: OrbitEnvironmentModel,
    config_path: str,
    q_init: Quaternion,
    P0: np.ndarray,
) -> EstimatorResult:
    """Run standalone ESKF."""
    print("  Running ESKF...")

    eskf = ESKF(P0=P0, config_path=config_path)

    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    times = []
    errors = []
    bias_errors = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = t - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        jd = sim_data.jd[k]

        # Prediction
        x = eskf.predict(x, omega, dt)

        # Measurements
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            r_eci = env.get_r_eci(jd)
            B_n = env.get_B_eci(r_eci, jd)
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            s_n = env.get_sun_eci(jd)
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.st_meas[k])):
            z_st = Quaternion.from_array(sim_data.st_meas[k])
            try:
                x = eskf.update(x, z_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # Compute errors
        q_true = Quaternion.from_array(sim_data.q_true[k])
        error = compute_attitude_error(q_true, x.nom.ori)
        bias_error = compute_bias_error(sim_data.b_g_true[k], x.nom.gyro_bias)

        times.append(t)
        errors.append(error)
        bias_errors.append(bias_error)

    result = EstimatorResult(
        name="ESKF",
        times=np.array(times),
        errors_deg=np.array(errors),
        bias_errors=np.array(bias_errors),
    )
    result.compute_statistics()
    return result


def run_smoother_standalone(
    sim_data,
    env: OrbitEnvironmentModel,
    config_path: str,
    q_init: Quaternion,
    smoother_lag: float = 60.0,
) -> EstimatorResult:
    """Run standalone IncrementalFixedLagSmoother."""
    print("  Running Standalone Smoother...")

    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=smoother_lag,
        use_robust=True,
    )

    # Initialize
    smoother.initialize(sim_data.t[0], q_init, np.zeros(3))

    times = []
    errors = []
    bias_errors = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = t - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        jd = sim_data.jd[k]

        # Integrate gyro
        smoother.integrate_gyro(omega, dt)

        # Check for measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_eci = None
        s_eci = None
        if z_mag is not None:
            r_eci = env.get_r_eci(jd)
            B_eci = env.get_B_eci(r_eci, jd)
        if z_sun is not None:
            s_eci = env.get_sun_eci(jd)

        # Add measurement if present - ONLY record error when smoother updates
        if z_mag is not None or z_sun is not None or z_st is not None:
            state = smoother.add_measurement(
                t=t, jd=jd,
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=B_eci, s_eci=s_eci,
            )

            if state is not None:
                q_true = Quaternion.from_array(sim_data.q_true[k])
                error = compute_attitude_error(q_true, state.ori)
                bias_error = compute_bias_error(sim_data.b_g_true[k], state.gyro_bias)

                times.append(t)
                errors.append(error)
                bias_errors.append(bias_error)

    result = EstimatorResult(
        name="Smoother",
        times=np.array(times),
        errors_deg=np.array(errors),
        bias_errors=np.array(bias_errors),
    )
    result.compute_statistics()
    return result


def run_redundant(
    sim_data,
    env: OrbitEnvironmentModel,
    config_path: str,
    q_init: Quaternion,
    P0: np.ndarray,
    smoother_lag: float = 60.0,
) -> Tuple[EstimatorResult, EstimatorResult, Dict]:
    """
    Run redundant architecture (ESKF + Smoother).

    Returns results for both ESKF and Smoother components, plus statistics.
    """
    print("  Running Redundant Architecture...")

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=smoother_lag,
        use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
        agreement_threshold_deg=0.5,
        consecutive_agreements_to_recover=10,
    )

    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    times = []
    redundant_errors = []  # Error from whichever estimator is PRIMARY
    redundant_bias_errors = []
    disagreements = []
    primaries = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = t - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        jd = sim_data.jd[k]

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_n = None
        s_n = None
        if z_mag is not None:
            r_eci = env.get_r_eci(jd)
            B_n = env.get_B_eci(r_eci, jd)
        if z_sun is not None:
            s_n = env.get_sun_eci(jd)

        # Step redundant estimator
        x, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x,
            t=t, jd=jd,
            omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st,
            B_n=B_n, s_n=s_n,
        )

        # Compute error from the SELECTED PRIMARY estimator
        q_true = Quaternion.from_array(sim_data.q_true[k])

        if primary == "ESKF":
            primary_state = x.nom
        else:
            primary_state = smoother_state

        redundant_error = compute_attitude_error(q_true, primary_state.ori)
        redundant_bias_error = compute_bias_error(sim_data.b_g_true[k], primary_state.gyro_bias)

        times.append(t)
        redundant_errors.append(redundant_error)
        redundant_bias_errors.append(redundant_bias_error)
        disagreements.append(disagreement)
        primaries.append(primary)

    # Create single result for redundant architecture (selected primary)
    redundant_result = EstimatorResult(
        name="Redundant",
        times=np.array(times),
        errors_deg=np.array(redundant_errors),
        bias_errors=np.array(redundant_bias_errors),
    )
    redundant_result.compute_statistics()

    # Additional statistics
    stats = redundant.get_statistics()
    stats['disagreements'] = np.array(disagreements)
    stats['primaries'] = primaries

    return redundant_result, stats


def plot_comparison(
    results: List[EstimatorResult],
    title: str,
    output_path: str,
    redundant_stats: Optional[Dict] = None,
):
    """Create comparison plot."""
    n_plots = 3 if redundant_stats is None else 4
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)

    colors = {'ESKF': 'blue', 'Smoother': 'red', 'Redundant': 'green'}

    # Attitude errors
    ax = axes[0]
    for result in results:
        color = colors.get(result.name, 'gray')
        ax.semilogy(result.times, result.errors_deg, color=color, linewidth=0.5,
                    alpha=0.7, label=f"{result.name} (SS: {result.ss_mean_error_deg:.4f}°)")
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bias errors
    ax = axes[1]
    for result in results:
        color = colors.get(result.name, 'gray')
        ax.plot(result.times, result.bias_errors, color=color, linewidth=0.5, alpha=0.7, label=result.name)
    ax.set_ylabel('Bias Error [deg/h]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Zoomed steady-state
    ax = axes[2]
    for result in results:
        color = colors.get(result.name, 'gray')
        ss_mask = result.times > 30
        if np.any(ss_mask):
            ax.plot(result.times[ss_mask], result.errors_deg[ss_mask], color=color,
                   linewidth=0.5, alpha=0.7, label=result.name)
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Steady-State (t > 30s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Disagreement (if available)
    if redundant_stats is not None and 'disagreements' in redundant_stats:
        ax = axes[3]
        times = results[0].times  # Use first result's times
        ax.plot(times, redundant_stats['disagreements'], 'purple', linewidth=0.5, alpha=0.7)
        ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
        ax.set_ylabel('Disagreement [deg]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {output_path}")


def print_results_table(results: List[EstimatorResult], title: str):
    """Print results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"{'Estimator':<20} {'Mean [deg]':>12} {'Std [deg]':>12} {'SS Mean':>12} {'SS Std':>12}")
    print(f"{'-' * 70}")
    for r in results:
        print(f"{r.name:<20} {r.mean_error_deg:>12.6f} {r.std_error_deg:>12.6f} "
              f"{r.ss_mean_error_deg:>12.6f} {r.ss_std_error_deg:>12.6f}")
    print(f"{'=' * 70}")


def run_comparison(
    config_path: str,
    output_prefix: str,
    sim_run_id: Optional[int] = None,
    db_path: str = "simulations.db",
    smoother_lag: float = 60.0,
    regenerate_sim: bool = False,
    data_gen_config: Optional[str] = None,  # Separate config for data generation
):
    """
    Run full comparison of all estimators.

    Args:
        config_path: Path to configuration file for ESTIMATORS
        output_prefix: Prefix for output files
        sim_run_id: Simulation run ID (if None, generates new simulation)
        db_path: Path to simulation database
        smoother_lag: Fixed-lag smoother window duration
        regenerate_sim: If True, regenerate simulation even if sim_run_id provided
        data_gen_config: Config for data generation (if different from estimator config)
    """
    print(f"\n{'#' * 70}")
    print(f"BASELINE COMPARISON: {output_prefix}")
    print(f"{'#' * 70}")
    print(f"Estimator config: {config_path}")
    if data_gen_config:
        print(f"Data generation config: {data_gen_config}")
    print(f"Smoother lag: {smoother_lag}s")

    # Load or generate simulation
    db = SimulationDatabase(db_path)

    if sim_run_id is None or regenerate_sim:
        print("\nGenerating new simulation...")
        gen_config_path = data_gen_config if data_gen_config else config_path
        config = load_yaml(gen_config_path)
        generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=gen_config_path)
        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd'],
            run_name=config['simulation']['run_name']
        )
        sim_run_id = generator.run(sim_cfg)
        print(f"  Created simulation run ID: {sim_run_id}")

    sim = db.load_run(sim_run_id)
    env = OrbitEnvironmentModel()

    print(f"\nSimulation: {len(sim.t)} samples ({sim.t[-1]:.1f}s)")

    # Initial conditions - perturbed attitude
    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.3, 0.3, 0.3]))  # ~30 deg initial error
    q_init = q_true_0 @ perturb

    P0 = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6])

    print("\nRunning estimators...")

    # Run all estimators
    eskf_result = run_eskf(sim, env, config_path, q_init, P0)
    smoother_result = run_smoother_standalone(sim, env, config_path, q_init, smoother_lag)
    redundant_result, redundant_stats = run_redundant(
        sim, env, config_path, q_init, P0, smoother_lag
    )

    # Collect results: ESKF, Smoother, Redundant (selected primary)
    all_results = [eskf_result, smoother_result, redundant_result]

    # Print results
    print_results_table(all_results, f"Results: {output_prefix}")

    # Print redundant architecture statistics
    print(f"\nRedundant Architecture Statistics:")
    print(f"  Switch events: {redundant_stats['total_switch_events']}")
    print(f"  Final primary: {redundant_stats['primary']}")
    print(f"  Max disagreement: {redundant_stats['max_disagreement_deg']:.4f}°")

    # Create plot
    plot_path = f"{output_prefix}_comparison.png"
    plot_comparison(all_results, f"Baseline Comparison: {output_prefix}", plot_path, redundant_stats)

    return all_results, redundant_stats


def main():
    """Run baseline comparison tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline performance comparison")
    parser.add_argument("--zero-noise", action="store_true", help="Run zero-noise verification")
    parser.add_argument("--normal", action="store_true", help="Run normal noise comparison")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--lag", type=float, default=60.0, help="Smoother lag in seconds")
    args = parser.parse_args()

    if args.all or (not args.zero_noise and not args.normal):
        args.zero_noise = True
        args.normal = True

    results = {}

    # Zero-noise verification
    # Generate data with PERFECT measurements, run estimators with NORMAL config
    if args.zero_noise:
        print("\n" + "=" * 70)
        print("ZERO-NOISE VERIFICATION TEST")
        print("=" * 70)
        print("Purpose: Verify implementations converge to near-zero error")
        print("Data: Perfect measurements (zero noise)")
        print("Estimators: Normal tuning (default config)")
        print("Expected: All estimators should achieve < 0.01° error")

        results['zero_noise'], _ = run_comparison(
            config_path="configs/config_baseline_short.yaml",  # Estimator config
            data_gen_config="configs/config_perfect_measurements.yaml",  # Data gen config
            output_prefix="baseline_zero_noise",
            regenerate_sim=True,
            smoother_lag=args.lag,
        )

    # Normal noise comparison
    if args.normal:
        print("\n" + "=" * 70)
        print("NORMAL NOISE BASELINE TEST")
        print("=" * 70)
        print("Purpose: Compare estimator performance under nominal conditions")

        results['normal'], _ = run_comparison(
            config_path="configs/config_baseline_short.yaml",
            output_prefix="baseline_normal",
            regenerate_sim=True,
            smoother_lag=args.lag,
        )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
