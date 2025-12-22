#!/usr/bin/env python3
"""
Analysis script for:
1. Per-axis (roll/pitch/yaw) estimation performance across scenarios
2. Monte Carlo investigation of ESKF eclipse case high variance

Specifically investigating:
- Which axes perform worse in different scenarios
- Why ESKF shows high variance (0.064° ± 0.144°) in eclipse conditions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO
from utilities.utils import load_yaml

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['legend.fontsize'] = 14
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['lines.linewidth'] = 1.5
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['grid.linewidth'] = 0.6
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = 'gray'
rcParams['mathtext.fontset'] = 'dejavuserif'


@dataclass
class AxisErrors:
    """Per-axis error statistics."""
    roll_errors: np.ndarray  # Time series
    pitch_errors: np.ndarray
    yaw_errors: np.ndarray
    total_errors: np.ndarray
    times: np.ndarray


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    eclipse: bool = False
    mag_bias: Optional[np.ndarray] = None
    true_gyro_bias: Optional[np.ndarray] = None


def compute_euler_errors(q_est: Quaternion, q_true: Quaternion) -> np.ndarray:
    """Compute roll, pitch, yaw errors in degrees.

    Returns:
        [roll_err, pitch_err, yaw_err] in degrees
    """
    # Compute error quaternion: q_err = q_true^-1 @ q_est
    q_err = q_true.conjugate() @ q_est
    q_err = q_err.normalize()

    # Convert to Euler angles
    euler_err = q_err.as_euler()  # [roll, pitch, yaw] in radians

    return np.rad2deg(euler_err)


def compute_total_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute total attitude error magnitude in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def modify_sim_data(sim_data, scenario: ScenarioConfig):
    """Apply scenario modifications to simulation data."""
    class ModifiedData:
        pass

    modified = ModifiedData()
    modified.t = sim_data.t.copy()
    modified.jd = sim_data.jd.copy()
    modified.q_true = sim_data.q_true.copy()
    modified.omega_meas = sim_data.omega_meas.copy()
    modified.mag_meas = sim_data.mag_meas.copy()
    modified.sun_meas = sim_data.sun_meas.copy()
    modified.st_meas = sim_data.st_meas.copy()
    modified.b_eci = sim_data.b_eci.copy()
    modified.s_eci = sim_data.s_eci.copy()
    modified.b_g_true = sim_data.b_g_true.copy()

    if scenario.eclipse:
        modified.sun_meas = np.full_like(modified.sun_meas, np.nan)

    if scenario.mag_bias is not None:
        for k in range(len(modified.mag_meas)):
            if not np.any(np.isnan(modified.mag_meas[k])):
                biased = modified.mag_meas[k] + scenario.mag_bias
                modified.mag_meas[k] = biased / np.linalg.norm(biased)

    if scenario.true_gyro_bias is not None:
        modified.omega_meas = modified.omega_meas + scenario.true_gyro_bias
        modified.b_g_true = modified.b_g_true + scenario.true_gyro_bias

    return modified


def run_eskf_with_axis_tracking(sim_data, config_path: str, ss_start: float = 100.0) -> AxisErrors:
    """Run ESKF and track per-axis errors over time."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    eskf = ESKF(P0=P0, config_path=config_path, chi2_threshold=1e10)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    total_errors = []
    times = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        x_est = eskf.predict(x_est, omega, dt)

        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.mag_meas[k],
                                   SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except:
                pass

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k],
                                   SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except:
                pass

        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except:
                pass

        # Track errors
        euler_err = compute_euler_errors(x_est.nom.ori, q_true)
        total_err = compute_total_error(x_est.nom.ori, q_true)

        times.append(t)
        roll_errors.append(abs(euler_err[0]))
        pitch_errors.append(abs(euler_err[1]))
        yaw_errors.append(abs(euler_err[2]))
        total_errors.append(total_err)

    return AxisErrors(
        roll_errors=np.array(roll_errors),
        pitch_errors=np.array(pitch_errors),
        yaw_errors=np.array(yaw_errors),
        total_errors=np.array(total_errors),
        times=np.array(times)
    )


def run_monte_carlo_eclipse(n_runs: int, db: SimulationDatabase, config_path: str,
                            ss_start: float = 100.0) -> Dict:
    """Run Monte Carlo analysis for eclipse case with detailed tracking."""

    results = {
        'roll_means': [],
        'pitch_means': [],
        'yaw_means': [],
        'total_means': [],
        'roll_stds': [],
        'pitch_stds': [],
        'yaw_stds': [],
        'total_stds': [],
        'run_details': [],  # Store detailed info about each run
    }

    # Get run IDs
    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(run_ids) < n_runs:
        print(f"  Warning: Only {len(run_ids)} runs available, requested {n_runs}")
        n_runs = len(run_ids)

    eclipse_scenario = ScenarioConfig(name="Eclipse", eclipse=True)

    for i, run_id in enumerate(run_ids):
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)

        sim_data_base = db.load_run(run_id)
        sim_data = modify_sim_data(sim_data_base, eclipse_scenario)

        try:
            axis_errors = run_eskf_with_axis_tracking(sim_data, config_path, ss_start)

            # Compute steady-state statistics
            ss_mask = axis_errors.times >= ss_start

            ss_roll = axis_errors.roll_errors[ss_mask]
            ss_pitch = axis_errors.pitch_errors[ss_mask]
            ss_yaw = axis_errors.yaw_errors[ss_mask]
            ss_total = axis_errors.total_errors[ss_mask]

            results['roll_means'].append(np.mean(ss_roll))
            results['pitch_means'].append(np.mean(ss_pitch))
            results['yaw_means'].append(np.mean(ss_yaw))
            results['total_means'].append(np.mean(ss_total))

            results['roll_stds'].append(np.std(ss_roll))
            results['pitch_stds'].append(np.std(ss_pitch))
            results['yaw_stds'].append(np.std(ss_yaw))
            results['total_stds'].append(np.std(ss_total))

            # Store run details for outlier analysis
            run_detail = {
                'run_id': run_id,
                'total_mean': np.mean(ss_total),
                'roll_mean': np.mean(ss_roll),
                'pitch_mean': np.mean(ss_pitch),
                'yaw_mean': np.mean(ss_yaw),
                'max_total': np.max(ss_total),
                'n_st': np.sum(~np.isnan(sim_data.st_meas[:, 0])),
                'n_mag': np.sum(~np.isnan(sim_data.mag_meas[:, 0])),
            }
            results['run_details'].append(run_detail)

            print(f"total={np.mean(ss_total):.4f}°, roll={np.mean(ss_roll):.4f}°, "
                  f"pitch={np.mean(ss_pitch):.4f}°, yaw={np.mean(ss_yaw):.4f}°")

        except Exception as e:
            print(f"FAILED: {e}")

    return results


def analyze_scenarios_per_axis(db: SimulationDatabase, config_path: str,
                                n_runs: int = 10, ss_start: float = 100.0) -> Dict:
    """Analyze per-axis performance across different scenarios."""

    scenarios = [
        ScenarioConfig(name="Baseline"),
        ScenarioConfig(name="Eclipse", eclipse=True),
        ScenarioConfig(name="Gyro Drift", true_gyro_bias=np.deg2rad(np.array([1.0, 1.0, 1.0])) / 3600),
        ScenarioConfig(name="Mag Bias", mag_bias=np.array([0.05, 0.0, 0.0])),
    ]

    all_results = {}

    # Get run IDs
    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.name}")
        print(f"{'='*60}")

        scenario_results = {
            'roll_means': [],
            'pitch_means': [],
            'yaw_means': [],
            'total_means': [],
        }

        for i, run_id in enumerate(run_ids):
            print(f"  Run {i+1}/{len(run_ids)}...", end=" ")

            sim_data_base = db.load_run(run_id)
            sim_data = modify_sim_data(sim_data_base, scenario)

            try:
                axis_errors = run_eskf_with_axis_tracking(sim_data, config_path, ss_start)

                ss_mask = axis_errors.times >= ss_start
                scenario_results['roll_means'].append(np.mean(axis_errors.roll_errors[ss_mask]))
                scenario_results['pitch_means'].append(np.mean(axis_errors.pitch_errors[ss_mask]))
                scenario_results['yaw_means'].append(np.mean(axis_errors.yaw_errors[ss_mask]))
                scenario_results['total_means'].append(np.mean(axis_errors.total_errors[ss_mask]))

                print("done")
            except Exception as e:
                print(f"FAILED: {e}")

        # Compute statistics
        all_results[scenario.name] = {
            'roll': {'mean': np.mean(scenario_results['roll_means']),
                     'std': np.std(scenario_results['roll_means'])},
            'pitch': {'mean': np.mean(scenario_results['pitch_means']),
                      'std': np.std(scenario_results['pitch_means'])},
            'yaw': {'mean': np.mean(scenario_results['yaw_means']),
                    'std': np.std(scenario_results['yaw_means'])},
            'total': {'mean': np.mean(scenario_results['total_means']),
                      'std': np.std(scenario_results['total_means'])},
        }

        print(f"\nResults for {scenario.name}:")
        print(f"  Roll:  {all_results[scenario.name]['roll']['mean']:.4f} ± "
              f"{all_results[scenario.name]['roll']['std']:.4f}°")
        print(f"  Pitch: {all_results[scenario.name]['pitch']['mean']:.4f} ± "
              f"{all_results[scenario.name]['pitch']['std']:.4f}°")
        print(f"  Yaw:   {all_results[scenario.name]['yaw']['mean']:.4f} ± "
              f"{all_results[scenario.name]['yaw']['std']:.4f}°")
        print(f"  Total: {all_results[scenario.name]['total']['mean']:.4f} ± "
              f"{all_results[scenario.name]['total']['std']:.4f}°")

    return all_results


def plot_axis_comparison(results: Dict, save_path: str):
    """Create bar plot comparing per-axis errors across scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = list(results.keys())
    x = np.arange(len(scenarios))
    width = 0.2

    colors = {'roll': 'C0', 'pitch': 'C1', 'yaw': 'C2', 'total': 'C3'}

    for i, axis in enumerate(['roll', 'pitch', 'yaw', 'total']):
        means = [results[s][axis]['mean'] for s in scenarios]
        stds = [results[s][axis]['std'] for s in scenarios]

        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=axis.capitalize(),
               color=colors[axis], capsize=4, alpha=0.8)

    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_xlabel('Scenario')
    ax.set_title('Per-Axis Estimation Errors Across Scenarios (ESKF)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_eclipse_mc_distribution(results: Dict, save_path: str):
    """Plot distribution of errors in eclipse Monte Carlo."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total error histogram
    ax = axes[0, 0]
    ax.hist(results['total_means'], bins=20, alpha=0.7, color='C0', edgecolor='black')
    ax.axvline(np.mean(results['total_means']), color='red', linestyle='--',
               label=f'Mean: {np.mean(results["total_means"]):.4f}°')
    ax.axvline(np.median(results['total_means']), color='green', linestyle=':',
               label=f'Median: {np.median(results["total_means"]):.4f}°')
    ax.set_xlabel('Mean Total Error [deg]')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Total Errors (Eclipse)')
    ax.legend()

    # Per-axis comparison
    ax = axes[0, 1]
    ax.hist(results['roll_means'], bins=15, alpha=0.6, label='Roll', color='C0')
    ax.hist(results['pitch_means'], bins=15, alpha=0.6, label='Pitch', color='C1')
    ax.hist(results['yaw_means'], bins=15, alpha=0.6, label='Yaw', color='C2')
    ax.set_xlabel('Mean Error [deg]')
    ax.set_ylabel('Count')
    ax.set_title('Per-Axis Error Distribution (Eclipse)')
    ax.legend()

    # Scatter: roll vs yaw
    ax = axes[1, 0]
    ax.scatter(results['roll_means'], results['yaw_means'], alpha=0.6, s=50)
    ax.set_xlabel('Roll Error [deg]')
    ax.set_ylabel('Yaw Error [deg]')
    ax.set_title('Roll vs Yaw Errors')
    ax.grid(True, alpha=0.3)

    # Run details - identify outliers
    ax = axes[1, 1]
    run_details = results['run_details']
    total_errors = [r['total_mean'] for r in run_details]
    st_counts = [r['n_st'] for r in run_details]

    scatter = ax.scatter(st_counts, total_errors, c=range(len(run_details)),
                         cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Number of Star Tracker Measurements')
    ax.set_ylabel('Mean Total Error [deg]')
    ax.set_title('Error vs Star Tracker Availability')
    ax.grid(True, alpha=0.3)

    # Mark outliers (>2 std from mean)
    mean_err = np.mean(total_errors)
    std_err = np.std(total_errors)
    for i, (st, err) in enumerate(zip(st_counts, total_errors)):
        if err > mean_err + 2 * std_err:
            ax.annotate(f'Run {run_details[i]["run_id"]}', (st, err), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_outlier_runs(results: Dict):
    """Analyze what makes some runs have high error."""
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS")
    print("=" * 60)

    run_details = results['run_details']
    total_errors = [r['total_mean'] for r in run_details]

    mean_err = np.mean(total_errors)
    std_err = np.std(total_errors)
    median_err = np.median(total_errors)

    print(f"\nOverall Statistics:")
    print(f"  Mean: {mean_err:.4f}°")
    print(f"  Std:  {std_err:.4f}°")
    print(f"  Median: {median_err:.4f}°")
    print(f"  Min: {np.min(total_errors):.4f}°")
    print(f"  Max: {np.max(total_errors):.4f}°")

    # Find outliers
    outlier_threshold = mean_err + 2 * std_err
    outliers = [r for r in run_details if r['total_mean'] > outlier_threshold]

    print(f"\nOutliers (>{outlier_threshold:.4f}°):")
    if outliers:
        for r in outliers:
            print(f"  Run {r['run_id']}: total={r['total_mean']:.4f}°, "
                  f"roll={r['roll_mean']:.4f}°, pitch={r['pitch_mean']:.4f}°, "
                  f"yaw={r['yaw_mean']:.4f}°, ST={r['n_st']}, MAG={r['n_mag']}")
    else:
        print("  No outliers found")

    # Find which axis contributes most to high-error runs
    high_error_runs = sorted(run_details, key=lambda x: x['total_mean'], reverse=True)[:5]
    low_error_runs = sorted(run_details, key=lambda x: x['total_mean'])[:5]

    print(f"\nTop 5 Highest Error Runs:")
    for r in high_error_runs:
        dominant = max([('roll', r['roll_mean']), ('pitch', r['pitch_mean']),
                        ('yaw', r['yaw_mean'])], key=lambda x: x[1])
        print(f"  Run {r['run_id']}: {r['total_mean']:.4f}° (dominant: {dominant[0]}={dominant[1]:.4f}°)")

    print(f"\nTop 5 Lowest Error Runs:")
    for r in low_error_runs:
        print(f"  Run {r['run_id']}: {r['total_mean']:.4f}°")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    st_counts = [r['n_st'] for r in run_details]
    correlation = np.corrcoef(st_counts, total_errors)[0, 1]
    print(f"  Star tracker count vs error correlation: {correlation:.3f}")

    # Axis-specific variance
    print(f"\nAxis-Specific Variance:")
    roll_var = np.var(results['roll_means'])
    pitch_var = np.var(results['pitch_means'])
    yaw_var = np.var(results['yaw_means'])
    total_var = np.var(results['total_means'])

    print(f"  Roll variance:  {roll_var:.6f} deg²")
    print(f"  Pitch variance: {pitch_var:.6f} deg²")
    print(f"  Yaw variance:   {yaw_var:.6f} deg²")
    print(f"  Total variance: {total_var:.6f} deg²")

    # Eclipse-specific observation
    print("\n" + "-" * 60)
    print("ECLIPSE SCENARIO INSIGHTS:")
    print("-" * 60)
    print("""
In eclipse conditions, the sun sensor is unavailable. This affects observability:

1. YAW (Z-axis rotation):
   - Sun sensor provides direct yaw information when sun is not aligned with body axis
   - Without sun sensor, yaw observability depends on:
     * Star tracker (most accurate, but infrequent: ~0.2 Hz)
     * Magnetometer (provides some heading info, but less direct)

2. Why variance is high:
   - The high variance (±0.144°) likely comes from runs where:
     * Star tracker measurements are sparse or rejected
     * Magnetometer-only estimation has higher uncertainty
     * Yaw angle drifts between star tracker updates

3. ESKF vs iSAM2 behavior:
   - ESKF: Sequential updates mean yaw can drift between measurements
   - iSAM2: Batch optimization can better constrain yaw through factor graph

4. Check if yaw dominates the high-variance runs.
""")


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 20  # Monte Carlo runs
    ss_start = 100.0  # Steady-state start time

    print("=" * 70)
    print("AXIS-SPECIFIC ANALYSIS AND ECLIPSE MONTE CARLO")
    print("=" * 70)

    db = SimulationDatabase(db_path)

    # Part 1: Per-axis analysis across scenarios
    print("\n" + "=" * 70)
    print("PART 1: PER-AXIS ANALYSIS ACROSS SCENARIOS")
    print("=" * 70)

    axis_results = analyze_scenarios_per_axis(db, base_config, n_runs=n_runs, ss_start=ss_start)
    plot_axis_comparison(axis_results, 'axis_comparison_scenarios.png')

    # Part 2: Detailed Eclipse Monte Carlo
    print("\n" + "=" * 70)
    print("PART 2: ECLIPSE MONTE CARLO ANALYSIS")
    print("=" * 70)

    eclipse_results = run_monte_carlo_eclipse(n_runs, db, base_config, ss_start)
    plot_eclipse_mc_distribution(eclipse_results, 'eclipse_mc_analysis.png')
    analyze_outlier_runs(eclipse_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nPer-Axis Performance Summary:")
    for scenario, results in axis_results.items():
        worst_axis = max(['roll', 'pitch', 'yaw'], key=lambda x: results[x]['mean'])
        print(f"  {scenario}: worst axis = {worst_axis} ({results[worst_axis]['mean']:.4f}°)")

    print("\nEclipse Monte Carlo Summary:")
    print(f"  Total error: {np.mean(eclipse_results['total_means']):.4f} ± "
          f"{np.std(eclipse_results['total_means']):.4f}°")
    print(f"  Roll error:  {np.mean(eclipse_results['roll_means']):.4f} ± "
          f"{np.std(eclipse_results['roll_means']):.4f}°")
    print(f"  Pitch error: {np.mean(eclipse_results['pitch_means']):.4f} ± "
          f"{np.std(eclipse_results['pitch_means']):.4f}°")
    print(f"  Yaw error:   {np.mean(eclipse_results['yaw_means']):.4f} ± "
          f"{np.std(eclipse_results['yaw_means']):.4f}°")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
