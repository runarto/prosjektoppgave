#!/usr/bin/env python3
"""
Monte Carlo evaluation of transient star tracker fault scenario.

A random 1-10 degree error is applied to star tracker measurements
for a 60-second window, then removed. Tests estimator robustness
and recovery capability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, Tuple
import time

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO
from estimation.redundant_estimator import RedundantEstimator
from utilities.utils import load_yaml

# =============================================================================
# Publication-quality plot settings
# =============================================================================
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


class MockEnvironment:
    """Mock environment using pre-computed simulation data."""
    def __init__(self, sim_data):
        self.sim_data = sim_data

    def _find_idx(self, jd):
        return np.argmin(np.abs(self.sim_data.jd - jd))

    def get_r_eci(self, jd):
        return np.array([7000e3, 0, 0])

    def get_B_eci(self, r_eci, jd):
        return self.sim_data.b_eci[self._find_idx(jd)]

    def get_sun_eci(self, jd):
        return self.sim_data.s_eci[self._find_idx(jd)]


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error magnitude in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def apply_star_tracker_fault(sim_data, fault_start: float, fault_duration: float,
                              error_magnitude_deg: float):
    """
    Apply transient error to star tracker measurements.

    Args:
        sim_data: Original simulation data
        fault_start: Start time of fault (seconds)
        fault_duration: Duration of fault (seconds)
        error_magnitude_deg: Error magnitude in degrees (random direction)

    Returns:
        Modified simulation data
    """
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

    fault_end = fault_start + fault_duration

    # Generate random error axis (unit vector)
    error_axis = np.random.randn(3)
    error_axis = error_axis / np.linalg.norm(error_axis)
    error_angle_rad = np.deg2rad(error_magnitude_deg)

    # Create error quaternion
    q_error = Quaternion.from_avec(error_axis * error_angle_rad)

    for k in range(len(modified.t)):
        t = modified.t[k]
        if fault_start <= t < fault_end:
            if not np.any(np.isnan(modified.st_meas[k])):
                # Apply error to star tracker measurement
                q_meas = Quaternion.from_array(modified.st_meas[k])
                q_corrupted = (q_meas @ q_error).normalize()
                modified.st_meas[k] = q_corrupted.as_array()

    return modified


def run_eskf_full(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run ESKF and return full time series."""
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

    times, errors = [], []

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

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors)


def run_isam2_full(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run iSAM2 and return full time series."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(config_path=config_path, use_isam2=True, use_rk4=True)

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    errors = []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.array(times), np.array(errors)


def run_redundant_full(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Run Redundant estimator and return full time series + modes."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(P0=P0, config_path=config_path)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times, errors, modes = [], [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_n = sim_data.b_eci[k] if z_mag is not None else None
        s_n = sim_data.s_eci[k] if z_sun is not None else None

        x_est, _, _, mode = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))
        modes.append(mode)

    return np.array(times), np.array(errors), modes


def run_monte_carlo(n_runs: int, config_path: str, db: SimulationDatabase,
                    fault_start: float, fault_duration: float) -> Dict:
    """Run Monte Carlo for star tracker fault scenario."""

    # Get run IDs
    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    results = {
        'ESKF': {'errors_during': [], 'errors_after': [], 'max_during': [], 'recovery_time': []},
        'iSAM2': {'errors_during': [], 'errors_after': [], 'max_during': [], 'recovery_time': []},
        'Redundant': {'errors_during': [], 'errors_after': [], 'max_during': [], 'recovery_time': []},
    }

    # Store one example run for plotting
    example_run = None

    fault_end = fault_start + fault_duration
    recovery_threshold = 0.1  # degrees - consider "recovered" when error < this

    for i, run_id in enumerate(run_ids):
        # Random error magnitude between 1-10 degrees
        error_mag = np.random.uniform(1.0, 10.0)

        print(f"  Run {i+1}/{n_runs} (error={error_mag:.1f} deg)...", end=" ", flush=True)

        sim_data_base = db.load_run(run_id)
        sim_data = apply_star_tracker_fault(sim_data_base, fault_start, fault_duration, error_mag)

        # Run ESKF
        try:
            times, errors = run_eskf_full(sim_data, config_path)
            during_mask = (times >= fault_start) & (times < fault_end)
            after_mask = times >= fault_end + 30  # 30s after fault ends

            results['ESKF']['errors_during'].append(np.mean(errors[during_mask]))
            results['ESKF']['errors_after'].append(np.mean(errors[after_mask]))
            results['ESKF']['max_during'].append(np.max(errors[during_mask]))

            # Find recovery time
            post_fault = times >= fault_end
            recovered_idx = np.where(post_fault & (errors < recovery_threshold))[0]
            if len(recovered_idx) > 0:
                recovery_time = times[recovered_idx[0]] - fault_end
            else:
                recovery_time = np.inf
            results['ESKF']['recovery_time'].append(recovery_time)

            if example_run is None:
                example_run = {'times': times, 'ESKF': errors}
        except Exception as e:
            print(f"ESKF failed: {e}")

        # Run iSAM2
        try:
            times, errors = run_isam2_full(sim_data, config_path)
            during_mask = (times >= fault_start) & (times < fault_end)
            after_mask = times >= fault_end + 30

            results['iSAM2']['errors_during'].append(np.mean(errors[during_mask]))
            results['iSAM2']['errors_after'].append(np.mean(errors[after_mask]))
            results['iSAM2']['max_during'].append(np.max(errors[during_mask]))

            post_fault = times >= fault_end
            recovered_idx = np.where(post_fault & (errors < recovery_threshold))[0]
            if len(recovered_idx) > 0:
                recovery_time = times[recovered_idx[0]] - fault_end
            else:
                recovery_time = np.inf
            results['iSAM2']['recovery_time'].append(recovery_time)

            if example_run is not None:
                example_run['iSAM2'] = errors
        except Exception as e:
            print(f"iSAM2 failed: {e}")

        # Run Redundant
        try:
            times, errors, modes = run_redundant_full(sim_data, config_path)
            during_mask = (times >= fault_start) & (times < fault_end)
            after_mask = times >= fault_end + 30

            results['Redundant']['errors_during'].append(np.mean(errors[during_mask]))
            results['Redundant']['errors_after'].append(np.mean(errors[after_mask]))
            results['Redundant']['max_during'].append(np.max(errors[during_mask]))

            post_fault = times >= fault_end
            recovered_idx = np.where(post_fault & (errors < recovery_threshold))[0]
            if len(recovered_idx) > 0:
                recovery_time = times[recovered_idx[0]] - fault_end
            else:
                recovery_time = np.inf
            results['Redundant']['recovery_time'].append(recovery_time)

            if example_run is not None:
                example_run['Redundant'] = errors
                example_run['modes'] = modes
        except Exception as e:
            print(f"Redundant failed: {e}")

        print("done")

    return results, example_run


def create_config(base_config_path: str, output_path: str):
    """Create config with standard noise scale."""
    config = load_yaml(base_config_path)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return output_path


def plot_example_run(example_run: Dict, fault_start: float, fault_duration: float,
                     save_path: str):
    """Plot example run showing fault and recovery."""
    fig, ax = plt.subplots(figsize=(12, 6))

    times = example_run['times']
    fault_end = fault_start + fault_duration

    # Shade fault region
    ax.axvspan(fault_start, fault_end, alpha=0.3, color='red', label='Fault Period')

    # Align arrays to minimum length
    min_len = min(len(times), len(example_run['ESKF']),
                  len(example_run.get('iSAM2', times)),
                  len(example_run.get('Redundant', times)))
    times = times[:min_len]

    # Plot errors
    ax.semilogy(times, example_run['ESKF'][:min_len], 'C0-', linewidth=1.5, label='ESKF')
    if 'iSAM2' in example_run:
        ax.semilogy(times, example_run['iSAM2'][:min_len], 'C1-', linewidth=1.5, label='iSAM2')
    if 'Redundant' in example_run:
        ax.semilogy(times, example_run['Redundant'][:min_len], 'C2--', linewidth=1.5, label='Redundant')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Star Tracker Transient Fault: Example Run')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, times[-1]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_mc_results(results: Dict, save_path: str):
    """Plot Monte Carlo results as bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    estimators = ['ESKF', 'iSAM2', 'Redundant']
    colors = ['C0', 'C1', 'C2']
    x = np.arange(len(estimators))

    # Max error during fault
    ax = axes[0]
    means = [np.mean(results[e]['max_during']) for e in estimators]
    stds = [np.std(results[e]['max_during']) for e in estimators]
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.set_ylabel('Max Error [deg]')
    ax.set_title('Peak Error During Fault')
    ax.grid(True, alpha=0.3, axis='y')

    # Mean error after recovery
    ax = axes[1]
    means = [np.mean(results[e]['errors_after']) for e in estimators]
    stds = [np.std(results[e]['errors_after']) for e in estimators]
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.set_ylabel('Mean Error [deg]')
    ax.set_title('Steady-State After Recovery')
    ax.grid(True, alpha=0.3, axis='y')

    # Recovery time
    ax = axes[2]
    recovery_times = []
    for e in estimators:
        rt = [r for r in results[e]['recovery_time'] if r < np.inf]
        recovery_times.append(rt if rt else [0])
    means = [np.mean(rt) for rt in recovery_times]
    stds = [np.std(rt) for rt in recovery_times]
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.set_ylabel('Recovery Time [s]')
    ax.set_title('Time to Recover (<0.1$^\\circ$)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for results."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Star Tracker Transient Fault: Monte Carlo Results ($N=10$ runs, 1--10$^\circ$ error for 60s)}
\label{tab:st_fault}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Estimator} & \textbf{Peak Error} [deg] & \textbf{Post-Recovery} [deg] & \textbf{Recovery Time} [s] \\
\midrule
"""
    for est in ['ESKF', 'iSAM2', 'Redundant']:
        peak = np.mean(results[est]['max_during'])
        peak_std = np.std(results[est]['max_during'])
        post = np.mean(results[est]['errors_after'])
        post_std = np.std(results[est]['errors_after'])
        rt = [r for r in results[est]['recovery_time'] if r < np.inf]
        if rt:
            rt_mean = np.mean(rt)
            rt_std = np.std(rt)
            rt_str = f"${rt_mean:.1f} \\pm {rt_std:.1f}$"
        else:
            rt_str = "N/A"

        table += f"{est} & ${peak:.2f} \\pm {peak_std:.2f}$ & ${post:.4f} \\pm {post_std:.4f}$ & {rt_str} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}"""
    return table


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 10
    fault_start = 100.0  # Start fault after convergence
    fault_duration = 60.0  # 60 second fault

    print("=" * 70)
    print("STAR TRACKER TRANSIENT FAULT - MONTE CARLO")
    print("=" * 70)
    print(f"Runs: {n_runs}")
    print(f"Fault window: {fault_start}s - {fault_start + fault_duration}s")
    print(f"Error magnitude: 1-10 degrees (random per run)")

    config_path = create_config(base_config, 'configs/config_st_fault.yaml')
    db = SimulationDatabase(db_path)

    start_time = time.time()
    results, example_run = run_monte_carlo(n_runs, config_path, db, fault_start, fault_duration)
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for est in ['ESKF', 'iSAM2', 'Redundant']:
        peak = np.mean(results[est]['max_during'])
        post = np.mean(results[est]['errors_after'])
        rt = [r for r in results[est]['recovery_time'] if r < np.inf]
        rt_str = f"{np.mean(rt):.1f}s" if rt else "N/A"
        print(f"{est:12s}: Peak={peak:.2f} deg, Post-recovery={post:.4f} deg, Recovery={rt_str}")

    # Generate outputs
    print("\n--- Generating Outputs ---")

    if example_run:
        plot_example_run(example_run, fault_start, fault_duration, 'st_fault_example.png')

    plot_mc_results(results, 'st_fault_mc_results.png')

    latex_table = generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)

    with open('st_fault_results.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved: st_fault_results.tex")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
