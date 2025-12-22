#!/usr/bin/env python3
"""
Compare different robust loss functions and thresholds for iSAM2 smoother.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, Tuple, List
import time

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState
from estimation.keyframe_fgo import KeyframeFGO
from utilities.utils import load_yaml

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12


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
    """Apply transient error to star tracker measurements."""
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

    error_axis = np.random.randn(3)
    error_axis = error_axis / np.linalg.norm(error_axis)
    error_angle_rad = np.deg2rad(error_magnitude_deg)
    q_error = Quaternion.from_avec(error_axis * error_angle_rad)

    for k in range(len(modified.t)):
        t = modified.t[k]
        if fault_start <= t < fault_end:
            if not np.any(np.isnan(modified.st_meas[k])):
                q_meas = Quaternion.from_array(modified.st_meas[k])
                q_corrupted = (q_meas @ q_error).normalize()
                modified.st_meas[k] = q_corrupted.as_array()

    return modified


def run_isam2(sim_data, config_path: str, kernel: str, param: float) -> Tuple[np.ndarray, np.ndarray]:
    """Run iSAM2 with specified robust kernel."""
    env = MockEnvironment(sim_data)

    use_robust = kernel != "none"

    fgo = KeyframeFGO(
        config_path=config_path,
        use_isam2=True,
        use_rk4=True,
        use_robust=use_robust,
        robust_kernel=kernel if use_robust else "huber",
        robust_param=param,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    errors = []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.array(times), np.array(errors)


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


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 10
    fault_start = 100.0
    fault_duration = 60.0
    fault_end = fault_start + fault_duration

    # Configurations to test
    # Format: (name, kernel, param)
    configs = [
        # No robust
        ("L2 (no robust)", "none", 1.0),
        # Huber with different thresholds
        ("Huber k=1.345", "huber", 1.345),
        ("Huber k=0.5", "huber", 0.5),
        ("Huber k=0.1", "huber", 0.1),
        # Other kernels (more aggressive outlier rejection)
        ("Cauchy k=1.0", "cauchy", 1.0),
        ("Cauchy k=0.1", "cauchy", 0.1),
        ("Welsch k=1.0", "welsch", 1.0),
        ("Tukey k=4.685", "tukey", 4.685),
        ("Tukey k=1.0", "tukey", 1.0),
        ("GemanMcClure k=1.0", "geman", 1.0),
    ]

    print("=" * 70)
    print("ROBUST KERNEL COMPARISON FOR iSAM2")
    print("=" * 70)
    print(f"Fault: {fault_start}s - {fault_end}s, 1-10 deg error")
    print(f"Monte Carlo runs: {n_runs}")
    print(f"\nConfigurations: {len(configs)}")
    for name, kernel, param in configs:
        print(f"  - {name}")
    print()

    config_path = create_config(base_config, 'configs/config_kernel_test.yaml')
    db = SimulationDatabase(db_path)

    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Results storage
    results = {name: {'peak_during': [], 'mean_after': []} for name, _, _ in configs}

    start_time = time.time()

    for i, run_id in enumerate(run_ids):
        error_mag = np.random.uniform(1.0, 10.0)
        print(f"Run {i+1}/{n_runs} (error={error_mag:.1f} deg)")

        sim_data_base = db.load_run(run_id)
        sim_data = apply_star_tracker_fault(sim_data_base, fault_start, fault_duration, error_mag)

        for name, kernel, param in configs:
            try:
                times, errors = run_isam2(sim_data, config_path, kernel, param)

                during_mask = (times >= fault_start) & (times < fault_end)
                after_mask = times >= fault_end + 30

                results[name]['peak_during'].append(np.max(errors[during_mask]))
                results[name]['mean_after'].append(np.mean(errors[after_mask]))

                print(f"  {name}: peak={np.max(errors[during_mask]):.2f} deg")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                results[name]['peak_during'].append(np.nan)
                results[name]['mean_after'].append(np.nan)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Peak During [deg]':>20} {'Post-Recovery [deg]':>20}")
    print("-" * 70)

    for name, _, _ in configs:
        peak_vals = [v for v in results[name]['peak_during'] if not np.isnan(v)]
        after_vals = [v for v in results[name]['mean_after'] if not np.isnan(v)]
        if peak_vals:
            peak_mean = np.mean(peak_vals)
            peak_std = np.std(peak_vals)
            after_mean = np.mean(after_vals)
            after_std = np.std(after_vals)
            print(f"{name:<25} {peak_mean:>8.2f} ± {peak_std:<8.2f} {after_mean:>8.4f} ± {after_std:<8.4f}")
        else:
            print(f"{name:<25} {'FAILED':>20} {'FAILED':>20}")

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of robust loss functions for iSAM2 during star tracker fault ($N=""" + str(n_runs) + r"""$ runs)}
\label{tab:robust_kernels}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Loss Function} & \textbf{Peak Error} & \textbf{Post-Recovery} \\
\midrule
"""
    for name, _, _ in configs:
        peak_vals = [v for v in results[name]['peak_during'] if not np.isnan(v)]
        after_vals = [v for v in results[name]['mean_after'] if not np.isnan(v)]
        if peak_vals:
            peak_mean = np.mean(peak_vals)
            peak_std = np.std(peak_vals)
            after_mean = np.mean(after_vals)
            after_std = np.std(after_vals)
            latex += f"{name} & ${peak_mean:.2f}^\\circ \\pm {peak_std:.2f}^\\circ$ & ${after_mean:.3f}^\\circ \\pm {after_std:.3f}^\\circ$ \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print(latex)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [name for name, _, _ in configs]
    x = np.arange(len(names))

    # Peak error
    ax = axes[0]
    means = []
    stds = []
    for name, _, _ in configs:
        peak_vals = [v for v in results[name]['peak_during'] if not np.isnan(v)]
        means.append(np.mean(peak_vals) if peak_vals else 0)
        stds.append(np.std(peak_vals) if peak_vals else 0)

    colors = ['C3'] + ['C0']*3 + ['C1']*2 + ['C2'] + ['C4']*2 + ['C5']
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Peak Error [deg]')
    ax.set_title('Peak Error During Fault')
    ax.grid(True, alpha=0.3, axis='y')

    # Post-recovery
    ax = axes[1]
    means = []
    stds = []
    for name, _, _ in configs:
        after_vals = [v for v in results[name]['mean_after'] if not np.isnan(v)]
        means.append(np.mean(after_vals) if after_vals else 0)
        stds.append(np.std(after_vals) if after_vals else 0)

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Error [deg]')
    ax.set_title('Steady-State After Recovery')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('robust_kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('robust_kernel_comparison.pdf', dpi=150, bbox_inches='tight')
    print(f"\nSaved: robust_kernel_comparison.png")


if __name__ == "__main__":
    main()
