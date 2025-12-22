#!/usr/bin/env python3
"""
Compare robustness mechanisms for star tracker transient fault.

Tests:
1. ESKF with chi-squared gate (default)
2. ESKF without chi-squared gate
3. iSAM2 with M-estimator (Huber)
4. iSAM2 without M-estimator (L2)
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
rcParams['legend.fontsize'] = 12
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


def run_eskf(sim_data, config_path: str, use_chi2: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run ESKF with or without chi-squared gating."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    # chi2_threshold: low value enables gating, high value disables it
    chi2_thresh = 7.81 if use_chi2 else 1e10

    eskf = ESKF(P0=P0, config_path=config_path,
                chi2_threshold=chi2_thresh,
                chi2_threshold_sun=chi2_thresh,
                chi2_threshold_star=chi2_thresh)
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


def run_isam2(sim_data, config_path: str, use_robust: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run iSAM2 with or without M-estimator."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(
        config_path=config_path,
        use_isam2=True,
        use_rk4=True,
        use_robust=use_robust,  # Toggle M-estimator
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    errors = []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.array(times), np.array(errors)


def run_redundant(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run Redundant estimator with chi-squared (ESKF) and M-estimator (smoother)."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    # Initialize with chi-squared and M-estimator enabled
    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        use_robust=True,  # Enable M-estimator in smoother
        disagreement_threshold_deg=1.0,
        consecutive_disagreements_to_switch=3,
    )

    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        # Prepare measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = None
        if not np.any(np.isnan(sim_data.st_meas[k])):
            z_st = Quaternion.from_array(sim_data.st_meas[k])

        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        try:
            x_est, output_nom, disagreement, mode = redundant.step(
                x_est, t, jd, omega, dt,
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_n=B_n, s_n=s_n
            )
        except Exception:
            pass

        times.append(t)
        errors.append(compute_attitude_error_deg(output_nom.ori, q_true))

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


def run_monte_carlo(n_runs: int, config_path: str, db: SimulationDatabase,
                    fault_start: float, fault_duration: float) -> Dict:
    """Run Monte Carlo for all configurations."""

    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    configs = [
        ('ESKF + $\\chi^2$', 'eskf', True),
        ('ESKF (no gate)', 'eskf', False),
        ('iSAM2 + Huber', 'isam2', True),
        ('iSAM2 (L2)', 'isam2', False),
        ('Redundant', 'redundant', None),
    ]

    results = {name: {'errors_during': [], 'errors_after': [], 'max_during': []}
               for name, _, _ in configs}

    fault_end = fault_start + fault_duration

    # Store example run
    example_run = None

    for i, run_id in enumerate(run_ids):
        error_mag = np.random.uniform(1.0, 10.0)
        print(f"  Run {i+1}/{n_runs} (error={error_mag:.1f} deg)...", end=" ", flush=True)

        sim_data_base = db.load_run(run_id)
        sim_data = apply_star_tracker_fault(sim_data_base, fault_start, fault_duration, error_mag)

        for name, method, flag in configs:
            try:
                if method == 'eskf':
                    times, errors = run_eskf(sim_data, config_path, use_chi2=flag)
                elif method == 'isam2':
                    times, errors = run_isam2(sim_data, config_path, use_robust=flag)
                elif method == 'redundant':
                    times, errors = run_redundant(sim_data, config_path)

                during_mask = (times >= fault_start) & (times < fault_end)
                after_mask = times >= fault_end + 30

                results[name]['errors_during'].append(np.mean(errors[during_mask]))
                results[name]['errors_after'].append(np.mean(errors[after_mask]))
                results[name]['max_during'].append(np.max(errors[during_mask]))

                if example_run is None:
                    example_run = {'times': times}
                if i == 0:  # Store first run for example plot
                    example_run[name] = errors

            except Exception as e:
                print(f"{name} failed: {e}")

        print("done")

    return results, example_run


def plot_example(example_run: Dict, fault_start: float, fault_duration: float, save_path: str):
    """Plot example run comparing all configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    times = example_run['times']
    fault_end = fault_start + fault_duration

    ax.axvspan(fault_start, fault_end, alpha=0.2, color='red', label='Fault Period')

    colors = {'ESKF + $\\chi^2$': 'C0', 'ESKF (no gate)': 'C0',
              'iSAM2 + Huber': 'C1', 'iSAM2 (L2)': 'C1',
              'Redundant': 'C2'}
    styles = {'ESKF + $\\chi^2$': '-', 'ESKF (no gate)': '--',
              'iSAM2 + Huber': '-', 'iSAM2 (L2)': '--',
              'Redundant': '-'}

    for name in ['ESKF + $\\chi^2$', 'ESKF (no gate)', 'iSAM2 + Huber', 'iSAM2 (L2)', 'Redundant']:
        if name in example_run:
            min_len = min(len(times), len(example_run[name]))
            ax.semilogy(times[:min_len], example_run[name][:min_len],
                       color=colors[name], linestyle=styles[name],
                       linewidth=1.5, label=name)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Robustness Mechanism Comparison: Star Tracker Fault')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, times[-1]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_mc_comparison(results: Dict, save_path: str):
    """Plot Monte Carlo comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = ['ESKF + $\\chi^2$', 'ESKF (no gate)', 'iSAM2 + Huber', 'iSAM2 (L2)', 'Redundant']
    colors = ['C0', 'C0', 'C1', 'C1', 'C2']
    hatches = ['', '//', '', '//', '']
    x = np.arange(len(names))

    # Peak error during fault
    ax = axes[0]
    means = [np.mean(results[n]['max_during']) for n in names]
    stds = [np.std(results[n]['max_during']) for n in names]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(['ESKF\n+$\\chi^2$', 'ESKF\n(no gate)', 'iSAM2\n+Huber', 'iSAM2\n(L2)', 'Redundant'])
    ax.set_ylabel('Peak Error [deg]')
    ax.set_title('Peak Error During Fault')
    ax.grid(True, alpha=0.3, axis='y')

    # Post-recovery error
    ax = axes[1]
    means = [np.mean(results[n]['errors_after']) for n in names]
    stds = [np.std(results[n]['errors_after']) for n in names]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(['ESKF\n+$\\chi^2$', 'ESKF\n(no gate)', 'iSAM2\n+Huber', 'iSAM2\n(L2)', 'Redundant'])
    ax.set_ylabel('Mean Error [deg]')
    ax.set_title('Steady-State After Recovery')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Effect of Robustness Mechanisms on Star Tracker Fault Handling ($N=10$ runs)}
\label{tab:robustness_comparison}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Configuration} & \textbf{Peak Error During Fault} [deg] & \textbf{Post-Recovery} [deg] \\
\midrule
"""
    for name in ['ESKF + $\\chi^2$', 'ESKF (no gate)', 'iSAM2 + Huber', 'iSAM2 (L2)', 'Redundant']:
        peak = np.mean(results[name]['max_during'])
        peak_std = np.std(results[name]['max_during'])
        post = np.mean(results[name]['errors_after'])
        post_std = np.std(results[name]['errors_after'])

        # LaTeX-safe name
        latex_name = name.replace('$\\chi^2$', '$\\chi^2$')

        table += f"{latex_name} & ${peak:.2f} \\pm {peak_std:.2f}$ & ${post:.4f} \\pm {post_std:.4f}$ \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}"""
    return table


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 10
    fault_start = 100.0
    fault_duration = 60.0

    print("=" * 70)
    print("ROBUSTNESS MECHANISM COMPARISON")
    print("=" * 70)
    print(f"Configurations tested:")
    print(f"  1. ESKF with chi-squared gate (threshold = 7.81)")
    print(f"  2. ESKF without chi-squared gate")
    print(f"  3. iSAM2 with Huber M-estimator")
    print(f"  4. iSAM2 with L2 cost (no robust)")
    print(f"  5. Redundant (chi-squared + M-estimator)")
    print(f"\nFault: {fault_start}s - {fault_start + fault_duration}s, 1-10 deg error")
    print(f"Monte Carlo runs: {n_runs}")

    config_path = create_config(base_config, 'configs/config_robust_test.yaml')
    db = SimulationDatabase(db_path)

    start_time = time.time()
    results, example_run = run_monte_carlo(n_runs, config_path, db, fault_start, fault_duration)
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<20} {'Peak During [deg]':>20} {'Post-Recovery [deg]':>20}")
    print("-" * 70)
    for name in ['ESKF + $\\chi^2$', 'ESKF (no gate)', 'iSAM2 + Huber', 'iSAM2 (L2)', 'Redundant']:
        peak = np.mean(results[name]['max_during'])
        post = np.mean(results[name]['errors_after'])
        display_name = name.replace('$\\chi^2$', 'χ²')
        print(f"{display_name:<20} {peak:>20.2f} {post:>20.4f}")

    # Generate outputs
    print("\n--- Generating Outputs ---")

    if example_run:
        plot_example(example_run, fault_start, fault_duration, 'robustness_example.png')

    plot_mc_comparison(results, 'robustness_mc_comparison.png')

    latex_table = generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)

    with open('robustness_comparison.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved: robustness_comparison.tex")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
