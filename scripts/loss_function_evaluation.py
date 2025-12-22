#!/usr/bin/env python3
"""
Comprehensive evaluation of robust loss functions across multiple scenarios.

Uses the FixedLagAttitudeSmoother (incremental) for consistency with fault_scenario_comparison.

Scenarios:
1. Baseline - nominal operation
2. Magnetometer bias - persistent 3° bias
3. Star tracker transient fault - 5° error for 50s
4. Eclipse - sun sensor dropout for 100s

Loss functions:
- L2 (no robust)
- Huber k=1.345 (default)
- Huber k=0.1 (aggressive)
- Cauchy k=1.0
- Cauchy k=0.1
- Geman-McClure k=1.0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, Tuple, List, Optional
import time

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error magnitude in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


# =============================================================================
# Scenario Modifiers
# =============================================================================

def apply_baseline(sim_data):
    """No modification - baseline scenario."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci', 'b_g_true']:
        setattr(modified, attr, getattr(sim_data, attr).copy())
    return modified


def apply_mag_bias(sim_data, bias_magnitude: float = 0.62):
    """Apply persistent magnetometer bias."""
    modified = apply_baseline(sim_data)

    # Bias direction (fixed in body frame)
    bias_dir = np.array([1.0, 0.5, 0.3])
    bias_dir = bias_dir / np.linalg.norm(bias_dir)
    bias = bias_magnitude * bias_dir

    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.mag_meas[k])):
            modified.mag_meas[k] = modified.mag_meas[k] + bias

    return modified


def apply_st_fault(sim_data, fault_start: float = 100.0, fault_end: float = 150.0,
                   error_deg: float = 5.0):
    """Apply star tracker fault with fixed error."""
    modified = apply_baseline(sim_data)

    # Fixed rotation axis
    axis = np.array([0.5, 0.5, 0.707])
    axis = axis / np.linalg.norm(axis)
    error_rad = np.deg2rad(error_deg)
    q_error = Quaternion.from_avec(axis * error_rad)

    for k in range(len(modified.t)):
        t = modified.t[k]
        if fault_start <= t < fault_end:
            if not np.any(np.isnan(modified.st_meas[k])):
                q_meas = Quaternion.from_array(modified.st_meas[k])
                q_corrupted = (q_meas @ q_error).normalize()
                modified.st_meas[k] = q_corrupted.as_array()

    return modified


def apply_eclipse(sim_data, eclipse_start: float = 100.0, eclipse_end: float = 200.0):
    """Apply eclipse - sun sensor dropout."""
    modified = apply_baseline(sim_data)

    for k in range(len(modified.t)):
        t = modified.t[k]
        if eclipse_start <= t < eclipse_end:
            modified.sun_meas[k] = np.array([np.nan, np.nan, np.nan])

    return modified


# =============================================================================
# Estimator Runner
# =============================================================================

def run_smoother(sim_data, config_path: str, kernel: str, param: float,
                 lag: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """Run FixedLagAttitudeSmoother with specified robust kernel."""
    use_robust = kernel != "none"

    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=lag,
        use_robust=use_robust,
        robust_kernel=kernel if use_robust else "huber",
        robust_param=param,
        normalize_mag=True,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    smoother.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    dt = sim_data.t[1] - sim_data.t[0]

    # Store keyframe times and states for interpolation
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
                t=sim_data.t[k],
                jd=sim_data.jd[k],
                z_mag=z_mag,
                z_sun=z_sun,
                z_st=z_st,
                B_eci=sim_data.b_eci[k],
                s_eci=sim_data.s_eci[k],
            )
            state = smoother.get_state()
            if state is not None:
                keyframe_times.append(sim_data.t[k])
                keyframe_states.append(state.ori.copy())

    # Interpolate states at all timesteps using SLERP
    keyframe_times = np.array(keyframe_times)
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
            t0 = keyframe_times[idx - 1]
            t1 = keyframe_times[idx]
            q0 = keyframe_states[idx - 1]
            q1 = keyframe_states[idx]

            if t1 > t0:
                alpha = (t_k - t0) / (t1 - t0)
                q_interp = q0.slerp(q1, alpha)
            else:
                q_interp = q0

        err = compute_attitude_error_deg(q_interp, q_true)
        errors.append(err)

    return sim_data.t, np.array(errors)


def main():
    config_path = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 5

    # Loss functions to test
    loss_functions = [
        ("L2", "none", 1.0),
        ("Huber k=1.345", "huber", 1.345),
        ("Huber k=0.1", "huber", 0.1),
        ("Cauchy k=1.0", "cauchy", 1.0),
        ("Cauchy k=0.1", "cauchy", 0.1),
        ("Geman-McClure", "geman", 1.0),
    ]

    # Scenarios with consistent parameters
    scenarios = [
        ("Baseline", "baseline"),
        ("Mag Bias (0.62)", "mag_bias"),
        ("ST Fault (5°)", "st_fault"),
        ("Eclipse (100s)", "eclipse"),
    ]

    print("=" * 80)
    print("LOSS FUNCTION EVALUATION - FixedLagAttitudeSmoother (Incremental)")
    print("=" * 80)
    print(f"Monte Carlo runs per configuration: {n_runs}")
    print(f"Loss functions: {len(loss_functions)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total configurations: {len(loss_functions) * len(scenarios)}")
    print()

    db = SimulationDatabase(db_path)

    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(run_ids) < n_runs:
        print(f"Warning: Only {len(run_ids)} runs available")
        n_runs = len(run_ids)

    # Results storage: results[scenario][loss_function] = list of mean errors
    results = {s_name: {l_name: [] for l_name, _, _ in loss_functions}
               for s_name, _ in scenarios}

    start_time = time.time()

    for scenario_name, scenario_type in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        for i, run_id in enumerate(run_ids):
            print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)

            sim_data_base = db.load_run(run_id)

            # Apply scenario modification
            if scenario_type == "baseline":
                sim_data = apply_baseline(sim_data_base)
            elif scenario_type == "mag_bias":
                sim_data = apply_mag_bias(sim_data_base, bias_magnitude=0.62)
            elif scenario_type == "st_fault":
                sim_data = apply_st_fault(sim_data_base, fault_start=100.0,
                                          fault_end=150.0, error_deg=5.0)
            elif scenario_type == "eclipse":
                sim_data = apply_eclipse(sim_data_base, eclipse_start=100.0,
                                         eclipse_end=200.0)

            for loss_name, kernel, param in loss_functions:
                try:
                    times, errors = run_smoother(sim_data, config_path, kernel, param)

                    # Measure steady-state performance (last 50% of simulation)
                    mask = times >= times[-1] * 0.5
                    mean_error = np.mean(errors[mask])
                    results[scenario_name][loss_name].append(mean_error)
                except Exception as e:
                    print(f"\n    {loss_name} failed: {e}")
                    results[scenario_name][loss_name].append(np.nan)

            print("done")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ==========================================================================
    # Generate Results Table
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - Mean Attitude Error [deg]")
    print("=" * 80)

    # Header
    header = f"{'Loss Function':<20}"
    for s_name, _ in scenarios:
        header += f" {s_name[:15]:>15}"
    print(header)
    print("-" * 80)

    for loss_name, _, _ in loss_functions:
        row = f"{loss_name:<20}"
        for s_name, _ in scenarios:
            vals = [v for v in results[s_name][loss_name] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                row += f" {mean:>15.4f}"
            else:
                row += f" {'FAILED':>15}"
        print(row)

    # ==========================================================================
    # Generate LaTeX Table
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Mean attitude error for different loss functions across scenarios ($N=""" + str(n_runs) + r"""$ runs, FixedLagSmoother)}
\label{tab:loss_function_evaluation}
\begin{tabular}{@{}l""" + "c" * len(scenarios) + r"""@{}}
\toprule
\textbf{Loss Function} """

    for s_name, _ in scenarios:
        latex += f"& \\textbf{{{s_name}}} "
    latex += r"\\" + "\n" + r"\midrule" + "\n"

    for loss_name, _, _ in loss_functions:
        latex += f"{loss_name} "
        for s_name, _ in scenarios:
            vals = [v for v in results[s_name][loss_name] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                latex += f"& ${mean:.3f}^\\circ \\pm {std:.3f}^\\circ$ "
            else:
                latex += "& -- "
        latex += r"\\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(latex)

    # ==========================================================================
    # Generate Plot
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['C3', 'C0', 'C0', 'C1', 'C1', 'C2']
    hatches = ['', '', '//', '', '//', '']

    for ax_idx, (s_name, _) in enumerate(scenarios):
        ax = axes[ax_idx]

        names = [l[0] for l in loss_functions]
        x = np.arange(len(names))

        means = []
        stds = []
        for loss_name, _, _ in loss_functions:
            vals = [v for v in results[s_name][loss_name] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in names],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Error [deg]')
        ax.set_title(s_name)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('loss_function_evaluation.png', dpi=150, bbox_inches='tight')
    plt.savefig('loss_function_evaluation.pdf', dpi=150, bbox_inches='tight')
    print(f"\nSaved: loss_function_evaluation.png")

    # ==========================================================================
    # Save results to file
    # ==========================================================================
    with open('loss_function_evaluation.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: loss_function_evaluation.tex")


if __name__ == "__main__":
    main()
