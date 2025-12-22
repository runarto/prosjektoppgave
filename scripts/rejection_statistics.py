#!/usr/bin/env python3
"""
Analyze chi-squared rejection statistics for star tracker fault.
"""

import numpy as np
from typing import Tuple, Dict
import time

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from utilities.utils import load_yaml


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


def run_eskf_with_rejection_tracking(sim_data, config_path: str,
                                      fault_start: float, fault_end: float,
                                      use_chi2: bool) -> Dict:
    """Run ESKF and track rejection statistics."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    chi2_thresh = 7.81 if use_chi2 else 1e10

    eskf = ESKF(P0=P0, config_path=config_path,
                chi2_threshold=chi2_thresh,
                chi2_threshold_sun=chi2_thresh,
                chi2_threshold_star=chi2_thresh)

    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    # Track statistics
    stats = {
        'during_fault': {'total': 0, 'accepted': 0, 'rejected': 0},
        'before_fault': {'total': 0, 'accepted': 0, 'rejected': 0},
        'after_fault': {'total': 0, 'accepted': 0, 'rejected': 0},
    }

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]

        x_est = eskf.predict(x_est, omega, dt)

        # Magnetometer update
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.mag_meas[k],
                                   SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except:
                pass

        # Sun sensor update
        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k],
                                   SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except:
                pass

        # Star tracker update with rejection tracking
        if not np.any(np.isnan(sim_data.st_meas[k])):
            # Determine period
            if t < fault_start:
                period = 'before_fault'
            elif t < fault_end:
                period = 'during_fault'
            else:
                period = 'after_fault'

            stats[period]['total'] += 1

            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
                stats[period]['accepted'] += 1
            except ValueError:
                stats[period]['rejected'] += 1

    return stats


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
    n_runs = 20
    fault_start = 100.0
    fault_duration = 60.0
    fault_end = fault_start + fault_duration

    print("=" * 70)
    print("CHI-SQUARED REJECTION STATISTICS")
    print("=" * 70)
    print(f"Fault period: {fault_start}s - {fault_end}s")
    print(f"Monte Carlo runs: {n_runs}")
    print()

    config_path = create_config(base_config, 'configs/config_rejection_test.yaml')
    db = SimulationDatabase(db_path)

    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Collect statistics across runs
    all_stats = []
    error_magnitudes = []

    start_time = time.time()

    for i, run_id in enumerate(run_ids):
        error_mag = np.random.uniform(1.0, 10.0)
        error_magnitudes.append(error_mag)
        print(f"  Run {i+1}/{n_runs} (error={error_mag:.1f} deg)...", end=" ", flush=True)

        sim_data_base = db.load_run(run_id)
        sim_data = apply_star_tracker_fault(sim_data_base, fault_start, fault_duration, error_mag)

        stats = run_eskf_with_rejection_tracking(
            sim_data, config_path, fault_start, fault_end, use_chi2=True
        )
        all_stats.append(stats)

        rejection_rate = stats['during_fault']['rejected'] / stats['during_fault']['total'] * 100
        print(f"rejected {stats['during_fault']['rejected']}/{stats['during_fault']['total']} ({rejection_rate:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for period in ['before_fault', 'during_fault', 'after_fault']:
        totals = [s[period]['total'] for s in all_stats]
        accepted = [s[period]['accepted'] for s in all_stats]
        rejected = [s[period]['rejected'] for s in all_stats]

        if sum(totals) > 0:
            mean_total = np.mean(totals)
            mean_accepted = np.mean(accepted)
            mean_rejected = np.mean(rejected)
            rejection_rates = [r/t*100 if t > 0 else 0 for r, t in zip(rejected, totals)]
            mean_rate = np.mean(rejection_rates)
            std_rate = np.std(rejection_rates)

            period_name = period.replace('_', ' ').title()
            print(f"\n{period_name}:")
            print(f"  Total ST measurements: {mean_total:.1f} per run")
            print(f"  Accepted: {mean_accepted:.1f} ({100-mean_rate:.1f}%)")
            print(f"  Rejected: {mean_rejected:.1f} ({mean_rate:.1f}% ± {std_rate:.1f}%)")

    # Analysis by error magnitude
    print("\n" + "=" * 70)
    print("REJECTION RATE VS ERROR MAGNITUDE")
    print("=" * 70)

    # Sort by error magnitude
    sorted_data = sorted(zip(error_magnitudes, all_stats), key=lambda x: x[0])

    print(f"{'Error [deg]':>12} {'Rejected':>10} {'Total':>8} {'Rate':>10}")
    print("-" * 45)
    for err_mag, stats in sorted_data:
        rej = stats['during_fault']['rejected']
        tot = stats['during_fault']['total']
        rate = rej / tot * 100 if tot > 0 else 0
        print(f"{err_mag:>12.1f} {rej:>10} {tot:>8} {rate:>9.1f}%")

    # LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)

    # During fault statistics
    during_totals = [s['during_fault']['total'] for s in all_stats]
    during_rejected = [s['during_fault']['rejected'] for s in all_stats]
    during_rates = [r/t*100 if t > 0 else 0 for r, t in zip(during_rejected, during_totals)]

    before_totals = [s['before_fault']['total'] for s in all_stats]
    before_rejected = [s['before_fault']['rejected'] for s in all_stats]
    before_rates = [r/t*100 if t > 0 else 0 for r, t in zip(before_rejected, before_totals)]

    after_totals = [s['after_fault']['total'] for s in all_stats]
    after_rejected = [s['after_fault']['rejected'] for s in all_stats]
    after_rates = [r/t*100 if t > 0 else 0 for r, t in zip(after_rejected, after_totals)]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Chi-squared rejection statistics for star tracker measurements ($N=""" + str(n_runs) + r"""$ runs)}
\label{tab:rejection_stats}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Period} & \textbf{Total Measurements} & \textbf{Rejected} & \textbf{Rejection Rate} \\
\midrule
"""
    latex += f"Before fault & {np.mean(before_totals):.0f} & {np.mean(before_rejected):.1f} & ${np.mean(before_rates):.1f}^\\circ \\pm {np.std(before_rates):.1f}^\\circ$ \\\\\n"
    latex += f"During fault & {np.mean(during_totals):.0f} & {np.mean(during_rejected):.1f} & ${np.mean(during_rates):.1f}\\% \\pm {np.std(during_rates):.1f}\\%$ \\\\\n"
    latex += f"After fault & {np.mean(after_totals):.0f} & {np.mean(after_rejected):.1f} & ${np.mean(after_rates):.1f}\\% \\pm {np.std(after_rates):.1f}\\%$ \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(latex)


if __name__ == "__main__":
    main()
