#!/usr/bin/env python3
"""
Compare ESKF, iSAM2, and Redundant estimators on degraded scenarios.

Scenarios:
1. Eclipse - Sun sensor unavailable throughout simulation
2. Gyro drift - Initial 1°/h per axis bias, unknown to estimator
3. Magnetometer bias - 5% hard-iron bias on X-axis
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


def apply_eclipse(sim_data):
    """Eclipse - Sun sensor unavailable throughout simulation."""
    modified = apply_baseline(sim_data)

    # Set all sun measurements to NaN
    for k in range(len(modified.t)):
        modified.sun_meas[k] = np.array([np.nan, np.nan, np.nan])

    return modified


def apply_gyro_drift(sim_data, drift_deg_per_hour: float = 1.0):
    """Gyro drift - Initial bias unknown to estimator.

    The true gyro measurements already include the bias from simulation.
    We modify the gyro measurements to add additional drift.
    """
    modified = apply_baseline(sim_data)

    # Add 1 deg/h = 1/3600 deg/s per axis
    drift_rad_per_s = np.deg2rad(drift_deg_per_hour / 3600.0)
    bias_offset = np.array([drift_rad_per_s, drift_rad_per_s, drift_rad_per_s])

    # Add bias to gyro measurements
    for k in range(len(modified.t)):
        modified.omega_meas[k] = modified.omega_meas[k] + bias_offset

    return modified


def apply_mag_bias(sim_data, bias_fraction: float = 0.05):
    """Magnetometer bias - 5% hard-iron bias on X-axis."""
    modified = apply_baseline(sim_data)

    # Apply 5% bias on X-axis
    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.mag_meas[k])):
            mag_norm = np.linalg.norm(modified.mag_meas[k])
            modified.mag_meas[k][0] += bias_fraction * mag_norm

    return modified


# =============================================================================
# Estimator Runners
# =============================================================================

def run_eskf(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run ESKF with chi-squared gating."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    eskf = ESKF(P0=P0, config_path=config_path,
                chi2_threshold=7.81,
                chi2_threshold_sun=7.81,
                chi2_threshold_star=7.81)

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


def run_isam2(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run iSAM2 with Cauchy k=0.1."""
    env = MockEnvironment(sim_data)

    fgo = KeyframeFGO(
        config_path=config_path,
        use_isam2=True,
        use_rk4=True,
        use_robust=True,
        robust_kernel="huber",
        robust_param=0.1,
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
    """Run Redundant estimator (ESKF + smoother with switching)."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        use_robust=True,
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


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 10

    # Scenarios
    scenarios = [
        ("Baseline", apply_baseline),
        ("Eclipse", apply_eclipse),
        ("Gyro Drift", apply_gyro_drift),
        ("Mag Bias", apply_mag_bias),
    ]

    # Estimators
    estimators = [
        ("ESKF", run_eskf),
        ("iSAM2", run_isam2),
        ("Redundant", run_redundant),
    ]

    print("=" * 70)
    print("DEGRADED SCENARIOS - ESKF vs iSAM2 vs Redundant")
    print("=" * 70)
    print(f"Monte Carlo runs: {n_runs}")
    print(f"ESKF: chi-squared gating")
    print(f"iSAM2: Cauchy k=0.1")
    print(f"Redundant: ESKF + smoother with switching")
    print()

    config_path = create_config(base_config, 'configs/config_degraded_comparison.yaml')
    db = SimulationDatabase(db_path)

    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Results storage: results[scenario][estimator] = list of mean errors
    results = {s_name: {e_name: [] for e_name, _ in estimators}
               for s_name, _ in scenarios}

    start_time = time.time()

    for scenario_name, scenario_fn in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        for i, run_id in enumerate(run_ids):
            print(f"  Run {i+1}/{n_runs}:", end=" ", flush=True)

            sim_data_base = db.load_run(run_id)
            sim_data = scenario_fn(sim_data_base)

            for est_name, est_fn in estimators:
                try:
                    times, errors = est_fn(sim_data, config_path)

                    # Use steady-state (last 50% of simulation)
                    mask = times >= times[-1] * 0.5
                    mean_error = np.mean(errors[mask])
                    results[scenario_name][est_name].append(mean_error)
                    print(f"{est_name}={mean_error:.3f}°", end=" ", flush=True)
                except Exception as e:
                    print(f"{est_name}=FAIL", end=" ", flush=True)
                    results[scenario_name][est_name].append(np.nan)

            print()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - Mean Attitude Error [deg]")
    print("=" * 80)

    header = f"{'Scenario':<15}"
    for est_name, _ in estimators:
        header += f" {est_name:>20}"
    print(header)
    print("-" * 80)

    for scenario_name, _ in scenarios:
        row = f"{scenario_name:<15}"
        for est_name, _ in estimators:
            vals = [v for v in results[scenario_name][est_name] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row += f" {mean:>8.3f} ± {std:<8.3f}"
            else:
                row += f" {'FAILED':>20}"
        print(row)

    # ==========================================================================
    # LaTeX Table
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of estimators on degraded scenarios ($N=""" + str(n_runs) + r"""$ runs)}
\label{tab:degraded_comparison}
\begin{tabular}{@{}l""" + "c" * len(estimators) + r"""@{}}
\toprule
\textbf{Scenario} """

    for est_name, _ in estimators:
        latex += f"& \\textbf{{{est_name}}} "
    latex += r"\\" + "\n" + r"\midrule" + "\n"

    for scenario_name, _ in scenarios:
        latex += f"{scenario_name} "
        for est_name, _ in estimators:
            vals = [v for v in results[scenario_name][est_name] if not np.isnan(v)]
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
    # Plot
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = [s[0] for s in scenarios]
    est_names = [e[0] for e in estimators]
    x = np.arange(len(scenario_names))
    width = 0.25

    colors = ['C0', 'C1', 'C2']

    for i, (est_name, _) in enumerate(estimators):
        means = []
        stds = []
        for scenario_name, _ in scenarios:
            vals = [v for v in results[scenario_name][est_name] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=est_name,
                     color=colors[i], capsize=3, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_title('Estimator Comparison on Degraded Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('degraded_scenarios_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('degraded_scenarios_comparison.pdf', dpi=150, bbox_inches='tight')
    print(f"\nSaved: degraded_scenarios_comparison.png")


if __name__ == "__main__":
    main()
