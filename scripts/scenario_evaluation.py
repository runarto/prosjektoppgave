#!/usr/bin/env python3
"""
Scenario-based evaluation of attitude estimators.

Tests ESKF, iSAM2, and Redundant estimator under:
1. Eclipse - no sun sensor measurements
2. Non-zero initial gyro bias (simulating long-term drift)
3. Magnetometer constant bias
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy

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


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    description: str
    # Data modifications
    eclipse: bool = False  # Remove sun measurements
    mag_bias: Optional[np.ndarray] = None  # Constant magnetometer bias
    # Initial condition modifications
    true_gyro_bias: Optional[np.ndarray] = None  # True gyro bias (estimator assumes zero)


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


def modify_sim_data(sim_data, scenario: ScenarioConfig):
    """Apply scenario modifications to simulation data."""
    # Create a copy to avoid modifying original
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

    # Eclipse: remove sun measurements
    if scenario.eclipse:
        modified.sun_meas = np.full_like(modified.sun_meas, np.nan)

    # Magnetometer bias: add constant offset to measurements
    if scenario.mag_bias is not None:
        for k in range(len(modified.mag_meas)):
            if not np.any(np.isnan(modified.mag_meas[k])):
                # Add bias and renormalize (since mag is unit vector)
                biased = modified.mag_meas[k] + scenario.mag_bias
                modified.mag_meas[k] = biased / np.linalg.norm(biased)

    # Gyro bias: modify the true bias (affects omega_meas interpretation)
    if scenario.true_gyro_bias is not None:
        # The gyro measurements already include the true bias from simulation
        # We're simulating that the TRUE bias is different from what estimator expects
        # So we add additional bias to measurements
        modified.omega_meas = modified.omega_meas + scenario.true_gyro_bias
        # Update true bias record
        modified.b_g_true = modified.b_g_true + scenario.true_gyro_bias

    return modified


def run_eskf(sim_data, config_path: str, scenario: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ESKF and return times, attitude errors, and bias errors."""
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

    times, att_errors, bias_errors = [], [], []

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
        att_errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))
        # Bias error in deg/h
        bias_err = np.linalg.norm(x_est.nom.gyro_bias - sim_data.b_g_true[k])
        bias_errors.append(np.rad2deg(bias_err) * 3600)

    return np.array(times), np.array(att_errors), np.array(bias_errors)


def run_isam2(sim_data, config_path: str, scenario: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run iSAM2 and return times, attitude errors, and bias errors."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(
        config_path=config_path,
        use_isam2=True,
        use_rk4=True,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    att_errors, bias_errors = [], []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        att_errors.append(compute_attitude_error_deg(states[i].ori, q_true))
        bias_err = np.linalg.norm(states[i].gyro_bias - sim_data.b_g_true[idx])
        bias_errors.append(np.rad2deg(bias_err) * 3600)

    return np.array(times), np.array(att_errors), np.array(bias_errors)


def run_redundant(sim_data, config_path: str, scenario: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Redundant estimator and return times, attitude errors, and bias errors."""
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

    times, att_errors, bias_errors = [], [], []

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

        x_est, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        att_errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))
        bias_err = np.linalg.norm(x_est.nom.gyro_bias - sim_data.b_g_true[k])
        bias_errors.append(np.rad2deg(bias_err) * 3600)

    return np.array(times), np.array(att_errors), np.array(bias_errors)


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


def plot_scenario_results(results: Dict, scenario: ScenarioConfig, save_prefix: str):
    """Plot attitude and bias errors for a scenario."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = {'ESKF': 'C0', 'iSAM2': 'C1', 'Redundant': 'C2'}
    styles = {'ESKF': '-', 'iSAM2': '-', 'Redundant': '--'}

    # Attitude error plot
    ax = axes[0]
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if name in results:
            times, att_err, _ = results[name]
            ax.semilogy(times, att_err, color=colors[name], linestyle=styles[name],
                       linewidth=1.5, label=name)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(f'{scenario.name}: Attitude Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Bias error plot
    ax = axes[1]
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if name in results:
            times, _, bias_err = results[name]
            ax.semilogy(times, bias_err, color=colors[name], linestyle=styles[name],
                       linewidth=1.5, label=name)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Gyro Bias Error [deg/h]')
    ax.set_title(f'{scenario.name}: Gyro Bias Estimation Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_prefix}.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}.png")
    plt.close()


def compute_stats(results: Dict, ss_start: float = 100.0) -> Dict:
    """Compute steady-state statistics for all estimators."""
    stats = {}
    for name, (times, att_err, bias_err) in results.items():
        ss_mask = times >= ss_start
        stats[name] = {
            'att_mean': np.mean(att_err[ss_mask]),
            'att_std': np.std(att_err[ss_mask]),
            'bias_mean': np.mean(bias_err[ss_mask]),
            'bias_std': np.std(bias_err[ss_mask]),
        }
    return stats


def print_scenario_stats(scenario: ScenarioConfig, stats: Dict):
    """Print statistics for a scenario."""
    print(f"\n{scenario.name}")
    print(f"  {scenario.description}")
    print("-" * 70)
    print(f"{'Estimator':<12} {'Att Mean [deg]':>14} {'Att Std':>12} {'Bias Mean [°/h]':>16} {'Bias Std':>12}")
    print("-" * 70)
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if name in stats:
            s = stats[name]
            print(f"{name:<12} {s['att_mean']:>14.4f} {s['att_std']:>12.4f} {s['bias_mean']:>16.2f} {s['bias_std']:>12.2f}")


def generate_combined_table(all_stats: Dict[str, Dict], scenarios: List[ScenarioConfig]) -> str:
    """Generate LaTeX table for all scenarios."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Estimator Performance Across Challenging Scenarios}
\label{tab:scenario_comparison}
\begin{tabular}{llcccc}
\toprule
Scenario & Estimator & \multicolumn{2}{c}{Attitude Error [deg]} & \multicolumn{2}{c}{Bias Error [deg/h]} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & Mean & Std & Mean & Std \\
\midrule
"""
    for scenario in scenarios:
        stats = all_stats[scenario.name]
        first = True
        for name in ['ESKF', 'iSAM2', 'Redundant']:
            if name not in stats:
                continue
            s = stats[name]
            scenario_col = scenario.name if first else ""
            first = False
            table += f"{scenario_col} & {name} & {s['att_mean']:.4f} & {s['att_std']:.4f} & {s['bias_mean']:.2f} & {s['bias_std']:.2f} \\\\\n"
        table += r"\midrule" + "\n"

    # Remove last midrule and add bottomrule
    table = table.rsplit(r"\midrule", 1)[0]
    table += r"""\bottomrule
\end{tabular}
\end{table}"""
    return table


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    ss_start = 100.0

    print("=" * 70)
    print("SCENARIO-BASED EVALUATION")
    print("=" * 70)

    config_path = create_config(base_config, 'configs/config_scenario.yaml')
    db = SimulationDatabase(db_path)

    # Load base simulation data (normal measurements)
    print("\nLoading simulation data (run_id=237)...")
    sim_data_base = db.load_run(237)

    # Define scenarios
    scenarios = [
        ScenarioConfig(
            name="Eclipse",
            description="No sun sensor measurements available",
            eclipse=True
        ),
        ScenarioConfig(
            name="Gyro Drift",
            description="Initial gyro bias of 1 deg/h per axis (unknown to estimator)",
            true_gyro_bias=np.deg2rad(np.array([1.0, 1.0, 1.0])) / 3600  # 1 deg/h per axis
        ),
        ScenarioConfig(
            name="Mag Bias",
            description="Magnetometer has 5% constant bias on X-axis",
            mag_bias=np.array([0.05, 0.0, 0.0])  # 5% bias on X
        ),
    ]

    all_stats = {}
    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.name}")
        print(f"{'='*70}")
        print(f"Description: {scenario.description}")

        # Modify simulation data for this scenario
        sim_data = modify_sim_data(sim_data_base, scenario)

        results = {}

        # Run ESKF
        print("\n  Running ESKF...")
        try:
            results['ESKF'] = run_eskf(sim_data, config_path, scenario)
        except Exception as e:
            print(f"    Failed: {e}")

        # Run iSAM2
        print("  Running iSAM2...")
        try:
            results['iSAM2'] = run_isam2(sim_data, config_path, scenario)
        except Exception as e:
            print(f"    Failed: {e}")

        # Run Redundant
        print("  Running Redundant...")
        try:
            results['Redundant'] = run_redundant(sim_data, config_path, scenario)
        except Exception as e:
            print(f"    Failed: {e}")

        # Compute and print stats
        stats = compute_stats(results, ss_start)
        print_scenario_stats(scenario, stats)

        all_stats[scenario.name] = stats
        all_results[scenario.name] = results

        # Plot results
        save_name = f"scenario_{scenario.name.lower().replace(' ', '_')}"
        plot_scenario_results(results, scenario, save_name)

    # Generate combined LaTeX table
    print("\n" + "=" * 70)
    print("COMBINED LaTeX TABLE")
    print("=" * 70)
    latex_table = generate_combined_table(all_stats, scenarios)
    print(latex_table)

    with open('scenario_results_table.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved: scenario_results_table.tex")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
