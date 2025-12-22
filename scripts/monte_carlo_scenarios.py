#!/usr/bin/env python3
"""
Monte Carlo evaluation of attitude estimators across multiple scenarios.

Scenarios:
1. Baseline (normal measurements)
2. Eclipse (no sun sensor)
3. Gyro drift (unknown initial bias)
4. Magnetometer bias (constant offset)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
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
    short_name: str
    description: str
    eclipse: bool = False
    mag_bias: Optional[np.ndarray] = None
    true_gyro_bias: Optional[np.ndarray] = None


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


def run_eskf(sim_data, config_path: str, ss_start: float) -> Tuple[float, float]:
    """Run ESKF and return steady-state mean and std error."""
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

    ss_errors = []

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

        if t >= ss_start:
            ss_errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.mean(ss_errors), np.std(ss_errors)


def run_isam2(sim_data, config_path: str, ss_start: float) -> Tuple[float, float]:
    """Run iSAM2 and return steady-state mean and std error."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(config_path=config_path, use_isam2=True, use_rk4=True)

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    ss_errors = []
    for i, t in enumerate(times):
        if t >= ss_start:
            idx = np.argmin(np.abs(sim_data.t - t))
            q_true = Quaternion.from_array(sim_data.q_true[idx])
            ss_errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.mean(ss_errors), np.std(ss_errors)


def run_redundant(sim_data, config_path: str, ss_start: float) -> Tuple[float, float]:
    """Run Redundant estimator and return steady-state mean and std error."""
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

    ss_errors = []

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

        x_est, _, _, _ = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        if t >= ss_start:
            ss_errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.mean(ss_errors), np.std(ss_errors)


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


def run_monte_carlo_scenario(scenario: ScenarioConfig, n_runs: int,
                              config_path: str, db: SimulationDatabase,
                              ss_start: float = 100.0) -> Dict:
    """Run Monte Carlo for a single scenario."""
    results = {
        'ESKF': {'means': [], 'stds': []},
        'iSAM2': {'means': [], 'stds': []},
        'Redundant': {'means': [], 'stds': []},
    }

    # Get available run IDs (use mc_run_* from previous Monte Carlo)
    import sqlite3
    conn = sqlite3.connect(db.path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs WHERE name LIKE 'mc_run_%' ORDER BY id LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(run_ids) < n_runs:
        print(f"  Warning: Only {len(run_ids)} runs available, requested {n_runs}")
        n_runs = len(run_ids)

    for i, run_id in enumerate(run_ids):
        print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)

        # Load and modify data
        sim_data_base = db.load_run(run_id)
        sim_data = modify_sim_data(sim_data_base, scenario)

        # Run ESKF
        try:
            mean, std = run_eskf(sim_data, config_path, ss_start)
            results['ESKF']['means'].append(mean)
            results['ESKF']['stds'].append(std)
        except Exception as e:
            print(f"ESKF failed: {e}")

        # Run iSAM2
        try:
            mean, std = run_isam2(sim_data, config_path, ss_start)
            results['iSAM2']['means'].append(mean)
            results['iSAM2']['stds'].append(std)
        except Exception as e:
            print(f"iSAM2 failed: {e}")

        # Run Redundant
        try:
            mean, std = run_redundant(sim_data, config_path, ss_start)
            results['Redundant']['means'].append(mean)
            results['Redundant']['stds'].append(std)
        except Exception as e:
            print(f"Redundant failed: {e}")

        print("done")

    # Compute aggregate statistics
    agg_results = {}
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if results[name]['means']:
            agg_results[name] = {
                'mean': np.mean(results[name]['means']),
                'std': np.std(results[name]['means']),
                'mean_std': np.mean(results[name]['stds']),
            }

    return agg_results


def generate_latex_table(all_results: Dict[str, Dict], scenarios: List[ScenarioConfig]) -> str:
    """Generate LaTeX table for all scenarios."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo Performance Comparison Across Scenarios (N runs each)}
\label{tab:mc_scenario_comparison}
\begin{tabular}{llcc}
\toprule
Scenario & Estimator & Mean Error [deg] & Std [deg] \\
\midrule
"""
    for scenario in scenarios:
        results = all_results[scenario.name]
        first = True
        for name in ['ESKF', 'iSAM2', 'Redundant']:
            if name not in results:
                continue
            r = results[name]
            scenario_col = scenario.short_name if first else ""
            first = False
            table += f"{scenario_col} & {name} & {r['mean']:.4f} $\\pm$ {r['std']:.4f} & {r['mean_std']:.4f} \\\\\n"
        table += r"\midrule" + "\n"

    table = table.rsplit(r"\midrule", 1)[0]
    table += r"""\bottomrule
\end{tabular}
\end{table}"""
    return table


def plot_monte_carlo_results(all_results: Dict[str, Dict], scenarios: List[ScenarioConfig],
                              save_path: str):
    """Create bar plot comparing estimators across scenarios."""
    fig, ax = plt.subplots(figsize=(14, 8))

    scenario_names = [s.short_name for s in scenarios]
    x = np.arange(len(scenario_names))
    width = 0.25

    colors = {'ESKF': 'C0', 'iSAM2': 'C1', 'Redundant': 'C2'}

    for i, name in enumerate(['ESKF', 'iSAM2', 'Redundant']):
        means = []
        stds = []
        for scenario in scenarios:
            if name in all_results[scenario.name]:
                means.append(all_results[scenario.name][name]['mean'])
                stds.append(all_results[scenario.name][name]['std'])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=name,
                     color=colors[name], capsize=5, alpha=0.8)

    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_title('Monte Carlo Comparison Across Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 10  # Number of Monte Carlo runs per scenario
    ss_start = 100.0

    print("=" * 70)
    print("MONTE CARLO SCENARIO EVALUATION")
    print("=" * 70)
    print(f"Runs per scenario: {n_runs}")
    print(f"Steady-state start: {ss_start}s")

    config_path = create_config(base_config, 'configs/config_mc_scenario.yaml')
    db = SimulationDatabase(db_path)

    # Define scenarios
    scenarios = [
        ScenarioConfig(
            name="Baseline",
            short_name="Baseline",
            description="Normal operation with all sensors",
        ),
        ScenarioConfig(
            name="Eclipse",
            short_name="Eclipse",
            description="No sun sensor measurements",
            eclipse=True
        ),
        ScenarioConfig(
            name="Gyro Drift",
            short_name="Gyro Drift",
            description="Unknown 1 deg/h gyro bias per axis",
            true_gyro_bias=np.deg2rad(np.array([1.0, 1.0, 1.0])) / 3600
        ),
        ScenarioConfig(
            name="Mag Bias",
            short_name="Mag Bias",
            description="5% magnetometer bias on X-axis",
            mag_bias=np.array([0.05, 0.0, 0.0])
        ),
    ]

    all_results = {}
    start_time = time.time()

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.name}")
        print(f"{'='*70}")
        print(f"Description: {scenario.description}")

        results = run_monte_carlo_scenario(
            scenario, n_runs, config_path, db, ss_start
        )
        all_results[scenario.name] = results

        # Print results for this scenario
        print(f"\n  Results for {scenario.name}:")
        for name in ['ESKF', 'iSAM2', 'Redundant']:
            if name in results:
                r = results[name]
                print(f"    {name:12s}: {r['mean']:.4f} ± {r['std']:.4f} deg")

    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed/60:.1f} minutes")

    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    # LaTeX table
    latex_table = generate_latex_table(all_results, scenarios)
    print("\nLaTeX Table:")
    print(latex_table)

    with open('mc_scenario_results.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved: mc_scenario_results.tex")

    # Bar plot
    plot_monte_carlo_results(all_results, scenarios, 'mc_scenario_comparison.png')

    print("\n" + "=" * 70)
    print("MONTE CARLO EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
