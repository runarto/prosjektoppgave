#!/usr/bin/env python3
"""
Generate baseline performance comparison table and convergence plots.

Compares ESKF, iSAM2, and Redundant estimator for:
1. Perfect measurements (no measurement noise)
2. Normal measurements (standard noise)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
    """Mock environment that uses pre-computed values from simulation data."""

    def __init__(self, sim_data):
        self.sim_data = sim_data
        self._jd_to_idx = {jd: i for i, jd in enumerate(sim_data.jd)}

    def _find_idx(self, jd):
        """Find closest index for given Julian date."""
        idx = np.argmin(np.abs(self.sim_data.jd - jd))
        return idx

    def get_r_eci(self, jd):
        """Return dummy position (not used for factors)."""
        return np.array([7000e3, 0, 0])  # ~7000 km orbit

    def get_B_eci(self, r_eci, jd):
        """Return magnetic field from simulation data."""
        idx = self._find_idx(jd)
        return self.sim_data.b_eci[idx]

    def get_sun_eci(self, jd):
        """Return sun direction from simulation data."""
        idx = self._find_idx(jd)
        return self.sim_data.s_eci[idx]


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error magnitude in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def run_eskf(sim_data, config_path: str):
    """Run ESKF on simulation data."""
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

    times = []
    errors = []

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


def run_isam2(sim_data, config_path: str):
    """Run iSAM2 on simulation data."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(
        config_path=config_path,
        use_isam2=True,
        use_rk4=True,
    )

    kf_times, kf_states = fgo.process_simulation(sim_data, env)

    # Interpolate to full rate
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    # Compute errors
    errors = []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.array(times), np.array(errors)


def run_redundant(sim_data, config_path: str):
    """Run Redundant estimator on simulation data."""
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

    times = []
    errors = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        # Get measurements (None if NaN)
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_n = sim_data.b_eci[k] if z_mag is not None else None
        s_n = sim_data.s_eci[k] if z_sun is not None else None

        # Use the step method
        x_est, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_est,
            t=t,
            jd=jd,
            omega_meas=omega,
            dt=dt,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=B_n,
            s_n=s_n,
        )

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors)


def create_config(base_config_path: str, output_path: str, noise_scale: float = 1.0):
    """Create a config with specified noise scale."""
    config = load_yaml(base_config_path)

    config['sensors']['mag']['scaling']['noise_scale'] = noise_scale
    config['sensors']['sun']['scaling']['noise_scale'] = noise_scale
    config['sensors']['star']['scaling']['noise_scale'] = noise_scale

    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def plot_convergence(times_dict, errors_dict, title, save_path):
    """Plot convergence of all estimators."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'ESKF': 'C0', 'iSAM2': 'C1', 'Redundant': 'C2'}
    styles = {'ESKF': '-', 'iSAM2': '-', 'Redundant': '--'}

    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if name in times_dict:
            ax.semilogy(times_dict[name], errors_dict[name],
                       color=colors[name], linestyle=styles[name],
                       linewidth=1.5, label=name)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Set consistent x limits
    all_times = [t for t in times_dict.values()]
    ax.set_xlim([min(t[0] for t in all_times), max(t[-1] for t in all_times)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_latex_table(stats_perfect, stats_normal):
    """Generate LaTeX table."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Baseline Performance Comparison: Steady-State Attitude Error}
\label{tab:baseline_performance}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Perfect Measurements} & \multicolumn{2}{c}{Normal Measurements} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Estimator & Mean [deg] & Std [deg] & Mean [deg] & Std [deg] \\
\midrule
"""
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        if name not in stats_perfect:
            continue
        p_mean = stats_perfect[name]['mean']
        p_std = stats_perfect[name]['std']
        n_mean = stats_normal[name]['mean']
        n_std = stats_normal[name]['std']

        # Format numbers appropriately
        if p_mean < 0.001:
            p_mean_str = f"{p_mean:.6f}"
            p_std_str = f"{p_std:.6f}"
        elif p_mean < 0.01:
            p_mean_str = f"{p_mean:.5f}"
            p_std_str = f"{p_std:.5f}"
        else:
            p_mean_str = f"{p_mean:.4f}"
            p_std_str = f"{p_std:.4f}"

        n_mean_str = f"{n_mean:.4f}"
        n_std_str = f"{n_std:.4f}"

        table += f"{name} & {p_mean_str} & {p_std_str} & {n_mean_str} & {n_std_str} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}"""

    return table


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    ss_start = 100.0  # Steady-state start time

    print("=" * 60)
    print("BASELINE PERFORMANCE COMPARISON")
    print("=" * 60)

    config_path = create_config(base_config, 'configs/config_temp.yaml', noise_scale=1.0)

    db = SimulationDatabase(db_path)

    print("\n--- Loading Perfect Measurements (run_id=202) ---")
    sim_data_perfect = db.load_run(202)

    print("\n--- Loading Normal Measurements (run_id=237) ---")
    sim_data_normal = db.load_run(237)

    # Results storage
    times_p = {}
    errors_p = {}
    times_n = {}
    errors_n = {}

    # Run ESKF
    print("\n--- Running ESKF ---")
    print("  Perfect measurements...")
    times_p['ESKF'], errors_p['ESKF'] = run_eskf(sim_data_perfect, config_path)
    print("  Normal measurements...")
    times_n['ESKF'], errors_n['ESKF'] = run_eskf(sim_data_normal, config_path)

    # Run iSAM2
    print("\n--- Running iSAM2 ---")
    try:
        print("  Perfect measurements...")
        times_p['iSAM2'], errors_p['iSAM2'] = run_isam2(sim_data_perfect, config_path)
        print("  Normal measurements...")
        times_n['iSAM2'], errors_n['iSAM2'] = run_isam2(sim_data_normal, config_path)
    except Exception as e:
        print(f"  iSAM2 failed: {e}")

    # Run Redundant
    print("\n--- Running Redundant ---")
    try:
        print("  Perfect measurements...")
        times_p['Redundant'], errors_p['Redundant'] = run_redundant(sim_data_perfect, config_path)
        print("  Normal measurements...")
        times_n['Redundant'], errors_n['Redundant'] = run_redundant(sim_data_normal, config_path)
    except Exception as e:
        print(f"  Redundant failed: {e}")

    # Compute steady-state statistics
    stats_perfect = {}
    stats_normal = {}

    for name in times_p.keys():
        ss_mask = times_p[name] >= ss_start
        stats_perfect[name] = {
            'mean': np.mean(errors_p[name][ss_mask]),
            'std': np.std(errors_p[name][ss_mask])
        }

    for name in times_n.keys():
        ss_mask = times_n[name] >= ss_start
        stats_normal[name] = {
            'mean': np.mean(errors_n[name][ss_mask]),
            'std': np.std(errors_n[name][ss_mask])
        }

    # Print results
    print("\n" + "=" * 60)
    print("STEADY-STATE RESULTS (t >= {}s)".format(ss_start))
    print("=" * 60)

    print("\nPerfect Measurements:")
    for name, s in stats_perfect.items():
        print(f"  {name:12s}: Mean = {s['mean']:.6f} deg, Std = {s['std']:.6f} deg")

    print("\nNormal Measurements:")
    for name, s in stats_normal.items():
        print(f"  {name:12s}: Mean = {s['mean']:.4f} deg, Std = {s['std']:.4f} deg")

    # Generate LaTeX table
    latex_table = generate_latex_table(stats_perfect, stats_normal)
    print("\n" + "=" * 60)
    print("LaTeX TABLE")
    print("=" * 60)
    print(latex_table)

    with open('baseline_results_table.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved: baseline_results_table.tex")

    # Generate convergence plots
    plot_convergence(times_p, errors_p,
                    'Attitude Error Convergence (Perfect Measurements)',
                    'convergence_perfect.png')

    plot_convergence(times_n, errors_n,
                    'Attitude Error Convergence (Normal Measurements)',
                    'convergence_normal.png')

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
