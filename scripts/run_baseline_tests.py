"""
Run baseline tests for thesis Section 6.1.

Compares ESKF, iSAM2 (Fixed-Lag Smoother), and Redundant architecture
under two conditions:
1. Perfect measurements (zero noise) - implementation verification
2. Normal measurements (baseline noise) - nominal performance

Outputs:
- LaTeX table with steady-state results
- Convergence plots for both scenarios
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import EskfState, NominalState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml
from environment.environment import OrbitEnvironmentModel


def compute_attitude_error(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    dq = q_true @ q_est.conjugate()
    angle = 2 * np.arccos(np.clip(abs(dq.mu), 0, 1))
    return np.rad2deg(angle)


def run_all_estimators(sim, env, config_path, q_init, P0, smoother_lag=60.0):
    """Run all three estimators on the same simulation data."""

    results = {}

    # === ESKF ===
    print("  Running ESKF...")
    eskf = ESKF(P0=P0, config_path=config_path)
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    eskf_times, eskf_errors = [], []
    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        x_eskf = eskf.predict(x_eskf, omega, dt)

        if not np.any(np.isnan(sim.mag_meas[k])):
            B_n = env.get_B_eci(env.get_r_eci(jd), jd)
            try: x_eskf = eskf.update(x_eskf, sim.mag_meas[k], SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError: pass
        if not np.any(np.isnan(sim.sun_meas[k])):
            s_n = env.get_sun_eci(jd)
            try: x_eskf = eskf.update(x_eskf, sim.sun_meas[k], SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError: pass
        if not np.any(np.isnan(sim.st_meas[k])):
            try: x_eskf = eskf.update(x_eskf, Quaternion.from_array(sim.st_meas[k]), SensorType.STAR_TRACKER)
            except ValueError: pass

        q_true = Quaternion.from_array(sim.q_true[k])
        eskf_times.append(t)
        eskf_errors.append(compute_attitude_error(q_true, x_eskf.nom.ori))

    results['ESKF'] = {'times': np.array(eskf_times), 'errors': np.array(eskf_errors)}

    # === iSAM2 (Fixed-Lag Smoother) ===
    print("  Running iSAM2...")
    smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=smoother_lag, use_robust=True)
    smoother.initialize(sim.t[0], q_init, np.zeros(3))

    isam_times, isam_errors = [], []
    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        smoother.integrate_gyro(omega, dt)

        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim.st_meas[k]) if not np.any(np.isnan(sim.st_meas[k])) else None

        B_eci = env.get_B_eci(env.get_r_eci(jd), jd) if z_mag is not None else None
        s_eci = env.get_sun_eci(jd) if z_sun is not None else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            state = smoother.add_measurement(t, jd, z_mag, z_sun, z_st, B_eci, s_eci)
            if state is not None:
                q_true = Quaternion.from_array(sim.q_true[k])
                isam_times.append(t)
                isam_errors.append(compute_attitude_error(q_true, state.ori))

    results['iSAM2'] = {'times': np.array(isam_times), 'errors': np.array(isam_errors)}

    # === Redundant ===
    print("  Running Redundant...")
    redundant = RedundantEstimator(P0=P0, config_path=config_path, smoother_lag=smoother_lag)
    x_red = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    red_times, red_errors, red_primaries = [], [], []
    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim.st_meas[k]) if not np.any(np.isnan(sim.st_meas[k])) else None

        B_n = env.get_B_eci(env.get_r_eci(jd), jd) if z_mag is not None else None
        s_n = env.get_sun_eci(jd) if z_sun is not None else None

        x_red, smoother_state, disagreement, primary = redundant.step(
            x_red, t, jd, omega, dt, z_mag, z_sun, z_st, B_n, s_n
        )

        q_true = Quaternion.from_array(sim.q_true[k])
        primary_state = x_red.nom if primary == "ESKF" else smoother_state

        red_times.append(t)
        red_errors.append(compute_attitude_error(q_true, primary_state.ori))
        red_primaries.append(primary)

    results['Redundant'] = {
        'times': np.array(red_times),
        'errors': np.array(red_errors),
        'primaries': red_primaries,
        'stats': redundant.get_statistics()
    }

    return results


def compute_statistics(results, ss_start=30.0):
    """Compute steady-state statistics for each estimator."""
    stats = {}
    for name, data in results.items():
        times, errors = data['times'], data['errors']
        ss_mask = times > ss_start

        stats[name] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'ss_mean': np.mean(errors[ss_mask]) if np.any(ss_mask) else np.nan,
            'ss_std': np.std(errors[ss_mask]) if np.any(ss_mask) else np.nan,
            'max': np.max(errors),
            'final': errors[-1],
        }
    return stats


def generate_latex_table(perfect_stats, normal_stats):
    """Generate LaTeX table comparing results."""

    latex = r"""
\begin{table}[htbp]
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
        p = perfect_stats[name]
        n = normal_stats[name]
        latex += f"{name} & {p['ss_mean']:.6f} & {p['ss_std']:.6f} & {n['ss_mean']:.4f} & {n['ss_std']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def create_convergence_plot(results, title, filename):
    """Create convergence plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    colors = {'ESKF': '#1f77b4', 'iSAM2': '#ff7f0e', 'Redundant': '#2ca02c'}

    # Top: Log scale (full view)
    ax = axes[0]
    for name, data in results.items():
        ax.semilogy(data['times'], data['errors'], color=colors[name],
                   linewidth=0.8, alpha=0.8, label=name)
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-5, 100])

    # Bottom: Linear scale (steady-state zoom, t > 20s)
    ax = axes[1]
    for name, data in results.items():
        mask = data['times'] > 20
        if np.any(mask):
            ax.plot(data['times'][mask], data['errors'][mask], color=colors[name],
                   linewidth=0.8, alpha=0.8, label=name)
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Steady-State (t > 20s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 70)
    print("BASELINE PERFORMANCE TESTS")
    print("=" * 70)

    db_path = "simulations.db"
    config_baseline = "configs/config_baseline_short.yaml"
    config_perfect = "configs/config_perfect_measurements.yaml"

    db = SimulationDatabase(db_path)
    env = OrbitEnvironmentModel()

    # Initial conditions
    P0 = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6])

    all_stats = {}

    # === Test 1: Perfect Measurements ===
    print("\n" + "-" * 70)
    print("TEST 1: Perfect Measurements (Implementation Verification)")
    print("-" * 70)

    print("Generating simulation with perfect measurements...")
    config = load_yaml(config_perfect)
    generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=config_perfect)
    sim_cfg = SimulationConfig(
        T=config['time']['sim_T'], dt=config['time']['sim_dt'],
        start_jd=config['time']['start_jd'], run_name="perfect_measurements"
    )
    run_id = generator.run(sim_cfg)
    sim = db.load_run(run_id)

    # Perturbed initial attitude
    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.3, 0.3, 0.3]))
    q_init = q_true_0 @ perturb

    print("Running estimators with baseline config...")
    perfect_results = run_all_estimators(sim, env, config_baseline, q_init, P0)
    all_stats['perfect'] = compute_statistics(perfect_results)

    create_convergence_plot(perfect_results,
                           'Convergence: Perfect Measurements',
                           'baseline_perfect_convergence.png')

    # === Test 2: Normal Measurements ===
    print("\n" + "-" * 70)
    print("TEST 2: Normal Measurements (Baseline Performance)")
    print("-" * 70)

    print("Generating simulation with normal measurements...")
    config = load_yaml(config_baseline)
    generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=config_baseline)
    sim_cfg = SimulationConfig(
        T=config['time']['sim_T'], dt=config['time']['sim_dt'],
        start_jd=config['time']['start_jd'], run_name="baseline_normal"
    )
    run_id = generator.run(sim_cfg)
    sim = db.load_run(run_id)

    q_true_0 = Quaternion.from_array(sim.q_true[0])
    q_init = q_true_0 @ perturb

    print("Running estimators...")
    normal_results = run_all_estimators(sim, env, config_baseline, q_init, P0)
    all_stats['normal'] = compute_statistics(normal_results)

    create_convergence_plot(normal_results,
                           'Convergence: Normal Measurements',
                           'baseline_normal_convergence.png')

    # === Print Results ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nPerfect Measurements (Steady-State, t > 30s):")
    print(f"{'Estimator':<12} {'Mean [deg]':>14} {'Std [deg]':>14}")
    print("-" * 42)
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        s = all_stats['perfect'][name]
        print(f"{name:<12} {s['ss_mean']:>14.6f} {s['ss_std']:>14.6f}")

    print("\nNormal Measurements (Steady-State, t > 30s):")
    print(f"{'Estimator':<12} {'Mean [deg]':>14} {'Std [deg]':>14}")
    print("-" * 42)
    for name in ['ESKF', 'iSAM2', 'Redundant']:
        s = all_stats['normal'][name]
        print(f"{name:<12} {s['ss_mean']:>14.6f} {s['ss_std']:>14.6f}")

    # === Generate LaTeX Table ===
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    latex = generate_latex_table(all_stats['perfect'], all_stats['normal'])
    print(latex)

    # Save LaTeX to file
    with open('baseline_results_table.tex', 'w') as f:
        f.write(latex)
    print("Saved: baseline_results_table.tex")

    # Print redundant architecture stats
    print("\n" + "=" * 70)
    print("REDUNDANT ARCHITECTURE STATISTICS")
    print("=" * 70)
    for test_name in ['perfect', 'normal']:
        stats = normal_results['Redundant']['stats'] if test_name == 'normal' else perfect_results['Redundant']['stats']
        print(f"\n{test_name.capitalize()} Measurements:")
        print(f"  Switch events: {stats['total_switch_events']}")
        print(f"  Final primary: {stats['primary']}")

    return all_stats


if __name__ == "__main__":
    main()
