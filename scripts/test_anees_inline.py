#!/usr/bin/env python3
"""
Monte Carlo ANEES (Average Normalized Estimation Error Squared) Analysis.

Runs multiple simulations with standard measurement noise and standard noise
assumptions in the estimator to validate filter consistency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import chi2
from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
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


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> np.ndarray:
    """Compute attitude error in body frame (right-multiply convention)."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    if theta < 1e-8:
        return np.zeros(3)
    return theta * q_err.eta / np.sin(theta / 2.0)


def run_single_monte_carlo(sim_data, config_path: str, ss_start: float = 60.0):
    """
    Run ESKF on a single simulation and return NEES time series.

    Args:
        sim_data: Simulation data from database
        config_path: Path to config file
        ss_start: Start time for steady-state analysis

    Returns:
        times: Array of time values (steady-state only)
        nees_vals: Array of NEES values (steady-state only)
    """
    # Initial covariance
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initial state with perturbation
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    # Initialize ESKF
    eskf = ESKF(P0=P0, config_path=config_path, chi2_threshold=1e10)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times = []
    nees_vals = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]

        # Predict
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

        # Star tracker update
        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except:
                pass

        # Compute NEES (only in steady-state)
        if t >= ss_start:
            q_true = Quaternion.from_array(sim_data.q_true[k])
            att_err = compute_attitude_error(x_est.nom.ori, q_true)
            bias_err = x_est.nom.gyro_bias - sim_data.b_g_true[k]
            delta_x = np.concatenate([att_err, bias_err])

            try:
                nees = delta_x.T @ np.linalg.solve(x_est.err.cov, delta_x)
                times.append(t)
                nees_vals.append(nees)
            except:
                pass

    return np.array(times), np.array(nees_vals)


def run_monte_carlo(n_runs: int, config_path: str, db_path: str, ss_start: float = 60.0):
    """
    Run Monte Carlo NEES analysis.

    Args:
        n_runs: Number of Monte Carlo runs
        config_path: Path to config file
        db_path: Path to simulation database
        ss_start: Start time for steady-state analysis

    Returns:
        all_nees: List of NEES arrays from each run
        anees_per_timestep: ANEES averaged across runs at each timestep
        times: Time array (from first run)
        nees_matrix: Matrix of NEES values (n_runs x n_timesteps)
    """
    config = load_yaml(config_path)
    db = SimulationDatabase(db_path)

    all_nees = []
    times = None

    print(f"\nRunning {n_runs} Monte Carlo simulations...")
    print("-" * 50)

    for i in range(n_runs):
        # Generate new simulation data
        generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=config_path)
        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd'],
            run_name=f'mc_run_{i}'
        )
        run_id = generator.run(sim_cfg)
        sim_data = db.load_run(run_id)

        # Run ESKF
        t, nees = run_single_monte_carlo(sim_data, config_path, ss_start)
        all_nees.append(nees)

        if times is None:
            times = t

        # Progress
        mean_nees = np.mean(nees)
        print(f"  Run {i+1:3d}/{n_runs}: Mean NEES = {mean_nees:.2f}")

    # Compute ANEES (average across runs at each timestep)
    min_len = min(len(n) for n in all_nees)
    nees_matrix = np.array([n[:min_len] for n in all_nees])
    anees_per_timestep = np.mean(nees_matrix, axis=0)
    times = times[:min_len]

    return all_nees, anees_per_timestep, times, nees_matrix


def plot_anees_results(times, anees, nees_matrix, n_runs: int, save_path: str):
    """Plot ANEES results."""
    n = 6  # State dimension

    # Chi-squared bounds
    lower_bound = chi2.ppf(0.025, n)
    upper_bound = chi2.ppf(0.975, n)
    lower_25 = chi2.ppf(0.25, n)
    upper_75 = chi2.ppf(0.75, n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: ANEES over time
    ax = axes[0]
    ax.plot(times, anees, 'C0', linewidth=1.0, label='ANEES')
    ax.axhline(n, color='C1', linestyle='-', linewidth=2.0, alpha=0.7,
               label=f'Expected (n={n})')
    ax.axhline(lower_25, color='C2', linestyle='--', linewidth=1.5,
               label=f'25th percentile ({lower_25:.2f})')
    ax.axhline(upper_75, color='C3', linestyle='--', linewidth=1.5,
               label=f'75th percentile ({upper_75:.2f})')

    ax.set_ylabel('ANEES')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'Average NEES over {n_runs} Monte Carlo Runs')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([times[0], times[-1]])

    # Stats box
    mean_anees = np.mean(anees)
    std_anees = np.std(anees)
    textstr = f'Mean ANEES: {mean_anees:.2f}\nStd: {std_anees:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Bottom plot: Histogram of all NEES values
    ax = axes[1]
    all_nees_flat = nees_matrix.flatten()

    # Theoretical chi-squared PDF
    x = np.linspace(0.01, 25, 500)
    pdf = chi2.pdf(x, n)

    ax.hist(all_nees_flat, bins=50, density=True, alpha=0.7, color='C0',
            label=f'NEES histogram (N={len(all_nees_flat)})')
    ax.plot(x, pdf, 'C1-', linewidth=2.0, label=f'$\\chi^2({n})$ PDF')
    ax.axvline(n, color='C3', linestyle='--', linewidth=1.5, label=f'Expected mean ({n})')

    ax.set_xlabel('NEES')
    ax.set_ylabel('Density')
    ax.set_title('NEES Distribution vs Theoretical $\\chi^2$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 25])

    # Stats for histogram
    empirical_mean = np.mean(all_nees_flat)
    empirical_std = np.std(all_nees_flat)
    theoretical_std = np.sqrt(2 * n)

    textstr = f'Empirical mean: {empirical_mean:.2f} (expected: {n})\nEmpirical std: {empirical_std:.2f} (expected: {theoretical_std:.2f})'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


def compute_statistics(nees_matrix, n: int = 6):
    """Compute and print NEES statistics."""
    all_nees = nees_matrix.flatten()

    lower_25 = chi2.ppf(0.25, n)
    upper_75 = chi2.ppf(0.75, n)

    below = np.sum(all_nees < lower_25) / len(all_nees) * 100
    within = np.sum((all_nees >= lower_25) & (all_nees <= upper_75)) / len(all_nees) * 100
    above = np.sum(all_nees > upper_75) / len(all_nees) * 100

    return {
        'mean': np.mean(all_nees),
        'std': np.std(all_nees),
        'median': np.median(all_nees),
        'below_25': below,
        'within_25_75': within,
        'above_75': above,
    }


def main():
    config_path = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 20  # Number of Monte Carlo runs
    ss_start = 60.0  # Steady-state start time
    n = 6  # State dimension

    print("=" * 60)
    print("MONTE CARLO ANEES ANALYSIS")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Monte Carlo runs: {n_runs}")
    print(f"Steady-state start: {ss_start}s")
    print(f"State dimension: {n}")

    # Run Monte Carlo
    all_nees, anees, times, nees_matrix = run_monte_carlo(
        n_runs, config_path, db_path, ss_start
    )

    # Compute statistics
    stats = compute_statistics(nees_matrix, n)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nNEES Statistics (pooled across all runs):")
    print(f"  Mean:       {stats['mean']:.2f} (expected: {n})")
    print(f"  Std:        {stats['std']:.2f} (expected: {np.sqrt(2*n):.2f})")
    print(f"  Median:     {stats['median']:.2f} (expected: {chi2.ppf(0.5, n):.2f})")
    print(f"\nDistribution:")
    print(f"  Below 25th percentile:  {stats['below_25']:.1f}% (expected: 25%)")
    print(f"  Within 25-75th:         {stats['within_25_75']:.1f}% (expected: 50%)")
    print(f"  Above 75th percentile:  {stats['above_75']:.1f}% (expected: 25%)")

    # Plot results
    plot_anees_results(times, anees, nees_matrix, n_runs, 'nees_monte_carlo.png')

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
