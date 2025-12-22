#!/usr/bin/env python3
"""
Euler Angle Analysis: True angles and estimation errors.

Creates publication-quality plots with:
1. True roll, pitch, yaw angles over time
2. Roll, pitch, yaw estimation errors for ESKF, iSAM2, and Redundant estimator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sqlite3
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import SimulationDatabase
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml

# =============================================================================
# Publication-quality plot settings - LARGE FONTS for paper printing
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['legend.fontsize'] = 14
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['figure.titlesize'] = 22
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


def quat_to_euler_deg(q: Quaternion) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    euler_rad = q.as_euler()
    return np.rad2deg(euler_rad)


def compute_euler_error(q_est: Quaternion, q_true: Quaternion) -> np.ndarray:
    """
    Compute roll, pitch, yaw errors in degrees.
    Returns the difference in Euler angles, wrapped to [-180, 180].
    """
    euler_est = quat_to_euler_deg(q_est)
    euler_true = quat_to_euler_deg(q_true)
    error = euler_est - euler_true
    # Wrap to [-180, 180]
    error = np.mod(error + 180, 360) - 180
    return error


def generate_data(config_path: str, db_path: str) -> int:
    """Generate simulation data."""
    print(f"Generating simulation with config: {config_path}")

    generator = EnhancedAttitudeDataGenerator(
        db_path=db_path,
        config_path=config_path
    )

    config = load_yaml(config_path)
    sim_cfg = SimulationConfig(
        T=config['time']['sim_T'],
        dt=config['time']['sim_dt'],
        start_jd=config['time']['start_jd'],
        run_name=config['simulation']['run_name']
    )

    run_id = generator.run(sim_cfg)
    print(f"Generated run ID: {run_id}")
    return run_id


def run_all_estimators(sim_data, estimator_config: str):
    """
    Run ESKF, iSAM2 smoother, and Redundant estimator.
    """
    att_err_rad = np.deg2rad(5.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize estimators
    eskf = ESKF(P0=P0, config_path=estimator_config)
    smoother = FixedLagAttitudeSmoother(
        config_path=estimator_config, lag=60.0, use_robust=True
    )
    redundant = RedundantEstimator(
        P0=P0, config_path=estimator_config, smoother_lag=60.0, use_robust=True,
        disagreement_threshold_deg=2.0, consecutive_disagreements_to_switch=5,
    )

    # Initial state with small perturbation
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    x_eskf = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )
    x_redundant = EskfState(
        nom=NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    # Results storage
    results = {
        'times': [],
        'true_euler': [],
        'eskf_euler': [],
        'smoother_euler': [],
        'redundant_euler': [],
        'eskf_error': [],
        'smoother_error': [],
        'redundant_error': [],
    }

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]
        omega_k = sim_data.omega_meas[k].copy()

        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1].copy()

        # Predict
        x_eskf = eskf.predict(x_eskf, omega_k, dt_k)
        smoother.integrate_gyro(omega_k, dt_k, t=t)

        B_n, s_n = sim_data.b_eci[k], sim_data.s_eci[k]

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        # ESKF updates
        if z_mag is not None:
            try: x_eskf = eskf.update(x_eskf, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except: pass
        if z_sun is not None:
            try: x_eskf = eskf.update(x_eskf, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except: pass
        if z_st is not None:
            try: x_eskf = eskf.update(x_eskf, z_st, SensorType.STAR_TRACKER)
            except: pass

        # Smoother updates
        has_meas = z_mag is not None or z_sun is not None or z_st is not None
        if has_meas:
            smoother_state = smoother.add_measurement(
                t=t, jd=jd, z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_eci=B_n, s_eci=s_n
            )
        else:
            smoother_state = smoother.get_propagated_state()

        # Redundant estimator
        x_redundant, redundant_smoother_state, _, primary = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        if primary == 'SMOOTHER':
            redundant_state = redundant_smoother_state
        else:
            redundant_state = x_redundant.nom

        # True state
        q_true = Quaternion.from_array(sim_data.q_true[k])

        # Compute Euler angles
        true_euler = quat_to_euler_deg(q_true)
        eskf_euler = quat_to_euler_deg(x_eskf.nom.ori)
        smoother_euler = quat_to_euler_deg(smoother_state.ori) if smoother_state else eskf_euler
        redundant_euler = quat_to_euler_deg(redundant_state.ori)

        # Compute errors
        eskf_err = compute_euler_error(x_eskf.nom.ori, q_true)
        smoother_err = compute_euler_error(smoother_state.ori, q_true) if smoother_state else eskf_err
        redundant_err = compute_euler_error(redundant_state.ori, q_true)

        # Store results
        results['times'].append(t)
        results['true_euler'].append(true_euler)
        results['eskf_euler'].append(eskf_euler)
        results['smoother_euler'].append(smoother_euler)
        results['redundant_euler'].append(redundant_euler)
        results['eskf_error'].append(eskf_err)
        results['smoother_error'].append(smoother_err)
        results['redundant_error'].append(redundant_err)

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_true_euler_angles(results: dict, save_path: str):
    """Plot the true roll, pitch, yaw angles over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t = results['times']
    euler = results['true_euler']

    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, euler[:, i], color=color, linewidth=2.0)
        ax.set_ylabel(f'{label} [deg]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t[-1]])

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('True Attitude (Euler Angles)', fontsize=22, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_euler_errors(results: dict, save_path: str, title_suffix: str = "", t_start: float = 60.0):
    """Plot roll, pitch, yaw errors for all estimators."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t = results['times']
    mask = t >= t_start
    t_plot = t[mask]

    labels = ['Roll Error', 'Pitch Error', 'Yaw Error']

    # Compute global y-limits for consistent axes
    all_errors = np.concatenate([
        results['eskf_error'][mask],
        results['smoother_error'][mask],
        results['redundant_error'][mask]
    ])
    y_max = np.max(np.abs(all_errors)) * 1.15  # 15% margin
    y_lim = (-y_max, y_max)

    # Subsample for markers (every 100th point to avoid clutter)
    marker_stride = 100

    for i, (ax, label) in enumerate(zip(axes, labels)):
        # ESKF: solid line with circle markers
        ax.plot(t_plot, results['eskf_error'][mask, i],
                color='#1f77b4', linewidth=1.5, label='ESKF', alpha=0.9)
        ax.plot(t_plot[::marker_stride], results['eskf_error'][mask, i][::marker_stride],
                'o', color='#1f77b4', markersize=5, alpha=0.7)

        # iSAM2: solid line (green)
        ax.plot(t_plot, results['smoother_error'][mask, i],
                color='#2ca02c', linewidth=1.5, label='iSAM2', alpha=0.9)

        # Redundant: dashed line with triangle markers
        ax.plot(t_plot, results['redundant_error'][mask, i],
                color='#d62728', linewidth=2.0, linestyle='--',
                label='Redundant', alpha=0.9)
        ax.plot(t_plot[::marker_stride], results['redundant_error'][mask, i][::marker_stride],
                '^', color='#d62728', markersize=5, alpha=0.7)

        ax.set_ylabel(f'{label} [deg]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([t_start, t[-1]])
        ax.set_ylim(y_lim)  # Consistent y-axis

        if i == 0:
            ax.legend(loc='upper right', ncol=3)

    axes[-1].set_xlabel('Time [s]')
    title = f'Attitude Estimation Errors{title_suffix}'
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_combined(results: dict, save_path: str, title: str = "", t_start: float = 60.0):
    """Create a combined figure with true angles and errors."""
    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.3)

    t = results['times']
    mask = t >= t_start
    t_full = t
    t_plot = t[mask]

    euler = results['true_euler']

    angle_labels = ['Roll', 'Pitch', 'Yaw']
    angle_colors = ['#1f77b4', '#2ca02c', '#d62728']

    # Left column: True Euler angles
    for i, (label, color) in enumerate(zip(angle_labels, angle_colors)):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t_full, euler[:, i], color=color, linewidth=2.0)
        ax.set_ylabel(f'{label} [deg]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t_full[-1]])
        if i == 0:
            ax.set_title('True Attitude', fontsize=20, fontweight='bold')
        if i == 2:
            ax.set_xlabel('Time [s]')

    # Right column: Estimation errors
    est_colors = {
        'ESKF': '#1f77b4',
        'iSAM2': '#2ca02c',
        'Redundant': '#d62728',
    }

    for i, label in enumerate(angle_labels):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t_plot, results['eskf_error'][mask, i],
                color=est_colors['ESKF'], linewidth=1.5, label='ESKF', alpha=0.9)
        ax.plot(t_plot, results['smoother_error'][mask, i],
                color=est_colors['iSAM2'], linewidth=1.5, label='iSAM2', alpha=0.9)
        ax.plot(t_plot, results['redundant_error'][mask, i],
                color=est_colors['Redundant'], linewidth=2.0, linestyle='--',
                label='Redundant', alpha=0.9)

        ax.set_ylabel(f'{label} Error [deg]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([t_start, t[-1]])

        if i == 0:
            ax.set_title('Estimation Errors', fontsize=20, fontweight='bold')
            ax.legend(loc='upper right', ncol=3, fontsize=12)
        if i == 2:
            ax.set_xlabel('Time [s]')

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_statistics(results: dict, t_start: float = 60.0, label: str = ""):
    """Print error statistics for each estimator."""
    t = results['times']
    mask = t >= t_start

    print(f"\n{'='*70}")
    print(f"EULER ANGLE ERROR STATISTICS {label}")
    print(f"(t >= {t_start}s, after convergence)")
    print("="*70)

    estimators = ['eskf', 'smoother', 'redundant']
    names = ['ESKF', 'iSAM2', 'Redundant']
    angles = ['Roll', 'Pitch', 'Yaw']

    for est, name in zip(estimators, names):
        errors = results[f'{est}_error'][mask]
        print(f"\n{name}:")
        for i, angle in enumerate(angles):
            mean_err = np.mean(np.abs(errors[:, i]))
            std_err = np.std(errors[:, i])
            max_err = np.max(np.abs(errors[:, i]))
            print(f"  {angle:5s}: mean={mean_err:.4f}°, std={std_err:.4f}°, max={max_err:.4f}°")

        total_err = np.sqrt(np.sum(errors**2, axis=1))
        print(f"  Total: mean={np.mean(total_err):.4f}°, std={np.std(total_err):.4f}°, max={np.max(total_err):.4f}°")


def main():
    db_path = "simulations.db"
    estimator_config = "configs/config_baseline_short.yaml"

    print("="*70)
    print("EULER ANGLE ANALYSIS")
    print("="*70)

    # =========================================================================
    # Part 1: Perfect Measurements
    # =========================================================================
    print("\n" + "-"*70)
    print("Part 1: Perfect Measurements (zero noise)")
    print("-"*70)

    perfect_config = "configs/config_perfect_meas.yaml"
    run_id_perfect = generate_data(perfect_config, db_path)

    db = SimulationDatabase(db_path)
    sim_data_perfect = db.load_run(run_id_perfect)
    print(f"Loaded {len(sim_data_perfect.t)} samples")

    print("\nRunning estimators with standard noise assumptions...")
    results_perfect = run_all_estimators(sim_data_perfect, estimator_config)

    print_statistics(results_perfect, label="(Perfect Measurements)")

    plot_true_euler_angles(results_perfect, 'true_euler_angles.png')
    plot_euler_errors(results_perfect, 'euler_errors_perfect.png',
                      title_suffix=' (Perfect Measurements)')
    plot_combined(results_perfect, 'euler_combined_perfect.png',
                  title='Perfect Measurements')

    # =========================================================================
    # Part 2: Realistic Noisy Measurements
    # =========================================================================
    print("\n" + "-"*70)
    print("Part 2: Realistic Noisy Measurements")
    print("-"*70)

    noisy_config = "configs/config_baseline_short.yaml"
    run_id_noisy = generate_data(noisy_config, db_path)

    sim_data_noisy = db.load_run(run_id_noisy)
    print(f"Loaded {len(sim_data_noisy.t)} samples")

    print("\nRunning estimators...")
    results_noisy = run_all_estimators(sim_data_noisy, estimator_config)

    print_statistics(results_noisy, label="(Noisy Measurements)")

    plot_euler_errors(results_noisy, 'euler_errors_noisy.png',
                      title_suffix=' (Realistic Noise)')
    plot_combined(results_noisy, 'euler_combined_noisy.png',
                  title='Realistic Sensor Noise')

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
