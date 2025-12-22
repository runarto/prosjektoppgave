#!/usr/bin/env python3
"""
Monte Carlo comparison: ESKF chi2 test vs Smoother M-estimator.

Tests outlier rejection under frequent measurement spikes on all sensors.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, List, Dict
from dataclasses import dataclass

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from utilities.utils import load_yaml

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['legend.fontsize'] = 14
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
class SpikeConfig:
    """Configuration for measurement spikes."""
    mag_spike_prob: float = 0.05  # 5% of mag measurements are spikes
    sun_spike_prob: float = 0.05  # 5% of sun measurements are spikes
    star_spike_prob: float = 0.05  # 5% of star tracker measurements are spikes
    spike_magnitude: float = 0.3  # Spike magnitude (unit vector perturbation)


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def add_spikes(sim_data, spike_config: SpikeConfig, seed: int = None):
    """Add measurement spikes to simulation data."""
    if seed is not None:
        np.random.seed(seed)

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

    n_mag_spikes = 0
    n_sun_spikes = 0
    n_star_spikes = 0

    for k in range(len(modified.t)):
        # Magnetometer spikes
        if not np.any(np.isnan(modified.mag_meas[k])):
            if np.random.random() < spike_config.mag_spike_prob:
                # Add random spike
                spike = np.random.randn(3)
                spike = spike / np.linalg.norm(spike) * spike_config.spike_magnitude
                spiked = modified.mag_meas[k] + spike
                modified.mag_meas[k] = spiked / np.linalg.norm(spiked)
                n_mag_spikes += 1

        # Sun sensor spikes
        if not np.any(np.isnan(modified.sun_meas[k])):
            if np.random.random() < spike_config.sun_spike_prob:
                spike = np.random.randn(3)
                spike = spike / np.linalg.norm(spike) * spike_config.spike_magnitude
                spiked = modified.sun_meas[k] + spike
                modified.sun_meas[k] = spiked / np.linalg.norm(spiked)
                n_sun_spikes += 1

        # Star tracker spikes
        if not np.any(np.isnan(modified.st_meas[k])):
            if np.random.random() < spike_config.star_spike_prob:
                # Add random rotation spike to quaternion
                spike_angle = spike_config.spike_magnitude * 10  # Convert to radians-ish
                spike_axis = np.random.randn(3)
                spike_axis = spike_axis / np.linalg.norm(spike_axis) * spike_angle
                q_spike = Quaternion.from_avec(spike_axis)
                q_orig = Quaternion.from_array(modified.st_meas[k])
                q_spiked = (q_orig @ q_spike).normalize()
                modified.st_meas[k] = q_spiked.as_array()
                n_star_spikes += 1

    return modified, (n_mag_spikes, n_sun_spikes, n_star_spikes)


def run_eskf_with_chi2(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run ESKF with chi2 outlier rejection."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    eskf = ESKF(P0=P0, config_path=config_path)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times, errors = [], []
    rejected_count = 0

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        x_est = eskf.predict(x_est, omega, dt)

        # Magnetometer
        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.mag_meas[k],
                                   SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except ValueError:
                rejected_count += 1

        # Sun sensor
        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k],
                                   SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                rejected_count += 1

        # Star tracker
        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except ValueError:
                rejected_count += 1

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors), rejected_count


def run_smoother_with_mestimator(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run fixed-lag smoother with M-estimator (robust mode)."""
    att_err_rad = np.deg2rad(17.0)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=True,  # Enable M-estimator
    )
    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        # Integrate gyro
        smoother.integrate_gyro(omega, dt, t=t)

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_eci = sim_data.b_eci[k] if z_mag is not None else None
        s_eci = sim_data.s_eci[k] if z_sun is not None else None

        # Add measurement if any sensor has data
        if z_mag is not None or z_sun is not None or z_st is not None:
            state = smoother.add_measurement(
                t=t, jd=jd,
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=B_eci, s_eci=s_eci,
            )
        else:
            state = smoother.get_propagated_state()

        if state is not None:
            times.append(t)
            errors.append(compute_attitude_error_deg(state.ori, q_true))

    return np.array(times), np.array(errors)


def run_monte_carlo(sim_data_base, config_path: str, spike_config: SpikeConfig,
                    n_runs: int = 20) -> Dict:
    """Run Monte Carlo comparison."""
    eskf_errors_all = []
    smoother_errors_all = []
    eskf_rejected_all = []
    total_spikes_all = []

    print(f"\nRunning {n_runs} Monte Carlo simulations...")
    print("-" * 60)

    for i in range(n_runs):
        # Add spikes with different seed each run
        sim_data, spike_counts = add_spikes(sim_data_base, spike_config, seed=i*42)
        total_spikes = sum(spike_counts)

        # Run ESKF
        times_eskf, errors_eskf, rejected = run_eskf_with_chi2(sim_data, config_path)

        # Run Smoother
        times_sm, errors_sm = run_smoother_with_mestimator(sim_data, config_path)

        # Store steady-state errors (t >= 100s)
        ss_mask_eskf = times_eskf >= 100.0
        ss_mask_sm = times_sm >= 100.0

        eskf_errors_all.append(errors_eskf[ss_mask_eskf])
        smoother_errors_all.append(errors_sm[ss_mask_sm])
        eskf_rejected_all.append(rejected)
        total_spikes_all.append(total_spikes)

        print(f"  Run {i+1:3d}/{n_runs}: Spikes={total_spikes}, "
              f"ESKF rejected={rejected}, "
              f"ESKF mean={np.mean(errors_eskf[ss_mask_eskf]):.4f}°, "
              f"Smoother mean={np.mean(errors_sm[ss_mask_sm]):.4f}°")

    return {
        'eskf_errors': eskf_errors_all,
        'smoother_errors': smoother_errors_all,
        'eskf_rejected': eskf_rejected_all,
        'total_spikes': total_spikes_all,
        'times_eskf': times_eskf,
        'times_sm': times_sm,
    }


def plot_results(results: Dict, spike_config: SpikeConfig, save_path: str):
    """Plot Monte Carlo results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Flatten errors for histogram
    eskf_flat = np.concatenate(results['eskf_errors'])
    smoother_flat = np.concatenate(results['smoother_errors'])

    # 1. Error histogram
    ax = axes[0, 0]
    bins = np.logspace(-3, 1, 50)
    ax.hist(eskf_flat, bins=bins, alpha=0.7, label=f'ESKF ($\\chi^2$ test)', density=True)
    ax.hist(smoother_flat, bins=bins, alpha=0.7, label='Smoother (M-estimator)', density=True)
    ax.set_xscale('log')
    ax.set_xlabel('Attitude Error [deg]')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution (Steady-State)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax = axes[0, 1]
    eskf_means = [np.mean(e) for e in results['eskf_errors']]
    smoother_means = [np.mean(e) for e in results['smoother_errors']]
    bp = ax.boxplot([eskf_means, smoother_means], labels=['ESKF\n($\\chi^2$ test)', 'Smoother\n(M-estimator)'])
    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_title('Mean Error per Run')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Rejection rate vs spikes
    ax = axes[1, 0]
    ax.scatter(results['total_spikes'], results['eskf_rejected'], alpha=0.7, s=50)
    ax.set_xlabel('Total Spikes Injected')
    ax.set_ylabel('Measurements Rejected by $\\chi^2$ Test')
    ax.set_title('ESKF Outlier Rejection')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(results['total_spikes'], results['eskf_rejected'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(results['total_spikes']), max(results['total_spikes']), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend')
    ax.legend()

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    Monte Carlo Results ({len(results['eskf_errors'])} runs)
    {'='*45}

    Spike Configuration:
      Mag spike prob:   {spike_config.mag_spike_prob*100:.1f}%
      Sun spike prob:   {spike_config.sun_spike_prob*100:.1f}%
      Star spike prob:  {spike_config.star_spike_prob*100:.1f}%
      Spike magnitude:  {spike_config.spike_magnitude}

    ESKF (χ² test):
      Mean error:       {np.mean(eskf_flat):.4f}°
      Std error:        {np.std(eskf_flat):.4f}°
      Median error:     {np.median(eskf_flat):.4f}°
      95th percentile:  {np.percentile(eskf_flat, 95):.4f}°
      Avg rejected:     {np.mean(results['eskf_rejected']):.1f}

    Smoother (M-estimator):
      Mean error:       {np.mean(smoother_flat):.4f}°
      Std error:        {np.std(smoother_flat):.4f}°
      Median error:     {np.median(smoother_flat):.4f}°
      95th percentile:  {np.percentile(smoother_flat, 95):.4f}°
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'

    print("=" * 60)
    print("CHI2 vs M-ESTIMATOR MONTE CARLO COMPARISON")
    print("=" * 60)

    # Create config
    config = load_yaml(base_config)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    config_path = 'configs/config_spike_test.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load base simulation data
    print("\nLoading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    # Configure spikes
    spike_config = SpikeConfig(
        mag_spike_prob=0.10,   # 10% spike rate
        sun_spike_prob=0.10,
        star_spike_prob=0.10,
        spike_magnitude=0.3,
    )

    print(f"\nSpike configuration:")
    print(f"  Mag spike probability:  {spike_config.mag_spike_prob*100:.0f}%")
    print(f"  Sun spike probability:  {spike_config.sun_spike_prob*100:.0f}%")
    print(f"  Star spike probability: {spike_config.star_spike_prob*100:.0f}%")
    print(f"  Spike magnitude:        {spike_config.spike_magnitude}")

    # Run Monte Carlo
    results = run_monte_carlo(sim_data_base, config_path, spike_config, n_runs=20)

    # Plot results
    plot_results(results, spike_config, 'chi2_vs_mestimator_mc.png')

    # Print summary
    eskf_flat = np.concatenate(results['eskf_errors'])
    smoother_flat = np.concatenate(results['smoother_errors'])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nESKF (χ² test):")
    print(f"  Mean error:      {np.mean(eskf_flat):.4f}°")
    print(f"  Std error:       {np.std(eskf_flat):.4f}°")
    print(f"  Avg rejected:    {np.mean(results['eskf_rejected']):.1f} measurements")

    print(f"\nSmoother (M-estimator):")
    print(f"  Mean error:      {np.mean(smoother_flat):.4f}°")
    print(f"  Std error:       {np.std(smoother_flat):.4f}°")

    improvement = (np.mean(eskf_flat) - np.mean(smoother_flat)) / np.mean(eskf_flat) * 100
    print(f"\nImprovement: {improvement:+.1f}% (positive = smoother better)")


if __name__ == "__main__":
    main()
