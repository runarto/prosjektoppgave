#!/usr/bin/env python3
"""
Transient Fault Recovery Test.

Injects brief severe sensor faults and measures recovery time for:
- ESKF (with χ² test)
- Fixed-Lag Smoother (with M-estimator)
- Redundant Estimator (combined approach)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
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
class TransientFault:
    """Definition of a transient fault."""
    start_time: float  # seconds
    duration: float    # seconds
    sensor: str        # 'mag', 'sun', 'star', or 'all'
    magnitude: float   # fault magnitude


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def inject_transient_fault(sim_data, fault: TransientFault, seed: int = None):
    """Inject a transient fault into simulation data."""
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

    fault_end = fault.start_time + fault.duration

    for k in range(len(modified.t)):
        t = modified.t[k]

        if fault.start_time <= t <= fault_end:
            # Apply fault based on sensor type
            if fault.sensor in ['mag', 'all']:
                if not np.any(np.isnan(modified.mag_meas[k])):
                    spike = np.random.randn(3)
                    spike = spike / np.linalg.norm(spike) * fault.magnitude
                    spiked = modified.mag_meas[k] + spike
                    modified.mag_meas[k] = spiked / np.linalg.norm(spiked)

            if fault.sensor in ['sun', 'all']:
                if not np.any(np.isnan(modified.sun_meas[k])):
                    spike = np.random.randn(3)
                    spike = spike / np.linalg.norm(spike) * fault.magnitude
                    spiked = modified.sun_meas[k] + spike
                    modified.sun_meas[k] = spiked / np.linalg.norm(spiked)

            if fault.sensor in ['star', 'all']:
                if not np.any(np.isnan(modified.st_meas[k])):
                    spike_angle = fault.magnitude * 10  # radians
                    spike_axis = np.random.randn(3)
                    spike_axis = spike_axis / np.linalg.norm(spike_axis) * spike_angle
                    q_spike = Quaternion.from_avec(spike_axis)
                    q_orig = Quaternion.from_array(modified.st_meas[k])
                    q_spiked = (q_orig @ q_spike).normalize()
                    modified.st_meas[k] = q_spiked.as_array()

    return modified


def run_eskf(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run ESKF with chi2 test."""
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
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k],
                                   SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass

        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors)


def run_smoother(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run fixed-lag smoother with M-estimator."""
    att_err_rad = np.deg2rad(17.0)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=True,
    )
    smoother.initialize(sim_data.t[0], q0_est, np.zeros(3))

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        smoother.integrate_gyro(omega, dt, t=t)

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_eci = sim_data.b_eci[k] if z_mag is not None else None
        s_eci = sim_data.s_eci[k] if z_sun is not None else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            state = smoother.add_measurement(t=t, jd=jd, z_mag=z_mag, z_sun=z_sun,
                                            z_st=z_st, B_eci=B_eci, s_eci=s_eci)
        else:
            state = smoother.get_propagated_state()

        if state is not None:
            times.append(t)
            errors.append(compute_attitude_error_deg(state.ori, q_true))

    return np.array(times), np.array(errors)


def run_redundant(sim_data, config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run redundant estimator."""
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

    times, errors = [], []

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
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors)


def compute_recovery_time(times: np.ndarray, errors: np.ndarray,
                          fault_end: float, threshold: float = 0.05) -> Optional[float]:
    """
    Compute time to recover to within threshold after fault ends.

    Returns None if never recovers within the data window.
    """
    # Find index where fault ends
    fault_end_idx = np.searchsorted(times, fault_end)

    # Look for first time error drops below threshold after fault
    for i in range(fault_end_idx, len(errors)):
        if errors[i] < threshold:
            return times[i] - fault_end

    return None  # Never recovered


def run_monte_carlo(sim_data_base, config_path: str, n_runs: int = 20) -> Dict:
    """Run Monte Carlo transient fault recovery test."""

    results = {
        'eskf': {'recovery_times': [], 'max_errors': [], 'errors_during_fault': []},
        'smoother': {'recovery_times': [], 'max_errors': [], 'errors_during_fault': []},
        'redundant': {'recovery_times': [], 'max_errors': [], 'errors_during_fault': []},
    }

    # Store one example run for plotting
    example_run = None

    print(f"\nRunning {n_runs} Monte Carlo simulations...")
    print("-" * 70)

    for i in range(n_runs):
        # Random fault parameters
        np.random.seed(i * 42)
        fault = TransientFault(
            start_time=100.0 + np.random.uniform(0, 50),  # Fault between 100-150s
            duration=3.0,  # 3 second fault
            sensor='star',  # Star tracker fault (most impactful)
            magnitude=0.5,  # Severe fault
        )

        # Inject fault
        sim_data = inject_transient_fault(sim_data_base, fault, seed=i*42)
        fault_end = fault.start_time + fault.duration

        # Run estimators
        times_eskf, errors_eskf = run_eskf(sim_data, config_path)
        times_sm, errors_sm = run_smoother(sim_data, config_path)
        times_red, errors_red = run_redundant(sim_data, config_path)

        # Compute metrics
        recovery_threshold = 0.05  # degrees

        # ESKF
        rec_eskf = compute_recovery_time(times_eskf, errors_eskf, fault_end, recovery_threshold)
        fault_mask_eskf = (times_eskf >= fault.start_time) & (times_eskf <= fault_end + 30)
        max_err_eskf = np.max(errors_eskf[fault_mask_eskf]) if np.any(fault_mask_eskf) else 0

        # Smoother
        rec_sm = compute_recovery_time(times_sm, errors_sm, fault_end, recovery_threshold)
        fault_mask_sm = (times_sm >= fault.start_time) & (times_sm <= fault_end + 30)
        max_err_sm = np.max(errors_sm[fault_mask_sm]) if np.any(fault_mask_sm) else 0

        # Redundant
        rec_red = compute_recovery_time(times_red, errors_red, fault_end, recovery_threshold)
        fault_mask_red = (times_red >= fault.start_time) & (times_red <= fault_end + 30)
        max_err_red = np.max(errors_red[fault_mask_red]) if np.any(fault_mask_red) else 0

        # Store results
        results['eskf']['recovery_times'].append(rec_eskf if rec_eskf else 60.0)
        results['eskf']['max_errors'].append(max_err_eskf)

        results['smoother']['recovery_times'].append(rec_sm if rec_sm else 60.0)
        results['smoother']['max_errors'].append(max_err_sm)

        results['redundant']['recovery_times'].append(rec_red if rec_red else 60.0)
        results['redundant']['max_errors'].append(max_err_red)

        # Save first run for example plot
        if i == 0:
            example_run = {
                'fault': fault,
                'eskf': (times_eskf, errors_eskf),
                'smoother': (times_sm, errors_sm),
                'redundant': (times_red, errors_red),
            }

        print(f"  Run {i+1:3d}/{n_runs}: Fault @ {fault.start_time:.1f}s, "
              f"Recovery: ESKF={rec_eskf:.1f}s, SM={rec_sm:.1f}s, RED={rec_red:.1f}s"
              if rec_eskf and rec_sm and rec_red else
              f"  Run {i+1:3d}/{n_runs}: Fault @ {fault.start_time:.1f}s, "
              f"Recovery: ESKF={'N/A' if not rec_eskf else f'{rec_eskf:.1f}s'}, "
              f"SM={'N/A' if not rec_sm else f'{rec_sm:.1f}s'}, "
              f"RED={'N/A' if not rec_red else f'{rec_red:.1f}s'}")

    results['example'] = example_run
    return results


def plot_results(results: Dict, save_path: str):
    """Plot transient fault recovery results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Example time series
    ax = axes[0, 0]
    example = results['example']
    fault = example['fault']

    times_eskf, errors_eskf = example['eskf']
    times_sm, errors_sm = example['smoother']
    times_red, errors_red = example['redundant']

    # Zoom to fault region
    t_start = fault.start_time - 10
    t_end = fault.start_time + fault.duration + 40

    mask_eskf = (times_eskf >= t_start) & (times_eskf <= t_end)
    mask_sm = (times_sm >= t_start) & (times_sm <= t_end)
    mask_red = (times_red >= t_start) & (times_red <= t_end)

    ax.semilogy(times_eskf[mask_eskf], errors_eskf[mask_eskf], 'C0-', label='ESKF', linewidth=1.5)
    ax.semilogy(times_sm[mask_sm], errors_sm[mask_sm], 'C1-', label='Smoother', linewidth=1.5)
    ax.semilogy(times_red[mask_red], errors_red[mask_red], 'C2--', label='Redundant', linewidth=1.5)

    # Mark fault region
    ax.axvspan(fault.start_time, fault.start_time + fault.duration,
               alpha=0.3, color='red', label='Fault period')
    ax.axhline(0.05, color='gray', linestyle=':', label='Recovery threshold')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Example: Transient Star Tracker Fault')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Recovery time box plot
    ax = axes[0, 1]
    data = [results['eskf']['recovery_times'],
            results['smoother']['recovery_times'],
            results['redundant']['recovery_times']]
    bp = ax.boxplot(data, tick_labels=['ESKF', 'Smoother', 'Redundant'])
    ax.set_ylabel('Recovery Time [s]')
    ax.set_title('Time to Recover Below 0.05°')
    ax.grid(True, alpha=0.3, axis='y')

    # Max error box plot
    ax = axes[1, 0]
    data = [results['eskf']['max_errors'],
            results['smoother']['max_errors'],
            results['redundant']['max_errors']]
    bp = ax.boxplot(data, tick_labels=['ESKF', 'Smoother', 'Redundant'])
    ax.set_ylabel('Maximum Error [deg]')
    ax.set_title('Peak Error During/After Fault')
    ax.grid(True, alpha=0.3, axis='y')

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    def safe_mean(lst):
        valid = [x for x in lst if x is not None and x < 60]
        return np.mean(valid) if valid else float('nan')

    def safe_std(lst):
        valid = [x for x in lst if x is not None and x < 60]
        return np.std(valid) if valid else float('nan')

    stats_text = f"""
    Transient Fault Recovery Results ({len(results['eskf']['recovery_times'])} runs)
    {'='*50}

    Fault Configuration:
      Sensor:     Star Tracker
      Duration:   3.0 seconds
      Magnitude:  0.5 (severe)

    Recovery Time to < 0.05° [seconds]:
                    Mean      Std       Median
      ESKF:         {np.mean(results['eskf']['recovery_times']):6.2f}    {np.std(results['eskf']['recovery_times']):6.2f}    {np.median(results['eskf']['recovery_times']):6.2f}
      Smoother:     {np.mean(results['smoother']['recovery_times']):6.2f}    {np.std(results['smoother']['recovery_times']):6.2f}    {np.median(results['smoother']['recovery_times']):6.2f}
      Redundant:    {np.mean(results['redundant']['recovery_times']):6.2f}    {np.std(results['redundant']['recovery_times']):6.2f}    {np.median(results['redundant']['recovery_times']):6.2f}

    Peak Error During Fault [degrees]:
                    Mean      Std       Median
      ESKF:         {np.mean(results['eskf']['max_errors']):6.2f}    {np.std(results['eskf']['max_errors']):6.2f}    {np.median(results['eskf']['max_errors']):6.2f}
      Smoother:     {np.mean(results['smoother']['max_errors']):6.2f}    {np.std(results['smoother']['max_errors']):6.2f}    {np.median(results['smoother']['max_errors']):6.2f}
      Redundant:    {np.mean(results['redundant']['max_errors']):6.2f}    {np.std(results['redundant']['max_errors']):6.2f}    {np.median(results['redundant']['max_errors']):6.2f}
    """
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'

    print("=" * 70)
    print("TRANSIENT FAULT RECOVERY TEST")
    print("=" * 70)

    # Create config
    config = load_yaml(base_config)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    config_path = 'configs/config_transient_test.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load simulation data
    print("\nLoading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    # Run Monte Carlo
    results = run_monte_carlo(sim_data_base, config_path, n_runs=20)

    # Plot results
    plot_results(results, 'transient_fault_recovery.png')

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMean Recovery Time [s]:")
    print(f"  ESKF:      {np.mean(results['eskf']['recovery_times']):.2f}")
    print(f"  Smoother:  {np.mean(results['smoother']['recovery_times']):.2f}")
    print(f"  Redundant: {np.mean(results['redundant']['recovery_times']):.2f}")

    print("\nMean Peak Error [deg]:")
    print(f"  ESKF:      {np.mean(results['eskf']['max_errors']):.2f}")
    print(f"  Smoother:  {np.mean(results['smoother']['max_errors']):.2f}")
    print(f"  Redundant: {np.mean(results['redundant']['max_errors']):.2f}")


if __name__ == "__main__":
    main()
