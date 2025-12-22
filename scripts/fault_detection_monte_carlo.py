#!/usr/bin/env python3
"""
Monte Carlo Fault Detection Analysis.

Runs multiple trials to get statistically meaningful results for:
1. Persistent faults - graceful degradation
2. Transient faults - recovery capability
3. Detection performance - delay and false alarm rates
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

from data.db import SimulationDatabase
from data.classes import SimulationConfig
from estimation.eskf import ESKF
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


@dataclass
class TrialResult:
    """Results from a single Monte Carlo trial."""
    # Time series
    times: np.ndarray
    eskf_only_errors: np.ndarray
    redundant_eskf_errors: np.ndarray
    redundant_smoother_errors: np.ndarray
    redundant_selected_errors: np.ndarray
    disagreements: np.ndarray

    # Scalar metrics
    eskf_only_post_fault_mean: float
    redundant_post_fault_mean: float
    detection_delay: Optional[float]
    n_switch_events: int

    # Fault config
    fault_onset: float
    fault_duration: Optional[float]  # None for persistent


def run_single_trial(
    sim_data,
    config_path: str,
    fault_config: Dict,
    initial_error_deg: float = 10.0,
) -> TrialResult:
    """Run a single Monte Carlo trial."""

    att_err_rad = np.deg2rad(initial_error_deg)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    # Initialize both estimators with same initial conditions
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    # ESKF only
    eskf = ESKF(P0=P0.copy(), config_path=config_path)
    nom_eskf = NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3))
    err_eskf = MultiVarGauss(np.zeros(6), P0.copy())
    x_eskf_only = EskfState(nom=nom_eskf, err=err_eskf)

    # Redundant
    redundant = RedundantEstimator(
        P0=P0.copy(),
        config_path=config_path,
        smoother_lag=60.0,
        use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )
    nom_red = NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3))
    err_red = MultiVarGauss(np.zeros(6), P0.copy())
    x_redundant = EskfState(nom=nom_red, err=err_red)

    # Storage
    times = []
    eskf_only_errors = []
    redundant_eskf_errors = []
    redundant_smoother_errors = []
    redundant_selected_errors = []
    disagreements = []

    fault_onset = fault_config['onset_time']
    fault_duration = fault_config.get('duration', None)  # None = persistent
    fault_magnitude = fault_config['magnitude']

    injected_bias = np.zeros(3)
    fault_active = False
    first_detection_time = None

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]

        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        # Apply fault
        if fault_duration is None:  # Persistent
            if t >= fault_onset:
                if not fault_active:
                    fault_active = True
                    injected_bias = fault_magnitude * np.array([1, 0.5, 0.3])
                    injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_magnitude
                omega_k = omega_k + injected_bias
        else:  # Transient
            if fault_onset <= t < fault_onset + fault_duration:
                if not fault_active:
                    fault_active = True
                    injected_bias = fault_magnitude * np.array([1, 0.5, 0.3])
                    injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_magnitude
                omega_k = omega_k + injected_bias
            else:
                fault_active = False
                injected_bias = np.zeros(3)

        # Get measurements
        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # === ESKF only ===
        x_eskf_only = eskf.predict(x_eskf_only, omega_k, dt_k)
        if z_mag is not None:
            try:
                x_eskf_only = eskf.update(x_eskf_only, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError:
                pass
        if z_sun is not None:
            try:
                x_eskf_only = eskf.update(x_eskf_only, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError:
                pass
        if z_st is not None:
            try:
                x_eskf_only = eskf.update(x_eskf_only, z_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # === Redundant ===
        x_redundant, smoother_state, disagreement_deg, primary_str = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        # Track first detection
        if first_detection_time is None and t >= fault_onset and disagreement_deg > 2.0:
            first_detection_time = t

        # Compute errors
        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_only_err = compute_attitude_error(x_eskf_only.nom.ori, q_true)
        red_eskf_err = compute_attitude_error(x_redundant.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true)
        selected_err = smoother_err if primary_str == 'SMOOTHER' else red_eskf_err

        times.append(t)
        eskf_only_errors.append(eskf_only_err)
        redundant_eskf_errors.append(red_eskf_err)
        redundant_smoother_errors.append(smoother_err)
        redundant_selected_errors.append(selected_err)
        disagreements.append(disagreement_deg)

    times = np.array(times)
    eskf_only_errors = np.array(eskf_only_errors)
    redundant_selected_errors = np.array(redundant_selected_errors)

    # Compute post-fault metrics
    if fault_duration is None:  # Persistent
        post_mask = times >= fault_onset + 30  # After 30s of fault
    else:  # Transient
        post_mask = times >= fault_onset + fault_duration  # After fault clears

    eskf_post_mean = np.mean(eskf_only_errors[post_mask])
    red_post_mean = np.mean(redundant_selected_errors[post_mask])

    detection_delay = first_detection_time - fault_onset if first_detection_time else None

    stats = redundant.get_statistics()

    return TrialResult(
        times=times,
        eskf_only_errors=eskf_only_errors,
        redundant_eskf_errors=np.array(redundant_eskf_errors),
        redundant_smoother_errors=np.array(redundant_smoother_errors),
        redundant_selected_errors=redundant_selected_errors,
        disagreements=np.array(disagreements),
        eskf_only_post_fault_mean=eskf_post_mean,
        redundant_post_fault_mean=red_post_mean,
        detection_delay=detection_delay,
        n_switch_events=len(stats.get('switch_events', [])),
        fault_onset=fault_onset,
        fault_duration=fault_duration,
    )


def run_monte_carlo(
    config_path: str,
    db_path: str,
    n_trials: int,
    fault_config: Dict,
    verbose: bool = True,
) -> List[TrialResult]:
    """Run Monte Carlo simulation."""

    db = SimulationDatabase(db_path)
    config = load_yaml(config_path)

    # Get generator
    from data.generator_enhanced import EnhancedAttitudeDataGenerator

    results = []

    for i in range(n_trials):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Trial {i+1}/{n_trials}")

        # Generate new simulation data
        generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=config_path)
        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd']
        )
        sim_id = generator.run(sim_cfg)
        sim_data = db.load_run(sim_id)

        # Run trial
        result = run_single_trial(sim_data, config_path, fault_config)
        results.append(result)

    return results


def plot_monte_carlo_results(
    results_persistent: List[TrialResult],
    results_transient: List[TrialResult],
    results_nominal: List[TrialResult],
    save_path: str = None,
):
    """Create comprehensive Monte Carlo results figure."""

    fig = plt.figure(figsize=(16, 12))

    # === Row 1: Error distributions (box plots) ===
    ax1 = fig.add_subplot(2, 3, 1)

    # Collect post-fault errors
    eskf_persistent = [r.eskf_only_post_fault_mean for r in results_persistent]
    red_persistent = [r.redundant_post_fault_mean for r in results_persistent]
    eskf_transient = [r.eskf_only_post_fault_mean for r in results_transient]
    red_transient = [r.redundant_post_fault_mean for r in results_transient]
    eskf_nominal = [r.eskf_only_post_fault_mean for r in results_nominal]
    red_nominal = [r.redundant_post_fault_mean for r in results_nominal]

    positions = [1, 2, 4, 5, 7, 8]
    data = [eskf_nominal, red_nominal, eskf_persistent, red_persistent, eskf_transient, red_transient]
    colors = ['lightcoral', 'lightblue'] * 3

    bp = ax1.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xticks([1.5, 4.5, 7.5])
    ax1.set_xticklabels(['Nominal', 'Persistent\nFault', 'Transient\nFault'])
    ax1.set_ylabel('Mean Post-Fault Error [deg]')
    ax1.set_title('Error Distribution by Scenario')
    ax1.legend([Patch(facecolor='lightcoral'), Patch(facecolor='lightblue')],
               ['ESKF only', 'Redundant'], loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # === Row 1: Improvement histogram ===
    ax2 = fig.add_subplot(2, 3, 2)

    improvement_persistent = [(e - r) / e * 100 for e, r in zip(eskf_persistent, red_persistent)]
    improvement_transient = [(e - r) / e * 100 for e, r in zip(eskf_transient, red_transient)]

    bins = np.linspace(-20, 60, 25)
    ax2.hist(improvement_persistent, bins=bins, alpha=0.7, label='Persistent fault', color='orange', edgecolor='black')
    ax2.hist(improvement_transient, bins=bins, alpha=0.7, label='Transient fault', color='green', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(np.mean(improvement_persistent), color='orange', linestyle='-', linewidth=2,
                label=f'Persistent mean: {np.mean(improvement_persistent):.1f}%')
    ax2.axvline(np.mean(improvement_transient), color='green', linestyle='-', linewidth=2,
                label=f'Transient mean: {np.mean(improvement_transient):.1f}%')

    ax2.set_xlabel('Error Reduction [%]')
    ax2.set_ylabel('Count')
    ax2.set_title('Improvement from Redundant Architecture')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === Row 1: Detection delay histogram ===
    ax3 = fig.add_subplot(2, 3, 3)

    delays_persistent = [r.detection_delay for r in results_persistent if r.detection_delay is not None]
    delays_transient = [r.detection_delay for r in results_transient if r.detection_delay is not None]

    if delays_persistent and delays_transient:
        bins = np.linspace(0, max(max(delays_persistent), max(delays_transient)) + 5, 20)
        ax3.hist(delays_persistent, bins=bins, alpha=0.7, label='Persistent', color='orange', edgecolor='black')
        ax3.hist(delays_transient, bins=bins, alpha=0.7, label='Transient', color='green', edgecolor='black')
        ax3.axvline(np.mean(delays_persistent), color='orange', linestyle='-', linewidth=2)
        ax3.axvline(np.mean(delays_transient), color='green', linestyle='-', linewidth=2)

    ax3.set_xlabel('Detection Delay [s]')
    ax3.set_ylabel('Count')
    ax3.set_title('Fault Detection Delay Distribution')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # === Row 2: Time series (mean ± std) ===
    ax4 = fig.add_subplot(2, 3, 4)

    # Stack time series from all trials
    t_common = results_persistent[0].times
    eskf_stack = np.array([r.eskf_only_errors for r in results_persistent])
    red_stack = np.array([r.redundant_selected_errors for r in results_persistent])

    eskf_mean = np.mean(eskf_stack, axis=0)
    eskf_std = np.std(eskf_stack, axis=0)
    red_mean = np.mean(red_stack, axis=0)
    red_std = np.std(red_stack, axis=0)

    t_start = 100
    mask = t_common >= t_start

    ax4.semilogy(t_common[mask], eskf_mean[mask], 'r-', linewidth=1.5, label='ESKF only (mean)')
    ax4.fill_between(t_common[mask],
                     np.maximum(eskf_mean[mask] - eskf_std[mask], 0.001),
                     eskf_mean[mask] + eskf_std[mask],
                     color='red', alpha=0.2)
    ax4.semilogy(t_common[mask], red_mean[mask], 'b-', linewidth=1.5, label='Redundant (mean)')
    ax4.fill_between(t_common[mask],
                     np.maximum(red_mean[mask] - red_std[mask], 0.001),
                     red_mean[mask] + red_std[mask],
                     color='blue', alpha=0.2)

    fault_onset = results_persistent[0].fault_onset
    ax4.axvline(fault_onset, color='black', linestyle='--', linewidth=1, label='Fault onset')

    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Attitude Error [deg]')
    ax4.set_title(f'Persistent Fault: Error Evolution (N={len(results_persistent)} trials)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([1e-2, 100])

    # === Row 2: Transient fault time series ===
    ax5 = fig.add_subplot(2, 3, 5)

    t_common = results_transient[0].times
    eskf_stack = np.array([r.eskf_only_errors for r in results_transient])
    red_stack = np.array([r.redundant_selected_errors for r in results_transient])

    eskf_mean = np.mean(eskf_stack, axis=0)
    eskf_std = np.std(eskf_stack, axis=0)
    red_mean = np.mean(red_stack, axis=0)
    red_std = np.std(red_stack, axis=0)

    mask = t_common >= t_start

    ax5.semilogy(t_common[mask], eskf_mean[mask], 'r-', linewidth=1.5, label='ESKF only (mean)')
    ax5.fill_between(t_common[mask],
                     np.maximum(eskf_mean[mask] - eskf_std[mask], 0.001),
                     eskf_mean[mask] + eskf_std[mask],
                     color='red', alpha=0.2)
    ax5.semilogy(t_common[mask], red_mean[mask], 'b-', linewidth=1.5, label='Redundant (mean)')
    ax5.fill_between(t_common[mask],
                     np.maximum(red_mean[mask] - red_std[mask], 0.001),
                     red_mean[mask] + red_std[mask],
                     color='blue', alpha=0.2)

    fault_onset = results_transient[0].fault_onset
    fault_duration = results_transient[0].fault_duration
    ax5.axvspan(fault_onset, fault_onset + fault_duration, color='red', alpha=0.15, label='Fault active')

    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Attitude Error [deg]')
    ax5.set_title(f'Transient Fault: Error Evolution (N={len(results_transient)} trials)')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([1e-2, 100])

    # === Row 2: Summary statistics table ===
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Compute statistics
    def stats_str(data):
        return f"{np.mean(data):.2f} ± {np.std(data):.2f}"

    table_data = [
        ['Metric', 'Nominal', 'Persistent', 'Transient'],
        ['ESKF Error [deg]',
         stats_str(eskf_nominal),
         stats_str(eskf_persistent),
         stats_str(eskf_transient)],
        ['Redundant Error [deg]',
         stats_str(red_nominal),
         stats_str(red_persistent),
         stats_str(red_transient)],
        ['Improvement [%]',
         '-',
         stats_str(improvement_persistent),
         stats_str(improvement_transient)],
        ['Detection Rate',
         '-',
         f"{len(delays_persistent)}/{len(results_persistent)} ({100*len(delays_persistent)/len(results_persistent):.0f}%)",
         f"{len(delays_transient)}/{len(results_transient)} ({100*len(delays_transient)/len(results_transient):.0f}%)"],
        ['Detection Delay [s]',
         '-',
         stats_str(delays_persistent) if delays_persistent else 'N/A',
         stats_str(delays_transient) if delays_transient else 'N/A'],
    ]

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Summary Statistics (mean ± std)', pad=20, fontweight='bold')

    plt.suptitle('Monte Carlo Fault Detection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Number of Monte Carlo trials')
    parser.add_argument('--config', type=str, default='configs/config_baseline_short.yaml')
    parser.add_argument('--db', type=str, default='simulations.db')
    args = parser.parse_args()

    n_trials = args.trials
    config_path = args.config
    db_path = args.db

    print("="*70)
    print(f"MONTE CARLO FAULT DETECTION ANALYSIS (N={n_trials} trials)")
    print("="*70)

    # Fault configurations
    fault_onset = 150.0
    fault_magnitude = 0.03  # 30 mrad/s

    persistent_config = {
        'type': 'gyro_bias_step',
        'onset_time': fault_onset,
        'magnitude': fault_magnitude,
        'duration': None,  # Persistent
    }

    transient_config = {
        'type': 'gyro_bias_step',
        'onset_time': 120.0,
        'magnitude': 0.05,  # 50 mrad/s
        'duration': 30.0,  # 30 seconds
    }

    nominal_config = {
        'type': 'none',
        'onset_time': float('inf'),
        'magnitude': 0.0,
    }

    start_time = time.time()

    # Run Monte Carlo for each scenario
    print(f"\n1. Running NOMINAL scenario ({n_trials} trials)...")
    results_nominal = run_monte_carlo(config_path, db_path, n_trials, nominal_config)

    print(f"\n2. Running PERSISTENT fault scenario ({n_trials} trials)...")
    results_persistent = run_monte_carlo(config_path, db_path, n_trials, persistent_config)

    print(f"\n3. Running TRANSIENT fault scenario ({n_trials} trials)...")
    results_transient = run_monte_carlo(config_path, db_path, n_trials, transient_config)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Generate plots
    print("\nGenerating plots...")
    plot_monte_carlo_results(
        results_persistent,
        results_transient,
        results_nominal,
        save_path='fault_detection_monte_carlo.pdf'
    )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    eskf_persistent = [r.eskf_only_post_fault_mean for r in results_persistent]
    red_persistent = [r.redundant_post_fault_mean for r in results_persistent]
    eskf_transient = [r.eskf_only_post_fault_mean for r in results_transient]
    red_transient = [r.redundant_post_fault_mean for r in results_transient]

    improvement_persistent = [(e - r) / e * 100 for e, r in zip(eskf_persistent, red_persistent)]
    improvement_transient = [(e - r) / e * 100 for e, r in zip(eskf_transient, red_transient)]

    print(f"\nPersistent Fault ({fault_magnitude*1000:.0f} mrad/s):")
    print(f"  ESKF only:  {np.mean(eskf_persistent):.2f} ± {np.std(eskf_persistent):.2f}°")
    print(f"  Redundant:  {np.mean(red_persistent):.2f} ± {np.std(red_persistent):.2f}°")
    print(f"  Improvement: {np.mean(improvement_persistent):.1f} ± {np.std(improvement_persistent):.1f}%")

    print(f"\nTransient Fault (50 mrad/s for 30s):")
    print(f"  ESKF only:  {np.mean(eskf_transient):.2f} ± {np.std(eskf_transient):.2f}°")
    print(f"  Redundant:  {np.mean(red_transient):.2f} ± {np.std(red_transient):.2f}°")
    print(f"  Improvement: {np.mean(improvement_transient):.1f} ± {np.std(improvement_transient):.1f}%")

    delays_persistent = [r.detection_delay for r in results_persistent if r.detection_delay is not None]
    delays_transient = [r.detection_delay for r in results_transient if r.detection_delay is not None]

    print(f"\nDetection Performance:")
    print(f"  Persistent: {len(delays_persistent)}/{n_trials} detected, delay = {np.mean(delays_persistent):.1f} ± {np.std(delays_persistent):.1f}s")
    print(f"  Transient:  {len(delays_transient)}/{n_trials} detected, delay = {np.mean(delays_transient):.1f} ± {np.std(delays_transient):.1f}s")

    print("="*70)


if __name__ == "__main__":
    main()
