#!/usr/bin/env python3
"""
Improved Monte Carlo Fault Detection Analysis.

Changes:
1. Better plot readability (larger fonts, clearer colors)
2. Investigate oscillation issue
3. Compare: switching vs always-smoother during fault
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

# Better plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


@dataclass
class TrialResult:
    times: np.ndarray
    eskf_only_errors: np.ndarray
    redundant_selected_errors: np.ndarray
    smoother_always_errors: np.ndarray  # What if we just use smoother?
    disagreements: np.ndarray
    primaries: List[str]

    eskf_only_post_fault_mean: float
    redundant_post_fault_mean: float
    smoother_always_post_fault_mean: float
    detection_delay: Optional[float]
    n_switches: int

    fault_onset: float
    fault_duration: Optional[float]


def run_single_trial(sim_data, config_path: str, fault_config: Dict) -> TrialResult:
    """Run a single trial comparing ESKF-only, Redundant, and Smoother-always."""

    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    # ESKF only
    eskf = ESKF(P0=P0.copy(), config_path=config_path)
    x_eskf_only = EskfState(
        nom=NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    # Redundant (with switching)
    redundant = RedundantEstimator(
        P0=P0.copy(), config_path=config_path,
        smoother_lag=60.0, use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )
    x_redundant = EskfState(
        nom=NominalState(ori=q0_est.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times, eskf_only_errors, redundant_selected, smoother_always = [], [], [], []
    disagreements, primaries = [], []

    fault_onset = fault_config['onset_time']
    fault_duration = fault_config.get('duration', None)
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

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        # ESKF only
        x_eskf_only = eskf.predict(x_eskf_only, omega_k, dt_k)
        if z_mag is not None:
            try: x_eskf_only = eskf.update(x_eskf_only, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except ValueError: pass
        if z_sun is not None:
            try: x_eskf_only = eskf.update(x_eskf_only, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except ValueError: pass
        if z_st is not None:
            try: x_eskf_only = eskf.update(x_eskf_only, z_st, SensorType.STAR_TRACKER)
            except ValueError: pass

        # Redundant
        x_redundant, smoother_state, disagreement_deg, primary_str = redundant.step(
            x_eskf=x_redundant, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        if first_detection_time is None and t >= fault_onset and disagreement_deg > 2.0:
            first_detection_time = t

        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_eskf_only.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true)
        selected_err = smoother_err if primary_str == 'SMOOTHER' else compute_attitude_error(x_redundant.nom.ori, q_true)

        times.append(t)
        eskf_only_errors.append(eskf_err)
        redundant_selected.append(selected_err)
        smoother_always.append(smoother_err)  # What if we always use smoother?
        disagreements.append(disagreement_deg)
        primaries.append(primary_str)

    times = np.array(times)

    # Post-fault metrics
    if fault_duration is None:
        post_mask = times >= fault_onset + 30
    else:
        post_mask = times >= fault_onset + fault_duration

    stats = redundant.get_statistics()
    n_switches = len([e for e in stats.get('switch_events', []) if e[0] >= fault_onset])

    return TrialResult(
        times=times,
        eskf_only_errors=np.array(eskf_only_errors),
        redundant_selected_errors=np.array(redundant_selected),
        smoother_always_errors=np.array(smoother_always),
        disagreements=np.array(disagreements),
        primaries=primaries,
        eskf_only_post_fault_mean=np.mean(np.array(eskf_only_errors)[post_mask]),
        redundant_post_fault_mean=np.mean(np.array(redundant_selected)[post_mask]),
        smoother_always_post_fault_mean=np.mean(np.array(smoother_always)[post_mask]),
        detection_delay=first_detection_time - fault_onset if first_detection_time else None,
        n_switches=n_switches,
        fault_onset=fault_onset,
        fault_duration=fault_duration,
    )


def run_monte_carlo(config_path: str, db_path: str, n_trials: int, fault_config: Dict) -> List[TrialResult]:
    db = SimulationDatabase(db_path)
    config = load_yaml(config_path)
    from data.generator_enhanced import EnhancedAttitudeDataGenerator

    results = []
    for i in range(n_trials):
        if (i + 1) % 5 == 0:
            print(f"  Trial {i+1}/{n_trials}")

        generator = EnhancedAttitudeDataGenerator(db_path=db_path, config_path=config_path)
        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd']
        )
        sim_id = generator.run(sim_cfg)
        sim_data = db.load_run(sim_id)

        result = run_single_trial(sim_data, config_path, fault_config)
        results.append(result)

    return results


def plot_results(results_persistent: List[TrialResult],
                 results_transient: List[TrialResult],
                 save_path: str = None):
    """Create improved, readable plots."""

    fig = plt.figure(figsize=(14, 10))

    # Colors
    c_eskf = '#E74C3C'      # Red
    c_redundant = '#3498DB'  # Blue
    c_smoother = '#27AE60'   # Green

    # === Plot 1: Persistent fault - time evolution ===
    ax1 = fig.add_subplot(2, 2, 1)

    t_common = results_persistent[0].times
    t_start = 100
    mask = t_common >= t_start
    fault_onset = results_persistent[0].fault_onset

    # Stack and compute mean/std
    eskf_stack = np.array([r.eskf_only_errors for r in results_persistent])
    red_stack = np.array([r.redundant_selected_errors for r in results_persistent])
    smooth_stack = np.array([r.smoother_always_errors for r in results_persistent])

    for stack, color, label in [
        (eskf_stack, c_eskf, 'ESKF only'),
        (smooth_stack, c_smoother, 'Smoother (always)'),
        (red_stack, c_redundant, 'Redundant (switching)'),
    ]:
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        ax1.semilogy(t_common[mask], mean[mask], color=color, linewidth=2, label=label)
        ax1.fill_between(t_common[mask],
                         np.maximum(mean[mask] - std[mask], 0.01),
                         mean[mask] + std[mask],
                         color=color, alpha=0.15)

    ax1.axvline(fault_onset, color='black', linestyle='--', linewidth=1.5, label='Fault onset')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_title(f'Persistent Fault (30 mrad/s)\nN={len(results_persistent)} trials, mean ± std')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.005, 100])
    ax1.set_xlim([t_start, t_common[-1]])

    # === Plot 2: Transient fault - time evolution ===
    ax2 = fig.add_subplot(2, 2, 2)

    t_common = results_transient[0].times
    fault_onset = results_transient[0].fault_onset
    fault_end = fault_onset + results_transient[0].fault_duration

    eskf_stack = np.array([r.eskf_only_errors for r in results_transient])
    red_stack = np.array([r.redundant_selected_errors for r in results_transient])
    smooth_stack = np.array([r.smoother_always_errors for r in results_transient])

    for stack, color, label in [
        (eskf_stack, c_eskf, 'ESKF only'),
        (smooth_stack, c_smoother, 'Smoother (always)'),
        (red_stack, c_redundant, 'Redundant (switching)'),
    ]:
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        ax2.semilogy(t_common[mask], mean[mask], color=color, linewidth=2, label=label)
        ax2.fill_between(t_common[mask],
                         np.maximum(mean[mask] - std[mask], 0.01),
                         mean[mask] + std[mask],
                         color=color, alpha=0.15)

    ax2.axvspan(fault_onset, fault_end, color='red', alpha=0.15, label='Fault active')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Attitude Error [deg]')
    ax2.set_title(f'Transient Fault (50 mrad/s for 30s)\nN={len(results_transient)} trials, mean ± std')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.005, 100])
    ax2.set_xlim([t_start, t_common[-1]])

    # === Plot 3: Bar chart comparison ===
    ax3 = fig.add_subplot(2, 2, 3)

    # Collect data
    scenarios = ['Persistent\nFault', 'Transient\nFault']
    eskf_means = [np.mean([r.eskf_only_post_fault_mean for r in results_persistent]),
                  np.mean([r.eskf_only_post_fault_mean for r in results_transient])]
    eskf_stds = [np.std([r.eskf_only_post_fault_mean for r in results_persistent]),
                 np.std([r.eskf_only_post_fault_mean for r in results_transient])]
    red_means = [np.mean([r.redundant_post_fault_mean for r in results_persistent]),
                 np.mean([r.redundant_post_fault_mean for r in results_transient])]
    red_stds = [np.std([r.redundant_post_fault_mean for r in results_persistent]),
                np.std([r.redundant_post_fault_mean for r in results_transient])]
    smooth_means = [np.mean([r.smoother_always_post_fault_mean for r in results_persistent]),
                    np.mean([r.smoother_always_post_fault_mean for r in results_transient])]
    smooth_stds = [np.std([r.smoother_always_post_fault_mean for r in results_persistent]),
                   np.std([r.smoother_always_post_fault_mean for r in results_transient])]

    x = np.arange(len(scenarios))
    width = 0.25

    bars1 = ax3.bar(x - width, eskf_means, width, yerr=eskf_stds, label='ESKF only',
                    color=c_eskf, capsize=5, edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x, smooth_means, width, yerr=smooth_stds, label='Smoother (always)',
                    color=c_smoother, capsize=5, edgecolor='black', linewidth=1)
    bars3 = ax3.bar(x + width, red_means, width, yerr=red_stds, label='Redundant (switching)',
                    color=c_redundant, capsize=5, edgecolor='black', linewidth=1)

    ax3.set_ylabel('Mean Post-Fault Error [deg]')
    ax3.set_title('Post-Fault Error Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}°',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # === Plot 4: Switching analysis ===
    ax4 = fig.add_subplot(2, 2, 4)

    # Count switches per trial
    switches_persistent = [r.n_switches for r in results_persistent]
    switches_transient = [r.n_switches for r in results_transient]

    # Histogram
    bins = np.arange(0, max(max(switches_persistent), max(switches_transient)) + 2) - 0.5
    ax4.hist(switches_persistent, bins=bins, alpha=0.7, label='Persistent fault',
             color='orange', edgecolor='black')
    ax4.hist(switches_transient, bins=bins, alpha=0.7, label='Transient fault',
             color='purple', edgecolor='black')

    ax4.axvline(np.mean(switches_persistent), color='orange', linestyle='--', linewidth=2,
                label=f'Persistent mean: {np.mean(switches_persistent):.1f}')
    ax4.axvline(np.mean(switches_transient), color='purple', linestyle='--', linewidth=2,
                label=f'Transient mean: {np.mean(switches_transient):.1f}')

    ax4.set_xlabel('Number of Switches (after fault onset)')
    ax4.set_ylabel('Count')
    ax4.set_title('Switching Frequency\n(This causes the oscillation!)')
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--config', type=str, default='configs/config_baseline_short.yaml')
    parser.add_argument('--db', type=str, default='simulations.db')
    args = parser.parse_args()

    print("="*70)
    print(f"IMPROVED MONTE CARLO ANALYSIS (N={args.trials})")
    print("="*70)

    persistent_config = {
        'onset_time': 150.0,
        'magnitude': 0.03,
        'duration': None,
    }

    transient_config = {
        'onset_time': 120.0,
        'magnitude': 0.05,
        'duration': 30.0,
    }

    start_time = time.time()

    print(f"\n1. Running PERSISTENT fault ({args.trials} trials)...")
    results_persistent = run_monte_carlo(args.config, args.db, args.trials, persistent_config)

    print(f"\n2. Running TRANSIENT fault ({args.trials} trials)...")
    results_transient = run_monte_carlo(args.config, args.db, args.trials, transient_config)

    print(f"\nTotal time: {time.time() - start_time:.1f}s")

    print("\nGenerating plots...")
    plot_results(results_persistent, results_transient, 'fault_mc_improved.pdf')

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Why does redundant oscillate?")
    print("="*70)

    switches_p = [r.n_switches for r in results_persistent]
    switches_t = [r.n_switches for r in results_transient]

    print(f"\nPersistent fault: {np.mean(switches_p):.1f} ± {np.std(switches_p):.1f} switches")
    print(f"Transient fault:  {np.mean(switches_t):.1f} ± {np.std(switches_t):.1f} switches")

    print("""
The oscillation happens because:
1. ESKF diverges → disagreement > 2° → switch to smoother
2. ESKF gets RESET to smoother state → disagreement ≈ 0
3. After 10 agreements (< 0.5°) → switch BACK to ESKF
4. ESKF immediately diverges again (fault still present!)
5. Repeat...

This rapid switching causes the oscillation in the "redundant" curve.

SOLUTION: For persistent faults, should STAY on smoother, not keep switching.
The current hysteresis isn't long enough.
""")

    print("\nPost-fault errors:")
    print(f"{'Scenario':<20} {'ESKF':<12} {'Smoother':<12} {'Redundant':<12}")
    print("-"*56)

    for name, results in [('Persistent', results_persistent), ('Transient', results_transient)]:
        eskf = np.mean([r.eskf_only_post_fault_mean for r in results])
        smooth = np.mean([r.smoother_always_post_fault_mean for r in results])
        red = np.mean([r.redundant_post_fault_mean for r in results])
        print(f"{name:<20} {eskf:>10.2f}° {smooth:>10.2f}° {red:>10.2f}°")

    print("\n" + "="*70)
    print("KEY INSIGHT: 'Smoother (always)' is best for persistent faults!")
    print("The switching mechanism hurts performance when fault persists.")
    print("="*70)


if __name__ == "__main__":
    main()
