#!/usr/bin/env python3
"""
Fault Detection with Persistent Sun Sensor Error.

Tests fault detection when sun sensor has a persistent bias from t=0,
simulating a calibration error or hard-iron type bias.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from estimation.redundant_estimator import RedundantEstimator

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['lines.linewidth'] = 1.5


@dataclass
class FaultScenario:
    name: str
    sensor: str
    fault_type: str
    magnitude: float
    start_time: float = 0.0  # Persistent from start


@dataclass
class DetectionResult:
    detected: bool
    detection_time: Optional[float]
    max_disagreement: float
    mean_disagreement: float
    disagreement_history: np.ndarray
    time_history: np.ndarray


def make_star_tracker_sparse(sim_data, dropout_prob: float = 0.7, seed: int = None):
    """Make star tracker measurements sparse."""
    if seed is not None:
        np.random.seed(seed)

    class ModifiedData:
        pass

    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas',
                 'st_meas', 'b_eci', 's_eci', 'b_g_true']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.st_meas[k])):
            if np.random.random() < dropout_prob:
                modified.st_meas[k] = np.array([np.nan, np.nan, np.nan, np.nan])

    return modified


def inject_persistent_sun_bias(sim_data, bias_magnitude: float):
    """Inject persistent sun sensor bias from t=0."""
    class ModifiedData:
        pass

    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas',
                 'st_meas', 'b_eci', 's_eci', 'b_g_true']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    bias = np.array([bias_magnitude, 0.5*bias_magnitude, 0.3*bias_magnitude])

    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.sun_meas[k])):
            biased = modified.sun_meas[k] + bias
            modified.sun_meas[k] = biased / np.linalg.norm(biased)

    return modified


def inject_fault(sim_data, fault: FaultScenario, seed: int = None):
    """Inject a fault into simulation data."""
    if seed is not None:
        np.random.seed(seed)

    class ModifiedData:
        pass

    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas',
                 'st_meas', 'b_eci', 's_eci', 'b_g_true']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    for k in range(len(modified.t)):
        t = modified.t[k]
        if t < fault.start_time:
            continue

        if fault.sensor == 'mag' and not np.any(np.isnan(modified.mag_meas[k])):
            if fault.fault_type == 'bias':
                bias = np.array([fault.magnitude, 0.5*fault.magnitude, 0.3*fault.magnitude])
                biased = modified.mag_meas[k] + bias
                modified.mag_meas[k] = biased / np.linalg.norm(biased)

        elif fault.sensor == 'sun' and not np.any(np.isnan(modified.sun_meas[k])):
            if fault.fault_type == 'bias':
                bias = np.array([fault.magnitude, 0.5*fault.magnitude, 0.3*fault.magnitude])
                biased = modified.sun_meas[k] + bias
                modified.sun_meas[k] = biased / np.linalg.norm(biased)

        elif fault.sensor == 'star' and not np.any(np.isnan(modified.st_meas[k])):
            if fault.fault_type == 'bias':
                bias_axis = np.array([1, 0.5, 0.3]) * fault.magnitude
                q_bias = Quaternion.from_avec(bias_axis)
                q_orig = Quaternion.from_array(modified.st_meas[k])
                q_biased = (q_orig @ q_bias).normalize()
                modified.st_meas[k] = q_biased.as_array()

    return modified


def run_estimator(sim_data, config_path: str, fault: Optional[FaultScenario] = None,
                  threshold: float = 2.0) -> DetectionResult:
    """Run redundant estimator and track disagreement."""
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        disagreement_threshold_deg=threshold,
    )

    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    disagreements = []
    times = []
    detection_time = None
    fault_start = fault.start_time if fault else float('inf')

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]

        if np.any(np.isnan(omega)):
            omega = sim_data.omega_meas[k-1]

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_n = sim_data.b_eci[k] if z_mag is not None else None
        s_n = sim_data.s_eci[k] if z_sun is not None else None

        x_est, _, disagreement, _ = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        disagreements.append(disagreement)

        if t >= fault_start and detection_time is None:
            if disagreement > threshold:
                detection_time = t - fault_start

    disagreements = np.array(disagreements)
    times = np.array(times)

    # Compute mean over last 2/3 of simulation (after convergence)
    late_mask = times > times[-1] * 0.33
    mean_dis = np.mean(disagreements[late_mask])

    return DetectionResult(
        detected=detection_time is not None,
        detection_time=detection_time,
        max_disagreement=np.max(disagreements),
        mean_disagreement=mean_dis,
        disagreement_history=disagreements,
        time_history=times,
    )


def main():
    print("=" * 70)
    print("FAULT DETECTION WITH PERSISTENT SUN SENSOR ERROR")
    print("=" * 70)

    base_config = 'configs/config_baseline_short.yaml'
    st_dropout = 0.7

    # Test different persistent sun bias magnitudes
    sun_bias_magnitudes = [0.0, 0.1, 0.2, 0.3]

    # Load simulation
    print("\n1. Loading simulation data...")
    db = SimulationDatabase('simulations.db')
    sim_data_raw = db.load_run(237)
    print(f"   Loaded: {sim_data_raw.t[-1]:.1f}s simulation")

    # Make star tracker sparse
    print(f"\n2. Making star tracker sparse ({st_dropout*100:.0f}% dropout)...")
    sim_data_sparse = make_star_tracker_sparse(sim_data_raw, dropout_prob=st_dropout, seed=123)
    n_st = np.sum(~np.any(np.isnan(sim_data_sparse.st_meas), axis=1))
    print(f"   Star tracker available: {n_st}/{len(sim_data_sparse.t)} timesteps")

    # Store results for plotting
    results_by_sun_bias = {}

    for sun_bias in sun_bias_magnitudes:
        print(f"\n{'='*70}")
        print(f"Testing with persistent sun bias = {sun_bias}")
        print(f"{'='*70}")

        # Apply persistent sun bias
        if sun_bias > 0:
            sim_data_base = inject_persistent_sun_bias(sim_data_sparse, sun_bias)
        else:
            sim_data_base = sim_data_sparse

        # Run nominal (no additional fault)
        print("\n   Running nominal baseline...")
        nominal_result = run_estimator(sim_data_base, base_config, fault=None)
        print(f"   Nominal: mean={nominal_result.mean_disagreement:.3f}°, max={nominal_result.max_disagreement:.3f}°")

        results_by_sun_bias[sun_bias] = {
            'nominal': nominal_result,
            'mag_faults': {},
            'sun_faults': {},
        }

        # Test magnetometer faults on top of sun bias
        print("\n   Testing magnetometer faults...")
        mag_magnitudes = [0.1, 0.2, 0.3]
        for mag_mag in mag_magnitudes:
            fault = FaultScenario(
                name=f"mag_bias_{mag_mag}",
                sensor='mag',
                fault_type='bias',
                magnitude=mag_mag,
                start_time=100.0,
            )
            sim_data_faulty = inject_fault(sim_data_base, fault, seed=42)
            result = run_estimator(sim_data_faulty, base_config, fault)
            results_by_sun_bias[sun_bias]['mag_faults'][mag_mag] = result
            det_str = f"detected at {result.detection_time:.1f}s" if result.detected else "NOT detected"
            print(f"      mag bias={mag_mag}: {det_str}, max_dis={result.max_disagreement:.2f}°")

        # Test additional sun sensor faults (on top of persistent bias)
        print("\n   Testing additional sun sensor faults...")
        sun_magnitudes = [0.1, 0.2, 0.3]
        for sun_mag in sun_magnitudes:
            fault = FaultScenario(
                name=f"sun_bias_{sun_mag}",
                sensor='sun',
                fault_type='bias',
                magnitude=sun_mag,
                start_time=100.0,
            )
            sim_data_faulty = inject_fault(sim_data_base, fault, seed=42)
            result = run_estimator(sim_data_faulty, base_config, fault)
            results_by_sun_bias[sun_bias]['sun_faults'][sun_mag] = result
            det_str = f"detected at {result.detection_time:.1f}s" if result.detected else "NOT detected"
            print(f"      sun bias={sun_mag}: {det_str}, max_dis={result.max_disagreement:.2f}°")

    # Create comparison plot
    print("\n3. Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sun_bias_magnitudes)))

    # Plot 1: Disagreement over time for nominal cases
    ax1 = axes[0, 0]
    for i, sun_bias in enumerate(sun_bias_magnitudes):
        result = results_by_sun_bias[sun_bias]['nominal']
        label = f"Sun bias = {sun_bias}" if sun_bias > 0 else "No sun bias"
        ax1.plot(result.time_history, result.disagreement_history,
                color=colors[i], label=label, alpha=0.8)
    ax1.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Threshold (2°)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Disagreement [deg]')
    ax1.set_title('Nominal Disagreement with Persistent Sun Bias')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 15])

    # Plot 2: Disagreement with magnetometer fault
    ax2 = axes[0, 1]
    mag_fault_mag = 0.2
    for i, sun_bias in enumerate(sun_bias_magnitudes):
        result = results_by_sun_bias[sun_bias]['mag_faults'].get(mag_fault_mag)
        if result:
            label = f"Sun bias = {sun_bias}" if sun_bias > 0 else "No sun bias"
            ax2.plot(result.time_history, result.disagreement_history,
                    color=colors[i], label=label, alpha=0.8)
    ax2.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.axvline(100.0, color='gray', linestyle=':', alpha=0.5, label='Fault onset')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title(f'Magnetometer Fault (bias={mag_fault_mag}) Detection')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 15])

    # Plot 3: Disagreement with sun sensor fault
    ax3 = axes[1, 0]
    sun_fault_mag = 0.2
    for i, sun_bias in enumerate(sun_bias_magnitudes):
        result = results_by_sun_bias[sun_bias]['sun_faults'].get(sun_fault_mag)
        if result:
            label = f"Sun bias = {sun_bias}" if sun_bias > 0 else "No sun bias"
            ax3.plot(result.time_history, result.disagreement_history,
                    color=colors[i], label=label, alpha=0.8)
    ax3.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax3.axvline(100.0, color='gray', linestyle=':', alpha=0.5, label='Fault onset')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Disagreement [deg]')
    ax3.set_title(f'Additional Sun Fault (bias={sun_fault_mag}) Detection')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 15])

    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    x = np.arange(len(sun_bias_magnitudes))
    width = 0.25

    # Get mean disagreements
    nominal_means = [results_by_sun_bias[sb]['nominal'].mean_disagreement for sb in sun_bias_magnitudes]
    mag_fault_means = [results_by_sun_bias[sb]['mag_faults'][0.2].mean_disagreement for sb in sun_bias_magnitudes]
    sun_fault_means = [results_by_sun_bias[sb]['sun_faults'][0.2].mean_disagreement for sb in sun_bias_magnitudes]

    bars1 = ax4.bar(x - width, nominal_means, width, label='Nominal', color='C0')
    bars2 = ax4.bar(x, mag_fault_means, width, label='Mag fault (0.2)', color='C1')
    bars3 = ax4.bar(x + width, sun_fault_means, width, label='Sun fault (0.2)', color='C2')

    ax4.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax4.set_xlabel('Persistent Sun Bias')
    ax4.set_ylabel('Mean Disagreement [deg]')
    ax4.set_title('Mean Disagreement Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(sb) for sb in sun_bias_magnitudes])
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fault_detection_persistent_sun.png', dpi=150, bbox_inches='tight')
    plt.savefig('fault_detection_persistent_sun.pdf', dpi=150, bbox_inches='tight')
    print("   Saved to fault_detection_persistent_sun.png/pdf")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Persistent Sun Bias':<20} {'Nominal Mean':<15} {'Mag Fault Det.':<15} {'Sun Fault Det.':<15}")
    print("-" * 70)

    for sun_bias in sun_bias_magnitudes:
        nom_mean = results_by_sun_bias[sun_bias]['nominal'].mean_disagreement
        mag_det = results_by_sun_bias[sun_bias]['mag_faults'][0.2].detected
        sun_det = results_by_sun_bias[sun_bias]['sun_faults'][0.2].detected

        mag_str = "Yes" if mag_det else "No"
        sun_str = "Yes" if sun_det else "No"

        print(f"{sun_bias:<20} {nom_mean:<15.2f} {mag_str:<15} {sun_str:<15}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Check if persistent sun bias helps detection
    no_bias_sun_det = results_by_sun_bias[0.0]['sun_faults'][0.2].detected
    with_bias_sun_det = results_by_sun_bias[0.2]['sun_faults'][0.2].detected

    if with_bias_sun_det and not no_bias_sun_det:
        print("- Persistent sun bias ENABLES detection of additional sun faults")
    elif not with_bias_sun_det:
        print("- Sun sensor faults remain undetectable even with persistent bias")

    # Compare disagreement levels
    no_bias_nom = results_by_sun_bias[0.0]['nominal'].mean_disagreement
    high_bias_nom = results_by_sun_bias[0.3]['nominal'].mean_disagreement

    print(f"- Nominal disagreement increases from {no_bias_nom:.2f}° to {high_bias_nom:.2f}° with sun bias")

    return 0


if __name__ == "__main__":
    sys.exit(main())
