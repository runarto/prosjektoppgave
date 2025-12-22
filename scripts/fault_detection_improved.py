#!/usr/bin/env python3
"""
Improved Fault Detection Analysis for Thesis Section 5.5.

This script addresses the weakness in fault detection analysis by:
1. Testing fault detection with sparse star tracker (rapid tumbling scenario)
2. Testing wider range of fault magnitudes to find Minimum Detectable Fault (MDF)
3. Generating ROC curves with proper false alarm rate analysis
4. Creating comprehensive tables for thesis inclusion

Key insight: When star tracker is available frequently, it dominates the
attitude solution and masks magnetometer/sun sensor faults. Testing with
sparse star tracker measurements reveals the true fault detection capability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import sys

from data.db import SimulationDatabase
from data.classes import SimulationConfig
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from estimation.redundant_estimator import RedundantEstimator
from utilities.utils import load_yaml

# Lazy import for generator
EnhancedAttitudeDataGenerator = None

def get_generator():
    global EnhancedAttitudeDataGenerator
    if EnhancedAttitudeDataGenerator is None:
        from data.generator_enhanced import EnhancedAttitudeDataGenerator as Gen
        EnhancedAttitudeDataGenerator = Gen
    return EnhancedAttitudeDataGenerator

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
class FaultScenario:
    """Configuration for a fault injection scenario."""
    name: str
    sensor: str  # 'mag', 'sun', 'star'
    fault_type: str  # 'bias', 'stuck', 'noise'
    magnitude: float
    start_time: float = 100.0


@dataclass
class DetectionResult:
    """Result from a single fault detection run."""
    detected: bool
    detection_time: Optional[float]  # Time after fault onset
    max_disagreement: float
    mean_disagreement_post_fault: float
    switched_to_smoother: bool
    switch_time: Optional[float]


@dataclass
class ROCPoint:
    """Single point on ROC curve."""
    threshold: float
    tpr: float  # True Positive Rate
    fpr: float  # False Positive Rate
    mean_delay: float


def create_sparse_st_config(base_config_path: str, st_dropout_prob: float = 0.7) -> str:
    """Create config with sparse star tracker measurements."""
    config = load_yaml(base_config_path)

    # Increase star tracker dropout to simulate sparse measurements
    config['sensors']['star']['dropout']['probability'] = st_dropout_prob
    config['sensors']['star']['dt'] = 10.0  # Also reduce measurement rate

    # Keep other sensors at baseline
    config['simulation']['run_name'] = f'sparse_st_dropout_{int(st_dropout_prob*100)}'

    output_path = 'configs/config_sparse_st_fault_test.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def inject_fault(sim_data, fault: FaultScenario, seed: int = None):
    """Inject a persistent fault into simulation data."""
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

    stuck_value = None

    for k in range(len(modified.t)):
        t = modified.t[k]

        if t < fault.start_time:
            continue

        if fault.sensor == 'mag' and not np.any(np.isnan(modified.mag_meas[k])):
            if fault.fault_type == 'bias':
                bias = np.array([fault.magnitude, 0.5*fault.magnitude, 0.3*fault.magnitude])
                biased = modified.mag_meas[k] + bias
                modified.mag_meas[k] = biased / np.linalg.norm(biased)
            elif fault.fault_type == 'stuck':
                if stuck_value is None:
                    stuck_value = modified.mag_meas[k].copy()
                modified.mag_meas[k] = stuck_value
            elif fault.fault_type == 'noise':
                noise = np.random.randn(3) * fault.magnitude
                noisy = modified.mag_meas[k] + noise
                modified.mag_meas[k] = noisy / np.linalg.norm(noisy)

        elif fault.sensor == 'sun' and not np.any(np.isnan(modified.sun_meas[k])):
            if fault.fault_type == 'bias':
                bias = np.array([fault.magnitude, 0.5*fault.magnitude, 0.3*fault.magnitude])
                biased = modified.sun_meas[k] + bias
                modified.sun_meas[k] = biased / np.linalg.norm(biased)
            elif fault.fault_type == 'stuck':
                if stuck_value is None:
                    stuck_value = modified.sun_meas[k].copy()
                modified.sun_meas[k] = stuck_value
            elif fault.fault_type == 'noise':
                noise = np.random.randn(3) * fault.magnitude
                noisy = modified.sun_meas[k] + noise
                modified.sun_meas[k] = noisy / np.linalg.norm(noisy)

        elif fault.sensor == 'star' and not np.any(np.isnan(modified.st_meas[k])):
            if fault.fault_type == 'bias':
                bias_axis = np.array([1, 0.5, 0.3]) * fault.magnitude
                q_bias = Quaternion.from_avec(bias_axis)
                q_orig = Quaternion.from_array(modified.st_meas[k])
                q_biased = (q_orig @ q_bias).normalize()
                modified.st_meas[k] = q_biased.as_array()
            elif fault.fault_type == 'stuck':
                if stuck_value is None:
                    stuck_value = modified.st_meas[k].copy()
                modified.st_meas[k] = stuck_value

    return modified


def run_estimator(sim_data, config_path: str, fault: Optional[FaultScenario] = None,
                  disagreement_threshold: float = 2.0) -> DetectionResult:
    """Run redundant estimator and check for fault detection."""
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        disagreement_threshold_deg=disagreement_threshold,
    )

    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    disagreements = []
    times = []
    modes = []
    detection_time = None
    switch_time = None

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

        x_est, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        disagreements.append(disagreement)
        modes.append(primary)

        # Check for detection after fault onset
        if t >= fault_start and detection_time is None:
            if disagreement > disagreement_threshold:
                detection_time = t - fault_start

        # Check for mode switch
        if t >= fault_start and switch_time is None:
            if primary in ['SMOOTHER', 'CONSERVATIVE']:
                switch_time = t - fault_start

    disagreements = np.array(disagreements)
    times = np.array(times)

    # Compute post-fault statistics
    if fault:
        post_fault_mask = times >= fault_start
        mean_dis_post = np.mean(disagreements[post_fault_mask]) if np.any(post_fault_mask) else 0
        max_dis = np.max(disagreements)
    else:
        # For nominal runs, use last 2/3 of simulation
        late_mask = times > times[-1] * 0.33
        mean_dis_post = np.mean(disagreements[late_mask])
        max_dis = np.max(disagreements[late_mask])

    return DetectionResult(
        detected=detection_time is not None,
        detection_time=detection_time,
        max_disagreement=max_dis,
        mean_disagreement_post_fault=mean_dis_post,
        switched_to_smoother=switch_time is not None,
        switch_time=switch_time,
    )


def compute_roc_curve(nominal_results: List[DetectionResult],
                      faulty_results: List[DetectionResult],
                      thresholds: np.ndarray) -> List[ROCPoint]:
    """Compute ROC curve from nominal and faulty run results."""
    roc_points = []

    for thresh in thresholds:
        # False positive rate from nominal runs
        fp_count = sum(1 for r in nominal_results if r.max_disagreement > thresh)
        fpr = fp_count / len(nominal_results) if nominal_results else 0

        # True positive rate and detection delays from faulty runs
        tp_count = 0
        delays = []
        for r in faulty_results:
            # Re-check detection with this threshold
            if r.max_disagreement > thresh:
                tp_count += 1
                if r.detection_time is not None:
                    delays.append(r.detection_time)

        tpr = tp_count / len(faulty_results) if faulty_results else 0
        mean_delay = np.mean(delays) if delays else float('inf')

        roc_points.append(ROCPoint(
            threshold=thresh,
            tpr=tpr,
            fpr=fpr,
            mean_delay=mean_delay,
        ))

    return roc_points


def run_mdf_analysis(sim_data_base, config_path: str, sensor: str,
                     magnitudes: np.ndarray, n_runs: int = 20,
                     threshold: float = 2.0) -> Dict:
    """
    Find Minimum Detectable Fault (MDF) for a sensor.

    Returns detection rate at each fault magnitude.
    """
    results = {}

    for mag in magnitudes:
        fault = FaultScenario(
            name=f"{sensor}_bias_{mag:.3f}",
            sensor=sensor,
            fault_type='bias',
            magnitude=mag,
        )

        detections = 0
        delays = []

        for run_idx in range(n_runs):
            # Inject fault
            sim_data = inject_fault(sim_data_base, fault, seed=run_idx * 42)

            # Run estimator
            result = run_estimator(sim_data, config_path, fault,
                                   disagreement_threshold=threshold)

            if result.detected:
                detections += 1
                if result.detection_time is not None:
                    delays.append(result.detection_time)

        detection_rate = detections / n_runs * 100
        mean_delay = np.mean(delays) if delays else None

        results[mag] = {
            'detection_rate': detection_rate,
            'mean_delay': mean_delay,
            'n_detected': detections,
            'n_total': n_runs,
        }

        delay_str = f"{mean_delay:.1f}s" if mean_delay else "N/A"
        print(f"    {sensor} bias={mag:.3f}: {detection_rate:.0f}% detected, delay={delay_str}")

    return results


def plot_mdf_analysis(mdf_results: Dict[str, Dict], save_path: str):
    """Plot Minimum Detectable Fault analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'mag': 'C0', 'sun': 'C1', 'star': 'C2'}
    labels = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'star': 'Star Tracker'}

    # Plot 1: Detection rate vs fault magnitude
    ax1 = axes[0]
    for sensor, results in mdf_results.items():
        magnitudes = sorted(results.keys())
        rates = [results[m]['detection_rate'] for m in magnitudes]
        ax1.plot(magnitudes, rates, 'o-', color=colors[sensor],
                label=labels[sensor], linewidth=2, markersize=8)

        # Find 50% detection threshold (MDF)
        for i, (m, r) in enumerate(zip(magnitudes, rates)):
            if r >= 50:
                ax1.axvline(m, color=colors[sensor], linestyle=':', alpha=0.5)
                break

    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax1.set_xlabel('Fault Magnitude (bias)')
    ax1.set_ylabel('Detection Rate [%]')
    ax1.set_title('Fault Detection Rate vs Magnitude')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot 2: Detection delay vs fault magnitude
    ax2 = axes[1]
    for sensor, results in mdf_results.items():
        magnitudes = sorted(results.keys())
        delays = [results[m]['mean_delay'] for m in magnitudes]
        # Filter out None values
        valid = [(m, d) for m, d in zip(magnitudes, delays) if d is not None]
        if valid:
            mags, dels = zip(*valid)
            ax2.plot(mags, dels, 's-', color=colors[sensor],
                    label=labels[sensor], linewidth=2, markersize=8)

    ax2.set_xlabel('Fault Magnitude (bias)')
    ax2.set_ylabel('Mean Detection Delay [s]')
    ax2.set_title('Detection Latency vs Fault Magnitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved MDF analysis to {save_path}")
    plt.close()


def plot_roc_curves(roc_data: Dict[str, List[ROCPoint]], save_path: str):
    """Plot ROC curves for different sensors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'mag': 'C0', 'sun': 'C1', 'star': 'C2'}
    labels = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'star': 'Star Tracker'}

    # Plot 1: ROC curve
    ax1 = axes[0]
    for sensor, roc_points in roc_data.items():
        fprs = [p.fpr for p in roc_points]
        tprs = [p.tpr for p in roc_points]

        # Sort by FPR for proper curve
        sorted_idx = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[sorted_idx]
        tprs_sorted = np.array(tprs)[sorted_idx]

        # Compute AUC
        auc = np.trapz(tprs_sorted, fprs_sorted)

        ax1.plot(fprs_sorted, tprs_sorted, '-', color=colors[sensor],
                label=f'{labels[sensor]} (AUC={auc:.2f})', linewidth=2)
        ax1.fill_between(fprs_sorted, tprs_sorted, alpha=0.1, color=colors[sensor])

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves for Sensor Fault Detection')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: TPR vs threshold with delay
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    for sensor, roc_points in roc_data.items():
        thresholds = [p.threshold for p in roc_points]
        tprs = [p.tpr for p in roc_points]
        delays = [p.mean_delay if p.mean_delay != float('inf') else np.nan
                  for p in roc_points]

        ax2.plot(thresholds, tprs, '-', color=colors[sensor],
                label=f'{labels[sensor]} TPR', linewidth=2)
        ax2_twin.plot(thresholds, delays, '--', color=colors[sensor],
                     alpha=0.7, linewidth=1.5)

    ax2.set_xlabel('Detection Threshold [deg]')
    ax2.set_ylabel('True Positive Rate', color='black')
    ax2_twin.set_ylabel('Detection Delay [s]', color='gray')
    ax2.set_title('Detection Performance vs Threshold')
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved ROC curves to {save_path}")
    plt.close()


def generate_latex_table(mdf_results: Dict[str, Dict], threshold: float) -> str:
    """Generate LaTeX table for thesis."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Minimum Detectable Fault (MDF) Analysis with Sparse Star Tracker}")
    lines.append(r"\label{tab:mdf_analysis}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Sensor & MDF$_{50\%}$ & MDF$_{90\%}$ & Det. Rate at 0.1 & Det. Rate at 0.2 & Mean Delay \\")
    lines.append(r"\midrule")

    sensor_names = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'star': 'Star Tracker'}

    for sensor, results in mdf_results.items():
        magnitudes = sorted(results.keys())
        rates = [results[m]['detection_rate'] for m in magnitudes]

        # Find MDF at 50% and 90%
        mdf_50 = None
        mdf_90 = None
        for m, r in zip(magnitudes, rates):
            if r >= 50 and mdf_50 is None:
                mdf_50 = m
            if r >= 90 and mdf_90 is None:
                mdf_90 = m

        # Get rates at specific magnitudes
        rate_01 = results.get(0.1, {}).get('detection_rate', '-')
        rate_02 = results.get(0.2, {}).get('detection_rate', '-')

        # Get mean delay at highest tested magnitude
        max_mag = max(magnitudes)
        mean_delay = results[max_mag].get('mean_delay')
        delay_str = f"{mean_delay:.1f}s" if mean_delay else "N/A"

        mdf_50_str = f"{mdf_50:.2f}" if mdf_50 else "$>$0.5"
        mdf_90_str = f"{mdf_90:.2f}" if mdf_90 else "$>$0.5"
        rate_01_str = f"{rate_01:.0f}\\%" if isinstance(rate_01, (int, float)) else rate_01
        rate_02_str = f"{rate_02:.0f}\\%" if isinstance(rate_02, (int, float)) else rate_02

        lines.append(f"{sensor_names[sensor]} & {mdf_50_str} & {mdf_90_str} & {rate_01_str} & {rate_02_str} & {delay_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\\\[0.5em]")
    lines.append(f"\\footnotesize{{Detection threshold: {threshold}°, Star tracker dropout: 70\\%}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def make_star_tracker_sparse(sim_data, dropout_prob: float = 0.7, seed: int = None):
    """
    Make star tracker measurements sparse by introducing dropouts.

    This simulates a scenario where star tracker is unavailable frequently,
    e.g., due to rapid tumbling or interference.
    """
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

    # Drop star tracker measurements randomly
    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.st_meas[k])):
            if np.random.random() < dropout_prob:
                modified.st_meas[k] = np.array([np.nan, np.nan, np.nan, np.nan])

    return modified


def main():
    print("=" * 70)
    print("IMPROVED FAULT DETECTION ANALYSIS")
    print("For Thesis Section 5.5")
    print("=" * 70)

    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_mc_runs = 15  # Monte Carlo runs per configuration
    st_dropout_prob = 0.7  # 70% star tracker dropout

    # Use existing simulation
    print("\n1. Loading existing simulation data...")
    db = SimulationDatabase(db_path)
    sim_id = 237  # Use existing simulation
    sim_data_raw = db.load_run(sim_id)
    print(f"   Loaded simulation: {sim_data_raw.t[-1]:.1f}s, ID={sim_id}")

    # Make star tracker sparse
    print(f"\n2. Making star tracker sparse (dropout={st_dropout_prob*100:.0f}%)...")
    sim_data_base = make_star_tracker_sparse(sim_data_raw, dropout_prob=st_dropout_prob, seed=123)

    # Count available measurements
    n_st_available = np.sum(~np.any(np.isnan(sim_data_base.st_meas), axis=1))
    n_total = len(sim_data_base.t)
    print(f"   Star tracker available: {n_st_available}/{n_total} ({n_st_available/n_total*100:.1f}%)")

    sparse_config = base_config  # Use baseline config for estimator

    # Run nominal baseline (no fault)
    print("\n3. Running nominal baseline...")
    nominal_results = []
    for i in range(n_mc_runs):
        result = run_estimator(sim_data_base, sparse_config, fault=None)
        nominal_results.append(result)

    mean_nominal_dis = np.mean([r.mean_disagreement_post_fault for r in nominal_results])
    max_nominal_dis = np.max([r.max_disagreement for r in nominal_results])
    print(f"   Nominal: mean disagreement={mean_nominal_dis:.4f}°, max={max_nominal_dis:.4f}°")

    # Minimum Detectable Fault Analysis
    print("\n4. Running Minimum Detectable Fault (MDF) analysis...")

    # Test magnitudes for vector sensors (mag, sun)
    vector_magnitudes = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    # Test magnitudes for star tracker (in radians)
    star_magnitudes = np.deg2rad(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]))

    mdf_results = {}

    print("\n   Magnetometer bias faults:")
    mdf_results['mag'] = run_mdf_analysis(
        sim_data_base, sparse_config, 'mag', vector_magnitudes, n_runs=n_mc_runs
    )

    print("\n   Sun sensor bias faults:")
    mdf_results['sun'] = run_mdf_analysis(
        sim_data_base, sparse_config, 'sun', vector_magnitudes, n_runs=n_mc_runs
    )

    print("\n   Star tracker bias faults:")
    mdf_results['star'] = run_mdf_analysis(
        sim_data_base, sparse_config, 'star', star_magnitudes, n_runs=n_mc_runs
    )

    # ROC Curve Analysis
    print("\n5. Computing ROC curves...")
    thresholds = np.linspace(0.5, 5.0, 30)
    roc_data = {}

    for sensor in ['mag', 'sun', 'star']:
        print(f"   {sensor}...")

        # Use medium fault magnitude for ROC analysis
        if sensor == 'star':
            fault_mag = np.deg2rad(3.0)
        else:
            fault_mag = 0.20

        fault = FaultScenario(
            name=f"{sensor}_roc_test",
            sensor=sensor,
            fault_type='bias',
            magnitude=fault_mag,
        )

        faulty_results = []
        for i in range(n_mc_runs):
            sim_data = inject_fault(sim_data_base, fault, seed=i * 42)
            # Run with different thresholds implicitly captured by max_disagreement
            result = run_estimator(sim_data, sparse_config, fault,
                                   disagreement_threshold=10.0)  # High threshold to capture all
            faulty_results.append(result)

        roc_data[sensor] = compute_roc_curve(nominal_results, faulty_results, thresholds)

    # Generate plots
    print("\n6. Generating plots...")
    plot_mdf_analysis(mdf_results, 'fault_mdf_analysis.png')
    plot_roc_curves(roc_data, 'fault_roc_curves.png')

    # Generate LaTeX table
    print("\n7. Generating LaTeX table...")
    latex_table = generate_latex_table(mdf_results, threshold=2.0)
    with open('fault_mdf_table.tex', 'w') as f:
        f.write(latex_table)
    print("   Saved to fault_mdf_table.tex")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nConfiguration: Star tracker dropout = {st_dropout_prob*100:.0f}%")
    print(f"Monte Carlo runs per configuration: {n_mc_runs}")
    print(f"\nMinimum Detectable Fault (50% detection rate):")

    for sensor, results in mdf_results.items():
        magnitudes = sorted(results.keys())
        rates = [results[m]['detection_rate'] for m in magnitudes]
        mdf_50 = None
        for m, r in zip(magnitudes, rates):
            if r >= 50:
                mdf_50 = m
                break

        if sensor == 'star':
            mdf_str = f"{np.rad2deg(mdf_50):.1f}°" if mdf_50 else ">10°"
        else:
            mdf_str = f"{mdf_50:.2f}" if mdf_50 else ">0.5"
        print(f"  {sensor:12s}: {mdf_str}")

    print("\nROC AUC values:")
    for sensor, roc_points in roc_data.items():
        fprs = [p.fpr for p in roc_points]
        tprs = [p.tpr for p in roc_points]
        sorted_idx = np.argsort(fprs)
        auc = np.trapz(np.array(tprs)[sorted_idx], np.array(fprs)[sorted_idx])
        print(f"  {sensor:12s}: {auc:.3f}")

    print("\nGenerated files:")
    print("  - fault_mdf_analysis.png/pdf")
    print("  - fault_roc_curves.png/pdf")
    print("  - fault_mdf_table.tex")
    print("  - configs/config_sparse_st_fault_test.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
