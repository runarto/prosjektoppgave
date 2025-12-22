#!/usr/bin/env python3
"""
Fast Fault Detection Analysis for Thesis Section 5.5.

Streamlined version with fewer Monte Carlo runs for quicker results.
Tests sparse star tracker scenario with bias faults.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from estimation.redundant_estimator import RedundantEstimator

# Publication-quality plot settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['lines.linewidth'] = 1.5

@dataclass
class FaultScenario:
    name: str
    sensor: str  # 'mag', 'sun', 'star'
    fault_type: str  # 'bias', 'stuck'
    magnitude: float
    start_time: float = 100.0

@dataclass
class DetectionResult:
    detected: bool
    detection_time: Optional[float]
    max_disagreement: float
    mean_disagreement: float

@dataclass
class ROCPoint:
    threshold: float
    tpr: float
    fpr: float


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

        elif fault.sensor == 'sun' and not np.any(np.isnan(modified.sun_meas[k])):
            if fault.fault_type == 'bias':
                bias = np.array([fault.magnitude, 0.5*fault.magnitude, 0.3*fault.magnitude])
                biased = modified.sun_meas[k] + bias
                modified.sun_meas[k] = biased / np.linalg.norm(biased)
            elif fault.fault_type == 'stuck':
                if stuck_value is None:
                    stuck_value = modified.sun_meas[k].copy()
                modified.sun_meas[k] = stuck_value

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
    """Run redundant estimator and check for fault detection."""
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

    if fault:
        post_fault_mask = times >= fault_start
        mean_dis = np.mean(disagreements[post_fault_mask]) if np.any(post_fault_mask) else 0
    else:
        mean_dis = np.mean(disagreements[times > times[-1] * 0.33])

    return DetectionResult(
        detected=detection_time is not None,
        detection_time=detection_time,
        max_disagreement=np.max(disagreements),
        mean_disagreement=mean_dis,
    )


def run_mdf_analysis(sim_data_base, config_path: str, sensor: str,
                     magnitudes: np.ndarray, n_runs: int = 5) -> Dict:
    """Find Minimum Detectable Fault (MDF) for a sensor."""
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
            sim_data = inject_fault(sim_data_base, fault, seed=run_idx * 42)
            result = run_estimator(sim_data, config_path, fault)

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
        }

        delay_str = f"{mean_delay:.1f}s" if mean_delay else "N/A"
        print(f"    {sensor} bias={mag:.3f}: {detection_rate:.0f}% ({detections}/{n_runs}), delay={delay_str}")

    return results


def compute_roc_curve(nominal_results: List[DetectionResult],
                      faulty_results: List[DetectionResult],
                      thresholds: np.ndarray) -> List[ROCPoint]:
    """Compute ROC curve from nominal and faulty run results."""
    roc_points = []

    for thresh in thresholds:
        fp_count = sum(1 for r in nominal_results if r.max_disagreement > thresh)
        fpr = fp_count / len(nominal_results) if nominal_results else 0

        tp_count = sum(1 for r in faulty_results if r.max_disagreement > thresh)
        tpr = tp_count / len(faulty_results) if faulty_results else 0

        roc_points.append(ROCPoint(threshold=thresh, tpr=tpr, fpr=fpr))

    return roc_points


def plot_results(mdf_results: Dict, roc_data: Dict, save_prefix: str):
    """Generate combined results plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'mag': 'C0', 'sun': 'C1', 'star': 'C2'}
    labels = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'star': 'Star Tracker'}

    # Plot 1: Detection rate vs magnitude (vector sensors)
    ax1 = axes[0, 0]
    for sensor in ['mag', 'sun']:
        if sensor in mdf_results:
            results = mdf_results[sensor]
            mags = sorted(results.keys())
            rates = [results[m]['detection_rate'] for m in mags]
            ax1.plot(mags, rates, 'o-', color=colors[sensor], label=labels[sensor], linewidth=2, markersize=8)

    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax1.set_xlabel('Fault Magnitude (bias)')
    ax1.set_ylabel('Detection Rate [%]')
    ax1.set_title('Vector Sensor Fault Detection')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Plot 2: Detection rate for star tracker
    ax2 = axes[0, 1]
    if 'star' in mdf_results:
        results = mdf_results['star']
        mags = sorted(results.keys())
        mags_deg = [np.rad2deg(m) for m in mags]
        rates = [results[m]['detection_rate'] for m in mags]
        ax2.plot(mags_deg, rates, 'o-', color=colors['star'], label=labels['star'], linewidth=2, markersize=8)

    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax2.set_xlabel('Fault Magnitude [deg]')
    ax2.set_ylabel('Detection Rate [%]')
    ax2.set_title('Star Tracker Fault Detection')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    # Plot 3: ROC curves
    ax3 = axes[1, 0]
    for sensor, roc_points in roc_data.items():
        fprs = [p.fpr for p in roc_points]
        tprs = [p.tpr for p in roc_points]
        sorted_idx = np.argsort(fprs)
        fprs_s = np.array(fprs)[sorted_idx]
        tprs_s = np.array(tprs)[sorted_idx]
        auc = np.trapz(tprs_s, fprs_s)
        ax3.plot(fprs_s, tprs_s, '-', color=colors[sensor],
                label=f'{labels[sensor]} (AUC={auc:.2f})', linewidth=2)

    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])

    # Plot 4: Detection delay vs magnitude
    ax4 = axes[1, 1]
    for sensor in ['mag', 'sun']:
        if sensor in mdf_results:
            results = mdf_results[sensor]
            mags = sorted(results.keys())
            delays = [results[m]['mean_delay'] for m in mags]
            valid = [(m, d) for m, d in zip(mags, delays) if d is not None]
            if valid:
                m_vals, d_vals = zip(*valid)
                ax4.plot(m_vals, d_vals, 's-', color=colors[sensor],
                        label=labels[sensor], linewidth=2, markersize=8)

    ax4.set_xlabel('Fault Magnitude (bias)')
    ax4.set_ylabel('Mean Detection Delay [s]')
    ax4.set_title('Detection Latency')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_prefix}.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_prefix}.png/pdf")
    plt.close()


def generate_latex_table(mdf_results: Dict) -> str:
    """Generate LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Minimum Detectable Fault Analysis (Sparse Star Tracker)}")
    lines.append(r"\label{tab:mdf_sparse_st}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Sensor & MDF$_{50\%}$ & MDF$_{90\%}$ & Max Det. Rate & Mean Delay \\")
    lines.append(r"\midrule")

    sensor_names = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'star': 'Star Tracker'}

    for sensor, results in mdf_results.items():
        magnitudes = sorted(results.keys())
        rates = [results[m]['detection_rate'] for m in magnitudes]

        mdf_50 = mdf_90 = None
        for m, r in zip(magnitudes, rates):
            if r >= 50 and mdf_50 is None:
                mdf_50 = m
            if r >= 90 and mdf_90 is None:
                mdf_90 = m

        max_rate = max(rates)
        max_mag = magnitudes[rates.index(max_rate)]
        mean_delay = results[max_mag].get('mean_delay')

        if sensor == 'star':
            mdf_50_str = f"{np.rad2deg(mdf_50):.1f}$^\\circ$" if mdf_50 else "$>$10$^\\circ$"
            mdf_90_str = f"{np.rad2deg(mdf_90):.1f}$^\\circ$" if mdf_90 else "$>$10$^\\circ$"
        else:
            mdf_50_str = f"{mdf_50:.2f}" if mdf_50 else "$>$0.5"
            mdf_90_str = f"{mdf_90:.2f}" if mdf_90 else "$>$0.5"

        delay_str = f"{mean_delay:.1f}s" if mean_delay else "N/A"

        lines.append(f"{sensor_names[sensor]} & {mdf_50_str} & {mdf_90_str} & {max_rate:.0f}\\% & {delay_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[0.5em]")
    lines.append(r"\footnotesize{Star tracker dropout: 70\%, threshold: 2.0$^\circ$}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("FAST FAULT DETECTION ANALYSIS")
    print("=" * 60)

    base_config = 'configs/config_baseline_short.yaml'
    n_mc_runs = 5  # Reduced for speed
    st_dropout = 0.7

    # Load existing simulation
    print("\n1. Loading simulation data...")
    db = SimulationDatabase('simulations.db')
    sim_data_raw = db.load_run(237)
    print(f"   Loaded: {sim_data_raw.t[-1]:.1f}s simulation")

    # Make star tracker sparse
    print(f"\n2. Making star tracker sparse ({st_dropout*100:.0f}% dropout)...")
    sim_data_base = make_star_tracker_sparse(sim_data_raw, dropout_prob=st_dropout, seed=123)
    n_st = np.sum(~np.any(np.isnan(sim_data_base.st_meas), axis=1))
    print(f"   Star tracker available: {n_st}/{len(sim_data_base.t)} timesteps")

    # Run nominal baseline
    print("\n3. Running nominal baseline...")
    nominal_results = []
    for i in range(n_mc_runs):
        result = run_estimator(sim_data_base, base_config, fault=None)
        nominal_results.append(result)
    print(f"   Nominal max disagreement: {np.max([r.max_disagreement for r in nominal_results]):.3f} deg")

    # MDF Analysis
    print("\n4. MDF Analysis...")

    # Reduced magnitude sets
    vector_mags = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    star_mags = np.deg2rad(np.array([1.0, 2.0, 3.0, 5.0, 7.0]))

    mdf_results = {}

    print("\n   Magnetometer:")
    mdf_results['mag'] = run_mdf_analysis(sim_data_base, base_config, 'mag', vector_mags, n_mc_runs)

    print("\n   Sun sensor:")
    mdf_results['sun'] = run_mdf_analysis(sim_data_base, base_config, 'sun', vector_mags, n_mc_runs)

    print("\n   Star tracker:")
    mdf_results['star'] = run_mdf_analysis(sim_data_base, base_config, 'star', star_mags, n_mc_runs)

    # ROC Analysis
    print("\n5. ROC Analysis...")
    thresholds = np.linspace(0.5, 4.0, 20)
    roc_data = {}

    for sensor in ['mag', 'sun', 'star']:
        fault_mag = np.deg2rad(3.0) if sensor == 'star' else 0.25
        fault = FaultScenario(name=f"{sensor}_roc", sensor=sensor, fault_type='bias', magnitude=fault_mag)

        faulty_results = []
        for i in range(n_mc_runs):
            sim_data = inject_fault(sim_data_base, fault, seed=i * 42)
            result = run_estimator(sim_data, base_config, fault, threshold=10.0)
            faulty_results.append(result)

        roc_data[sensor] = compute_roc_curve(nominal_results, faulty_results, thresholds)

        # Compute AUC
        fprs = [p.fpr for p in roc_data[sensor]]
        tprs = [p.tpr for p in roc_data[sensor]]
        sorted_idx = np.argsort(fprs)
        auc = np.trapz(np.array(tprs)[sorted_idx], np.array(fprs)[sorted_idx])
        print(f"   {sensor}: AUC = {auc:.3f}")

    # Generate outputs
    print("\n6. Generating outputs...")
    plot_results(mdf_results, roc_data, 'fault_detection_results')

    latex_table = generate_latex_table(mdf_results)
    with open('fault_detection_mdf.tex', 'w') as f:
        f.write(latex_table)
    print("   Saved LaTeX table to fault_detection_mdf.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTest configuration: ST dropout = {st_dropout*100:.0f}%, MC runs = {n_mc_runs}")
    print("\nMinimum Detectable Fault (50% detection):")
    for sensor, results in mdf_results.items():
        mags = sorted(results.keys())
        rates = [results[m]['detection_rate'] for m in mags]
        mdf_50 = None
        for m, r in zip(mags, rates):
            if r >= 50:
                mdf_50 = m
                break
        if sensor == 'star':
            mdf_str = f"{np.rad2deg(mdf_50):.1f} deg" if mdf_50 else ">7 deg"
        else:
            mdf_str = f"{mdf_50:.2f}" if mdf_50 else ">0.5"
        print(f"  {sensor:12s}: {mdf_str}")

    print("\nOutput files:")
    print("  - fault_detection_results.png/pdf")
    print("  - fault_detection_mdf.tex")

    return 0


if __name__ == "__main__":
    sys.exit(main())
