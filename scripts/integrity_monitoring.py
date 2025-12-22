#!/usr/bin/env python3
"""
Test 3: Integrity Monitoring Analysis.

Evaluates the Redundant estimator's false alarm and missed detection rates
across multiple scenarios to characterize its reliability as a fault monitor.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Tuple
from dataclasses import dataclass

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
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
class FaultConfig:
    """Fault configuration."""
    name: str
    start_time: float
    sensor: str
    fault_type: str
    magnitude: float


def inject_fault(sim_data, fault: FaultConfig, seed: int = None):
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

        if t >= fault.start_time:
            if fault.sensor == 'mag' and not np.any(np.isnan(modified.mag_meas[k])):
                if fault.fault_type == 'bias':
                    bias = np.array([fault.magnitude, 0, 0])
                    biased = modified.mag_meas[k] + bias
                    modified.mag_meas[k] = biased / np.linalg.norm(biased)
                elif fault.fault_type == 'stuck':
                    if stuck_value is None:
                        stuck_value = modified.mag_meas[k].copy()
                    modified.mag_meas[k] = stuck_value

            elif fault.sensor == 'star' and not np.any(np.isnan(modified.st_meas[k])):
                if fault.fault_type == 'bias':
                    bias_axis = np.array([1, 0, 0]) * fault.magnitude
                    q_bias = Quaternion.from_avec(bias_axis)
                    q_orig = Quaternion.from_array(modified.st_meas[k])
                    q_biased = (q_orig @ q_bias).normalize()
                    modified.st_meas[k] = q_biased.as_array()
                elif fault.fault_type == 'stuck':
                    if stuck_value is None:
                        stuck_value = modified.st_meas[k].copy()
                    modified.st_meas[k] = stuck_value

    return modified


def add_noise_variation(sim_data, noise_scale: float, seed: int = None):
    """Add noise variation to nominal data without faults."""
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

    # Add small noise variations
    for k in range(len(modified.t)):
        if not np.any(np.isnan(modified.mag_meas[k])):
            noise = np.random.randn(3) * noise_scale * 0.01
            modified.mag_meas[k] += noise
            modified.mag_meas[k] /= np.linalg.norm(modified.mag_meas[k])

        if not np.any(np.isnan(modified.sun_meas[k])):
            noise = np.random.randn(3) * noise_scale * 0.01
            modified.sun_meas[k] += noise
            modified.sun_meas[k] /= np.linalg.norm(modified.sun_meas[k])

    return modified


def run_redundant_estimator(sim_data, config_path: str, threshold: float = 2.0) -> Dict:
    """Run redundant estimator and track fault indicators."""
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

    times = []
    disagreements = []
    modes = []
    fault_flagged = False
    flag_time = None

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]

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

        # Check if fault was flagged (disagreement exceeds threshold)
        if not fault_flagged and disagreement > threshold:
            fault_flagged = True
            flag_time = t

    return {
        'times': np.array(times),
        'disagreements': np.array(disagreements),
        'modes': modes,
        'fault_flagged': fault_flagged,
        'flag_time': flag_time,
        'max_disagreement': np.max(disagreements),
    }


def run_integrity_analysis(sim_data_base, config_path: str, n_runs: int = 50) -> Dict:
    """
    Run comprehensive integrity analysis.

    Tests:
    1. Nominal runs - check for false alarms
    2. Faulty runs with detectable faults - check for missed detections
    """
    threshold = 2.0  # degrees
    results = {
        'nominal': {'runs': [], 'false_alarms': 0},
        'mag_bias_5pct': {'runs': [], 'missed': 0},
        'mag_bias_10pct': {'runs': [], 'missed': 0},
        'mag_stuck': {'runs': [], 'missed': 0},
        'star_stuck': {'runs': [], 'missed': 0},
    }

    faults = {
        'mag_bias_5pct': FaultConfig("Mag Bias 5%", 100.0, 'mag', 'bias', 0.05),
        'mag_bias_10pct': FaultConfig("Mag Bias 10%", 100.0, 'mag', 'bias', 0.10),
        'mag_stuck': FaultConfig("Mag Stuck", 100.0, 'mag', 'stuck', 0.0),
        'star_stuck': FaultConfig("Star Stuck", 100.0, 'star', 'stuck', 0.0),
    }

    # 1. Nominal runs (no faults - check false alarm rate)
    print("\n  Running nominal scenarios (false alarm analysis)...")
    for i in range(n_runs):
        # Add slight noise variation per run
        sim_data = add_noise_variation(sim_data_base, noise_scale=1.0, seed=i*42)
        run_result = run_redundant_estimator(sim_data, config_path, threshold)
        results['nominal']['runs'].append(run_result)

        # Check for false alarm (flagged but no fault present)
        if run_result['fault_flagged']:
            results['nominal']['false_alarms'] += 1

        if (i + 1) % 10 == 0:
            print(f"    Completed {i+1}/{n_runs} nominal runs")

    # 2. Faulty runs (check missed detection rate for detectable faults)
    for fault_key, fault in faults.items():
        print(f"\n  Running {fault.name} scenarios (missed detection analysis)...")
        for i in range(n_runs):
            sim_data = inject_fault(sim_data_base, fault, seed=i*42)
            run_result = run_redundant_estimator(sim_data, config_path, threshold)
            results[fault_key]['runs'].append(run_result)

            # Check for missed detection (fault present but not flagged after fault onset)
            if not run_result['fault_flagged'] or (run_result['flag_time'] and run_result['flag_time'] < fault.start_time):
                results[fault_key]['missed'] += 1

            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{n_runs} runs")

    return results


def plot_integrity_results(results: Dict, n_runs: int, save_path: str):
    """Plot integrity monitoring results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. False alarm rate (single bar)
    ax = axes[0, 0]
    false_alarm_rate = results['nominal']['false_alarms'] / n_runs * 100

    ax.bar(['Nominal\n(No Fault)'], [false_alarm_rate], color='steelblue', alpha=0.8, width=0.4)
    ax.axhline(5, color='red', linestyle='--', linewidth=1.5, label='5% Target')
    ax.set_ylabel('False Alarm Rate [%]')
    ax.set_title('False Alarm Rate')
    ax.set_ylim([0, max(20, false_alarm_rate + 5)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.text(0, false_alarm_rate + 1, f'{false_alarm_rate:.1f}%', ha='center', va='bottom', fontsize=14)

    # 2. Missed detection rates
    ax = axes[0, 1]
    fault_names = ['Mag Bias\n(5%)', 'Mag Bias\n(10%)', 'Mag\nStuck', 'Star\nStuck']
    fault_keys = ['mag_bias_5pct', 'mag_bias_10pct', 'mag_stuck', 'star_stuck']

    missed_rates = [results[k]['missed'] / n_runs * 100 for k in fault_keys]
    detection_rates = [100 - r for r in missed_rates]

    x = np.arange(len(fault_names))
    bars = ax.bar(x, detection_rates, color='green', alpha=0.7, label='Detected')
    ax.bar(x, missed_rates, bottom=detection_rates, color='red', alpha=0.7, label='Missed')

    ax.set_xticks(x)
    ax.set_xticklabels(fault_names)
    ax.set_ylabel('Rate [%]')
    ax.set_title('Detection Rate by Fault Type')
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Disagreement distribution - nominal vs faulty
    ax = axes[1, 0]

    nominal_max_disag = [r['max_disagreement'] for r in results['nominal']['runs']]
    mag_stuck_max_disag = [r['max_disagreement'] for r in results['mag_stuck']['runs']]

    bins = np.linspace(0, 20, 30)
    ax.hist(nominal_max_disag, bins=bins, alpha=0.6, label='Nominal', color='steelblue', density=True)
    ax.hist(mag_stuck_max_disag, bins=bins, alpha=0.6, label='Mag Stuck Fault', color='orange', density=True)
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xlabel('Maximum Disagreement [deg]')
    ax.set_ylabel('Density')
    ax.set_title('Disagreement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary metrics table
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate metrics
    total_faulty_runs = n_runs * 4  # 4 fault types
    total_detected = sum(n_runs - results[k]['missed'] for k in fault_keys)

    summary_data = [
        ['Total Runs (Nominal)', f'{n_runs}'],
        ['Total Runs (Faulty)', f'{total_faulty_runs}'],
        ['False Alarm Rate', f'{false_alarm_rate:.1f}%'],
        ['Overall Detection Rate', f'{total_detected/total_faulty_runs*100:.1f}%'],
        ['', ''],
        ['Mag Bias (5%) Detection', f'{100-results["mag_bias_5pct"]["missed"]/n_runs*100:.0f}%'],
        ['Mag Bias (10%) Detection', f'{100-results["mag_bias_10pct"]["missed"]/n_runs*100:.0f}%'],
        ['Mag Stuck Detection', f'{100-results["mag_stuck"]["missed"]/n_runs*100:.0f}%'],
        ['Star Stuck Detection', f'{100-results["star_stuck"]["missed"]/n_runs*100:.0f}%'],
    ]

    table = ax.table(
        cellText=summary_data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.6, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Integrity Monitoring Summary', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    n_runs = 50

    print("=" * 70)
    print("TEST 3: INTEGRITY MONITORING ANALYSIS")
    print("=" * 70)

    # Create config
    config = load_yaml(base_config)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    config_path = 'configs/config_integrity_test.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load simulation data
    print("\nLoading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    print(f"\nRunning integrity analysis ({n_runs} runs per scenario)...")
    results = run_integrity_analysis(sim_data_base, config_path, n_runs=n_runs)

    # Plot results
    plot_integrity_results(results, n_runs, 'integrity_monitoring.png')

    # Print summary
    print("\n" + "=" * 70)
    print("INTEGRITY MONITORING SUMMARY")
    print("=" * 70)

    false_alarm_rate = results['nominal']['false_alarms'] / n_runs * 100
    print(f"\nFalse Alarm Rate: {false_alarm_rate:.1f}% ({results['nominal']['false_alarms']}/{n_runs} runs)")
    print("\nDetection Rates:")
    for key in ['mag_bias_5pct', 'mag_bias_10pct', 'mag_stuck', 'star_stuck']:
        det_rate = (n_runs - results[key]['missed']) / n_runs * 100
        print(f"  {key}: {det_rate:.0f}% ({n_runs - results[key]['missed']}/{n_runs} detections)")


if __name__ == "__main__":
    main()
