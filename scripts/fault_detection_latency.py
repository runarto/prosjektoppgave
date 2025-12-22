#!/usr/bin/env python3
"""
Fault Detection Latency Test.

Measures how quickly the Redundant estimator detects faults through
ESKF-Smoother disagreement monitoring.

Tests different fault types and severities.
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
    """Configuration for a persistent fault."""
    name: str
    start_time: float
    sensor: str  # 'mag', 'sun', 'star'
    fault_type: str  # 'bias', 'noise', 'stuck'
    magnitude: float


def inject_persistent_fault(sim_data, fault: FaultConfig, seed: int = None):
    """Inject a persistent fault starting at fault.start_time."""
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

    # Store stuck value if needed
    stuck_value_mag = None
    stuck_value_sun = None
    stuck_value_st = None

    for k in range(len(modified.t)):
        t = modified.t[k]

        if t >= fault.start_time:
            if fault.sensor == 'mag' and not np.any(np.isnan(modified.mag_meas[k])):
                if fault.fault_type == 'bias':
                    # Add constant bias
                    bias = np.array([fault.magnitude, 0, 0])
                    biased = modified.mag_meas[k] + bias
                    modified.mag_meas[k] = biased / np.linalg.norm(biased)
                elif fault.fault_type == 'noise':
                    # Add extra noise
                    noise = np.random.randn(3) * fault.magnitude
                    noisy = modified.mag_meas[k] + noise
                    modified.mag_meas[k] = noisy / np.linalg.norm(noisy)
                elif fault.fault_type == 'stuck':
                    # Sensor stuck at first faulty value
                    if stuck_value_mag is None:
                        stuck_value_mag = modified.mag_meas[k].copy()
                    modified.mag_meas[k] = stuck_value_mag

            elif fault.sensor == 'sun' and not np.any(np.isnan(modified.sun_meas[k])):
                if fault.fault_type == 'bias':
                    bias = np.array([fault.magnitude, 0, 0])
                    biased = modified.sun_meas[k] + bias
                    modified.sun_meas[k] = biased / np.linalg.norm(biased)
                elif fault.fault_type == 'noise':
                    noise = np.random.randn(3) * fault.magnitude
                    noisy = modified.sun_meas[k] + noise
                    modified.sun_meas[k] = noisy / np.linalg.norm(noisy)
                elif fault.fault_type == 'stuck':
                    if stuck_value_sun is None:
                        stuck_value_sun = modified.sun_meas[k].copy()
                    modified.sun_meas[k] = stuck_value_sun

            elif fault.sensor == 'star' and not np.any(np.isnan(modified.st_meas[k])):
                if fault.fault_type == 'bias':
                    # Add constant rotation bias
                    bias_axis = np.array([1, 0, 0]) * fault.magnitude
                    q_bias = Quaternion.from_avec(bias_axis)
                    q_orig = Quaternion.from_array(modified.st_meas[k])
                    q_biased = (q_orig @ q_bias).normalize()
                    modified.st_meas[k] = q_biased.as_array()
                elif fault.fault_type == 'noise':
                    noise_axis = np.random.randn(3) * fault.magnitude
                    q_noise = Quaternion.from_avec(noise_axis)
                    q_orig = Quaternion.from_array(modified.st_meas[k])
                    q_noisy = (q_orig @ q_noise).normalize()
                    modified.st_meas[k] = q_noisy.as_array()
                elif fault.fault_type == 'stuck':
                    if stuck_value_st is None:
                        stuck_value_st = modified.st_meas[k].copy()
                    modified.st_meas[k] = stuck_value_st

    return modified


def run_redundant_with_detection(sim_data, config_path: str, fault: FaultConfig) -> Dict:
    """Run redundant estimator and track disagreement/detection."""
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
    detection_time = None
    switch_time = None

    disagreement_threshold = 2.0  # degrees (from RedundantEstimator default)

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

        # Check for detection (first time disagreement exceeds threshold after fault)
        if t >= fault.start_time and detection_time is None:
            if disagreement > disagreement_threshold:
                detection_time = t - fault.start_time

        # Check for mode switch
        if t >= fault.start_time and switch_time is None:
            if primary in ['SMOOTHER', 'CONSERVATIVE']:
                switch_time = t - fault.start_time

    return {
        'times': np.array(times),
        'disagreements': np.array(disagreements),
        'modes': modes,
        'detection_time': detection_time,
        'switch_time': switch_time,
    }


def run_fault_detection_test(sim_data_base, config_path: str, faults: List[FaultConfig],
                              n_runs: int = 10) -> Dict:
    """Run fault detection latency tests."""
    results = {}

    for fault in faults:
        print(f"\n  Testing: {fault.name}")
        detection_times = []
        switch_times = []
        example_run = None

        for i in range(n_runs):
            # Inject fault with different seed each run
            sim_data = inject_persistent_fault(sim_data_base, fault, seed=i*42)

            # Run redundant estimator
            run_result = run_redundant_with_detection(sim_data, config_path, fault)

            if run_result['detection_time'] is not None:
                detection_times.append(run_result['detection_time'])
            if run_result['switch_time'] is not None:
                switch_times.append(run_result['switch_time'])

            # Save first run for plotting
            if i == 0:
                example_run = run_result

        results[fault.name] = {
            'fault': fault,
            'detection_times': detection_times,
            'switch_times': switch_times,
            'detection_rate': len(detection_times) / n_runs * 100,
            'switch_rate': len(switch_times) / n_runs * 100,
            'mean_detection': np.mean(detection_times) if detection_times else None,
            'mean_switch': np.mean(switch_times) if switch_times else None,
            'example': example_run,
        }

        det_str = f"{np.mean(detection_times):.2f}s" if detection_times else "N/A"
        sw_str = f"{np.mean(switch_times):.2f}s" if switch_times else "N/A"
        print(f"    Detection: {len(detection_times)}/{n_runs} runs, mean={det_str}")
        print(f"    Switch:    {len(switch_times)}/{n_runs} runs, mean={sw_str}")

    return results


def plot_results(results: Dict, save_path: str):
    """Plot fault detection latency results."""
    n_faults = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Example disagreement time series
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_faults))

    for i, (name, data) in enumerate(results.items()):
        example = data['example']
        fault = data['fault']

        # Zoom to fault region
        t_start = fault.start_time - 20
        t_end = fault.start_time + 80
        mask = (example['times'] >= t_start) & (example['times'] <= t_end)

        ax.plot(example['times'][mask] - fault.start_time,
                example['disagreements'][mask],
                color=colors[i], label=name, linewidth=1.5)

    ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.0, label='Fault onset')
    ax.set_xlabel('Time since fault onset [s]')
    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('Disagreement Evolution After Fault')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-20, 80])

    # 2. Detection latency bar chart
    ax = axes[0, 1]
    names = list(results.keys())
    detection_means = [results[n]['mean_detection'] if results[n]['mean_detection'] else 0
                       for n in names]
    detection_rates = [results[n]['detection_rate'] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, detection_means, color='steelblue', alpha=0.8)

    # Add detection rate labels
    for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=10)
    ax.set_ylabel('Detection Latency [s]')
    ax.set_title('Mean Time to Detect Fault')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Detection rate vs fault magnitude
    ax = axes[1, 0]

    # Group by sensor
    sensors = ['mag', 'sun', 'star']
    sensor_colors = {'mag': 'C0', 'sun': 'C1', 'star': 'C2'}

    for sensor in sensors:
        sensor_faults = [(n, d) for n, d in results.items() if d['fault'].sensor == sensor]
        if sensor_faults:
            magnitudes = [d['fault'].magnitude for _, d in sensor_faults]
            rates = [d['detection_rate'] for _, d in sensor_faults]
            ax.scatter(magnitudes, rates, color=sensor_colors[sensor],
                      s=100, label=sensor.capitalize(), alpha=0.8)

    ax.set_xlabel('Fault Magnitude')
    ax.set_ylabel('Detection Rate [%]')
    ax.set_title('Detection Rate vs Fault Severity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for name, data in results.items():
        det_time = f"{data['mean_detection']:.2f}" if data['mean_detection'] else "N/A"
        sw_time = f"{data['mean_switch']:.2f}" if data['mean_switch'] else "N/A"
        table_data.append([
            name,
            f"{data['detection_rate']:.0f}%",
            det_time,
            f"{data['switch_rate']:.0f}%",
            sw_time
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Fault', 'Det. Rate', 'Det. Time [s]', 'Switch Rate', 'Switch Time [s]'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Fault Detection Summary', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'

    print("=" * 70)
    print("FAULT DETECTION LATENCY TEST")
    print("=" * 70)

    # Create config
    config = load_yaml(base_config)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    config_path = 'configs/config_detection_test.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load simulation data
    print("\nLoading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    # Define fault scenarios
    faults = [
        # Magnetometer faults
        FaultConfig("Mag Bias (5%)", 100.0, 'mag', 'bias', 0.05),
        FaultConfig("Mag Bias (10%)", 100.0, 'mag', 'bias', 0.10),
        FaultConfig("Mag Stuck", 100.0, 'mag', 'stuck', 0.0),

        # Sun sensor faults
        FaultConfig("Sun Bias (5%)", 100.0, 'sun', 'bias', 0.05),
        FaultConfig("Sun Bias (10%)", 100.0, 'sun', 'bias', 0.10),
        FaultConfig("Sun Stuck", 100.0, 'sun', 'stuck', 0.0),

        # Star tracker faults
        FaultConfig("Star Bias (1°)", 100.0, 'star', 'bias', np.deg2rad(1.0)),
        FaultConfig("Star Bias (5°)", 100.0, 'star', 'bias', np.deg2rad(5.0)),
        FaultConfig("Star Stuck", 100.0, 'star', 'stuck', 0.0),
    ]

    print("\nRunning fault detection tests...")
    results = run_fault_detection_test(sim_data_base, config_path, faults, n_runs=10)

    # Plot results
    plot_results(results, 'fault_detection_latency.png')

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Fault':<20} {'Det. Rate':>10} {'Det. Time':>12} {'Switch Rate':>12}")
    print("-" * 60)
    for name, data in results.items():
        det_time = f"{data['mean_detection']:.2f}s" if data['mean_detection'] else "N/A"
        print(f"{name:<20} {data['detection_rate']:>9.0f}% {det_time:>12} {data['switch_rate']:>11.0f}%")


if __name__ == "__main__":
    main()
