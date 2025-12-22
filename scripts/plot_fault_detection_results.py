#!/usr/bin/env python3
"""
Plot Fault Detection Results

Generates publication-quality figures for fault detection evaluation:
1. NIS time series during faults (per sensor)
2. Detection latency comparison
3. Attitude error comparison
4. Summary heatmap
"""

import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 150

CONFIG_PATH = "configs/config_baseline_short.yaml"
FAULT_START = 100.0
FAULT_END = 150.0
CHI2_THRESHOLD = 7.81


def apply_fault(sim_data, sensor_type: str, magnitude: float):
    """Apply fault to specified sensor."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)
    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END:
            if sensor_type == 'sun' and not np.isnan(modified.sun_meas[k, 0]):
                perturbation = np.random.randn(3)
                perturbation = perturbation / np.linalg.norm(perturbation) * magnitude
                modified.sun_meas[k] = modified.sun_meas[k] + perturbation
                modified.sun_meas[k] = modified.sun_meas[k] / np.linalg.norm(modified.sun_meas[k])
            elif sensor_type == 'mag' and not np.isnan(modified.mag_meas[k, 0]):
                perturbation = np.random.randn(3)
                perturbation = perturbation / np.linalg.norm(perturbation) * magnitude
                modified.mag_meas[k] = modified.mag_meas[k] + perturbation
                modified.mag_meas[k] = modified.mag_meas[k] / np.linalg.norm(modified.mag_meas[k])
            elif sensor_type == 'st' and not np.isnan(modified.st_meas[k, 0]):
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                q_error = Quaternion.from_avec(axis * magnitude)
                q_meas = Quaternion.from_array(modified.st_meas[k])
                q_corrupted = (q_meas @ q_error).normalize()
                modified.st_meas[k] = q_corrupted.as_array()
    return modified


def run_eskf_get_nis(sim_data):
    """Run ESKF and return per-sensor NIS time series."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    mag_nis = {'t': [], 'nis': []}
    sun_nis = {'t': [], 'nis': []}
    st_nis = {'t': [], 'nis': []}

    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]

        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        if not np.isnan(sim_data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except ValueError:
                pass
            mag_nis['t'].append(t)
            mag_nis['nis'].append(eskf.last_nis)

        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass
            sun_nis['t'].append(t)
            sun_nis['nis'].append(eskf.last_nis)

        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass
            st_nis['t'].append(t)
            st_nis['nis'].append(eskf.last_nis)

    return mag_nis, sun_nis, st_nis


def plot_nis_time_series():
    """Plot 1: NIS time series for each sensor fault type."""
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()
    sim_data = db.load_run(run_id)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    sensors = [('mag', 'Magnetometer', 1.0), ('sun', 'Sun Sensor', 1.0), ('st', 'Star Tracker', 0.2)]
    colors = {'mag': 'C0', 'sun': 'C1', 'st': 'C2'}

    for ax, (sensor, name, magnitude) in zip(axes, sensors):
        # Apply fault
        modified = apply_fault(sim_data, sensor, magnitude)
        mag_nis, sun_nis, st_nis = run_eskf_get_nis(modified)

        # Select the faulty sensor's NIS
        if sensor == 'mag':
            nis_data = mag_nis
        elif sensor == 'sun':
            nis_data = sun_nis
        else:
            nis_data = st_nis

        t = np.array(nis_data['t'])
        nis = np.array(nis_data['nis'])

        # Plot NIS
        ax.plot(t, nis, color=colors[sensor], linewidth=0.8, alpha=0.8)

        # Threshold line
        ax.axhline(y=CHI2_THRESHOLD, color='r', linestyle='--', linewidth=1.5, label=f'$\\chi^2$ threshold ({CHI2_THRESHOLD})')

        # Fault window shading
        ax.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red', label='Fault window')

        ax.set_ylabel('NIS')
        ax.set_title(f'{name} Fault (magnitude={magnitude})')
        ax.legend(loc='upper right')
        ax.set_ylim([0, min(100, np.max(nis) * 1.1)])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('fault_detection_nis_timeseries.pdf', bbox_inches='tight')
    plt.savefig('fault_detection_nis_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved: fault_detection_nis_timeseries.pdf/png")


def plot_detection_latency():
    """Plot 2: Detection latency comparison bar chart."""
    # Data from evaluation results
    sensors = ['Magnetometer', 'Sun Sensor', 'Star Tracker']
    eskf_latency = [1.0, 2.75, 25.0]
    redundant_latency = [1.0, 2.75, 25.0]

    x = np.arange(len(sensors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, eskf_latency, width, label='ESKF', color='C0', alpha=0.8)
    bars2 = ax.bar(x + width/2, redundant_latency, width, label='Redundant', color='C2', alpha=0.8)

    ax.set_ylabel('Detection Latency [s]')
    ax.set_title('Fault Detection Latency by Sensor Type')
    ax.set_xticks(x)
    ax.set_xticklabels(sensors)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('fault_detection_latency.pdf', bbox_inches='tight')
    plt.savefig('fault_detection_latency.png', dpi=150, bbox_inches='tight')
    print("Saved: fault_detection_latency.pdf/png")


def plot_error_comparison():
    """Plot 3: Attitude error during fault comparison."""
    # Data from evaluation results
    sensors = ['Mag\n(0.5-3.0)', 'Sun\n(0.5-3.0)', 'ST\n(2.9°-86°)']
    eskf_error = [0.022, 0.021, 0.021]
    smoother_error = [0.018, 0.020, 0.042]
    redundant_error = [0.012, 0.012, 0.021]

    x = np.arange(len(sensors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, eskf_error, width, label='ESKF', color='C0', alpha=0.8)
    bars2 = ax.bar(x, smoother_error, width, label='Smoother', color='C1', alpha=0.8)
    bars3 = ax.bar(x + width, redundant_error, width, label='Redundant', color='C2', alpha=0.8)

    ax.set_ylabel('Mean Attitude Error During Fault [deg]')
    ax.set_title('Estimator Accuracy During Sensor Faults')
    ax.set_xticks(x)
    ax.set_xticklabels(sensors)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fault_detection_error_comparison.pdf', bbox_inches='tight')
    plt.savefig('fault_detection_error_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: fault_detection_error_comparison.pdf/png")


def plot_summary_heatmap():
    """Plot 4: Summary heatmap of detection success."""
    # Detection success (1 = detected, 0 = not detected)
    # Rows: sensor types, Columns: magnitudes
    mag_mags = [0.5, 1.0, 2.0, 3.0]
    sun_mags = [0.5, 1.0, 2.0, 3.0]
    st_mags = [2.9, 11.5, 28.6, 85.9]  # in degrees

    # All detected for ESKF and Redundant
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ESKF detection
    eskf_data = np.ones((3, 4))  # All detected
    im1 = axes[0].imshow(eskf_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(['Low', 'Med-Low', 'Med-High', 'High'])
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(['Magnetometer', 'Sun Sensor', 'Star Tracker'])
    axes[0].set_title('ESKF Detection')
    axes[0].set_xlabel('Fault Magnitude')

    # Add latency annotations
    latencies = [[1.0, 1.0, 1.0, 1.0],
                 [3.0, 3.0, 2.5, 2.5],
                 [25.0, 25.0, 25.0, 25.0]]
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f'{latencies[i][j]:.1f}s', ha='center', va='center', fontsize=9)

    # Redundant detection
    redundant_data = np.ones((3, 4))
    im2 = axes[1].imshow(redundant_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(['Low', 'Med-Low', 'Med-High', 'High'])
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(['Magnetometer', 'Sun Sensor', 'Star Tracker'])
    axes[1].set_title('Redundant Detection')
    axes[1].set_xlabel('Fault Magnitude')

    for i in range(3):
        for j in range(4):
            axes[1].text(j, i, f'{latencies[i][j]:.1f}s', ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('fault_detection_summary.pdf', bbox_inches='tight')
    plt.savefig('fault_detection_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: fault_detection_summary.pdf/png")


def plot_combined_figure():
    """Plot 5: Combined 2x2 figure for publication."""
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()
    sim_data = db.load_run(run_id)

    fig = plt.figure(figsize=(12, 10))

    # Panel A: NIS time series for magnetometer fault
    ax1 = fig.add_subplot(2, 2, 1)
    modified = apply_fault(sim_data, 'mag', 1.0)
    mag_nis, sun_nis, st_nis = run_eskf_get_nis(modified)

    t = np.array(mag_nis['t'])
    nis = np.array(mag_nis['nis'])
    ax1.plot(t, nis, 'C0', linewidth=0.8)
    ax1.axhline(y=CHI2_THRESHOLD, color='r', linestyle='--', linewidth=1.5)
    ax1.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red')
    ax1.set_ylabel('Magnetometer NIS')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('(a) Magnetometer Fault Detection')
    ax1.set_ylim([0, min(80, np.max(nis) * 1.1)])
    ax1.grid(True, alpha=0.3)

    # Panel B: NIS time series for star tracker fault
    ax2 = fig.add_subplot(2, 2, 2)
    modified = apply_fault(sim_data, 'st', 0.2)
    mag_nis, sun_nis, st_nis = run_eskf_get_nis(modified)

    t = np.array(st_nis['t'])
    nis = np.array(st_nis['nis'])
    ax2.plot(t, nis, 'C2', linewidth=0.8)
    ax2.axhline(y=CHI2_THRESHOLD, color='r', linestyle='--', linewidth=1.5)
    ax2.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red')
    ax2.set_ylabel('Star Tracker NIS')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('(b) Star Tracker Fault Detection')
    ax2.set_ylim([0, min(150, np.max(nis) * 1.1)])
    ax2.grid(True, alpha=0.3)

    # Panel C: Detection latency
    ax3 = fig.add_subplot(2, 2, 3)
    sensors = ['Mag', 'Sun', 'ST']
    latency = [1.0, 2.75, 25.0]
    colors = ['C0', 'C1', 'C2']
    bars = ax3.bar(sensors, latency, color=colors, alpha=0.8)
    ax3.set_ylabel('Detection Latency [s]')
    ax3.set_title('(c) Detection Latency by Sensor')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, lat in zip(bars, latency):
        ax3.annotate(f'{lat:.1f}s', xy=(bar.get_x() + bar.get_width()/2, lat),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # Panel D: Error comparison
    ax4 = fig.add_subplot(2, 2, 4)
    sensors = ['Mag', 'Sun', 'ST']
    x = np.arange(len(sensors))
    width = 0.25
    eskf_err = [0.022, 0.021, 0.021]
    smoother_err = [0.018, 0.020, 0.042]
    redundant_err = [0.012, 0.012, 0.021]

    ax4.bar(x - width, eskf_err, width, label='ESKF', color='C0', alpha=0.8)
    ax4.bar(x, smoother_err, width, label='Smoother', color='C1', alpha=0.8)
    ax4.bar(x + width, redundant_err, width, label='Redundant', color='C2', alpha=0.8)
    ax4.set_ylabel('Error During Fault [deg]')
    ax4.set_title('(d) Accuracy During Faults')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sensors)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fault_detection_combined.pdf', bbox_inches='tight')
    plt.savefig('fault_detection_combined.png', dpi=150, bbox_inches='tight')
    print("Saved: fault_detection_combined.pdf/png")


def main():
    print("Generating fault detection plots...")
    print()

    print("1. NIS time series...")
    plot_nis_time_series()

    print("2. Detection latency...")
    plot_detection_latency()

    print("3. Error comparison...")
    plot_error_comparison()

    print("4. Summary heatmap...")
    plot_summary_heatmap()

    print("5. Combined figure...")
    plot_combined_figure()

    print()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
