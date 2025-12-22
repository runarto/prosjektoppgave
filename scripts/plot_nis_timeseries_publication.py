#!/usr/bin/env python3
"""
NIS Time Series Plots - Publication Quality

Creates separate, readable plots for each sensor fault type.
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


def plot_single_sensor(ax, t, nis, sensor_name, color, fault_start, fault_end, threshold):
    """Plot NIS for a single sensor with fault window."""
    ax.plot(t, nis, color=color, linewidth=1.2, alpha=0.9)

    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'$\\chi^2$ threshold ({threshold})')

    # Fault window shading
    ax.axvspan(fault_start, fault_end, alpha=0.15, color='red', label='Fault window')

    ax.set_ylabel('NIS')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)

    # Set reasonable y-limit
    max_nis = np.max(nis)
    ax.set_ylim([0, min(100, max_nis * 1.1)])


def main():
    # Load data
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()
    sim_data = db.load_run(run_id)

    sensors = [
        ('mag', 'Magnetometer', 1.0, '#1f77b4'),
        ('sun', 'Sun Sensor', 1.0, '#ff7f0e'),
        ('st', 'Star Tracker', 0.2, '#2ca02c'),
    ]

    # Option 1: Single figure with 3 rows (tall figure)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, (sensor, name, magnitude, color) in zip(axes, sensors):
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

        plot_single_sensor(ax, t, nis, name, color, FAULT_START, FAULT_END, CHI2_THRESHOLD)

        if sensor == 'st':
            title = f'{name} Fault ({np.rad2deg(magnitude):.1f}°)'
        else:
            title = f'{name} Fault (magnitude = {magnitude})'
        ax.set_title(title)

    # Only show x-label on bottom plot
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')

    plt.tight_layout()
    plt.savefig('fault_detection_nis_3panel.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fault_detection_nis_3panel.png', bbox_inches='tight', dpi=300)
    print("Saved: fault_detection_nis_3panel.pdf/png")
    plt.close()

    # Option 2: Individual plots for each sensor (most readable)
    for sensor, name, magnitude, color in sensors:
        fig, ax = plt.subplots(figsize=(10, 5))

        modified = apply_fault(sim_data, sensor, magnitude)
        mag_nis, sun_nis, st_nis = run_eskf_get_nis(modified)

        if sensor == 'mag':
            nis_data = mag_nis
        elif sensor == 'sun':
            nis_data = sun_nis
        else:
            nis_data = st_nis

        t = np.array(nis_data['t'])
        nis = np.array(nis_data['nis'])

        plot_single_sensor(ax, t, nis, name, color, FAULT_START, FAULT_END, CHI2_THRESHOLD)

        if sensor == 'st':
            title = f'{name} Fault Detection ({np.rad2deg(magnitude):.1f}°)'
        else:
            title = f'{name} Fault Detection (magnitude = {magnitude})'
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(f'fault_detection_nis_{sensor}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fault_detection_nis_{sensor}.png', bbox_inches='tight', dpi=300)
        print(f"Saved: fault_detection_nis_{sensor}.pdf/png")
        plt.close()

    # Option 3: 2-panel figure with Magnetometer and Star Tracker only (most contrasting)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (sensor, name, magnitude, color) in zip(axes, [sensors[0], sensors[2]]):
        modified = apply_fault(sim_data, sensor, magnitude)
        mag_nis, sun_nis, st_nis = run_eskf_get_nis(modified)

        if sensor == 'mag':
            nis_data = mag_nis
            panel = '(a)'
        else:
            nis_data = st_nis
            panel = '(b)'

        t = np.array(nis_data['t'])
        nis = np.array(nis_data['nis'])

        plot_single_sensor(ax, t, nis, name, color, FAULT_START, FAULT_END, CHI2_THRESHOLD)

        if sensor == 'st':
            title = f'{panel} {name} ({np.rad2deg(magnitude):.1f}°)'
        else:
            title = f'{panel} {name} (magnitude = {magnitude})'
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig('fault_detection_nis_2panel.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fault_detection_nis_2panel.png', bbox_inches='tight', dpi=300)
    print("Saved: fault_detection_nis_2panel.pdf/png")


if __name__ == "__main__":
    main()
