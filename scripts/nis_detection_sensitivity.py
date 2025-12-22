#!/usr/bin/env python3
"""
NIS-Based Fault Detection Sensitivity Analysis

Evaluates how detection performance depends on:
- Detection window length Nh: {5, 10, 20}
- Exceedance fraction f: {0.5, 0.6, 0.8}

Test scenarios (one representative fault per sensor):
- Magnetometer: perturbation magnitude 2.0
- Sun sensor: perturbation magnitude 2.0
- Star tracker: rotation error 0.5 rad (~28.6 deg)

Fault window: t=100s to t=150s (100% fault probability)
"""

import numpy as np
import sqlite3
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)

# =============================================================================
# Publication-quality plot settings
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['lines.linewidth'] = 1.5
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['grid.linewidth'] = 0.6
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = 'gray'
rcParams['mathtext.fontset'] = 'dejavuserif'

# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = "configs/config_baseline_short.yaml"

FAULT_START = 100.0  # seconds
FAULT_END = 150.0    # seconds

# Representative fault magnitudes
FAULT_MAGNITUDES = {
    'mag': 2.0,        # Magnetometer perturbation
    'sun': 2.0,        # Sun sensor perturbation
    'st': 0.5,         # Star tracker rotation error (rad)
}

# Detection parameters to sweep
WINDOW_LENGTHS = [5, 10, 20]      # Nh
EXCEEDANCE_FRACTIONS = [0.5, 0.6, 0.8]  # f

# Chi-squared threshold (95% confidence, 3 DOF)
CHI2_THRESHOLD = 7.81


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParameterizedNISTracker:
    """Track NIS values with configurable detection parameters."""
    name: str
    threshold: float
    window_size: int
    detection_fraction: float

    nis_history: deque = field(default_factory=deque)
    times: List[float] = field(default_factory=list)
    nis_values: List[float] = field(default_factory=list)
    detection_flags: List[bool] = field(default_factory=list)

    first_detection_time: Optional[float] = None
    is_detected: bool = False

    def __post_init__(self):
        self.nis_history = deque(maxlen=self.window_size)

    def update(self, t: float, nis: float):
        """Update tracker with new NIS value."""
        self.times.append(t)
        self.nis_values.append(nis)
        self.nis_history.append(nis)

        # Check detection based on sliding window
        if len(self.nis_history) >= 3:
            exceeding = sum(1 for n in self.nis_history if n > self.threshold)
            fraction = exceeding / len(self.nis_history)
            detected = fraction >= self.detection_fraction
        else:
            detected = False

        self.detection_flags.append(detected)

        # Record first detection time
        if detected and self.first_detection_time is None:
            self.first_detection_time = t
            self.is_detected = True

    def get_false_detections(self, fault_start: float, fault_end: float) -> int:
        """Count detections outside fault window."""
        count = 0
        for t, flag in zip(self.times, self.detection_flags):
            if flag and (t < fault_start or t > fault_end):
                count += 1
        return count

    def reset(self):
        """Reset tracker state."""
        self.nis_history.clear()
        self.times.clear()
        self.nis_values.clear()
        self.detection_flags.clear()
        self.first_detection_time = None
        self.is_detected = False


@dataclass
class SensitivityResult:
    """Result from one parameter combination test."""
    sensor: str
    window_length: int  # Nh
    exceedance_fraction: float  # f
    estimator: str
    detected: bool
    detection_latency: Optional[float]
    false_positives: int


# =============================================================================
# Fault Injection
# =============================================================================

def apply_fault(sim_data, sensor_type: str, magnitude: float):
    """Apply transient fault to specified sensor during fault window."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)  # Reproducible results
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


# =============================================================================
# Estimator Runners with Parameterized Detection
# =============================================================================

def run_eskf_with_detection(
    sim_data,
    faulty_sensor: str,
    window_length: int,
    exceedance_fraction: float
) -> Tuple[Dict[str, ParameterizedNISTracker], List[float], List[float]]:
    """
    Run ESKF with parameterized NIS-based detection.
    """
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    # Initialize trackers with specified parameters
    trackers = {
        'mag': ParameterizedNISTracker('magnetometer', CHI2_THRESHOLD, window_length, exceedance_fraction),
        'sun': ParameterizedNISTracker('sun_sensor', CHI2_THRESHOLD, window_length, exceedance_fraction),
        'st': ParameterizedNISTracker('star_tracker', CHI2_THRESHOLD, window_length, exceedance_fraction),
    }

    times = []
    errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        times.append(t)
        q_err = x.nom.ori.conjugate() @ q_true
        theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
        errors.append(np.rad2deg(theta))

        # Prediction
        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        # Magnetometer update
        if not np.isnan(sim_data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except ValueError:
                pass
            trackers['mag'].update(t, eskf.last_nis)

        # Sun sensor update
        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass
            trackers['sun'].update(t, eskf.last_nis)

        # Star tracker update
        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass
            trackers['st'].update(t, eskf.last_nis)

    return trackers, times, errors


def run_redundant_with_detection(
    sim_data,
    faulty_sensor: str,
    window_length: int,
    exceedance_fraction: float
) -> Tuple[Dict[str, ParameterizedNISTracker], List[float], List[float]]:
    """
    Run Redundant estimator with parameterized NIS-based detection.
    """
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)

    redundant = RedundantEstimator(
        P0=P0,
        config_path=CONFIG_PATH,
        smoother_lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )
    redundant.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    # Initialize trackers with specified parameters
    trackers = {
        'mag': ParameterizedNISTracker('magnetometer', CHI2_THRESHOLD, window_length, exceedance_fraction),
        'sun': ParameterizedNISTracker('sun_sensor', CHI2_THRESHOLD, window_length, exceedance_fraction),
        'st': ParameterizedNISTracker('star_tracker', CHI2_THRESHOLD, window_length, exceedance_fraction),
    }

    times = []
    errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        z_mag = sim_data.mag_meas[k] if not np.isnan(sim_data.mag_meas[k, 0]) else None
        z_sun = sim_data.sun_meas[k] if not np.isnan(sim_data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.isnan(sim_data.st_meas[k, 0]) else None

        x_eskf, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_eskf, t=t, jd=sim_data.jd[k], dt=dt,
            omega_meas=sim_data.omega_meas[k],
            z_mag=z_mag, z_sun=z_sun, z_st=z_st,
            B_n=sim_data.b_eci[k], s_n=sim_data.s_eci[k],
        )

        # Track NIS from internal ESKF
        if z_mag is not None and len(redundant.nis_history_mag) > 0:
            trackers['mag'].update(t, redundant.nis_history_mag[-1])
        if z_sun is not None and len(redundant.nis_history_sun) > 0:
            trackers['sun'].update(t, redundant.nis_history_sun[-1])
        if z_st is not None and len(redundant.nis_history_st) > 0:
            trackers['st'].update(t, redundant.nis_history_st[-1])

        # Compute error based on primary
        if primary == "SMOOTHER" and smoother_state is not None:
            q_est = smoother_state.ori
        else:
            q_est = x_eskf.nom.ori

        q_err = q_est.conjugate() @ q_true
        theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))

        times.append(t)
        errors.append(np.rad2deg(theta))

    return trackers, times, errors


# =============================================================================
# Sensitivity Test Runner
# =============================================================================

def run_sensitivity_test(
    sim_data,
    sensor: str,
    window_length: int,
    exceedance_fraction: float,
) -> List[SensitivityResult]:
    """
    Run sensitivity test for one parameter combination.

    Returns results for ESKF and Redundant estimators.
    """
    # Apply fault to the specified sensor
    magnitude = FAULT_MAGNITUDES[sensor]
    modified_data = apply_fault(sim_data, sensor, magnitude)

    results = []

    # Run ESKF
    trackers_eskf, _, _ = run_eskf_with_detection(
        modified_data, sensor, window_length, exceedance_fraction
    )

    faulty_tracker = trackers_eskf[sensor]
    detection_latency = None
    if faulty_tracker.first_detection_time is not None:
        if faulty_tracker.first_detection_time >= FAULT_START:
            detection_latency = faulty_tracker.first_detection_time - FAULT_START

    # Count false positives on healthy sensors
    healthy_sensors = [s for s in ['mag', 'sun', 'st'] if s != sensor]
    false_positives = sum(
        trackers_eskf[s].get_false_detections(FAULT_START, FAULT_END)
        for s in healthy_sensors
    )

    results.append(SensitivityResult(
        sensor=sensor,
        window_length=window_length,
        exceedance_fraction=exceedance_fraction,
        estimator='ESKF',
        detected=faulty_tracker.is_detected,
        detection_latency=detection_latency,
        false_positives=false_positives,
    ))

    # Run Redundant
    trackers_red, _, _ = run_redundant_with_detection(
        modified_data, sensor, window_length, exceedance_fraction
    )

    faulty_tracker = trackers_red[sensor]
    detection_latency = None
    if faulty_tracker.first_detection_time is not None:
        if faulty_tracker.first_detection_time >= FAULT_START:
            detection_latency = faulty_tracker.first_detection_time - FAULT_START

    false_positives = sum(
        trackers_red[s].get_false_detections(FAULT_START, FAULT_END)
        for s in healthy_sensors
    )

    results.append(SensitivityResult(
        sensor=sensor,
        window_length=window_length,
        exceedance_fraction=exceedance_fraction,
        estimator='Redundant',
        detected=faulty_tracker.is_detected,
        detection_latency=detection_latency,
        false_positives=false_positives,
    ))

    return results


# =============================================================================
# Output Functions
# =============================================================================

def save_results_csv(results: List[SensitivityResult], filename: str):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sensor', 'Nh', 'f', 'estimator',
            'detected', 'detection_latency', 'false_positives'
        ])
        for r in results:
            writer.writerow([
                r.sensor, r.window_length, r.exceedance_fraction, r.estimator,
                1 if r.detected else 0,
                r.detection_latency if r.detection_latency is not None else '',
                r.false_positives
            ])


def save_results_sqlite(results: List[SensitivityResult], db_path: str):
    """Save results to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nis_sensitivity_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            Nh INTEGER,
            f REAL,
            estimator TEXT,
            detected INTEGER,
            detection_latency REAL,
            false_positives INTEGER
        )
    ''')

    timestamp = datetime.now().isoformat()

    for r in results:
        cursor.execute('''
            INSERT INTO nis_sensitivity_results
            (timestamp, sensor, Nh, f, estimator, detected, detection_latency, false_positives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, r.sensor, r.window_length, r.exceedance_fraction,
            r.estimator, 1 if r.detected else 0, r.detection_latency,
            r.false_positives
        ))

    conn.commit()
    conn.close()


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_detection_rate_vs_f(results: List[SensitivityResult], estimator: str = 'ESKF'):
    """
    Plot detection rate vs exceedance fraction f for each Nh.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    sensors = ['mag', 'sun', 'st']
    sensor_names = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}
    colors = {'5': '#1f77b4', '10': '#ff7f0e', '20': '#2ca02c'}
    markers = {'5': 'o', '10': 's', '20': '^'}

    for ax, sensor in zip(axes, sensors):
        for Nh in WINDOW_LENGTHS:
            f_values = []
            detection_rates = []

            for f in EXCEEDANCE_FRACTIONS:
                matching = [r for r in results
                           if r.sensor == sensor
                           and r.window_length == Nh
                           and r.exceedance_fraction == f
                           and r.estimator == estimator]
                if matching:
                    rate = sum(1 for r in matching if r.detected) / len(matching)
                    f_values.append(f)
                    detection_rates.append(rate * 100)

            ax.plot(f_values, detection_rates, f'-{markers[str(Nh)]}',
                   color=colors[str(Nh)], label=f'$N_h$ = {Nh}',
                   markersize=8, linewidth=2)

        ax.set_xlabel('Exceedance Fraction $f$')
        ax.set_title(sensor_names[sensor])
        ax.set_ylim([-5, 105])
        ax.set_xticks(EXCEEDANCE_FRACTIONS)
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel('Detection Rate [%]')
    axes[0].legend(loc='lower left')

    plt.suptitle(f'{estimator}: Detection Rate vs. Exceedance Fraction', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'sensitivity_detection_rate_vs_f_{estimator.lower()}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'sensitivity_detection_rate_vs_f_{estimator.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_latency_vs_Nh(results: List[SensitivityResult], estimator: str = 'ESKF'):
    """
    Plot detection latency vs window length Nh for each f.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    sensors = ['mag', 'sun', 'st']
    sensor_names = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}
    colors = {'0.5': '#1f77b4', '0.6': '#ff7f0e', '0.8': '#2ca02c'}
    markers = {'0.5': 'o', '0.6': 's', '0.8': '^'}

    for ax, sensor in zip(axes, sensors):
        for f in EXCEEDANCE_FRACTIONS:
            Nh_values = []
            latencies = []

            for Nh in WINDOW_LENGTHS:
                matching = [r for r in results
                           if r.sensor == sensor
                           and r.window_length == Nh
                           and r.exceedance_fraction == f
                           and r.estimator == estimator
                           and r.detected
                           and r.detection_latency is not None]
                if matching:
                    avg_latency = np.mean([r.detection_latency for r in matching])
                    Nh_values.append(Nh)
                    latencies.append(avg_latency)

            if Nh_values:
                ax.plot(Nh_values, latencies, f'-{markers[str(f)]}',
                       color=colors[str(f)], label=f'$f$ = {f}',
                       markersize=8, linewidth=2)

        ax.set_xlabel('Window Length $N_h$')
        ax.set_title(sensor_names[sensor])
        ax.set_xticks(WINDOW_LENGTHS)
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel('Detection Latency [s]')
    axes[0].legend(loc='upper left')

    plt.suptitle(f'{estimator}: Detection Latency vs. Window Length', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'sensitivity_latency_vs_Nh_{estimator.lower()}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'sensitivity_latency_vs_Nh_{estimator.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_combined_sensitivity(results: List[SensitivityResult]):
    """
    Create a combined 2x2 plot showing key sensitivity results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sensors = ['mag', 'sun', 'st']
    sensor_names = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}
    sensor_colors = {'mag': '#1f77b4', 'sun': '#ff7f0e', 'st': '#2ca02c'}

    # (a) Detection rate vs f for ESKF, aggregated across sensors
    ax = axes[0, 0]
    for sensor in sensors:
        f_values = []
        rates = []
        for f in EXCEEDANCE_FRACTIONS:
            matching = [r for r in results
                       if r.sensor == sensor
                       and r.exceedance_fraction == f
                       and r.estimator == 'ESKF']
            if matching:
                rate = sum(1 for r in matching if r.detected) / len(matching) * 100
                f_values.append(f)
                rates.append(rate)
        ax.plot(f_values, rates, '-o', color=sensor_colors[sensor],
               label=sensor_names[sensor], markersize=8, linewidth=2)

    ax.set_xlabel('Exceedance Fraction $f$')
    ax.set_ylabel('Detection Rate [%]')
    ax.set_title('(a) ESKF: Detection Rate vs. $f$')
    ax.set_ylim([-5, 105])
    ax.set_xticks(EXCEEDANCE_FRACTIONS)
    ax.legend()
    ax.grid(True, alpha=0.4)

    # (b) Detection latency vs Nh for ESKF
    ax = axes[0, 1]
    for sensor in sensors:
        Nh_values = []
        latencies = []
        for Nh in WINDOW_LENGTHS:
            matching = [r for r in results
                       if r.sensor == sensor
                       and r.window_length == Nh
                       and r.estimator == 'ESKF'
                       and r.detected
                       and r.detection_latency is not None]
            if matching:
                avg_lat = np.mean([r.detection_latency for r in matching])
                Nh_values.append(Nh)
                latencies.append(avg_lat)
        if Nh_values:
            ax.plot(Nh_values, latencies, '-o', color=sensor_colors[sensor],
                   label=sensor_names[sensor], markersize=8, linewidth=2)

    ax.set_xlabel('Window Length $N_h$')
    ax.set_ylabel('Detection Latency [s]')
    ax.set_title('(b) ESKF: Latency vs. $N_h$')
    ax.set_xticks(WINDOW_LENGTHS)
    ax.legend()
    ax.grid(True, alpha=0.4)

    # (c) False positives heatmap for ESKF
    ax = axes[1, 0]
    fp_matrix = np.zeros((len(WINDOW_LENGTHS), len(EXCEEDANCE_FRACTIONS)))
    for i, Nh in enumerate(WINDOW_LENGTHS):
        for j, f in enumerate(EXCEEDANCE_FRACTIONS):
            matching = [r for r in results
                       if r.window_length == Nh
                       and r.exceedance_fraction == f
                       and r.estimator == 'ESKF']
            if matching:
                fp_matrix[i, j] = sum(r.false_positives for r in matching)

    im = ax.imshow(fp_matrix, cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(EXCEEDANCE_FRACTIONS)))
    ax.set_xticklabels([str(f) for f in EXCEEDANCE_FRACTIONS])
    ax.set_yticks(range(len(WINDOW_LENGTHS)))
    ax.set_yticklabels([str(Nh) for Nh in WINDOW_LENGTHS])
    ax.set_xlabel('Exceedance Fraction $f$')
    ax.set_ylabel('Window Length $N_h$')
    ax.set_title('(c) ESKF: Total False Positives')

    # Add text annotations
    for i in range(len(WINDOW_LENGTHS)):
        for j in range(len(EXCEEDANCE_FRACTIONS)):
            text = ax.text(j, i, f'{int(fp_matrix[i, j])}',
                          ha='center', va='center', color='black', fontsize=12)
    plt.colorbar(im, ax=ax, label='False Positives')

    # (d) Summary table as text
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    summary_lines = [
        "Summary of Sensitivity Analysis",
        "=" * 40,
        "",
        f"Fault window: {FAULT_START}s - {FAULT_END}s",
        f"Chi-squared threshold: {CHI2_THRESHOLD} (95%, 3 DOF)",
        "",
        "Fault magnitudes:",
        f"  Magnetometer: {FAULT_MAGNITUDES['mag']}",
        f"  Sun Sensor: {FAULT_MAGNITUDES['sun']}",
        f"  Star Tracker: {FAULT_MAGNITUDES['st']} rad ({np.rad2deg(FAULT_MAGNITUDES['st']):.1f} deg)",
        "",
        "Key findings:",
    ]

    # Compute best parameters
    best_detection = None
    best_latency = None
    min_fp = float('inf')

    for Nh in WINDOW_LENGTHS:
        for f in EXCEEDANCE_FRACTIONS:
            matching = [r for r in results
                       if r.window_length == Nh and r.exceedance_fraction == f
                       and r.estimator == 'ESKF']
            if matching:
                det_rate = sum(1 for r in matching if r.detected) / len(matching)
                fp = sum(r.false_positives for r in matching)
                latencies = [r.detection_latency for r in matching
                            if r.detected and r.detection_latency is not None]
                avg_lat = np.mean(latencies) if latencies else float('inf')

                if det_rate == 1.0 and fp < min_fp:
                    min_fp = fp
                    best_detection = (Nh, f, avg_lat)

    if best_detection:
        summary_lines.append(f"  Best params: Nh={best_detection[0]}, f={best_detection[1]}")
        summary_lines.append(f"  Avg latency: {best_detection[2]:.2f}s")

    summary_text = '\n'.join(summary_lines)
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('nis_sensitivity_combined.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('nis_sensitivity_combined.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_latex_table(results: List[SensitivityResult]) -> str:
    """Generate LaTeX table for sensitivity results."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{NIS-based fault detection sensitivity analysis}
\label{tab:nis_sensitivity}
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Sensor} & \textbf{$N_h$} & \textbf{$f$} & \textbf{ESKF Det.} & \textbf{ESKF Lat. [s]} & \textbf{Red. Det.} & \textbf{Red. Lat. [s]} \\
\midrule
"""

    for sensor in ['mag', 'sun', 'st']:
        sensor_name = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}[sensor]
        first_row = True

        for Nh in WINDOW_LENGTHS:
            for f in EXCEEDANCE_FRACTIONS:
                eskf_results = [r for r in results
                               if r.sensor == sensor and r.window_length == Nh
                               and r.exceedance_fraction == f and r.estimator == 'ESKF']
                red_results = [r for r in results
                              if r.sensor == sensor and r.window_length == Nh
                              and r.exceedance_fraction == f and r.estimator == 'Redundant']

                if eskf_results and red_results:
                    e = eskf_results[0]
                    r = red_results[0]

                    sensor_str = sensor_name if first_row else ""
                    first_row = False

                    e_det = "\\checkmark" if e.detected else "--"
                    e_lat = f"{e.detection_latency:.2f}" if e.detection_latency is not None else "--"
                    r_det = "\\checkmark" if r.detected else "--"
                    r_lat = f"{r.detection_latency:.2f}" if r.detection_latency is not None else "--"

                    latex += f"{sensor_str} & {Nh} & {f} & {e_det} & {e_lat} & {r_det} & {r_lat} \\\\\n"

        if sensor != 'st':
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    return latex


# =============================================================================
# Main
# =============================================================================

def main():
    db_path = "simulations.db"

    # Get simulation run
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()

    db = SimulationDatabase(db_path)
    sim_data = db.load_run(run_id)

    print("=" * 80)
    print("NIS-BASED FAULT DETECTION SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Fault window: {FAULT_START}s - {FAULT_END}s")
    print(f"Window lengths Nh: {WINDOW_LENGTHS}")
    print(f"Exceedance fractions f: {EXCEEDANCE_FRACTIONS}")
    print(f"Fault magnitudes: {FAULT_MAGNITUDES}")
    print()

    all_results = []

    # Run parameter sweep
    total_tests = len(['mag', 'sun', 'st']) * len(WINDOW_LENGTHS) * len(EXCEEDANCE_FRACTIONS)
    test_num = 0

    for sensor in ['mag', 'sun', 'st']:
        sensor_name = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}[sensor]
        print(f"\nTesting {sensor_name} faults...")

        for Nh, f in product(WINDOW_LENGTHS, EXCEEDANCE_FRACTIONS):
            test_num += 1
            print(f"  [{test_num}/{total_tests}] Nh={Nh}, f={f}...", end=" ", flush=True)

            results = run_sensitivity_test(sim_data, sensor, Nh, f)
            all_results.extend(results)
            print("done")

    # Save results
    save_results_csv(all_results, 'nis_sensitivity_results.csv')
    save_results_sqlite(all_results, 'nis_sensitivity_results.db')
    print("\nSaved: nis_sensitivity_results.csv")
    print("Saved: nis_sensitivity_results.db")

    # Generate plots
    print("\nGenerating plots...")
    plot_detection_rate_vs_f(all_results, 'ESKF')
    plot_detection_rate_vs_f(all_results, 'Redundant')
    plot_latency_vs_Nh(all_results, 'ESKF')
    plot_latency_vs_Nh(all_results, 'Redundant')
    plot_combined_sensitivity(all_results)

    print("Saved: sensitivity_detection_rate_vs_f_eskf.pdf/png")
    print("Saved: sensitivity_detection_rate_vs_f_redundant.pdf/png")
    print("Saved: sensitivity_latency_vs_Nh_eskf.pdf/png")
    print("Saved: sensitivity_latency_vs_Nh_redundant.pdf/png")
    print("Saved: nis_sensitivity_combined.pdf/png")

    # Generate LaTeX table
    latex_table = generate_latex_table(all_results)
    with open('nis_sensitivity_table.tex', 'w') as f:
        f.write(latex_table)
    print("Saved: nis_sensitivity_table.tex")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for sensor in ['mag', 'sun', 'st']:
        sensor_name = {'mag': 'Magnetometer', 'sun': 'Sun Sensor', 'st': 'Star Tracker'}[sensor]
        print(f"\n{sensor_name}:")
        print(f"{'Nh':>4} {'f':>6} {'ESKF Det':>10} {'ESKF Lat':>10} {'Red Det':>10} {'Red Lat':>10} {'FP':>6}")
        print("-" * 60)

        for Nh in WINDOW_LENGTHS:
            for f in EXCEEDANCE_FRACTIONS:
                eskf = [r for r in all_results
                       if r.sensor == sensor and r.window_length == Nh
                       and r.exceedance_fraction == f and r.estimator == 'ESKF']
                red = [r for r in all_results
                      if r.sensor == sensor and r.window_length == Nh
                      and r.exceedance_fraction == f and r.estimator == 'Redundant']

                if eskf and red:
                    e, r = eskf[0], red[0]
                    e_det = 'Yes' if e.detected else 'No'
                    e_lat = f'{e.detection_latency:.2f}s' if e.detection_latency else 'N/A'
                    r_det = 'Yes' if r.detected else 'No'
                    r_lat = f'{r.detection_latency:.2f}s' if r.detection_latency else 'N/A'
                    fp = e.false_positives + r.false_positives

                    print(f"{Nh:>4} {f:>6.1f} {e_det:>10} {e_lat:>10} {r_det:>10} {r_lat:>10} {fp:>6}")

    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(latex_table)


if __name__ == "__main__":
    main()
