#!/usr/bin/env python3
"""
Fault Detection Evaluation Framework

Tests transient sensor faults with configurable magnitudes:
- Sun sensor: perturbation magnitudes {0.5, 1.0, 2.0, 3.0}
- Magnetometer: perturbation magnitudes {0.5, 1.0, 2.0, 3.0}
- Star tracker: rotation errors (rad) {0.05, 0.2, 0.5, 1.5}

Fault window: t=100s to t=150s (100% fault probability)

Reports:
- Detection success/failure
- Detection latency
- False detections on healthy sensors
"""

import numpy as np
import sqlite3
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = "configs/config_baseline_short.yaml"

FAULT_START = 100.0  # seconds
FAULT_END = 150.0    # seconds

# Fault magnitudes per sensor type
SUN_MAGNITUDES = [0.5, 1.0, 2.0, 3.0]
MAG_MAGNITUDES = [0.5, 1.0, 2.0, 3.0]
ST_MAGNITUDES = [0.05, 0.2, 0.5, 1.5]  # radians

# Detection parameters
CHI2_THRESHOLD_MAG = 7.81   # 3 DOF, 95%
CHI2_THRESHOLD_SUN = 7.81   # 3 DOF, 95%
CHI2_THRESHOLD_ST = 7.81    # 3 DOF, 95%
DETECTION_WINDOW = 10       # samples for sliding window
DETECTION_FRACTION = 0.6    # fraction of window exceeding threshold to flag


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SensorNISTracker:
    """Track NIS values and detection state for a single sensor."""
    name: str
    threshold: float
    window_size: int = DETECTION_WINDOW
    detection_fraction: float = DETECTION_FRACTION

    nis_history: deque = field(default_factory=lambda: deque(maxlen=DETECTION_WINDOW))
    times: List[float] = field(default_factory=list)
    nis_values: List[float] = field(default_factory=list)
    detection_flags: List[bool] = field(default_factory=list)

    first_detection_time: Optional[float] = None
    is_detected: bool = False

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
class FaultDetectionResult:
    """Result of a single fault detection test."""
    sensor_type: str
    fault_magnitude: float
    estimator: str

    # Detection metrics
    detected: bool
    detection_latency: Optional[float]  # seconds after fault start
    false_detections_mag: int
    false_detections_sun: int
    false_detections_st: int

    # Performance metrics
    mean_error_during_fault: float
    mean_error_after_fault: float
    max_error_during_fault: float


# =============================================================================
# Fault Injection
# =============================================================================

def apply_sun_fault(sim_data, magnitude: float):
    """Apply transient sun sensor fault during window."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)  # Reproducible
    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END:
            if not np.isnan(modified.sun_meas[k, 0]):
                # Add random perturbation vector before normalization
                perturbation = np.random.randn(3)
                perturbation = perturbation / np.linalg.norm(perturbation) * magnitude
                modified.sun_meas[k] = modified.sun_meas[k] + perturbation
                # Re-normalize
                modified.sun_meas[k] = modified.sun_meas[k] / np.linalg.norm(modified.sun_meas[k])

    return modified


def apply_mag_fault(sim_data, magnitude: float):
    """Apply transient magnetometer fault during window."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)
    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END:
            if not np.isnan(modified.mag_meas[k, 0]):
                perturbation = np.random.randn(3)
                perturbation = perturbation / np.linalg.norm(perturbation) * magnitude
                modified.mag_meas[k] = modified.mag_meas[k] + perturbation
                modified.mag_meas[k] = modified.mag_meas[k] / np.linalg.norm(modified.mag_meas[k])

    return modified


def apply_st_fault(sim_data, magnitude_rad: float):
    """Apply transient star tracker fault during window."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)
    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END:
            if not np.isnan(modified.st_meas[k, 0]):
                # Random rotation axis
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                # Create error quaternion
                q_error = Quaternion.from_avec(axis * magnitude_rad)
                # Apply to measurement
                q_meas = Quaternion.from_array(modified.st_meas[k])
                q_corrupted = (q_meas @ q_error).normalize()
                modified.st_meas[k] = q_corrupted.as_array()

    return modified


# =============================================================================
# Estimator Runners with NIS Tracking
# =============================================================================

def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def run_eskf_with_nis_tracking(sim_data) -> Tuple[Dict[str, SensorNISTracker], List[float], List[float]]:
    """
    Run ESKF and track per-sensor NIS values.

    Returns:
        - Dict of sensor NIS trackers
        - Times list
        - Errors list
    """
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    # Initialize trackers
    trackers = {
        'mag': SensorNISTracker('magnetometer', CHI2_THRESHOLD_MAG),
        'sun': SensorNISTracker('sun_sensor', CHI2_THRESHOLD_SUN),
        'st': SensorNISTracker('star_tracker', CHI2_THRESHOLD_ST),
    }

    times = []
    errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        times.append(t)
        errors.append(compute_attitude_error_deg(x.nom.ori, q_true))

        # Prediction
        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        # Magnetometer update
        if not np.isnan(sim_data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
                trackers['mag'].update(t, eskf.last_nis)
            except ValueError:
                # Rejected - still record the NIS
                trackers['mag'].update(t, eskf.last_nis)

        # Sun sensor update
        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
                trackers['sun'].update(t, eskf.last_nis)
            except ValueError:
                trackers['sun'].update(t, eskf.last_nis)

        # Star tracker update
        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
                trackers['st'].update(t, eskf.last_nis)
            except ValueError:
                trackers['st'].update(t, eskf.last_nis)

    return trackers, times, errors


def run_smoother_with_tracking(sim_data) -> Tuple[List[float], List[float]]:
    """Run smoother and return times and errors."""
    smoother = FixedLagAttitudeSmoother(
        config_path=CONFIG_PATH,
        lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        normalize_mag=True,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    smoother.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    dt = sim_data.t[1] - sim_data.t[0]
    keyframe_times = [sim_data.t[0]]
    keyframe_states = [q_init.copy()]

    for k in range(len(sim_data.t)):
        if not np.isnan(sim_data.omega_meas[k, 0]):
            smoother.integrate_gyro(sim_data.omega_meas[k], dt, t=sim_data.t[k])

        z_mag = sim_data.mag_meas[k] if not np.isnan(sim_data.mag_meas[k, 0]) else None
        z_sun = sim_data.sun_meas[k] if not np.isnan(sim_data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.isnan(sim_data.st_meas[k, 0]) else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            smoother.add_measurement(
                t=sim_data.t[k], jd=sim_data.jd[k],
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=sim_data.b_eci[k], s_eci=sim_data.s_eci[k],
            )
            state = smoother.get_state()
            if state is not None:
                keyframe_times.append(sim_data.t[k])
                keyframe_states.append(state.ori.copy())

    # Interpolate errors
    keyframe_times = np.array(keyframe_times)
    times = []
    errors = []

    for k in range(len(sim_data.t)):
        t_k = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        idx = np.searchsorted(keyframe_times, t_k)
        if idx == 0:
            q_interp = keyframe_states[0]
        elif idx >= len(keyframe_times):
            q_interp = keyframe_states[-1]
        else:
            t0, t1 = keyframe_times[idx-1], keyframe_times[idx]
            q0, q1 = keyframe_states[idx-1], keyframe_states[idx]
            alpha = (t_k - t0) / (t1 - t0) if t1 > t0 else 0
            q_interp = q0.slerp(q1, alpha)

        times.append(t_k)
        errors.append(compute_attitude_error_deg(q_interp, q_true))

    return times, errors


def run_redundant_with_tracking(sim_data) -> Tuple[Dict[str, SensorNISTracker], List[float], List[float], List[str]]:
    """
    Run redundant estimator and track NIS values and primary estimator.
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

    # Use the internal ESKF's trackers
    trackers = {
        'mag': SensorNISTracker('magnetometer', CHI2_THRESHOLD_MAG),
        'sun': SensorNISTracker('sun_sensor', CHI2_THRESHOLD_SUN),
        'st': SensorNISTracker('star_tracker', CHI2_THRESHOLD_ST),
    }

    times = []
    errors = []
    primaries = []
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

        times.append(t)
        errors.append(compute_attitude_error_deg(q_est, q_true))
        primaries.append(primary)

    return trackers, times, errors, primaries


# =============================================================================
# Test Runner
# =============================================================================

def run_fault_test(
    sim_data,
    sensor_type: str,
    fault_magnitude: float,
) -> List[FaultDetectionResult]:
    """
    Run fault test for a single sensor-magnitude combination.

    Returns results for all three estimators.
    """
    # Apply fault
    if sensor_type == 'sun':
        modified_data = apply_sun_fault(sim_data, fault_magnitude)
        faulty_sensor = 'sun'
    elif sensor_type == 'mag':
        modified_data = apply_mag_fault(sim_data, fault_magnitude)
        faulty_sensor = 'mag'
    elif sensor_type == 'st':
        modified_data = apply_st_fault(sim_data, fault_magnitude)
        faulty_sensor = 'st'
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    results = []

    # Run ESKF
    trackers_eskf, times_eskf, errors_eskf = run_eskf_with_nis_tracking(modified_data)

    # Compute metrics
    times_arr = np.array(times_eskf)
    errors_arr = np.array(errors_eskf)

    fault_mask = (times_arr >= FAULT_START) & (times_arr < FAULT_END)
    after_mask = times_arr >= FAULT_END

    faulty_tracker = trackers_eskf[faulty_sensor]
    detection_latency = None
    if faulty_tracker.first_detection_time is not None:
        if faulty_tracker.first_detection_time >= FAULT_START:
            detection_latency = faulty_tracker.first_detection_time - FAULT_START

    # Count false detections on OTHER sensors
    healthy_sensors = [s for s in ['mag', 'sun', 'st'] if s != faulty_sensor]
    false_mag = trackers_eskf['mag'].get_false_detections(FAULT_START, FAULT_END) if 'mag' in healthy_sensors else 0
    false_sun = trackers_eskf['sun'].get_false_detections(FAULT_START, FAULT_END) if 'sun' in healthy_sensors else 0
    false_st = trackers_eskf['st'].get_false_detections(FAULT_START, FAULT_END) if 'st' in healthy_sensors else 0

    results.append(FaultDetectionResult(
        sensor_type=sensor_type,
        fault_magnitude=fault_magnitude,
        estimator='ESKF',
        detected=faulty_tracker.is_detected,
        detection_latency=detection_latency,
        false_detections_mag=false_mag,
        false_detections_sun=false_sun,
        false_detections_st=false_st,
        mean_error_during_fault=np.mean(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
        mean_error_after_fault=np.mean(errors_arr[after_mask]) if np.any(after_mask) else 0,
        max_error_during_fault=np.max(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
    ))

    # Run Smoother
    times_sm, errors_sm = run_smoother_with_tracking(modified_data)
    times_arr = np.array(times_sm)
    errors_arr = np.array(errors_sm)

    fault_mask = (times_arr >= FAULT_START) & (times_arr < FAULT_END)
    after_mask = times_arr >= FAULT_END

    results.append(FaultDetectionResult(
        sensor_type=sensor_type,
        fault_magnitude=fault_magnitude,
        estimator='Smoother',
        detected=False,  # Smoother doesn't have NIS-based detection
        detection_latency=None,
        false_detections_mag=0,
        false_detections_sun=0,
        false_detections_st=0,
        mean_error_during_fault=np.mean(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
        mean_error_after_fault=np.mean(errors_arr[after_mask]) if np.any(after_mask) else 0,
        max_error_during_fault=np.max(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
    ))

    # Run Redundant
    trackers_red, times_red, errors_red, primaries = run_redundant_with_tracking(modified_data)
    times_arr = np.array(times_red)
    errors_arr = np.array(errors_red)

    fault_mask = (times_arr >= FAULT_START) & (times_arr < FAULT_END)
    after_mask = times_arr >= FAULT_END

    faulty_tracker = trackers_red[faulty_sensor]
    detection_latency = None
    if faulty_tracker.first_detection_time is not None:
        if faulty_tracker.first_detection_time >= FAULT_START:
            detection_latency = faulty_tracker.first_detection_time - FAULT_START

    healthy_sensors = [s for s in ['mag', 'sun', 'st'] if s != faulty_sensor]
    false_mag = trackers_red['mag'].get_false_detections(FAULT_START, FAULT_END) if 'mag' in healthy_sensors else 0
    false_sun = trackers_red['sun'].get_false_detections(FAULT_START, FAULT_END) if 'sun' in healthy_sensors else 0
    false_st = trackers_red['st'].get_false_detections(FAULT_START, FAULT_END) if 'st' in healthy_sensors else 0

    results.append(FaultDetectionResult(
        sensor_type=sensor_type,
        fault_magnitude=fault_magnitude,
        estimator='Redundant',
        detected=faulty_tracker.is_detected,
        detection_latency=detection_latency,
        false_detections_mag=false_mag,
        false_detections_sun=false_sun,
        false_detections_st=false_st,
        mean_error_during_fault=np.mean(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
        mean_error_after_fault=np.mean(errors_arr[after_mask]) if np.any(after_mask) else 0,
        max_error_during_fault=np.max(errors_arr[fault_mask]) if np.any(fault_mask) else 0,
    ))

    return results


def save_results_csv(results: List[FaultDetectionResult], filename: str):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sensor_type', 'fault_magnitude', 'estimator',
            'detected', 'detection_latency_s',
            'false_det_mag', 'false_det_sun', 'false_det_st',
            'mean_error_during_deg', 'mean_error_after_deg', 'max_error_during_deg'
        ])
        for r in results:
            writer.writerow([
                r.sensor_type, r.fault_magnitude, r.estimator,
                r.detected, r.detection_latency if r.detection_latency else '',
                r.false_detections_mag, r.false_detections_sun, r.false_detections_st,
                f'{r.mean_error_during_fault:.4f}',
                f'{r.mean_error_after_fault:.4f}',
                f'{r.max_error_during_fault:.4f}'
            ])


def save_results_sqlite(results: List[FaultDetectionResult], db_path: str):
    """Save results to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fault_detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor_type TEXT,
            fault_magnitude REAL,
            estimator TEXT,
            detected INTEGER,
            detection_latency_s REAL,
            false_det_mag INTEGER,
            false_det_sun INTEGER,
            false_det_st INTEGER,
            mean_error_during_deg REAL,
            mean_error_after_deg REAL,
            max_error_during_deg REAL
        )
    ''')

    timestamp = datetime.now().isoformat()

    for r in results:
        cursor.execute('''
            INSERT INTO fault_detection_results
            (timestamp, sensor_type, fault_magnitude, estimator, detected,
             detection_latency_s, false_det_mag, false_det_sun, false_det_st,
             mean_error_during_deg, mean_error_after_deg, max_error_during_deg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, r.sensor_type, r.fault_magnitude, r.estimator,
            1 if r.detected else 0, r.detection_latency,
            r.false_detections_mag, r.false_detections_sun, r.false_detections_st,
            r.mean_error_during_fault, r.mean_error_after_fault, r.max_error_during_fault
        ))

    conn.commit()
    conn.close()


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
    print("FAULT DETECTION EVALUATION")
    print("=" * 80)
    print(f"Fault window: {FAULT_START}s - {FAULT_END}s")
    print(f"Detection window: {DETECTION_WINDOW} samples")
    print(f"Detection fraction: {DETECTION_FRACTION}")
    print()

    all_results = []

    # Sun sensor faults
    print("Testing Sun Sensor Faults...")
    for mag in SUN_MAGNITUDES:
        print(f"  Magnitude {mag}...", end=" ", flush=True)
        results = run_fault_test(sim_data, 'sun', mag)
        all_results.extend(results)
        print("done")

    # Magnetometer faults
    print("\nTesting Magnetometer Faults...")
    for mag in MAG_MAGNITUDES:
        print(f"  Magnitude {mag}...", end=" ", flush=True)
        results = run_fault_test(sim_data, 'mag', mag)
        all_results.extend(results)
        print("done")

    # Star tracker faults
    print("\nTesting Star Tracker Faults...")
    for mag in ST_MAGNITUDES:
        print(f"  Magnitude {mag} rad ({np.rad2deg(mag):.1f} deg)...", end=" ", flush=True)
        results = run_fault_test(sim_data, 'st', mag)
        all_results.extend(results)
        print("done")

    # Save results
    save_results_csv(all_results, 'fault_detection_results.csv')
    save_results_sqlite(all_results, 'fault_detection_results.db')
    print("\nSaved: fault_detection_results.csv")
    print("Saved: fault_detection_results.db")

    # Print summary tables
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY BY SENSOR TYPE")
    print("=" * 80)

    for sensor in ['sun', 'mag', 'st']:
        sensor_results = [r for r in all_results if r.sensor_type == sensor]
        if not sensor_results:
            continue

        sensor_name = {'sun': 'Sun Sensor', 'mag': 'Magnetometer', 'st': 'Star Tracker'}[sensor]
        print(f"\n{sensor_name}:")
        print(f"{'Magnitude':>10} {'Estimator':>12} {'Detected':>10} {'Latency':>10} {'Error During':>12} {'Error After':>12}")
        print("-" * 70)

        for r in sensor_results:
            latency_str = f"{r.detection_latency:.2f}s" if r.detection_latency is not None else "N/A"
            print(f"{r.fault_magnitude:>10.2f} {r.estimator:>12} {'Yes' if r.detected else 'No':>10} "
                  f"{latency_str:>10} {r.mean_error_during_fault:>12.4f}° {r.mean_error_after_fault:>12.4f}°")

    # False detection summary
    print("\n" + "=" * 80)
    print("FALSE DETECTION SUMMARY")
    print("=" * 80)

    total_false = {'mag': 0, 'sun': 0, 'st': 0}
    for r in all_results:
        if r.estimator in ['ESKF', 'Redundant']:
            total_false['mag'] += r.false_detections_mag
            total_false['sun'] += r.false_detections_sun
            total_false['st'] += r.false_detections_st

    print(f"Total false detections on healthy sensors:")
    print(f"  Magnetometer: {total_false['mag']}")
    print(f"  Sun Sensor:   {total_false['sun']}")
    print(f"  Star Tracker: {total_false['st']}")

    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Fault detection evaluation: transient sensor faults (t=100-150s)}
\label{tab:fault_detection}
\begin{tabular}{@{}llcccc@{}}
\toprule
\textbf{Sensor} & \textbf{Magnitude} & \textbf{Estimator} & \textbf{Detected} & \textbf{Latency [s]} & \textbf{Error During} \\
\midrule
"""

    for sensor in ['sun', 'mag', 'st']:
        sensor_results = [r for r in all_results if r.sensor_type == sensor]
        sensor_name = {'sun': 'Sun', 'mag': 'Mag', 'st': 'ST'}[sensor]

        for i, r in enumerate(sensor_results):
            if i % 3 == 0:  # New magnitude group
                if r.sensor_type == 'st':
                    mag_str = f"{np.rad2deg(r.fault_magnitude):.1f}$^\\circ$"
                else:
                    mag_str = f"{r.fault_magnitude}"
            else:
                mag_str = ""

            sensor_str = sensor_name if i == 0 else ""
            latency_str = f"{r.detection_latency:.2f}" if r.detection_latency is not None else "--"
            detected_str = "\\checkmark" if r.detected else "--"

            latex += f"{sensor_str} & {mag_str} & {r.estimator} & {detected_str} & {latency_str} & {r.mean_error_during_fault:.3f}$^\\circ$ \\\\\n"

        if sensor != 'st':
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(latex)

    with open('fault_detection_evaluation.tex', 'w') as f:
        f.write(latex)
    print("\nSaved: fault_detection_evaluation.tex")


if __name__ == "__main__":
    main()
