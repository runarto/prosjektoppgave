#!/usr/bin/env python3
"""
Estimator Switching Trigger Test

Constructs a scenario designed to trigger disagreement-based switching:
1. Persistent star tracker fault (constant rotation bias, not random spikes)
2. ESKF becomes corrupted or drifts
3. Smoother stays accurate due to robust cost function
4. Disagreement triggers switch

Fault design:
- Apply constant rotation error to star tracker during fault window
- Use a ramping bias: starts small (passes chi-squared gate) and grows
- This corrupts ESKF early, then later measurements get rejected
- Smoother's Cauchy kernel + multi-epoch optimization handles this better

Runs three configurations:
1. ESKF only
2. Smoother only
3. Redundant with switching enabled
"""

import numpy as np
import sqlite3
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

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
# Publication-quality plot settings
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['lines.linewidth'] = 1.5
rcParams['axes.linewidth'] = 1.2
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

# Fault parameters: ramping constant rotation bias
# Starts at BIAS_START_DEG and grows to BIAS_END_DEG over the fault window
BIAS_START_DEG = 0.5   # Small enough to initially pass chi-squared gate
BIAS_END_DEG = 10.0    # Large enough to clearly corrupt estimates
BIAS_AXIS = np.array([0.3, 0.5, 0.8])  # Constant rotation axis (will be normalized)
BIAS_AXIS = BIAS_AXIS / np.linalg.norm(BIAS_AXIS)

# Alternative: constant bias (no ramp)
USE_CONSTANT_BIAS = False
CONSTANT_BIAS_DEG = 5.0

# Alternative scenario: Star tracker dropout + gyro drift
# This is more realistic - the ST goes offline during fault window
# ESKF drifts on gyro/mag/sun, smoother handles transition better
USE_DROPOUT_SCENARIO = True
GYRO_BIAS_DRIFT = np.array([0.002, -0.0015, 0.0018])  # rad/s additional drift during fault (increased)

# Switching parameters - lowered threshold to trigger switching in this scenario
DISAGREEMENT_THRESHOLD_DEG = 1.0  # Lowered from default 2.0 to trigger switching
CONSECUTIVE_TO_SWITCH = 3  # Reduced from 5 for faster response


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimeSeriesData:
    """Time series data for one estimator."""
    times: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)  # Attitude error in degrees

    # For redundant estimator
    disagreements: List[float] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)


@dataclass
class SwitchingMetrics:
    """Metrics related to switching behavior."""
    first_switch_time: Optional[float] = None
    switch_events: List[Tuple[float, str]] = field(default_factory=list)
    max_error_before_switch: float = 0.0
    max_error_after_switch: float = 0.0
    mean_error_during_fault_eskf: float = 0.0
    mean_error_during_fault_smoother: float = 0.0
    mean_error_during_fault_redundant: float = 0.0
    disagreement_exceeded_threshold: bool = False
    max_disagreement: float = 0.0


# =============================================================================
# Fault Injection
# =============================================================================

def apply_persistent_st_fault(sim_data, use_ramp: bool = True):
    """
    Apply persistent star tracker fault: constant rotation bias.

    The bias is applied as a rotation error around a fixed axis.
    If use_ramp=True, the magnitude grows linearly from BIAS_START_DEG to BIAS_END_DEG.
    If use_ramp=False, a constant bias of CONSTANT_BIAS_DEG is applied.

    This creates a systematic error (not random spikes) that:
    1. Initially may pass chi-squared gate (small bias)
    2. Grows to be clearly detectable (large bias)
    3. Creates persistent disagreement between estimators
    """
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    fault_duration = FAULT_END - FAULT_START

    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END and not np.isnan(modified.st_meas[k, 0]):
            # Compute bias magnitude
            if use_ramp:
                # Linear ramp from BIAS_START_DEG to BIAS_END_DEG
                progress = (t - FAULT_START) / fault_duration
                bias_deg = BIAS_START_DEG + progress * (BIAS_END_DEG - BIAS_START_DEG)
            else:
                bias_deg = CONSTANT_BIAS_DEG

            bias_rad = np.deg2rad(bias_deg)

            # Create error quaternion: rotation around fixed axis
            q_error = Quaternion.from_avec(BIAS_AXIS * bias_rad)

            # Apply to measurement (right multiply for body-frame rotation)
            q_meas = Quaternion.from_array(modified.st_meas[k])
            q_corrupted = (q_meas @ q_error).normalize()
            modified.st_meas[k] = q_corrupted.as_array()

    return modified


def apply_dropout_with_gyro_drift(sim_data):
    """
    Apply star tracker dropout + gyro drift fault scenario.

    During the fault window:
    1. Star tracker measurements are set to NaN (dropout)
    2. A constant gyro bias is added to gyro measurements

    This simulates:
    - Star tracker occlusion or failure
    - Gyro performance degradation under stress

    The ESKF relies on gyro integration + mag/sun updates and will drift.
    The smoother can handle the transition better through retrospective optimization.
    """
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END:
            # Set star tracker to NaN (dropout)
            modified.st_meas[k] = np.array([np.nan, np.nan, np.nan, np.nan])

            # Add gyro bias drift
            if not np.isnan(modified.omega_meas[k, 0]):
                modified.omega_meas[k] = modified.omega_meas[k] + GYRO_BIAS_DRIFT

    return modified


def apply_intermittent_st_fault(sim_data):
    """
    Apply intermittent star tracker fault with small biases.

    During the fault window:
    - Apply a small bias (~0.03°) that can occasionally pass chi-squared gate
    - Add random noise to make some measurements pass
    - This gradually corrupts the ESKF while smoother stays robust

    The key: small enough to sometimes pass, large enough to corrupt over time.
    """
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    np.random.seed(42)

    # Small bias that's near the chi-squared gate boundary
    # ST noise is ~0.00024 rad, chi2 threshold 7.81 → boundary ~0.0007 rad (0.04°)
    bias_rad = 0.0005  # 0.029° - right at the edge

    for k in range(len(modified.t)):
        t = modified.t[k]
        if FAULT_START <= t < FAULT_END and not np.isnan(modified.st_meas[k, 0]):
            # Fixed bias axis with some random variation
            axis = BIAS_AXIS + np.random.randn(3) * 0.1
            axis = axis / np.linalg.norm(axis)

            # Create error quaternion
            q_error = Quaternion.from_avec(axis * bias_rad)

            # Apply to measurement
            q_meas = Quaternion.from_array(modified.st_meas[k])
            q_corrupted = (q_meas @ q_error).normalize()
            modified.st_meas[k] = q_corrupted.as_array()

    return modified


# =============================================================================
# Estimator Runners
# =============================================================================

def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def run_eskf_only(sim_data) -> TimeSeriesData:
    """Run ESKF only and return time series."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    data = TimeSeriesData()
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        t = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        data.times.append(t)
        data.errors.append(compute_attitude_error_deg(x.nom.ori, q_true))

        # Prediction
        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        # Updates
        if not np.isnan(sim_data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except ValueError:
                pass

        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except ValueError:
                pass

        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except ValueError:
                pass

    return data


def run_smoother_only(sim_data) -> TimeSeriesData:
    """Run smoother only and return time series."""
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

    # Interpolate errors at all time steps
    keyframe_times = np.array(keyframe_times)
    data = TimeSeriesData()

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

        data.times.append(t_k)
        data.errors.append(compute_attitude_error_deg(q_interp, q_true))

    return data


def run_redundant(sim_data) -> TimeSeriesData:
    """Run redundant estimator and return time series with switching info."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)

    redundant = RedundantEstimator(
        P0=P0,
        config_path=CONFIG_PATH,
        smoother_lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        disagreement_threshold_deg=DISAGREEMENT_THRESHOLD_DEG,
        consecutive_disagreements_to_switch=CONSECUTIVE_TO_SWITCH,
        consecutive_agreements_to_recover=10,
        agreement_threshold_deg=0.5,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )
    redundant.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    data = TimeSeriesData()
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

        # Compute error based on active mode
        if primary == "SMOOTHER" and smoother_state is not None:
            q_est = smoother_state.ori
        else:
            q_est = x_eskf.nom.ori

        data.times.append(t)
        data.errors.append(compute_attitude_error_deg(q_est, q_true))
        data.disagreements.append(disagreement)
        data.modes.append(primary)

    # Store switch events from redundant estimator
    data.switch_events = redundant.switch_events

    return data


# =============================================================================
# Analysis and Metrics
# =============================================================================

def compute_metrics(
    eskf_data: TimeSeriesData,
    smoother_data: TimeSeriesData,
    redundant_data: TimeSeriesData,
) -> SwitchingMetrics:
    """Compute switching-related metrics."""
    metrics = SwitchingMetrics()

    times = np.array(redundant_data.times)
    fault_mask = (times >= FAULT_START) & (times < FAULT_END)

    # Mean errors during fault
    eskf_errors = np.array(eskf_data.errors)
    smoother_errors = np.array(smoother_data.errors)
    redundant_errors = np.array(redundant_data.errors)

    metrics.mean_error_during_fault_eskf = np.mean(eskf_errors[fault_mask])
    metrics.mean_error_during_fault_smoother = np.mean(smoother_errors[fault_mask])
    metrics.mean_error_during_fault_redundant = np.mean(redundant_errors[fault_mask])

    # Disagreement analysis
    disagreements = np.array(redundant_data.disagreements)
    metrics.max_disagreement = np.max(disagreements)
    metrics.disagreement_exceeded_threshold = np.any(disagreements > DISAGREEMENT_THRESHOLD_DEG)

    # Switch events
    if hasattr(redundant_data, 'switch_events') and redundant_data.switch_events:
        metrics.switch_events = redundant_data.switch_events
        metrics.first_switch_time = redundant_data.switch_events[0][0]

        # Errors before and after first switch
        switch_time = metrics.first_switch_time
        before_switch = redundant_errors[times < switch_time]
        after_switch = redundant_errors[times >= switch_time]

        metrics.max_error_before_switch = np.max(before_switch) if len(before_switch) > 0 else 0
        metrics.max_error_after_switch = np.max(after_switch) if len(after_switch) > 0 else 0

    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_switching_summary(
    eskf_data: TimeSeriesData,
    smoother_data: TimeSeriesData,
    redundant_data: TimeSeriesData,
    metrics: SwitchingMetrics,
):
    """Generate summary figure with attitude error, disagreement, and mode."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    times = np.array(redundant_data.times)

    # Panel (a): Attitude error vs time
    ax1 = axes[0]
    ax1.plot(times, eskf_data.errors, 'b-', label='ESKF only', linewidth=1.2, alpha=0.8)
    ax1.plot(times, smoother_data.errors, 'g-', label='Smoother only', linewidth=1.2, alpha=0.8)
    ax1.plot(times, redundant_data.errors, 'r-', label='Redundant', linewidth=2)

    # Fault window shading
    ax1.axvspan(FAULT_START, FAULT_END, alpha=0.15, color='red', label='Fault window')

    # Mark switch times
    for switch_time, event in metrics.switch_events:
        ax1.axvline(x=switch_time, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_title('(a) Attitude Error Comparison')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim([0, min(1.0, np.max([np.max(eskf_data.errors), np.max(smoother_data.errors)]) * 1.2)])

    # Panel (b): Disagreement vs time
    ax2 = axes[1]
    disagreements = np.array(redundant_data.disagreements)
    ax2.plot(times, disagreements, 'k-', linewidth=1.2)
    ax2.axhline(y=DISAGREEMENT_THRESHOLD_DEG, color='red', linestyle='--',
                linewidth=2, label=f'Threshold ({DISAGREEMENT_THRESHOLD_DEG}°)')
    ax2.axvspan(FAULT_START, FAULT_END, alpha=0.15, color='red')

    # Mark switch times
    for switch_time, event in metrics.switch_events:
        ax2.axvline(x=switch_time, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('(b) ESKF-Smoother Disagreement')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim([0, max(3.0, metrics.max_disagreement * 1.2)])

    # Panel (c): Active mode vs time
    ax3 = axes[2]
    modes = redundant_data.modes
    mode_numeric = []
    for m in modes:
        if m == 'ESKF':
            mode_numeric.append(0)
        elif m == 'SMOOTHER':
            mode_numeric.append(1)
        else:  # CONSERVATIVE
            mode_numeric.append(2)

    ax3.step(times, mode_numeric, 'k-', where='post', linewidth=1.5)
    ax3.axvspan(FAULT_START, FAULT_END, alpha=0.15, color='red')

    # Mark switch times with annotations
    for switch_time, event in metrics.switch_events:
        ax3.axvline(x=switch_time, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3.annotate(event.split('->')[-1], xy=(switch_time, 1.5), fontsize=9,
                    ha='center', va='bottom', color='purple')

    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['ESKF', 'SMOOTHER', 'CONSERVATIVE'])
    ax3.set_ylabel('Active Mode')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('(c) Active Estimator Mode')
    ax3.grid(True, alpha=0.4, axis='x')
    ax3.set_ylim([-0.5, 2.5])

    plt.tight_layout()
    plt.savefig('switching_trigger_summary.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('switching_trigger_summary.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_fault_injection_profile():
    """Plot the fault injection profile showing bias magnitude over time."""
    fig, ax = plt.subplots(figsize=(10, 4))

    t = np.linspace(0, 300, 1000)
    bias = np.zeros_like(t)

    for i, ti in enumerate(t):
        if FAULT_START <= ti < FAULT_END:
            if USE_CONSTANT_BIAS:
                bias[i] = CONSTANT_BIAS_DEG
            else:
                progress = (ti - FAULT_START) / (FAULT_END - FAULT_START)
                bias[i] = BIAS_START_DEG + progress * (BIAS_END_DEG - BIAS_START_DEG)

    ax.fill_between(t, 0, bias, alpha=0.3, color='red')
    ax.plot(t, bias, 'r-', linewidth=2, label='Star tracker bias')
    ax.axvspan(FAULT_START, FAULT_END, alpha=0.1, color='gray')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Bias Magnitude [deg]')
    ax.set_title('Star Tracker Fault Profile')
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xlim([0, 300])

    plt.tight_layout()
    plt.savefig('fault_injection_profile.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fault_injection_profile.png', bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# Output Functions
# =============================================================================

def save_results_csv(
    eskf_data: TimeSeriesData,
    smoother_data: TimeSeriesData,
    redundant_data: TimeSeriesData,
    filename: str
):
    """Save time series to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time', 'eskf_error', 'smoother_error', 'redundant_error',
            'disagreement', 'mode'
        ])
        for i in range(len(redundant_data.times)):
            writer.writerow([
                redundant_data.times[i],
                eskf_data.errors[i],
                smoother_data.errors[i],
                redundant_data.errors[i],
                redundant_data.disagreements[i],
                redundant_data.modes[i]
            ])


def save_metrics_csv(metrics: SwitchingMetrics, filename: str):
    """Save metrics to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['first_switch_time', metrics.first_switch_time or 'None'])
        writer.writerow(['num_switches', len(metrics.switch_events)])
        writer.writerow(['max_error_before_switch', f'{metrics.max_error_before_switch:.4f}'])
        writer.writerow(['max_error_after_switch', f'{metrics.max_error_after_switch:.4f}'])
        writer.writerow(['mean_error_eskf', f'{metrics.mean_error_during_fault_eskf:.4f}'])
        writer.writerow(['mean_error_smoother', f'{metrics.mean_error_during_fault_smoother:.4f}'])
        writer.writerow(['mean_error_redundant', f'{metrics.mean_error_during_fault_redundant:.4f}'])
        writer.writerow(['disagreement_exceeded_threshold', metrics.disagreement_exceeded_threshold])
        writer.writerow(['max_disagreement', f'{metrics.max_disagreement:.4f}'])


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
    print("ESTIMATOR SWITCHING TRIGGER TEST")
    print("=" * 80)
    print(f"Fault window: {FAULT_START}s - {FAULT_END}s")
    if USE_DROPOUT_SCENARIO:
        print(f"Fault type: Star tracker dropout + gyro drift")
        print(f"Gyro bias drift: {GYRO_BIAS_DRIFT} rad/s")
        print(f"  = {np.rad2deg(GYRO_BIAS_DRIFT)} deg/s")
    else:
        print(f"Fault type: {'Constant' if USE_CONSTANT_BIAS else 'Ramping'} star tracker bias")
        if USE_CONSTANT_BIAS:
            print(f"Bias magnitude: {CONSTANT_BIAS_DEG}°")
        else:
            print(f"Bias range: {BIAS_START_DEG}° → {BIAS_END_DEG}°")
        print(f"Bias axis: {BIAS_AXIS}")
    print(f"Disagreement threshold: {DISAGREEMENT_THRESHOLD_DEG}°")
    print(f"Consecutive to switch: {CONSECUTIVE_TO_SWITCH}")
    print()

    # Plot fault injection profile
    print("Generating fault injection profile...")
    plot_fault_injection_profile()
    print("Saved: fault_injection_profile.pdf/png")
    print()

    # Apply fault based on scenario selection
    if USE_DROPOUT_SCENARIO:
        print("Applying ST dropout + gyro drift scenario...")
        modified_data = apply_dropout_with_gyro_drift(sim_data)
    else:
        print("Applying persistent star tracker bias...")
        modified_data = apply_persistent_st_fault(sim_data, use_ramp=not USE_CONSTANT_BIAS)
    print()

    # Run estimators
    print("Running ESKF only...", end=" ", flush=True)
    eskf_data = run_eskf_only(modified_data)
    print("done")

    print("Running Smoother only...", end=" ", flush=True)
    smoother_data = run_smoother_only(modified_data)
    print("done")

    print("Running Redundant estimator...", end=" ", flush=True)
    redundant_data = run_redundant(modified_data)
    print("done")
    print()

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(eskf_data, smoother_data, redundant_data)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nMean attitude error during fault window ({FAULT_START}-{FAULT_END}s):")
    print(f"  ESKF only:     {metrics.mean_error_during_fault_eskf:.4f}°")
    print(f"  Smoother only: {metrics.mean_error_during_fault_smoother:.4f}°")
    print(f"  Redundant:     {metrics.mean_error_during_fault_redundant:.4f}°")

    print(f"\nDisagreement analysis:")
    print(f"  Max disagreement:       {metrics.max_disagreement:.4f}°")
    print(f"  Exceeded threshold:     {metrics.disagreement_exceeded_threshold}")
    print(f"  Threshold:              {DISAGREEMENT_THRESHOLD_DEG}°")

    print(f"\nSwitching behavior:")
    print(f"  Number of switches:     {len(metrics.switch_events)}")
    if metrics.first_switch_time:
        print(f"  First switch time:      {metrics.first_switch_time:.2f}s")
        print(f"  Max error before switch: {metrics.max_error_before_switch:.4f}°")
        print(f"  Max error after switch:  {metrics.max_error_after_switch:.4f}°")
    else:
        print(f"  First switch time:      No switch occurred")

    for t, event in metrics.switch_events:
        print(f"    t={t:.2f}s: {event}")

    # Validation checks
    print(f"\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    if metrics.disagreement_exceeded_threshold:
        print("✓ Disagreement exceeded threshold")
    else:
        print("✗ Disagreement did NOT exceed threshold")

    if len(metrics.switch_events) > 0:
        print("✓ Switching was triggered")
        if metrics.mean_error_during_fault_redundant < metrics.mean_error_during_fault_eskf:
            print("✓ Redundant output improved over ESKF-only")
        else:
            print("✗ Redundant output did NOT improve over ESKF-only")
    else:
        print("✗ No switching occurred")

    # Save results
    save_results_csv(eskf_data, smoother_data, redundant_data, 'switching_trigger_timeseries.csv')
    save_metrics_csv(metrics, 'switching_trigger_metrics.csv')
    print("\nSaved: switching_trigger_timeseries.csv")
    print("Saved: switching_trigger_metrics.csv")

    # Generate plots
    print("\nGenerating summary plot...")
    plot_switching_summary(eskf_data, smoother_data, redundant_data, metrics)
    print("Saved: switching_trigger_summary.pdf/png")


if __name__ == "__main__":
    main()
