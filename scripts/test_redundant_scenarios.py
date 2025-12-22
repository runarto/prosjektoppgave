#!/usr/bin/env python3
"""
Redundant Architecture Stress Testing

Tests scenarios that demonstrate the value of the redundant ESKF + iSAM2 architecture:
1. ESKF divergence from overconfident initialization
2. Sensor dropout and recovery
3. Measurement outliers/spikes

For each scenario, compares:
- ESKF only
- iSAM2 only
- Redundant architecture

Shows switching events and demonstrates when redundant outperforms standalone.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import EskfState, NominalState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel


@dataclass
class ScenarioResult:
    name: str
    times: np.ndarray
    errors: np.ndarray
    switch_times: List[float] = None
    switch_types: List[str] = None

    def ss_mean(self, start=30.0):
        mask = self.times > start
        return np.mean(self.errors[mask]) if np.any(mask) else np.nan

    def ss_max(self, start=30.0):
        mask = self.times > start
        return np.max(self.errors[mask]) if np.any(mask) else np.nan


def compute_attitude_error(q_true: Quaternion, q_est: Quaternion) -> float:
    dq = q_true @ q_est.conjugate()
    return np.rad2deg(2 * np.arccos(np.clip(abs(dq.mu), 0, 1)))


def run_eskf(sim, env, config_path, q_init, P0, inject_outliers=None):
    """Run standalone ESKF."""
    eskf = ESKF(P0=P0, config_path=config_path)
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    times, errors = [], []

    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        x = eskf.predict(x, omega, dt)

        # Magnetometer
        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        if z_mag is not None:
            # Inject outlier if specified
            if inject_outliers and inject_outliers['type'] == 'mag' and inject_outliers['start'] <= t <= inject_outliers['end']:
                z_mag = z_mag + np.random.randn(3) * inject_outliers['magnitude']
                z_mag = z_mag / np.linalg.norm(z_mag)

            B_n = env.get_B_eci(env.get_r_eci(jd), jd)
            try:
                x = eskf.update(x, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
            except:
                pass

        # Sun sensor
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        if z_sun is not None:
            s_n = env.get_sun_eci(jd)
            try:
                x = eskf.update(x, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
            except:
                pass

        # Star tracker (with possible dropout)
        z_st = None
        if not np.any(np.isnan(sim.st_meas[k])):
            if inject_outliers and inject_outliers['type'] == 'st_dropout':
                if not (inject_outliers['start'] <= t <= inject_outliers['end']):
                    z_st = Quaternion.from_array(sim.st_meas[k])
            else:
                z_st = Quaternion.from_array(sim.st_meas[k])

        if z_st is not None:
            try:
                x = eskf.update(x, z_st, SensorType.STAR_TRACKER)
            except:
                pass

        q_true = Quaternion.from_array(sim.q_true[k])
        times.append(t)
        errors.append(compute_attitude_error(q_true, x.nom.ori))

    return ScenarioResult("ESKF", np.array(times), np.array(errors))


def run_smoother(sim, env, config_path, q_init, inject_outliers=None):
    """Run standalone iSAM2 smoother."""
    smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=60.0, use_robust=True)
    smoother.initialize(sim.t[0], q_init, np.zeros(3))

    times, errors = [], []

    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        smoother.integrate_gyro(omega, dt, t=t)

        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        z_st = None

        # Apply outlier injection
        if z_mag is not None and inject_outliers and inject_outliers['type'] == 'mag':
            if inject_outliers['start'] <= t <= inject_outliers['end']:
                z_mag = z_mag + np.random.randn(3) * inject_outliers['magnitude']
                z_mag = z_mag / np.linalg.norm(z_mag)

        if not np.any(np.isnan(sim.st_meas[k])):
            if inject_outliers and inject_outliers['type'] == 'st_dropout':
                if not (inject_outliers['start'] <= t <= inject_outliers['end']):
                    z_st = Quaternion.from_array(sim.st_meas[k])
            else:
                z_st = Quaternion.from_array(sim.st_meas[k])

        B_eci = env.get_B_eci(env.get_r_eci(jd), jd) if z_mag is not None else None
        s_eci = env.get_sun_eci(jd) if z_sun is not None else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            state = smoother.add_measurement(t, jd, z_mag, z_sun, z_st, B_eci, s_eci)
            if state:
                q_true = Quaternion.from_array(sim.q_true[k])
                times.append(t)
                errors.append(compute_attitude_error(q_true, state.ori))

    return ScenarioResult("iSAM2", np.array(times), np.array(errors))


def run_redundant(sim, env, config_path, q_init, P0, inject_outliers=None):
    """Run redundant architecture."""
    redundant = RedundantEstimator(
        P0=P0, config_path=config_path, smoother_lag=60.0,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
        agreement_threshold_deg=0.5,
        consecutive_agreements_to_recover=10,
    )

    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    times, errors = [], []
    primaries = []
    switch_times, switch_types = [], []
    last_primary = "ESKF"

    for k in range(1, len(sim.t)):
        t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
        jd, omega = sim.jd[k], sim.omega_meas[k]

        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        z_st = None

        # Apply outlier injection
        if z_mag is not None and inject_outliers and inject_outliers['type'] == 'mag':
            if inject_outliers['start'] <= t <= inject_outliers['end']:
                z_mag = z_mag + np.random.randn(3) * inject_outliers['magnitude']
                z_mag = z_mag / np.linalg.norm(z_mag)

        if not np.any(np.isnan(sim.st_meas[k])):
            if inject_outliers and inject_outliers['type'] == 'st_dropout':
                if not (inject_outliers['start'] <= t <= inject_outliers['end']):
                    z_st = Quaternion.from_array(sim.st_meas[k])
            else:
                z_st = Quaternion.from_array(sim.st_meas[k])

        B_n = env.get_B_eci(env.get_r_eci(jd), jd) if z_mag is not None else None
        s_n = env.get_sun_eci(jd) if z_sun is not None else None

        x, smoother_state, disagreement, primary = redundant.step(
            x, t, jd, omega, dt, z_mag, z_sun, z_st, B_n, s_n
        )

        # Track switches
        if primary != last_primary:
            switch_times.append(t)
            switch_types.append(f"{last_primary}→{primary}")
            last_primary = primary

        q_true = Quaternion.from_array(sim.q_true[k])
        primary_state = x.nom if primary == "ESKF" else smoother_state

        times.append(t)
        errors.append(compute_attitude_error(q_true, primary_state.ori))
        primaries.append(primary)

    result = ScenarioResult("Redundant", np.array(times), np.array(errors))
    result.switch_times = switch_times
    result.switch_types = switch_types
    result.primaries = primaries
    return result


def plot_scenario(results: Dict[str, ScenarioResult], title: str, filename: str,
                  event_window: Tuple[float, float] = None):
    """Plot scenario comparison with switching events."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    colors = {'ESKF': '#1f77b4', 'iSAM2': '#ff7f0e', 'Redundant': '#2ca02c'}

    # Top: Full error comparison (log scale)
    ax = axes[0]
    for name, result in results.items():
        ax.semilogy(result.times, result.errors, color=colors[name],
                   linewidth=0.8, alpha=0.8, label=f"{name} (SS: {result.ss_mean():.4f}°)")

    # Mark event window
    if event_window:
        ax.axvspan(event_window[0], event_window[1], alpha=0.2, color='red', label='Event')

    # Mark switch times for redundant
    if 'Redundant' in results and results['Redundant'].switch_times:
        for st, stype in zip(results['Redundant'].switch_times, results['Redundant'].switch_types):
            ax.axvline(x=st, color='purple', linestyle=':', alpha=0.7)
            ax.annotate(stype, (st, ax.get_ylim()[1]*0.5), rotation=90, fontsize=8, color='purple')

    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-4, 100])

    # Middle: Linear scale zoom on event
    ax = axes[1]
    for name, result in results.items():
        ax.plot(result.times, result.errors, color=colors[name], linewidth=0.8, alpha=0.8, label=name)

    if event_window:
        ax.axvspan(event_window[0], event_window[1], alpha=0.2, color='red')
        ax.set_xlim([max(0, event_window[0]-20), event_window[1]+40])

    if 'Redundant' in results and results['Redundant'].switch_times:
        for st in results['Redundant'].switch_times:
            ax.axvline(x=st, color='purple', linestyle=':', alpha=0.7)

    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Event Window Detail')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom: Primary estimator (redundant only)
    ax = axes[2]
    if 'Redundant' in results and hasattr(results['Redundant'], 'primaries'):
        primaries = results['Redundant'].primaries
        times = results['Redundant'].times
        primary_numeric = np.array([1 if p == 'ESKF' else 0 for p in primaries])

        # Create colored regions for each estimator
        ax.fill_between(times, 0, 1, where=primary_numeric == 1,
                       color='#1f77b4', alpha=0.6, step='post', label='ESKF active')
        ax.fill_between(times, 0, 1, where=primary_numeric == 0,
                       color='#ff7f0e', alpha=0.6, step='post', label='iSAM2 active')

        if event_window:
            ax.axvspan(event_window[0], event_window[1], alpha=0.15, color='red', zorder=0)

        # Mark switch times with annotations (alternate positions to avoid overlap)
        for i, (st, stype) in enumerate(zip(results['Redundant'].switch_times, results['Redundant'].switch_types)):
            ax.axvline(x=st, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            # Alternate y position for annotations
            y_pos = 0.75 if i % 2 == 0 else 0.25
            x_offset = 3 if i % 2 == 0 else 5
            ax.annotate(stype, xy=(st, 0.5), xytext=(st + x_offset, y_pos),
                       fontsize=9, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

        ax.set_ylabel('Active Estimator')
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', ncol=2, fontsize=9)

    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def run_scenario_1_overconfident_init(sim, env, config_path):
    """
    Scenario 1: Overconfident initialization

    ESKF starts with large attitude error but small covariance (overconfident).
    This causes slow convergence. The smoother should detect disagreement and take over.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Overconfident Initialization")
    print("="*70)
    print("Setup: 45° initial error, but P0 says only 1° uncertainty")
    print("Expected: ESKF converges slowly, redundant switches to smoother early")

    # Large initial error (~45 degrees)
    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.5, 0.5, 0.5]))  # ~50 deg error
    q_init = q_true_0 @ perturb

    # Overconfident P0 (small attitude uncertainty)
    P0_overconfident = np.diag([0.001, 0.001, 0.001, 1e-6, 1e-6, 1e-6])  # 0.06 deg uncertainty
    P0_normal = np.diag([0.3, 0.3, 0.3, 1e-6, 1e-6, 1e-6])  # ~17 deg uncertainty

    print("\nRunning estimators...")
    results = {}

    # ESKF with overconfident init
    results['ESKF'] = run_eskf(sim, env, config_path, q_init, P0_overconfident)
    print(f"  ESKF: SS mean = {results['ESKF'].ss_mean():.4f}°")

    # iSAM2 (doesn't use P0)
    results['iSAM2'] = run_smoother(sim, env, config_path, q_init)
    print(f"  iSAM2: SS mean = {results['iSAM2'].ss_mean():.4f}°")

    # Redundant with overconfident ESKF
    results['Redundant'] = run_redundant(sim, env, config_path, q_init, P0_overconfident)
    print(f"  Redundant: SS mean = {results['Redundant'].ss_mean():.4f}°")

    if results['Redundant'].switch_times:
        print(f"\n  Switch events: {list(zip(results['Redundant'].switch_times, results['Redundant'].switch_types))}")

    plot_scenario(results, "Scenario 1: Overconfident Initialization (45° error, tiny P0)",
                  "scenario1_overconfident.png", event_window=(0, 20))

    return results


def run_scenario_2_star_tracker_dropout(sim, env, config_path):
    """
    Scenario 2: Star tracker dropout

    Star tracker unavailable for 60 seconds. ESKF must rely on mag/sun only.
    Smoother with robust cost and different structure may handle this differently.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Star Tracker Dropout")
    print("="*70)
    print("Setup: Star tracker unavailable from t=50s to t=110s (60s dropout)")
    print("Expected: Both estimators degrade, but redundant can detect issues")

    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.3, 0.3, 0.3]))
    q_init = q_true_0 @ perturb

    P0 = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6])

    inject = {'type': 'st_dropout', 'start': 50.0, 'end': 110.0}

    print("\nRunning estimators...")
    results = {}

    results['ESKF'] = run_eskf(sim, env, config_path, q_init, P0, inject_outliers=inject)
    print(f"  ESKF: SS mean = {results['ESKF'].ss_mean():.4f}°")

    results['iSAM2'] = run_smoother(sim, env, config_path, q_init, inject_outliers=inject)
    print(f"  iSAM2: SS mean = {results['iSAM2'].ss_mean():.4f}°")

    results['Redundant'] = run_redundant(sim, env, config_path, q_init, P0, inject_outliers=inject)
    print(f"  Redundant: SS mean = {results['Redundant'].ss_mean():.4f}°")

    if results['Redundant'].switch_times:
        print(f"\n  Switch events: {list(zip(results['Redundant'].switch_times, results['Redundant'].switch_types))}")

    plot_scenario(results, "Scenario 2: Star Tracker Dropout (t=50-110s)",
                  "scenario2_st_dropout.png", event_window=(50, 110))

    return results


def run_scenario_3_magnetometer_spikes(sim, env, config_path):
    """
    Scenario 3: Magnetometer measurement spikes

    Periodic large errors in magnetometer readings. Tests robustness to outliers.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Magnetometer Measurement Spikes")
    print("="*70)
    print("Setup: Large magnetometer noise from t=80s to t=140s")
    print("Expected: iSAM2 with Huber robust cost handles spikes better")

    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.3, 0.3, 0.3]))
    q_init = q_true_0 @ perturb

    P0 = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6])

    inject = {'type': 'mag', 'start': 80.0, 'end': 140.0, 'magnitude': 0.5}  # Large spikes

    print("\nRunning estimators...")
    results = {}

    results['ESKF'] = run_eskf(sim, env, config_path, q_init, P0, inject_outliers=inject)
    print(f"  ESKF: SS mean = {results['ESKF'].ss_mean():.4f}°")

    results['iSAM2'] = run_smoother(sim, env, config_path, q_init, inject_outliers=inject)
    print(f"  iSAM2: SS mean = {results['iSAM2'].ss_mean():.4f}°")

    results['Redundant'] = run_redundant(sim, env, config_path, q_init, P0, inject_outliers=inject)
    print(f"  Redundant: SS mean = {results['Redundant'].ss_mean():.4f}°")

    if results['Redundant'].switch_times:
        print(f"\n  Switch events: {list(zip(results['Redundant'].switch_times, results['Redundant'].switch_types))}")

    plot_scenario(results, "Scenario 3: Magnetometer Spikes (t=80-140s, σ=0.5)",
                  "scenario3_mag_spikes.png", event_window=(80, 140))

    return results


def run_scenario_4_measurement_blackout(sim, env, config_path):
    """
    Scenario 4: Measurement blackout with gyro bias step

    Normal operation for 30s, then NO measurements (only gyro) for 30s WITH
    an injected gyro bias step at t=30s. This causes drift during blackout.
    When measurements return, the ESKF may struggle to recover if overconfident.
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Measurement Blackout + Gyro Bias Step")
    print("="*70)
    print("Setup: Normal ops 0-30s, blackout 30-60s with bias step, normal ops 60s+")
    print("       Gyro bias step of 0.5 deg/s injected at t=30s")
    print("Expected: ESKF may struggle to recover, smoother refits when measurements return")

    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.1, 0.1, 0.1]))  # Small initial error
    q_init = q_true_0 @ perturb

    # Overconfident P0 - ESKF won't grow covariance enough during blackout
    P0 = np.diag([0.01, 0.01, 0.01, 1e-8, 1e-8, 1e-8])  # Very confident in bias

    # Blackout config
    blackout_start, blackout_end = 30.0, 60.0

    # Gyro bias step (0.5 deg/s = 0.00873 rad/s)
    bias_step = np.array([0.5, 0.0, 0.0]) * np.pi / 180.0  # rad/s

    print("\nRunning estimators...")
    results = {}

    # Custom run functions for blackout scenario
    def run_with_blackout(estimator_type):
        if estimator_type == 'eskf':
            eskf = ESKF(P0=P0, config_path=config_path)
            x = EskfState(
                nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
                err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
            )
        elif estimator_type == 'smoother':
            smoother = FixedLagAttitudeSmoother(config_path=config_path, lag=60.0, use_robust=True)
            smoother.initialize(sim.t[0], q_init, np.zeros(3))
        else:  # redundant
            redundant = RedundantEstimator(
                P0=P0, config_path=config_path, smoother_lag=60.0,
                disagreement_threshold_deg=2.0,
                consecutive_disagreements_to_switch=5,
                agreement_threshold_deg=0.5,
                consecutive_agreements_to_recover=10,
            )
            x = EskfState(
                nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
                err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
            )

        times, errors = [], []
        primaries = [] if estimator_type == 'redundant' else None
        switch_times, switch_types = [], []
        last_primary = "ESKF"

        for k in range(1, len(sim.t)):
            t, dt = sim.t[k], sim.t[k] - sim.t[k-1]
            jd, omega = sim.jd[k], sim.omega_meas[k]

            # Check if in blackout period
            in_blackout = blackout_start <= t <= blackout_end

            # Inject gyro bias step during blackout
            if in_blackout:
                omega = omega + bias_step  # Add bias to gyro measurement

            # Get measurements (None during blackout)
            if in_blackout:
                z_mag, z_sun, z_st = None, None, None
            else:
                z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
                z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
                z_st = Quaternion.from_array(sim.st_meas[k]) if not np.any(np.isnan(sim.st_meas[k])) else None

            B_n = env.get_B_eci(env.get_r_eci(jd), jd) if z_mag is not None else None
            s_n = env.get_sun_eci(jd) if z_sun is not None else None

            if estimator_type == 'eskf':
                x = eskf.predict(x, omega, dt)
                if z_mag is not None:
                    try: x = eskf.update(x, z_mag, SensorType.MAGNETOMETER, B_n=B_n)
                    except: pass
                if z_sun is not None:
                    try: x = eskf.update(x, z_sun, SensorType.SUN_VECTOR, s_n=s_n)
                    except: pass
                if z_st is not None:
                    try: x = eskf.update(x, z_st, SensorType.STAR_TRACKER)
                    except: pass
                q_est = x.nom.ori

            elif estimator_type == 'smoother':
                smoother.integrate_gyro(omega, dt, t=t)  # May create auto-keyframe during gaps
                if z_mag is not None or z_sun is not None or z_st is not None:
                    state = smoother.add_measurement(t, jd, z_mag, z_sun, z_st, B_n, s_n)
                    if state:
                        q_est = state.ori
                    else:
                        continue
                else:
                    # No measurements - get current state (may have been updated by auto-keyframe)
                    state = smoother.get_state()
                    if state:
                        q_est = state.ori
                    else:
                        continue

            else:  # redundant
                x, smoother_state, disagreement, primary = redundant.step(
                    x, t, jd, omega, dt, z_mag, z_sun, z_st, B_n, s_n
                )
                if primary != last_primary:
                    switch_times.append(t)
                    switch_types.append(f"{last_primary}→{primary}")
                    last_primary = primary
                q_est = x.nom.ori if primary == "ESKF" else smoother_state.ori
                primaries.append(primary)

            q_true = Quaternion.from_array(sim.q_true[k])
            times.append(t)
            errors.append(compute_attitude_error(q_true, q_est))

        result = ScenarioResult(
            estimator_type.upper() if estimator_type != 'redundant' else 'Redundant',
            np.array(times), np.array(errors)
        )
        if estimator_type == 'redundant':
            result.switch_times = switch_times
            result.switch_types = switch_types
            result.primaries = primaries
        return result

    results['ESKF'] = run_with_blackout('eskf')
    print(f"  ESKF: SS mean = {results['ESKF'].ss_mean(start=70):.4f}° (post-recovery)")

    results['iSAM2'] = run_with_blackout('smoother')
    print(f"  iSAM2: SS mean = {results['iSAM2'].ss_mean(start=70):.4f}° (post-recovery)")

    results['Redundant'] = run_with_blackout('redundant')
    print(f"  Redundant: SS mean = {results['Redundant'].ss_mean(start=70):.4f}° (post-recovery)")

    if results['Redundant'].switch_times:
        print(f"\n  Switch events: {list(zip(results['Redundant'].switch_times, results['Redundant'].switch_types))}")

    # Custom plot for blackout scenario
    plot_scenario(results, "Scenario 4: Measurement Blackout (t=30-60s, gyro only)",
                  "scenario4_blackout.png", event_window=(30, 60))

    return results


def print_summary_table(all_results: Dict[str, Dict[str, ScenarioResult]]):
    """Print LaTeX summary table."""
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)

    print("\nSteady-State Mean Error [deg]:")
    print("-"*70)
    print(f"{'Scenario':<35} {'ESKF':>10} {'iSAM2':>10} {'Redundant':>10}")
    print("-"*70)

    for scenario_name, results in all_results.items():
        eskf_err = results['ESKF'].ss_mean()
        isam_err = results['iSAM2'].ss_mean()
        red_err = results['Redundant'].ss_mean()
        print(f"{scenario_name:<35} {eskf_err:>10.4f} {isam_err:>10.4f} {red_err:>10.4f}")

    # LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Redundant Architecture Performance Under Degraded Conditions}
\label{tab:redundant_scenarios}
\begin{tabular}{lccc}
\toprule
Scenario & ESKF & iSAM2 & Redundant \\
\midrule
"""

    for scenario_name, results in all_results.items():
        eskf_err = results['ESKF'].ss_mean()
        isam_err = results['iSAM2'].ss_mean()
        red_err = results['Redundant'].ss_mean()

        # Bold the best performer
        errors = [eskf_err, isam_err, red_err]
        best_idx = np.argmin(errors)

        eskf_str = f"\\textbf{{{eskf_err:.4f}}}" if best_idx == 0 else f"{eskf_err:.4f}"
        isam_str = f"\\textbf{{{isam_err:.4f}}}" if best_idx == 1 else f"{isam_err:.4f}"
        red_str = f"\\textbf{{{red_err:.4f}}}" if best_idx == 2 else f"{red_err:.4f}"

        latex += f"{scenario_name} & {eskf_str} & {isam_str} & {red_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    print(latex)


def main():
    print("="*70)
    print("REDUNDANT ARCHITECTURE STRESS TESTING")
    print("="*70)

    db = SimulationDatabase('simulations.db')
    env = OrbitEnvironmentModel()
    config_path = 'configs/config_baseline_short.yaml'

    # Use baseline simulation
    sim = db.load_run(33)
    print(f"Using simulation with {len(sim.t)} samples ({sim.t[-1]:.1f}s)")

    all_results = {}

    # Run all scenarios
    all_results['1. Overconfident Init'] = run_scenario_1_overconfident_init(sim, env, config_path)
    all_results['2. Star Tracker Dropout'] = run_scenario_2_star_tracker_dropout(sim, env, config_path)
    all_results['3. Magnetometer Spikes'] = run_scenario_3_magnetometer_spikes(sim, env, config_path)

    # Print summary
    print_summary_table(all_results)

    # Print switching summary
    print("\n" + "="*70)
    print("SWITCHING EVENTS SUMMARY")
    print("="*70)
    for scenario_name, results in all_results.items():
        red = results['Redundant']
        print(f"\n{scenario_name}:")
        if red.switch_times:
            for t, s in zip(red.switch_times, red.switch_types):
                print(f"  t={t:.2f}s: {s}")
        else:
            print("  No switches")

    return all_results


if __name__ == "__main__":
    main()
