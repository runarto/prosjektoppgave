#!/usr/bin/env python3
"""
Analyze switching behavior of the Redundant estimator during faults.

Questions:
1. Does the system switch when a fault is detected?
2. What triggers the switch (disagreement vs NIS)?
3. How does switching affect accuracy?
"""

import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)

CONFIG_PATH = "configs/config_baseline_short.yaml"
FAULT_START = 100.0
FAULT_END = 150.0


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


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


def run_redundant_with_tracking(sim_data):
    """Run redundant estimator and track switching behavior."""
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

    times = []
    errors = []
    primaries = []
    disagreements = []

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

        # Compute error based on primary
        if primary == "SMOOTHER" and smoother_state is not None:
            q_est = smoother_state.ori
        else:
            q_est = x_eskf.nom.ori

        times.append(t)
        errors.append(compute_attitude_error_deg(q_est, q_true))
        primaries.append(primary)
        disagreements.append(disagreement)

    # Get switch events
    switch_events = redundant.switch_events

    return {
        'times': np.array(times),
        'errors': np.array(errors),
        'primaries': primaries,
        'disagreements': np.array(disagreements),
        'switch_events': switch_events,
    }


def main():
    db = SimulationDatabase("simulations.db")
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()[0]
    conn.close()
    sim_data = db.load_run(run_id)

    print("=" * 80)
    print("REDUNDANT ESTIMATOR SWITCHING BEHAVIOR ANALYSIS")
    print("=" * 80)
    print(f"Fault window: {FAULT_START}s - {FAULT_END}s")
    print()

    # Test scenarios
    scenarios = [
        ('mag', 1.0, 'Magnetometer (1.0)'),
        ('sun', 1.0, 'Sun Sensor (1.0)'),
        ('st', 0.2, 'Star Tracker (0.2 rad = 11.5°)'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    for row, (sensor, mag, name) in enumerate(scenarios):
        print(f"\n{name}:")

        # Apply fault
        modified = apply_fault(sim_data, sensor, mag)

        # Run redundant estimator
        result = run_redundant_with_tracking(modified)

        # Analyze switching
        switch_events = result['switch_events']
        primaries = result['primaries']

        print(f"  Switch events: {len(switch_events)}")
        for t, event in switch_events:
            print(f"    t={t:.1f}s: {event}")

        # Count time in each mode
        eskf_count = sum(1 for p in primaries if p == 'ESKF')
        smoother_count = sum(1 for p in primaries if p == 'SMOOTHER')
        conservative_count = sum(1 for p in primaries if p == 'CONSERVATIVE')
        total = len(primaries)

        print(f"  Time in ESKF: {100*eskf_count/total:.1f}%")
        print(f"  Time in SMOOTHER: {100*smoother_count/total:.1f}%")
        print(f"  Time in CONSERVATIVE: {100*conservative_count/total:.1f}%")

        # Plot 1: Primary estimator over time
        ax1 = axes[row, 0]
        primary_numeric = [{'ESKF': 0, 'SMOOTHER': 1, 'CONSERVATIVE': 2}[p] for p in primaries]
        ax1.plot(result['times'], primary_numeric, 'k-', linewidth=1)
        ax1.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['ESKF', 'SMOOTHER', 'CONSERV.'])
        ax1.set_ylabel('Primary')
        ax1.set_title(f'{name}')
        ax1.grid(True, alpha=0.3)
        if row == 2:
            ax1.set_xlabel('Time [s]')

        # Plot 2: Disagreement over time
        ax2 = axes[row, 1]
        ax2.plot(result['times'], result['disagreements'], 'b-', linewidth=0.8)
        ax2.axhline(y=2.0, color='r', linestyle='--', label='Threshold (2°)')
        ax2.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red')
        ax2.set_ylabel('Disagreement [deg]')
        ax2.set_ylim([0, min(10, np.max(result['disagreements']) * 1.1)])
        ax2.grid(True, alpha=0.3)
        if row == 0:
            ax2.legend(loc='upper right')
        if row == 2:
            ax2.set_xlabel('Time [s]')

        # Plot 3: Attitude error over time
        ax3 = axes[row, 2]
        ax3.plot(result['times'], result['errors'], 'g-', linewidth=0.8)
        ax3.axvspan(FAULT_START, FAULT_END, alpha=0.2, color='red')
        ax3.set_ylabel('Attitude Error [deg]')
        ax3.set_ylim([0, min(0.1, np.max(result['errors']) * 1.1)])
        ax3.grid(True, alpha=0.3)
        if row == 2:
            ax3.set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig('redundant_switching_analysis.pdf', bbox_inches='tight')
    plt.savefig('redundant_switching_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: redundant_switching_analysis.pdf/png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Redundant estimator uses DISAGREEMENT-based switching, not NIS-based:

1. ESKF and Smoother run in PARALLEL at all times
2. Disagreement = angular difference between ESKF and Smoother estimates
3. If disagreement > threshold (2°) for N consecutive measurements → switch

Key observations:
- Vector sensor faults (Mag/Sun): ESKF rejects bad measurements via chi-squared gate,
  so ESKF estimate stays accurate. Disagreement stays LOW because both estimators
  reject/downweight the faulty measurements. NO SWITCH needed.

- Star tracker faults: The Smoother's robust kernel may not fully reject large
  ST errors, potentially causing higher Smoother error and triggering a switch
  to CONSERVATIVE mode.

The NIS-based detection is used for:
- Per-sensor health monitoring
- Identifying WHICH sensor is faulty
- Deciding between SMOOTHER vs CONSERVATIVE mode when switching
""")


if __name__ == "__main__":
    main()
