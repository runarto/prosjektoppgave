#!/usr/bin/env python3
"""
Focused Fault Detection Demonstration.

Creates clear visualizations showing:
1. What fault detection looks like in practice
2. When faults are detected vs missed
3. False alarm vs detection trade-off
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.db import SimulationDatabase
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from utilities.utils import load_yaml


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true @ q_est.conjugate()
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_with_fault_injection(sim_data, config_path, fault_config=None):
    """
    Run redundant estimator with optional fault injection.

    Returns time series of: times, eskf_errors, smoother_errors, disagreements
    """
    att_err_rad = np.deg2rad(10.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=60.0,
        use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )

    # Initial state with error
    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad, att_err_rad, att_err_rad]) / np.sqrt(3)
    q0_est = q0_true @ Quaternion.from_avec(perturb)
    q0_est = q0_est.normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0.copy())
    x_est = EskfState(nom=nom0, err=err0)

    times, eskf_errors, smoother_errors, disagreements = [], [], [], []
    fault_active = False
    injected_bias = np.zeros(3)

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt_k = t - sim_data.t[k-1]

        omega_k = sim_data.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim_data.omega_meas[k-1]

        # Fault injection
        if fault_config and t >= fault_config.get('onset_time', float('inf')):
            if not fault_active:
                fault_active = True
                if fault_config['type'] == 'gyro_bias_step':
                    injected_bias = fault_config['magnitude'] * np.array([1, 0.5, 0.3])
                    injected_bias = injected_bias / np.linalg.norm(injected_bias) * fault_config['magnitude']
            omega_k = omega_k + injected_bias

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None
        B_n = sim_data.b_eci[k]
        s_n = sim_data.s_eci[k]

        x_est, smoother_state, disagreement_deg, primary_str = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega_k, dt=dt_k,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        q_true = Quaternion.from_array(sim_data.q_true[k])
        eskf_err = compute_attitude_error(x_est.nom.ori, q_true)
        smoother_err = compute_attitude_error(smoother_state.ori, q_true)

        times.append(t)
        eskf_errors.append(eskf_err)
        smoother_errors.append(smoother_err)
        disagreements.append(disagreement_deg)

    stats = redundant.get_statistics()
    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'smoother_errors': np.array(smoother_errors),
        'disagreements': np.array(disagreements),
        'switch_events': stats.get('switch_events', []),
    }


def create_fault_detection_figure(nominal_result, faulty_results, fault_onset, save_path=None):
    """
    Create a clear fault detection demonstration figure.

    Shows steady-state behavior with and without faults.
    """
    fig = plt.figure(figsize=(14, 10))

    # Focus on steady-state (after t=50s convergence)
    t_start = 50.0
    t_end = None

    # Filter data to steady-state
    nom_mask = nominal_result['times'] >= t_start
    nom_t = nominal_result['times'][nom_mask]
    nom_dis = nominal_result['disagreements'][nom_mask]

    # === Plot 1: Nominal disagreement (what "healthy" looks like) ===
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(nom_t, nom_dis, 'b-', linewidth=0.8, alpha=0.8)
    ax1.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='Detection threshold (2°)')
    ax1.fill_between(nom_t, 0, nom_dis, alpha=0.3, color='blue')
    ax1.set_ylabel('Disagreement [deg]')
    ax1.set_title('Nominal Operation (No Fault)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(3.0, np.percentile(nom_dis, 99) * 1.2)])

    # Statistics box
    nom_mean = np.mean(nom_dis)
    nom_std = np.std(nom_dis)
    nom_max = np.max(nom_dis)
    false_alarms = np.sum(nom_dis > 2.0) / len(nom_dis) * 100
    stats_text = f'Mean: {nom_mean:.3f}°\nStd: {nom_std:.3f}°\nMax: {nom_max:.2f}°\nFalse alarms: {false_alarms:.1f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === Plot 2: Faulty disagreement (what fault looks like) ===
    ax2 = fig.add_subplot(3, 2, 2)

    # Use the medium fault (0.02 rad/s)
    faulty = faulty_results['medium']
    faulty_mask = faulty['times'] >= t_start
    faulty_t = faulty['times'][faulty_mask]
    faulty_dis = faulty['disagreements'][faulty_mask]

    ax2.plot(faulty_t, faulty_dis, 'r-', linewidth=0.8, alpha=0.8)
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(fault_onset, color='green', linestyle='-', linewidth=2, label=f'Fault onset (t={fault_onset}s)')
    ax2.fill_between(faulty_t, 0, faulty_dis, alpha=0.3, color='red')

    # Mark detection point
    post_fault_mask = faulty_t >= fault_onset
    post_fault_dis = faulty_dis[post_fault_mask]
    post_fault_t = faulty_t[post_fault_mask]
    if len(post_fault_dis) > 0:
        detection_idx = np.where(post_fault_dis > 2.0)[0]
        if len(detection_idx) > 0:
            detection_time = post_fault_t[detection_idx[0]]
            detection_delay = detection_time - fault_onset
            ax2.axvline(detection_time, color='purple', linestyle=':', linewidth=2,
                       label=f'Detection (delay: {detection_delay:.1f}s)')

    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_title('With Gyro Bias Fault (0.02 rad/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(5.0, np.max(faulty_dis) * 1.1)])

    # === Plot 3: Comparison of fault magnitudes ===
    ax3 = fig.add_subplot(3, 2, 3)

    colors = {'small': 'orange', 'medium': 'red', 'large': 'darkred'}
    labels = {'small': '0.01 rad/s', 'medium': '0.02 rad/s', 'large': '0.05 rad/s'}

    for key in ['small', 'medium', 'large']:
        if key in faulty_results:
            res = faulty_results[key]
            mask = res['times'] >= fault_onset - 10
            ax3.plot(res['times'][mask], res['disagreements'][mask],
                    color=colors[key], linewidth=1, alpha=0.8, label=labels[key])

    ax3.axhline(2.0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(fault_onset, color='green', linestyle='-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Disagreement [deg]')
    ax3.set_title('Detection by Fault Magnitude')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # === Plot 4: Detection delay vs fault magnitude ===
    ax4 = fig.add_subplot(3, 2, 4)

    magnitudes = []
    delays = []
    detected = []

    for key, mag in [('small', 0.01), ('medium', 0.02), ('large', 0.05)]:
        if key in faulty_results:
            res = faulty_results[key]
            post_mask = res['times'] >= fault_onset
            post_dis = res['disagreements'][post_mask]
            post_t = res['times'][post_mask]

            det_idx = np.where(post_dis > 2.0)[0]
            magnitudes.append(mag)
            if len(det_idx) > 0:
                delays.append(post_t[det_idx[0]] - fault_onset)
                detected.append(True)
            else:
                delays.append(np.nan)
                detected.append(False)

    colors_bar = ['green' if d else 'red' for d in detected]
    bars = ax4.bar(range(len(magnitudes)), delays, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(magnitudes)))
    ax4.set_xticklabels([f'{m*1000:.0f} mrad/s' for m in magnitudes])
    ax4.set_ylabel('Detection Delay [s]')
    ax4.set_xlabel('Fault Magnitude')
    ax4.set_title('Detection Delay by Fault Size')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, delay) in enumerate(zip(bars, delays)):
        if not np.isnan(delay):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{delay:.1f}s', ha='center', va='bottom', fontsize=10)

    # === Plot 5: ESKF error with and without fault detection ===
    ax5 = fig.add_subplot(3, 2, 5)

    # Nominal ESKF error
    nom_err_mask = nominal_result['times'] >= t_start
    ax5.semilogy(nominal_result['times'][nom_err_mask],
                 nominal_result['eskf_errors'][nom_err_mask],
                 'b-', linewidth=1, alpha=0.7, label='Nominal')

    # Faulty ESKF error (shows divergence if not corrected)
    faulty = faulty_results['medium']
    ax5.semilogy(faulty['times'][faulty_mask],
                 faulty['eskf_errors'][faulty_mask],
                 'r-', linewidth=1, alpha=0.7, label='With fault')

    ax5.axvline(fault_onset, color='green', linestyle='-', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('ESKF Attitude Error [deg]')
    ax5.set_title('Impact of Fault on Estimation Error')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([1e-3, 20])

    # === Plot 6: Summary statistics table ===
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    # Create summary table
    table_data = [
        ['Metric', 'Nominal', 'Small Fault', 'Med Fault', 'Large Fault'],
        ['Mean Disagree.', f'{nom_mean:.3f}°', '', '', ''],
        ['Detection Rate', 'N/A', '', '', ''],
        ['Detection Delay', 'N/A', '', '', ''],
        ['False Alarm Rate', f'{false_alarms:.1f}%', 'N/A', 'N/A', 'N/A'],
    ]

    # Fill in fault data
    for i, key in enumerate(['small', 'medium', 'large']):
        if key in faulty_results:
            res = faulty_results[key]
            post_mask = res['times'] >= fault_onset
            post_dis = res['disagreements'][post_mask]

            mean_dis = np.mean(post_dis)
            table_data[1][i+2] = f'{mean_dis:.2f}°'

            det_idx = np.where(post_dis > 2.0)[0]
            if len(det_idx) > 0:
                table_data[2][i+2] = '✓ Yes'
                delay = res['times'][post_mask][det_idx[0]] - fault_onset
                table_data[3][i+2] = f'{delay:.1f}s'
            else:
                table_data[2][i+2] = '✗ No'
                table_data[3][i+2] = 'N/A'

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(5):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Summary Statistics', pad=20)

    plt.suptitle('Fault Detection Analysis: ESKF vs Smoother Disagreement', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def main():
    config_path = "configs/config_baseline_short.yaml"
    db = SimulationDatabase("simulations.db")

    # Load most recent simulation
    import sqlite3
    conn = sqlite3.connect("simulations.db")
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM runs ORDER BY id DESC LIMIT 1')
    sim_id = cursor.fetchone()[0]
    conn.close()

    print(f"Loading simulation ID: {sim_id}")
    sim_data = db.load_run(sim_id)
    print(f"Simulation duration: {sim_data.t[-1]:.1f}s")

    fault_onset = 150.0  # Inject fault at t=150s (well after convergence)

    print("\nRunning nominal (no fault)...")
    nominal = run_with_fault_injection(sim_data, config_path, fault_config=None)

    print("Running with small fault (0.01 rad/s)...")
    small_fault = run_with_fault_injection(sim_data, config_path,
        fault_config={'type': 'gyro_bias_step', 'onset_time': fault_onset, 'magnitude': 0.01})

    print("Running with medium fault (0.02 rad/s)...")
    medium_fault = run_with_fault_injection(sim_data, config_path,
        fault_config={'type': 'gyro_bias_step', 'onset_time': fault_onset, 'magnitude': 0.02})

    print("Running with large fault (0.05 rad/s)...")
    large_fault = run_with_fault_injection(sim_data, config_path,
        fault_config={'type': 'gyro_bias_step', 'onset_time': fault_onset, 'magnitude': 0.05})

    faulty_results = {
        'small': small_fault,
        'medium': medium_fault,
        'large': large_fault,
    }

    print("\nGenerating figure...")
    create_fault_detection_figure(nominal, faulty_results, fault_onset,
                                  save_path="fault_detection_demo.pdf")

    # Print summary
    print("\n" + "="*60)
    print("FAULT DETECTION SUMMARY")
    print("="*60)

    # Nominal stats
    nom_mask = nominal['times'] >= 50
    nom_dis = nominal['disagreements'][nom_mask]
    print(f"\nNominal (healthy) operation (t>50s):")
    print(f"  Mean disagreement: {np.mean(nom_dis):.3f}°")
    print(f"  Std disagreement:  {np.std(nom_dis):.3f}°")
    print(f"  Max disagreement:  {np.max(nom_dis):.2f}°")
    print(f"  False alarm rate:  {np.sum(nom_dis > 2.0)/len(nom_dis)*100:.1f}%")

    print(f"\nFault detection results (fault onset at t={fault_onset}s):")
    print("-"*60)

    for name, mag in [('Small', 0.01), ('Medium', 0.02), ('Large', 0.05)]:
        key = name.lower()
        res = faulty_results[key]
        post_mask = res['times'] >= fault_onset
        post_dis = res['disagreements'][post_mask]
        post_t = res['times'][post_mask]

        det_idx = np.where(post_dis > 2.0)[0]

        print(f"\n{name} fault ({mag*1000:.0f} mrad/s = {np.rad2deg(mag):.2f} deg/s):")
        print(f"  Mean post-fault disagreement: {np.mean(post_dis):.2f}°")
        print(f"  Max post-fault disagreement:  {np.max(post_dis):.2f}°")

        if len(det_idx) > 0:
            delay = post_t[det_idx[0]] - fault_onset
            print(f"  Detected: YES (delay = {delay:.1f}s)")
        else:
            print(f"  Detected: NO")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
