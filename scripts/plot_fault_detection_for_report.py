#!/usr/bin/env python3
"""
Generate report-quality plots for fault detection evaluation section.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run the fault detection test to get results
from scripts.fault_detection_comprehensive_test import (
    FaultScenario, generate_simulation_data, run_eskf, run_smoother, run_redundant
)

# =============================================================================
# Publication-quality plot settings (matching test_anees_inline.py)
# =============================================================================
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


def plot_star_tracker_fault_detail():
    """Create detailed plot of star tracker fault scenario."""

    base_config = "configs/config_fault_detection_test.yaml"

    scenario = FaultScenario(
        name="Star Tracker Fault",
        sensor="star",
        fault_start=100.0,
        fault_end=150.0,
        spike_probability=1.0,
        spike_magnitude=1.5,
    )

    print("Generating star tracker fault data...")
    sim_data = generate_simulation_data(base_config, scenario, T=300.0, dt=0.02)

    print("Running estimators...")
    eskf_result = run_eskf(sim_data, base_config, use_gating=True, gating_threshold=9.21)
    smoother_result = run_smoother(sim_data, base_config, use_robust=True, robust_param=0.1, lag=60.0)
    redundant_result = run_redundant(sim_data, base_config, use_robust=True, robust_param=0.1,
                                      smoother_lag=60.0, disagreement_threshold_deg=0.5,
                                      consecutive_to_switch=3)

    # Create figure with more space
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.25})

    t = eskf_result.times
    fault_start, fault_end = 100.0, 150.0

    # --- Top panel: Attitude errors ---
    ax1 = axes[0]
    ax1.semilogy(t, eskf_result.attitude_errors_deg, 'C0-', label='ESKF', alpha=0.9)
    ax1.semilogy(t, smoother_result.attitude_errors_deg, 'C1-', label='Smoother', alpha=0.9)
    ax1.semilogy(t, redundant_result.attitude_errors_deg, 'C2--', label='Redundant', alpha=0.9)

    # Shade fault window
    ax1.axvspan(fault_start, fault_end, alpha=0.15, color='red', label='Fault active')

    # Mark switch events
    if redundant_result.switch_events:
        for t_switch, event in redundant_result.switch_events:
            ax1.axvline(x=t_switch, color='C2', linestyle=':', alpha=0.)

    ax1.set_ylabel('Attitude error [deg]')
    ax1.set_ylim([1e-3, 10])
    ax1.set_xlim([0, 300])
    ax1.legend(loc='upper right', ncol=2)
    ax1.set_title('Star Tracker Fault Detection')
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: Disagreement ---
    ax2 = axes[1]
    ax2.plot(t, redundant_result.disagreement_deg, 'C4-', label='Disagreement')
    ax2.axhline(y=0.5, color='C3', linestyle='--', alpha=0.7, label='Threshold')
    ax2.axvspan(fault_start, fault_end, alpha=0.15, color='red')

    # Mark switch events with colored vertical lines
    detect_plotted = False
    recover_plotted = False
    if redundant_result.switch_events:
        for t_switch, event in redundant_result.switch_events:
            if 'CONSERVATIVE' in event and 'ESKF->' in event:
                ax2.axvline(x=t_switch, color='C3', linestyle=':', linewidth=4.0,
                           label='Detection' if not detect_plotted else None)
                detect_plotted = True
            else:
                ax2.axvline(x=t_switch, color='C2', linestyle=':', linewidth=4.0,
                           label='Recovery' if not recover_plotted else None)
                recover_plotted = True

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Disagreement [deg]')
    ax2.set_ylim([0, max(2.5, np.max(redundant_result.disagreement_deg) * 1.1)])
    ax2.set_xlim([0, 300])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.savefig('fault_detection_star_tracker_detail.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fault_detection_star_tracker_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fault_detection_star_tracker_detail.pdf")


def plot_disagreement_comparison():
    """Create comparison of disagreement across all three fault scenarios."""

    base_config = "configs/config_fault_detection_test.yaml"

    scenarios = [
        FaultScenario("Sun Sensor", "sun", 100.0, 150.0, 1.0, 2.0),
        FaultScenario("Magnetometer", "mag", 100.0, 150.0, 1.0, 2.0),
        FaultScenario("Star Tracker", "star", 100.0, 150.0, 1.0, 1.5),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'hspace': 0.15})

    for i, scenario in enumerate(scenarios):
        print(f"Processing {scenario.name}...")
        sim_data = generate_simulation_data(base_config, scenario, T=300.0, dt=0.02)
        result = run_redundant(sim_data, base_config, use_robust=True, robust_param=0.1,
                               smoother_lag=60.0, disagreement_threshold_deg=0.5,
                               consecutive_to_switch=3)

        ax = axes[i]
        t = result.times

        ax.plot(t, result.disagreement_deg, 'C4-', label='Disagreement')
        ax.axhline(y=0.5, color='C3', linestyle='--', alpha=0.7, label='Threshold')
        ax.axvspan(100.0, 150.0, alpha=0.15, color='red', label='Fault active' if i == 0 else None)

        # Mark detection if it occurred
        if result.switch_events:
            for t_switch, event in result.switch_events:
                if 'CONSERVATIVE' in event and 'ESKF->' in event:
                    ax.axvline(x=t_switch, color='C2', linestyle=':', alpha=0.8)

        ax.set_ylabel(f'{scenario.name} [deg]')
        ax.set_ylim([0, max(3.0, np.max(result.disagreement_deg) * 1.1)])
        ax.set_xlim([0, 300])
        ax.grid(True, alpha=0.3)

        # Add detection status
        detected = any('ESKF->' in e[1] and 'CONSERVATIVE' in e[1] for e in result.switch_events) if result.switch_events else False
        status = 'Detected' if detected else 'Not detected'
        props = dict(boxstyle='round', facecolor='wheat' if detected else 'lightgray', alpha=0.9)
        ax.text(0.98, 0.85, status, transform=ax.transAxes, ha='right', fontsize=14,
                color='C2' if detected else 'gray', bbox=props)

        if i == 0:
            ax.legend(loc='upper left', ncol=3)

    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title('ESKF-Smoother Disagreement Across Fault Scenarios')

    plt.savefig('fault_detection_disagreement_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fault_detection_disagreement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fault_detection_disagreement_comparison.pdf")


if __name__ == "__main__":
    print("Generating report plots for fault detection evaluation...")
    print("=" * 60)

    plot_star_tracker_fault_detail()
    print()
    plot_disagreement_comparison()

    print()
    print("=" * 60)
    print("Done! Generated plots:")
    print("  - fault_detection_star_tracker_detail.pdf")
    print("  - fault_detection_disagreement_comparison.pdf")
