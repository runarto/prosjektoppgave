#!/usr/bin/env python3
"""
Fault Detection Summary Plot

Two-panel figure:
(a) Detection latency by sensor
(b) Accuracy during faults by estimator
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

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


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Detection latency by sensor
    ax1 = axes[0]
    sensors = ['Magnetometer', 'Sun Sensor', 'Star Tracker']
    latency = [1.0, 2.75, 25.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    bars = ax1.bar(sensors, latency, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Detection Latency [s]')
    ax1.set_title('(a) Detection Latency')
    ax1.grid(True, alpha=0.4, axis='y')
    ax1.set_ylim([0, 30])

    # Add value labels on bars
    for bar, lat in zip(bars, latency):
        ax1.annotate(f'{lat:.1f} s',
                    xy=(bar.get_x() + bar.get_width()/2, lat),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    # Panel (b): Accuracy during faults
    ax2 = axes[1]
    sensors_short = ['Magnetometer', 'Sun Sensor', 'Star Tracker']
    x = np.arange(len(sensors_short))
    width = 0.25

    eskf_err = [0.022, 0.021, 0.021]
    smoother_err = [0.018, 0.020, 0.042]
    redundant_err = [0.012, 0.012, 0.021]

    bars1 = ax2.bar(x - width, eskf_err, width, label='ESKF', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x, smoother_err, width, label='Smoother', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars3 = ax2.bar(x + width, redundant_err, width, label='Redundant', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('Mean Attitude Error [deg]')
    ax2.set_title('(b) Accuracy During Faults')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sensors_short)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_ylim([0, 0.05])

    plt.tight_layout()
    plt.savefig('fault_detection_summary_2panel.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fault_detection_summary_2panel.png', bbox_inches='tight', dpi=300)
    print("Saved: fault_detection_summary_2panel.pdf/png")


if __name__ == "__main__":
    main()
