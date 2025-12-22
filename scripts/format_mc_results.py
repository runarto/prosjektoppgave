#!/usr/bin/env python3
"""
Format Monte Carlo results into publication-quality table and plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================================================================
# Publication-quality plot settings
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


# Monte Carlo results (from the simulation)
results = {
    'Baseline': {
        'ESKF': {'mean': 0.0147, 'std': 0.0017, 'mean_std': 0.0059},
        'iSAM2': {'mean': 0.0368, 'std': 0.0017, 'mean_std': 0.0200},
        'Redundant': {'mean': 0.0147, 'std': 0.0017, 'mean_std': 0.0059},
    },
    'Eclipse': {
        'ESKF': {'mean': 0.0515, 'std': 0.1075, 'mean_std': 0.1625},
        'iSAM2': {'mean': 0.0351, 'std': 0.0018, 'mean_std': 0.0197},
        'Redundant': {'mean': 0.0505, 'std': 0.1043, 'mean_std': 0.1581},
    },
    'Gyro Drift': {
        'ESKF': {'mean': 0.0147, 'std': 0.0017, 'mean_std': 0.0059},
        'iSAM2': {'mean': 0.0368, 'std': 0.0018, 'mean_std': 0.0201},
        'Redundant': {'mean': 0.0147, 'std': 0.0017, 'mean_std': 0.0059},
    },
    'Mag Bias': {
        'ESKF': {'mean': 3.1201, 'std': 0.1153, 'mean_std': 0.3563},
        'iSAM2': {'mean': 0.2127, 'std': 0.0189, 'mean_std': 0.1846},
        'Redundant': {'mean': 3.1301, 'std': 0.1255, 'mean_std': 0.3556},
    },
}

scenarios = ['Baseline', 'Eclipse', 'Gyro Drift', 'Mag Bias']
estimators = ['ESKF', 'iSAM2', 'Redundant']


def generate_latex_table():
    """Generate publication-quality LaTeX table."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo Attitude Estimation Performance ($N=10$ runs per scenario).
Mean steady-state attitude error and standard deviation across runs are reported.}
\label{tab:mc_results}
\begin{tabular}{@{}llcc@{}}
\toprule
\textbf{Scenario} & \textbf{Estimator} & \textbf{Mean Error} [deg] & \textbf{Std Across Runs} [deg] \\
\midrule
"""

    for i, scenario in enumerate(scenarios):
        for j, est in enumerate(estimators):
            r = results[scenario][est]

            # Scenario name only on first row
            sc_name = scenario if j == 0 else ""

            # Format based on magnitude
            if r['mean'] >= 1.0:
                mean_str = f"{r['mean']:.2f}"
                std_str = f"{r['std']:.2f}"
            elif r['mean'] >= 0.1:
                mean_str = f"{r['mean']:.3f}"
                std_str = f"{r['std']:.3f}"
            else:
                mean_str = f"{r['mean']:.4f}"
                std_str = f"{r['std']:.4f}"

            table += f"{sc_name} & {est} & {mean_str} & {std_str} \\\\\n"

        # Add midrule between scenarios (except after last)
        if i < len(scenarios) - 1:
            table += r"\midrule" + "\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}"""

    return table


def generate_compact_table():
    """Generate compact table with scenarios as columns."""
    table = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo Steady-State Attitude Error Comparison ($N=10$ runs).
Values show mean $\pm$ std [deg] across Monte Carlo runs.}
\label{tab:mc_compact}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Estimator} & \textbf{Baseline} & \textbf{Eclipse} & \textbf{Gyro Drift} & \textbf{Mag Bias} \\
\midrule
"""

    for est in estimators:
        row = f"{est}"
        for scenario in scenarios:
            r = results[scenario][est]
            if r['mean'] >= 1.0:
                val = f"${r['mean']:.2f} \\pm {r['std']:.2f}$"
            elif r['mean'] >= 0.1:
                val = f"${r['mean']:.3f} \\pm {r['std']:.3f}$"
            else:
                val = f"${r['mean']:.4f} \\pm {r['std']:.4f}$"
            row += f" & {val}"
        row += r" \\" + "\n"
        table += row

    table += r"""\bottomrule
\end{tabular}
\end{table}"""

    return table


def plot_bar_comparison():
    """Create bar chart comparing estimators across scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(scenarios))
    width = 0.25

    colors = {'ESKF': '#1f77b4', 'iSAM2': '#ff7f0e', 'Redundant': '#2ca02c'}

    for i, est in enumerate(estimators):
        means = [results[s][est]['mean'] for s in scenarios]
        stds = [results[s][est]['std'] for s in scenarios]

        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=est, color=colors[est], capsize=4, alpha=0.85,
                     error_kw={'linewidth': 1.5})

    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    ax.set_ylim([0.005, 10])

    # Add horizontal reference lines
    ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig('mc_results_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig('mc_results_bar.pdf', dpi=150, bbox_inches='tight')
    print("Saved: mc_results_bar.png")
    plt.close()


def plot_heatmap():
    """Create heatmap of mean errors."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create data matrix
    data = np.array([[results[s][e]['mean'] for s in scenarios] for e in estimators])

    # Use log scale for color
    im = ax.imshow(np.log10(data), cmap='RdYlGn_r', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(estimators)))
    ax.set_xticklabels(scenarios)
    ax.set_yticklabels(estimators)

    # Annotate cells
    for i in range(len(estimators)):
        for j in range(len(scenarios)):
            val = data[i, j]
            if val >= 1.0:
                text = f"{val:.2f}"
            elif val >= 0.1:
                text = f"{val:.3f}"
            else:
                text = f"{val:.4f}"

            # Choose text color based on background
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=14)

    ax.set_title('Mean Attitude Error [deg]')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(Error [deg])')

    plt.tight_layout()
    plt.savefig('mc_results_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig('mc_results_heatmap.pdf', dpi=150, bbox_inches='tight')
    print("Saved: mc_results_heatmap.png")
    plt.close()


def plot_scenario_focus():
    """Create focused plots for each scenario type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'ESKF': '#1f77b4', 'iSAM2': '#ff7f0e', 'Redundant': '#2ca02c'}

    # Left: Nominal scenarios (Baseline, Eclipse, Gyro Drift)
    ax = axes[0]
    nominal_scenarios = ['Baseline', 'Eclipse', 'Gyro Drift']
    x = np.arange(len(nominal_scenarios))
    width = 0.25

    for i, est in enumerate(estimators):
        means = [results[s][est]['mean'] for s in nominal_scenarios]
        stds = [results[s][est]['std'] for s in nominal_scenarios]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=est,
               color=colors[est], capsize=4, alpha=0.85)

    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_xticks(x)
    ax.set_xticklabels(nominal_scenarios)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Nominal Scenarios')
    ax.set_ylim([0, 0.2])

    # Right: Fault scenario (Mag Bias)
    ax = axes[1]
    fault_scenario = 'Mag Bias'
    x = np.arange(len(estimators))

    means = [results[fault_scenario][e]['mean'] for e in estimators]
    stds = [results[fault_scenario][e]['std'] for e in estimators]

    bars = ax.bar(x, means, 0.6, yerr=stds,
                  color=[colors[e] for e in estimators], capsize=5, alpha=0.85)

    ax.set_ylabel('Mean Attitude Error [deg]')
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Magnetometer Bias Scenario')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{mean:.2f}°', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig('mc_results_split.png', dpi=150, bbox_inches='tight')
    plt.savefig('mc_results_split.pdf', dpi=150, bbox_inches='tight')
    print("Saved: mc_results_split.png")
    plt.close()


def main():
    print("=" * 60)
    print("FORMATTING MONTE CARLO RESULTS")
    print("=" * 60)

    # Generate LaTeX tables
    print("\n--- Vertical Table ---")
    table1 = generate_latex_table()
    print(table1)

    with open('mc_results_vertical.tex', 'w') as f:
        f.write(table1)
    print("\nSaved: mc_results_vertical.tex")

    print("\n--- Compact Table ---")
    table2 = generate_compact_table()
    print(table2)

    with open('mc_results_compact.tex', 'w') as f:
        f.write(table2)
    print("\nSaved: mc_results_compact.tex")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_bar_comparison()
    plot_heatmap()
    plot_scenario_focus()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
