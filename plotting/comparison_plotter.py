"""
Publication-quality comparison plots for ESKF vs Factor Graph Optimization.

Generates clean, professional plots suitable for technical reports and papers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Publication settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.0,
    'grid.alpha': 0.3,
})


@dataclass
class EstimationResults:
    """Container for estimation results."""
    t: np.ndarray                    # Time vector
    attitude_error: np.ndarray       # Attitude error (degrees)
    bias_error: np.ndarray           # Bias error (rad/s)
    sigma_attitude: np.ndarray       # Attitude uncertainty (degrees)
    sigma_bias: np.ndarray           # Bias uncertainty (rad/s)
    method_name: str                 # "ESKF" or "FGO"
    scenario_name: str               # Scenario description


class ComparisonPlotter:
    """Generate comparison plots for ESKF vs FGO."""

    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        self.figsize = figsize

    def plot_attitude_comparison(
        self,
        results: List[EstimationResults],
        filename: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Plot attitude error comparison between methods.

        Args:
            results: List of EstimationResults for each method
            filename: Output filename (if None, shows plot)
            title: Plot title (if None, uses scenario name)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        colors = {'ESKF': '#1f77b4', 'FGO': '#ff7f0e'}

        for res in results:
            color = colors.get(res.method_name, '#2ca02c')

            # Attitude error
            ax1.plot(res.t, res.attitude_error, label=res.method_name,
                    color=color, alpha=0.8)

            # Uncertainty (3-sigma bound)
            if res.sigma_attitude is not None:
                ax1.fill_between(res.t,
                                -3*res.sigma_attitude,
                                3*res.sigma_attitude,
                                color=color, alpha=0.15,
                                label=f'{res.method_name} ±3σ')

        ax1.set_ylabel('Attitude Error (deg)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        if title:
            ax1.set_title(title)
        elif results:
            ax1.set_title(f'Attitude Estimation: {results[0].scenario_name}')

        # Bias error (if available)
        for res in results:
            if res.bias_error is not None:
                color = colors.get(res.method_name, '#2ca02c')
                ax2.semilogy(res.t, np.abs(res.bias_error),
                           label=res.method_name, color=color, alpha=0.8)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Bias Error (rad/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
        else:
            plt.show()

    def plot_statistics_comparison(
        self,
        results: List[EstimationResults],
        filename: Optional[str] = None
    ):
        """
        Plot statistical comparison (mean, std, max errors).

        Args:
            results: List of EstimationResults for each method
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        methods = [r.method_name for r in results]
        att_mean = [np.mean(r.attitude_error) for r in results]
        att_std = [np.std(r.attitude_error) for r in results]
        att_max = [np.max(np.abs(r.attitude_error)) for r in results]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Mean error
        axes[0].bar(methods, att_mean, color=colors[:len(methods)])
        axes[0].set_ylabel('Mean Error (deg)')
        axes[0].set_title('Mean Attitude Error')
        axes[0].grid(True, axis='y', alpha=0.3)

        # Std deviation
        axes[1].bar(methods, att_std, color=colors[:len(methods)])
        axes[1].set_ylabel('Std Dev (deg)')
        axes[1].set_title('Attitude Error Std Dev')
        axes[1].grid(True, axis='y', alpha=0.3)

        # Max error
        axes[2].bar(methods, att_max, color=colors[:len(methods)])
        axes[2].set_ylabel('Max Error (deg)')
        axes[2].set_title('Maximum Attitude Error')
        axes[2].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
        else:
            plt.show()

    def plot_multi_scenario_comparison(
        self,
        scenarios: Dict[str, List[EstimationResults]],
        filename: Optional[str] = None
    ):
        """
        Plot comparison across multiple scenarios.

        Args:
            scenarios: Dict mapping scenario name to list of results
            filename: Output filename
        """
        n_scenarios = len(scenarios)
        fig, axes = plt.subplots(n_scenarios, 1, figsize=(10, 3*n_scenarios), sharex=True)

        if n_scenarios == 1:
            axes = [axes]

        colors = {'ESKF': '#1f77b4', 'FGO': '#ff7f0e'}

        for ax, (scenario_name, results) in zip(axes, scenarios.items()):
            for res in results:
                color = colors.get(res.method_name, '#2ca02c')
                ax.plot(res.t, res.attitude_error,
                       label=res.method_name, color=color, alpha=0.8)

            ax.set_ylabel('Attitude Error (deg)')
            ax.set_title(scenario_name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
        else:
            plt.show()

    def plot_convergence_comparison(
        self,
        results: List[EstimationResults],
        window_size: int = 100,
        filename: Optional[str] = None
    ):
        """
        Plot convergence comparison using moving average.

        Args:
            results: List of EstimationResults
            window_size: Moving average window size
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = {'ESKF': '#1f77b4', 'FGO': '#ff7f0e'}

        for res in results:
            color = colors.get(res.method_name, '#2ca02c')

            # Compute moving average
            moving_avg = np.convolve(
                np.abs(res.attitude_error),
                np.ones(window_size)/window_size,
                mode='valid'
            )
            t_avg = res.t[:len(moving_avg)]

            ax.semilogy(t_avg, moving_avg,
                       label=f'{res.method_name} (MA-{window_size})',
                       color=color, alpha=0.8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Attitude Error (deg, moving avg)')
        ax.set_title('Convergence Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
        else:
            plt.show()


def create_summary_table(results_dict: Dict[str, List[EstimationResults]]) -> str:
    """
    Create a LaTeX summary table of results.

    Args:
        results_dict: Dict mapping scenario to list of results

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\hline")
    lines.append(r"Scenario & \multicolumn{2}{c|}{Mean Error (deg)} & \multicolumn{2}{c}{Max Error (deg)} \\")
    lines.append(r" & ESKF & FGO & ESKF & FGO \\")
    lines.append(r"\hline")

    for scenario_name, results in results_dict.items():
        eskf_res = next((r for r in results if r.method_name == "ESKF"), None)
        fgo_res = next((r for r in results if r.method_name == "FGO"), None)

        if eskf_res and fgo_res:
            eskf_mean = np.mean(np.abs(eskf_res.attitude_error))
            fgo_mean = np.mean(np.abs(fgo_res.attitude_error))
            eskf_max = np.max(np.abs(eskf_res.attitude_error))
            fgo_max = np.max(np.abs(fgo_res.attitude_error))

            lines.append(f"{scenario_name} & {eskf_mean:.4f} & {fgo_mean:.4f} & {eskf_max:.4f} & {fgo_max:.4f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Attitude estimation performance comparison}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
