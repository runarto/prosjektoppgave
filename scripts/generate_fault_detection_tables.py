#!/usr/bin/env python3
"""
Generate LaTeX tables for fault detection results.

Creates two tables:
1. Detection latency by sensor
2. Attitude error during faults by estimator
"""


def generate_detection_latency_table():
    """Generate LaTeX table for detection latency."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Fault detection latency by sensor type}
\label{tab:detection_latency}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Sensor} & \textbf{Measurement Rate [Hz]} & \textbf{Detection Latency [s]} \\
\midrule
Magnetometer & 1.0 & 1.0 \\
Sun Sensor & 0.4 & 2.75 \\
Star Tracker & 0.04 & 25.0 \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_accuracy_table():
    """Generate LaTeX table for attitude error during faults."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Mean attitude error during sensor faults (fault window: 100--150\,s)}
\label{tab:fault_accuracy}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Faulty Sensor} & \textbf{ESKF} & \textbf{Smoother} & \textbf{Redundant} \\
\midrule
Magnetometer & 0.022$^\circ$ & 0.018$^\circ$ & 0.012$^\circ$ \\
Sun Sensor & 0.021$^\circ$ & 0.020$^\circ$ & 0.012$^\circ$ \\
Star Tracker & 0.021$^\circ$ & 0.042$^\circ$ & 0.021$^\circ$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_combined_table():
    """Generate a combined table with all fault detection results."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Fault detection performance summary}
\label{tab:fault_detection_summary}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{Detection}} & \multicolumn{3}{c}{\textbf{Mean Error During Fault [$^\circ$]}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-7}
\textbf{Sensor} & \textbf{Rate [Hz]} & \textbf{Success} & \textbf{Latency [s]} & \textbf{ESKF} & \textbf{Smoother} & \textbf{Redundant} \\
\midrule
Magnetometer & 1.0 & 100\% & 1.0 & 0.022 & 0.018 & 0.012 \\
Sun Sensor & 0.4 & 100\% & 2.75 & 0.021 & 0.020 & 0.012 \\
Star Tracker & 0.04 & 100\% & 25.0 & 0.021 & 0.042 & 0.021 \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    # Generate individual tables
    latency_table = generate_detection_latency_table()
    accuracy_table = generate_accuracy_table()
    combined_table = generate_combined_table()

    # Save detection latency table
    with open('fault_detection_latency_table.tex', 'w') as f:
        f.write(latency_table)
    print("Saved: fault_detection_latency_table.tex")

    # Save accuracy table
    with open('fault_detection_accuracy_table.tex', 'w') as f:
        f.write(accuracy_table)
    print("Saved: fault_detection_accuracy_table.tex")

    # Save combined table
    with open('fault_detection_summary_table.tex', 'w') as f:
        f.write(combined_table)
    print("Saved: fault_detection_summary_table.tex")

    # Print tables to console
    print("\n" + "=" * 60)
    print("DETECTION LATENCY TABLE")
    print("=" * 60)
    print(latency_table)

    print("\n" + "=" * 60)
    print("ACCURACY DURING FAULTS TABLE")
    print("=" * 60)
    print(accuracy_table)

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY TABLE")
    print("=" * 60)
    print(combined_table)


if __name__ == "__main__":
    main()
