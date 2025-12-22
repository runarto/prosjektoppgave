#!/usr/bin/env python3
"""
Plot smoother vs ESKF for magnetometer bias scenario.
Shows how the redundant estimator detects and handles the faulty sensor.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.redundant_estimator import RedundantEstimator
from utilities.utils import load_yaml

# =============================================================================
# Publication-quality plot settings
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['legend.fontsize'] = 12
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


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error magnitude in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def add_mag_bias(sim_data, mag_bias: np.ndarray):
    """Add magnetometer bias to simulation data."""
    class ModifiedData:
        pass

    modified = ModifiedData()
    modified.t = sim_data.t.copy()
    modified.jd = sim_data.jd.copy()
    modified.q_true = sim_data.q_true.copy()
    modified.omega_meas = sim_data.omega_meas.copy()
    modified.mag_meas = sim_data.mag_meas.copy()
    modified.sun_meas = sim_data.sun_meas.copy()
    modified.st_meas = sim_data.st_meas.copy()
    modified.b_eci = sim_data.b_eci.copy()
    modified.s_eci = sim_data.s_eci.copy()
    modified.b_g_true = sim_data.b_g_true.copy()

    for k in range(len(modified.mag_meas)):
        if not np.any(np.isnan(modified.mag_meas[k])):
            biased = modified.mag_meas[k] + mag_bias
            modified.mag_meas[k] = biased / np.linalg.norm(biased)

    return modified


def create_config(base_config_path: str, output_path: str):
    """Create config with standard noise scale."""
    config = load_yaml(base_config_path)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return output_path


def run_redundant_detailed(sim_data, config_path: str):
    """Run Redundant estimator and return detailed results."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(P0=P0, config_path=config_path)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times = []
    eskf_errors = []
    smoother_errors = []
    disagreements = []
    primaries = []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        jd = sim_data.jd[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        z_mag = sim_data.mag_meas[k] if not np.any(np.isnan(sim_data.mag_meas[k])) else None
        z_sun = sim_data.sun_meas[k] if not np.any(np.isnan(sim_data.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.any(np.isnan(sim_data.st_meas[k])) else None

        B_n = sim_data.b_eci[k] if z_mag is not None else None
        s_n = sim_data.s_eci[k] if z_sun is not None else None

        x_est, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        eskf_errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

        if smoother_state is not None:
            smoother_errors.append(compute_attitude_error_deg(smoother_state.ori, q_true))
        else:
            smoother_errors.append(np.nan)

        disagreements.append(disagreement)
        primaries.append(primary)

    return {
        'times': np.array(times),
        'eskf_errors': np.array(eskf_errors),
        'smoother_errors': np.array(smoother_errors),
        'disagreements': np.array(disagreements),
        'primaries': primaries,
    }


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'

    print("Loading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    # Add magnetometer bias
    mag_bias = np.array([0.05, 0.0, 0.0])  # 5% bias on X
    sim_data = add_mag_bias(sim_data_base, mag_bias)

    config_path = create_config(base_config, 'configs/config_smoother_plot.yaml')

    print("Running Redundant estimator...")
    results = run_redundant_detailed(sim_data, config_path)

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    times = results['times']
    eskf_err = results['eskf_errors']
    smoother_err = results['smoother_errors']
    disagreements = results['disagreements']
    primaries = results['primaries']

    # Plot 1: Attitude errors
    ax = axes[0]
    ax.semilogy(times, eskf_err, 'C0-', linewidth=1.5, label='ESKF', alpha=0.8)
    ax.semilogy(times, smoother_err, 'C2-', linewidth=1.5, label='Fixed-Lag Smoother', alpha=0.8)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Magnetometer Bias Scenario: ESKF vs Smoother')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([times[0], times[-1]])

    # Plot 2: Disagreement
    ax = axes[1]
    ax.plot(times, disagreements, 'C1-', linewidth=1.0)
    ax.axhline(2.0, color='r', linestyle='--', linewidth=1.5, label='Switch threshold (2°)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('ESKF-Smoother Disagreement')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([times[0], times[-1]])

    # Plot 3: Primary estimator selection
    ax = axes[2]

    # Convert primaries to numeric for plotting
    primary_map = {'ESKF': 0, 'SMOOTHER': 1, 'CONSERVATIVE': 0.5}
    primary_numeric = [primary_map.get(p, 0) for p in primaries]

    ax.fill_between(times, 0, primary_numeric, alpha=0.5, step='post')
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['ESKF', 'Conservative', 'Smoother'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Primary Estimator')
    ax.set_title('Estimator Selection')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig('smoother_mag_bias.png', dpi=150, bbox_inches='tight')
    plt.savefig('smoother_mag_bias.pdf', dpi=150, bbox_inches='tight')
    print("Saved: smoother_mag_bias.png")

    # Print statistics
    ss_mask = times >= 100.0
    print(f"\nSteady-state statistics (t >= 100s):")
    print(f"  ESKF mean error:     {np.mean(eskf_err[ss_mask]):.4f} deg")
    print(f"  Smoother mean error: {np.nanmean(smoother_err[ss_mask]):.4f} deg")
    print(f"  Mean disagreement:   {np.mean(disagreements[ss_mask]):.4f} deg")

    # Count mode switches
    conservative_count = sum(1 for p in primaries if p == 'CONSERVATIVE')
    smoother_count = sum(1 for p in primaries if p == 'SMOOTHER')
    print(f"\n  Time in ESKF mode:        {sum(1 for p in primaries if p == 'ESKF') / len(primaries) * 100:.1f}%")
    print(f"  Time in Conservative mode: {conservative_count / len(primaries) * 100:.1f}%")
    print(f"  Time in Smoother mode:     {smoother_count / len(primaries) * 100:.1f}%")


if __name__ == "__main__":
    main()
