#!/usr/bin/env python3
"""
Detailed analysis of magnetometer bias scenario.

Shows:
1. Attitude error comparison between all estimators
2. Redundant system switching behavior and reasons
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from collections import deque

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
from estimation.keyframe_fgo import KeyframeFGO
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


class MockEnvironment:
    """Mock environment using pre-computed simulation data."""
    def __init__(self, sim_data):
        self.sim_data = sim_data

    def _find_idx(self, jd):
        return np.argmin(np.abs(self.sim_data.jd - jd))

    def get_r_eci(self, jd):
        return np.array([7000e3, 0, 0])

    def get_B_eci(self, r_eci, jd):
        return self.sim_data.b_eci[self._find_idx(jd)]

    def get_sun_eci(self, jd):
        return self.sim_data.s_eci[self._find_idx(jd)]


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


def run_eskf(sim_data, config_path: str):
    """Run ESKF and return detailed results."""
    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    eskf = ESKF(P0=P0, config_path=config_path, chi2_threshold=1e10)
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times, errors = [], []

    for k in range(1, len(sim_data.t)):
        t = sim_data.t[k]
        dt = sim_data.t[k] - sim_data.t[k-1]
        omega = sim_data.omega_meas[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        x_est = eskf.predict(x_est, omega, dt)

        if not np.any(np.isnan(sim_data.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.mag_meas[k],
                                   SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except:
                pass

        if not np.any(np.isnan(sim_data.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim_data.sun_meas[k],
                                   SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except:
                pass

        if not np.any(np.isnan(sim_data.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim_data.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except:
                pass

        times.append(t)
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))

    return np.array(times), np.array(errors)


def run_isam2(sim_data, config_path: str):
    """Run iSAM2 and return detailed results."""
    env = MockEnvironment(sim_data)
    fgo = KeyframeFGO(config_path=config_path, use_isam2=True, use_rk4=True)

    kf_times, kf_states = fgo.process_simulation(sim_data, env)
    times, states = fgo.interpolate_full_rate(kf_times, kf_states, sim_data)

    errors = []
    for i, t in enumerate(times):
        idx = np.argmin(np.abs(sim_data.t - t))
        q_true = Quaternion.from_array(sim_data.q_true[idx])
        errors.append(compute_attitude_error_deg(states[i].ori, q_true))

    return np.array(times), np.array(errors)


def run_redundant_detailed(sim_data, config_path: str):
    """Run Redundant estimator and return detailed diagnostics."""
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

    # Storage for detailed diagnostics
    times = []
    errors = []
    disagreements = []
    modes = []  # "ESKF", "SMOOTHER", "CONSERVATIVE"
    nis_mag_list = []
    nis_sun_list = []
    nis_st_list = []

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
        errors.append(compute_attitude_error_deg(x_est.nom.ori, q_true))
        disagreements.append(disagreement)
        modes.append(primary)

        # Get latest NIS values (use NaN if not available)
        nis_mag_list.append(list(redundant.nis_history_mag)[-1] if redundant.nis_history_mag else np.nan)
        nis_sun_list.append(list(redundant.nis_history_sun)[-1] if redundant.nis_history_sun else np.nan)
        nis_st_list.append(list(redundant.nis_history_st)[-1] if redundant.nis_history_st else np.nan)

    # Get switch events
    switch_events = redundant.switch_events

    return {
        'times': np.array(times),
        'errors': np.array(errors),
        'disagreements': np.array(disagreements),
        'modes': modes,
        'nis_mag': np.array(nis_mag_list),
        'nis_sun': np.array(nis_sun_list),
        'nis_st': np.array(nis_st_list),
        'switch_events': switch_events,
        'estimator': redundant,
    }


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


def plot_mag_bias_comparison(times_eskf, errors_eskf, times_isam, errors_isam,
                              redundant_data, save_path):
    """Plot attitude error comparison for mag bias scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(times_eskf, errors_eskf, 'C0-', linewidth=1.5, label='ESKF')
    ax.semilogy(times_isam, errors_isam, 'C1-', linewidth=1.5, label='iSAM2')
    ax.semilogy(redundant_data['times'], redundant_data['errors'], 'C2--',
               linewidth=1.5, label='Redundant')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Magnetometer Bias Scenario: Attitude Error Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([times_eskf[0], times_eskf[-1]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_redundant_switching(redundant_data, save_path):
    """Plot detailed redundant system switching behavior."""
    times = redundant_data['times']
    errors = redundant_data['errors']
    disagreements = redundant_data['disagreements']
    modes = redundant_data['modes']
    nis_mag = redundant_data['nis_mag']
    nis_sun = redundant_data['nis_sun']
    nis_st = redundant_data['nis_st']
    switch_events = redundant_data['switch_events']

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel 1: Attitude Error with mode shading
    ax = axes[0]
    ax.semilogy(times, errors, 'C2-', linewidth=1.5, label='Redundant')

    # Color background by mode
    mode_colors = {'ESKF': 'lightblue', 'SMOOTHER': 'lightgreen', 'CONSERVATIVE': 'lightyellow'}
    current_mode = modes[0]
    start_idx = 0

    for i, mode in enumerate(modes):
        if mode != current_mode or i == len(modes) - 1:
            end_idx = i if mode != current_mode else i + 1
            ax.axvspan(times[start_idx], times[end_idx-1],
                      alpha=0.3, color=mode_colors.get(current_mode, 'white'))
            current_mode = mode
            start_idx = i

    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Redundant Estimator: Attitude Error with Mode Indication')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add mode legend
    legend_elements = [Patch(facecolor=mode_colors['ESKF'], alpha=0.3, label='ESKF Mode'),
                       Patch(facecolor=mode_colors['SMOOTHER'], alpha=0.3, label='Smoother Mode'),
                       Patch(facecolor=mode_colors['CONSERVATIVE'], alpha=0.3, label='Conservative Mode')]
    ax.legend(handles=legend_elements, loc='upper left')

    # Panel 2: ESKF-Smoother Disagreement
    ax = axes[1]
    ax.plot(times, disagreements, 'C3-', linewidth=1.0)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, label='Threshold (0.5°)')
    ax.set_ylabel('Disagreement [deg]')
    ax.set_title('ESKF vs Smoother Disagreement')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(2.0, np.nanmax(disagreements) * 1.1)])

    # Panel 3: Per-Sensor NIS
    ax = axes[2]
    ax.semilogy(times, nis_mag, 'C0-', linewidth=1.0, alpha=0.7, label='Magnetometer')
    ax.semilogy(times, nis_sun, 'C1-', linewidth=1.0, alpha=0.7, label='Sun Sensor')
    ax.semilogy(times, nis_st, 'C2-', linewidth=1.0, alpha=0.7, label='Star Tracker')
    ax.axhline(y=7.81, color='r', linestyle='--', linewidth=1.5, label='χ²(3, 0.95) = 7.81')
    ax.set_ylabel('NIS')
    ax.set_title('Per-Sensor Normalized Innovation Squared (NIS)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0.1, 100])

    # Panel 4: Mode timeline
    ax = axes[3]
    mode_to_num = {'ESKF': 0, 'SMOOTHER': 1, 'CONSERVATIVE': 2}
    mode_nums = [mode_to_num.get(m, 0) for m in modes]
    ax.step(times, mode_nums, 'k-', linewidth=2, where='post')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['ESKF', 'Smoother', 'Conservative'])
    ax.set_ylabel('Active Mode')
    ax.set_xlabel('Time [s]')
    ax.set_title('Primary Estimator Mode Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 2.5])

    # Mark switch events
    for t_switch, transition in switch_events:
        ax.axvline(x=t_switch, color='r', linestyle=':', alpha=0.7)
        ax.annotate(transition, xy=(t_switch, 2.3), fontsize=8, rotation=45,
                   ha='left', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'
    mag_bias = np.array([0.05, 0.0, 0.0])  # 5% bias on X-axis

    print("=" * 70)
    print("MAGNETOMETER BIAS SCENARIO ANALYSIS")
    print("=" * 70)

    config_path = create_config(base_config, 'configs/config_mag_analysis.yaml')
    db = SimulationDatabase(db_path)

    print("\nLoading simulation data...")
    sim_data_base = db.load_run(237)

    print(f"Adding magnetometer bias: {mag_bias}")
    sim_data = add_mag_bias(sim_data_base, mag_bias)

    # Run all estimators
    print("\nRunning ESKF...")
    times_eskf, errors_eskf = run_eskf(sim_data, config_path)

    print("Running iSAM2...")
    times_isam, errors_isam = run_isam2(sim_data, config_path)

    print("Running Redundant (with detailed diagnostics)...")
    redundant_data = run_redundant_detailed(sim_data, config_path)

    # Print switch events
    print("\n" + "=" * 70)
    print("REDUNDANT SYSTEM SWITCH EVENTS")
    print("=" * 70)
    if redundant_data['switch_events']:
        for t_switch, transition in redundant_data['switch_events']:
            print(f"  t = {t_switch:7.2f}s: {transition}")
    else:
        print("  No switch events occurred")

    # Print diagnostics summary
    print("\n" + "=" * 70)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 70)
    stats = redundant_data['estimator'].get_statistics()
    print(f"  Final primary estimator: {stats['primary']}")
    print(f"  Total switch events: {stats['total_switch_events']}")
    print(f"  ESKF health score: {stats['eskf_health_score']:.1%}")
    print(f"  Faulty sensor: {stats['faulty_sensor']}")
    print(f"  Sensor health scores: {stats['sensor_health_scores']}")

    # Generate plots
    print("\nGenerating plots...")
    plot_mag_bias_comparison(times_eskf, errors_eskf, times_isam, errors_isam,
                             redundant_data, 'mag_bias_comparison.png')

    plot_redundant_switching(redundant_data, 'redundant_switching_analysis.png')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
