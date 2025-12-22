#!/usr/bin/env python3
"""
Plot smoother attitude error for magnetometer bias scenario.
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

# Publication-quality plot settings
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
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


def add_mag_bias(sim_data, mag_bias: np.ndarray):
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


def main():
    base_config = 'configs/config_baseline_short.yaml'
    db_path = 'simulations.db'

    print("Loading simulation data...")
    db = SimulationDatabase(db_path)
    sim_data_base = db.load_run(237)

    # Add magnetometer bias
    mag_bias = np.array([0.05, 0.0, 0.0])
    sim_data = add_mag_bias(sim_data_base, mag_bias)

    config = load_yaml(base_config)
    config['sensors']['mag']['scaling']['noise_scale'] = 1.0
    config['sensors']['sun']['scaling']['noise_scale'] = 1.0
    config['sensors']['star']['scaling']['noise_scale'] = 1.0

    import yaml
    with open('configs/config_smoother_plot.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    att_err_rad = np.deg2rad(17.0)
    P0 = np.diag([att_err_rad**2] * 3 + [1e-6] * 3)

    q0_true = Quaternion.from_array(sim_data.q_true[0])
    perturb = np.array([att_err_rad] * 3) / np.sqrt(3)
    q0_est = (q0_true @ Quaternion.from_avec(perturb)).normalize()

    redundant = RedundantEstimator(P0=P0, config_path='configs/config_smoother_plot.yaml')
    x_est = EskfState(
        nom=NominalState(ori=q0_est, gyro_bias=np.zeros(3)),
        err=MultiVarGauss(np.zeros(6), P0.copy())
    )

    times = []
    smoother_errors = []

    print("Running estimator...")
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

        x_est, smoother_state, _, _ = redundant.step(
            x_eskf=x_est, t=t, jd=jd, omega_meas=omega, dt=dt,
            z_mag=z_mag, z_sun=z_sun, z_st=z_st, B_n=B_n, s_n=s_n,
        )

        times.append(t)
        if smoother_state is not None:
            smoother_errors.append(compute_attitude_error_deg(smoother_state.ori, q_true))
        else:
            smoother_errors.append(np.nan)

    times = np.array(times)
    smoother_errors = np.array(smoother_errors)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(times, smoother_errors, 'C0-', linewidth=1.5)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Fixed-Lag Smoother: Magnetometer Bias Scenario')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([times[0], times[-1]])

    plt.tight_layout()
    plt.savefig('smoother_mag_bias.png', dpi=150, bbox_inches='tight')
    plt.savefig('smoother_mag_bias.pdf', dpi=150, bbox_inches='tight')
    print("Saved: smoother_mag_bias.png")


if __name__ == "__main__":
    main()
