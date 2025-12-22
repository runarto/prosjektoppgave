"""
Generate plots for the Simulation Setup chapter.

Plots:
1. Sensor measurement timeline (availability)
2. Example sensor measurements
3. Gyro bias evolution (random walk)
4. Reference vectors (B_eci, s_eci)
5. Noise characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import SimulationDatabase
from utilities.quaternion import Quaternion

# Use thesis-consistent style
from matplotlib import rcParams
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
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'


def load_baseline_data(db_path: str = "simulations.db"):
    """Load baseline simulation data."""
    db = SimulationDatabase(db_path)

    # Query runs directly
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs ORDER BY id DESC;")
    runs = cur.fetchall()

    if not runs:
        raise ValueError("No simulation runs found")

    # Find a baseline run
    run_id = None
    for rid, name in runs:
        if 'baseline' in name.lower():
            run_id = rid
            break

    # Use most recent run if no baseline found
    if run_id is None:
        run_id = runs[0][0]

    print(f"Loading run {run_id}")
    return db.load_run(run_id)


def plot_sensor_timeline(data, output_dir: Path):
    """Plot sensor measurement availability timeline."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    t = data.t

    # Gyro (always available)
    gyro_avail = ~np.isnan(data.omega_meas[:, 0])
    axes[0].fill_between(t, 0, gyro_avail.astype(float), alpha=0.7, color='C0', step='mid')
    axes[0].set_ylabel('Gyro')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['N/A', 'OK'])

    # Magnetometer
    mag_avail = ~np.isnan(data.mag_meas[:, 0])
    axes[1].fill_between(t, 0, mag_avail.astype(float), alpha=0.7, color='C1', step='mid')
    axes[1].set_ylabel('Mag')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['N/A', 'OK'])

    # Sun sensor
    sun_avail = ~np.isnan(data.sun_meas[:, 0])
    axes[2].fill_between(t, 0, sun_avail.astype(float), alpha=0.7, color='C2', step='mid')
    axes[2].set_ylabel('Sun')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['N/A', 'OK'])

    # Star tracker
    st_avail = ~np.isnan(data.st_meas[:, 0])
    axes[3].fill_between(t, 0, st_avail.astype(float), alpha=0.7, color='C3', step='mid')
    axes[3].set_ylabel('Star Tr.')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['N/A', 'OK'])

    axes[3].set_xlabel('Time [s]')
    axes[0].set_title('Sensor Measurement Availability')

    # Add measurement counts
    n_gyro = np.sum(gyro_avail)
    n_mag = np.sum(mag_avail)
    n_sun = np.sum(sun_avail)
    n_st = np.sum(st_avail)

    axes[0].text(0.98, 0.5, f'{n_gyro}', transform=axes[0].transAxes, ha='right', va='center')
    axes[1].text(0.98, 0.5, f'{n_mag}', transform=axes[1].transAxes, ha='right', va='center')
    axes[2].text(0.98, 0.5, f'{n_sun}', transform=axes[2].transAxes, ha='right', va='center')
    axes[3].text(0.98, 0.5, f'{n_st}', transform=axes[3].transAxes, ha='right', va='center')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'sensor_timeline.{ext}')
    plt.close(fig)
    print(f"Saved sensor_timeline.pdf/png")


def plot_sensor_measurements(data, output_dir: Path):
    """Plot example sensor measurements."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = data.t

    # Gyro measurements
    ax = axes[0, 0]
    valid = ~np.isnan(data.omega_meas[:, 0])
    ax.plot(t[valid], np.rad2deg(data.omega_meas[valid, 0]), 'C0-', alpha=0.7, label=r'$\omega_x$', linewidth=0.5)
    ax.plot(t[valid], np.rad2deg(data.omega_meas[valid, 1]), 'C1-', alpha=0.7, label=r'$\omega_y$', linewidth=0.5)
    ax.plot(t[valid], np.rad2deg(data.omega_meas[valid, 2]), 'C2-', alpha=0.7, label=r'$\omega_z$', linewidth=0.5)
    ax.plot(t, np.rad2deg(data.omega_true[:, 0]), 'C0--', alpha=0.5, linewidth=1)
    ax.plot(t, np.rad2deg(data.omega_true[:, 1]), 'C1--', alpha=0.5, linewidth=1)
    ax.plot(t, np.rad2deg(data.omega_true[:, 2]), 'C2--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [deg/s]')
    ax.set_title('Gyroscope measurements')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 50])  # Show first 50s for detail

    # Magnetometer measurements
    ax = axes[0, 1]
    valid = ~np.isnan(data.mag_meas[:, 0])
    t_mag = t[valid]
    ax.plot(t_mag, data.mag_meas[valid, 0], 'C0.', markersize=2, label=r'$B_x$')
    ax.plot(t_mag, data.mag_meas[valid, 1], 'C1.', markersize=2, label=r'$B_y$')
    ax.plot(t_mag, data.mag_meas[valid, 2], 'C2.', markersize=2, label=r'$B_z$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Magnetic field (body, normalized)')
    ax.set_title('Magnetometer measurements')
    ax.legend(loc='upper right')

    # Sun sensor measurements
    ax = axes[1, 0]
    valid = ~np.isnan(data.sun_meas[:, 0])
    t_sun = t[valid]
    ax.plot(t_sun, data.sun_meas[valid, 0], 'C0.', markersize=3, label=r'$s_x$')
    ax.plot(t_sun, data.sun_meas[valid, 1], 'C1.', markersize=3, label=r'$s_y$')
    ax.plot(t_sun, data.sun_meas[valid, 2], 'C2.', markersize=3, label=r'$s_z$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Sun direction (body)')
    ax.set_title('Sun sensor measurements')
    ax.legend(loc='upper right')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='FOV limit')

    # Star tracker measurements (as Euler angles)
    ax = axes[1, 1]
    valid = ~np.isnan(data.st_meas[:, 0])
    t_st = t[valid]

    # Convert to Euler angles for visualization
    euler_meas = []
    euler_true = []
    for i, v in enumerate(valid):
        if v:
            q_meas = Quaternion.from_array(data.st_meas[i])
            q_true = Quaternion.from_array(data.q_true[i])
            euler_meas.append(np.rad2deg(q_meas.as_euler()))
            euler_true.append(np.rad2deg(q_true.as_euler()))

    euler_meas = np.array(euler_meas)
    euler_true = np.array(euler_true)

    ax.plot(t_st, euler_meas[:, 0], 'C0.', markersize=4, label='Roll (meas)')
    ax.plot(t_st, euler_meas[:, 1], 'C1.', markersize=4, label='Pitch (meas)')
    ax.plot(t_st, euler_meas[:, 2], 'C2.', markersize=4, label='Yaw (meas)')
    ax.plot(t_st, euler_true[:, 0], 'C0-', alpha=0.5, linewidth=1)
    ax.plot(t_st, euler_true[:, 1], 'C1-', alpha=0.5, linewidth=1)
    ax.plot(t_st, euler_true[:, 2], 'C2-', alpha=0.5, linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Euler angles [deg]')
    ax.set_title('Star tracker measurements')
    ax.legend(loc='upper right')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'sensor_measurements.{ext}')
    plt.close(fig)
    print(f"Saved sensor_measurements.pdf/png")


def plot_gyro_bias_evolution(data, output_dir: Path):
    """Plot gyroscope bias random walk evolution."""
    fig, ax = plt.subplots(figsize=(10, 5))

    t = data.t

    # Convert to deg/s for readability
    bias_deg = np.rad2deg(data.b_g_true) * 3600  # deg/h for visibility

    ax.plot(t, bias_deg[:, 0], 'C0-', label=r'$b_{g,x}$', linewidth=1)
    ax.plot(t, bias_deg[:, 1], 'C1-', label=r'$b_{g,y}$', linewidth=1)
    ax.plot(t, bias_deg[:, 2], 'C2-', label=r'$b_{g,z}$', linewidth=1)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Gyro bias [deg/h]')
    ax.set_title('Gyroscope Bias Evolution (Random Walk)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add theoretical 1-sigma bounds
    # RRW = 0.5 deg/h/sqrt(h), so after time T (in hours), sigma = RRW * sqrt(T)
    RRW = 0.5  # deg/h/sqrt(h)
    t_hours = t / 3600
    sigma_1 = RRW * np.sqrt(t_hours)  # deg/h
    ax.fill_between(t, -sigma_1, sigma_1, alpha=0.2, color='gray', label=r'$\pm 1\sigma$ (theoretical)')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'gyro_bias_evolution.{ext}')
    plt.close(fig)
    print(f"Saved gyro_bias_evolution.pdf/png")


def plot_reference_vectors(data, output_dir: Path):
    """Plot reference vectors (magnetic field and sun direction in ECI)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t = data.t

    # Magnetic field in ECI
    ax = axes[0]
    B_norm = np.linalg.norm(data.b_eci, axis=1)
    ax.plot(t, data.b_eci[:, 0] / B_norm, 'C0-', label=r'$B_x$', linewidth=0.8)
    ax.plot(t, data.b_eci[:, 1] / B_norm, 'C1-', label=r'$B_y$', linewidth=0.8)
    ax.plot(t, data.b_eci[:, 2] / B_norm, 'C2-', label=r'$B_z$', linewidth=0.8)
    ax.set_ylabel('Magnetic field (ECI, normalized)')
    ax.set_title('Reference Vectors in ECI Frame')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Sun direction in ECI
    ax = axes[1]
    s_norm = np.linalg.norm(data.s_eci, axis=1)
    ax.plot(t, data.s_eci[:, 0] / s_norm, 'C0-', label=r'$s_x$', linewidth=0.8)
    ax.plot(t, data.s_eci[:, 1] / s_norm, 'C1-', label=r'$s_y$', linewidth=0.8)
    ax.plot(t, data.s_eci[:, 2] / s_norm, 'C2-', label=r'$s_z$', linewidth=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Sun direction (ECI)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'reference_vectors.{ext}')
    plt.close(fig)
    print(f"Saved reference_vectors.pdf/png")


def plot_measurement_noise_histogram(data, output_dir: Path):
    """Plot histogram of measurement residuals to show noise characteristics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gyro noise: difference between measured and true (minus bias)
    ax = axes[0, 0]
    valid = ~np.isnan(data.omega_meas[:, 0])
    omega_true_biased = data.omega_true + data.b_g_true
    gyro_noise = data.omega_meas[valid] - omega_true_biased[valid]
    gyro_noise_flat = gyro_noise.flatten() * 1000  # Convert to mrad/s

    ax.hist(gyro_noise_flat, bins=50, density=True, alpha=0.7, color='C0')

    # Overlay theoretical Gaussian
    sigma_theoretical = 0.309  # mrad/s from calculation
    x = np.linspace(-2, 2, 100)
    ax.plot(x, 1/(sigma_theoretical * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma_theoretical**2)),
            'r-', linewidth=2, label=f'Theory ($\\sigma$={sigma_theoretical:.2f} mrad/s)')

    ax.set_xlabel('Noise [mrad/s]')
    ax.set_ylabel('Density')
    ax.set_title('Gyroscope Measurement Noise')
    ax.legend()

    # Magnetometer noise
    ax = axes[0, 1]
    valid = ~np.isnan(data.mag_meas[:, 0])

    # Compute expected measurement
    mag_expected = []
    for i in range(len(data.t)):
        if valid[i]:
            q = Quaternion.from_array(data.q_true[i])
            R_bn = q.as_rotmat().T
            B_b = R_bn @ data.b_eci[i]
            B_b_norm = B_b / np.linalg.norm(data.b_eci[i])
            mag_expected.append(B_b_norm)

    mag_expected = np.array(mag_expected)
    mag_meas_valid = data.mag_meas[valid]
    mag_noise = mag_meas_valid - mag_expected
    mag_noise_flat = mag_noise.flatten()

    ax.hist(mag_noise_flat, bins=50, density=True, alpha=0.7, color='C1')

    sigma_mag = 0.015
    x = np.linspace(-0.1, 0.1, 100)
    ax.plot(x, 1/(sigma_mag * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma_mag**2)),
            'r-', linewidth=2, label=f'Theory ($\\sigma$={sigma_mag})')

    ax.set_xlabel('Noise (normalized)')
    ax.set_ylabel('Density')
    ax.set_title('Magnetometer Measurement Noise')
    ax.legend()

    # Sun sensor angular error
    ax = axes[1, 0]
    valid = ~np.isnan(data.sun_meas[:, 0])

    sun_errors = []
    for i in range(len(data.t)):
        if valid[i]:
            q = Quaternion.from_array(data.q_true[i])
            R_bn = q.as_rotmat().T
            s_b_true = R_bn @ data.s_eci[i]
            s_b_true = s_b_true / np.linalg.norm(s_b_true)
            s_b_meas = data.sun_meas[i]

            # Angular error
            dot = np.clip(np.dot(s_b_true, s_b_meas), -1, 1)
            angle_err = np.arccos(dot)
            sun_errors.append(np.rad2deg(angle_err))

    sun_errors = np.array(sun_errors)
    ax.hist(sun_errors, bins=30, density=True, alpha=0.7, color='C2')
    ax.axvline(1.5, color='r', linestyle='--', label=f'$\\sigma$ = 1.5°')
    ax.set_xlabel('Angular error [deg]')
    ax.set_ylabel('Density')
    ax.set_title('Sun Sensor Angular Error')
    ax.legend()

    # Star tracker angular error
    ax = axes[1, 1]
    valid = ~np.isnan(data.st_meas[:, 0])

    st_errors = []
    for i in range(len(data.t)):
        if valid[i]:
            q_true = Quaternion.from_array(data.q_true[i])
            q_meas = Quaternion.from_array(data.st_meas[i])

            # Error quaternion
            q_err = q_true.conjugate() @ q_meas
            angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
            st_errors.append(np.rad2deg(angle_err) * 3600)  # arcsec

    st_errors = np.array(st_errors)
    ax.hist(st_errors, bins=30, density=True, alpha=0.7, color='C3')
    ax.axvline(50, color='r', linestyle='--', label=r'$\sigma$ = 50 arcsec')
    ax.set_xlabel('Angular error [arcsec]')
    ax.set_ylabel('Density')
    ax.set_title('Star Tracker Angular Error')
    ax.legend()

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'measurement_noise.{ext}')
    plt.close(fig)
    print(f"Saved measurement_noise.pdf/png")


def plot_true_attitude_and_rates(data, output_dir: Path):
    """Plot true Euler angles and angular velocities."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t = data.t

    # Convert quaternions to Euler angles
    euler_angles = []
    for i in range(len(t)):
        q = Quaternion.from_array(data.q_true[i])
        euler_angles.append(np.rad2deg(q.as_euler()))
    euler_angles = np.array(euler_angles)

    # Euler angles
    ax = axes[0]
    ax.plot(t, euler_angles[:, 0], 'C0-', label='Roll')
    ax.plot(t, euler_angles[:, 1], 'C1-', label='Pitch')
    ax.plot(t, euler_angles[:, 2], 'C2-', label='Yaw')
    ax.set_ylabel('Euler angles [deg]')
    ax.set_title('True Attitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Angular velocities
    ax = axes[1]
    ax.plot(t, np.rad2deg(data.omega_true[:, 0]), 'C0-', label=r'$\omega_x$')
    ax.plot(t, np.rad2deg(data.omega_true[:, 1]), 'C1-', label=r'$\omega_y$')
    ax.plot(t, np.rad2deg(data.omega_true[:, 2]), 'C2-', label=r'$\omega_z$')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [deg/s]')
    ax.set_title('True Angular Velocity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'true_euler_angles.{ext}')
    plt.close(fig)
    print(f"Saved true_euler_angles.pdf/png")


def plot_angular_velocity_profile(data, output_dir: Path):
    """Plot angular velocity profile showing the sinusoidal components."""
    fig, ax = plt.subplots(figsize=(10, 5))

    t = data.t

    ax.plot(t, np.rad2deg(data.omega_true[:, 0]), 'C0-', label=r'$\omega_x$ ($A=0.02$, $f=0.01$ Hz)')
    ax.plot(t, np.rad2deg(data.omega_true[:, 1]), 'C1-', label=r'$\omega_y$ ($A=0.01$, $f=0.008$ Hz)')
    ax.plot(t, np.rad2deg(data.omega_true[:, 2]), 'C2-', label=r'$\omega_z$ ($A=0.015$, $f=0.005$ Hz)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [deg/s]')
    ax.set_title('Sinusoidal Angular Velocity Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add amplitude markers
    ax.axhline(np.rad2deg(0.02), color='C0', linestyle='--', alpha=0.3)
    ax.axhline(-np.rad2deg(0.02), color='C0', linestyle='--', alpha=0.3)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'angular_velocity_profile.{ext}')
    plt.close(fig)
    print(f"Saved angular_velocity_profile.pdf/png")


def main():
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading simulation data...")
    data = load_baseline_data()

    print("\nGenerating plots...")

    # Generate all plots
    plot_true_attitude_and_rates(data, output_dir)
    plot_angular_velocity_profile(data, output_dir)
    plot_sensor_timeline(data, output_dir)
    plot_sensor_measurements(data, output_dir)
    plot_gyro_bias_evolution(data, output_dir)
    plot_reference_vectors(data, output_dir)
    plot_measurement_noise_histogram(data, output_dir)

    print("\nAll plots saved to figures/")


if __name__ == "__main__":
    main()
