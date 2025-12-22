"""
Compare convergence of ESKF, Smoother, and Redundant estimator
under nominal conditions with 10 degree initial error.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

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

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss


def load_nominal_data(db_path: str = "simulations.db"):
    """Load a nominal simulation run."""
    db = SimulationDatabase(db_path)
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs ORDER BY id DESC;")
    runs = cur.fetchall()

    # Find baseline/nominal run
    run_id = None
    for rid, name in runs:
        if 'baseline' in name.lower() or 'nominal' in name.lower():
            run_id = rid
            break

    if run_id is None:
        run_id = runs[0][0]

    print(f"Loading run {run_id}")
    return db.load_run(run_id)


def create_initial_error(q_true: Quaternion, error_deg: float, axis: np.ndarray = None) -> Quaternion:
    """Create initial quaternion with specified error from truth."""
    if axis is None:
        # Random axis
        axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    error_rad = np.deg2rad(error_deg)
    q_error = Quaternion.from_avec(axis * error_rad)

    # Apply error: q_init = q_true ⊗ q_error
    return q_true @ q_error


def compute_attitude_error(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate() @ q_est
    angle = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle)


def run_eskf(data, q_init: Quaternion, config_path: str = "configs/config_baseline_short.yaml"):
    """Run ESKF and return attitude errors over time."""
    # Initial covariance
    P0 = np.diag([
        np.deg2rad(10)**2, np.deg2rad(10)**2, np.deg2rad(10)**2,  # attitude
        1e-6, 1e-6, 1e-6  # bias
    ])

    eskf = ESKF(P0=P0, config_path=config_path)

    # Initialize state
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    errors = []
    covariances = []

    for k in range(len(data.t)):
        # Get true quaternion
        q_true = Quaternion.from_array(data.q_true[k])

        # Compute error before update
        err = compute_attitude_error(x.nom.ori, q_true)
        errors.append(err)
        covariances.append(np.sqrt(np.trace(x.err.cov[:3, :3])))

        # Prediction
        if not np.isnan(data.omega_meas[k, 0]):
            dt = data.t[1] - data.t[0] if k == 0 else data.t[k] - data.t[k-1]
            x = eskf.predict(x, data.omega_meas[k], dt)

        # Magnetometer update
        if not np.isnan(data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, data.mag_meas[k], SensorType.MAGNETOMETER, B_n=data.b_eci[k])
            except:
                pass

        # Sun sensor update
        if not np.isnan(data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, data.sun_meas[k], SensorType.SUN_VECTOR, s_n=data.s_eci[k])
            except:
                pass

        # Star tracker update
        if not np.isnan(data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except:
                pass

    return np.array(errors), np.array(covariances)


def run_smoother(data, q_init: Quaternion, config_path: str = "configs/config_baseline_short.yaml"):
    """Run fixed-lag smoother and return attitude errors over time."""
    smoother = FixedLagAttitudeSmoother(
        config_path=config_path,
        lag=60.0,
        use_robust=False,  # No robust kernels for fair comparison
        normalize_mag=True,
    )

    # Initialize
    smoother.initialize(data.t[0], q_init.copy(), np.zeros(3))

    errors = []
    dt = data.t[1] - data.t[0]

    for k in range(len(data.t)):
        q_true = Quaternion.from_array(data.q_true[k])

        # Get current smoother state
        state = smoother.get_state()
        if state is not None:
            err = compute_attitude_error(state.ori, q_true)
        else:
            err = compute_attitude_error(q_init, q_true)
        errors.append(err)

        # Integrate gyro
        if not np.isnan(data.omega_meas[k, 0]):
            smoother.integrate_gyro(data.omega_meas[k], dt, t=data.t[k])

        # Add measurements
        z_mag = data.mag_meas[k] if not np.isnan(data.mag_meas[k, 0]) else None
        z_sun = data.sun_meas[k] if not np.isnan(data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(data.st_meas[k]) if not np.isnan(data.st_meas[k, 0]) else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            smoother.add_measurement(
                t=data.t[k],
                jd=data.jd[k],
                z_mag=z_mag,
                z_sun=z_sun,
                z_st=z_st,
                B_eci=data.b_eci[k],
                s_eci=data.s_eci[k],
            )

    return np.array(errors)


def run_redundant(data, q_init: Quaternion, config_path: str = "configs/config_baseline_short.yaml"):
    """Run redundant estimator and return attitude errors over time."""
    P0 = np.diag([
        np.deg2rad(10)**2, np.deg2rad(10)**2, np.deg2rad(10)**2,
        1e-6, 1e-6, 1e-6
    ])

    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=60.0,
        use_robust=False,
    )

    # Initialize ESKF state
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    # Initialize redundant estimator
    redundant.initialize(data.t[0], q_init.copy(), np.zeros(3))

    errors_eskf = []
    errors_smoother = []
    errors_output = []
    primary_modes = []

    dt = data.t[1] - data.t[0]

    for k in range(len(data.t)):
        q_true = Quaternion.from_array(data.q_true[k])

        # Prepare measurements
        z_mag = data.mag_meas[k] if not np.isnan(data.mag_meas[k, 0]) else None
        z_sun = data.sun_meas[k] if not np.isnan(data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(data.st_meas[k]) if not np.isnan(data.st_meas[k, 0]) else None

        # Step redundant estimator
        x_eskf, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_eskf,
            t=data.t[k],
            jd=data.jd[k],
            omega_meas=data.omega_meas[k],
            dt=dt,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=data.b_eci[k],
            s_n=data.s_eci[k],
        )

        # Compute errors
        err_eskf = compute_attitude_error(x_eskf.nom.ori, q_true)
        err_smoother = compute_attitude_error(smoother_state.ori, q_true)

        # Get output error based on primary
        output_state = redundant.get_primary_state(x_eskf)
        err_output = compute_attitude_error(output_state.ori, q_true)

        errors_eskf.append(err_eskf)
        errors_smoother.append(err_smoother)
        errors_output.append(err_output)
        primary_modes.append(primary)

    return np.array(errors_eskf), np.array(errors_smoother), np.array(errors_output), primary_modes


def plot_convergence(data, errors_eskf, errors_smoother, errors_redundant, output_dir: Path):
    """Plot convergence comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    t = data.t

    ax.semilogy(t, errors_eskf, 'C0-', label='ESKF', alpha=0.9)
    ax.semilogy(t, errors_smoother, 'C1-', label='Smoother', alpha=0.9)
    ax.semilogy(t, errors_redundant, 'C2--', label='Redundant (output)', alpha=0.9)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Convergence from 10° Initial Error (Nominal Conditions)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([1e-3, 20])

    # Add reference lines
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='1°')
    ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5, label='0.1°')
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5, label='0.01°')

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'convergence_comparison.{ext}')
    plt.close(fig)
    print(f"Saved convergence_comparison.pdf/png")


def plot_convergence_detailed(data, errors_eskf, errors_smoother,
                               red_eskf, red_smoother, red_output,
                               output_dir: Path):
    """Plot detailed convergence with separate panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = data.t

    # Top-left: All methods comparison
    ax = axes[0, 0]
    ax.semilogy(t, errors_eskf, 'C0-', label='ESKF', linewidth=2)
    ax.semilogy(t, errors_smoother, 'C1-', label='Smoother', linewidth=2)
    ax.semilogy(t, red_output, 'C2--', label='Redundant', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Convergence Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-3, 20])

    # Top-right: Early convergence (first 60s)
    ax = axes[0, 1]
    mask = t <= 60
    ax.semilogy(t[mask], errors_eskf[mask], 'C0-', label='ESKF', linewidth=2)
    ax.semilogy(t[mask], errors_smoother[mask], 'C1-', label='Smoother', linewidth=2)
    ax.semilogy(t[mask], red_output[mask], 'C2--', label='Redundant', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Early Convergence (0-60s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-2, 15])

    # Bottom-left: Steady-state (last 100s)
    ax = axes[1, 0]
    mask = t >= 200
    ax.plot(t[mask], errors_eskf[mask], 'C0-', label='ESKF', linewidth=1.5, alpha=0.8)
    ax.plot(t[mask], errors_smoother[mask], 'C1-', label='Smoother', linewidth=1.5, alpha=0.8)
    ax.plot(t[mask], red_output[mask], 'C2--', label='Redundant', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Steady-State Performance (200-300s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Redundant estimator internal comparison
    ax = axes[1, 1]
    ax.semilogy(t, red_eskf, 'C0-', label='Redundant ESKF', alpha=0.7)
    ax.semilogy(t, red_smoother, 'C1-', label='Redundant Smoother', alpha=0.7)
    ax.semilogy(t, red_output, 'C2-', label='Redundant Output', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Redundant Estimator Internal States')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-3, 20])

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'convergence_detailed.{ext}')
    plt.close(fig)
    print(f"Saved convergence_detailed.pdf/png")


def plot_steady_state_stats(data, errors_eskf, errors_smoother, red_output, output_dir: Path):
    """Plot steady-state error statistics."""
    t = data.t

    # Use last 100 seconds for steady-state
    mask = t >= 200

    eskf_ss = errors_eskf[mask]
    smoother_ss = errors_smoother[mask]
    redundant_ss = red_output[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plot
    ax = axes[0]
    data_box = [eskf_ss, smoother_ss, redundant_ss]
    bp = ax.boxplot(data_box, labels=['ESKF', 'Smoother', 'Redundant'], patch_artist=True)
    colors = ['C0', 'C1', 'C2']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Attitude error [deg]')
    ax.set_title('Steady-State Error Distribution (t > 200s)')
    ax.grid(True, alpha=0.3)

    # Right: Statistics table as text
    ax = axes[1]
    ax.axis('off')

    stats_text = "Steady-State Statistics (t > 200s)\n"
    stats_text += "=" * 45 + "\n\n"

    for name, errs in [('ESKF', eskf_ss), ('Smoother', smoother_ss), ('Redundant', redundant_ss)]:
        stats_text += f"{name}:\n"
        stats_text += f"  Mean:   {np.mean(errs):.4f}°\n"
        stats_text += f"  Std:    {np.std(errs):.4f}°\n"
        stats_text += f"  RMS:    {np.sqrt(np.mean(errs**2)):.4f}°\n"
        stats_text += f"  Max:    {np.max(errs):.4f}°\n"
        stats_text += f"  95%:    {np.percentile(errs, 95):.4f}°\n\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'convergence_statistics.{ext}')
    plt.close(fig)
    print(f"Saved convergence_statistics.pdf/png")

    # Print to console
    print("\n" + "=" * 50)
    print("Steady-State Performance (t > 200s)")
    print("=" * 50)
    print(f"{'Method':<15} {'Mean':>10} {'Std':>10} {'RMS':>10} {'Max':>10}")
    print("-" * 50)
    print(f"{'ESKF':<15} {np.mean(eskf_ss):>10.4f} {np.std(eskf_ss):>10.4f} {np.sqrt(np.mean(eskf_ss**2)):>10.4f} {np.max(eskf_ss):>10.4f}")
    print(f"{'Smoother':<15} {np.mean(smoother_ss):>10.4f} {np.std(smoother_ss):>10.4f} {np.sqrt(np.mean(smoother_ss**2)):>10.4f} {np.max(smoother_ss):>10.4f}")
    print(f"{'Redundant':<15} {np.mean(redundant_ss):>10.4f} {np.std(redundant_ss):>10.4f} {np.sqrt(np.mean(redundant_ss**2)):>10.4f} {np.max(redundant_ss):>10.4f}")


def main():
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    config_path = "configs/config_baseline_short.yaml"

    print("Loading simulation data...")
    data = load_nominal_data()

    # Create initial quaternion with 10 degree error
    q_true_init = Quaternion.from_array(data.q_true[0])
    q_init = create_initial_error(q_true_init, error_deg=10.0, axis=np.array([1, 1, 1]))

    initial_error = compute_attitude_error(q_init, q_true_init)
    print(f"Initial attitude error: {initial_error:.2f} degrees")

    print("\nRunning ESKF...")
    errors_eskf, cov_eskf = run_eskf(data, q_init, config_path)

    print("Running Smoother...")
    errors_smoother = run_smoother(data, q_init, config_path)

    print("Running Redundant Estimator...")
    red_eskf, red_smoother, red_output, modes = run_redundant(data, q_init, config_path)

    print("\nGenerating plots...")
    plot_convergence(data, errors_eskf, errors_smoother, red_output, output_dir)
    plot_convergence_detailed(data, errors_eskf, errors_smoother,
                               red_eskf, red_smoother, red_output, output_dir)
    plot_steady_state_stats(data, errors_eskf, errors_smoother, red_output, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
