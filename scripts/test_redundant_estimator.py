"""
Test script for the redundant estimator.

Runs ESKF + Fixed-Lag Smoother in parallel and compares results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import EskfState, NominalState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel


def run_redundant_test(
    sim_run_id: int = 1,
    db_path: str = "simulations.db",
    config_path: str = "configs/config_baseline_short.yaml",
    smoother_lag: float = 60.0,
):
    """Run redundant estimator test."""
    print("=" * 70)
    print("REDUNDANT ESTIMATOR TEST")
    print("=" * 70)

    # Load simulation
    db = SimulationDatabase(db_path)
    sim = db.load_run(sim_run_id)
    env = OrbitEnvironmentModel()

    N = len(sim.t)
    print(f"Simulation: {N} samples ({sim.t[-1]:.1f}s)")

    # Initialize with perturbed attitude (large initial error)
    q_true_0 = Quaternion.from_array(sim.q_true[0])
    perturb = Quaternion.from_avec(np.array([0.5, 0.5, 0.5]))  # ~50 deg error
    q_init = q_true_0 @ perturb

    # Initial covariance
    P0 = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6])

    # Create redundant estimator
    redundant = RedundantEstimator(
        P0=P0,
        config_path=config_path,
        smoother_lag=smoother_lag,
        use_robust=True,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
        agreement_threshold_deg=0.5,
        consecutive_agreements_to_recover=10,
    )

    # Initialize ESKF state
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy()),
    )

    # Storage
    times = []
    errors_eskf = []
    errors_smoother = []
    disagreements = []
    primaries = []

    print("\nRunning simulation...")
    print_interval = N // 10

    for k in range(1, N):
        t = sim.t[k]
        jd = sim.jd[k]
        dt = t - sim.t[k-1]
        omega = sim.omega_meas[k]

        # Get measurements
        z_mag = sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None
        z_sun = sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None
        z_st = Quaternion.from_array(sim.st_meas[k]) if not np.any(np.isnan(sim.st_meas[k])) else None

        # Get reference vectors
        B_n = None
        s_n = None
        if z_mag is not None:
            r_eci = env.get_r_eci(jd)
            B_n = env.get_B_eci(r_eci, jd)
        if z_sun is not None:
            s_n = env.get_sun_eci(jd)

        # Step redundant estimator
        x_eskf, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_eskf,
            t=t,
            jd=jd,
            omega_meas=omega,
            dt=dt,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=B_n,
            s_n=s_n,
        )

        # Compute errors
        q_true = Quaternion.from_array(sim.q_true[k])

        # ESKF error
        dq_eskf = q_true @ x_eskf.nom.ori.conjugate()
        err_eskf = np.rad2deg(2 * np.arccos(np.clip(abs(dq_eskf.mu), 0, 1)))

        # Smoother error
        dq_smoother = q_true @ smoother_state.ori.conjugate()
        err_smoother = np.rad2deg(2 * np.arccos(np.clip(abs(dq_smoother.mu), 0, 1)))

        times.append(t)
        errors_eskf.append(err_eskf)
        errors_smoother.append(err_smoother)
        disagreements.append(disagreement)
        primaries.append(primary)

        # Progress
        if k % print_interval == 0:
            print(f"  t={t:.1f}s: ESKF={err_eskf:.4f}°, Smoother={err_smoother:.4f}°, "
                  f"Disagree={disagreement:.4f}°, Primary={primary}")

    # Statistics
    times = np.array(times)
    errors_eskf = np.array(errors_eskf)
    errors_smoother = np.array(errors_smoother)
    disagreements = np.array(disagreements)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Skip first 30 seconds for steady-state analysis
    ss_idx = times > 30
    if np.any(ss_idx):
        print(f"\nSteady-state (t > 30s):")
        print(f"  ESKF:     Mean={np.mean(errors_eskf[ss_idx]):.6f}°, "
              f"Std={np.std(errors_eskf[ss_idx]):.6f}°")
        print(f"  Smoother: Mean={np.mean(errors_smoother[ss_idx]):.6f}°, "
              f"Std={np.std(errors_smoother[ss_idx]):.6f}°")
        print(f"  Disagreement: Mean={np.mean(disagreements[ss_idx]):.6f}°, "
              f"Max={np.max(disagreements[ss_idx]):.6f}°")

    # Get statistics
    stats = redundant.get_statistics()
    print(f"\nEstimator Statistics:")
    print(f"  Final primary: {stats['primary']}")
    print(f"  Switch events: {stats['total_switch_events']}")
    print(f"  Total disagreements > threshold: {stats['total_disagreements_over_threshold']}")
    print(f"  Max disagreement: {stats['max_disagreement_deg']:.4f}°")
    print(f"  Smoother keyframes: {stats['smoother']['total_keyframes_added']}")
    print(f"  Smoother marginalized: {stats['smoother']['total_keyframes_marginalized']}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Attitude errors
    ax = axes[0]
    ax.semilogy(times, errors_eskf, 'b-', alpha=0.7, linewidth=0.5, label='ESKF')
    ax.semilogy(times, errors_smoother, 'r-', alpha=0.7, linewidth=0.5, label='Smoother')
    ax.set_ylabel('Attitude Error [deg]')
    ax.set_title('Redundant Estimator: ESKF vs Fixed-Lag Smoother')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Disagreement
    ax = axes[1]
    ax.plot(times, disagreements, 'g-', linewidth=0.5)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Switch threshold')
    ax.axhline(y=0.5, color='b', linestyle='--', alpha=0.5, label='Agreement threshold')
    ax.set_ylabel('Disagreement [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Primary estimator
    ax = axes[2]
    primary_numeric = [1 if p == 'ESKF' else 0 for p in primaries]
    ax.fill_between(times, primary_numeric, alpha=0.3)
    ax.set_ylabel('Primary')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Smoother', 'ESKF'])
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('redundant_estimator_test.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to redundant_estimator_test.png")

    return times, errors_eskf, errors_smoother, disagreements


if __name__ == "__main__":
    run_redundant_test()
