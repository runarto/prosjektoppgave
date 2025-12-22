#!/usr/bin/env python3
"""
Complete script to:
1. Run Factor Graph Optimization (FGO) on a simulation
2. Save results to database
3. Generate plots (attitude error split view, bias error, Euler comparison)
"""

import numpy as np
from data.db import SimulationDatabase
from estimation.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.quaternion import Quaternion
from utilities.states import NominalState
from environment.environment import OrbitEnvironmentModel
from plotting.attitude_plotter import AttitudePlotter


def run_fgo_and_save(sim_run_id: int, db_path: str = "simulations.db",
                     config_path: str = "configs/config_baseline_short.yaml",
                     window_size: int = 500,
                     optimize_stride: int = 500) -> int:
    """
    Run FGO on a simulation and save results to database.

    Args:
        sim_run_id: ID of simulation run to estimate
        db_path: Path to database
        config_path: Path to configuration file
        window_size: Number of samples per optimization window
        optimize_stride: Optimize every N samples

    Returns:
        est_run_id: ID of saved estimation run
    """
    print("=" * 70)
    print("RUNNING FACTOR GRAPH OPTIMIZATION (FGO)")
    print("=" * 70)

    # Load simulation data
    db = SimulationDatabase(db_path)
    sim = db.load_run(sim_run_id)

    print(f"\nLoaded simulation run {sim_run_id}")
    print(f"Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

    # Initialize FGO components
    fgo = GtsamFGO(config_path=config_path)
    env = OrbitEnvironmentModel()
    window = SlidingWindow(max_len=window_size)

    print(f"\nFGO configuration:")
    print(f"  Window size: {window_size} samples")
    print(f"  Optimization stride: {optimize_stride} samples")
    print(f"  Robust kernel: {fgo.robust_kernel} (k={fgo.robust_param})")

    # Initial nominal state (use true initial attitude, zero bias)
    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = np.zeros(3)
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    # Storage for all estimated states and corresponding times
    est_states = []
    est_times = []
    est_jds = []

    n_mag_used = 0
    n_sun_used = 0
    n_st_used = 0
    n_optimizations = 0

    print(f"\nRunning FGO on {len(sim.t)} samples...")

    for k in range(len(sim.t)):
        if k == 0:
            dt = 0.02  # Assume 50 Hz
        else:
            dt = sim.t[k] - sim.t[k - 1]

        omega_k = sim.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim.omega_meas[k - 1] if k > 0 else np.zeros(3)

        # Propagate nominal state
        q_pred = x_nom_prev.ori.propagate(omega_k - x_nom_prev.gyro_bias, dt)
        x_nom = NominalState(ori=q_pred, gyro_bias=x_nom_prev.gyro_bias)

        # Build WindowSample for this step
        z_mag = None if np.any(np.isnan(sim.mag_meas[k])) else sim.mag_meas[k]
        z_sun = None if np.any(np.isnan(sim.sun_meas[k])) else sim.sun_meas[k]
        z_st = None
        if not np.any(np.isnan(sim.st_meas[k])):
            z_st = Quaternion.from_array(sim.st_meas[k])

        if z_mag is not None:
            n_mag_used += 1
        if z_sun is not None:
            n_sun_used += 1
        if z_st is not None:
            n_st_used += 1

        sample = WindowSample(
            t=float(sim.t[k]),
            jd=float(sim.jd[k]),
            x_nom=x_nom,
            omega_meas=omega_k,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
        )

        window.add(sample)
        x_nom_prev = x_nom

        # When window is full AND we've reached the optimization stride, run smoothing
        if window.ready and (k % optimize_stride == 0 or k == window_size - 1):
            try:
                window_samples = list(window.samples)
                window_states = fgo.optimize_window(window_samples, env)
                n_optimizations += 1

                # Store all states from this optimization batch
                for i, state in enumerate(window_states):
                    global_k = k - len(window_states) + 1 + i

                    # Skip if we've already stored this state
                    if global_k < len(est_states):
                        continue

                    est_states.append(state)
                    est_times.append(sim.t[global_k])
                    est_jds.append(sim.jd[global_k])

                # Update x_nom_prev with the last optimized state
                if window_states:
                    x_nom_prev = window_states[-1]

                # Progress reporting
                if n_optimizations % 10 == 0 or n_optimizations <= 3:
                    print(f"  Optimization {n_optimizations}: processed up to t={sim.t[k]:.1f}s, "
                          f"total states: {len(est_states)}")

                window.samples.clear()  # Clear window after optimization

            except Exception as e:
                print(f"\nOptimization failed at step {k}: {e}")
                import traceback
                traceback.print_exc()
                break

    print("\nFGO complete!")
    print(f"\nMeasurement Statistics:")
    print(f"  Magnetometer: {n_mag_used} used")
    print(f"  Sun sensor:   {n_sun_used} used")
    print(f"  Star tracker: {n_st_used} used")
    print(f"  Total optimizations: {n_optimizations}")

    # Compute final error - right-multiply: q_err = q_true ⊗ q_est^{-1}
    if len(est_states) > 0:
        # Find matching true state for final estimate
        final_idx = len(est_states) - 1
        q_true_final = Quaternion.from_array(sim.q_true[final_idx])
        q_est_final = est_states[-1].ori
        q_err = q_true_final.multiply(q_est_final.conjugate())
        angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
        print(f"\nFinal attitude error: {np.rad2deg(angle_err):.4f} deg")

    # Save to database
    print("\nSaving results to database...")
    est_run_id = db.insert_fgo_run(
        sim_run_id=sim_run_id,
        t=np.array(est_times),
        jd=np.array(est_jds),
        states=est_states,
        name="fgo_all_sensors"
    )
    print(f"Saved as estimation run {est_run_id}")

    return est_run_id


def generate_plots(sim_run_id: int, est_run_id: int, db_path: str = "simulations.db",
                   output_prefix: str = "fgo"):
    """
    Generate plots for the FGO estimation results.

    Args:
        sim_run_id: ID of simulation run
        est_run_id: ID of estimation run
        db_path: Path to database
        output_prefix: Prefix for output filenames
    """
    print("\n" + "=" * 70)
    print("GENERATING FGO PLOTS")
    print("=" * 70)

    plotter = AttitudePlotter(db_path=db_path)

    # 1. Attitude error with split view (first 5 seconds + steady-state)
    print("\n1. Attitude error (split view: convergence + steady-state)...")
    plotter.plot_attitude_error(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        split_view=True,
        split_time=5.0,
        fname=f"{output_prefix}_attitude_error_split"
    )

    # 2. Attitude error with log scale
    print("\n2. Attitude error (log scale)...")
    plotter.plot_attitude_error(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        log_scale=True,
        fname=f"{output_prefix}_attitude_error_log"
    )

    # 3. Euler angle comparison
    print("\n3. Euler angle comparison...")
    plotter.plot_euler_comparison(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        fname=f"{output_prefix}_euler_comparison"
    )

    # 4. Bias error (steady-state, no covariance bounds for FGO)
    print("\n4. Bias estimation error...")
    plot_fgo_bias_error(
        db_path=db_path,
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        fname=f"{output_prefix}_bias_error",
        steady_state_start=20.0
    )

    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {output_prefix}_attitude_error_split.pdf/png  (convergence + steady-state)")
    print(f"  - {output_prefix}_attitude_error_log.pdf/png    (log scale)")
    print(f"  - {output_prefix}_euler_comparison.pdf/png      (roll, pitch, yaw)")
    print(f"  - {output_prefix}_bias_error.pdf/png            (gyro bias estimation)")


def plot_fgo_bias_error(db_path: str, sim_run_id: int, est_run_id: int,
                        fname: str | None = None, steady_state_start: float = 20.0):
    """
    Plot gyro bias estimation error for FGO (without covariance bounds).

    FGO doesn't provide covariance estimates, so this plot shows only the
    bias estimation error without 3-sigma bounds.

    Args:
        db_path: Path to database
        sim_run_id: Simulation run ID
        est_run_id: Estimation run ID
        fname: Optional filename prefix for saving plots
        steady_state_start: Time (s) after which to show (default: 20s)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.1)

    db = SimulationDatabase(db_path)
    sim = db.load_run(sim_run_id)
    est = db.load_estimated_states(est_run_id)

    t = est.t
    b_true = sim.b_g_true[:len(t)]
    b_est = est.bg_est

    # Find steady-state start index
    ss_idx = np.searchsorted(t, steady_state_start)
    t = t[ss_idx:]
    b_true = b_true[ss_idx:]
    b_est = b_est[ss_idx:]

    # Bias error
    b_err = b_est - b_true  # (N, 3) in rad/s

    # Convert to deg/s
    b_err_deg_s = np.rad2deg(b_err)

    axis_labels = ['X', 'Y', 'Z']
    colors = ['C0', 'C1', 'C2']

    # Single plot with all 3 axes
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(3):
        ax.plot(t, b_err_deg_s[:, i], colors[i], linewidth=1,
                label=f'{axis_labels[i]}-axis')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Bias Error [deg/s]')
    ax.set_title('FGO Gyro Bias Estimation Error - Steady State')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if fname is not None:
        fig.savefig(fname + ".pdf", bbox_inches="tight")
        fig.savefig(fname + ".png", dpi=400, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    # Print summary
    print(f"\nBias Estimation Summary (FGO):")
    print(f"  Final bias error (deg/s): [{b_err_deg_s[-1, 0]:.6f}, {b_err_deg_s[-1, 1]:.6f}, {b_err_deg_s[-1, 2]:.6f}]")
    print(f"  RMS bias error (deg/s):   [{np.sqrt(np.mean(b_err_deg_s[:, 0]**2)):.6f}, "
          f"{np.sqrt(np.mean(b_err_deg_s[:, 1]**2)):.6f}, {np.sqrt(np.mean(b_err_deg_s[:, 2]**2)):.6f}]")


def main():
    """Run complete pipeline: FGO estimation + plotting."""
    # Configuration
    sim_run_id = 1  # Use run with STIM300 parameters
    db_path = "simulations.db"
    config_path = "configs/config_baseline_short.yaml"

    # FGO parameters
    window_size = 500      # 10 seconds at 50 Hz
    optimize_stride = 500  # Optimize every 10 seconds

    print("FGO ESTIMATION AND PLOTTING PIPELINE")
    print("=" * 70)
    print(f"Simulation run ID: {sim_run_id}")
    print(f"Database: {db_path}")
    print(f"Config: {config_path}")

    # Step 1: Run FGO and save results
    est_run_id = run_fgo_and_save(
        sim_run_id=sim_run_id,
        db_path=db_path,
        config_path=config_path,
        window_size=window_size,
        optimize_stride=optimize_stride
    )

    # Step 2: Generate all plots
    generate_plots(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        db_path=db_path,
        output_prefix="fgo"
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
