#!/usr/bin/env python3
"""
Complete script to:
1. Run ESKF on a simulation
2. Save results to database
3. Generate enhanced plots (log scale, split view, NEES/ANEES)
"""

import numpy as np
import copy
from data.db import SimulationDatabase
from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from plotting.attitude_plotter import AttitudePlotter


def run_eskf_and_save(sim_run_id: int, db_path: str = "simulations.db",
                       config_path: str = "configs/config_baseline_short.yaml") -> int:
    """
    Run ESKF on a simulation and save results to database.

    Args:
        sim_run_id: ID of simulation run to estimate
        db_path: Path to database
        config_path: Path to configuration file

    Returns:
        est_run_id: ID of saved estimation run
    """
    print("=" * 70)
    print("RUNNING ESKF")
    print("=" * 70)

    # Load simulation data
    db = SimulationDatabase(db_path)
    sim = db.load_run(sim_run_id)

    print(f"\nLoaded simulation run {sim_run_id}")
    print(f"Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

    # Initialize ESKF
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path=config_path)

    # Initial state (with some error)
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()
    b0_est = np.zeros(3)

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    # Storage for all states
    states = [copy.deepcopy(x_est)]
    rejected_count = {"mag": 0, "sun": 0, "star": 0}
    accepted_count = {"mag": 0, "sun": 0, "star": 0}

    print("\nRunning filter...")

    # Run filter
    for k in range(1, len(sim.t)):
        omega_k = sim.omega_meas[k] if not np.any(np.isnan(sim.omega_meas[k])) else sim.omega_meas[k-1]
        dt_k = sim.t[k] - sim.t[k-1]

        # Predict
        x_est = eskf.predict(x_est, omega_k, dt_k)

        # Environment
        B_n = sim.b_eci[k]
        s_n = sim.s_eci[k]

        # Try magnetometer update
        mag_meas = sim.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=B_n
                )
                accepted_count["mag"] += 1
            except ValueError:
                rejected_count["mag"] += 1

        # Try sun sensor update
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=s_n
                )
                accepted_count["sun"] += 1
            except ValueError:
                rejected_count["sun"] += 1

        # Try star tracker update
        st_meas = sim.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(
                    x_est=x_est,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER
                )
                accepted_count["star"] += 1
            except ValueError:
                rejected_count["star"] += 1

        states.append(copy.deepcopy(x_est))

        if (k + 1) % 1000 == 0:
            print(f"  Progress: {k+1}/{len(sim.t)} samples")

    print("\nFilter complete!")
    print(f"\nMeasurement Statistics:")
    print(f"  Magnetometer: {accepted_count['mag']} accepted, {rejected_count['mag']} rejected")
    print(f"  Sun sensor:   {accepted_count['sun']} accepted, {rejected_count['sun']} rejected")
    print(f"  Star tracker: {accepted_count['star']} accepted, {rejected_count['star']} rejected")

    # Compute final error
    q_true_final = Quaternion.from_array(sim.q_true[-1])
    q_est_final = states[-1].nom.ori
    q_err = q_true_final.conjugate().multiply(q_est_final)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    print(f"\nFinal attitude error: {np.rad2deg(angle_err):.4f}°")

    # Save to database
    print("\nSaving results to database...")
    est_run_id = db.insert_eskf_run(
        sim_run_id=sim_run_id,
        t=sim.t,
        jd=sim.jd,
        states=states,
        name="eskf_all_sensors"
    )
    print(f"Saved as estimation run {est_run_id}")

    return est_run_id


def generate_plots(sim_run_id: int, est_run_id: int, db_path: str = "simulations.db"):
    """
    Generate all enhanced plots for the estimation results.

    Args:
        sim_run_id: ID of simulation run
        est_run_id: ID of estimation run
        db_path: Path to database
    """
    print("\n" + "=" * 70)
    print("GENERATING ENHANCED PLOTS")
    print("=" * 70)

    plotter = AttitudePlotter(db_path=db_path)

    # 1. Attitude error with log scale
    print("\n1. Attitude error (log scale)...")
    plotter.plot_attitude_error(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        log_scale=True,
        fname="attitude_error_log"
    )

    # 2. Attitude error with split view
    print("\n2. Attitude error (split view: convergence + steady-state)...")
    plotter.plot_attitude_error(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        split_view=True,
        split_time=5.0,
        fname="attitude_error_split"
    )

    # 3. NEES/ANEES with chi-squared confidence bounds
    print("\n3. NEES/ANEES analysis with χ² confidence bounds...")
    plotter.plot_nees(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        alpha=0.05,  # 95% confidence
        fname="nees_analysis"
    )

    # 4. Euler angle comparison (optional)
    print("\n4. Euler angle comparison...")
    plotter.plot_euler_comparison(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        fname="euler_comparison"
    )

    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - attitude_error_log.pdf/png      (log scale)")
    print("  - attitude_error_split.pdf/png    (convergence + steady-state)")
    print("  - nees_analysis.pdf/png           (filter consistency check)")
    print("  - euler_comparison.pdf/png        (roll, pitch, yaw)")


def main():
    """Run complete pipeline: ESKF estimation + enhanced plotting."""
    # Configuration
    sim_run_id = 1  # Change this to your simulation run ID
    db_path = "simulations.db"
    config_path = "configs/config_baseline_short.yaml"

    print("ESKF ESTIMATION AND ENHANCED PLOTTING PIPELINE")
    print("=" * 70)
    print(f"Simulation run ID: {sim_run_id}")
    print(f"Database: {db_path}")
    print(f"Config: {config_path}")

    # Step 1: Run ESKF and save results
    est_run_id = run_eskf_and_save(
        sim_run_id=sim_run_id,
        db_path=db_path,
        config_path=config_path
    )

    # Step 2: Generate all enhanced plots
    generate_plots(
        sim_run_id=sim_run_id,
        est_run_id=est_run_id,
        db_path=db_path
    )

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
