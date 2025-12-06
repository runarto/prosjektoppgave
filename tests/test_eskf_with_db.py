#!/usr/bin/env python3
"""
Test ESKF (Error-State Kalman Filter) using real simulation data from database.

This script:
1. Loads simulation data from the database
2. Runs ESKF with predict-update loop
3. Computes attitude and bias errors against ground truth
4. Analyzes performance metrics
5. Compares with factor graph results (if available)
"""

import numpy as np
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import copy

from data.db import SimulationDatabase
from data.classes import SimulationResult
from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel
from plotting.comparison_plotter import ComparisonPlotter, EstimationResults


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    # Ensure same hemisphere
    q_true_arr = q_true.as_array()
    q_est_arr = q_est.as_array()
    if np.dot(q_true_arr, q_est_arr) < 0:
        q_est_arr = -q_est_arr
        q_est = Quaternion.from_array(q_est_arr)

    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def compute_bias_error(b_true: np.ndarray, b_est: np.ndarray) -> np.ndarray:
    """Compute gyro bias error."""
    return b_est - b_true


class PerformanceMetrics:
    """Container for performance metrics."""
    def __init__(self):
        self.attitude_errors_deg: List[float] = []
        self.bias_errors: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.attitude_sigmas: List[np.ndarray] = []  # 3-sigma bounds
        self.bias_sigmas: List[np.ndarray] = []

    def add(self, t: float, att_err_deg: float, bias_err: np.ndarray,
            att_sigma: Optional[np.ndarray] = None,
            bias_sigma: Optional[np.ndarray] = None):
        self.timestamps.append(t)
        self.attitude_errors_deg.append(att_err_deg)
        self.bias_errors.append(bias_err)
        if att_sigma is not None:
            self.attitude_sigmas.append(att_sigma)
        if bias_sigma is not None:
            self.bias_sigmas.append(bias_sigma)

    def print_summary(self):
        """Print summary statistics."""
        if not self.attitude_errors_deg:
            print("No data to summarize")
            return

        att_errs = np.array(self.attitude_errors_deg)
        bias_errs = np.array(self.bias_errors)

        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        print(f"\nAttitude Errors:")
        print(f"  Mean:   {np.mean(att_errs):.4f} deg")
        print(f"  Median: {np.median(att_errs):.4f} deg")
        print(f"  Max:    {np.max(att_errs):.4f} deg")
        print(f"  Min:    {np.min(att_errs):.4f} deg")
        print(f"  Std:    {np.std(att_errs):.4f} deg")
        print(f"  RMS:    {np.sqrt(np.mean(att_errs**2)):.4f} deg")

        print(f"\nGyro Bias Errors (rad/s):")
        bias_rms = np.sqrt(np.mean(bias_errs**2, axis=0))
        print(f"  X-axis RMS: {bias_rms[0]:.6e} rad/s")
        print(f"  Y-axis RMS: {bias_rms[1]:.6e} rad/s")
        print(f"  Z-axis RMS: {bias_rms[2]:.6e} rad/s")
        print(f"  Magnitude RMS: {np.linalg.norm(bias_rms):.6e} rad/s")

        if self.attitude_sigmas:
            att_sigmas = np.array(self.attitude_sigmas)
            print(f"\nAttitude 3σ Bounds (final):")
            print(f"  X: {att_sigmas[-1,0]:.4f} deg")
            print(f"  Y: {att_sigmas[-1,1]:.4f} deg")
            print(f"  Z: {att_sigmas[-1,2]:.4f} deg")

        if self.bias_sigmas:
            bias_sigmas = np.array(self.bias_sigmas)
            print(f"\nBias 3σ Bounds (final, rad/s):")
            print(f"  X: {bias_sigmas[-1,0]:.6e}")
            print(f"  Y: {bias_sigmas[-1,1]:.6e}")
            print(f"  Z: {bias_sigmas[-1,2]:.6e}")

        print(f"\nData Statistics:")
        print(f"  Total samples: {len(self.attitude_errors_deg)}")
        print(f"  Time span: {self.timestamps[-1] - self.timestamps[0]:.2f} seconds")
        print(f"  Duration: {(self.timestamps[-1] - self.timestamps[0])/60:.2f} minutes")


def run_eskf_on_simulation(
    db: SimulationDatabase,
    config_path: str,
    sim_run_id: int,
    max_samples: Optional[int] = None,
    save_results: bool = False,
    est_name: str = "eskf_test",
    scenario_name: str = "Unknown"
) -> tuple[PerformanceMetrics, Optional[EstimationResults]]:
    """
    Run ESKF on simulation data.

    Args:
        db: Database connection
        sim_run_id: Simulation run ID to process
        max_samples: Maximum number of samples to process (None = all)
        save_results: Whether to save results back to database
        est_name: Name for estimation run in database

    Returns:
        PerformanceMetrics object with results
    """
    print("\n" + "="*60)
    print("ESKF TEST WITH DATABASE")
    print("="*60)

    # Load simulation data
    print(f"\nLoading simulation run {sim_run_id} from database...")
    sim: SimulationResult = db.load_run(sim_run_id)

    N = sim.t.shape[0]
    print(f"  Total samples: {N}")
    print(f"  Duration: {sim.t[-1]:.2f} seconds ({sim.t[-1]/60:.2f} minutes)")

    if max_samples is not None:
        N = min(N, max_samples)
        print(f"  Processing: {N} samples (limited)")

    # Initialize
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path=config_path)
    metrics = PerformanceMetrics()

    # Storage for estimated states
    estimated_states: List[EskfState] = []

    # Initial guess (use true state with some error)
    q0_true = Quaternion.from_array(sim.q_true[0])
    # Add small initial error
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()
    b0_est = np.zeros(3)  # Wrong initial bias
    s_n = sim.s_eci
    b_n = sim.b_eci
 
    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    estimated_states.append(copy.deepcopy(x_est))

    # Initial error
    att_err_0 = compute_attitude_error_deg(q0_true, q0_est)
    bias_err_0 = compute_bias_error(sim.b_g_true[0], b0_est)
    print(f"\nInitial errors:")
    print(f"  Attitude: {att_err_0:.4f} deg")
    print(f"  Bias magnitude: {np.linalg.norm(bias_err_0):.6e} rad/s")

    print(f"\nProcessing with ESKF...")
    print("Progress: ", end="", flush=True)

    progress_step = max(1, N // 20)

    # Main loop
    for k in range(1, N):
        if k % progress_step == 0:
            print(".", end="", flush=True)

        # Get measurements
        omega_k = sim.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim.omega_meas[k-1]

        dt_k = sim.t[k] - sim.t[k-1]

        # Predict
        x_est = eskf.predict(x_est, omega_k, dt_k)
        att_err = compute_attitude_error_deg(Quaternion.from_array(sim.q_true[k]), x_est.nom.ori)
        

        # Environment fields
        B_eci = b_n[k]
        s_eci = s_n[k]

        # Update with available measurements

        # Magnetometer
        mag_meas = sim.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=B_eci
                )
            except ValueError:
                pass  # Skip outliers rejected by chi-squared test

        # Sun vector
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=s_eci
                )
            except ValueError:
                pass  # Skip outliers

        # Star tracker
        st_meas = sim.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(
                    x_est=x_est,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER
                )
            except ValueError:
                pass  # Skip outliers

        # Compute errors
        q_true = Quaternion.from_array(sim.q_true[k])
        b_true = sim.b_g_true[k]

        att_err = compute_attitude_error_deg(q_true, x_est.nom.ori)
        bias_err = compute_bias_error(b_true, x_est.nom.gyro_bias)

        # Compute 3-sigma bounds
        att_sigma = 3 * np.sqrt(np.diag(x_est.err.cov)[:3]) * 180/np.pi  # deg
        bias_sigma = 3 * np.sqrt(np.diag(x_est.err.cov)[3:])  # rad/s

        # Store
        metrics.add(sim.t[k], att_err, bias_err, att_sigma, bias_sigma)
        estimated_states.append(copy.deepcopy(x_est))

    print(" Done!")

    # Save results to database if requested
    if save_results and len(estimated_states) > 0:
        print(f"\nSaving {len(estimated_states)} estimated states to database...")
        try:
            est_run_id = db.insert_eskf_run(
                sim_run_id=sim_run_id,
                t=sim.t[:len(estimated_states)],
                jd=sim.jd[:len(estimated_states)],
                states=estimated_states,
                name=est_name
            )
            print(f"Saved as estimation run {est_run_id}")
        except Exception as e:
            print(f"Failed to save to database: {e}")

    # Print summary
    metrics.print_summary()

    # Create EstimationResults for plotting (if we have data)
    estimation_results = None
    if len(metrics.attitude_errors_deg) > 0:
        att_errors = np.array(metrics.attitude_errors_deg)
        bias_errors = np.array(metrics.bias_errors)
        bias_error_norms = np.linalg.norm(bias_errors, axis=1)
        timestamps = np.array(metrics.timestamps)

        # Get sigma bounds from metrics
        if len(metrics.attitude_sigmas) > 0:
            att_sigmas = np.array(metrics.attitude_sigmas)
            # Convert 3-sigma bounds to 1-sigma by dividing by 3, then take mean across axes
            sigma_att = np.mean(att_sigmas, axis=1) / 3.0  # Mean of 3-sigma bounds
        else:
            sigma_att = np.zeros_like(att_errors)

        if len(metrics.bias_sigmas) > 0:
            bias_sigmas = np.array(metrics.bias_sigmas)
            sigma_bias = np.mean(bias_sigmas, axis=1) / 3.0  # Mean of 3-sigma bounds
        else:
            sigma_bias = np.zeros_like(bias_error_norms)

        estimation_results = EstimationResults(
            t=timestamps,
            attitude_error=att_errors,
            bias_error=bias_error_norms,
            sigma_attitude=sigma_att,
            sigma_bias=sigma_bias,
            method_name="ESKF",
            scenario_name=scenario_name
        )

    return metrics, estimation_results


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test ESKF on simulation data")
    parser.add_argument("--plot", action="store_true", help="Generate plots of results")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ESKF - Database Test Suite")
    print("="*60)

    # Create output directory if plotting is enabled
    output_dir = None
    plotter = None
    if args.plot:
        output_dir = Path("output/eskf")
        output_dir.mkdir(parents=True, exist_ok=True)
        plotter = ComparisonPlotter()
        print(f"\nPlotting enabled. Plots will be saved to: {output_dir}/")

    # Connect to database
    db_path = "simulations.db"
    print(f"\nConnecting to database: {db_path}")
    db = SimulationDatabase(path=db_path)

    # Check available runs
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs;")
    runs = cur.fetchall()

    if not runs:
        print("\nERROR: No simulation runs found in database!")
        print("Please generate simulation data first.")
        return 1

    print(f"\nAvailable simulation runs:")
    for run_id, name in runs:
        cur.execute("SELECT COUNT(*) FROM samples WHERE run_id=?;", (run_id,))
        count = cur.fetchone()[0]
        print(f"  [{run_id}] {name} ({count} samples)")

    # Test configuration
    TEST_CONFIGS = [
        {
            "name": "Baseline measurements",
            "path": "configs/config_baseline_short.yaml",
            "sim_run_id": 1,
            "max_samples": None,
            "skip_samples": 1,
            "save_results": False,
        },
        # {
        #     "name": "Rapid tumbling",
        #     "path": "configs/config_rapid_tumbling.yaml",
        #     "sim_run_id": 2,
        #     "max_samples": None,
        #     "skip_samples": 1,
        #     "save_results": False,
        # },
        # {
        #     "name": "Eclipse",
        #     "path": "configs/config_eclipse.yaml",
        #     "sim_run_id": 3,
        #     "max_samples": None,
        #     "skip_samples": 2,
        #     "save_results": False,
        # },
        # {
        #     "name": "Measurement spikes",
        #     "path": "configs/config_measurement_spikes.yaml",
        #     "sim_run_id": 4,
        #     "max_samples": None,
        #     "skip_samples": 2,
        #     "save_results": False,
        # },
    ]

    results = {}

    # Run tests
    for i, config in enumerate(TEST_CONFIGS):
        print("\n" + "="*60)
        print(f"Test {i+1}/{len(TEST_CONFIGS)}: {config['name']}")
        print("="*60)

        try:
            metrics, est_results = run_eskf_on_simulation(
                db=db,
                config_path=config["path"],
                sim_run_id=config["sim_run_id"],
                max_samples=config.get("max_samples"),
                save_results=config.get("save_results", False),
                est_name=f"eskf_test_{i+1}",
                scenario_name=config["name"]
            )
            results[config["name"]] = {
                "success": True,
                "metrics": metrics
            }

            # Generate plot if enabled and results are available
            if args.plot and plotter and est_results:
                scenario_filename = config["name"].lower().replace(" ", "_")
                plot_path = output_dir / f"{scenario_filename}_eskf.png"
                plotter.plot_attitude_comparison(
                    [est_results],
                    filename=plot_path,
                    title=f"{config['name']}: ESKF"
                )
                print(f"  Plot saved to: {plot_path}")
        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[config["name"]] = {
                "success": False,
                "error": str(e)
            }

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results.items():
        status = "PASSED" if result["success"] else "FAILED"
        print(f"\n{name}: {status}")

        if result["success"]:
            metrics = result["metrics"]
            if metrics.attitude_errors_deg:
                mean_err = np.mean(metrics.attitude_errors_deg)
                max_err = np.max(metrics.attitude_errors_deg)
                print(f"  Attitude error: {mean_err:.4f}° mean, {max_err:.4f}° max")

                bias_errs = np.array(metrics.bias_errors)
                bias_rms = np.sqrt(np.mean(bias_errs**2, axis=0))
                print(f"  Bias RMS: {np.linalg.norm(bias_rms):.6e} rad/s")
        else:
            print(f"  Error: {result['error']}")

    # Overall result
    all_passed = all(r["success"] for r in results.values())
    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
