#!/usr/bin/env python3
"""
Test GTSAM Factor Graph Optimization using real simulation data from database.

This script:
1. Loads simulation data from the database
2. Runs GTSAM-based factor graph optimization
3. Computes attitude errors against ground truth
4. Analyzes performance metrics
5. Creates visualizations (optional)
"""

import numpy as np
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from data.db import SimulationDatabase
from data.classes import SimulationResult
from estimation.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.quaternion import Quaternion
from utilities.states import NominalState
from environment.environment import OrbitEnvironmentModel
from plotting.comparison_plotter import ComparisonPlotter, EstimationResults


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
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

    def add(self, t: float, att_err_deg: float, bias_err: np.ndarray):
        self.timestamps.append(t)
        self.attitude_errors_deg.append(att_err_deg)
        self.bias_errors.append(bias_err)

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

        print(f"\nData Statistics:")
        print(f"  Total samples: {len(self.attitude_errors_deg)}")
        print(f"  Time span: {self.timestamps[-1] - self.timestamps[0]:.2f} seconds")
        print(f"  Duration: {(self.timestamps[-1] - self.timestamps[0])/60:.2f} minutes")


def run_gtsam_on_simulation(
    db: SimulationDatabase,
    config_path: str,
    sim_run_id: int,
    window_size: int = 200,
    max_samples: Optional[int] = None,
    skip_samples: int = 1,
    save_results: bool = False,
    est_name: str = "gtsam_fgo_test",
    scenario_name: str = "Unknown"
) -> tuple[PerformanceMetrics, Optional[EstimationResults]]:
    """
    Run GTSAM factor graph optimization on simulation data.

    Args:
        db: Database connection
        sim_run_id: Simulation run ID to process
        window_size: Size of sliding window
        max_samples: Maximum number of samples to process (None = all)
        skip_samples: Process every N-th sample (for speed)
        save_results: Whether to save results back to database
        est_name: Name for estimation run in database

    Returns:
        PerformanceMetrics object with results
    """
    print("\n" + "="*60)
    print("GTSAM FACTOR GRAPH OPTIMIZATION TEST")
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

    if skip_samples > 1:
        print(f"  Subsampling: every {skip_samples} sample(s)")

    # Initialize
    fgo = GtsamFGO(config_path=config_path)
    env = OrbitEnvironmentModel()
    window = SlidingWindow(max_len=window_size)
    metrics = PerformanceMetrics()

    # Storage for all estimated states
    estimated_states: List[NominalState] = []
    estimated_times: List[float] = []
    estimated_jds: List[float] = []

    # Initial guess (use true state for first sample)
    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = sim.b_g_true[0].copy()
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    print(f"\nProcessing with window size {window_size}...")
    print("Progress: ", end="", flush=True)

    progress_step = max(1, N // 20)  # Show 20 progress markers

    # Main loop
    for k in range(0, N, skip_samples):
        # Progress indicator
        if k % progress_step == 0:
            print(".", end="", flush=True)

        # Get measurements
        omega_k = sim.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim.omega_meas[max(0, k-1)]

        # Propagate nominal state
        if k > 0:
            dt = sim.t[k] - sim.t[k-skip_samples]
        else:
            dt = 0.02

        q_pred = x_nom_prev.ori.propagate(omega_k - x_nom_prev.gyro_bias, dt)
        x_nom = NominalState(ori=q_pred, gyro_bias=x_nom_prev.gyro_bias)

        # Prepare measurements (handle NaN)
        z_mag = None if np.any(np.isnan(sim.mag_meas[k])) else sim.mag_meas[k]
        z_sun = None if np.any(np.isnan(sim.sun_meas[k])) else sim.sun_meas[k]
        z_st = None
        if not np.any(np.isnan(sim.st_meas[k])):
            z_st = Quaternion.from_array(sim.st_meas[k])

        # Create window sample
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

        # Optimize when window is full
        if window.ready:
            window_samples = list(window.samples)

            try:
                optimized_states = fgo.optimize_window(window_samples, env)

                # Store results and compute errors for the window
                for i, opt_state in enumerate(optimized_states):
                    global_k = k - len(window_samples) + 1 + i
                    if global_k < 0 or global_k >= N:
                        continue

                    # Get ground truth
                    q_true = Quaternion.from_array(sim.q_true[global_k])
                    b_true = sim.b_g_true[global_k]

                    # Compute errors
                    att_err = compute_attitude_error_deg(q_true, opt_state.ori)
                    bias_err = compute_bias_error(b_true, opt_state.gyro_bias)

                    # Store
                    metrics.add(sim.t[global_k], att_err, bias_err)

                    # Only keep states we haven't stored yet
                    if len(estimated_states) <= global_k:
                        estimated_states.append(opt_state)
                        estimated_times.append(sim.t[global_k])
                        estimated_jds.append(sim.jd[global_k])

                # Slide window (keep second half)
                window.clear_half()

            except Exception as e:
                print(f"\nOptimization failed at k={k}: {e}")
                # Continue with next window
                window.samples.clear()

    print(" Done!")

    # Handle remaining samples in window
    if len(window.samples) > 0:
        print(f"\nProcessing final window with {len(window.samples)} samples...")
        try:
            window_samples = list(window.samples)
            optimized_states = fgo.optimize_window(window_samples, env)

            for i, opt_state in enumerate(optimized_states):
                global_k = N - len(window_samples) + i
                if global_k < N:
                    q_true = Quaternion.from_array(sim.q_true[global_k])
                    b_true = sim.b_g_true[global_k]

                    att_err = compute_attitude_error_deg(q_true, opt_state.ori)
                    bias_err = compute_bias_error(b_true, opt_state.gyro_bias)

                    metrics.add(sim.t[global_k], att_err, bias_err)

                    if len(estimated_states) <= global_k:
                        estimated_states.append(opt_state)
                        estimated_times.append(sim.t[global_k])
                        estimated_jds.append(sim.jd[global_k])
        except Exception as e:
            print(f"Final optimization failed: {e}")

    # Save results to database if requested
    if save_results and len(estimated_states) > 0:
        print(f"\nSaving {len(estimated_states)} estimated states to database...")
        try:
            t_arr = np.array(estimated_times)
            jd_arr = np.array(estimated_jds)
            est_run_id = db.insert_fgo_run(
                sim_run_id=sim_run_id,
                t=t_arr,
                jd=jd_arr,
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
        # Convert metrics to arrays for EstimationResults
        att_errors = np.array(metrics.attitude_errors_deg)
        bias_errors = np.array(metrics.bias_errors)
        bias_error_norms = np.linalg.norm(bias_errors, axis=1)
        timestamps = np.array(metrics.timestamps)

        # We don't have sigma bounds from FGO, so use zeros
        sigma_att = np.zeros_like(att_errors)
        sigma_bias = np.zeros_like(bias_error_norms)

        estimation_results = EstimationResults(
            t=timestamps,
            attitude_error=att_errors,
            bias_error=bias_error_norms,
            sigma_attitude=sigma_att,
            sigma_bias=sigma_bias,
            method_name="FGO",
            scenario_name=scenario_name
        )

    return metrics, estimation_results


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test GTSAM Factor Graph Optimization")
    parser.add_argument("--plot", action="store_true", help="Generate plots of results")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GTSAM Factor Graph - Database Test Suite")
    print("="*60)

    # Create output directory if plotting is enabled
    output_dir = None
    plotter = None
    if args.plot:
        output_dir = Path("output/fgo")
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
            "name": "Rapid tumbling",
            "path": "configs/config_baseline_short.yaml",
            "sim_run_id": 1,
            "window_size": 400,
            "max_samples": None,
            "skip_samples": 1,
            "save_results": False,
        },
        # {
        #     "name": "Baseline Test (6000 samples, window=300)",
        #     "path": "configs/config_baseline.yaml",
        #     "sim_run_id": 2,
        #     "window_size": 300,
        #     "max_samples": None,
        #     "skip_samples": 1,
        #     "save_results": False,
        # },
        # {
        #     "name": "Spikes Test (6000 samples, window=300)",
        #     "path": "configs/config_spikes.yaml",
        #     "sim_run_id": 3,
        #     "window_size": 300,
        #     "max_samples": None,
        #     "skip_samples": 2,
        #     "save_results": False,
        # },
        # {
        #     "name": "Anomalies Test (9000 samples, window=300)",
        #     "path": "configs/config_anomalies.yaml",
        #     "sim_run_id": 4,
        #     "window_size": 300,
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
            metrics, est_results = run_gtsam_on_simulation(
                db=db,
                config_path=config["path"],
                sim_run_id=config["sim_run_id"],
                window_size=config["window_size"],
                max_samples=config.get("max_samples"),
                skip_samples=config.get("skip_samples", 1),
                save_results=config.get("save_results", False),
                est_name=f"gtsam_test_{i+1}",
                scenario_name=config["name"]
            )
            results[config["name"]] = {
                "success": True,
                "metrics": metrics
            }

            # Generate plot if enabled and results are available
            if args.plot and plotter and est_results:
                scenario_filename = config["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
                plot_path = output_dir / f"{scenario_filename}_fgo.png"
                plotter.plot_attitude_comparison(
                    [est_results],
                    filename=plot_path,
                    title=f"{config['name']}: Factor Graph Optimization"
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
