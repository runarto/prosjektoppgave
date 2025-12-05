#!/usr/bin/env python3
"""
Run ESKF and FGO on all scenarios and generate comparison plots.

This script:
1. Runs ESKF on specified simulation runs
2. Runs FGO on the same runs
3. Generates publication-quality comparison plots
4. Saves results to output/ directory
"""

import numpy as np
import sys
from pathlib import Path

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel
from plotting.comparison_plotter import ComparisonPlotter, EstimationResults


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_eskf(sim, config_path: str) -> EstimationResults:
    """Run ESKF on simulation data."""
    print("\n  Running ESKF...", flush=True)

    # Initialize filter
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path=config_path)

    # Initial state
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()
    b0_est = np.zeros(3)

    nom = NominalState(ori=q0_est, gyro_bias=b0_est)
    err = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom, err=err)

    errors = []
    bias_errors = []
    sigma_att = []
    sigma_bias = []

    for k in range(1, len(sim.t)):
        omega_k = sim.omega_meas[k] if not np.any(np.isnan(sim.omega_meas[k])) else sim.omega_meas[k-1]
        dt_k = sim.t[k] - sim.t[k-1]

        # Predict
        x_est = eskf.predict(x_est, omega_k, dt_k)

        # Update with magnetometer
        if not np.any(np.isnan(sim.mag_meas[k])):
            try:
                x_est = eskf.update(x_est, sim.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim.b_eci[k])
            except ValueError:
                pass  # Skip outliers

        # Update with sun sensor (if available)
        if not np.any(np.isnan(sim.sun_meas[k])):
            try:
                x_est = eskf.update(x_est, sim.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim.s_eci[k])
            except ValueError:
                pass

        # Update with star tracker
        if not np.any(np.isnan(sim.st_meas[k])):
            try:
                q_meas = Quaternion.from_array(sim.st_meas[k])
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except ValueError:
                pass

        # Compute errors
        q_true_k = Quaternion.from_array(sim.q_true[k])
        att_err = compute_attitude_error_deg(q_true_k, x_est.nom.ori)
        errors.append(att_err)

        bias_err = np.linalg.norm(x_est.nom.gyro_bias - sim.b_g_true[k])
        bias_errors.append(bias_err)

        # Extract uncertainties
        P_diag = np.diag(x_est.err.cov)
        sigma_att.append(np.rad2deg(np.sqrt(np.mean(P_diag[0:3]))))
        sigma_bias.append(np.sqrt(np.mean(P_diag[3:6])))

    return EstimationResults(
        t=sim.t[1:],
        attitude_error=np.array(errors),
        bias_error=np.array(bias_errors),
        sigma_attitude=np.array(sigma_att),
        sigma_bias=np.array(sigma_bias),
        method_name="ESKF",
        scenario_name=""
    )


def run_fgo(sim, config_path: str, window_size: int = 100, stride: int = 50) -> EstimationResults:
    """Run FGO on simulation data (optimized for short datasets)."""
    print("\n  Running FGO...", flush=True)

    window = SlidingWindow(max_len=window_size)
    fgo = GtsamFGO(config_path=config_path)
    env = OrbitEnvironmentModel()

    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = np.zeros(3)
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    errors = []
    bias_errors = []
    t_out = []

    for k in range(len(sim.t)):
        dt = 0.02 if k == 0 else sim.t[k] - sim.t[k-1]
        omega_k = sim.omega_meas[k] if not np.any(np.isnan(sim.omega_meas[k])) else sim.omega_meas[k-1]

        q_pred = x_nom_prev.ori.propagate(omega_k - x_nom_prev.gyro_bias, dt)
        x_nom = NominalState(ori=q_pred, gyro_bias=x_nom_prev.gyro_bias)

        sample = WindowSample(
            t=float(sim.t[k]),
            jd=float(sim.jd[k]),
            x_nom=x_nom,
            omega_meas=omega_k,
            z_mag=None if np.any(np.isnan(sim.mag_meas[k])) else sim.mag_meas[k],
            z_sun=None if np.any(np.isnan(sim.sun_meas[k])) else sim.sun_meas[k],
            z_st=None if np.any(np.isnan(sim.st_meas[k])) else Quaternion.from_array(sim.st_meas[k]),
        )

        window.add(sample)
        x_nom_prev = x_nom

        if window.ready and (k % stride == 0 or k == window_size - 1):
            try:
                window_states = fgo.optimize_window(list(window.samples), env)

                for i, state in enumerate(window_states):
                    global_k = k - len(window_states) + 1 + i
                    if global_k >= len(errors):
                        q_true_k = Quaternion.from_array(sim.q_true[global_k])
                        att_err = compute_attitude_error_deg(q_true_k, state.ori)
                        errors.append(att_err)
                        bias_errors.append(np.linalg.norm(state.gyro_bias - sim.b_g_true[global_k]))
                        t_out.append(sim.t[global_k])

            except Exception as e:
                print(f"    FGO failed at step {k}: {e}")
                break

    return EstimationResults(
        t=np.array(t_out),
        attitude_error=np.array(errors),
        bias_error=np.array(bias_errors),
        sigma_attitude=None,
        sigma_bias=None,
        method_name="FGO",
        scenario_name=""
    )


def main():
    print("="*70)
    print("ESKF VS FGO COMPARISON TEST")
    print("="*70)

    # Define scenarios to test (run_id: (scenario_name, config_path))
    scenarios = {
        5: ("Baseline", "configs/config_baseline_short.yaml"),
        6: ("Rapid Tumbling", "configs/config_rapid_tumbling_short.yaml"),
        7: ("Eclipse", "configs/config_eclipse_short.yaml"),
        8: ("Measurement Spikes", "configs/config_measurement_spikes_short.yaml"),
    }

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Load database
    db = SimulationDatabase("simulations.db")

    plotter = ComparisonPlotter()
    all_results = {}

    for run_id, (scenario_name, config_path) in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name} (run_id={run_id})")
        print(f"{'='*70}")

        # Load simulation
        sim = db.load_run(run_id)
        print(f"  Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

        # Run both methods
        eskf_results = run_eskf(sim, config_path)
        eskf_results.scenario_name = scenario_name

        fgo_results = run_fgo(sim, config_path)
        fgo_results.scenario_name = scenario_name

        all_results[scenario_name] = [eskf_results, fgo_results]

        # Generate plots for this scenario
        plotter.plot_attitude_comparison(
            [eskf_results, fgo_results],
            filename=output_dir / f"{scenario_name.lower().replace(' ', '_')}_comparison.png",
            title=f"{scenario_name}: ESKF vs FGO"
        )

        plotter.plot_statistics_comparison(
            [eskf_results, fgo_results],
            filename=output_dir / f"{scenario_name.lower().replace(' ', '_')}_statistics.png"
        )

        # Print summary
        print(f"\n  Results Summary:")
        print(f"    ESKF - Mean: {np.mean(np.abs(eskf_results.attitude_error)):.4f}°, "
              f"Max: {np.max(np.abs(eskf_results.attitude_error)):.4f}°")
        print(f"    FGO  - Mean: {np.mean(np.abs(fgo_results.attitude_error)):.4f}°, "
              f"Max: {np.max(np.abs(fgo_results.attitude_error)):.4f}°")

    # Generate multi-scenario comparison
    if len(all_results) > 1:
        plotter.plot_multi_scenario_comparison(
            all_results,
            filename=output_dir / "all_scenarios_comparison.png"
        )

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nPlots saved to: {output_dir}/")
    print(f"Tested {len(scenarios)} scenarios")

    return 0


if __name__ == "__main__":
    sys.exit(main())
