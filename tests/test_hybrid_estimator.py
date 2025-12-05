#!/usr/bin/env python3
"""
Test Hybrid Estimator (ESKF Frontend + FGO Backend)

This script tests the hybrid architecture with automatic mode switching.
"""

import numpy as np
import sys
from pathlib import Path

from data.db import SimulationDatabase
from estimation.hybrid_estimator import HybridEstimator, EstimatorMode
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
from environment.environment import OrbitEnvironmentModel
from plotting.comparison_plotter import ComparisonPlotter, EstimationResults


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def run_hybrid(sim, config_path: str, scenario_name: str) -> EstimationResults:
    """Run hybrid estimator on simulation data."""
    print(f"\n  Running Hybrid Estimator...")

    # Initialize
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    hybrid = HybridEstimator(
        P0=P0,
        config_path=config_path,
        fgo_window_size=100,
        fgo_optimize_interval=10.0,  # Optimize every 10 seconds
        use_robust=True
        # Using default thresholds: degraded_threshold=0.2, mode_switch_hysteresis=50, warmup_samples=100
    )

    # Initial state
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()
    b0_est = np.zeros(3)

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    errors = []
    bias_errors = []
    sigma_att = []
    sigma_bias = []
    mode_log = []
    fgo_update_times = []

    for k in range(len(sim.t)):
        omega_k = sim.omega_meas[k] if not np.any(np.isnan(sim.omega_meas[k])) else sim.omega_meas[k-1]
        dt_k = sim.t[k] - sim.t[k-1]

        # Get reference fields
        B_eci = sim.b_eci[k]
        s_eci = sim.s_eci[k]

        # Prepare measurements
        z_mag = None if np.any(np.isnan(sim.mag_meas[k])) else sim.mag_meas[k]
        z_sun = None if np.any(np.isnan(sim.sun_meas[k])) else sim.sun_meas[k]
        z_st = None if np.any(np.isnan(sim.st_meas[k])) else Quaternion.from_array(sim.st_meas[k])

        # Hybrid step
        x_est, fgo_updated = hybrid.step(
            x_eskf=x_est,
            t=sim.t[k],
            jd=sim.jd[k],
            omega_meas=omega_k,
            dt=dt_k,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=z_st,
            B_n=B_eci,
            s_n=s_eci
        )

        if fgo_updated:
            fgo_update_times.append(sim.t[k])

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

        # Log mode
        mode_log.append(1 if hybrid.mode == EstimatorMode.DUAL else 2)

        if k % 1000 == 0:
            print(f"    Progress: {k}/{len(sim.t)}, Mode: {hybrid.mode.value}, "
                  f"Meas rate: {hybrid.measurement_health.measurement_rate:.1%}")

    print(f"  FGO optimizations: {len(fgo_update_times)}")
    print(f"  FGO times: {fgo_update_times[:5]}... (showing first 5)")

    return EstimationResults(
        t=sim.t[1:],
        attitude_error=np.array(errors),
        bias_error=np.array(bias_errors),
        sigma_attitude=np.array(sigma_att),
        sigma_bias=np.array(sigma_bias),
        method_name="Hybrid",
        scenario_name=scenario_name
    )


def main():
    print("="*70)
    print("HYBRID ESTIMATOR TEST")
    print("="*70)

    # Define scenarios
    scenarios = {
        5: ("Baseline", "configs/config_baseline_short.yaml"),
        6: ("Rapid Tumbling", "configs/config_rapid_tumbling_short.yaml"),
        7: ("Eclipse", "configs/config_eclipse_short.yaml"),
        8: ("Measurement Spikes", "configs/config_measurement_spikes_short.yaml"),
    }

    # Create output directory
    output_dir = Path("output/hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load database
    db = SimulationDatabase("simulations.db")

    plotter = ComparisonPlotter()

    for run_id, (scenario_name, config_path) in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name} (run_id={run_id})")
        print(f"{'='*70}")

        # Load simulation
        sim = db.load_run(run_id)
        print(f"  Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

        # Run hybrid estimator
        try:
            hybrid_results = run_hybrid(sim, config_path, scenario_name)

            # Print summary
            print(f"\n  Results Summary:")
            print(f"    Mean error: {np.mean(np.abs(hybrid_results.attitude_error)):.4f}°")
            print(f"    Max error: {np.max(np.abs(hybrid_results.attitude_error)):.4f}°")
            print(f"    Final 3σ attitude: {hybrid_results.sigma_attitude[-1]:.4f}°")

            # Save individual plot
            plotter.plot_attitude_comparison(
                [hybrid_results],
                filename=output_dir / f"{scenario_name.lower().replace(' ', '_')}_hybrid.png",
                title=f"{scenario_name}: Hybrid Estimator"
            )

        except Exception as e:
            print(f"\n  Test failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("HYBRID ESTIMATOR TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nPlots saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
