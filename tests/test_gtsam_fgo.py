#!/usr/bin/env python3
"""
Test GTSAM Factor Graph Optimization on realistic space-grade data.
"""

import numpy as np
import sys

from data.db import SimulationDatabase
from estimation.gtsam_fg import GtsamFGO, WindowSample, SlidingWindow
from utilities.quaternion import Quaternion
from utilities.states import NominalState
from environment.environment import OrbitEnvironmentModel


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def main():
    print("="*70, flush=True)
    print("GTSAM FACTOR GRAPH TEST - REALISTIC SPACE-GRADE DATA", flush=True)
    print("="*70, flush=True)

    print("\nLoading database...", flush=True)
    # Load simulation data
    db = SimulationDatabase("simulations.db")
    print("Loading simulation run 9 (baseline)...", flush=True)
    sim = db.load_run(9)
    print("Simulation loaded!", flush=True)

    print(f"\nLoaded simulation run 9:", flush=True)
    print(f"  Duration: {sim.t[-1]:.1f} seconds", flush=True)
    print(f"  Samples: {len(sim.t)}", flush=True)

    # Check bias
    max_bias = np.max(np.linalg.norm(sim.b_g_true, axis=1))
    print(f"  Max true gyro bias: {max_bias:.2e} rad/s")

    # Initialize factor graph optimizer
    window_size = 500  # 6 seconds at 50 Hz
    optimize_stride = 1000  # Optimize every 500 steps (20 second)
    window = SlidingWindow(max_len=window_size)
    fgo = GtsamFGO()
    env = OrbitEnvironmentModel()

    print(f"\nFactor graph configuration:", flush=True)
    print(f"  Window size: {window_size} samples", flush=True)
    print(f"  Optimization stride: {optimize_stride} samples", flush=True)
    print(f"  Optimization: Batched sliding window smoother", flush=True)

    # Initial nominal state
    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = np.zeros(3)
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    # Initial error
    att_err_0 = compute_attitude_error_deg(q0, q0)
    print(f"\nInitial attitude error: {att_err_0:.4f}°", flush=True)

    # Run factor graph optimizer
    print(f"\nRunning GTSAM FGO on {len(sim.t)} samples...", flush=True)

    est_states = []
    errors = []
    bias_errors = []

    n_mag_used = 0
    n_st_used = 0
    n_sun_used = 0
    n_optimizations = 0

    for k in range(30000):
        print(f"Processing sample {k+1}/{len(sim.t)}", flush=True)
        if k == 0:
            dt = 0.02
        else:
            dt = sim.t[k] - sim.t[k-1]

        omega_k = sim.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim.omega_meas[k-1]

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
            print(f"  Optimizing at step {k} (window size: {len(window.samples)})...", flush=True)
            try:
                window_samples = list(window.samples)
                window_states = fgo.optimize_window(window_samples, env)
                n_optimizations += 1

                # Store all states from this optimization batch
                for i, state in enumerate(window_states):
                    # Only store states we haven't stored yet
                    global_k = k - len(window_states) + 1 + i

                    # Skip if we've already stored this state
                    if global_k < len(est_states):
                        continue

                    q_true_k = Quaternion.from_array(sim.q_true[global_k])
                    att_err = compute_attitude_error_deg(q_true_k, state.ori)
                    errors.append(att_err)

                    bias_err = np.linalg.norm(state.gyro_bias - sim.b_g_true[global_k])
                    bias_errors.append(bias_err)

                    est_states.append(state)

                # Progress reporting
                if n_optimizations in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]:
                    last_k = k - len(window_states) + len(window_states)
                    print(f"  Optimization {n_optimizations:4d} complete: stored {len(window_states)} states, "
                          f"total states: {len(est_states)}", flush=True)
                    
                window.samples.clear()  # Clear window after optimization
                

            except Exception as e:
                print(f"\nOptimization failed at step {k}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

    # Results
    errors = np.array(errors)
    bias_errors = np.array(bias_errors)

    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Samples processed: {len(est_states)}/{len(sim.t)}")
    print(f"Optimizations run: {n_optimizations}")
    print(f"\nMeasurements used:")
    print(f"  Magnetometer: {n_mag_used}")
    print(f"  Sun sensor: {n_sun_used}")
    print(f"  Star tracker: {n_st_used}")

    if len(errors) > 0:
        print(f"\nAttitude Error:")
        print(f"  Mean:  {errors.mean():.4f}°")
        print(f"  Std:   {errors.std():.4f}°")
        print(f"  Max:   {errors.max():.4f}°")
        print(f"  Final: {errors[-1]:.4f}°")
        print(f"\nBias Error:")
        print(f"  Mean:  {bias_errors.mean():.2e} rad/s")
        print(f"  Max:   {bias_errors.max():.2e} rad/s")
        print(f"  Final: {bias_errors[-1]:.2e} rad/s")

        if len(errors) >= len(sim.t) - window_size:
            print(f"\n✓ SUCCESS - GTSAM FGO converged!")
            return 0
        else:
            print(f"\n✗ PARTIAL - Only processed {len(errors)} samples")
            return 1
    else:
        print(f"\n✗ FAILED - No states estimated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
