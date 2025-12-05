#!/usr/bin/env python3
"""
Fast FGO test with optimized parameters for long datasets.

Optimization strategies:
1. Larger stride (optimize every 1000 steps = 20 seconds)
2. Smaller window (200 samples = 4 seconds)
3. Non-overlapping batches after initial convergence
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
    print("="*70)
    print("FAST GTSAM FGO TEST - OPTIMIZED FOR LONG DATASETS")
    print("="*70)

    # Load simulation data
    db = SimulationDatabase("simulations.db")
    sim = db.load_run(9)

    print(f"\nLoaded simulation run 9:")
    print(f"  Duration: {sim.t[-1]:.1f} seconds")
    print(f"  Samples: {len(sim.t)}")

    # Optimized parameters for speed
    window_size = 200           # Reduced from 300
    optimize_stride = 1000       # Increased from 50 (optimize every 20 seconds)

    window = SlidingWindow(max_len=window_size)
    fgo = GtsamFGO(config_path="configs/config_spacegrade_realistic.yaml")
    env = OrbitEnvironmentModel()

    print(f"\nOptimized FGO configuration:")
    print(f"  Window size: {window_size} samples (4 seconds)")
    print(f"  Stride: {optimize_stride} samples (20 seconds)")
    print(f"  Expected optimizations: ~{len(sim.t) // optimize_stride}")

    # Initial state
    q0 = Quaternion.from_array(sim.q_true[0])
    b0 = np.zeros(3)
    x_nom_prev = NominalState(ori=q0, gyro_bias=b0)

    errors = []
    bias_errors = []
    t_out = []
    n_optimizations = 0

    print(f"\nRunning optimized FGO...")

    for k in range(len(sim.t)):
        dt = 0.02 if k == 0 else sim.t[k] - sim.t[k-1]
        omega_k = sim.omega_meas[k] if not np.any(np.isnan(sim.omega_meas[k])) else sim.omega_meas[k-1]

        # Propagate nominal
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

        # Optimize with large stride
        if window.ready and (k % optimize_stride == 0 or k == window_size - 1):
            try:
                window_states = fgo.optimize_window(list(window.samples), env)
                n_optimizations += 1

                # Store all states from this batch
                for i, state in enumerate(window_states):
                    global_k = k - len(window_states) + 1 + i
                    if global_k >= len(errors):
                        q_true_k = Quaternion.from_array(sim.q_true[global_k])
                        att_err = compute_attitude_error_deg(q_true_k, state.ori)
                        errors.append(att_err)
                        bias_errors.append(np.linalg.norm(state.gyro_bias - sim.b_g_true[global_k]))
                        t_out.append(sim.t[global_k])

                # Progress
                if n_optimizations % 10 == 0:
                    print(f"  Optimization {n_optimizations:3d}: step {k:6d}/{len(sim.t)-1}, "
                          f"states: {len(errors)}, time: {sim.t[k]:.1f}s", flush=True)

                # Clear window for non-overlapping batches (faster)
                window.samples.clear()

            except Exception as e:
                print(f"  FGO failed at step {k}: {e}")
                break

    # Results
    errors = np.array(errors)
    bias_errors = np.array(bias_errors)

    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Optimizations run: {n_optimizations}")
    print(f"States estimated: {len(errors)}/{len(sim.t)}")

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

        print(f"\n✓ Fast FGO completed successfully!")
        return 0
    else:
        print(f"\n✗ FGO failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
