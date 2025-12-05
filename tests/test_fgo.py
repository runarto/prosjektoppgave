#!/usr/bin/env python3
"""
Standalone test script for Factor Graph Optimization implementations.
This demonstrates both the custom FGO and GTSAM-based implementations.
"""

import numpy as np
import sys
from utilities.quaternion import Quaternion
from utilities.states import NominalState, SensorType
from estimation.fg import FGO, WindowSample
from estimation.gtsam_fg import GtsamFGO

def create_synthetic_data(N=200, dt=0.02):
    """Create synthetic measurement data for testing."""

    # Initial state
    q0 = Quaternion(1.0, np.zeros(3))
    b0 = np.array([0.01, -0.005, 0.003])  # small gyro bias

    # True angular velocity (slow rotation)
    omega_true = np.array([0.05, 0.02, -0.03])  # rad/s

    # Generate synthetic true trajectory
    q_true_list = [q0]
    t_list = [0.0]
    jd_list = [2460000.0]  # arbitrary julian date

    for i in range(1, N):
        t = i * dt
        q_prev = q_true_list[-1]
        q_new = q_prev.propagate(omega_true, dt)
        q_true_list.append(q_new)
        t_list.append(t)
        jd_list.append(jd_list[0] + t / 86400.0)  # convert to days

    # Generate measurements
    samples = []
    gyro_noise_std = 0.001  # rad/s
    mag_noise_std = 0.01    # nT (normalized)
    sun_noise_std = 0.005   # unit vector error

    # Reference vectors in inertial frame
    B_n = np.array([0.3, 0.1, 0.5])  # magnetic field (arbitrary)
    B_n = B_n / np.linalg.norm(B_n)

    s_n = np.array([1.0, 0.0, 0.0])  # sun vector (arbitrary)
    s_n = s_n / np.linalg.norm(s_n)

    for i, q_true in enumerate(q_true_list):
        # Gyro measurement
        omega_meas = omega_true + b0 + np.random.normal(0, gyro_noise_std, 3)

        # Magnetometer measurement (every 5 steps)
        if i % 5 == 0:
            R_bn = q_true.as_rotmat().T
            z_mag = R_bn @ B_n + np.random.normal(0, mag_noise_std, 3)
            z_mag = z_mag / np.linalg.norm(z_mag)
        else:
            z_mag = None

        # Sun sensor measurement (every 3 steps)
        if i % 3 == 0:
            R_bn = q_true.as_rotmat().T
            z_sun = R_bn @ s_n + np.random.normal(0, sun_noise_std, 3)
            z_sun = z_sun / np.linalg.norm(z_sun)
        else:
            z_sun = None

        # Create nominal state (initial guess with small error)
        if i == 0:
            x_nom = NominalState(ori=q_true, gyro_bias=b0)
        else:
            # Propagate from previous
            x_nom = NominalState(ori=q_true, gyro_bias=b0)

        sample = WindowSample(
            t=t_list[i],
            jd=jd_list[i],
            x_nom=x_nom,
            omega_meas=omega_meas,
            z_mag=z_mag,
            z_sun=z_sun,
            z_st=None,  # no star tracker for simplicity
        )
        samples.append(sample)

    return samples, q_true_list, B_n, s_n


def test_custom_fgo():
    """Test the custom factor graph implementation."""
    print("\n" + "="*60)
    print("Testing Custom FGO Implementation (sim/fg.py)")
    print("="*60)

    samples, q_true_list, B_n, s_n = create_synthetic_data()

    fgo = FGO()
    fgo.build_graph(samples)

    print(f"\nBuilt graph with {len(fgo.states)} states and {len(fgo.factors)} factors")
    print(f"Factor types: ", end="")
    factor_counts = {}
    for f in fgo.factors:
        factor_counts[f.factor_type] = factor_counts.get(f.factor_type, 0) + 1
    for ftype, count in factor_counts.items():
        print(f"{ftype}:{count} ", end="")
    print()

    # Run optimization
    print("\nRunning optimization...")
    try:
        optimized_states = fgo.optimize(max_nfev=50)
        print(f"\nOptimization succeeded!")
        print(f"Optimized {len(optimized_states)} states")

        # Compute errors
        errors_deg = []
        for i, (opt_state, q_true) in enumerate(zip(optimized_states, q_true_list)):
            q_err = q_true.conjugate().multiply(opt_state.ori)
            angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
            errors_deg.append(np.rad2deg(angle_err))

        print(f"\nAttitude errors:")
        print(f"  Mean: {np.mean(errors_deg):.4f} deg")
        print(f"  Max:  {np.max(errors_deg):.4f} deg")
        print(f"  RMS:  {np.sqrt(np.mean(np.array(errors_deg)**2)):.4f} deg")

        return True
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gtsam_fgo():
    """Test the GTSAM factor graph implementation."""
    print("\n" + "="*60)
    print("Testing GTSAM FGO Implementation (sim/gtsam_fg.py)")
    print("="*60)

    # Create a mock environment for GTSAM
    class MockEnv:
        def get_r_eci(self, jd):
            return np.array([7000e3, 0, 0])  # arbitrary orbit position

        def get_B_eci(self, r_eci, jd):
            B = np.array([0.3, 0.1, 0.5])
            return B / np.linalg.norm(B)

        def get_sun_eci(self, jd):
            s = np.array([1.0, 0.0, 0.0])
            return s / np.linalg.norm(s)

    samples, q_true_list, _, _ = create_synthetic_data(N=1000)

    fgo = GtsamFGO(max_iters=30)
    env = MockEnv()

    print(f"\nOptimizing window with {len(samples)} samples...")
    try:
        optimized_states = fgo.optimize_window(samples, env)
        print(f"\nOptimization succeeded!")
        print(f"Optimized {len(optimized_states)} states")

        # Compute errors
        errors_deg = []
        for i, (opt_state, q_true) in enumerate(zip(optimized_states, q_true_list)):
            q_err = q_true.conjugate().multiply(opt_state.ori)
            angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
            errors_deg.append(np.rad2deg(angle_err))

        print(f"\nAttitude errors:")
        print(f"  Mean: {np.mean(errors_deg):.4f} deg")
        print(f"  Max:  {np.max(errors_deg):.4f} deg")
        print(f"  RMS:  {np.sqrt(np.mean(np.array(errors_deg)**2)):.4f} deg")

        return True
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("Factor Graph Optimization Test Suite")
    print("="*60)

    results = {}

    # Test custom implementation
    #results['custom_fgo'] = test_custom_fgo()

    # Test GTSAM implementation
    results['gtsam_fgo'] = test_gtsam_fgo()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  {name:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
