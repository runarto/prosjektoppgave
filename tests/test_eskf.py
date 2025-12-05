#!/usr/bin/env python3
"""
Test script for Error-State Kalman Filter (ESKF) implementation.

This script tests the ESKF with synthetic data to verify:
1. Prediction step correctness
2. Update step correctness
3. Error injection mechanism
4. Covariance propagation
5. Overall filter performance
"""

import numpy as np
import sys
from typing import List

from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss


def create_synthetic_trajectory(N=100, dt=0.02):
    """Create a simple synthetic trajectory with known dynamics."""

    # True initial state
    q_true = Quaternion(1.0, np.zeros(3))
    b_true = np.array([0.01, -0.005, 0.003])  # rad/s

    # True angular velocity (constant for simplicity)
    omega_true = np.array([0.1, 0.05, -0.02])  # rad/s

    # Storage
    q_true_list = [q_true]
    b_true_list = [b_true]
    omega_meas_list = []
    mag_meas_list = []
    sun_meas_list = []
    st_meas_list = []

    # Reference vectors
    B_n = np.array([0.3, 0.1, 0.5])
    B_n = B_n / np.linalg.norm(B_n)

    s_n = np.array([1.0, 0.0, 0.0])
    s_n = s_n / np.linalg.norm(s_n)

    # Noise parameters
    gyro_noise_std = 0.0001  # rad/s
    mag_noise_std = 0.00005   # unit vector error
    sun_noise_std = 0.00001   # unit vector error
    st_noise_std = 0.000005   # rad (small angle)

    # Generate trajectory
    for i in range(N):
        # Gyro measurement
        omega_meas = omega_true + b_true + np.random.normal(0, gyro_noise_std, 3)
        omega_meas_list.append(omega_meas)

        # Magnetometer measurement (every 5 steps)
        if i % 5 == 0:
            R_bn = q_true.as_rotmat().T
            z_mag = R_bn @ B_n + np.random.normal(0, mag_noise_std, 3)
            z_mag = z_mag / np.linalg.norm(z_mag)
        else:
            z_mag = None
        mag_meas_list.append(z_mag)

        # Sun sensor measurement (every 3 steps)
        if i % 3 == 0:
            R_bn = q_true.as_rotmat().T
            z_sun = R_bn @ s_n + np.random.normal(0, sun_noise_std, 3)
            z_sun = z_sun / np.linalg.norm(z_sun)
        else:
            z_sun = None
        sun_meas_list.append(z_sun)

        # Star tracker measurement (every 10 steps)
        if i % 10 == 0:
            # Create small attitude error
            delta_theta = np.random.normal(0, st_noise_std, 3)
            delta_q = Quaternion.from_avec(delta_theta)
            q_st = q_true.multiply(delta_q).normalize()
        else:
            q_st = None
        st_meas_list.append(q_st)

        # Propagate true state
        if i < N - 1:
            q_true = q_true.propagate(omega_true, dt)
            b_true = b_true  # constant for now
            q_true_list.append(q_true)
            b_true_list.append(b_true)

    return {
        'q_true': q_true_list,
        'b_true': b_true_list,
        'omega_meas': omega_meas_list,
        'mag_meas': mag_meas_list,
        'sun_meas': sun_meas_list,
        'st_meas': st_meas_list,
        'B_n': B_n,
        's_n': s_n,
        'dt': dt
    }


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def test_eskf_prediction():
    """Test ESKF prediction step."""
    print("\n" + "="*60)
    print("Test 1: ESKF Prediction Step")
    print("="*60)

    # Initialize ESKF
    P0 = np.diag([0.01, 0.01, 0.01, 1e-5, 1e-5, 1e-5])
    eskf = ESKF(P0=P0)

    # Initial state
    q0 = Quaternion(1.0, np.zeros(3))
    b0 = np.zeros(3)
    nom0 = NominalState(ori=q0, gyro_bias=b0)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    # Test prediction
    omega_meas = np.array([0.1, 0.0, 0.0])
    dt = 0.02

    print(f"\nInitial state:")
    print(f"  Quaternion: {x_est.nom.ori.as_array()}")
    print(f"  Gyro bias: {x_est.nom.gyro_bias}")
    print(f"  P diag: {np.diag(x_est.err.cov)}")

    x_pred = eskf.predict(x_est, omega_meas, dt)

    print(f"\nAfter prediction (dt={dt}s, omega={omega_meas}):")
    print(f"  Quaternion: {x_pred.nom.ori.as_array()}")
    print(f"  Gyro bias: {x_pred.nom.gyro_bias}")
    print(f"  P diag: {np.diag(x_pred.err.cov)}")

    # Check that quaternion changed
    q_diff = q0.conjugate().multiply(x_pred.nom.ori)
    angle_change = 2 * np.arccos(np.clip(abs(q_diff.mu), 0, 1))
    print(f"\n  Attitude change: {np.rad2deg(angle_change):.4f} deg")

    # Check that covariance increased
    P_increase = np.diag(x_pred.err.cov) > np.diag(P0)
    print(f"  Covariance increased: {P_increase.all()}")

    if angle_change > 0 and P_increase.all():
        print("\n✓ Prediction test PASSED")
        return True
    else:
        print("\n✗ Prediction test FAILED")
        return False


def test_eskf_update():
    """Test ESKF update step with star tracker."""
    print("\n" + "="*60)
    print("Test 2: ESKF Update Step")
    print("="*60)

    # Initialize ESKF
    P0 = np.diag([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0)

    # Initial state with some error
    q_true = Quaternion(1.0, np.zeros(3))
    q_est_init = Quaternion.from_avec(np.array([0.01, 0.01, 0.01]))  # 1 deg error
    b_est = np.zeros(3)

    nom = NominalState(ori=q_est_init, gyro_bias=b_est)
    err = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom, err=err)

    print(f"\nInitial state (with error):")
    err_before = compute_attitude_error_deg(q_true, x_est.nom.ori)
    print(f"  Attitude error: {err_before:.4f} deg")
    print(f"  P diag: {np.diag(x_est.err.cov)}")

    # Update with perfect measurement
    x_upd = eskf.update(
        x_est=x_est,
        y=q_true,  # perfect measurement
        sensor_type=SensorType.STAR_TRACKER
    )

    print(f"\nAfter star tracker update:")
    err_after = compute_attitude_error_deg(q_true, x_upd.nom.ori)
    print(f"  Attitude error: {err_after:.4f} deg")
    print(f"  P diag: {np.diag(x_upd.err.cov)}")

    # Check that error decreased
    error_reduced = err_after < err_before
    # Check only attitude covariance (first 3 diagonal elements)
    # Star tracker doesn't directly observe bias, so bias covariance won't reduce
    p_att_reduced = np.diag(x_upd.err.cov)[:3] < np.diag(P0)[:3]

    print(f"\n  Error reduced: {error_reduced} ({err_before:.4f} → {err_after:.4f} deg)")
    print(f"  Attitude covariance reduced: {p_att_reduced.all()}")
    print(f"  P_att before: {np.diag(P0)[:3]}")
    print(f"  P_att after:  {np.diag(x_upd.err.cov)[:3]}")

    if error_reduced and p_att_reduced.all():
        print("\n✓ Update test PASSED")
        return True
    else:
        print("\n✗ Update test FAILED")
        return False


def test_eskf_full_filter():
    """Test complete ESKF with synthetic trajectory."""
    print("\n" + "="*60)
    print("Test 3: Full ESKF with Synthetic Data")
    print("="*60)

    # Generate synthetic data
    N = 100
    data = create_synthetic_trajectory(N=N, dt=0.02)

    print(f"\nGenerated trajectory with {N} samples")
    print(f"  True initial bias: {data['b_true'][0]}")
    print(f"  Angular velocity: constant")

    # Initialize ESKF
    P0 = np.diag([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3])
    eskf = ESKF(P0=P0)

    # Initial state (with small error)
    q0_true = data['q_true'][0]
    q0_est = Quaternion.from_avec(np.array([0.05, 0.05, 0.05]))  # ~3 deg error
    b0_est = np.zeros(3)  # wrong initial bias

    nom0 = NominalState(ori=q0_est, gyro_bias=b0_est)
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    # Track errors
    att_errors = []
    bias_errors = []

    print(f"\nInitial errors:")
    print(f"  Attitude: {compute_attitude_error_deg(q0_true, q0_est):.4f} deg")
    print(f"  Bias: {np.linalg.norm(b0_est - data['b_true'][0]):.6f} rad/s")

    # Run filter
    print(f"\nRunning filter...")
    for k in range(1, N):
        # Predict
        x_est = eskf.predict(x_est, data['omega_meas'][k], data['dt'])

        # Update with available measurements
        if data['mag_meas'][k] is not None:
            x_est = eskf.update(
                x_est=x_est,
                y=data['mag_meas'][k],
                sensor_type=SensorType.MAGNETOMETER,
                B_n=data['B_n']
            )

        if data['sun_meas'][k] is not None:
            x_est = eskf.update(
                x_est=x_est,
                y=data['sun_meas'][k],
                sensor_type=SensorType.SUN_VECTOR,
                s_n=data['s_n']
            )

        if data['st_meas'][k] is not None:
            x_est = eskf.update(
                x_est=x_est,
                y=data['st_meas'][k],
                sensor_type=SensorType.STAR_TRACKER
            )

        # Compute errors
        att_err = compute_attitude_error_deg(data['q_true'][k], x_est.nom.ori)
        bias_err = np.linalg.norm(x_est.nom.gyro_bias - data['b_true'][k])

        att_errors.append(att_err)
        bias_errors.append(bias_err)

    # Analyze results
    att_errors = np.array(att_errors)
    bias_errors = np.array(bias_errors)

    print(f"\nFinal errors:")
    print(f"  Attitude: {att_errors[-1]:.4f} deg")
    print(f"  Bias: {bias_errors[-1]:.6f} rad/s")

    print(f"\nAttitude error statistics:")
    print(f"  Mean:   {np.mean(att_errors):.4f} deg")
    print(f"  Median: {np.median(att_errors):.4f} deg")
    print(f"  Max:    {np.max(att_errors):.4f} deg")
    print(f"  RMS:    {np.sqrt(np.mean(att_errors**2)):.4f} deg")

    print(f"\nBias error statistics:")
    print(f"  Mean:   {np.mean(bias_errors):.6e} rad/s")
    print(f"  Final:  {bias_errors[-1]:.6e} rad/s")

    # Check convergence
    final_att_err = att_errors[-1]
    final_bias_err = bias_errors[-1]
    mean_att_err = np.mean(att_errors[-20:])  # last 20 samples

    # Success criteria
    att_converged = mean_att_err < 1.0  # < 1 deg on average
    # Bias convergence is slower - check if it's improving
    bias_improving = bias_errors[-1] < bias_errors[0]  # final < initial
    bias_reasonable = final_bias_err < 0.005  # < 5e-3 rad/s (more relaxed)

    print(f"\nConvergence check:")
    print(f"  Attitude converged: {att_converged} (mean last 20: {mean_att_err:.4f} deg)")
    print(f"  Bias improving: {bias_improving} ({bias_errors[0]:.6e} → {final_bias_err:.6e})")
    print(f"  Bias reasonable: {bias_reasonable} (< 5e-3 rad/s)")

    if att_converged and bias_improving and bias_reasonable:
        print("\n✓ Full filter test PASSED")
        return True
    else:
        print("\n✗ Full filter test FAILED")
        return False


def main():
    """Run all ESKF tests."""
    print("\n" + "="*60)
    print("ERROR-STATE KALMAN FILTER TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    try:
        results['prediction'] = test_eskf_prediction()
    except Exception as e:
        print(f"\nPrediction test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['prediction'] = False

    try:
        results['update'] = test_eskf_update()
    except Exception as e:
        print(f"\nUpdate test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['update'] = False

    try:
        results['full_filter'] = test_eskf_full_filter()
    except Exception as e:
        print(f"\nFull filter test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['full_filter'] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
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
