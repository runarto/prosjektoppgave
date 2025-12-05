#!/usr/bin/env python3
"""
Test manual ESKF implementation on space-grade data WITH measurement spikes.
"""

import numpy as np
import sys

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss


def compute_attitude_error_deg(q_true: Quaternion, q_est: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_true.conjugate().multiply(q_est)
    angle_err = 2 * np.arccos(np.clip(abs(q_err.mu), 0, 1))
    return np.rad2deg(angle_err)


def main():
    print("="*70)
    print("MANUAL ESKF TEST - SPACE-GRADE DATA WITH SPIKES")
    print("="*70)

    # Load simulation data
    db = SimulationDatabase("simulations.db")
    sim = db.load_run(11)

    print(f"\nLoaded simulation run 11:")
    print(f"  Duration: {sim.t[-1]:.1f} seconds")
    print(f"  Samples: {len(sim.t)}")

    # Check bias
    max_bias = np.max(np.linalg.norm(sim.b_g_true, axis=1))
    print(f"  Max true gyro bias: {max_bias:.2e} rad/s")

    # Initialize filter
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path="configs/config_spacegrade_large_spikes.yaml")

    # Initial state with small error
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()
    b0_est = np.zeros(3)

    nom = NominalState(ori=q0_est, gyro_bias=b0_est)
    err = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom, err=err)

    # Initial error
    att_err_0 = compute_attitude_error_deg(q0_true, q0_est)
    print(f"\nInitial attitude error: {att_err_0:.4f}°")

    # Run filter
    print(f"\nRunning manual ESKF on {len(sim.t)-1} samples...")
    errors = []
    bias_errors = []

    n_mag_updates = 0
    n_st_updates = 0
    n_sun_updates = 0

    n_mag_rejected = 0
    n_st_rejected = 0
    n_sun_rejected = 0

    for k in range(1, len(sim.t)):
        # Get measurements
        omega_k = sim.omega_meas[k]
        if np.any(np.isnan(omega_k)):
            omega_k = sim.omega_meas[k-1]
        dt_k = sim.t[k] - sim.t[k-1]

        # Predict
        try:
            x_est = eskf.predict(x_est, omega_k, dt_k)
        except Exception as e:
            print(f"\nPredict failed at step {k}: {e}")
            break

        # Update with magnetometer
        mag_meas = sim.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=mag_meas,
                    sensor_type=SensorType.MAGNETOMETER,
                    B_n=sim.b_eci[k]
                )
                n_mag_updates += 1
            except ValueError as e:
                # Innovation too large - likely a spike
                n_mag_rejected += 1
                if "innovation too large" in str(e).lower():
                    pass  # Skip this measurement
                else:
                    print(f"\nMagnetometer update failed at step {k}: {e}")
                    break
            except Exception as e:
                print(f"\nMagnetometer update failed at step {k}: {e}")
                break

        # Update with sun sensor
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(
                    x_est=x_est,
                    y=sun_meas,
                    sensor_type=SensorType.SUN_VECTOR,
                    s_n=sim.s_eci[k]
                )
                n_sun_updates += 1
            except ValueError as e:
                # Innovation too large - likely a spike
                n_sun_rejected += 1
                if "innovation too large" in str(e).lower():
                    pass  # Skip this measurement
                else:
                    print(f"\nSun sensor update failed at step {k}: {e}")
                    break
            except Exception as e:
                print(f"\nSun sensor update failed at step {k}: {e}")
                break

        # Update with star tracker
        st_meas = sim.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(
                    x_est=x_est,
                    y=q_meas,
                    sensor_type=SensorType.STAR_TRACKER
                )
                n_st_updates += 1
            except ValueError as e:
                # Innovation too large - likely a spike
                n_st_rejected += 1
                if "innovation too large" in str(e).lower():
                    pass  # Skip this measurement
                else:
                    print(f"\nStar tracker update failed at step {k}: {e}")
                    break
            except Exception as e:
                print(f"\nStar tracker update failed at step {k}: {e}")
                break

        # Compute errors
        q_true_k = Quaternion.from_array(sim.q_true[k])
        att_err = compute_attitude_error_deg(q_true_k, x_est.nom.ori)
        errors.append(att_err)

        bias_err = np.linalg.norm(x_est.nom.gyro_bias - sim.b_g_true[k])
        bias_errors.append(bias_err)

        # Check for divergence
        if att_err > 10.0:
            print(f"\nFilter diverged at step {k} (t={sim.t[k]:.2f}s): error={att_err:.2f}°")
            break

        # Progress reporting
        if k in [1, 10, 100, 500, 1000, 5000, 10000, 50000, 100000, 150000]:
            P_diag = np.diag(x_est.err.cov)
            sigma_att = np.rad2deg(np.sqrt(P_diag[0:3]))
            sigma_bias = np.sqrt(P_diag[3:6])
            print(f"  Step {k:6d}: att_err={att_err:.4f}°, bias_err={bias_err:.2e}, "
                  f"σ_att=[{sigma_att[0]:.3f},{sigma_att[1]:.3f},{sigma_att[2]:.3f}]°, "
                  f"σ_bias=[{sigma_bias[0]:.2e},{sigma_bias[1]:.2e},{sigma_bias[2]:.2e}]")

    # Results
    errors = np.array(errors)
    bias_errors = np.array(bias_errors)

    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Samples processed: {len(errors)}/{len(sim.t)-1}")
    print(f"\nMeasurement updates:")
    print(f"  Magnetometer: {n_mag_updates} (rejected: {n_mag_rejected})")
    print(f"  Sun sensor: {n_sun_updates} (rejected: {n_sun_rejected})")
    print(f"  Star tracker: {n_st_updates} (rejected: {n_st_rejected})")
    print(f"\nRejection rates:")
    total_mag = n_mag_updates + n_mag_rejected
    total_sun = n_sun_updates + n_sun_rejected
    total_st = n_st_updates + n_st_rejected
    if total_mag > 0:
        print(f"  Magnetometer: {100*n_mag_rejected/total_mag:.2f}%")
    if total_sun > 0:
        print(f"  Sun sensor: {100*n_sun_rejected/total_sun:.2f}%")
    if total_st > 0:
        print(f"  Star tracker: {100*n_st_rejected/total_st:.2f}%")

    print(f"\nAttitude Error:")
    print(f"  Mean:  {errors.mean():.4f}°")
    print(f"  Std:   {errors.std():.4f}°")
    print(f"  Max:   {errors.max():.4f}°")
    print(f"  Final: {errors[-1]:.4f}°")
    print(f"\nBias Error:")
    print(f"  Mean:  {bias_errors.mean():.2e} rad/s")
    print(f"  Max:   {bias_errors.max():.2e} rad/s")
    print(f"  Final: {bias_errors[-1]:.2e} rad/s")

    if len(errors) == len(sim.t) - 1:
        print(f"\n✓ SUCCESS - Manual ESKF converged on all samples!")
        print(f"  Filter successfully rejected {n_mag_rejected + n_sun_rejected + n_st_rejected} spike measurements")
        return 0
    else:
        print(f"\n✗ FAILED - Filter diverged")
        return 1


if __name__ == "__main__":
    sys.exit(main())
