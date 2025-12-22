#!/usr/bin/env python3
"""
Rapid Tumbling Scenario: Per-axis Euler angle error analysis.

Evaluates ESKF, Smoother, and Redundant estimator performance
in terms of roll, pitch, and yaw errors.
"""

import numpy as np
import sqlite3

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss

import logging
logging.disable(logging.INFO)


def quaternion_to_euler(q: Quaternion) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.

    Uses ZYX convention (yaw-pitch-roll).
    """
    arr = q.as_array()
    w, x, y, z = arr[0], arr[1], arr[2], arr[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(np.array([roll, pitch, yaw]))


def compute_euler_errors(q_est: Quaternion, q_true: Quaternion) -> np.ndarray:
    """
    Compute per-axis Euler angle errors in degrees.

    Returns: [roll_error, pitch_error, yaw_error]
    """
    euler_est = quaternion_to_euler(q_est)
    euler_true = quaternion_to_euler(q_true)

    errors = euler_est - euler_true

    # Wrap to [-180, 180]
    for i in range(3):
        while errors[i] > 180:
            errors[i] -= 360
        while errors[i] < -180:
            errors[i] += 360

    return errors


def apply_rapid_tumbling(sim_data):
    """No star tracker measurements (rapid tumbling scenario)."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())

    # Remove all star tracker measurements
    for k in range(len(modified.t)):
        modified.st_meas[k] = np.array([np.nan, np.nan, np.nan, np.nan])

    return modified


CONFIG_PATH = "configs/config_baseline_short.yaml"


def run_eskf_euler(sim_data) -> np.ndarray:
    """Run ESKF and return per-axis errors over time."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    euler_errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        q_true = Quaternion.from_array(sim_data.q_true[k])
        euler_errors.append(compute_euler_errors(x.nom.ori, q_true))

        if not np.isnan(sim_data.omega_meas[k, 0]):
            x = eskf.predict(x, sim_data.omega_meas[k], dt)

        if not np.isnan(sim_data.mag_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim_data.b_eci[k])
            except:
                pass

        if not np.isnan(sim_data.sun_meas[k, 0]):
            try:
                x = eskf.update(x, sim_data.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim_data.s_eci[k])
            except:
                pass

        if not np.isnan(sim_data.st_meas[k, 0]):
            try:
                q_st = Quaternion.from_array(sim_data.st_meas[k])
                x = eskf.update(x, q_st, SensorType.STAR_TRACKER)
            except:
                pass

    return np.array(euler_errors)


def run_smoother_euler(sim_data) -> np.ndarray:
    """Run smoother and return per-axis errors over time."""
    smoother = FixedLagAttitudeSmoother(
        config_path=CONFIG_PATH,
        lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        normalize_mag=True,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    smoother.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    dt = sim_data.t[1] - sim_data.t[0]
    keyframe_times = [sim_data.t[0]]
    keyframe_states = [q_init.copy()]

    for k in range(len(sim_data.t)):
        if not np.isnan(sim_data.omega_meas[k, 0]):
            smoother.integrate_gyro(sim_data.omega_meas[k], dt, t=sim_data.t[k])

        z_mag = sim_data.mag_meas[k] if not np.isnan(sim_data.mag_meas[k, 0]) else None
        z_sun = sim_data.sun_meas[k] if not np.isnan(sim_data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.isnan(sim_data.st_meas[k, 0]) else None

        if z_mag is not None or z_sun is not None or z_st is not None:
            smoother.add_measurement(
                t=sim_data.t[k], jd=sim_data.jd[k],
                z_mag=z_mag, z_sun=z_sun, z_st=z_st,
                B_eci=sim_data.b_eci[k], s_eci=sim_data.s_eci[k],
            )
            state = smoother.get_state()
            if state is not None:
                keyframe_times.append(sim_data.t[k])
                keyframe_states.append(state.ori.copy())

    # Interpolate and compute errors
    keyframe_times = np.array(keyframe_times)
    euler_errors = []

    for k in range(len(sim_data.t)):
        t_k = sim_data.t[k]
        q_true = Quaternion.from_array(sim_data.q_true[k])

        idx = np.searchsorted(keyframe_times, t_k)
        if idx == 0:
            q_interp = keyframe_states[0]
        elif idx >= len(keyframe_times):
            q_interp = keyframe_states[-1]
        else:
            t0, t1 = keyframe_times[idx-1], keyframe_times[idx]
            q0, q1 = keyframe_states[idx-1], keyframe_states[idx]
            alpha = (t_k - t0) / (t1 - t0) if t1 > t0 else 0
            q_interp = q0.slerp(q1, alpha)

        euler_errors.append(compute_euler_errors(q_interp, q_true))

    return np.array(euler_errors)


def run_redundant_euler(sim_data) -> np.ndarray:
    """Run redundant estimator and return per-axis errors over time."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)

    redundant = RedundantEstimator(
        P0=P0,
        config_path=CONFIG_PATH,
        smoother_lag=60.0,
        use_robust=True,
        robust_kernel="cauchy",
        robust_param=0.1,
        disagreement_threshold_deg=2.0,
        consecutive_disagreements_to_switch=5,
    )

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x_eskf = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )
    redundant.initialize(sim_data.t[0], q_init.copy(), np.zeros(3))

    euler_errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        q_true = Quaternion.from_array(sim_data.q_true[k])

        z_mag = sim_data.mag_meas[k] if not np.isnan(sim_data.mag_meas[k, 0]) else None
        z_sun = sim_data.sun_meas[k] if not np.isnan(sim_data.sun_meas[k, 0]) else None
        z_st = Quaternion.from_array(sim_data.st_meas[k]) if not np.isnan(sim_data.st_meas[k, 0]) else None

        x_eskf, smoother_state, disagreement, primary = redundant.step(
            x_eskf=x_eskf, t=sim_data.t[k], jd=sim_data.jd[k], dt=dt,
            omega_meas=sim_data.omega_meas[k],
            z_mag=z_mag, z_sun=z_sun, z_st=z_st,
            B_n=sim_data.b_eci[k], s_n=sim_data.s_eci[k],
        )

        if primary == "SMOOTHER" and smoother_state is not None:
            q_est = smoother_state.ori
        else:
            q_est = x_eskf.nom.ori

        euler_errors.append(compute_euler_errors(q_est, q_true))

    return np.array(euler_errors)


def main():
    db_path = "simulations.db"
    n_runs = 5

    # Get run IDs
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    db = SimulationDatabase(db_path)

    estimators = [
        ("ESKF", run_eskf_euler),
        ("Smoother", run_smoother_euler),
        ("Redundant", run_redundant_euler),
    ]

    print("=" * 80)
    print("RAPID TUMBLING SCENARIO: Per-Axis Euler Angle Error Analysis")
    print("=" * 80)
    print(f"Monte Carlo runs: {n_runs}")
    print()

    # Store results: results[estimator] = list of (roll_rmse, pitch_rmse, yaw_rmse)
    results = {name: [] for name, _ in estimators}

    for i, run_id in enumerate(run_ids):
        print(f"Run {i+1}/{n_runs}...", end=" ", flush=True)

        sim_data = db.load_run(run_id)
        modified_data = apply_rapid_tumbling(sim_data)

        for est_name, est_fn in estimators:
            try:
                euler_errors = est_fn(modified_data)

                # Compute steady-state performance (last 50%)
                mask = modified_data.t >= modified_data.t[-1] * 0.5
                ss_errors = euler_errors[mask]

                # Compute RMSE per axis
                roll_rmse = np.sqrt(np.mean(ss_errors[:, 0]**2))
                pitch_rmse = np.sqrt(np.mean(ss_errors[:, 1]**2))
                yaw_rmse = np.sqrt(np.mean(ss_errors[:, 2]**2))

                results[est_name].append((roll_rmse, pitch_rmse, yaw_rmse))
            except Exception as e:
                print(f"\n  {est_name} failed: {e}")
                results[est_name].append((np.nan, np.nan, np.nan))

        print("done")

    # Print results table
    print()
    print("=" * 80)
    print("RESULTS: Steady-State RMSE per Axis [deg]")
    print("=" * 80)
    print(f"{'Estimator':<15} {'Roll':>15} {'Pitch':>15} {'Yaw':>15} {'Total':>15}")
    print("-" * 75)

    for est_name, _ in estimators:
        vals = [v for v in results[est_name] if not np.isnan(v[0])]
        if vals:
            roll_mean = np.mean([v[0] for v in vals])
            roll_std = np.std([v[0] for v in vals])
            pitch_mean = np.mean([v[1] for v in vals])
            pitch_std = np.std([v[1] for v in vals])
            yaw_mean = np.mean([v[2] for v in vals])
            yaw_std = np.std([v[2] for v in vals])

            # Total RMSE (RSS of axis RMSEs)
            total = np.sqrt(roll_mean**2 + pitch_mean**2 + yaw_mean**2)

            print(f"{est_name:<15} {roll_mean:>6.3f}±{roll_std:.3f} {pitch_mean:>6.3f}±{pitch_std:.3f} {yaw_mean:>6.3f}±{yaw_std:.3f} {total:>15.3f}")
        else:
            print(f"{est_name:<15} {'FAILED':>15} {'FAILED':>15} {'FAILED':>15}")

    # LaTeX table
    print()
    print("=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Rapid tumbling scenario: per-axis RMSE (degrees, $N=""" + str(n_runs) + r"""$ runs)}
\label{tab:rapid_tumbling_euler}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Estimator} & \textbf{Roll} & \textbf{Pitch} & \textbf{Yaw} & \textbf{Total} \\
\midrule
"""

    for est_name, _ in estimators:
        vals = [v for v in results[est_name] if not np.isnan(v[0])]
        if vals:
            roll_mean = np.mean([v[0] for v in vals])
            roll_std = np.std([v[0] for v in vals])
            pitch_mean = np.mean([v[1] for v in vals])
            pitch_std = np.std([v[1] for v in vals])
            yaw_mean = np.mean([v[2] for v in vals])
            yaw_std = np.std([v[2] for v in vals])
            total = np.sqrt(roll_mean**2 + pitch_mean**2 + yaw_mean**2)

            latex += f"{est_name} & ${roll_mean:.3f}^\\circ \\pm {roll_std:.3f}^\\circ$ "
            latex += f"& ${pitch_mean:.3f}^\\circ \\pm {pitch_std:.3f}^\\circ$ "
            latex += f"& ${yaw_mean:.3f}^\\circ \\pm {yaw_std:.3f}^\\circ$ "
            latex += f"& ${total:.3f}^\\circ$ \\\\\n"
        else:
            latex += f"{est_name} & -- & -- & -- & -- \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(latex)

    with open("rapid_tumbling_euler.tex", "w") as f:
        f.write(latex)
    print("\nSaved: rapid_tumbling_euler.tex")


if __name__ == "__main__":
    main()
