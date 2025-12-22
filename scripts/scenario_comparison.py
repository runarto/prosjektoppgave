#!/usr/bin/env python3
"""
Scenario Comparison: ESKF vs Smoother vs Redundant

Scenarios:
1. Baseline - nominal measurements
2. Eclipse - sun sensor dropout
3. Rapid tumbling - no star tracker measurements
4. Magnetometer bias - constant 0.62 bias

Settings:
- Smoother: Cauchy k=0.1, noise_scale=1.0
- 10 Monte Carlo runs per configuration
"""

import numpy as np
import time
import sqlite3

from data.db import SimulationDatabase
from estimation.eskf import ESKF
from estimation.fixed_lag_smoother import FixedLagAttitudeSmoother
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss


def compute_attitude_error_deg(q_est: Quaternion, q_true: Quaternion) -> float:
    """Compute attitude error in degrees."""
    q_err = q_est.conjugate() @ q_true
    q_err = q_err.normalize()
    theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
    return np.rad2deg(theta)


# =============================================================================
# Scenario Modifiers
# =============================================================================

def apply_baseline(sim_data):
    """No modification."""
    class ModifiedData:
        pass
    modified = ModifiedData()
    for attr in ['t', 'jd', 'q_true', 'omega_meas', 'mag_meas', 'sun_meas', 'st_meas', 'b_eci', 's_eci']:
        setattr(modified, attr, getattr(sim_data, attr).copy())
    return modified


def apply_eclipse(sim_data):
    """Sun sensor unavailable throughout."""
    modified = apply_baseline(sim_data)
    for k in range(len(modified.t)):
        modified.sun_meas[k] = np.array([np.nan, np.nan, np.nan])
    return modified


def apply_rapid_tumbling(sim_data):
    """No star tracker (rapid tumbling)."""
    modified = apply_baseline(sim_data)
    for k in range(len(modified.t)):
        modified.st_meas[k] = np.array([np.nan, np.nan, np.nan, np.nan])
    return modified


def apply_mag_bias(sim_data, bias_magnitude: float = 0.62):
    """Constant magnetometer bias."""
    modified = apply_baseline(sim_data)
    bias_dir = np.array([1.0, 0.5, 0.3])
    bias_dir = bias_dir / np.linalg.norm(bias_dir)
    bias = bias_magnitude * bias_dir

    for k in range(len(modified.t)):
        if not np.isnan(modified.mag_meas[k, 0]):
            modified.mag_meas[k] = modified.mag_meas[k] + bias
    return modified


# =============================================================================
# Estimator Runners
# =============================================================================

CONFIG_PATH = "configs/config_baseline_short.yaml"

def run_eskf(sim_data) -> float:
    """Run ESKF and return mean steady-state error."""
    P0 = np.diag([np.deg2rad(5)**2]*3 + [1e-6]*3)
    eskf = ESKF(P0=P0, config_path=CONFIG_PATH)

    q_init = Quaternion.from_array(sim_data.q_true[0])
    x = EskfState(
        nom=NominalState(ori=q_init.copy(), gyro_bias=np.zeros(3)),
        err=MultiVarGauss(mean=np.zeros(6), cov=P0.copy())
    )

    errors = []
    dt = sim_data.t[1] - sim_data.t[0]

    for k in range(len(sim_data.t)):
        q_true = Quaternion.from_array(sim_data.q_true[k])
        errors.append(compute_attitude_error_deg(x.nom.ori, q_true))

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

    errors = np.array(errors)
    mask = sim_data.t >= sim_data.t[-1] * 0.5
    return np.mean(errors[mask])


def run_smoother(sim_data) -> float:
    """Run smoother (Cauchy k=0.1) and return mean steady-state error."""
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
    errors = []

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

        errors.append(compute_attitude_error_deg(q_interp, q_true))

    errors = np.array(errors)
    mask = sim_data.t >= sim_data.t[-1] * 0.5
    return np.mean(errors[mask])


def run_redundant(sim_data) -> float:
    """Run redundant estimator and return mean steady-state error."""
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

    errors = []
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

        errors.append(compute_attitude_error_deg(q_est, q_true))

    errors = np.array(errors)
    mask = sim_data.t >= sim_data.t[-1] * 0.5
    return np.mean(errors[mask])


def main():
    db_path = "simulations.db"
    n_runs = 10

    # Get run IDs
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT ?", (n_runs,))
    run_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    db = SimulationDatabase(db_path)

    scenarios = [
        ("Baseline", apply_baseline),
        ("Eclipse", apply_eclipse),
        ("Rapid Tumbling", apply_rapid_tumbling),
        ("Mag Bias (0.62)", apply_mag_bias),
    ]

    estimators = [
        ("ESKF", run_eskf),
        ("Smoother", run_smoother),
        ("Redundant", run_redundant),
    ]

    print("=" * 80)
    print("SCENARIO COMPARISON: ESKF vs Smoother (Cauchy k=0.1) vs Redundant")
    print("=" * 80)
    print(f"Runs per scenario: {n_runs}")
    print()

    results = {s[0]: {e[0]: [] for e in estimators} for s in scenarios}

    start_time = time.time()

    for scenario_name, scenario_fn in scenarios:
        print(f"\n{scenario_name}:")

        for i, run_id in enumerate(run_ids):
            print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
            sim_data = db.load_run(run_id)
            modified_data = scenario_fn(sim_data)

            for est_name, est_fn in estimators:
                try:
                    err = est_fn(modified_data)
                    results[scenario_name][est_name].append(err)
                except Exception as e:
                    results[scenario_name][est_name].append(np.nan)

            print("done")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - Mean Attitude Error [deg]")
    print("=" * 80)
    print(f"{'Scenario':<20} {'ESKF':>15} {'Smoother':>15} {'Redundant':>15}")
    print("-" * 65)

    for scenario_name, _ in scenarios:
        row = f"{scenario_name:<20}"
        for est_name, _ in estimators:
            vals = [v for v in results[scenario_name][est_name] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row += f" {mean:>6.3f} +/- {std:.3f}"
            else:
                row += f" {'FAILED':>15}"
        print(row)

    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Scenario comparison: ESKF vs Smoother (Cauchy $k$=0.1) vs Redundant ($N=""" + str(n_runs) + r"""$ runs)}
\label{tab:scenario_comparison}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Scenario} & \textbf{ESKF} & \textbf{Smoother} & \textbf{Redundant} \\
\midrule
"""

    for scenario_name, _ in scenarios:
        latex += f"{scenario_name} "
        for est_name, _ in estimators:
            vals = [v for v in results[scenario_name][est_name] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                latex += f"& ${mean:.3f}^\\circ \\pm {std:.3f}^\\circ$ "
            else:
                latex += "& -- "
        latex += r"\\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(latex)

    with open("scenario_comparison.tex", "w") as f:
        f.write(latex)
    print("\nSaved: scenario_comparison.tex")


if __name__ == "__main__":
    main()
