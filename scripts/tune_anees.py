#!/usr/bin/env python3
"""
ANEES tuning script - find optimal noise_scale parameters.

Goal: Find noise scales that give ANEES distribution close to:
  - 25% below 25th percentile
  - 50% within 25-75th percentile
  - 25% above 75th percentile
"""

import numpy as np
from scipy.stats import chi2
from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
from estimation.eskf import ESKF
import yaml
import copy


def run_eskf_anees(sim, config_path: str, mag_scale: float, sun_scale: float,
                   star_scale: float, gyro_scale: float = 1.0,
                   skip_first: int = 1000, verbose: bool = False):
    """
    Run ESKF and compute ANEES statistics.

    Returns dict with ANEES statistics.
    """
    # Load and modify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set noise scales
    config['sensors']['mag']['scaling'] = {'noise_scale': mag_scale}
    config['sensors']['sun']['scaling'] = {'noise_scale': sun_scale}
    config['sensors']['star']['scaling'] = {'noise_scale': star_scale}
    config['sensors']['gyro']['scaling'] = {'noise_scale': gyro_scale}

    # Write temp config
    temp_config = '/tmp/temp_config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Initialize ESKF
    P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
    eskf = ESKF(P0=P0, config_path=temp_config, chi2_threshold=1e10)

    # Initial state
    q0_true = Quaternion.from_array(sim.q_true[0])
    q0_est = Quaternion.from_avec(np.array([0.01, 0.01, 0.01])).multiply(q0_true).normalize()

    nom0 = NominalState(ori=q0_est, gyro_bias=np.zeros(3))
    err0 = MultiVarGauss(np.zeros(6), P0)
    x_est = EskfState(nom=nom0, err=err0)

    nees_vals = []
    N = len(sim.t)

    for k in range(1, N):
        omega_k = sim.omega_meas[k]
        dt_k = sim.t[k] - sim.t[k-1]
        x_est = eskf.predict(x_est, omega_k, dt_k)

        # Magnetometer update
        mag_meas = sim.mag_meas[k]
        if not np.any(np.isnan(mag_meas)):
            try:
                x_est = eskf.update(x_est, mag_meas, SensorType.MAGNETOMETER, B_n=sim.b_eci[k])
            except:
                pass

        # Sun sensor update
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            try:
                x_est = eskf.update(x_est, sun_meas, SensorType.SUN_VECTOR, s_n=sim.s_eci[k])
            except:
                pass

        # Star tracker update
        st_meas = sim.st_meas[k]
        if not np.any(np.isnan(st_meas)):
            try:
                q_meas = Quaternion.from_array(st_meas)
                x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)
            except:
                pass

        # Compute NEES
        q_true = Quaternion.from_array(sim.q_true[k])
        q_err = q_true.multiply(x_est.nom.ori.conjugate()).normalize()

        theta = 2.0 * np.arccos(np.clip(abs(q_err.mu), 0.0, 1.0))
        if theta < 1e-8:
            att_err = np.zeros(3)
        else:
            att_err = theta * q_err.eta / np.sin(theta / 2.0)

        bias_err = x_est.nom.gyro_bias - sim.b_g_true[k]
        delta_x = np.concatenate([att_err, bias_err])

        try:
            nees = delta_x.T @ np.linalg.solve(x_est.err.cov, delta_x)
        except:
            nees = np.inf
        nees_vals.append(nees)

    # Compute statistics (steady state only)
    nees_arr = np.array(nees_vals[skip_first:])
    n = 6
    lower_bound = chi2.ppf(0.25, n)
    upper_bound = chi2.ppf(0.75, n)

    below = np.sum(nees_arr < lower_bound) / len(nees_arr) * 100
    within = np.sum((nees_arr >= lower_bound) & (nees_arr <= upper_bound)) / len(nees_arr) * 100
    above = np.sum(nees_arr > upper_bound) / len(nees_arr) * 100

    return {
        'mean_nees': np.mean(nees_arr),
        'median_nees': np.median(nees_arr),
        'below_25': below,
        'within': within,
        'above_75': above,
        'score': abs(within - 50) + abs(below - 25) + abs(above - 25)  # Lower is better
    }


def grid_search(sim, config_path: str):
    """Grid search for optimal noise scales."""

    # Current values that give NEES ≈ 9 (overconfident)
    # Need to increase R (measurement noise) to make filter less confident

    # Search ranges (multiplicative factors on current values)
    mag_scales = [5.0, 7.5, 10.0, 12.5, 15.0]
    sun_scales = [5.0, 7.5, 10.0, 12.5, 15.0]
    star_scales = [1.0, 2.0, 3.0, 5.0]

    best_score = float('inf')
    best_params = None
    results = []

    total = len(mag_scales) * len(sun_scales) * len(star_scales)
    count = 0

    print(f"Grid search over {total} combinations...")
    print()

    for mag_s in mag_scales:
        for sun_s in sun_scales:
            for star_s in star_scales:
                count += 1
                print(f"[{count}/{total}] mag={mag_s}, sun={sun_s}, star={star_s}...", end=" ", flush=True)

                result = run_eskf_anees(sim, config_path, mag_s, sun_s, star_s)
                result['mag_scale'] = mag_s
                result['sun_scale'] = sun_s
                result['star_scale'] = star_s
                results.append(result)

                print(f"NEES={result['mean_nees']:.2f}, within={result['within']:.1f}%, score={result['score']:.1f}")

                if result['score'] < best_score:
                    best_score = result['score']
                    best_params = (mag_s, sun_s, star_s)

    print()
    print("="*70)
    print("BEST PARAMETERS")
    print("="*70)
    print(f"  mag_scale:  {best_params[0]}")
    print(f"  sun_scale:  {best_params[1]}")
    print(f"  star_scale: {best_params[2]}")
    print()

    # Run with best params and show details
    result = run_eskf_anees(sim, config_path, best_params[0], best_params[1], best_params[2])
    print("NEES Statistics:")
    print(f"  Mean NEES:  {result['mean_nees']:.2f} (expected: 6)")
    print(f"  Below 25th: {result['below_25']:.1f}% (expected: 25%)")
    print(f"  Within:     {result['within']:.1f}% (expected: 50%)")
    print(f"  Above 75th: {result['above_75']:.1f}% (expected: 25%)")

    return best_params, results


def fine_tune(sim, config_path: str, mag_center: float, sun_center: float, star_center: float):
    """Fine-tune around best parameters."""

    # Fine search around center values
    mag_scales = [mag_center * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
    sun_scales = [sun_center * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
    star_scales = [star_center * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]

    best_score = float('inf')
    best_params = None

    total = len(mag_scales) * len(sun_scales) * len(star_scales)
    count = 0

    print(f"Fine-tuning over {total} combinations...")
    print()

    for mag_s in mag_scales:
        for sun_s in sun_scales:
            for star_s in star_scales:
                count += 1
                result = run_eskf_anees(sim, config_path, mag_s, sun_s, star_s)

                if count % 25 == 0:
                    print(f"[{count}/{total}] Best so far: score={best_score:.1f}")

                if result['score'] < best_score:
                    best_score = result['score']
                    best_params = (mag_s, sun_s, star_s)

    print()
    print("="*70)
    print("FINE-TUNED PARAMETERS")
    print("="*70)
    print(f"  mag_scale:  {best_params[0]:.2f}")
    print(f"  sun_scale:  {best_params[1]:.2f}")
    print(f"  star_scale: {best_params[2]:.2f}")

    return best_params


def main():
    # Load simulation data
    db = SimulationDatabase('simulations.db')
    sim = db.load_run(1)
    config_path = 'configs/config_baseline_short.yaml'

    print(f"Loaded simulation with {len(sim.t)} samples ({sim.t[-1]:.1f}s)")
    print()

    # Coarse grid search
    best_params, results = grid_search(sim, config_path)

    # Fine tune
    print()
    print("="*70)
    print("FINE TUNING")
    print("="*70)
    final_params = fine_tune(sim, config_path, best_params[0], best_params[1], best_params[2])

    # Final verification
    print()
    print("="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    result = run_eskf_anees(sim, config_path, final_params[0], final_params[1], final_params[2])
    print(f"  mag_scale:  {final_params[0]:.2f}")
    print(f"  sun_scale:  {final_params[1]:.2f}")
    print(f"  star_scale: {final_params[2]:.2f}")
    print()
    print("NEES Statistics:")
    print(f"  Mean NEES:  {result['mean_nees']:.2f} (expected: 6)")
    print(f"  Below 25th: {result['below_25']:.1f}% (expected: 25%)")
    print(f"  Within:     {result['within']:.1f}% (expected: 50%)")
    print(f"  Above 75th: {result['above_75']:.1f}% (expected: 25%)")

    # Update config suggestion
    print()
    print("="*70)
    print("SUGGESTED CONFIG UPDATE")
    print("="*70)
    print(f"""
sensors:
  mag:
    scaling:
      noise_scale: {final_params[0]:.1f}
  sun:
    scaling:
      noise_scale: {final_params[1]:.1f}
  star:
    scaling:
      noise_scale: {final_params[2]:.1f}
""")


if __name__ == "__main__":
    main()
