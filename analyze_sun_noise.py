#!/usr/bin/env python3
"""
Analyze actual sun sensor measurement noise to find correct R matrix.
"""

import numpy as np
from data.db import SimulationDatabase
from utilities.quaternion import Quaternion


def main():
    print("="*70)
    print("SUN SENSOR NOISE ANALYSIS")
    print("="*70)

    # Load baseline scenario
    db = SimulationDatabase("simulations.db")
    sim = db.load_run(1)  # Updated database with fixed sun sensor noise

    print(f"\nScenario: Baseline")
    print(f"Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

    # Compute measurement residuals
    sun_residuals = []

    for k in range(len(sim.t)):
        sun_meas = sim.sun_meas[k]
        s_eci = sim.s_eci[k]
        q_true = Quaternion.from_array(sim.q_true[k])

        # Skip NaN measurements
        if np.any(np.isnan(sun_meas)) or np.any(np.isnan(s_eci)):
            continue

        # Compute expected measurement (perfect knowledge of attitude)
        # sun_expected = R(q_true) * s_eci
        s_body_expected = q_true.rotate(s_eci)

        # Check if measurements are normalized
        sun_meas_norm_val = np.linalg.norm(sun_meas)
        s_body_expected_norm_val = np.linalg.norm(s_body_expected)

        # Normalize both
        sun_meas_norm = sun_meas / sun_meas_norm_val
        s_body_expected_norm = s_body_expected / s_body_expected_norm_val

        # Residual = measurement - expected
        residual = sun_meas_norm - s_body_expected_norm
        sun_residuals.append(residual)

    sun_residuals = np.array(sun_residuals)  # Shape: (N, 3)

    print(f"\n{len(sun_residuals)} valid sun sensor measurements analyzed")

    # Compute statistics
    print("\n" + "="*70)
    print("SUN SENSOR RESIDUAL STATISTICS")
    print("="*70)

    # Check measurement norms
    sun_meas_norms = []
    for k in range(len(sim.t)):
        sun_meas = sim.sun_meas[k]
        if not np.any(np.isnan(sun_meas)):
            sun_meas_norms.append(np.linalg.norm(sun_meas))

    print(f"\nMeasurement norms:")
    print(f"  Min:  {np.min(sun_meas_norms):.6f}")
    print(f"  Max:  {np.max(sun_meas_norms):.6f}")
    print(f"  Mean: {np.mean(sun_meas_norms):.6f}")

    # Component-wise statistics
    for i, axis in enumerate(['X', 'Y', 'Z']):
        residuals_i = sun_residuals[:, i]
        print(f"\n{axis}-axis:")
        print(f"  Mean:   {np.mean(residuals_i):.6e}")
        print(f"  Std:    {np.std(residuals_i):.6e}")
        print(f"  Min:    {np.min(residuals_i):.6e}")
        print(f"  Max:    {np.max(residuals_i):.6e}")

    # Overall statistics
    residual_norms = np.linalg.norm(sun_residuals, axis=1)
    print(f"\nResidual norm:")
    print(f"  Mean:   {np.mean(residual_norms):.6e}")
    print(f"  Median: {np.median(residual_norms):.6e}")
    print(f"  Std:    {np.std(residual_norms):.6e}")
    print(f"  95%:    {np.percentile(residual_norms, 95):.6e}")
    print(f"  99%:    {np.percentile(residual_norms, 99):.6e}")
    print(f"  Max:    {np.max(residual_norms):.6e}")

    # Estimate appropriate sun_std
    component_stds = np.std(sun_residuals, axis=0)
    recommended_std = np.max(component_stds) * 3  # 3-sigma coverage

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"\nComponent-wise standard deviations:")
    print(f"  σ_x = {component_stds[0]:.6e}")
    print(f"  σ_y = {component_stds[1]:.6e}")
    print(f"  σ_z = {component_stds[2]:.6e}")

    print(f"\nCurrent filter setting: sun_std = 0.0001")
    print(f"Recommended setting:    sun_std = {recommended_std:.6e} (3σ of max component)")
    print(f"Conservative setting:   sun_std = {np.percentile(residual_norms, 99):.6e} (99th percentile)")

    # Check if measurements are actually unbiased
    mean_residual = np.mean(sun_residuals, axis=0)
    mean_norm = np.linalg.norm(mean_residual)
    print(f"\nMean residual vector: [{mean_residual[0]:.3e}, {mean_residual[1]:.3e}, {mean_residual[2]:.3e}]")
    print(f"Mean residual norm: {mean_norm:.6e}")
    if mean_norm < recommended_std:
        print("✓ Measurements appear unbiased (mean << std)")
    else:
        print("⚠️  WARNING: Measurements may have systematic bias!")


if __name__ == "__main__":
    main()
