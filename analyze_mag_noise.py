#!/usr/bin/env python3
"""
Analyze actual magnetometer measurement noise to find correct R matrix.
"""

import numpy as np
from data.db import SimulationDatabase
from utilities.quaternion import Quaternion


def main():
    print("="*70)
    print("MAGNETOMETER NOISE ANALYSIS")
    print("="*70)

    # Load baseline scenario
    db = SimulationDatabase("simulations.db")
    sim = db.load_run(1)

    print(f"\nScenario: Baseline")
    print(f"Duration: {sim.t[-1]:.1f}s, Samples: {len(sim.t)}")

    # Compute measurement residuals
    mag_residuals = []

    for k in range(len(sim.t)):
        mag_meas = sim.mag_meas[k]
        b_eci = sim.b_eci[k]
        q_true = Quaternion.from_array(sim.q_true[k])

        # Skip NaN measurements
        if np.any(np.isnan(mag_meas)) or np.any(np.isnan(b_eci)):
            continue

        # Compute expected measurement (perfect knowledge of attitude)
        # mag_expected = R(q_true) * b_eci
        b_body_expected = q_true.rotate(b_eci)

        # Normalize both (since filter normalizes)
        mag_meas_norm = mag_meas / np.linalg.norm(mag_meas)
        b_body_expected_norm = b_body_expected / np.linalg.norm(b_body_expected)

        # Residual = measurement - expected
        residual = mag_meas_norm - b_body_expected_norm
        mag_residuals.append(residual)

    mag_residuals = np.array(mag_residuals)  # Shape: (N, 3)

    print(f"\n{len(mag_residuals)} valid magnetometer measurements analyzed")

    # Compute statistics
    print("\n" + "="*70)
    print("MAGNETOMETER RESIDUAL STATISTICS")
    print("="*70)

    # Component-wise statistics
    for i, axis in enumerate(['X', 'Y', 'Z']):
        residuals_i = mag_residuals[:, i]
        print(f"\n{axis}-axis:")
        print(f"  Mean:   {np.mean(residuals_i):.6e}")
        print(f"  Std:    {np.std(residuals_i):.6e}")
        print(f"  Min:    {np.min(residuals_i):.6e}")
        print(f"  Max:    {np.max(residuals_i):.6e}")

    # Overall statistics
    residual_norms = np.linalg.norm(mag_residuals, axis=1)
    print(f"\nResidual norm:")
    print(f"  Mean:   {np.mean(residual_norms):.6e}")
    print(f"  Median: {np.median(residual_norms):.6e}")
    print(f"  Std:    {np.std(residual_norms):.6e}")
    print(f"  95%:    {np.percentile(residual_norms, 95):.6e}")
    print(f"  99%:    {np.percentile(residual_norms, 99):.6e}")
    print(f"  Max:    {np.max(residual_norms):.6e}")

    # Estimate appropriate mag_std
    # Use 99th percentile of component-wise std
    component_stds = np.std(mag_residuals, axis=0)
    recommended_std = np.max(component_stds) * 3  # 3-sigma coverage

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"\nComponent-wise standard deviations:")
    print(f"  σ_x = {component_stds[0]:.6e}")
    print(f"  σ_y = {component_stds[1]:.6e}")
    print(f"  σ_z = {component_stds[2]:.6e}")

    print(f"\nCurrent filter setting: mag_std = 5.0e-07")
    print(f"Recommended setting:    mag_std = {recommended_std:.6e} (3σ of max component)")
    print(f"Conservative setting:   mag_std = {np.percentile(residual_norms, 99):.6e} (99th percentile)")

    # Check if measurements are actually unbiased
    mean_residual = np.mean(mag_residuals, axis=0)
    mean_norm = np.linalg.norm(mean_residual)
    print(f"\nMean residual vector: [{mean_residual[0]:.3e}, {mean_residual[1]:.3e}, {mean_residual[2]:.3e}]")
    print(f"Mean residual norm: {mean_norm:.6e}")
    if mean_norm < recommended_std:
        print("✓ Measurements appear unbiased (mean << std)")
    else:
        print("⚠️  WARNING: Measurements may have systematic bias!")


if __name__ == "__main__":
    main()
