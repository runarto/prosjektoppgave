#!/usr/bin/env python3
"""
Comprehensive scenario validation script.

Validates simulation data quality, measurement availability, and anomaly configurations.
"""

import numpy as np
from data.db import SimulationDatabase
from utilities.quaternion import Quaternion
from typing import Dict, List, Tuple
import sys


class ValidationResult:
    """Store validation results with severity levels."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.pass_checks: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def add_pass(self, msg: str):
        self.pass_checks.append(msg)

    def print_summary(self, scenario_name: str):
        """Print formatted validation summary."""
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY: {scenario_name}")
        print(f"{'='*70}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  - {msg}")

        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  - {msg}")

        if self.info:
            print(f"\nℹ INFO ({len(self.info)}):")
            for msg in self.info:
                print(f"  - {msg}")

        if self.pass_checks:
            print(f"\n✓ PASSED ({len(self.pass_checks)}):")
            for msg in self.pass_checks:
                print(f"  - {msg}")

        # Overall status
        print(f"\n{'='*70}")
        if self.errors:
            print("STATUS: FAILED ❌")
            return False
        elif self.warnings:
            print("STATUS: PASSED WITH WARNINGS ⚠")
            return True
        else:
            print("STATUS: PASSED ✓")
            return True


def validate_data_quality(sim, result: ValidationResult):
    """Validate basic data quality (no NaN/Inf, quaternion normalization)."""

    # Check quaternion normalization
    q_norms = np.linalg.norm(sim.q_true, axis=1)
    max_norm_err = np.max(np.abs(q_norms - 1.0))

    if max_norm_err > 1e-3:
        result.add_error(f"Quaternion norm error too large: {max_norm_err:.2e} (max should be < 1e-3)")
    elif max_norm_err > 1e-6:
        result.add_warning(f"Quaternion norm error: {max_norm_err:.2e} (acceptable but could be better)")
    else:
        result.add_pass(f"Quaternion normalization: {max_norm_err:.2e}")

    # Check for NaN/Inf in truth data
    has_nan_q = np.any(~np.isfinite(sim.q_true))
    has_nan_omega = np.any(~np.isfinite(sim.omega_true))
    has_nan_bg = np.any(~np.isfinite(sim.b_g_true))

    if has_nan_q or has_nan_omega or has_nan_bg:
        result.add_error("Truth data contains NaN/Inf values")
    else:
        result.add_pass("Truth data has no NaN/Inf values")

    # Check gyro bias magnitude
    bg_norms = np.linalg.norm(sim.b_g_true, axis=1)
    final_bias_deg_h = np.rad2deg(bg_norms[-1] * 3600)

    if final_bias_deg_h < 0.01:
        result.add_warning(f"Gyro bias very small: {final_bias_deg_h:.4f} deg/h (should be ~0.5-10 deg/h for realism)")
    elif final_bias_deg_h > 100:
        result.add_warning(f"Gyro bias very large: {final_bias_deg_h:.1f} deg/h (may be unrealistic)")
    else:
        result.add_pass(f"Gyro bias magnitude: {final_bias_deg_h:.2f} deg/h")


def validate_measurement_availability(sim, result: ValidationResult):
    """Validate measurement availability rates."""
    N = len(sim.t)

    n_mag = np.sum(~np.isnan(sim.mag_meas[:, 0]))
    n_sun = np.sum(~np.isnan(sim.sun_meas[:, 0]))
    n_st = np.sum(~np.isnan(sim.st_meas[:, 0]))

    mag_rate = 100 * n_mag / N
    sun_rate = 100 * n_sun / N
    st_rate = 100 * n_st / N

    result.add_info(f"Gyro: {N}/{N} (100.0%)")
    result.add_info(f"Magnetometer: {n_mag}/{N} ({mag_rate:.1f}%)")
    result.add_info(f"Sun sensor: {n_sun}/{N} ({sun_rate:.1f}%)")
    result.add_info(f"Star tracker: {n_st}/{N} ({st_rate:.1f}%)")

    # Check for missing measurements
    if mag_rate < 5:
        result.add_error(f"Magnetometer availability too low: {mag_rate:.1f}%")
    elif mag_rate < 10:
        result.add_warning(f"Magnetometer availability low: {mag_rate:.1f}%")
    else:
        result.add_pass(f"Magnetometer availability: {mag_rate:.1f}%")

    if sun_rate == 0:
        result.add_error("Sun sensor has ZERO measurements (check FOV or initial attitude)")
    elif sun_rate < 5:
        result.add_warning(f"Sun sensor availability very low: {sun_rate:.1f}%")
    elif sun_rate > 80:
        result.add_warning(f"Sun sensor availability very high: {sun_rate:.1f}% (might indicate no FOV constraints)")
    else:
        result.add_pass(f"Sun sensor availability: {sun_rate:.1f}%")

    if st_rate < 1:
        result.add_warning(f"Star tracker availability very low: {st_rate:.1f}%")
    elif st_rate > 80:
        result.add_warning(f"Star tracker availability very high: {st_rate:.1f}% (might indicate no rate constraints)")
    else:
        result.add_pass(f"Star tracker availability: {st_rate:.1f}%")


def validate_sun_sensor_fov(sim, result: ValidationResult):
    """Validate sun sensor FOV constraints."""

    # Sample points throughout simulation
    n_samples = min(100, len(sim.t))
    indices = np.linspace(0, len(sim.t)-1, n_samples, dtype=int)

    z_components = []

    for k in indices:
        q_true = Quaternion.from_array(sim.q_true[k])
        s_eci = sim.s_eci[k]

        # Rotate sun vector to body frame
        R_nb = q_true.as_rotmat()
        R_bn = R_nb.T
        s_body = R_bn @ s_eci

        # Z-component (boresight direction)
        z_components.append(s_body[2])

    z_components = np.array(z_components)

    # FOV check: 120° FOV means half-angle = 60°, so cos(60°) = 0.5
    fov_half_angle_deg = 60.0
    cos_fov_half = np.cos(np.deg2rad(fov_half_angle_deg))

    in_fov = z_components > cos_fov_half
    fov_rate = 100 * np.sum(in_fov) / len(z_components)

    result.add_info(f"Sun in FOV: {np.sum(in_fov)}/{len(z_components)} samples ({fov_rate:.1f}%)")

    if fov_rate == 0:
        result.add_error(f"Sun NEVER in FOV! Max Z-component: {np.max(z_components):.3f} (need > {cos_fov_half:.3f})")
        result.add_info(f"  → FIX: Adjust initial attitude or reduce tumbling amplitude")
    elif fov_rate < 10:
        result.add_warning(f"Sun rarely in FOV: {fov_rate:.1f}%")
    else:
        result.add_pass(f"Sun FOV check: {fov_rate:.1f}% of samples")


def validate_star_tracker_rate(sim, result: ValidationResult):
    """Validate star tracker rate constraints."""

    # Check angular rate vs star tracker availability
    omega_norms = np.linalg.norm(sim.omega_true, axis=1)
    omega_deg_s = omega_norms * 180 / np.pi

    st_available = ~np.isnan(sim.st_meas[:, 0])

    if np.sum(st_available) > 0:
        omega_during_st = omega_deg_s[st_available]
        max_omega_during_st = np.max(omega_during_st)

        # Typical threshold is 2-5 deg/s
        if max_omega_during_st > 5.0:
            result.add_warning(f"Star tracker measurements during high rate: max {max_omega_during_st:.1f} deg/s (threshold usually 2-5 deg/s)")
        else:
            result.add_pass(f"Star tracker rate constraint: max {max_omega_during_st:.1f} deg/s during measurements")

        result.add_info(f"Overall angular rate: mean={np.mean(omega_deg_s):.1f} deg/s, max={np.max(omega_deg_s):.1f} deg/s")
    else:
        result.add_info(f"No star tracker measurements to validate rate constraint")


def validate_magnetometer_bias(sim, result: ValidationResult):
    """Validate magnetometer bias presence."""

    # Check if measurements show constant bias
    mag_available = ~np.isnan(sim.mag_meas[:, 0])

    if np.sum(mag_available) > 100:
        mag_meas_valid = sim.mag_meas[mag_available]

        # Check for field-aligned component (hard-iron should shift the mean)
        mag_mean = np.mean(mag_meas_valid, axis=0)
        mag_mean_norm = np.linalg.norm(mag_mean)

        # Expected Earth field is ~50 μT = 5e-5 T
        # Hard-iron bias should be ~100-1000 nT = 1e-7 to 1e-6 T
        # So bias/field ratio should be ~0.2-2%

        if mag_mean_norm < 1e-8:
            result.add_warning("Magnetometer bias appears to be zero or very small")
        else:
            result.add_info(f"Magnetometer mean measurement: {mag_mean_norm:.2e} T")
            result.add_pass("Magnetometer bias present")


def validate_scenario(db: SimulationDatabase, run_id: int, scenario_name: str = None) -> bool:
    """
    Validate a single scenario comprehensively.

    Returns:
        True if validation passed (possibly with warnings), False if failed
    """
    result = ValidationResult()

    # Get scenario name from database if not provided
    if scenario_name is None:
        cur = db.conn.cursor()
        cur.execute("SELECT name FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        scenario_name = row[0] if row else f"Run {run_id}"

    print(f"\n{'='*70}")
    print(f"VALIDATING: {scenario_name} (run_id={run_id})")
    print(f"{'='*70}")

    try:
        sim = db.load_run(run_id)
    except Exception as e:
        result.add_error(f"Failed to load simulation: {e}")
        result.print_summary(scenario_name)
        return False

    # Run all validation checks
    validate_data_quality(sim, result)
    validate_measurement_availability(sim, result)
    validate_sun_sensor_fov(sim, result)
    validate_star_tracker_rate(sim, result)
    validate_magnetometer_bias(sim, result)

    # Print summary
    passed = result.print_summary(scenario_name)

    return passed


def main():
    """Validate all scenarios in database."""
    print("="*70)
    print("COMPREHENSIVE SCENARIO VALIDATION")
    print("="*70)

    db = SimulationDatabase("simulations.db")

    # Get all runs from database
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs ORDER BY id")
    runs = cur.fetchall()

    if not runs:
        print("\n⚠ No simulation runs found in database!")
        print("\nGenerate scenarios first:")
        print("  python -m data.generate_scenarios_short")
        return 1

    print(f"\nFound {len(runs)} simulation runs:")
    for run_id, name in runs:
        print(f"  [{run_id}] {name}")

    # Validate each run
    results = {}
    for run_id, name in runs:
        passed = validate_scenario(db, run_id, name)
        results[run_id] = passed

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*70}")

    n_passed = sum(1 for passed in results.values() if passed)
    n_failed = len(results) - n_passed

    print(f"\nTotal scenarios: {len(results)}")
    print(f"  Passed: {n_passed}")
    print(f"  Failed: {n_failed}")

    if n_failed > 0:
        print(f"\n❌ {n_failed} scenario(s) failed validation")
        print("\nFailed scenarios:")
        for run_id, passed in results.items():
            if not passed:
                cur.execute("SELECT name FROM runs WHERE id = ?", (run_id,))
                name = cur.fetchone()[0]
                print(f"  [{run_id}] {name}")

        print("\nRecommended actions:")
        print("  1. Check config files for sensor parameters (FOV, max_rate, biases)")
        print("  2. Adjust initial attitude (q0) to point sensors toward references")
        print("  3. Reduce tumbling amplitude if sensors are always outside constraints")
        print("  4. Re-generate scenarios: python -m data.generate_scenarios_short")

        return 1
    else:
        print(f"\n✓ All scenarios passed validation!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
