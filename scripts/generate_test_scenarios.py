#!/usr/bin/env python3
"""
Generate multiple test scenario datasets for robustness testing.

This script creates:
1. Spike scenario - Frequent measurement outliers from all sensors
2. Anomaly scenario - Freezes, eclipses, and high spin rates
"""

import sys
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.utils import load_yaml


def generate_spike_scenario():
    """Generate dataset with frequent measurement spikes."""
    print("\n" + "="*70)
    print("GENERATING SPIKE SCENARIO")
    print("="*70)

    config = load_yaml("configs/config_spikes.yaml")
    generator = EnhancedAttitudeDataGenerator(
        db_path="simulations.db",
        config_path="configs/config_spikes.yaml"
    )

    sim_cfg = SimulationConfig(
        T=config["time"]["sim_T"],
        dt=config["time"]["sim_dt"],
        start_jd=config["time"]["start_jd"],
        run_name=config["simulation"]["run_name"],
        gyro_noise_scale=1.0,
        mag_noise_scale=1.0,
        sun_noise_scale=1.0,
    )

    run_id = generator.run(cfg=sim_cfg)
    print(f"\n✓ Spike scenario generated successfully (run_id={run_id})")
    return run_id


def generate_anomaly_scenario():
    """Generate dataset with measurement anomalies."""
    print("\n" + "="*70)
    print("GENERATING ANOMALY SCENARIO")
    print("="*70)

    config = load_yaml("configs/config_anomalies.yaml")
    generator = EnhancedAttitudeDataGenerator(
        db_path="simulations.db",
        config_path="configs/config_anomalies.yaml"
    )

    sim_cfg = SimulationConfig(
        T=config["time"]["sim_T"],
        dt=config["time"]["sim_dt"],
        start_jd=config["time"]["start_jd"],
        run_name=config["simulation"]["run_name"],
        gyro_noise_scale=1.0,
        mag_noise_scale=1.0,
        sun_noise_scale=1.0,
    )

    run_id = generator.run(cfg=sim_cfg)
    print(f"\n✓ Anomaly scenario generated successfully (run_id={run_id})")
    return run_id


def generate_perfect_scenario():
    """Generate near-perfect measurement scenario for verification."""
    print("\n" + "="*70)
    print("GENERATING PERFECT MEASUREMENT SCENARIO")
    print("="*70)

    config = load_yaml("configs/config_perfect.yaml")
    generator = EnhancedAttitudeDataGenerator(
        db_path="simulations.db",
        config_path="configs/config_perfect.yaml"
    )

    sim_cfg = SimulationConfig(
        T=config["time"]["sim_T"],
        dt=config["time"]["sim_dt"],
        start_jd=config["time"]["start_jd"],
        run_name=config["simulation"]["run_name"],
        gyro_noise_scale=1.0,
        mag_noise_scale=1.0,
        sun_noise_scale=1.0,
    )

    run_id = generator.run(cfg=sim_cfg)
    print(f"\n✓ Perfect scenario generated successfully (run_id={run_id})")
    return run_id


def generate_baseline_scenario():
    """Generate baseline scenario with fixed config (for comparison)."""
    print("\n" + "="*70)
    print("GENERATING BASELINE SCENARIO (with fixes)")
    print("="*70)
    
    config = load_yaml("configs/config_baseline.yaml")

    generator = EnhancedAttitudeDataGenerator(
        db_path="simulations.db",
        config_path="configs/config_baseline.yaml"
    )

    sim_cfg = SimulationConfig(
        T=config["time"]["sim_T"],
        dt=config["time"]["sim_dt"],
        start_jd=config["time"]["start_jd"],
        run_name=config["simulation"]["run_name"],
        gyro_noise_scale=1.0,
        mag_noise_scale=1.0,
        sun_noise_scale=1.0,
    )

    run_id = generator.run(cfg=sim_cfg)
    print(f"\n✓ Baseline scenario generated successfully (run_id={run_id})")

    return run_id


def main():
    """Generate all test scenarios."""
    print("="*70)
    print("TEST SCENARIO GENERATOR")
    print("="*70)
    print("\nThis script will generate four test datasets:")
    print("  1. Perfect - Near-perfect measurements for verification")
    print("  2. Baseline - Normal operation with fixed config")
    print("  3. Spikes - Frequent measurement outliers")
    print("  4. Anomalies - Freezes, eclipses, high spin rates")
    print()

    results = {}

    # Generate perfect scenario
    try:
        results['perfect'] = generate_perfect_scenario()
    except Exception as e:
        print(f"\n✗ Perfect scenario failed: {e}")
        import traceback
        traceback.print_exc()
        results['perfect'] = None

    # Generate baseline
    try:
        results['baseline'] = generate_baseline_scenario()
    except Exception as e:
        print(f"\n✗ Baseline scenario failed: {e}")
        import traceback
        traceback.print_exc()
        results['baseline'] = None

    # Generate spike scenario
    try:
        results['spikes'] = generate_spike_scenario()
    except Exception as e:
        print(f"\n✗ Spike scenario failed: {e}")
        import traceback
        traceback.print_exc()
        results['spikes'] = None

    # Generate anomaly scenario
    try:
        results['anomalies'] = generate_anomaly_scenario()
    except Exception as e:
        print(f"\n✗ Anomaly scenario failed: {e}")
        import traceback
        traceback.print_exc()
        results['anomalies'] = None

    # Summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)

    for name, run_id in results.items():
        if run_id is not None:
            print(f"  {name:15s}: ✓ run_id={run_id}")
        else:
            print(f"  {name:15s}: ✗ FAILED")

    # Check database
    from data.db import SimulationDatabase
    db = SimulationDatabase("simulations.db")
    cur = db.conn.cursor()
    cur.execute("SELECT id, name FROM runs ORDER BY id;")
    runs = cur.fetchall()

    print("\n" + "="*70)
    print("ALL RUNS IN DATABASE")
    print("="*70)
    for run_id, name in runs:
        cur.execute("SELECT COUNT(*) FROM samples WHERE run_id=?;", (run_id,))
        count = cur.fetchone()[0]
        print(f"  [{run_id}] {name:30s} - {count:5d} samples")

    successful = sum(1 for v in results.values() if v is not None)
    total = len(results)

    if successful == total:
        print("\n" + "="*70)
        print(f"✓ ALL {total} SCENARIOS GENERATED SUCCESSFULLY")
        print("="*70)
        print("\nYou can now test with:")
        print(f"  python test_gtsam_with_db.py  # Perfect: run_id={results.get('perfect', 'N/A')}")
        print(f"  python test_gtsam_with_db.py  # Baseline: run_id={results.get('baseline', 'N/A')}")
        print(f"  python test_eskf_with_db.py   # Spikes: run_id={results.get('spikes', 'N/A')}")
        return 0
    else:
        print("\n" + "="*70)
        print(f"⚠ {successful}/{total} scenarios generated successfully")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
