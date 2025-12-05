#!/usr/bin/env python3
"""
Generate SHORT test scenarios (5 minutes) for fast ESKF vs FGO comparison.

These datasets are designed for:
- Algorithm comparison
- Parameter tuning
- Plot generation
- Rapid iteration

For long-term stability analysis, use the 50-minute datasets with ESKF only.
"""

import sys
from pathlib import Path
import yaml
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.utils import load_yaml


def create_short_config(base_config_path: str, output_path: str, duration: float = 300.0):
    """Create a short version of a config file."""
    config = load_yaml(base_config_path)

    # Modify duration
    config['time']['sim_T'] = duration
    config['simulation']['run_name'] = config['simulation']['run_name'] + "_short"

    # Write to new file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Created: {output_path}")


def main():
    print("="*70)
    print("SHORT SCENARIO GENERATOR (5 minutes each)")
    print("="*70)

    # Base configs to convert
    base_configs = [
        ("configs/config_spacegrade_realistic.yaml", "configs/config_baseline_short.yaml"),
        ("configs/config_rapid_tumbling.yaml", "configs/config_rapid_tumbling_short.yaml"),
        ("configs/config_eclipse.yaml", "configs/config_eclipse_short.yaml"),
        ("configs/config_measurement_spikes.yaml", "configs/config_measurement_spikes_short.yaml"),
    ]

    # Create short config files
    print("\n1. Creating short config files...")
    for base_path, short_path in base_configs:
        if Path(base_path).exists():
            create_short_config(base_path, short_path, duration=300.0)
        else:
            print(f"  Warning: {base_path} not found, skipping")

    # Generate simulation data
    print("\n2. Generating short simulation datasets...")
    db_path = "simulations.db"
    run_ids = {}

    for _, short_path in base_configs:
        if not Path(short_path).exists():
            continue

        config = load_yaml(short_path)
        scenario_name = config['simulation']['run_name']

        print(f"\n  Generating: {scenario_name}")

        generator = EnhancedAttitudeDataGenerator(
            db_path=db_path,
            config_path=short_path
        )

        sim_cfg = SimulationConfig(
            T=config['time']['sim_T'],
            dt=config['time']['sim_dt'],
            start_jd=config['time']['start_jd']
        )

        print(f"    Duration: {sim_cfg.T:.0f}s, Timestep: {sim_cfg.dt}s")
        print(f"    Expected samples: {int(sim_cfg.T / sim_cfg.dt) + 1}")

        try:
            run_id = generator.run(sim_cfg)
            run_ids[scenario_name] = run_id
            print(f"    ✓ Generated with run_id={run_id}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SHORT SCENARIO GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated {len(run_ids)} short scenarios (5 minutes each):")
    for name, run_id in run_ids.items():
        print(f"  {run_id:3d} - {name}")

    print(f"\nThese datasets are optimized for:")
    print(f"  - Fast ESKF vs FGO comparison (~10 min runtime per scenario)")
    print(f"  - Algorithm testing and parameter tuning")
    print(f"  - Publication plot generation")

    print(f"\nNext steps:")
    print(f"  python tests/test_comparison.py  # Update run IDs to use short scenarios")
    print(f"  # Or test individually:")
    print(f"  python tests/test_manual_eskf.py")
    print(f"  python tests/test_fgo_fast.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
