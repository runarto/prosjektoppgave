#!/usr/bin/env python3
"""
Regenerate baseline scenario with corrected magnetometer noise for normalized measurements.

The original database used mag_std = 1e-7 Tesla, but measurements are stored as
normalized unit vectors. This caused chi-squared values of ~10^10.

The corrected mag_std = 0.002 is appropriate for normalized measurements (unitless),
corresponding to ~0.11° angular noise.
"""

import sys
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.utils import load_yaml

def main():
    print("="*70)
    print("REGENERATING BASELINE SCENARIO WITH CORRECTED MAG_STD")
    print("="*70)

    config_path = "configs/config_baseline_short.yaml"
    db_path = "simulations.db"

    # Load config
    config = load_yaml(config_path)

    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Duration: {config['time']['sim_T']:.1f}s")
    print(f"  Timestep: {config['time']['sim_dt']}s")
    print(f"  Magnetometer std: {config['sensors']['mag']['mag_std']}")
    print(f"  Expected samples: {int(config['time']['sim_T'] / config['time']['sim_dt']) + 1}")

    # Create generator
    print(f"\nInitializing generator...")
    generator = EnhancedAttitudeDataGenerator(
        db_path=db_path,
        config_path=config_path
    )

    # Create simulation config
    sim_cfg = SimulationConfig(
        T=config['time']['sim_T'],
        dt=config['time']['sim_dt'],
        start_jd=config['time']['start_jd'],
        run_name=config['simulation']['run_name']
    )

    # Generate data
    print(f"\nGenerating simulation data...")
    print(f"  This may take a minute...")

    try:
        run_id = generator.run(sim_cfg)
        print(f"\n✓ Successfully generated baseline scenario!")
        print(f"  Run ID: {run_id}")
        print(f"  Scenario: {config['simulation']['run_name']}")

        print(f"\nNext steps:")
        print(f"  1. Run diagnostics: python diagnose_eskf_divergence.py")
        print(f"  2. Test full filter: python tests/test_manual_eskf.py")

        return 0

    except Exception as e:
        print(f"\n✗ Failed to generate scenario: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
