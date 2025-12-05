#!/usr/bin/env python3
"""
Generate all test scenarios for attitude estimation comparison.

Scenarios:
1. Baseline - Realistic space-grade sensors with nominal dynamics
2. Rapid Tumbling - High angular rates causing star tracker dropout
3. Eclipse - Sun sensor unavailable periods
4. Measurement Spikes - Random measurement outliers

Each scenario generates a 3000-second (50-minute) dataset.
"""

import sys
from data.generator_enhanced import EnhancedAttitudeDataGenerator
from data.classes import SimulationConfig
from utilities.utils import load_yaml


SCENARIOS = [
    ("configs/config_spacegrade_realistic.yaml", "Baseline (Realistic Space-Grade)"),
    ("configs/config_rapid_tumbling.yaml", "Rapid Tumbling"),
    ("configs/config_eclipse.yaml", "Eclipse"),
    ("configs/config_measurement_spikes.yaml", "Measurement Spikes"),
]


def generate_scenario(config_path: str, description: str) -> int:
    """Generate a single scenario."""
    print("\n" + "="*70)
    print(f"GENERATING: {description}")
    print("="*70)

    config = load_yaml(config_path)

    generator = EnhancedAttitudeDataGenerator(
        db_path="simulations.db",
        config_path=config_path
    )

    sim_cfg = SimulationConfig(
        T=config["time"]["sim_T"],
        dt=config["time"]["sim_dt"],
        start_jd=config["time"]["start_jd"]
    )

    print(f"\nConfig: {config_path}")
    print(f"Duration: {sim_cfg.T:.1f}s")
    print(f"Timestep: {sim_cfg.dt}s")
    print(f"Expected samples: {int(sim_cfg.T / sim_cfg.dt) + 1}")

    run_id = generator.run(sim_cfg)

    print(f"\n✓ Scenario generated with run_id={run_id}")
    return run_id


def main():
    print("="*70)
    print("ATTITUDE ESTIMATION SCENARIO GENERATOR")
    print("="*70)
    print(f"\nWill generate {len(SCENARIOS)} scenarios:")
    for _, desc in SCENARIOS:
        print(f"  • {desc}")

    print("\n" + "="*70)
    input("Press Enter to start generation...")

    run_ids = {}

    for config_path, description in SCENARIOS:
        try:
            run_id = generate_scenario(config_path, description)
            run_ids[description] = run_id
        except Exception as e:
            print(f"\n✗ ERROR generating {description}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated scenarios:")
    for desc, run_id in run_ids.items():
        print(f"  {run_id:3d} - {desc}")

    print(f"\n✓ Successfully generated {len(run_ids)} scenarios!")
    print("\nUse these run IDs for testing:")
    print("  python tests/test_eskf_scenarios.py")
    print("  python tests/test_fgo_scenarios.py")
    print("  python tests/test_comparison.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
