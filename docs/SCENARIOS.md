# Scenario Generation and Validation

## Overview

This document describes the scenario generation system, validation tools, and recent fixes to ensure realistic spacecraft sensor measurements.

## Quick Start

```bash
# Generate scenarios
python -m data.generate_scenarios_short

# Validate scenarios
python validate_scenarios.py

# Run estimators
python tests/test_hybrid_estimator.py
```

## Scenario Generator

**Script:** `data/generator_enhanced.py`
**Config:** `configs/config_*_short.yaml`

### What It Does

Generates spacecraft attitude simulation data including:
- **Truth data**: Quaternion attitude, angular velocity, gyro bias
- **Sensor measurements**: Gyro, magnetometer, sun sensor, star tracker
- **Environment**: Magnetic field, sun vector (ECI frame)

### Key Features

1. **Sun-pointing initialization**: Spacecraft attitude initialized to point sun sensor (+Z body axis) toward sun, ensuring measurements from start
2. **Realistic sensor biases**: Gyro bias random walk, magnetometer hard-iron/soft-iron effects
3. **Sensor constraints**: Sun sensor FOV (120°), star tracker max rate (2-5 deg/s)
4. **Anomaly support**: Eclipse periods, measurement freezes, spikes

### Sensor Configuration

```yaml
sensors:
  gyro:
    noise_rw: 0.0001  # rad/s/√s - drives bias random walk
    bias_instability_deg_per_h: 1.0  # Realistic spacecraft gyro

  mag:
    hard_iron: [2e-7, -1.5e-7, 1e-7]  # Tesla (~200 nT constant bias)
    soft_iron: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Scale/rotation matrix

  sun:
    fov_deg: 120.0  # Cone half-angle = 60°, boresight = +Z body

  star:
    max_rate_deg_s: 2.0  # Dropouts when ||ω|| > 2 deg/s
```

### Important Assumptions

- **Coordinate frames**: ECI (inertial), body (spacecraft-fixed)
- **Quaternion convention**: `q = [q0, q1, q2, q3]` with `q0 = cos(θ/2)` (scalar first)
- **Rotation semantics**: `q` rotates body→ECI, `R_bn = q.as_rotmat().T` rotates ECI→body
- **Sun sensor boresight**: +Z body axis
- **Measurement timing**: Each sensor has its own `dt`, aligned to simulation `sim_dt`

## Validation Script

**Script:** `validate_scenarios.py`

### What It Does

Comprehensive validation of generated scenarios to catch common issues before running estimators.

### Checks Performed

1. **Data quality**
   - Quaternion normalization (|q| = 1 ± 1e-6)
   - No NaN/Inf in truth data

2. **Sensor availability**
   - Gyro: 100% (always available)
   - Magnetometer: >10% (sampled at lower rate)
   - Sun sensor: >5% (FOV constraints)
   - Star tracker: >1% (rate constraints)

3. **Sun sensor FOV**
   - Verifies sun vector enters FOV (Z-component > 0.5 for 120° FOV)
   - Detects initial attitude issues

4. **Star tracker rate**
   - Ensures dropout when ||ω|| > max_rate
   - Checks measurements only occur during slow rotation

5. **Gyro bias magnitude**
   - Warns if bias < 0.01 deg/h (unrealistically small)
   - Warns if bias > 100 deg/h (unrealistic)

6. **Magnetometer bias**
   - Verifies hard-iron bias is applied

### Example Output

```
✓ PASSED (7):
  - Quaternion normalization: 2.22e-16
  - Truth data has no NaN/Inf values
  - Gyro bias magnitude: 0.85 deg/h
  - Magnetometer availability: 20.0%
  - Sun sensor availability: 18.0%
  - Star tracker availability: 3.8%
  - Magnetometer bias present

ℹ INFO:
  - Sun in FOV: 45/100 samples (45.0%)
  - Overall angular rate: mean=1.2 deg/s, max=1.6 deg/s
```

## Recent Fixes

### 1. Sun Sensor FOV Issue (FIXED)

**Problem:** 3/4 scenarios had zero sun sensor measurements because spacecraft never pointed sensor toward sun.

**Root cause:** Initial attitude was identity quaternion (arbitrary), and small tumbling kept sun at ~63° from sensor (just outside 60° FOV limit).

**Fix:** `data/generator_enhanced.py:155-185` now initializes attitude to align +Z body with sun direction.

**Result:** Sun sensor measurements now available in all scenarios.

### 2. Gyro Bias Too Small (FIXED)

**Problem:** Gyro bias was ~0.003-0.008 deg/h (unrealistically small).

**Fix:** Increased `noise_rw` from 1e-7 to 1e-4 and `bias_instability_deg_per_h` from 0.01 to 1.0 in all config files.

**Result:** Bias now reaches realistic levels (~0.3-1.0 deg/h).

### 3. Magnetometer Bias Not Applied (FIXED)

**Problem:** Config had `hard_iron: [0, 0, 0]` AND sensor code didn't apply it even when configured.

**Fix:**
- `utilities/sensors.py`: Load and apply `hard_iron` and `soft_iron` matrices
- All config files: Set `hard_iron: [2e-7, -1.5e-7, 1e-7]` (~200 nT)

**Result:** Magnetometer now has realistic constant bias.

## Troubleshooting

### Sun sensor has zero measurements

**Diagnosis:**
```bash
python validate_scenarios.py
# Look for: "Sun NEVER in FOV! Max Z-component: X.XXX (need > 0.500)"
```

**Fix:** Re-generate scenarios with updated `data/generator_enhanced.py` that initializes sun-pointing attitude.

### Gyro bias too small

**Diagnosis:**
```bash
python validate_scenarios.py
# Look for: "Gyro bias very small: X.XX deg/h"
```

**Fix:** Check config files have:
```yaml
gyro:
  noise_rw: 0.0001
  bias_instability_deg_per_h: 1.0
```

### Star tracker always available during tumbling

**Diagnosis:**
```bash
python validate_scenarios.py
# Check: "Star tracker rate constraint: max X.X deg/s during measurements"
```

**Expected:** Should be < 5 deg/s (typically 2 deg/s threshold).

**If higher:** Check `max_rate_deg_s` in config and verify tumbling amplitude is sufficient.

## Files

### Generation
- `data/generator_enhanced.py` - Main generator with sun-pointing init
- `data/classes.py` - Data classes (SimulationConfig, SimulationResult)
- `data/db.py` - SQLite database interface

### Validation
- `validate_scenarios.py` - Comprehensive validation script

### Configuration
- `configs/config_baseline_short.yaml` - 5-minute baseline scenario
- `configs/config_rapid_tumbling_short.yaml` - High angular rates
- `configs/config_eclipse_short.yaml` - Sun sensor dropout periods
- `configs/config_measurement_spikes_short.yaml` - Outlier injection

## Testing
