# Test Scenarios Guide

## Overview

Three specialized test scenarios have been generated to evaluate filter robustness:

1. **Baseline** - Clean data with fixed configuration
2. **Spikes** - Frequent measurement outliers
3. **Anomalies** - Realistic operational issues

## Generated Datasets

| Run ID | Name | Duration | Description |
|--------|------|----------|-------------|
| 2 | baseline_fixed | 120s | Clean data with realistic noise levels |
| 3 | spikes_scenario | 120s | All sensors have spike generation enabled |
| 4 | anomalies_scenario | 180s | Freezes, eclipse, high spin |

## Scenario Details

### 1. Baseline Scenario (run_id=2)

**Purpose:** Establish baseline performance with clean, realistic data

**Configuration:**
- Duration: 2 minutes (6001 samples @ 20ms)
- Noise levels: Realistic (1e-3 for mag, 1e-4 for sun/star)
- **Biases disabled** (fixes model mismatch issue)
- No spikes or anomalies

**Expected Performance:**
- GTSAM: < 0.1° attitude error
- ESKF: < 1° attitude error (after fixes)

**Measurement Availability:**
- Gyro: 100%
- Magnetometer: 16.7% (dt=0.12s)
- Sun sensor: 5.0% (dt=0.4s)
- Star tracker: <0.1% (dt=0.7s)

### 2. Spike Scenario (run_id=3)

**Purpose:** Test outlier rejection and robustness to measurement spikes

**Configuration:**
- Duration: 2 minutes (6001 samples @ 20ms)
- **Spikes enabled** for all sensors:
  - Magnetometer: 5% probability, 0.5 rad magnitude (~28°)
  - Sun sensor: 8% probability, 0.3 rad magnitude (~17°)
  - Star tracker: 6% probability, 0.5 rad angle (~28°)
- Faster sampling for spike testing

**Expected Behavior:**
- ~50 magnetometer spikes (5% of 1001 measurements)
- ~24 sun sensor spikes (8% of 301 measurements)
- Occasional star tracker false solutions

**Note:** Spikes are added before normalization, so actual spike magnitude may be smaller after projecting onto unit sphere.

**Testing:**
- Filters should detect and reject outliers
- Chi-squared gating recommended
- GTSAM batch optimization naturally handles outliers better

### 3. Anomaly Scenario (run_id=4)

**Purpose:** Test filter behavior under realistic operational anomalies

**Configuration:**
- Duration: 3 minutes (9001 samples @ 20ms)
- High angular velocity profile (up to ~100 deg/s)

**Anomalies:**

#### Eclipse (60s - 90s)
- Sun sensor measurements unavailable
- Filters must rely on gyro + magnetometer + occasional star tracker
- **Expected:** Attitude drift without sun reference

#### Measurement Freeze (120s - 140s)
- Magnetometer readings stuck at last value
- Simulates sensor hang or communication issue
- **Expected:** Filter should detect and ignore frozen data

#### High Spin Period (30s - 55s)
- Angular rates: ~100 deg/s (well above 5 deg/s threshold)
- Star tracker automatically drops out (motion blur)
- **Expected:** Gyro + mag/sun only, increased attitude uncertainty

**Measurement Availability:**
- Magnetometer: 16.7% (normal)
- Sun sensor: 4.2% (reduced due to eclipse)
- Star tracker: <0.1% (extremely sparse due to high rates)

## How to Use

### Generate All Scenarios
```bash
source .venv/bin/activate
python generate_test_scenarios.py
```

### Validate Scenarios
```bash
python validate_scenarios.py
```

### Test with GTSAM
```bash
# Test baseline
python test_gtsam_with_db.py --sim_run_id 2

# Test with spikes
python test_gtsam_with_db.py --sim_run_id 3

# Test with anomalies
python test_gtsam_with_db.py --sim_run_id 4
```

Or modify the test scripts to specify the run_id:
```python
# In test_gtsam_with_db.py, change TEST_CONFIGS:
TEST_CONFIGS = [
    {
        "name": "Baseline Test",
        "sim_run_id": 2,  # ← Change this
        "window_size": 100,
        "max_samples": 1000,
    },
]
```

### Test with ESKF
```bash
python test_eskf_with_db.py  # Modify sim_run_id in script
```

## Expected Results

### Baseline Scenario
- **GTSAM:** ~0.01° mean error
- **ESKF:** ~0.5° mean error (if properly tuned)
- Both should converge smoothly

### Spike Scenario
- **GTSAM:** ~0.1° mean error (robust to outliers)
- **ESKF:** May diverge without outlier rejection
- **Recommendation:** Implement chi-squared gating for ESKF

### Anomaly Scenario
- **GTSAM:** ~0.5-1° mean error (graceful degradation)
- **ESKF:** May diverge during long eclipse period
- **Both:** Should show increased uncertainty during anomalies

## Troubleshooting

### No Spikes Detected
The spikes are added before measurement normalization, making them harder to detect with simple magnitude checks. They are present but manifest as angular deviations.

To verify spikes, compare predicted vs measured:
```python
# In filter code
innovation = z_meas - z_pred
if np.linalg.norm(innovation) > 3*sigma:
    print(f"Outlier detected at t={t}")
```

### ESKF Diverges
Try:
1. Use only baseline scenario first (run_id=2)
2. Increase process noise (Q_c) by 2-5x
3. Inflate measurement noise (R) by 2-3x
4. Add chi-squared outlier rejection

### Star Tracker Always Missing
This is expected! Star tracker has:
- Slow sampling (0.7s)
- Rate limit (5 deg/s)
- In anomaly scenario, rates exceed 50 deg/s most of the time

## Configuration Files

- `config.yaml` - Original configuration (has issues)
- `config_baseline_temp.yaml` - Auto-generated (fixed config)
- `config_spikes.yaml` - Spike scenario configuration
- `config_anomalies.yaml` - Anomaly scenario configuration

## Key Fixes Applied

The baseline and spike scenarios include these critical fixes:
```yaml
sensors:
  mag:
    mag_std: 1.0e-3        # Increased from 4e-5 (more realistic)
    bias_rw_std: 0.0       # Disabled (fixes model mismatch)

  sun:
    noise:
      sun_std: 1.0e-4      # Increased from 5e-6
    bias:
      rw_std: 0.0          # Disabled

  star:
    noise:
      st_std: 1.0e-4       # Increased from 7e-6
    bias:
      rw_std: 0.0          # Disabled
```

## References

- `generate_test_scenarios.py` - Generation script
- `validate_scenarios.py` - Validation script
- `data/generator_enhanced.py` - Enhanced generator with anomaly support
- `CRITICAL_FINDINGS.md` - Detailed analysis of issues found

## Next Steps

1. Test baseline scenario with both filters
2. Implement chi-squared gating for ESKF
3. Test spike scenario with outlier rejection
4. Evaluate anomaly scenario for graceful degradation
5. Compare GTSAM vs ESKF robustness across all scenarios
