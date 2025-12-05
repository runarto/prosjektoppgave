# Attitude Estimation: ESKF vs Factor Graph Optimization

Comparison of Error-State Kalman Filter (ESKF) and Factor Graph Optimization (FGO) for spacecraft attitude estimation using realistic space-grade sensor data.

## Documentation

- **[HYBRID_ESTIMATOR.md](HYBRID_ESTIMATOR.md)** - Architecture and implementation of the hybrid ESKF+FGO estimator
- **[docs/SCENARIOS.md](docs/SCENARIOS.md)** - Scenario generation, validation, and fixes
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Quick start guide
- **[docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md)** - Performance tuning guide
- **[docs/TEST_SCENARIOS_GUIDE.md](docs/TEST_SCENARIOS_GUIDE.md)** - Testing scenarios guide

## Project Structure

```
.
├── configs/                    # Scenario configurations
│   ├── config_spacegrade_realistic.yaml   # Baseline scenario
│   ├── config_rapid_tumbling.yaml         # High angular rates
│   ├── config_eclipse.yaml                # Sun sensor dropout
│   └── config_measurement_spikes.yaml     # Measurement outliers
│
├── scripts/                    # Data generation scripts
│   └── generate_all_scenarios.py          # Generate all test scenarios
│
├── tests/                      # Test and comparison scripts
│   ├── test_eskf_with_db.py              # ESKF on database runs
│   ├── test_gtsam_with_db.py             # FGO on database runs
│   ├── test_manual_eskf.py               # Manual ESKF test
│   ├── test_manual_eskf_spikes.py        # ESKF with outlier rejection
│   └── test_comparison.py                # ESKF vs FGO comparison
│
├── plotting/                   # Visualization tools
│   ├── attitude_plotter.py               # Attitude-specific plots
│   ├── comparison_plotter.py             # ESKF vs FGO comparison plots
│   └── data_plotter.py                   # General data plotting
│
├── sim/                        # Estimation implementations
│   ├── eskf.py                           # Manual ESKF implementation
│   ├── eskf_filterpy.py                  # FilterPy-based ESKF
│   ├── gtsam_fg.py                       # GTSAM factor graph
|   ├── hybrid_estimator                  # Hybrid and redundant architecture for KF + FG
│   └── fg.py                             # Custom factor graph
│
├── data/                       # Data generation and management
│   ├── generator_enhanced.py             # Simulation data generator
│   └── db.py                             # Database interface
│
├── utilities/                  # Core utilities
│   ├── quaternion.py                     # Quaternion math
│   ├── states.py                         # State representations
│   ├── sensors.py                        # Sensor models
│   └── process_model.py                  # Process dynamics
│
└── environment/                # Environment models
    └── environment.py                    # Orbit, magnetic field, sun vector
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install numpy scipy matplotlib pyyaml gtsam filterpy
```

**Note:** The `pyIGRF` library requires a coefficient file. Download it from the pyIGRF repository and place it in the appropriate location if you encounter errors.

## Test Scenarios

### 1. Baseline (Realistic Space-Grade)
- Nominal dynamics (~1°/s angular velocity)
- Space-grade sensors (FOG gyro, nT magnetometer, arcsecond star tracker)
- 5% star tracker dropout
- 50-minute duration

### 2. Rapid Tumbling
- High angular velocities (up to ~8.6°/s)
- Star tracker dropout above 2°/s
- Tests filter performance during aggressive maneuvers

### 3. Eclipse
- 30% sun sensor dropout (simulating eclipse periods)
- Tests degraded observability

### 4. Measurement Spikes
- Large random measurement outliers (500x-100x nominal noise)
- 0.5% probability of spikes on all sensors
- Tests outlier rejection mechanisms

## Usage

### 1. Generate Simulation Data

```bash
python scripts/generate_all_scenarios.py
```

This generates all four scenarios and stores them in `simulations.db`.

### 2. Run ESKF vs FGO Comparison

```bash
python tests/test_comparison.py
```

This:
- Runs both ESKF and FGO on all scenarios
- Generates comparison plots in `output/`
- Prints performance statistics

### 3. Run Individual Tests

```bash
# Test ESKF only
python tests/test_manual_eskf.py

# Test ESKF with outlier rejection
python tests/test_manual_eskf_spikes.py

# Test FGO
python tests/test_gtsam_with_db.py
```

## Results

All comparison plots and results are saved to the `output/` directory:

- `{scenario}_comparison.png` - Time-series comparison
- `{scenario}_statistics.png` - Statistical comparison (mean, std, max)
- `all_scenarios_comparison.png` - Multi-scenario overview

## Key Features

### ESKF Implementation (`sim/eskf.py`)
- Chi-squared outlier rejection (99.9% confidence)
- Adaptive innovation gating using Mahalanobis distance
- Joseph-form covariance update for numerical stability
- Covariance regularization to prevent numerical issues

### FGO Implementation (`sim/gtsam_fg.py`)
- GTSAM-based batch smoother
- Sliding window optimization (configurable window size)
- Between-factor process model
- Multi-sensor fusion (magnetometer, sun sensor, star tracker)

## Configuration

Scenario configs are in YAML format. Key parameters:

```yaml
sensors:
  gyro:
    noise:
      gyro_std: 1.0e-6        # White noise (rad/s)
      bias_instability_deg_per_h: 0.01  # Bias drift
  star:
    dropout:
      max_rate_deg_s: 2.0     # Dropout threshold
  spikes:
    enabled: true
    probability: 0.005        # Spike probability

omega_profile:
  amplitude:
    x: 0.15                   # Angular velocity (rad/s)
```

## Performance Metrics

The comparison evaluates:

1. **Attitude Error**: Angular deviation from true attitude (degrees)
2. **Bias Error**: Gyro bias estimation error (rad/s)
3. **Convergence**: Time to achieve steady-state accuracy
4. **Robustness**: Performance under anomalies (spikes, dropouts)
5. **Computational Cost**: Execution time

## Publication-Quality Plots

Plots use:
- Times New Roman font (serif fallback if unavailable)
- 300 DPI resolution
- Consistent color scheme (ESKF: blue, FGO: orange)
- Grid lines with 30% transparency
- LaTeX-compatible formatting

Generated plots are suitable for technical reports and conference papers.

## Example Results

**Baseline Scenario:**
- ESKF: Mean error 0.0004°, Final error 0.0004°
- FGO: Mean error 0.0023°, Final error 0.0215°

**Measurement Spikes:**
- ESKF: 0.49% rejection rate (173 spikes detected)
- Successfully maintained convergence with chi-squared outlier rejection
