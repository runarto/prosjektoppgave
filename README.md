# Spacecraft Attitude Estimation

Attitude estimation framework for spacecraft using multiple sensor fusion. Implements three estimators: ESKF (Error-State Kalman Filter), iSAM2 (incremental smoothing), and a Redundant estimator combining both.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pyyaml gtsam
```

## Project Structure

```
.
├── configs/                 # Scenario configuration files (YAML)
├── data/                    # Data generation
│   ├── generator_enhanced.py   # Simulation data generator
│   ├── db.py                    # SQLite database interface
│   └── classes.py               # Data classes
├── estimation/              # Estimator implementations
│   ├── eskf.py                  # Error-State Kalman Filter
│   ├── keyframe_fgo.py          # iSAM2 factor graph optimization
│   ├── fixed_lag_smoother.py    # Fixed-lag smoother for redundant
│   └── redundant_estimator.py   # ESKF + Smoother with fault detection
├── utilities/               # Core utilities
│   ├── quaternion.py            # Quaternion operations
│   ├── states.py                # State representations
│   ├── sensors.py               # Sensor models
│   └── process_model.py         # Process dynamics
├── environment/             # Environment models (orbit, mag field, sun)
├── plotting/                # Visualization tools
├── scripts/                 # Analysis and comparison scripts
└── simulations.db           # SQLite database with simulation data
```

## Quick Start

### 1. Generate Simulation Data

```python
from data.generator_enhanced import EnhancedAttitudeDataGenerator

generator = EnhancedAttitudeDataGenerator(
    db_path="simulations.db",
    config_path="configs/config_baseline_short.yaml"
)
run_id = generator.generate()
print(f"Generated simulation run ID: {run_id}")
```

### 2. Run ESKF

```python
from data.db import SimulationDatabase
from estimation.eskf import ESKF
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState, SensorType
from utilities.gaussian import MultiVarGauss
import numpy as np

# Load simulation
db = SimulationDatabase("simulations.db")
sim = db.load_run(run_id)

# Initialize ESKF
P0 = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])  # attitude (rad^2), bias
eskf = ESKF(P0=P0, config_path="configs/config_baseline_short.yaml")

# Initial state
q0 = Quaternion.from_array(sim.q_true[0])
nom0 = NominalState(ori=q0, gyro_bias=np.zeros(3))
err0 = MultiVarGauss(np.zeros(6), P0)
x_est = EskfState(nom=nom0, err=err0)

# Run filter
for k in range(1, len(sim.t)):
    dt = sim.t[k] - sim.t[k-1]

    # Predict with gyro
    x_est = eskf.predict(x_est, sim.omega_meas[k], dt)

    # Update with magnetometer
    if not np.any(np.isnan(sim.mag_meas[k])):
        x_est = eskf.update(x_est, sim.mag_meas[k], SensorType.MAGNETOMETER, B_n=sim.b_eci[k])

    # Update with sun sensor
    if not np.any(np.isnan(sim.sun_meas[k])):
        x_est = eskf.update(x_est, sim.sun_meas[k], SensorType.SUN_VECTOR, s_n=sim.s_eci[k])

    # Update with star tracker
    if not np.any(np.isnan(sim.st_meas[k])):
        q_meas = Quaternion.from_array(sim.st_meas[k])
        x_est = eskf.update(x_est, q_meas, SensorType.STAR_TRACKER)

# Access estimate
print(f"Final quaternion: {x_est.nom.ori}")
print(f"Final gyro bias: {x_est.nom.gyro_bias}")
```

### 3. Run iSAM2

```python
from estimation.keyframe_fgo import KeyframeFGO
from utilities.quaternion import Quaternion
from utilities.states import NominalState
import numpy as np

# Initialize iSAM2
fgo = KeyframeFGO(config_path="configs/config_baseline_short.yaml")

# Initial state
q0 = Quaternion.from_array(sim.q_true[0])
b0 = np.zeros(3)
nom0 = NominalState(ori=q0, gyro_bias=b0)

fgo.initialize(nom0, sim.t[0], sim.jd[0])

# Process measurements
for k in range(1, len(sim.t)):
    # Add gyro measurement (always)
    fgo.add_gyro(sim.omega_meas[k], sim.t[k], sim.jd[k])

    # Add vector measurements when available
    if not np.any(np.isnan(sim.mag_meas[k])):
        fgo.add_magnetometer(sim.mag_meas[k], sim.b_eci[k])

    if not np.any(np.isnan(sim.sun_meas[k])):
        fgo.add_sun_sensor(sim.sun_meas[k], sim.s_eci[k])

    if not np.any(np.isnan(sim.st_meas[k])):
        q_meas = Quaternion.from_array(sim.st_meas[k])
        fgo.add_star_tracker(q_meas)

# Get optimized estimate
result = fgo.get_current_estimate()
print(f"Optimized quaternion: {result.ori}")
print(f"Optimized gyro bias: {result.gyro_bias}")
```

### 4. Run Redundant Estimator

The redundant estimator runs ESKF and a fixed-lag smoother in parallel with automatic fault detection and switching.

```python
from estimation.redundant_estimator import RedundantEstimator
from utilities.quaternion import Quaternion
from utilities.states import NominalState, EskfState
from utilities.gaussian import MultiVarGauss
import numpy as np

# Initialize
P0 = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])
redundant = RedundantEstimator(
    P0=P0,
    config_path="configs/config_baseline_short.yaml",
    smoother_lag=60.0,              # 60s fixed-lag window
    disagreement_threshold_deg=2.0,  # Switch if >2 deg disagreement
)

# Initial state
q0 = Quaternion.from_array(sim.q_true[0])
nom0 = NominalState(ori=q0, gyro_bias=np.zeros(3))
err0 = MultiVarGauss(np.zeros(6), P0)
x_est = EskfState(nom=nom0, err=err0)

# Run estimator
for k in range(1, len(sim.t)):
    dt = sim.t[k] - sim.t[k-1]

    x_est, info = redundant.step(
        x_eskf=x_est,
        t=sim.t[k],
        jd=sim.jd[k],
        omega_meas=sim.omega_meas[k],
        dt=dt,
        z_mag=sim.mag_meas[k] if not np.any(np.isnan(sim.mag_meas[k])) else None,
        z_sun=sim.sun_meas[k] if not np.any(np.isnan(sim.sun_meas[k])) else None,
        z_st=Quaternion.from_array(sim.st_meas[k]) if not np.any(np.isnan(sim.st_meas[k])) else None,
        B_eci=sim.b_eci[k],
        s_eci=sim.s_eci[k],
    )

# Check which estimator is primary
print(f"Primary estimator: {redundant.primary}")
print(f"Final quaternion: {x_est.nom.ori}")
```

## Configuration

Scenarios are defined in YAML files under `configs/`. Key parameters:

```yaml
time:
  sim_dt: 0.02      # Simulation timestep (s)
  sim_T: 300.0      # Duration (s)

sensors:
  gyro:
    dt: 0.02
    noise:
      arw_deg: 0.15           # Angle random walk (deg/sqrt(h))
      rrw_deg: 0.5            # Rate random walk (deg/h/sqrt(h))

  mag:
    dt: 0.2
    mag_std: 0.015            # Measurement noise (normalized)

  sun:
    dt: 0.5
    fov_deg: 120.0            # Field of view
    noise:
      sun_std: 0.027

  star:
    dt: 5.0
    noise:
      st_std: 0.00024         # Very accurate
    dropout:
      max_rate_deg_s: 5.0     # Dropout above this rate

omega_profile:
  amplitude:
    x: 0.02                   # Angular velocity amplitude (rad/s)
    y: 0.01
    z: 0.015
  frequency:
    x: 0.01                   # Oscillation frequency (Hz)
    y: 0.008
    z: 0.005
```

## Estimators

| Estimator | Description | Use Case |
|-----------|-------------|----------|
| **ESKF** | Error-State Kalman Filter with chi-squared outlier rejection | Real-time, low latency |
| **iSAM2** | Incremental smoothing with keyframe-based preintegration | Higher accuracy, batch |
| **Redundant** | ESKF + Fixed-lag smoother with fault detection | Fault tolerance |

## Sensors

- **Gyro**: Angular velocity (always available)
- **Magnetometer**: Earth's magnetic field direction
- **Sun sensor**: Sun direction (limited FOV, eclipse dropouts)
- **Star tracker**: Absolute attitude (dropouts at high rates)

## Scripts

Example scripts are in `scripts/`:

- `run_eskf_and_plot.py` - Run ESKF and generate plots
- `run_fgo_and_plot.py` - Run iSAM2 and generate plots
- `baseline_comparison.py` - Compare ESKF vs iSAM2
- `fault_detection_*.py` - Fault detection analysis

## Quaternion Convention

Right-multiply convention (GTSAM compatible):
- `q` represents rotation from body to inertial frame
- Error: `q_err = q_true^{-1} * q_est`
- Propagation: `q_new = q * delta_q`
