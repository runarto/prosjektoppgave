# Redundant Estimator Architecture

The redundant estimator combines ESKF and a fixed-lag smoother running in parallel with automatic fault detection and switching.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REDUNDANT ESTIMATOR                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────┐       │
│  │       ESKF       │        │   Fixed-Lag Smoother     │       │
│  │   (Primary)      │        │   (60s window, iSAM2)    │       │
│  └────────┬─────────┘        └───────────┬──────────────┘       │
│           │                              │                      │
│           │ q_eskf                       │ q_smoother           │
│           │                              │                      │
│           └──────────┬───────────────────┘                      │
│                      │                                          │
│                      ▼                                          │
│           ┌──────────────────────┐                              │
│           │ Disagreement Monitor │                              │
│           │ ||q_eskf - q_smooth||│                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│           ┌──────────────────────┐                              │
│           │   Switching Logic    │                              │
│           │   (N consecutive)    │                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│              Output Estimate                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Operating Modes

### Normal Operation (ESKF Primary)
- ESKF provides real-time estimates with low latency
- Smoother runs in parallel, building up window
- Disagreement monitored continuously
- Output = ESKF estimate

### Fault Detected (Smoother Primary)
When disagreement exceeds threshold for N consecutive updates:
- Switch output to smoother estimate
- ESKF continues running (does not reset)
- Allows natural recovery when fault clears

### Recovery
When disagreement drops below threshold for M consecutive updates:
- Switch back to ESKF
- Normal operation resumes

## Key Parameters

```python
RedundantEstimator(
    P0=P0,                                    # Initial covariance (6x6)
    config_path="config.yaml",
    smoother_lag=60.0,                        # Fixed-lag window (seconds)
    use_robust=True,                          # Huber M-estimator
    disagreement_threshold_deg=2.0,           # Switch threshold (degrees)
    consecutive_disagreements_to_switch=5,    # N measurements
    consecutive_agreements_to_recover=10,     # M measurements
    agreement_threshold_deg=0.5,              # Recovery threshold
)
```

## Fault Detection Mechanism

The disagreement between ESKF and smoother is computed as:

```
disagreement = 2 * arccos(|q_eskf · q_smoother|)  [radians]
```

A history of disagreement events is maintained:
- If `disagreement > threshold` for N consecutive updates → switch to smoother
- If `disagreement < agreement_threshold` for M consecutive updates → switch back to ESKF

## Why Not Reset ESKF on Switch?

When switching to smoother, ESKF continues with its current state. This ensures:
1. Disagreement stays high during persistent faults (no premature switch-back)
2. Natural re-convergence detected when fault clears
3. Smooth recovery when ESKF naturally catches up

## Robust Estimation

The fixed-lag smoother uses Huber M-estimator for outlier robustness:

```python
use_robust=True,
robust_kernel="huber",  # or "cauchy"
robust_param=0.1,       # kernel parameter
```

## Usage Example

```python
from estimation.redundant_estimator import RedundantEstimator

redundant = RedundantEstimator(
    P0=P0,
    config_path="configs/config_baseline_short.yaml",
)

for k in range(1, len(sim.t)):
    x_est, info = redundant.step(
        x_eskf=x_est,
        t=sim.t[k],
        jd=sim.jd[k],
        omega_meas=sim.omega_meas[k],
        dt=dt,
        z_mag=mag_meas,
        z_sun=sun_meas,
        z_st=st_meas,
        B_eci=sim.b_eci[k],
        s_eci=sim.s_eci[k],
    )

    # Check current mode
    if redundant.primary == PrimaryEstimator.SMOOTHER:
        print("Fault detected - using smoother")
```

## Implementation Files

- `estimation/redundant_estimator.py` - Main class
- `estimation/eskf.py` - ESKF implementation
- `estimation/fixed_lag_smoother.py` - Fixed-lag smoother using iSAM2
