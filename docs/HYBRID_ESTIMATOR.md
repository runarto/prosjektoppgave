# Hybrid Estimator Architecture

## Overview

The Hybrid Estimator combines ESKF (frontend filter) with FGO (backend optimizer) in a dual-mode architecture that automatically switches between modes based on measurement availability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID ESTIMATOR                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌────────────────────────┐         │
│  │     ESKF     │         │    Sliding Window      │         │
│  │   Frontend   │────────▶│  (100 samples)         │         │
│  │              │  store  │                         │         │
│  └──────┬───────┘         └───────────┬─────────────┘         │
│         │                             │                       │
│         │ real-time                   │ every 10-20s          │
│         │ estimates                   ▼                       │
│         │                  ┌──────────────────────┐           │
│         │                  │    FGO Backend       │           │
│         │                  │  (Batch Optimizer)   │           │
│         │                  └──────────┬───────────┘           │
│         │                             │                       │
│         │ ◀───────────────────────────┘                       │
│         │   optimized state                                   │
│         │   + covariance                                      │
│         │                                                      │
│         ▼                                                      │
│   State Output                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Operating Modes

### 1. DUAL Mode (Normal Operation)

**When**: Measurement rate > 30%

**Behavior**:
- ESKF runs continuously, providing real-time state estimates
- All measurements processed by ESKF with chi-squared outlier rejection
- Measurements stored in sliding window (100 samples)
- FGO optimizes over window every 10 seconds
- Optimized state fed back to ESKF:
  - Nominal state (quaternion + bias) updated
  - Error covariance updated (if extraction successful)
  - Error mean reset to zero

**Benefits**:
- Real-time performance from ESKF
- Refined accuracy from periodic FGO smoothing
- Best of both worlds

### 2. FGO-ONLY Mode (Degraded Measurements)

**When**: Measurement rate < 30%

**Behavior**:
- ESKF prediction still runs (for propagation)
- ESKF updates skipped (measurements too sparse for reliable filtering)
- FGO optimization runs more frequently
- State estimates come entirely from batch optimization

**Benefits**:
- Handles sparse measurements better (temporal coherence)
- Avoids ESKF divergence during measurement dropout
- Automatic recovery when measurements resume

## Mode Switching Logic

```python
def check_mode_switch(self):
    measurement_rate = measurements_available / total_samples

    if measurement_rate < 0.3 and mode == DUAL:
        switch_to(FGO_ONLY)
    elif measurement_rate >= 0.3 and mode == FGO_ONLY:
        switch_to(DUAL)
```

## Key Features

### 1. Covariance Recovery

FGO provides point estimates without inherent uncertainty. We attempt to extract covariance using:

```python
marginals = gtsam.Marginals(graph, result)
P_R = marginals.marginalCovariance(R_key)  # 3x3 rotation
P_b = marginals.marginalCovariance(b_key)  # 3x3 bias

# Combine into 6x6 covariance
P = block_diag(P_R, P_b)
```

**Fallback Strategy**: Covariance extraction often fails when the bias is underconstrained (sparse measurements). In this case, we use a **reduced covariance** to reflect the fact that FGO has smoothed over a batch:

```python
if P_fgo is None:  # Extraction failed
    # After optimization, uncertainty should be lower
    P_reduced = 0.5 * P_eskf  # Reduce by 50%
    P_optimized = min(P_reduced, P0)  # But not lower than initial covariance
```

This prevents the large ESKF covariance from causing chi-squared rejections after FGO optimization.

### 2. State Feedback

After FGO optimization:
```python
# Update nominal state
x_eskf.nom.ori = q_optimized
x_eskf.nom.gyro_bias = b_optimized

# Update covariance (if available)
if P_fgo is not None:
    x_eskf.err.cov = P_fgo

# Reset error mean
x_eskf.err.mean = zeros(6)
```

### 3. Measurement Health Tracking

```python
class MeasurementHealth:
    mag_count: int
    sun_count: int
    star_count: int
    total_samples: int

    @property
    def measurement_rate(self) -> float:
        return (mag + sun + star) / (3 * total)
```

Tracks recent measurement availability with exponential decay to adapt to changing conditions.

### 4. Robust M-Estimators

Both ESKF and FGO use robust outlier rejection:
- **ESKF**: Chi-squared test (Mahalanobis distance > 16.27 rejected)
- **FGO**: Huber M-estimator (automatic robust weighting)

## Configuration Parameters

```python
HybridEstimator(
    P0=initial_covariance,           # 6x6
    config_path="config.yaml",
    fgo_window_size=100,             # samples
    fgo_optimize_interval=10.0,      # seconds
    degraded_threshold=0.3,          # measurement rate
    use_robust=True                  # M-estimator in FGO
)
```

## Usage Example

```python
from estimation.hybrid_estimator import HybridEstimator
from utilities.states import NominalState, EskfState

# Initialize
P0 = np.diag([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
hybrid = HybridEstimator(P0=P0, config_path="config.yaml")

# Initial state
x_est = EskfState(nom=nom0, err=err0)

# Process measurements
for k in range(len(timestamps)):
    x_est, fgo_updated = hybrid.step(
        x_eskf=x_est,
        t=t[k],
        jd=jd[k],
        omega_meas=omega[k],
        dt=dt[k],
        z_mag=mag[k],      # None if unavailable
        z_sun=sun[k],      # None if unavailable
        z_st=star[k],      # None if unavailable
        B_n=B_eci[k],
        s_n=s_eci[k]
    )

    # Use x_est.nom for state estimate
    # fgo_updated flag indicates if FGO ran this step
```

## Advantages Over Pure ESKF

1. **Better accuracy**: Batch smoothing leverages temporal coherence
2. **Sparse measurement handling**: FGO-only mode during degraded periods
3. **Outlier robustness**: M-estimators in both frontend and backend
4. **Automatic adaptation**: Mode switching based on measurement health

## Advantages Over Pure FGO

1. **Real-time performance**: ESKF provides immediate state estimates
2. **Lower computational cost**: FGO runs periodically, not every step
3. **Graceful degradation**: Falls back to ESKF if FGO fails

## Performance Characteristics

- **DUAL mode**:
  - ESKF: O(1) per step
  - FGO: O(W³) every N steps (W=100, N=500)
  - Net: ~O(1) amortized

- **FGO-ONLY mode**:
  - FGO: O(W³) every M steps (more frequent)
  - Handles sparse measurements better than filtering

## Test Results

See `tests/test_hybrid_estimator.py` for comprehensive testing on:
- Baseline scenario
- Rapid tumbling (star tracker dropout)
- Eclipse (sun sensor dropout)
- Measurement spikes (outlier rejection)

## Implementation Files

- `estimation/hybrid_estimator.py` - Main hybrid estimator class
- `estimation/eskf.py` - ESKF frontend
- `estimation/gtsam_fg.py` - FGO backend with M-estimators
- `tests/test_hybrid_estimator.py` - Test suite

## Future Enhancements

1. **Adaptive window size**: Adjust based on measurement density
2. **Marginal covariance computation**: More robust methods when system is underconstrained
3. **Loop closure**: Add constraints when revisiting similar states
4. **Multi-rate optimization**: Different intervals for rotation vs bias
