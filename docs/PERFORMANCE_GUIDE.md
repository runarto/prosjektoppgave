# FGO Performance Optimization Guide

Factor Graph Optimization is computationally expensive for long datasets. Here are strategies to make it practical.

## The Problem

FGO with GTSAM performs batch optimization over a sliding window. For a 3000-second dataset:
- 150,000 samples @ 50 Hz
- With stride=50, window=300: ~3000 optimizations
- Each optimization: solving a 300-sample nonlinear least squares problem
- **Total time: Several hours** ⚠️

## Solutions (Ranked by Practicality)

### ✅ 1. Use Shorter Test Datasets (RECOMMENDED)

**For research comparisons, 5-10 minute datasets are statistically sufficient.**

Create short versions of all scenarios:
```yaml
time:
  sim_T: 300.0  # 5 minutes instead of 3000 seconds
```

**Advantages:**
- 15,000 samples instead of 150,000 (10x faster)
- Still provides ~50-100 optimizations for convergence analysis
- Sufficient for statistical comparison
- ~10 minutes runtime for FGO

**Use case:**
- Initial testing and algorithm comparison
- Parameter tuning
- Generating plots for reports

**Create short configs:**
```bash
# Already created: config_baseline_short.yaml
# Create similar for other scenarios
```

### ✅ 2. Optimize FGO Parameters

Adjust window size and stride for long datasets:

| Parameter | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Window size | 300 | 200 | 1.5x |
| Stride | 50 | 1000 | 20x |
| **Total** | | | **~30x faster** |

**Trade-offs:**
- Larger stride: Fewer state estimates, but faster
- Smaller window: Less temporal smoothing, but faster convergence

**Implementation:**
```python
# Use test_fgo_fast.py
window_size = 200      # 4 seconds
optimize_stride = 1000  # 20 seconds between optimizations
```

**Results:**
- ~150 optimizations instead of 3000
- Runtime: ~30 minutes instead of hours
- Still captures long-term behavior

### ⚠️ 3. Downsample Data

Skip samples to reduce dataset size:

```python
# Process every Nth sample
downsample_factor = 5  # Use every 5th sample
for k in range(0, len(sim.t), downsample_factor):
    # Process sample k
```

**Trade-offs:**
- ✅ Faster processing
- ❌ Loses high-frequency dynamics
- ❌ May miss rapid attitude changes
- ❌ Not recommended for rapid tumbling scenarios

### 🔧 4. Non-Overlapping Windows

Clear window after each optimization instead of sliding:

```python
if window.ready and should_optimize:
    states = fgo.optimize_window(window.samples, env)
    # Store states...
    window.samples.clear()  # Non-overlapping batches
```

**Advantages:**
- Much faster (no re-optimization of same data)
- Reduces redundant computation

**Disadvantages:**
- No temporal smoothing across windows
- Discontinuities between batches

## Recommended Strategy for Your Report

### For Algorithm Comparison (ESKF vs FGO)

Use **short datasets (5-10 minutes)**:

1. **Generate short scenarios:**
   ```bash
   python scripts/generate_short_scenarios.py  # Create this
   ```

2. **Run comparison:**
   ```bash
   python tests/test_comparison.py  # Update to use short datasets
   ```

3. **Expected runtime:** 10-15 minutes per scenario

### For Long-Duration Stability Analysis

Use **ESKF only** on 50-minute datasets:

```bash
python tests/test_manual_eskf.py  # Already optimized, runs in ~1 minute
```

**Rationale:**
- ESKF is O(1) per timestep (Kalman update)
- FGO is O(N²) for window size N
- For stability over time, filtering is more appropriate than smoothing

### For Final Report

1. **Short datasets** (5-10 min) for detailed comparison:
   - All 4 scenarios
   - Both ESKF and FGO
   - Publication-quality plots
   - Statistical analysis

2. **Long dataset** (50 min) for ESKF only:
   - Demonstrates long-term stability
   - Shows no divergence over mission duration
   - One baseline scenario sufficient

3. **Performance table:**
   | Method | Short (5 min) | Long (50 min) |
   |--------|---------------|---------------|
   | ESKF   | ~5 seconds    | ~1 minute     |
   | FGO    | ~10 minutes   | ~Hours (impractical) |

## Implementation Scripts

### Fast FGO Test (Long Datasets)
```bash
python tests/test_fgo_fast.py
```
- Window: 200 samples
- Stride: 1000 samples
- Runtime: ~30 minutes for 50-minute dataset

### Comparison on Short Datasets
```bash
# 1. Update test_comparison.py to use short configs
# 2. Run comparison
python tests/test_comparison.py
```

## Computational Complexity

| Method | Per-Step | Per-Dataset (N samples) |
|--------|----------|------------------------|
| ESKF   | O(1)     | O(N)                   |
| FGO (stride S, window W) | O(W³) every S steps | O(N/S × W³) |

**Example (N=150,000, S=1000, W=200):**
- ESKF: 150,000 updates × O(1) = **O(150,000)**
- FGO: 150 optimizations × O(8,000,000) = **O(1,200,000,000)**

FGO is ~8000x more expensive per optimization!

## Bottom Line

**For your report:**
1. Use 5-10 minute datasets for ESKF vs FGO comparison ✅
2. Use 50-minute dataset for ESKF long-term stability ✅
3. Clearly state this is a practical limitation of batch smoothing
4. Note that real-time spacecraft use filtering (ESKF), not batch smoothing

This is scientifically valid - your comparison shows algorithm performance under realistic conditions, and the computational analysis is an important part of the trade-off discussion.
