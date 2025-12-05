# Quick Start Guide

## TL;DR - Fast Path to Results

```bash
# 1. Generate 5-minute test scenarios (fast!)
python scripts/generate_short_scenarios.py

# 2. Run comparison (ESKF vs FGO)
python tests/test_comparison.py

# 3. Get plots in output/
ls output/*.png
```

**Total time: ~1 hour** for all 4 scenarios

---

## Detailed Workflow

### Option 1: Fast Comparison (5-minute datasets) ⚡

**Best for:** Algorithm comparison, plots for report, rapid iteration

```bash
# Step 1: Generate short scenarios
python scripts/generate_short_scenarios.py
# Output: 4 scenarios × 15,000 samples each
# Time: ~5 minutes total

# Step 2: Run ESKF vs FGO comparison
python tests/test_comparison.py
# Update run IDs in the script first!
# Time: ~10 minutes per scenario = 40 minutes total

# Step 3: View results
ls output/
# - baseline_short_comparison.png
# - rapid_tumbling_short_comparison.png
# - eclipse_short_comparison.png
# - measurement_spikes_short_comparison.png
```

**Pros:**
- ✅ Fast (~1 hour total)
- ✅ Statistically valid (50-100 optimizations)
- ✅ All scenarios comparable
- ✅ Publication-quality plots

**Cons:**
- ❌ Doesn't show long-term stability (50 min)

---

### Option 2: Long-Term Stability (50-minute, ESKF only) 📊

**Best for:** Demonstrating no divergence over mission duration

```bash
# Use existing run 9 (already generated)
python tests/test_manual_eskf.py
# Time: ~1 minute
# Shows: 50 minutes of stable tracking
```

**Combine with Option 1** for complete report!

---

### Option 3: Fast FGO on Long Dataset 🐌

**Best for:** Testing if you really want FGO on long data

```bash
# Use existing run 9
python tests/test_fgo_fast.py
# Time: ~30 minutes (vs hours with original parameters)
# Trade-off: Fewer optimizations, but still shows performance
```

**Parameters:**
- Window: 200 samples (4 seconds)
- Stride: 1000 samples (20 seconds)
- ~150 optimizations total

---

## Recommended Report Structure

### For Your Thesis/Report

1. **Algorithm Comparison (5-minute datasets)**
   - Run all 4 short scenarios with both ESKF and FGO
   - Show side-by-side time-series plots
   - Include statistical comparison tables
   - **Source:** `test_comparison.py` with short datasets

2. **Long-Term Stability (50-minute, ESKF only)**
   - One baseline scenario showing 50 minutes
   - Demonstrate no divergence
   - Show steady-state performance
   - **Source:** `test_manual_eskf.py` on run 9

3. **Computational Analysis**
   - ESKF: O(N) complexity, ~1 minute for 50 min dataset
   - FGO: O(N/S × W³) complexity, ~hours for 50 min dataset
   - Justify use of short datasets for comparison
   - **Source:** PERFORMANCE_GUIDE.md

### Plot Organization

```
output/
├── baseline_short_comparison.png           # Figure 1
├── baseline_short_statistics.png           # Table 1 data
├── rapid_tumbling_short_comparison.png     # Figure 2
├── eclipse_short_comparison.png            # Figure 3
├── measurement_spikes_short_comparison.png # Figure 4
├── all_scenarios_comparison.png            # Figure 5 (overview)
└── baseline_long_eskf.png                  # Figure 6 (50 min stability)
```

---

## Troubleshooting

### "FGO takes too long!"
→ Use short datasets (5 minutes) or `test_fgo_fast.py`

### "Not enough data points for comparison"
→ 5 minutes = 15,000 samples with 50-100 optimizations is statistically sufficient

### "Need to show long-term behavior"
→ Use ESKF only for 50-minute dataset (FGO batch smoothing isn't meant for real-time long missions anyway)

### "Want faster iteration"
→ Use even shorter datasets (1-2 minutes) for algorithm development

---

## Performance Summary

| Configuration | Samples | ESKF Time | FGO Time (original) | FGO Time (fast) |
|---------------|---------|-----------|---------------------|-----------------|
| Short (5 min) | 15,000  | ~5 sec    | ~10 min            | ~3 min          |
| Long (50 min) | 150,000 | ~1 min    | ~Hours ⚠️          | ~30 min         |

**Recommendation:** Use short datasets for comparison, long dataset with ESKF only for stability demo.

---

## What Each Script Does

| Script | Purpose | Runtime | Output |
|--------|---------|---------|--------|
| `generate_short_scenarios.py` | Create 5-min datasets | ~5 min | 4 short scenarios in DB |
| `generate_all_scenarios.py` | Create 50-min datasets | ~30 min | 4 long scenarios in DB |
| `test_comparison.py` | ESKF vs FGO comparison | ~40 min (short) | Comparison plots |
| `test_manual_eskf.py` | ESKF on long dataset | ~1 min | ESKF performance |
| `test_fgo_fast.py` | Fast FGO on long dataset | ~30 min | FGO performance |

---

## Next Steps

1. **Generate short scenarios:**
   ```bash
   python scripts/generate_short_scenarios.py
   ```

2. **Update test_comparison.py run IDs** to match your newly generated short scenarios

3. **Run comparison:**
   ```bash
   python tests/test_comparison.py
   ```

4. **Copy plots to your report** from `output/`

5. **Run long ESKF** for stability demonstration:
   ```bash
   python tests/test_manual_eskf.py
   ```

Done! You now have all plots and data needed for your report. 🎉
