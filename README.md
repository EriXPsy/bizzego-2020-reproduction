# Reproduction of Bizzego et al. (2020) IAAFT Surrogate Analysis

**Status**: Partial reproduction — IBI surr<stim test failing (0/18 vs expected ~15-16/18)

**Goal**: Faithfully replicate the copresence/stimulus/surrogate three-tier comparison from Bizzego et al. (2020) "Strangers, Friends, Lovers," IEEE Trans. Affect. Comput.

## Pipeline Overview

```
IBI → resample 2Hz → IIR bandpass (0.04-0.1 Hz) → Z-score → IAAFT surrogate → 5s convolve → CC distance (Golland 2014) → MWU test
```

## Files

### Main Analysis
- **`iaaft_analysis.py`** — Full reproduction script (IBI + EDA + EMG)
  - Custom `cc_distance()` with per-lag re-standardization (pyphysio v1 replica)
  - Custom `iaaft_surrogate()` (Schreiber & Schmitz 1996)
  - Custom `compute_distances_golland()` — three-tier distance computation
  - All `np.convolve` calls use default `mode='full'` (matching Bizzego)

### Original Scripts (Reference)
- **`scripts_original/03_sync_ibi.py`** — Bizzego's IBI sync script
- **`scripts_original/04_test_cop-stim-surr.py`** — Bizzego's statistical test
- **`scripts_original/config.py`** — Bizzego's config

### Results (Current Run)
- **`results/iaaft_three_tier_test.csv`** — IBI three-tier test: **0/18 surr<stim significant**
- **`results/iaaft_cross_group_test.csv`** — IBI cross-group KW test: 0/6 significant
- **`results/iaaft_eda_three_tier_test.csv`** — EDA results (exploratory)
- **`results/iaaft_emg_three_tier_test.csv`** — EMG results (exploratory)
- **`results/iaaft_cross_modal_comparison.csv`** — Cross-modal comparison

## Bugfix History

| Version | Error | Fix | Effect |
|---------|-------|-----|--------|
| v1 | `abs(r)` + `corrcoef()/n` + `best_r=0.0` | Removed abs, /n, 0-init | Values now in 0.25-0.33 range (was ~0.001) |
| v2 | Used `np.corrcoef` directly | Same as v1 but cleaner | Still 0/18 surr<stim |
| v3 (current) | Per-lag re-standardization + dot-product mean | Matches pyphysio v3 `lagged_cross_corr` | Still 0/18 surr<stim |

## The Unsolved Problem

### Expected Results (Bizzego 2020 Paper)
- **surr < stim**: ~15-16/18 significant (after BH correction)
- **stim < cop**: ~4/18 significant (before BH correction)

### Current Results (Our Reproduction)
- **surr < stim**: 0/18 significant
- **stim < cop**: 1/18 significant (Lovers FEAR, p=0.014)

### Symptom Pattern
The surrogate mean *exceeds* the stimulus mean in many conditions:

| Condition | Surr Mean | Stim Mean | Direction |
|-----------|-----------|-----------|-----------|
| Friends HS | 0.332 | 0.251 | surr > stim (wrong) |
| Friends RELAX | 0.307 | 0.267 | surr > stim (wrong) |
| Strangers HS | 0.295 | 0.258 | surr > stim (wrong) |

This pattern should be reversed: actual-vs-surrogate cross-dyad distances should be *lower* than surrogate-vs-surrogate.

### What Has Been Verified
1. `cc_distance()` produces identical results to `np.corrcoef` for same-length signals (diagnostic test passed)
2. Per-lag re-standardization matches pyphysio v3 `lagged_cross_corr` logic
3. IAAFT surrogate preserves power spectrum (verified via spectral comparison)
4. Convolve uses `mode='full'` matching Bizzego's `np.convolve(s, kernel)` (no mode arg)
5. One-sided MWU test: `mannwhitneyu(surr, stim, alternative='less')` — matches Bizzego

### Hypotheses for Remaining Discrepancy

1. **`compute_distances_golland` internal logic (pyphysio v1)**:
   - This function was removed in pyphysio v3 and never published to PyPI
   - Bizzego calls: `compute_distances_golland(group_1_v, group_2_v, 'cc', 20, False, True)`
   - The `standardize=False` parameter is passed but its effect is unknown
   - The cross-dyad stimulus/surrogate pairing strategy (upper triangle? mean? per-dyad?) may differ

2. **IIRFilter implementation difference**:
   - Bizzego uses `ph.IIRFilter(fp=0.04, fs=0.1)` — pyphysio v1 internal implementation
   - We use `scipy.signal.butter(4, [0.04/nyq, 0.1/nyq], btype='band')` + `filtfilt`
   - Filter order may differ (pyphysio default vs our order=4)

3. **Resampling difference**:
   - Bizzego uses pyphysio's `signal.resample(F_RESAMP)` on `UnevenlySignal`
   - We use scipy `interp1d` with cubic interpolation on IBI midpoints
   - Different resampling can produce different 2Hz series, affecting all downstream results

4. **Segment boundary handling**:
   - Bizzego: `signal_M.segment_time(t_start, t_stop).get_values()`
   - Our implementation: `ibi_to_2hz()` which reconstructs IBI midpoints
   - Boundary samples may differ slightly

## Specific Questions for External Review

1. Is there a way to obtain pyphysio v1 `compute_distances_golland` source code? It was in `pyphysio.metrics.Metrics` but the module was removed.

2. Does the per-dyad cross-dyad aggregation in `compute_distances_golland` differ from our implementation? We do:
   ```python
   for i in range(n):
       mask = np.ones(n, dtype=bool); mask[i] = False
       stimulus[i] = np.mean(stimulus_matrix[i, mask])
       surrogate[i] = np.mean(surrogate_matrix[i, mask])
   ```

3. Could the `standardize=False` parameter in the call toggle additional re-standardization logic that we're missing?

4. Is there a known issue with the combination of per-lag re-standardization + `max(c)` (not `max(|c|)`) that would cause surrogates to systematically exceed stimulus distances?

## Data

Data is from OSF (https://osf.io/349su/). Not included in this repo due to size and licensing.

Path structure on local machine:
```
E:/OSF/Bizzego/
├── Processed/pkl/ibi/     # IBI data (per dyad, per gender)
├── Processed/pkl/trg/     # Trigger data (video boundaries)
├── RawData_1/ ... RawData_3/  # Raw .mat files (EDA, EMG)
└── Scripts/               # Bizzego's original scripts
```

## Running

```bash
# IBI analysis (full)
python iaaft_analysis.py --modality ibi

# Quick test (10 dyads)
python iaaft_analysis.py --modality ibi --max-dyads 10

# All modalities
python iaaft_analysis.py --modality all
```

**Dependencies**: numpy, scipy, pandas, scipy.io (for .mat files)

## References

- Bizzego, A. et al. (2020). "Strangers, Friends, and Lovers: The Effect of Relationship Quality on Physiological Synchrony." IEEE Trans. Affect. Comput.
- Golland, Y. et al. (2014). "A novel approach to analyzing the neural correlates of dyadic interaction." *Computational Intelligence and Neuroscience*.
- Schreiber, T. & Schmitz, A. (1996). "Improved surrogate data for nonlinearity tests." *Physical Review Letters*, 77(4), 635.
