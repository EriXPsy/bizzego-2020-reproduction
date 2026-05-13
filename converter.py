"""
Bizzego et al. (2020) "Strangers, Friends, Lovers" — MultiSync Converter
========================================================================
Converts cardiac IBI (Inter-Beat Interval) data into standardized format
for multisync-core dynamic features analysis.

Dataset: Bizzego, A. et al. (2020). IEEE Trans. Affect. Comput.
  N = 62 dyads (F=Friends 23, U=Strangers 20, L=Lovers 21)
  × 6 video conditions (HS, TIT, WD, RELAX, NH, PEN)
  × ECG → IBI per person (M/F)

Key design decisions:
  1. IBI is event-based (unevenly sampled). We interpolate to 2Hz
     following Bizzego's preprocessing (Golland 2014 method).
  2. Each dyad × video = one analysis unit.
  3. Analysis window: [video_start + 10s, video_end] (skip first 10s).
  4. Within-dyad Z-score is applied before WCC computation.
  5. This converter extracts IBI only; EDA/EMG analysis is future work.

Architecture (standard 5-phase):
  PKL (IBI, TRG) → interpolate 2Hz → Z-score → WCC → dynamic features
  → validate against Bizzego's copresence/stimulus/surrogate results.

Usage:
  python converter.py
  python converter.py --max-dyads 10   # quick test
  python converter.py --validate        # compare with Bizzego's results

Dependencies:
  pip install pandas numpy scipy
  pip install -e /path/to/multisync-core
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, signal as sig_filter

warnings.filterwarnings("ignore")

# ── Try importing multisync-core ─────────────────────────────────────────
try:
    from multisync.dynamic_features import extract_dynamic_features
    from multisync.dynamic_features import sliding_window_wcc
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    print("[WARNING] multisync-core not installed. Running in data-only mode.")

# ── Constants ─────────────────────────────────────────────────────────────
PKL_ROOT   = Path("E:/OSF/Bizzego/Processed/pkl")
OUTPUT_DIR = Path("E:/OSF/Bizzego/multisync_results")

F_RESAMP   = 2     # IBI resampling frequency (Hz) — matches Bizzego/Golland 2014
SKIP_SEC   = 10    # Skip first 10s of each video (habituation)
MIN_BEATS  = 100   # Minimum heartbeat events per video segment (relaxed from 235)

VIDEO_NAMES = ['HS', 'TIT', 'WD', 'RELAX', 'NH', 'PEN']
VIDEO_EMOTIONS = ['EMBRS', 'SAD', 'FEAR', 'CALMNESS', 'LOVE', 'PRIDE']

GROUP_LABELS = {
    'U': 'Strangers',
    'F': 'Friends',
    'L': 'Lovers',
}

# Bizzego 05_compute_ibi_mean.py excludes L04
EXCLUDE_DYADS = {'L04'}

WIN_SEC  = 10.0   # WCC window (seconds) — 20 samples at 2Hz
STEP_SEC = 5.0    # WCC step (seconds) — 10 samples at 2Hz


# ═══════════════════════════════════════════════════════════════════════
# PKL LOADING (pyphysio v1 backward-compat)
# ═══════════════════════════════════════════════════════════════════════

def _setup_pkl_stub() -> None:
    """Create pyphysio.Signal stub so we can load old pkl files."""
    if 'pyphysio.Signal' in sys.modules:
        return
    sig_mod = types.ModuleType('pyphysio.Signal')
    sys.modules['pyphysio.Signal'] = sig_mod

    class _Base(np.ndarray):
        def __new__(cls, values, times, *args, **kwargs):
            obj = np.asarray(values).view(cls)
            obj._times = times
            obj._values = np.asarray(values)
            return obj
        def get_times(self): return self._times
        def get_values(self): return self._values
        def __array_finalize__(self, obj):
            if obj is not None:
                self._times = getattr(obj, '_times', None)
                self._values = getattr(obj, '_values', None)

    sig_mod.EvenlySignal = type('EvenlySignal', (_Base,), {})
    sig_mod.UnevenlySignal = type('UnevenlySignal', (_Base,), {})

_setup_pkl_stub()


def load_pkl(path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a pyphysio v1 pkl file (gzip + pickle protocol 2).
    Returns (values_array, metadata_dict).
    """
    with gzip.open(str(path), 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) == 2:
        vals, meta = obj
        return np.asarray(vals), dict(meta)
    elif isinstance(obj, dict):
        # Fallback: some files might be plain dicts
        return np.asarray(obj.get('values', [])), obj
    else:
        # Maybe it has get_values/get_times
        return obj.get_values(), obj.__dict__


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def discover_dyads() -> List[Tuple[str, str]]:
    """
    Discover all available dyads from IBI directory.
    Returns list of (dyad_id, group) tuples.
    """
    ibi_dir = PKL_ROOT / "ibi"
    files = sorted(os.listdir(ibi_dir))

    # Extract unique dyad IDs and their group
    dyad_set = set()
    for f in files:
        if not f.endswith('.pkl'):
            continue
        dyad_id = f.replace('_F.pkl', '').replace('_M.pkl', '')
        group = dyad_id[0]  # F, U, or L
        dyad_set.add((dyad_id, group))

    # Exclude problematic dyads
    dyads = [(d, g) for d, g in sorted(dyad_set) if d not in EXCLUDE_DYADS]
    return dyads


def load_ibi(dyad_id: str, gender: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load IBI signal for one person.
    Returns (ibi_values_seconds, beat_times_seconds, sampling_freq).
    """
    pkl_path = PKL_ROOT / "ibi" / f"{dyad_id}_{gender}.pkl"
    vals, meta = load_pkl(pkl_path)

    # IBI values are in seconds (inter-beat intervals)
    ibi_seconds = vals.astype(np.float64)

    # Beat times from x_values (sample indices) / sampling_freq
    x_values = meta.get('x_values', None)
    fs = float(meta.get('sampling_freq', 2048))

    if x_values is not None:
        beat_times = np.asarray(x_values, dtype=np.float64) / fs
    else:
        # Fallback: cumulative sum of IBI
        beat_times = np.concatenate([[0.0], np.cumsum(ibi_seconds[:-1])])

    return ibi_seconds, beat_times, fs


def load_trigger(dyad_id: str) -> Tuple[np.ndarray, float]:
    """
    Load trigger signal to identify video segments.
    Returns (trigger_values, sampling_freq).
    """
    pkl_path = PKL_ROOT / "trg" / f"{dyad_id}.pkl"
    vals, meta = load_pkl(pkl_path)
    fs = float(meta.get('sampling_freq', 2048))
    return vals.astype(np.float64), fs


def get_video_segments(dyad_id: str) -> List[Dict]:
    """
    Extract video segment boundaries from trigger signal.
    Returns list of dicts with video metadata.
    """
    trg_vals, fs = load_trigger(dyad_id)
    segments = []

    for vid in range(1, 7):
        idx = np.where(trg_vals == vid)[0]
        if len(idx) == 0:
            continue
        t_start_raw = idx[0] / fs
        t_end = idx[-1] / fs
        t_start_analysis = t_start_raw + SKIP_SEC  # skip habituation

        duration = t_end - t_start_analysis
        segments.append({
            'video_id': vid,
            'video_name': VIDEO_NAMES[vid - 1],
            'video_emotion': VIDEO_EMOTIONS[vid - 1],
            't_start_raw': t_start_raw,
            't_start': t_start_analysis,
            't_end': t_end,
            'duration': duration,
        })

    return segments


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: PREPROCESSING (IBI → evenly-sampled time series)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_ibi(
    ibi_seconds: np.ndarray,
    beat_times: np.ndarray,
    t_start: float,
    t_end: float,
) -> Optional[np.ndarray]:
    """
    Convert event-based IBI to evenly-sampled time series at F_RESAMP Hz.

    IBI is an event series: ibi_seconds[k] = time between beat k and beat k+1.
    beat_times[k] = absolute time of beat k.

    Strategy:
      1. Find all IBI intervals that fall (even partially) within [t_start, t_end].
      2. For each such interval, use its midpoint as the IBI sample time.
      3. Interpolate onto a regular 2Hz grid.
      4. Clip physiological range [0.3s, 2.0s].

    Returns interpolated IBI series at F_RESAMP Hz, or None if too short.
    """
    # Segment to analysis window
    mask = (beat_times >= t_start) & (beat_times <= t_end)
    n_beats_in_segment = np.sum(mask)

    if n_beats_in_segment < MIN_BEATS:
        return None

    # Build IBI midpoints: midpoint of interval [beat_times[k], beat_times[k+1]]
    # For the k-th IBI, midpoint = beat_times[k] + ibi_seconds[k] / 2
    # Only keep intervals whose midpoint falls within [t_start, t_end]
    midpoints = beat_times[:-1] + ibi_seconds[:len(beat_times)-1] / 2.0
    ibi_vals = ibi_seconds[:len(beat_times)-1]

    seg_mask = (midpoints >= t_start) & (midpoints <= t_end)
    seg_mid = midpoints[seg_mask]
    seg_ibi = ibi_vals[seg_mask]

    if len(seg_ibi) < MIN_BEATS:
        return None

    # Remove physiological outliers
    valid = np.isfinite(seg_ibi) & (seg_ibi > 0.3) & (seg_ibi < 2.0)
    if np.sum(valid) < 10:
        return None

    seg_mid = seg_mid[valid]
    seg_ibi = seg_ibi[valid]

    # Ensure strictly monotonic midpoints (deduplicate)
    _, unique_idx = np.unique(seg_mid, return_index=True)
    seg_mid = seg_mid[unique_idx]
    seg_ibi = seg_ibi[unique_idx]

    if len(seg_mid) < 3:
        return None

    # Create evenly-spaced time grid at F_RESAMP Hz
    duration = t_end - t_start
    n_points = int(duration * F_RESAMP)
    if n_points < 20:
        return None

    t_grid = np.linspace(t_start, t_end, n_points)

    # Interpolate (cubic if enough points, else linear)
    kind = 'cubic' if len(seg_mid) >= 4 else 'linear'
    interp_func = interpolate.interp1d(
        seg_mid, seg_ibi, kind=kind, bounds_error=False, fill_value='extrapolate'
    )
    ibi_interp = interp_func(t_grid)

    # Clip to physiological range and remove any remaining artifacts
    ibi_interp = np.clip(ibi_interp, 0.3, 2.0)
    ibi_interp = np.nan_to_num(ibi_interp, nan=0.0, posinf=2.0, neginf=0.3)

    return ibi_interp


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: PAIRING + WCC + DYNAMIC FEATURES
# ═══════════════════════════════════════════════════════════════════════

def _compute_wcc(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> np.ndarray:
    """Sliding-window Pearson correlation at 2Hz with WIN/STEP settings."""
    hz = F_RESAMP
    window_samples = int(WIN_SEC * hz)
    step_samples = int(STEP_SEC * hz)

    if HAS_CORE:
        return sliding_window_wcc(
            signal_a, signal_b,
            window_size=window_samples,
            hz=hz,
            step_samples=step_samples,
        )

    # Inline fallback
    n = min(len(signal_a), len(signal_b))
    a, b = signal_a[:n], signal_b[:n]
    wcc_values = []
    i = 0
    while i + window_samples <= n:
        wa, wb = a[i:i+window_samples], b[i:i+window_samples]
        r = np.corrcoef(wa, wb)[0, 1]
        wcc_values.append(r if np.isfinite(r) else 0.0)
        i += step_samples
    return np.array(wcc_values)


def _extract_features(wcc: np.ndarray) -> Dict:
    """Extract 6 dynamic features from WCC series."""
    if not HAS_CORE or len(wcc) < 5:
        return {
            'onset_latency': None, 'rise_time': None, 'peak_amplitude': None,
            'recovery_time': None, 'mean_synchrony': None, 'synchrony_entropy': None,
        }
    try:
        feats = extract_dynamic_features(wcc, hz=1.0 / STEP_SEC)
        if isinstance(feats, dict):
            return feats
        elif hasattr(feats, '__dict__'):
            return dict(feats.__dict__)
        else:
            return {k: getattr(feats, k, None) for k in [
                'onset_latency', 'rise_time', 'peak_amplitude',
                'recovery_time', 'mean_synchrony', 'synchrony_entropy',
            ]}
    except Exception as e:
        return {k: None for k in [
            'onset_latency', 'rise_time', 'peak_amplitude',
            'recovery_time', 'mean_synchrony', 'synchrony_entropy',
        ]}


def process_dyad(dyad_id: str, group: str) -> List[Dict]:
    """
    Process one dyad: all 6 videos × IBI modality.
    Returns list of row dicts.
    """
    segments = get_video_segments(dyad_id)
    if not segments:
        return []

    # Load IBI for both persons
    try:
        ibi_m, beats_m, fs_m = load_ibi(dyad_id, 'M')
        ibi_f, beats_f, fs_f = load_ibi(dyad_id, 'F')
    except Exception as e:
        print(f"  [WARN] {dyad_id}: IBI load error: {e}")
        return []

    rows = []
    for seg in segments:
        vid = seg['video_id']
        t_start, t_end = seg['t_start'], seg['t_end']

        # Preprocess both persons
        ibi_series_m = preprocess_ibi(ibi_m, beats_m, t_start, t_end)
        ibi_series_f = preprocess_ibi(ibi_f, beats_f, t_start, t_end)

        if ibi_series_m is None or ibi_series_f is None:
            print(f"  [SKIP] {dyad_id} V{vid}: insufficient data")
            continue

        n = min(len(ibi_series_m), len(ibi_series_f))
        ibi_series_m = ibi_series_m[:n]
        ibi_series_f = ibi_series_f[:n]

        # Within-dyad Z-score (each person separately)
        z_m = (ibi_series_m - np.mean(ibi_series_m)) / np.std(ibi_series_m) if np.std(ibi_series_m) > 0 else ibi_series_m
        z_f = (ibi_series_f - np.mean(ibi_series_f)) / np.std(ibi_series_f) if np.std(ibi_series_f) > 0 else ibi_series_f

        # Compute WCC
        wcc = _compute_wcc(z_m, z_f)

        # Extract dynamic features
        feats = _extract_features(wcc)

        row = {
            'dyad_id':     dyad_id,
            'group':       group,
            'group_label': GROUP_LABELS.get(group, group),
            'video_id':    vid,
            'video_name':  seg['video_name'],
            'video_emotion': seg['video_emotion'],
            'duration_sec': seg['duration'],
            'n_samples':   n,
            'n_wcc_points': len(wcc),
            'mean_ibi_M':  float(ibi_series_m.mean()),
            'mean_ibi_F':  float(ibi_series_f.mean()),
            'hr_M':        float(60.0 / ibi_series_m.mean()),
            'hr_F':        float(60.0 / ibi_series_f.mean()),
            'wcc_mean':    float(wcc.mean()) if len(wcc) > 0 else None,
        }
        row.update(feats)

        # Store WCC series for downstream analysis
        if len(wcc) > 0:
            row['wcc_series'] = json.dumps(
                [round(float(x), 4) if np.isfinite(x) else None for x in wcc]
            )

        rows.append(row)

    return rows


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_results(df: pd.DataFrame) -> None:
    """
    Compare MultiSync results with Bizzego's key findings:
    1. Lovers > Friends > Strangers IBI synchrony gradient
    2. Video-specific effects (LOVE highest for Lovers?)
    """
    print("\n" + "=" * 60)
    print("VALIDATION: MultiSync vs Bizzego (2020)")
    print("=" * 60)

    # 1. Group-level WCC comparison
    print("\n── 1. WCC Mean by Group ──")
    for grp in ['U', 'F', 'L']:
        label = GROUP_LABELS[grp]
        sub = df[df['group'] == grp]
        wcc_mean = sub['wcc_mean'].dropna()
        if len(wcc_mean) > 0:
            m, s = wcc_mean.mean(), wcc_mean.std()
            print(f"  {label:10s}: M={m:.4f}, SD={s:.4f}, N={len(wcc_mean)}")

    # Kruskal-Wallis tests (replicating Bizzego's 04_test_cop_UFL.py)
    from scipy.stats import kruskal
    groups_wcc = []
    for grp in ['U', 'F', 'L']:
        groups_wcc.append(df[df['group'] == grp]['wcc_mean'].dropna().values)
    if all(len(g) > 2 for g in groups_wcc):
        stat, p = kruskal(*groups_wcc)
        print(f"\n  Kruskal-Wallis H={stat:.3f}, p={p:.4f}")

        # Pairwise
        from scipy.stats import mannwhitneyu
        pairs = [('U', 'F'), ('F', 'L'), ('U', 'L')]
        for g1, g2 in pairs:
            u, p_val = mannwhitneyu(groups_wcc[0 if g1=='U' else 1 if g1=='F' else 2],
                                     groups_wcc[0 if g2=='U' else 1 if g2=='F' else 2],
                                     alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            print(f"  {GROUP_LABELS[g1]:10s} vs {GROUP_LABELS[g2]:10s}: U={u:.1f}, p={p_val:.4f} {sig}")

    # 2. Dynamic features by group
    print("\n── 2. Dynamic Features by Group ──")
    feat_cols = ['onset_latency', 'rise_time', 'peak_amplitude', 'recovery_time', 'mean_synchrony', 'synchrony_entropy']
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        print(f"\n  {feat}:")
        for grp in ['U', 'F', 'L']:
            label = GROUP_LABELS[grp]
            vals = df[df['group'] == grp][feat].dropna()
            if len(vals) > 0:
                m, s = vals.mean(), vals.std()
                print(f"    {label:10s}: M={m:.4f}, SD={s:.4f}, N={len(vals)}")

        # KW test for this feature
        feat_groups = []
        for grp in ['U', 'F', 'L']:
            v = df[df['group'] == grp][feat].dropna().values
            if len(v) > 2:
                feat_groups.append(v)
        if len(feat_groups) >= 2:
            try:
                stat, p = kruskal(*feat_groups)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                print(f"    KW H={stat:.3f}, p={p:.4f} {sig}")
            except:
                pass

    # 3. Video-level effects
    print("\n── 3. WCC Mean by Video × Group ──")
    pivot = df.pivot_table(values='wcc_mean', index='video_name', columns='group', aggfunc='mean')
    print(pivot.round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bizzego 2020 → MultiSync converter")
    parser.add_argument('--max-dyads', type=int, default=None, help='Max dyads to process')
    parser.add_argument('--validate', action='store_true', help='Run validation comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("Bizzego (2020) Converter — MultiSync")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Discover dyads
    dyads = discover_dyads()
    print(f"\nFound {len(dyads)} dyads:")
    for grp in ['F', 'U', 'L']:
        grp_dyads = [d for d, g in dyads if g == grp]
        print(f"  {GROUP_LABELS[grp]:10s}: {len(grp_dyads)} ({grp_dyads[:3]}...)")

    if args.max_dyads:
        dyads = dyads[:args.max_dyads]
        print(f"\n[TEST MODE] Processing first {len(dyads)} dyads")

    # Process all dyads
    all_rows = []
    for dyad_id, group in dyads:
        print(f"\nProcessing {dyad_id} ({GROUP_LABELS[group]})...")
        rows = process_dyad(dyad_id, group)
        all_rows.extend(rows)
        print(f"  → {len(rows)} video segments")

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    print(f"\n{'=' * 60}")
    print(f"Total: {len(df)} dyad-video segments")
    print(f"Dyads: {df['dyad_id'].nunique()}")
    print(f"Groups: {df['group'].value_counts().to_dict()}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main results CSV (without WCC series — for stats)
    cols_no_wcc = [c for c in df.columns if c != 'wcc_series']
    df[cols_no_wcc].to_csv(OUTPUT_DIR / "bizzego_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'bizzego_results.csv'}")

    # Per-video WCC series (for visualization)
    wcc_cols = ['dyad_id', 'group', 'video_id', 'video_name', 'wcc_series']
    df_wcc = df[wcc_cols].dropna(subset=['wcc_series'])
    df_wcc.to_csv(OUTPUT_DIR / "bizzego_wcc_series.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'bizzego_wcc_series.csv'}")

    # Validation
    if args.validate:
        validate_results(df)


if __name__ == "__main__":
    main()
