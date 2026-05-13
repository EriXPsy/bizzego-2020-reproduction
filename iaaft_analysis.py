"""
Bizzego (2020) IAAFT Surrogate Analysis
=========================================
Replicates Bizzego's original copresence/stimulus/surrogate three-tier
comparison using IAAFT surrogates + CC distance (Golland 2014).

Then extends to EDA/EMG cross-modal synchrony.

Method (from Bizzego's 03_sync_ibi.py + 04_test_cop-stim-surr.py):
  1. Load IBI → resample to 2Hz → low-pass filter (0.04-0.1 Hz) → Z-score
  2. Generate IAAFT surrogate for each person's signal
  3. Smooth surrogate with 5s convolution window (CONV_WINDOW=5)
  4. CC distance = max lagged Pearson r over lag ∈ [-10s, +10s]
     (NO absolute value, NO /n normalization — matches Bizzego's
      compute_distances_golland with absolute=False)
  5. Three comparisons:
     - Copresence: actual_A vs actual_B (within dyad)
     - Stimulus:   actual_A vs surrogate_B (across dyads, same video)
     - Surrogate:  surrogate_A vs surrogate_B (across dyads, same video)
  6. Test: surrogate < stimulus < copresence (one-sided MWU)

Usage:
  python iaaft_analysis.py
  python iaaft_analysis.py --modality ibi     # IBI only (default)
  python iaaft_analysis.py --modality eda     # EDA cross-modal
  python iaaft_analysis.py --modality emg     # EMG cross-modal
  python iaaft_analysis.py --modality all     # All three
  python iaaft_analysis.py --max-dyads 10     # Quick test
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import types
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, signal as sig_filter, stats

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────
PKL_ROOT   = Path("E:/OSF/Bizzego/Processed/pkl")
RAW_ROOT   = Path("E:/OSF/Bizzego")
OUTPUT_DIR = Path("E:/OSF/Bizzego/multisync_results")

F_RESAMP   = 2     # Resampling frequency (Hz)
SKIP_SEC   = 10    # Skip first 10s (habituation)
MIN_SAMPLES = 235 * F_RESAMP  # Minimum samples (Bizzego threshold: 235 beats * 2 samples/beat)
MAX_DELAY  = 10    # Max lag for CC distance (seconds)
LAG        = MAX_DELAY * F_RESAMP  # Max lag in samples
CONV_WINDOW = 5    # Smoothing window for surrogate (seconds)
LP_FP      = 0.04  # Low-pass filter, passband (Hz)
LP_FS      = 0.1   # Low-pass filter, stopband (Hz)

VIDEO_NAMES = ['HS', 'TIT', 'WD', 'RELAX', 'NH', 'PEN']
VIDEO_EMOTIONS = ['EMBRS', 'SAD', 'FEAR', 'CALMNESS', 'LOVE', 'PRIDE']

GROUP_LABELS = {'U': 'Strangers', 'F': 'Friends', 'L': 'Lovers'}
EXCLUDE_DYADS = {'L04'}

RAW_FS = 2048  # Raw mat file sampling rate (Hz)


# ═══════════════════════════════════════════════════════════════════════
# PKL LOADING (pyphysio v1 backward-compat)
# ═══════════════════════════════════════════════════════════════════════

def _setup_pkl_stub():
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


def load_pkl(path):
    with gzip.open(str(path), 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) == 2:
        return np.asarray(obj[0]), dict(obj[1])
    elif isinstance(obj, dict):
        return np.asarray(obj.get('values', [])), obj
    else:
        return obj.get_values(), obj.__dict__


# ═══════════════════════════════════════════════════════════════════════
# IAAFT SURROGATE (Schreiber & Schmitz 1996)
# ═══════════════════════════════════════════════════════════════════════

def iaaft_surrogate(x, maxiter=1000, atol=1e-8, rtol=1e-10):
    """
    Iterative Amplitude Adjusted Fourier Transform surrogate.

    Generates a surrogate time series with the SAME power spectrum and
    SAME amplitude distribution as the original signal.

    Reference:
      Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for
      nonlinearity tests. Physical Review Letters, 77(4), 635.

    Source: pyphysio compare.py (nolitsa/surrogates.py)

    Parameters
    ----------
    x : np.ndarray — 1-D input time series
    maxiter : int — max iterations
    atol, rtol : float — convergence tolerances

    Returns
    -------
    y : np.ndarray — surrogate series
    n_iter : int — iterations performed
    rmsd : float — normalized RMSD of power spectra
    """
    x = np.asarray(x, dtype=np.float64)

    # "True" Fourier amplitudes and sorted values
    ampl = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)

    perr, cerr = -1.0, 1.0

    # Start with random permutation
    t = np.fft.rfft(np.random.permutation(x))

    for i in range(maxiter):
        # Match power spectrum
        s = np.real(np.fft.irfft(ampl * t / np.abs(t), n=x.shape[0]))

        # Match distribution by rank ordering
        y = sorted_x[np.argsort(np.argsort(s))]

        t = np.fft.rfft(y)
        cerr = np.sqrt(np.mean((ampl ** 2 - np.abs(t) ** 2) ** 2))

        if abs(cerr - perr) <= atol + rtol * abs(perr):
            break
        perr = cerr

    return y, i, cerr / np.mean(ampl ** 2)


def compute_IAAFT_surrogates(x):
    """Return only the surrogate series (convenience wrapper)."""
    y, _, _ = iaaft_surrogate(x)
    return y


# ═══════════════════════════════════════════════════════════════════════
# CC DISTANCE (Golland 2014)
# ═══════════════════════════════════════════════════════════════════════

def _get_lagged(data1, data2, idx_lag):
    """Extract lag-aligned slices from two signals."""
    if idx_lag == 0:
        return data1, data2
    elif idx_lag > 0:
        return data1[:-idx_lag], data2[idx_lag:]
    else:
        idx_lag = -idx_lag
        return data1[idx_lag:], data2[:-idx_lag]


def cc_distance(x, y, max_lag=LAG, normalize=True):
    """
    CC distance (Golland et al. 2014) — precise pyphysio v1 replica.

    Faithfully reproduces pyphysio's lagged_cross_corr behavior:
      1. For each lag, extract the aligned sub-window.
      2. Re-standardize EACH sub-window independently (local Z-score).
      3. Compute mean dot-product (= Pearson r for re-standardized windows).
      4. Take max(r) WITH sign (absolute=False), not max(|r|).

    Bugfix history (2026-05-13):
      v1: abs(r) + corrcoef / n + best_r=0.0 → 0/18 surr<stim (catastrophic)
      v2: removed abs/n/0-init, used corrcoef directly → still 0/18
      v3 (current): per-lag re-standardization + dot-product-mean → matches pyphysio

    Parameters
    ----------
    x, y : np.ndarray — 1-D time series (pre-Z-scored globally)
    max_lag : int — max lag in samples
    normalize : bool — kept for API compat; ignored (dot-product/N is the normalization)

    Returns
    -------
    dist : float — CC distance (higher = more synchronous)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    best_dist = float('-inf')
    n1, n2 = len(x), len(y)

    for curr_lag in range(-max_lag, max_lag + 1):
        # Compute aligned window boundaries
        start1 = max(0, -curr_lag)
        end1 = min(n1, n2 - curr_lag)
        start2 = max(0, curr_lag)
        end2 = min(n2, n1 + curr_lag)

        if end1 <= start1 or end2 <= start2:
            continue

        x_lag = x[start1:end1]
        y_lag = y[start2:end2]

        # Per-lag re-standardization (pyphysio lagged_cross_corr core logic)
        std_x = np.std(x_lag)
        std_y = np.std(y_lag)
        if std_x == 0 or std_y == 0:
            continue
        x_lag = (x_lag - np.mean(x_lag)) / std_x
        y_lag = (y_lag - np.mean(y_lag)) / std_y

        # Dot-product mean = Pearson r for re-standardized windows
        # Equivalent to scipy.signal.correlate / N
        c = np.sum(x_lag * y_lag) / len(x_lag)

        # absolute=False → take max positive correlation, not max |r|
        if c > best_dist:
            best_dist = c

    return best_dist if best_dist != float('-inf') else np.nan


# ═══════════════════════════════════════════════════════════════════════
# GOLLAND DISTANCE COMPUTATION (three-tier)
# ═══════════════════════════════════════════════════════════════════════

def compute_distances_golland(group1, group2, distance, lag, standardize, normalize):
    """
    Compute copresence, stimulus, and surrogate distances.

    This replicates the behavior of pyphysio.metrics.Metrics.compute_distances_golland
    as called in Bizzego's 03_sync_ibi.py.

    group1_v[i] = [actual_signal_i, surrogate_signal_i]  (e.g., Males)
    group2_v[i] = [actual_signal_i, surrogate_signal_i]  (e.g., Females)

    Returns
    -------
    copresence : np.ndarray — within-dyad actual-vs-actual distances
    stimulus : np.ndarray — cross-dyad actual-vs-surrogate (mean)
    surrogate : np.ndarray — cross-dyad surrogate-vs-surrogate (mean)
    """
    n = len(group1)
    if n != len(group2):
        raise ValueError(f"group1 ({n}) and group2 ({len(group2)}) must have same length")

    copresence = np.zeros(n)
    stimulus_matrix = np.zeros((n, n))
    surrogate_matrix = np.zeros((n, n))

    for i in range(n):
        x_actual_i = group1[i][0]  # actual signal for person i
        x_surr_i = group1[i][1]    # surrogate signal for person i
        y_actual_i = group2[i][0]  # actual signal for partner i
        y_surr_i = group2[i][1]    # surrogate signal for partner i

        # Copresence: within-dyad, actual vs actual
        copresence[i] = cc_distance(x_actual_i, y_actual_i, max_lag=lag, normalize=normalize)

        for j in range(n):
            if i == j:
                continue
            x_actual_j = group1[j][0]
            x_surr_j = group1[j][1]
            y_actual_j = group2[j][0]
            y_surr_j = group2[j][1]
            # Stimulus: actual_i vs surrogate_j (cross-dyad)
            stimulus_matrix[i, j] = cc_distance(x_actual_i, y_surr_j, max_lag=lag, normalize=normalize)

            # Surrogate: surrogate_i vs surrogate_j (cross-dyad)
            surrogate_matrix[i, j] = cc_distance(x_surr_i, y_surr_j, max_lag=lag, normalize=normalize)

    # Collapse cross-dyad matrices: take upper triangle mean per dyad (column-wise)
    # Following Bizzego's implementation: stimulus and surrogate are per-video vectors
    # The compute_distances_golland returns them as (video-level) means
    stimulus = np.zeros(n)
    surrogate = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        stimulus[i] = np.mean(stimulus_matrix[i, mask])
        surrogate[i] = np.mean(surrogate_matrix[i, mask])

    return copresence, stimulus, np.zeros(n), np.zeros(n), surrogate


# ═══════════════════════════════════════════════════════════════════════
# IBI SIGNAL PROCESSING (from Bizzego's pipeline)
# ═══════════════════════════════════════════════════════════════════════

def discover_dyads():
    """Discover all available dyads from IBI directory."""
    ibi_dir = PKL_ROOT / "ibi"
    files = sorted(os.listdir(ibi_dir))
    dyad_set = set()
    for f in files:
        if not f.endswith('.pkl'):
            continue
        dyad_id = f.replace('_F.pkl', '').replace('_M.pkl', '')
        group = dyad_id[0]
        dyad_set.add((dyad_id, group))
    dyads = [(d, g) for d, g in sorted(dyad_set) if d not in EXCLUDE_DYADS]
    return dyads


def load_ibi(dyad_id, gender):
    """Load IBI signal for one person from pkl."""
    pkl_path = PKL_ROOT / "ibi" / f"{dyad_id}_{gender}.pkl"
    vals, meta = load_pkl(pkl_path)
    ibi_seconds = vals.astype(np.float64)
    x_values = meta.get('x_values', None)
    fs = float(meta.get('sampling_freq', 2048))
    if x_values is not None:
        beat_times = np.asarray(x_values, dtype=np.float64) / fs
    else:
        beat_times = np.concatenate([[0.0], np.cumsum(ibi_seconds[:-1])])
    return ibi_seconds, beat_times, fs


def load_trigger(dyad_id):
    """Load trigger signal from pkl."""
    pkl_path = PKL_ROOT / "trg" / f"{dyad_id}.pkl"
    vals, meta = load_pkl(pkl_path)
    fs = float(meta.get('sampling_freq', 2048))
    return vals.astype(np.float64), fs


def get_video_segments_pkl(dyad_id):
    """Extract video segment boundaries from processed pkl trigger."""
    trg_vals, fs = load_trigger(dyad_id)
    segments = []
    for vid in range(1, 7):
        idx = np.where(trg_vals == vid)[0]
        if len(idx) == 0:
            continue
        t_start_raw = idx[0] / fs
        t_end = idx[-1] / fs
        t_start_analysis = t_start_raw + SKIP_SEC
        segments.append({
            'video_id': vid,
            'video_name': VIDEO_NAMES[vid - 1],
            'video_emotion': VIDEO_EMOTIONS[vid - 1],
            't_start': t_start_analysis,
            't_end': t_end,
            'duration': t_end - t_start_analysis,
        })
    return segments


def preprocess_ibi_signal(signal_2hz):
    """
    Apply Bizzego's preprocessing to a 2Hz IBI signal:
    1. Low-pass IIR filter (fp=0.04, fs=0.1 Hz)
    2. Z-score (standard_scale)
    """
    if len(signal_2hz) < MIN_SAMPLES:
        return None

    # Low-pass filter (Bizzego: ph.IIRFilter(fp=0.04, fs=0.1))
    # This is a very low-frequency filter that retains only < 0.04 Hz components
    nyq = F_RESAMP / 2.0
    low = LP_FP / nyq
    high = LP_FS / nyq
    if high >= 1.0:
        high = 0.99
    b, a = sig_filter.butter(4, [low, high], btype='band')
    filtered = sig_filter.filtfilt(b, a, signal_2hz)

    # Standard scale (Z-score)
    std = np.std(filtered)
    if std > 1e-10:
        filtered = (filtered - np.mean(filtered)) / std
    else:
        filtered = filtered - np.mean(filtered)

    return filtered


def ibi_to_2hz(ibi_seconds, beat_times, t_start, t_end):
    """Convert event-based IBI to 2Hz evenly-sampled time series."""
    mask = (beat_times[:-1] >= t_start) & (beat_times[:-1] + ibi_seconds[:len(beat_times)-1] <= t_end)
    midpoints = beat_times[:-1] + ibi_seconds[:len(beat_times)-1] / 2.0
    ibi_vals = ibi_seconds[:len(beat_times)-1]

    seg_mask = (midpoints >= t_start) & (midpoints <= t_end)
    seg_mid = midpoints[seg_mask]
    seg_ibi = ibi_vals[seg_mask]

    # Remove outliers
    valid = np.isfinite(seg_ibi) & (seg_ibi > 0.3) & (seg_ibi < 2.0)
    seg_mid = seg_mid[valid]
    seg_ibi = seg_ibi[valid]

    _, unique_idx = np.unique(seg_mid, return_index=True)
    seg_mid = seg_mid[unique_idx]
    seg_ibi = seg_ibi[unique_idx]

    if len(seg_mid) < 3:
        return None

    duration = t_end - t_start
    n_points = int(duration * F_RESAMP)
    if n_points < 20:
        return None

    t_grid = np.linspace(t_start, t_end, n_points)
    kind = 'cubic' if len(seg_mid) >= 4 else 'linear'
    interp_func = interpolate.interp1d(seg_mid, seg_ibi, kind=kind, bounds_error=False, fill_value='extrapolate')
    ibi_interp = interp_func(t_grid)
    ibi_interp = np.clip(ibi_interp, 0.3, 2.0)
    ibi_interp = np.nan_to_num(ibi_interp, nan=0.0, posinf=2.0, neginf=0.3)

    return ibi_interp


# ═══════════════════════════════════════════════════════════════════════
# EDA/EMG SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def load_raw_signal(dyad_id, gender, modality):
    """
    Load raw EDA or EMG signal from .mat file.

    Parameters
    ----------
    dyad_id : str — e.g., 'F01'
    gender : str — 'M' or 'F'
    modality : str — 'eda' or 'emg'
    """
    group = dyad_id[0]
    raw_dir = RAW_ROOT / f"RawData_{'1' if group == 'F' else '2' if group == 'U' else '3'}" / dyad_id
    mat_path = raw_dir / f"{modality}{gender}.mat"

    if not mat_path.exists():
        return None, None

    import scipy.io as sio
    data = sio.loadmat(str(mat_path))
    key = f"{modality}{gender}"
    signal = data[key].flatten().astype(np.float64)
    return signal, RAW_FS


def get_video_segments_raw(dyad_id):
    """
    Get video segments from raw trigger signal (.mat file).
    Raw trigger has continuous values, we use the processed pkl trigger
    to find video boundaries (both are at fs=2048).
    """
    trg_vals, fs = load_trigger(dyad_id)
    segments = []
    for vid in range(1, 7):
        idx = np.where(trg_vals == vid)[0]
        if len(idx) == 0:
            continue
        t_start_raw = idx[0] / fs
        t_end = idx[-1] / fs
        t_start_analysis = t_start_raw + SKIP_SEC
        segments.append({
            'video_id': vid,
            'video_name': VIDEO_NAMES[vid - 1],
            'video_emotion': VIDEO_EMOTIONS[vid - 1],
            'sample_start': int(t_start_analysis * fs),
            'sample_end': int(t_end * fs),
            't_start': t_start_analysis,
            't_end': t_end,
            'duration': t_end - t_start_analysis,
        })
    return segments


def preprocess_continuous_signal(raw_signal, t_start, t_end, fs, modality='eda'):
    """
    Preprocess EDA or EMG signal for synchrony analysis.

    Steps:
    1. Segment to [t_start, t_end]
    2. Resample to 2Hz
    3. Z-score
    """
    sample_start = int(t_start * fs)
    sample_end = int(t_end * fs)

    if sample_end > len(raw_signal):
        sample_end = len(raw_signal)
    if sample_start >= sample_end:
        return None

    segment = raw_signal[sample_start:sample_end]

    # Resample to 2Hz using scipy resample
    n_target = int(len(segment) * F_RESAMP / fs)
    if n_target < MIN_SAMPLES:
        return None

    resampled = sig_filter.resample(segment, num=n_target)

    # Z-score
    std = np.std(resampled)
    if std > 1e-10:
        resampled = (resampled - np.mean(resampled)) / std
    else:
        resampled = resampled - np.mean(resampled)

    return resampled


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_ibi_analysis(dyads, max_dyads=None):
    """
    Run full copresence/stimulus/surrogate analysis for IBI.

    Returns dict of {relation: {copresence_df, stimulus_df, surrogate_df}}.
    """
    if max_dyads:
        dyads = dyads[:max_dyads]

    print(f"\n{'='*60}")
    print("IBI ANALYSIS — Copresence / Stimulus / Surrogate")
    print(f"Dyads: {len(dyads)}, Videos: 6")
    print(f"IAAFT surrogate + CC distance (Golland 2014)")
    print(f"{'='*60}")

    # Phase 1: Load and preprocess all signals
    all_data = {}
    for RELATION in ['U', 'F', 'L']:
        selected_dyads = [d for d, g in dyads if g == RELATION]
        data_rel = {}
        for D in selected_dyads:
            print(f"  Loading {D}...", end="", flush=True)
            try:
                ibi_m, beats_m, fs_m = load_ibi(D, 'M')
                ibi_f, beats_f, fs_f = load_ibi(D, 'F')
                segments = get_video_segments_pkl(D)
            except Exception as e:
                print(f" ERROR: {e}")
                continue

            data_dyad = {}
            for seg in segments:
                vid = seg['video_id']
                t_start, t_end = seg['t_start'], seg['t_end']

                # Convert IBI to 2Hz and preprocess
                raw_2hz_m = ibi_to_2hz(ibi_m, beats_m, t_start, t_end)
                raw_2hz_f = ibi_to_2hz(ibi_f, beats_f, t_start, t_end)

                if raw_2hz_m is None or raw_2hz_f is None:
                    continue
                if len(raw_2hz_m) < MIN_SAMPLES or len(raw_2hz_f) < MIN_SAMPLES:
                    continue

                # Bizzego's pipeline: resample → LP filter → Z-score
                proc_m = preprocess_ibi_signal(raw_2hz_m)
                proc_f = preprocess_ibi_signal(raw_2hz_f)

                if proc_m is None or proc_f is None:
                    continue

                # Generate IAAFT surrogate + smooth with CONV_WINDOW
                surr_m = compute_IAAFT_surrogates(proc_m)
                surr_m = np.convolve(surr_m, np.ones(int(CONV_WINDOW * F_RESAMP)) / int(CONV_WINDOW * F_RESAMP))

                surr_f = compute_IAAFT_surrogates(proc_f)
                surr_f = np.convolve(surr_f, np.ones(int(CONV_WINDOW * F_RESAMP)) / int(CONV_WINDOW * F_RESAMP))

                data_dyad[vid] = {'M': [proc_m, surr_m], 'F': [proc_f, surr_f]}

            data_rel[D] = data_dyad
            n_videos = len(data_dyad)
            print(f" {n_videos} videos OK")

        all_data[RELATION] = data_rel

    # Phase 2: Compute distances
    results = {}
    for RELATION in ['U', 'F', 'L']:
        selected_dyads = [d for d, g in dyads if g == RELATION]
        data_rel = all_data[RELATION]

        copresence_list = []
        stimulus_list = []
        surrogate_list = []

        for v_idx in range(6):
            V = v_idx + 1
            group1_v = []  # Males
            group2_v = []  # Females

            for D in selected_dyads:
                if V in data_rel.get(D, {}):
                    group1_v.append(data_rel[D][V]['M'])
                    group2_v.append(data_rel[D][V]['F'])

            if len(group1_v) < 2:
                copresence_list.append(np.array([]))
                stimulus_list.append(np.array([]))
                surrogate_list.append(np.array([]))
                continue

            print(f"  Computing distances: {GROUP_LABELS[RELATION]} Video {VIDEO_NAMES[v_idx]} "
                  f"(N={len(group1_v)})...", end="", flush=True)

            cop_v, stim_v, _, _, surr_v = compute_distances_golland(
                group1_v, group2_v, 'cc', LAG, False, True
            )

            copresence_list.append(cop_v)
            stimulus_list.append(stim_v)
            surrogate_list.append(surr_v)
            print(f" cop={cop_v.mean():.6f}, stim={stim_v.mean():.6f}, surr={surr_v.mean():.6f}")

        copresence_df = pd.DataFrame(
            np.array(copresence_list, dtype=object),
            columns=selected_dyads,
            index=VIDEO_NAMES
        )
        stimulus_df = pd.DataFrame(
            np.array(stimulus_list, dtype=object),
            index=VIDEO_NAMES
        )
        surrogate_df = pd.DataFrame(
            np.array(surrogate_list, dtype=object),
            index=VIDEO_NAMES
        )

        results[RELATION] = {
            'copresence': copresence_df,
            'stimulus': stimulus_df,
            'surrogate': surrogate_df,
            'dyads': selected_dyads,
        }

    return results


def run_continuous_analysis(dyads, modality, max_dyads=None):
    """
    Run copresence/stimulus/surrogate analysis for EDA or EMG.

    Uses the same IAAFT + CC distance pipeline applied to continuous signals.
    """
    if max_dyads:
        dyads = dyads[:max_dyads]

    print(f"\n{'='*60}")
    print(f"{modality.upper()} ANALYSIS — Copresence / Stimulus / Surrogate")
    print(f"Dyads: {len(dyads)}, Videos: 6")
    print(f"IAAFT surrogate + CC distance")
    print(f"{'='*60}")

    all_data = {}
    for RELATION in ['U', 'F', 'L']:
        selected_dyads = [d for d, g in dyads if g == RELATION]
        data_rel = {}
        for D in selected_dyads:
            print(f"  Loading {D} ({modality})...", end="", flush=True)
            try:
                raw_m, fs_m = load_raw_signal(D, 'M', modality)
                raw_f, fs_f = load_raw_signal(D, 'F', modality)
                segments = get_video_segments_raw(D)
            except Exception as e:
                print(f" ERROR: {e}")
                continue

            if raw_m is None or raw_f is None:
                print(" no raw data")
                continue

            data_dyad = {}
            for seg in segments:
                vid = seg['video_id']
                t_start, t_end = seg['t_start'], seg['t_end']

                proc_m = preprocess_continuous_signal(raw_m, t_start, t_end, fs_m, modality)
                proc_f = preprocess_continuous_signal(raw_f, t_start, t_end, fs_f, modality)

                if proc_m is None or proc_f is None:
                    continue
                if len(proc_m) < MIN_SAMPLES or len(proc_f) < MIN_SAMPLES:
                    continue

                # IAAFT surrogate + smooth
                surr_m = compute_IAAFT_surrogates(proc_m)
                surr_m = np.convolve(surr_m, np.ones(int(CONV_WINDOW * F_RESAMP)) / int(CONV_WINDOW * F_RESAMP))

                surr_f = compute_IAAFT_surrogates(proc_f)
                surr_f = np.convolve(surr_f, np.ones(int(CONV_WINDOW * F_RESAMP)) / int(CONV_WINDOW * F_RESAMP))

                data_dyad[vid] = {'M': [proc_m, surr_m], 'F': [proc_f, surr_f]}

            data_rel[D] = data_dyad
            print(f" {len(data_dyad)} videos OK")

        all_data[RELATION] = data_rel

    # Phase 2: Compute distances
    results = {}
    for RELATION in ['U', 'F', 'L']:
        selected_dyads = [d for d, g in dyads if g == RELATION]
        data_rel = all_data[RELATION]

        copresence_list = []
        stimulus_list = []
        surrogate_list = []

        for v_idx in range(6):
            V = v_idx + 1
            group1_v = []
            group2_v = []
            for D in selected_dyads:
                if V in data_rel.get(D, {}):
                    group1_v.append(data_rel[D][V]['M'])
                    group2_v.append(data_rel[D][V]['F'])

            if len(group1_v) < 2:
                copresence_list.append(np.array([]))
                stimulus_list.append(np.array([]))
                surrogate_list.append(np.array([]))
                continue

            print(f"  Computing {modality} distances: {GROUP_LABELS[RELATION]} Video {VIDEO_NAMES[v_idx]} "
                  f"(N={len(group1_v)})...", end="", flush=True)

            cop_v, stim_v, _, _, surr_v = compute_distances_golland(
                group1_v, group2_v, 'cc', LAG, False, True
            )

            copresence_list.append(cop_v)
            stimulus_list.append(stim_v)
            surrogate_list.append(surr_v)
            print(f" cop={cop_v.mean():.6f}, stim={stim_v.mean():.6f}, surr={surr_v.mean():.6f}")

        copresence_df = pd.DataFrame(
            np.array(copresence_list, dtype=object),
            columns=selected_dyads,
            index=VIDEO_NAMES
        )
        stimulus_df = pd.DataFrame(
            np.array(stimulus_list, dtype=object),
            index=VIDEO_NAMES
        )
        surrogate_df = pd.DataFrame(
            np.array(surrogate_list, dtype=object),
            index=VIDEO_NAMES
        )

        results[RELATION] = {
            'copresence': copresence_df,
            'stimulus': stimulus_df,
            'surrogate': surrogate_df,
            'dyads': selected_dyads,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTING
# ═══════════════════════════════════════════════════════════════════════

def p2sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'


def test_three_tier(results):
    """
    Replicate Bizzego's 04_test_cop-stim-surr.py:
    For each relation × video:
      - surrogate < stimulus (one-sided MWU)
      - stimulus < copresence (one-sided MWU)
    """
    print("\n" + "=" * 70)
    print("THREE-TIER TEST: surrogate < stimulus < copresence")
    print("(One-sided Mann-Whitney U test)")
    print("=" * 70)

    all_rows = []

    for RELATION in ['U', 'F', 'L']:
        label = GROUP_LABELS[RELATION]
        print(f"\n── {label} ({RELATION}) ──")
        print(f"  {'Video':8s} {'Sur<Stim':>10s} {'Stim<Cop':>10s}")
        print(f"  {'─'*8} {'─'*10} {'─'*10}")

        cop_df = results[RELATION]['copresence']
        stim_df = results[RELATION]['stimulus']
        surr_df = results[RELATION]['surrogate']

        for v_idx, VIDEO in enumerate(VIDEO_EMOTIONS):
            surr_v = np.array(surr_df.iloc[v_idx], dtype=float)
            stim_v = np.array(stim_df.iloc[v_idx], dtype=float)
            cop_v = np.array(cop_df.iloc[v_idx], dtype=float)

            # Remove NaN
            surr_v = surr_v[~np.isnan(surr_v)]
            stim_v = stim_v[~np.isnan(stim_v)]
            cop_v = cop_v[~np.isnan(cop_v)]

            if len(surr_v) < 3 or len(stim_v) < 3 or len(cop_v) < 3:
                print(f"  {VIDEO:8s} {'N/A':>10s} {'N/A':>10s}")
                row = {
                    'relation': RELATION, 'relation_label': label,
                    'video': VIDEO, 'video_name': VIDEO_NAMES[v_idx],
                    'n_dyads': 0,
                    'mean_surr': np.nan, 'mean_stim': np.nan, 'mean_cop': np.nan,
                    'p_surr_stim': np.nan, 'sig_surr_stim': 'N/A',
                    'p_stim_cop': np.nan, 'sig_stim_cop': 'N/A',
                }
                all_rows.append(row)
                continue

            # Test: surrogate < stimulus (one-sided)
            try:
                T1, p1 = stats.mannwhitneyu(surr_v, stim_v, alternative='less')
            except:
                T1, p1 = np.nan, np.nan

            # Test: stimulus < copresence (one-sided)
            try:
                T2, p2 = stats.mannwhitneyu(stim_v, cop_v, alternative='less')
            except:
                T2, p2 = np.nan, np.nan

            sig1 = p2sig(p1) if np.isfinite(p1) else 'N/A'
            sig2 = p2sig(p2) if np.isfinite(p2) else 'N/A'

            print(f"  {VIDEO:8s} {sig1:>10s} {sig2:>10s}")

            row = {
                'relation': RELATION, 'relation_label': label,
                'video': VIDEO, 'video_name': VIDEO_NAMES[v_idx],
                'n_dyads': max(len(surr_v), len(stim_v), len(cop_v)),
                'mean_surr': np.mean(surr_v), 'sd_surr': np.std(surr_v),
                'mean_stim': np.mean(stim_v), 'sd_stim': np.std(stim_v),
                'mean_cop': np.mean(cop_v), 'sd_cop': np.std(cop_v),
                'U_surr_stim': T1, 'p_surr_stim': p1, 'sig_surr_stim': sig1,
                'U_stim_cop': T2, 'p_stim_cop': p2, 'sig_stim_cop': sig2,
            }
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def test_cross_group_copresence(results):
    """
    Replicate Bizzego's 04_test_cop_UFL.py:
    Kruskal-Wallis on copresence distances across U/F/L per video.
    """
    print("\n" + "=" * 70)
    print("CROSS-GROUP TEST: U vs F vs L (Kruskal-Wallis on copresence)")
    print("=" * 70)

    all_rows = []

    for v_idx in range(6):
        VIDEO = VIDEO_NAMES[v_idx]
        groups = []
        for RELATION in ['U', 'F', 'L']:
            cop_v = np.array(results[RELATION]['copresence'].iloc[v_idx], dtype=float)
            cop_v = cop_v[~np.isnan(cop_v)]
            groups.append(cop_v)

        print(f"\n── Video {VIDEO} ({VIDEO_EMOTIONS[v_idx]}) ──")
        for i, (RELATION, label) in enumerate(zip(['U', 'F', 'L'], ['Strangers', 'Friends', 'Lovers'])):
            if len(groups[i]) > 0:
                print(f"  {label:10s}: M={np.mean(groups[i]):.6f}, SD={np.std(groups[i]):.6f}, N={len(groups[i])}")

        if all(len(g) > 2 for g in groups):
            H, p = stats.kruskal(*groups)
            sig = p2sig(p)
            print(f"  Kruskal-Wallis H={H:.3f}, p={p:.4f} {sig}")

            # Pairwise
            pairs = [('U', 'F'), ('F', 'L'), ('U', 'L')]
            for g1, g2 in pairs:
                idx1 = 0 if g1 == 'U' else 1 if g1 == 'F' else 2
                idx2 = 0 if g2 == 'U' else 1 if g2 == 'F' else 2
                try:
                    U, p_val = stats.mannwhitneyu(groups[idx1], groups[idx2], alternative='two-sided')
                except:
                    U, p_val = np.nan, np.nan
                sig_p = p2sig(p_val) if np.isfinite(p_val) else 'N/A'
                print(f"    {GROUP_LABELS[g1]:10s} vs {GROUP_LABELS[g2]:10s}: U={U:.1f}, p={p_val:.4f} {sig_p}")

            row = {
                'video': VIDEO, 'video_emotion': VIDEO_EMOTIONS[v_idx],
                'H': H, 'p_KW': p, 'sig_KW': sig,
                'N_U': len(groups[0]), 'N_F': len(groups[1]), 'N_L': len(groups[2]),
                'mean_U': np.mean(groups[0]), 'mean_F': np.mean(groups[1]), 'mean_L': np.mean(groups[2]),
            }
            all_rows.append(row)
        else:
            print("  Insufficient data for KW test")

    return pd.DataFrame(all_rows)


def test_cross_modal(ibi_results, eda_results, emg_results):
    """Compare IBI vs EDA vs EMG copresence across groups and videos."""
    print("\n" + "=" * 70)
    print("CROSS-MODAL COMPARISON: IBI vs EDA vs EMG")
    print("=" * 70)

    # For each group × video, compare the three modalities
    rows = []
    for RELATION in ['U', 'F', 'L']:
        label = GROUP_LABELS[RELATION]
        for v_idx in range(6):
            VIDEO = VIDEO_NAMES[v_idx]
            row = {
                'relation': RELATION, 'relation_label': label,
                'video': VIDEO, 'video_emotion': VIDEO_EMOTIONS[v_idx],
            }

            for mod_name, mod_results in [('IBI', ibi_results), ('EDA', eda_results), ('EMG', emg_results)]:
                if mod_results is None:
                    row[f'{mod_name}_mean_cop'] = np.nan
                    row[f'{mod_name}_N'] = 0
                    continue
                cop_v = np.array(mod_results[RELATION]['copresence'].iloc[v_idx], dtype=float)
                cop_v = cop_v[~np.isnan(cop_v)]
                row[f'{mod_name}_mean_cop'] = np.mean(cop_v) if len(cop_v) > 0 else np.nan
                row[f'{mod_name}_N'] = len(cop_v)
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bizzego 2020 — IAAFT + CC Distance Analysis")
    parser.add_argument('--modality', choices=['ibi', 'eda', 'emg', 'all'], default='ibi',
                        help='Signal modality to analyze')
    parser.add_argument('--max-dyads', type=int, default=None, help='Max dyads to process')
    args = parser.parse_args()

    _setup_pkl_stub()
    np.random.seed(42)

    dyads = discover_dyads()
    print(f"Found {len(dyads)} dyads")
    for grp in ['F', 'U', 'L']:
        grp_dyads = [d for d, g in dyads if g == grp]
        print(f"  {GROUP_LABELS[grp]:10s}: {len(grp_dyads)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── IBI Analysis ──────────────────────────────────────────────────
    ibi_results = None
    ibi_three_tier = None
    ibi_cross_group = None

    if args.modality in ('ibi', 'all'):
        ibi_results = run_ibi_analysis(dyads, args.max_dyads)

        # Statistical tests
        ibi_three_tier = test_three_tier(ibi_results)
        ibi_cross_group = test_cross_group_copresence(ibi_results)

        # Save results
        for RELATION in ['U', 'F', 'L']:
            out_sub = OUTPUT_DIR / f"iaaft_{RELATION}"
            out_sub.mkdir(parents=True, exist_ok=True)
            for name in ['copresence', 'stimulus', 'surrogate']:
                ibi_results[RELATION][name].to_csv(out_sub / f"{name}.csv")

        ibi_three_tier.to_csv(OUTPUT_DIR / "iaaft_three_tier_test.csv", index=False)
        ibi_cross_group.to_csv(OUTPUT_DIR / "iaaft_cross_group_test.csv", index=False)

        print(f"\nResults saved to {OUTPUT_DIR}")

    # ── EDA Analysis ──────────────────────────────────────────────────
    eda_results = None
    if args.modality in ('eda', 'all'):
        eda_results = run_continuous_analysis(dyads, 'eda', args.max_dyads)

        eda_three_tier = test_three_tier(eda_results)
        eda_cross_group = test_cross_group_copresence(eda_results)

        for RELATION in ['U', 'F', 'L']:
            out_sub = OUTPUT_DIR / f"iaaft_eda_{RELATION}"
            out_sub.mkdir(parents=True, exist_ok=True)
            for name in ['copresence', 'stimulus', 'surrogate']:
                eda_results[RELATION][name].to_csv(out_sub / f"{name}.csv")

        eda_three_tier.to_csv(OUTPUT_DIR / "iaaft_eda_three_tier_test.csv", index=False)
        eda_cross_group.to_csv(OUTPUT_DIR / "iaaft_eda_cross_group_test.csv", index=False)

    # ── EMG Analysis ──────────────────────────────────────────────────
    emg_results = None
    if args.modality in ('emg', 'all'):
        emg_results = run_continuous_analysis(dyads, 'emg', args.max_dyads)

        emg_three_tier = test_three_tier(emg_results)
        emg_cross_group = test_cross_group_copresence(emg_results)

        for RELATION in ['U', 'F', 'L']:
            out_sub = OUTPUT_DIR / f"iaaft_emg_{RELATION}"
            out_sub.mkdir(parents=True, exist_ok=True)
            for name in ['copresence', 'stimulus', 'surrogate']:
                emg_results[RELATION][name].to_csv(out_sub / f"{name}.csv")

        emg_three_tier.to_csv(OUTPUT_DIR / "iaaft_emg_three_tier_test.csv", index=False)
        emg_cross_group.to_csv(OUTPUT_DIR / "iaaft_emg_cross_group_test.csv", index=False)

    # ── Cross-Modal Comparison ────────────────────────────────────────
    if args.modality == 'all' and ibi_results and eda_results and emg_results:
        cross_modal = test_cross_modal(ibi_results, eda_results, emg_results)
        cross_modal.to_csv(OUTPUT_DIR / "iaaft_cross_modal_comparison.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if ibi_three_tier is not None:
        sig_count = sum(1 for s in ibi_three_tier['sig_surr_stim'] if s in ('*', '**', '***'))
        sig_count2 = sum(1 for s in ibi_three_tier['sig_stim_cop'] if s in ('*', '**', '***'))
        print(f"\n  IBI Three-Tier Test:")
        print(f"    surrogate < stimulus:  {sig_count}/18 significant")
        print(f"    stimulus < copresence: {sig_count2}/18 significant")

    if ibi_cross_group is not None:
        sig_kw = sum(1 for s in ibi_cross_group.get('sig_KW', []) if s in ('*', '**', '***'))
        print(f"\n  IBI Cross-Group (KW):   {sig_kw}/6 videos significant")


if __name__ == "__main__":
    main()
