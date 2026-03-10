"""
EEG Data Loader for MUSE CSV Dataset
======================================

Loads EEG data from the new CSV-based MUSE wearable dataset.

Dataset folder structure:
    muse_wearable_data/
    ├── raw/
    │   └── <subject_id>/
    │       ├── muse/
    │       │   ├── ANGER_XXX.csv
    │       │   ├── FEAR_XXX.csv
    │       │   ├── HAPPINESS_XXX.csv
    │       │   └── SADNESS_XXX.csv
    │       └── order/   (elicitation order file)
    └── preprocessed/
        ├── unclean-signals/
        │   └── muse/
        │       └── 0.0078125/   (128 Hz data)
        └── clean-signals/
            └── muse/
                └── 0.0078125/   (128 Hz cleaned data)

CSV column format (MUSE headband):
    TimeStamp, RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10,
    [optional: Delta_TP9 ... HSI_TP9 ...]

Emotion → Quadrant mapping (Russell's circumplex model):
    HAPPINESS  → Q1  (Positive Valence, High Arousal)
    ANGER      → Q2  (Negative Valence, High Arousal)
    FEAR       → Q2  (Negative Valence, High Arousal)   ← same quadrant as ANGER
    SADNESS    → Q3  (Negative Valence, Low Arousal)

Sampling rate: 128 Hz (downsampled from 256 Hz)

Author: Final Year Project
Date: 2026
"""

import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ==================================================
# CONSTANTS
# ==================================================

# Column names for the 4 raw EEG channels
MUSE_RAW_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

# Possible HSI quality column names
HSI_COLS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

# Accepted sampling-rate subfolder names (the dataset uses 1/128 ≈ 0.0078125)
FS_SUBFOLDER = "0.0078125"

# Native sampling rate of this dataset
DATASET_FS = 128.0


# ==================================================
# UTILITY HELPERS
# ==================================================

def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _load_csv_channels(fpath: str):
    """
    Load a MUSE CSV file and return the 4 raw EEG channel arrays.

    Returns
    -------
    channels : list of 4 np.ndarray  (one per channel)  or  None on failure
    """
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None

    # Normalise column names (strip whitespace, handle case)
    df.columns = [c.strip() for c in df.columns]

    # Check all required channels are present
    missing = [c for c in MUSE_RAW_COLS if c not in df.columns]
    if missing:
        # Try case-insensitive match
        col_map = {c.upper(): c for c in df.columns}
        rename = {}
        still_missing = []
        for need in missing:
            if need.upper() in col_map:
                rename[col_map[need.upper()]] = need
            else:
                still_missing.append(need)
        if still_missing:
            return None
        df.rename(columns=rename, inplace=True)

    channels = []
    for col in MUSE_RAW_COLS:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        channels.append(_interp_nan(arr))

    return channels


def _apply_quality_mask(channels: list, df: pd.DataFrame, L: int) -> tuple:
    """
    Optionally apply HSI quality filtering if HSI columns exist in the CSV.

    Returns filtered channels and their new length.
    """
    mask = np.ones(L, dtype=bool)

    # Finite-value mask
    for ch in channels:
        mask &= np.isfinite(ch[:L])

    # HSI quality mask (only if all HSI columns present)
    hsi_present = all(c in df.columns for c in HSI_COLS)
    if hsi_present:
        for hsi_col in HSI_COLS:
            hsi = pd.to_numeric(df[hsi_col], errors="coerce").to_numpy(np.float64)[:L]
            mask &= np.isfinite(hsi) & (hsi <= 2)

    filtered = [ch[:L][mask] for ch in channels]
    return filtered, int(mask.sum())


# ==================================================
# BASELINE REDUCTION
# ==================================================

def apply_baseline_reduction(signal: np.ndarray,
                              baseline: np.ndarray,
                              eps: float = 1e-12) -> np.ndarray:
    """
    InvBase method: divide trial FFT by baseline FFT per channel.

    Args
    ----
    signal   : (T, C) trial signal
    baseline : (T, C) baseline signal (trimmed to same length internally)
    eps      : division guard

    Returns
    -------
    reduced  : (T, C) float32 baseline-reduced signal
    """
    common = min(len(signal), len(baseline))
    sig = signal[:common]
    bas = baseline[:common]

    FFT_s = np.fft.rfft(sig, axis=0)
    FFT_b = np.fft.rfft(bas, axis=0)
    FFT_r = FFT_s / (np.abs(FFT_b) + eps)
    reduced = np.fft.irfft(FFT_r, n=common, axis=0)
    return reduced.astype(np.float32)


# ==================================================
# FILE DISCOVERY
# ==================================================

def find_subject_dirs(data_root: str) -> list:
    """
    Return a sorted list of per-subject directories found under data_root.

    Handles two layouts:
      Layout A (raw):
        data_root/<subject_id>/muse/EMOTION_XXX.csv

      Layout B (preprocessed clean/unclean):
        data_root/clean-signals/muse/0.0078125/<subject_id>/EMOTION_XXX.csv
        data_root/unclean-signals/muse/0.0078125/<subject_id>/EMOTION_XXX.csv
    """
    subject_dirs = []

    # Layout A — raw: each immediate child that has a 'muse' subfolder
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "muse")):
            subject_dirs.append(("raw", entry, os.path.join(full, "muse")))

    # Layout B — preprocessed subtree
    for signal_type in ("clean-signals", "unclean-signals"):
        base = os.path.join(data_root, signal_type, "muse", FS_SUBFOLDER)
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                subject_dirs.append((signal_type, entry, full))

    return subject_dirs


def find_csv_files_for_subject(muse_dir: str) -> list:
    """Return all CSV files inside a subject's muse directory."""
    patterns = [
        os.path.join(muse_dir, "*.csv"),
        os.path.join(muse_dir, "**", "*.csv"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    return files


# ==================================================
# EMOTION PARSING & MAPPING
# ==================================================

def parse_emotion_from_filename(fname: str) -> str | None:
    """
    Extract the emotion label from a CSV filename.

    Supported patterns:
      ANGER_XXX.csv  /  HAPPINESS_XXX.csv  /  SADNESS_XXX.csv  /  FEAR_XXX.csv
      BASELINE_XXX.csv  (returned as 'BASELINE')
    """
    name = os.path.splitext(os.path.basename(fname))[0].upper()
    parts = name.split("_")
    if parts:
        return parts[0]
    return None


# ==================================================
# BASELINE LOADING
# ==================================================

def load_baselines_for_dataset(subject_dirs: list) -> dict:
    """
    For each subject, look for a BASELINE CSV file and load it.

    Returns
    -------
    baselines : dict  subject_id → (T, 4) float32 array
    """
    baselines = {}
    print("   Loading CSV baseline recordings...")

    for _, subject_id, muse_dir in subject_dirs:
        if subject_id in baselines:
            continue

        csv_files = find_csv_files_for_subject(muse_dir)
        for fpath in csv_files:
            emotion = parse_emotion_from_filename(fpath)
            if emotion != "BASELINE":
                continue

            channels = _load_csv_channels(fpath)
            if channels is None:
                continue

            L = min(len(ch) for ch in channels)
            if L == 0:
                continue

            signal = np.stack([ch[:L] for ch in channels], axis=1).astype(np.float32)
            signal -= signal.mean(axis=0, keepdims=True)
            baselines[subject_id] = signal
            break  # one baseline per subject is enough

    print(f"   ✅ Loaded {len(baselines)} CSV baselines")
    return baselines


# ==================================================
# WINDOWING
# ==================================================

def create_windows(signal: np.ndarray,
                   win_samples: int,
                   step_samples: int) -> list:
    """Slice a (T, C) signal into overlapping windows of shape (win_samples, C)."""
    windows = []
    L = len(signal)
    for start in range(0, L - win_samples + 1, step_samples):
        w = signal[start: start + win_samples]
        if len(w) == win_samples:
            windows.append(w)
    return windows


# ==================================================
# MAIN DATA LOADING
# ==================================================

def load_eeg_data(data_root: str, config) -> tuple:
    """
    Load EEG data from the CSV-based MUSE dataset.

    Walks through the dataset folder structure, reads CSV files,
    applies optional baseline reduction, windows the signals, and
    returns arrays ready for feature extraction / model training.

    Parameters
    ----------
    data_root : str
        Root of the dataset, e.g. ``muse_wearable_data/raw``  or
        ``muse_wearable_data/preprocessed/clean-signals``
    config : Config
        Pipeline configuration object.

    Returns
    -------
    X_raw      : (N, T, C)  float32  — windowed raw EEG
    y_labels   : (N,)       int64    — integer class labels
    subject_ids: (N,)       str      — subject ID per window
    label_to_id: dict       str→int  — label name to integer mapping
    clip_ids   : (N,)       str      — unique recording ID per window
    """
    print("\n" + "=" * 80)
    print("LOADING EEG DATA (MUSE CSV DATASET)")
    print("=" * 80)

    # ── Discover subject directories ──────────────────────────────────────────
    subject_dirs = find_subject_dirs(data_root)
    if not subject_dirs:
        raise ValueError(
            f"No subject directories found under '{data_root}'.\n"
            "Expected structure:  <data_root>/<subject_id>/muse/*.csv\n"
            "  OR:  <data_root>/clean-signals/muse/0.0078125/<subject_id>/*.csv"
        )
    print(f"Found {len(subject_dirs)} subject directories")
    for layout, sid, _ in subject_dirs[:5]:
        print(f"   [{layout}] Subject: {sid}")
    if len(subject_dirs) > 5:
        print(f"   ... and {len(subject_dirs) - 5} more")

    # ── Sampling rate note ─────────────────────────────────────────────────────
    # This dataset is recorded / downsampled at 128 Hz.
    # We override config.EEG_FS locally so windowing uses the correct value.
    fs = DATASET_FS
    win_samples = int(config.EEG_WINDOW_SEC * fs)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    print(f"\n⚙️  Sampling rate : {fs} Hz  (dataset native)")
    print(f"   Window        : {config.EEG_WINDOW_SEC}s  ({win_samples} samples)")
    print(f"   Overlap       : {config.EEG_OVERLAP * 100:.0f}%  (step = {step_samples} samples)")

    # ── Load baselines ─────────────────────────────────────────────────────────
    baselines: dict = {}
    if config.USE_BASELINE_REDUCTION:
        print(f"\n🔧 Baseline Reduction: ENABLED")
        baselines = load_baselines_for_dataset(subject_dirs)
    else:
        print(f"\n🔧 Baseline Reduction: DISABLED")

    # ── Process each subject / file ────────────────────────────────────────────
    all_windows: list   = []
    all_labels:  list   = []
    all_subjects: list  = []
    all_clip_ids: list  = []
    reduced_count       = 0
    not_reduced_count   = 0

    skipped = Counter()

    for layout, subject_id, muse_dir in subject_dirs:
        csv_files = find_csv_files_for_subject(muse_dir)

        for fpath in csv_files:
            emotion = parse_emotion_from_filename(fpath)
            if emotion is None:
                skipped["parse_error"] += 1
                continue
            if emotion == "BASELINE":
                skipped["baseline_file"] += 1
                continue
            if emotion not in config.MUSE_CSV_SUPERCLASS_MAP:
                skipped["unknown_emotion"] += 1
                continue

            superclass = config.MUSE_CSV_SUPERCLASS_MAP[emotion]

            # ── Load CSV ───────────────────────────────────────────────────────
            channels = _load_csv_channels(fpath)
            if channels is None:
                skipped["parse_error"] += 1
                continue

            L = min(len(ch) for ch in channels)
            if L == 0:
                skipped["no_data"] += 1
                continue

            # ── Quality filtering ──────────────────────────────────────────────
            try:
                df_raw = pd.read_csv(fpath)
                df_raw.columns = [c.strip() for c in df_raw.columns]
                channels_trimmed = [ch[:L] for ch in channels]
                channels_filtered, L_filt = _apply_quality_mask(
                    channels_trimmed, df_raw, L
                )
            except Exception:
                channels_filtered = [ch[:L] for ch in channels]
                L_filt = L

            if L_filt < win_samples:
                skipped["insufficient_length"] += 1
                continue

            # ── Build (T, 4) signal ────────────────────────────────────────────
            signal = np.stack(channels_filtered, axis=1).astype(np.float32)
            signal -= signal.mean(axis=0, keepdims=True)

            # ── Baseline reduction ─────────────────────────────────────────────
            if config.USE_BASELINE_REDUCTION and subject_id in baselines:
                signal = apply_baseline_reduction(signal, baselines[subject_id])
                reduced_count += 1
            else:
                not_reduced_count += 1

            # ── Windowing ──────────────────────────────────────────────────────
            windows = create_windows(signal, win_samples, step_samples)
            if not windows:
                skipped["insufficient_length"] += 1
                continue

            clip_id = f"{subject_id}_{emotion}"
            for w in windows:
                all_windows.append(w)
                all_labels.append(superclass)
                all_subjects.append(subject_id)
                all_clip_ids.append(clip_id)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n📊 Processing Summary:")
    print(f"   Subjects processed : {len(subject_dirs)}")
    print(f"   Windows extracted  : {len(all_windows)}")
    if skipped:
        print(f"   Skipped files:")
        for reason, cnt in skipped.items():
            print(f"      {reason}: {cnt}")

    if not all_windows:
        raise ValueError(
            "No valid EEG windows extracted from the MUSE CSV dataset.\n"
            f"Searched: {data_root}\n"
            "Check that DATA_ROOT points to a folder containing subject sub-directories."
        )

    # ── Assemble arrays ────────────────────────────────────────────────────────
    X_raw       = np.stack(all_windows).astype(np.float32)      # (N, T, 4)
    label_list  = sorted(set(all_labels))
    label_to_id = {lab: i for i, lab in enumerate(label_list)}
    y_labels    = np.array([label_to_id[l] for l in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    clip_ids    = np.array(all_clip_ids)

    print(f"\n✅ MUSE CSV data loaded: {X_raw.shape}  (N, T, C)")
    print(f"   Unique subjects    : {len(np.unique(subject_ids))}")
    print(f"   Unique recordings  : {len(np.unique(clip_ids))}")
    print(f"   Label distribution : {Counter(all_labels)}")

    if config.USE_BASELINE_REDUCTION:
        total = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction:")
        print(f"   ✅ Reduced  : {reduced_count}")
        print(f"   ⚠️  Skipped : {not_reduced_count}")
        if total:
            print(f"   Rate       : {100 * reduced_count / total:.1f}%")

    return X_raw, y_labels, subject_ids, label_to_id, clip_ids


# ==================================================
# DATA SPLITTING  (reuse same leak-free logic)
# ==================================================

def create_data_splits(y_labels, subject_ids, clip_ids, config,
                       train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Clip-independent or subject-independent train/val/test split.

    Identical strategy to ``eeg_data_loader_emognitionRaw.create_data_splits``
    so the rest of the pipeline is unaffected.
    """
    print("\n" + "=" * 80)
    print("CREATING DATA SPLIT — MUSE CSV (LEAK-FREE)")
    print("=" * 80)

    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT")
        unique_subjects = np.unique(subject_ids)
        np.random.shuffle(unique_subjects)

        n_test = max(1, int(len(unique_subjects) * test_ratio))
        n_val  = max(1, int(len(unique_subjects) * val_ratio))

        test_subj  = unique_subjects[:n_test]
        val_subj   = unique_subjects[n_test:n_test + n_val]
        train_subj = unique_subjects[n_test + n_val:]

        train_mask = np.isin(subject_ids, train_subj)
        val_mask   = np.isin(subject_ids, val_subj)
        test_mask  = np.isin(subject_ids, test_subj)

        print(f"  Train subjects : {len(train_subj)}")
        print(f"  Val subjects   : {len(val_subj)}")
        print(f"  Test subjects  : {len(test_subj)}")
    else:
        print("  Strategy: CLIP-INDEPENDENT (split by recordings)")
        unique_clips = np.unique(clip_ids)
        np.random.shuffle(unique_clips)

        n_test = max(1, int(len(unique_clips) * test_ratio))
        n_val  = max(1, int(len(unique_clips) * val_ratio))

        test_clips  = unique_clips[:n_test]
        val_clips   = unique_clips[n_test:n_test + n_val]
        train_clips = unique_clips[n_test + n_val:]

        train_mask = np.isin(clip_ids, train_clips)
        val_mask   = np.isin(clip_ids, val_clips)
        test_mask  = np.isin(clip_ids, test_clips)

        print(f"  Total recordings : {len(unique_clips)}")
        print(f"  Train            : {len(train_clips)}")
        print(f"  Val              : {len(val_clips)}")
        print(f"  Test             : {len(test_clips)}")

    split_indices = {
        "train": np.where(train_mask)[0],
        "val":   np.where(val_mask)[0],
        "test":  np.where(test_mask)[0],
    }

    print(f"\n📋 Window counts:")
    for name, idx in split_indices.items():
        dist = Counter(y_labels[idx].tolist())
        print(f"   {name.capitalize():5s}: {len(idx):5d} windows  |  class dist: {dict(dist)}")

    return split_indices
