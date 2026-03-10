"""
EEG Data Loader for MUSE CSV Dataset  (EmoKey / EKM-ED)
=========================================================

Actual folder structure on disk
--------------------------------
muse_wearable_data/
└── preprocessed/
    └── clean-signals/
        └── 0.0078125S/          ← note the trailing "S"
            ├── 1/               ← subject ID
            │   ├── ANGER.csv
            │   ├── FEAR.csv
            │   ├── HAPPINESS.csv
            │   ├── NEUTRAL_ANGER.csv      ← per-emotion neutral baseline
            │   ├── NEUTRAL_FEAR.csv
            │   ├── NEUTRAL_HAPPINESS.csv
            │   ├── NEUTRAL_SADNESS.csv
            │   └── SADNESS.csv
            └── 103/
                └── ...

DATA_ROOT should point to the folder that contains the subject-ID subfolders,
i.e. the full path ending in  .../clean-signals/0.0078125S

Emotion → Quadrant mapping (Russell's circumplex model):
    HAPPINESS  → Q1  (Positive Valence,  High Arousal)
    ANGER      → Q2  (Negative Valence,  High Arousal)
    FEAR       → Q2  (Negative Valence,  High Arousal)
    SADNESS    → Q3  (Negative Valence,  Low Arousal)
    NEUTRAL_*  → used as per-emotion baseline (not a class label)

Sampling rate: 128 Hz  (1 / 0.0078125 s)

Author: Final Year Project
Date: 2026
"""

import os
import glob
from collections import Counter

import numpy as np
import pandas as pd


# ==================================================
# CONSTANTS
# ==================================================

MUSE_RAW_COLS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
HSI_COLS      = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

# The actual subfolder name used in the dataset (1/128 ≈ 0.0078125, with "S")
FS_SUBFOLDER = "0.0078125S"

# Native sampling rate of this dataset
DATASET_FS = 128.0

# Emotions that are actual trial recordings (not neutrals / baselines)
TRIAL_EMOTIONS = {"ANGER", "FEAR", "HAPPINESS", "SADNESS"}


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
    Load a MUSE CSV and return the 4 raw EEG channel arrays.

    Returns list of 4 np.ndarray, or None on failure.
    """
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return None

    df.columns = [c.strip() for c in df.columns]

    # Case-insensitive column matching
    missing = [c for c in MUSE_RAW_COLS if c not in df.columns]
    if missing:
        col_map = {c.upper(): c for c in df.columns}
        rename, still_missing = {}, []
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
    """Apply HSI quality filtering if columns exist; otherwise just drop NaNs."""
    mask = np.ones(L, dtype=bool)
    for ch in channels:
        mask &= np.isfinite(ch[:L])

    if all(c in df.columns for c in HSI_COLS):
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
    """InvBase: divide trial FFT by baseline FFT per channel."""
    common = min(len(signal), len(baseline))
    sig, bas = signal[:common], baseline[:common]
    FFT_r = np.fft.rfft(sig, axis=0) / (np.abs(np.fft.rfft(bas, axis=0)) + eps)
    return np.fft.irfft(FFT_r, n=common, axis=0).astype(np.float32)


# ==================================================
# FILE DISCOVERY
# ==================================================

def _has_numeric_subdirs_with_csvs(folder: str) -> bool:
    """
    Return True if `folder` contains at least one numeric sub-directory
    that itself contains at least one CSV file.  This is the signature of
    the subject-ID level in the EmoKey dataset.
    """
    try:
        entries = os.listdir(folder)
    except PermissionError:
        return False
    for entry in entries:
        if not entry.isdigit():
            continue
        subdir = os.path.join(folder, entry)
        if not os.path.isdir(subdir):
            continue
        if glob.glob(os.path.join(subdir, "*.csv")):
            return True
    return False


def _find_subject_root(start: str, max_depth: int = 6) -> str | None:
    """
    Walk downward from `start` (BFS, up to `max_depth` levels) and return
    the first directory that passes `_has_numeric_subdirs_with_csvs`.

    This lets the loader work regardless of the Kaggle dataset slug or
    how many wrapper folders sit above the actual data.
    """
    from collections import deque
    queue = deque([(start, 0)])
    while queue:
        current, depth = queue.popleft()
        if _has_numeric_subdirs_with_csvs(current):
            return current
        if depth >= max_depth:
            continue
        try:
            children = sorted(os.listdir(current))
        except PermissionError:
            continue
        for child in children:
            full = os.path.join(current, child)
            if os.path.isdir(full):
                queue.append((full, depth + 1))
    return None


def find_subject_dirs(data_root: str) -> list:
    """
    Return a sorted list of (subject_id, subject_dir) tuples.

    Strategy
    --------
    1. Try `data_root` itself (already points at the right level).
    2. If that finds nothing, recursively search downward through the
       directory tree (BFS, max 6 levels) for a folder that contains
       numeric sub-directories with CSV files.

    This makes the loader resilient to any Kaggle dataset slug or extra
    wrapper folders added by the dataset uploader.
    """
    def _collect_numeric_subdirs(base: str) -> list:
        """Collect all numeric sub-dirs of `base` that contain CSVs."""
        result = []
        try:
            entries = os.listdir(base)
        except PermissionError:
            return result
        for entry in sorted(entries, key=lambda x: int(x) if x.isdigit() else float('inf')):
            if not entry.isdigit():
                continue
            full = os.path.join(base, entry)
            if os.path.isdir(full) and glob.glob(os.path.join(full, "*.csv")):
                result.append((entry, full))
        return result

    # ── Try data_root directly ────────────────────────────────────────────────
    subject_dirs = _collect_numeric_subdirs(data_root)
    if subject_dirs:
        return subject_dirs

    # ── Auto-discover: walk down until we find the right folder ───────────────
    print(f"   ⚠️  No subject dirs found directly in '{data_root}'")
    print(f"   🔍 Searching subdirectories for EmoKey structure ...")

    found_root = _find_subject_root(data_root, max_depth=6)
    if found_root and found_root != data_root:
        print(f"   ✅ Found subject root: {found_root}")
        return _collect_numeric_subdirs(found_root)

    return []


def find_csv_files_for_subject(subject_dir: str) -> list:
    """Return all CSV files directly inside the subject directory."""
    return sorted(glob.glob(os.path.join(subject_dir, "*.csv")))


# ==================================================
# FILENAME PARSING
# ==================================================

def parse_filename(fname: str) -> tuple:
    """
    Parse a CSV filename into (file_type, emotion).

    Returns
    -------
    file_type : 'trial'    — a proper emotion recording  (ANGER.csv etc.)
                'neutral'  — a neutral baseline recording (NEUTRAL_ANGER.csv etc.)
                'unknown'  — unrecognised filename
    emotion   : str  (e.g. 'ANGER') or None for 'unknown'

    Examples
    --------
    'ANGER.csv'           → ('trial',   'ANGER')
    'NEUTRAL_ANGER.csv'   → ('neutral', 'ANGER')
    'HAPPINESS.csv'       → ('trial',   'HAPPINESS')
    'NEUTRAL_SADNESS.csv' → ('neutral', 'SADNESS')
    """
    stem = os.path.splitext(os.path.basename(fname))[0].upper()

    if stem.startswith("NEUTRAL_"):
        # e.g. NEUTRAL_ANGER  →  emotion = ANGER
        emotion = stem[len("NEUTRAL_"):]
        return ("neutral", emotion) if emotion in TRIAL_EMOTIONS else ("unknown", None)

    if stem in TRIAL_EMOTIONS:
        return ("trial", stem)

    return ("unknown", None)


# ==================================================
# BASELINE LOADING
# ==================================================

def load_baselines_for_dataset(subject_dirs: list) -> dict:
    """
    Build a per-subject, per-emotion baseline dictionary from NEUTRAL_* files.

    Returns
    -------
    baselines : dict
        subject_id → { emotion → (T, 4) float32 array }
        e.g. baselines['1']['ANGER'] = array(...)
    """
    baselines: dict = {}
    print("   Loading per-emotion neutral baselines (NEUTRAL_*.csv)...")

    for subject_id, subject_dir in subject_dirs:
        csv_files = find_csv_files_for_subject(subject_dir)
        subj_baselines = {}

        for fpath in csv_files:
            file_type, emotion = parse_filename(fpath)
            if file_type != "neutral":
                continue

            channels = _load_csv_channels(fpath)
            if channels is None:
                continue
            L = min(len(ch) for ch in channels)
            if L == 0:
                continue

            signal = np.stack([ch[:L] for ch in channels], axis=1).astype(np.float32)
            signal -= signal.mean(axis=0, keepdims=True)
            subj_baselines[emotion] = signal

        if subj_baselines:
            baselines[subject_id] = subj_baselines

    loaded = sum(len(v) for v in baselines.values())
    print(f"   ✅ Loaded {loaded} neutral baselines across {len(baselines)} subjects")
    return baselines


# ==================================================
# WINDOWING
# ==================================================

def create_windows(signal: np.ndarray, win_samples: int, step_samples: int) -> list:
    """Slice (T, C) signal into overlapping windows of shape (win_samples, C)."""
    return [
        signal[s: s + win_samples]
        for s in range(0, len(signal) - win_samples + 1, step_samples)
        if len(signal[s: s + win_samples]) == win_samples
    ]


# ==================================================
# MAIN DATA LOADING
# ==================================================

def load_eeg_data(data_root: str, config) -> tuple:
    """
    Load EEG data from the EmoKey MUSE CSV dataset.

    Parameters
    ----------
    data_root : str
        Path to the folder that contains the numbered subject sub-directories.
        e.g.  .../muse_wearable_data/preprocessed/clean-signals/0.0078125S
        OR    .../muse_wearable_data/preprocessed/clean-signals
        (both layouts are detected automatically)

    config : Config
        Pipeline configuration.  Uses:
          config.EEG_WINDOW_SEC, config.EEG_OVERLAP,
          config.USE_BASELINE_REDUCTION, config.MUSE_CSV_SUPERCLASS_MAP

    Returns
    -------
    X_raw       : (N, T, 4)  float32
    y_labels    : (N,)       int64
    subject_ids : (N,)       str
    label_to_id : dict       str → int
    clip_ids    : (N,)       str
    """
    print("\n" + "=" * 80)
    print("LOADING EEG DATA  —  EmoKey MUSE CSV Dataset")
    print("=" * 80)

    # ── Discover subjects ─────────────────────────────────────────────────────
    subject_dirs = find_subject_dirs(data_root)
    if not subject_dirs:
        raise ValueError(
            f"No subject directories found under '{data_root}'.\n"
            f"Expected numbered sub-folders (e.g. 1/, 2/, 103/) containing CSV files.\n"
            f"Make sure DATA_ROOT_MUSE_CSV points to the folder with the subject IDs."
        )

    print(f"Found {len(subject_dirs)} subject directories")
    for sid, sdir in subject_dirs[:5]:
        print(f"   Subject {sid:>4s}  →  {sdir}")
    if len(subject_dirs) > 5:
        print(f"   ... and {len(subject_dirs) - 5} more")

    # ── Windowing parameters ──────────────────────────────────────────────────
    fs           = DATASET_FS
    win_samples  = int(config.EEG_WINDOW_SEC * fs)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    print(f"\n⚙️  Sampling rate : {fs} Hz")
    print(f"   Window        : {config.EEG_WINDOW_SEC}s  ({win_samples} samples)")
    print(f"   Overlap       : {config.EEG_OVERLAP * 100:.0f}%  (step = {step_samples} samples)")

    # ── Load baselines ────────────────────────────────────────────────────────
    baselines: dict = {}
    if config.USE_BASELINE_REDUCTION:
        print(f"\n🔧 Baseline Reduction: ENABLED  (using NEUTRAL_<EMOTION>.csv files)")
        baselines = load_baselines_for_dataset(subject_dirs)
    else:
        print(f"\n🔧 Baseline Reduction: DISABLED")

    # ── Process files ─────────────────────────────────────────────────────────
    all_windows:   list = []
    all_labels:    list = []
    all_subjects:  list = []
    all_clip_ids:  list = []
    reduced_count       = 0
    not_reduced_count   = 0
    skipped             = Counter()

    for subject_id, subject_dir in subject_dirs:
        csv_files = find_csv_files_for_subject(subject_dir)

        for fpath in csv_files:
            file_type, emotion = parse_filename(fpath)

            # Skip neutrals (used as baselines, not as class labels)
            if file_type == "neutral":
                skipped["neutral_baseline_file"] += 1
                continue
            if file_type == "unknown":
                skipped["unknown_filename"] += 1
                continue

            # Must be in the superclass map
            if emotion not in config.MUSE_CSV_SUPERCLASS_MAP:
                skipped["unknown_emotion"] += 1
                continue

            superclass = config.MUSE_CSV_SUPERCLASS_MAP[emotion]

            # ── Load channels ─────────────────────────────────────────────────
            channels = _load_csv_channels(fpath)
            if channels is None:
                skipped["parse_error"] += 1
                continue

            L = min(len(ch) for ch in channels)
            if L == 0:
                skipped["no_data"] += 1
                continue

            # ── Quality filtering ─────────────────────────────────────────────
            try:
                df_raw = pd.read_csv(fpath)
                df_raw.columns = [c.strip() for c in df_raw.columns]
                channels_filtered, L_filt = _apply_quality_mask(
                    [ch[:L] for ch in channels], df_raw, L
                )
            except Exception:
                channels_filtered = [ch[:L] for ch in channels]
                L_filt = L

            if L_filt < win_samples:
                skipped["insufficient_length"] += 1
                continue

            # ── Build signal array ────────────────────────────────────────────
            signal = np.stack(channels_filtered, axis=1).astype(np.float32)
            signal -= signal.mean(axis=0, keepdims=True)

            # ── Per-emotion baseline reduction ────────────────────────────────
            # Use NEUTRAL_<EMOTION>.csv as baseline for this specific emotion
            if (config.USE_BASELINE_REDUCTION
                    and subject_id in baselines
                    and emotion in baselines[subject_id]):
                baseline_sig = baselines[subject_id][emotion]
                signal = apply_baseline_reduction(signal, baseline_sig)
                reduced_count += 1
            else:
                not_reduced_count += 1

            # ── Window ───────────────────────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n📊 Processing Summary:")
    print(f"   Subjects processed : {len(subject_dirs)}")
    print(f"   Windows extracted  : {len(all_windows)}")
    if skipped:
        print(f"   Skipped files:")
        for reason, cnt in skipped.items():
            print(f"      {reason}: {cnt}")

    if not all_windows:
        raise ValueError(
            "No valid EEG windows extracted.\n"
            f"Searched: {data_root}\n"
            "Tip: Make sure DATA_ROOT_MUSE_CSV ends with the folder that contains "
            "the numbered subject sub-directories (e.g. .../0.0078125S)."
        )

    # ── Assemble output arrays ─────────────────────────────────────────────────
    X_raw       = np.stack(all_windows).astype(np.float32)
    label_list  = sorted(set(all_labels))
    label_to_id = {lab: i for i, lab in enumerate(label_list)}
    y_labels    = np.array([label_to_id[l] for l in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    clip_ids    = np.array(all_clip_ids)

    print(f"\n✅ EmoKey data loaded : {X_raw.shape}  (N windows, T samples, C channels)")
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
# DATA SPLITTING
# ==================================================

def create_data_splits(y_labels, subject_ids, clip_ids, config,
                       train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Leak-free train/val/test split — identical interface to the EmOgnition loader."""
    print("\n" + "=" * 80)
    print("CREATING DATA SPLIT — EmoKey MUSE CSV (LEAK-FREE)")
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
