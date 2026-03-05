"""
EEG Data Loader Module for Raw EmOgnition Dataset
==================================================

This module handles all data loading, preprocessing, and feature extraction
for the EEG emotion recognition pipeline using raw MUSE JSON files.

Features:
- MUSE headband EEG data loading from JSON files
- Baseline reduction (InvBase method)
- Quality filtering (HSI, HeadBandOn)
- Windowing with configurable overlap (LEAK-FREE)
- 26-feature extraction per channel (DE, PSD, temporal stats, etc.)
- Subject-independent and subject-dependent data splitting

Author: Final Year Project
Date: 2026
"""

import os
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ==================================================
# CONFIGURATION & CONSTANTS
# ==================================================

MUSE_CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
QUALITY_CHANNELS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
HSI_THRESHOLD = 2  # Maximum acceptable HSI value


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def _to_num(x):
    """Convert various input types to numeric numpy array."""
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    """Interpolate NaN values in a 1D array using linear interpolation."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


# ==================================================
# FILE DISCOVERY
# ==================================================

def find_muse_files(data_root):
    """
    Search for MUSE JSON files in the data directory.
    
    Args:
        data_root: Root directory containing MUSE JSON files
        
    Returns:
        List of file paths sorted alphabetically
    """
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*", "*_STIMULUS_MUSE.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    return files


def parse_filename(fname):
    """
    Parse MUSE filename to extract subject and emotion.
    
    Args:
        fname: Filename (e.g., "P001_ENTHUSIASM_STIMULUS_MUSE.json")
        
    Returns:
        Tuple of (subject, emotion) or (None, None) if parsing fails
    """
    parts = fname.split("_")
    if len(parts) < 2:
        return None, None
    
    subject = parts[0]
    emotion = parts[1].upper()
    
    return subject, emotion


# ==================================================
# BASELINE REDUCTION
# ==================================================

def apply_baseline_reduction(signal, baseline, eps=1e-12):
    """
    Apply InvBase method for baseline reduction using FFT division.
    
    Args:
        signal: (T, C) - trial signal array
        baseline: (T, C) - baseline signal array (same length as signal)
        eps: Small constant to prevent division by zero
    
    Returns:
        reduced_signal: (T, C) - baseline-reduced signal in time domain
    """
    FFT_trial = np.fft.rfft(signal, axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    return signal_reduced.astype(np.float32)


def load_single_baseline(fpath):
    """
    Load baseline signal from a single JSON file.
    
    Args:
        fpath: Path to baseline JSON file
        
    Returns:
        baseline_signal: (T, 4) array or None if loading fails
    """
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
        
        # Extract and interpolate channels
        channels = []
        for ch_name in MUSE_CHANNELS:
            ch_data = _interp_nan(_to_num(data.get(ch_name, [])))
            channels.append(ch_data)
        
        # Find minimum length
        L = min(len(ch) for ch in channels)
        if L == 0:
            return None
        
        # Stack and demean
        baseline_signal = np.stack([ch[:L] for ch in channels], axis=1)
        baseline_signal = baseline_signal - np.nanmean(baseline_signal, axis=0, keepdims=True)
        
        return baseline_signal
        
    except Exception as e:
        return None


def load_baseline_files(files, data_root):
    """
    Load baseline recordings for all subjects.
    
    Args:
        files: List of all MUSE file paths
        data_root: Root directory for data
    
    Returns:
        baseline_dict: Dictionary mapping subject IDs to baseline signals
    """
    baseline_dict = {}
    print("   Loading baseline recordings...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        subject, emotion = parse_filename(fname)
        
        if not subject or "BASELINE" in fname or subject in baseline_dict:
            continue
        
        # Try to find baseline file
        baseline_patterns = [
            os.path.join(data_root, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, subject, f"{subject}_BASELINE_STIMULUS_MUSE_cleaned", 
                       f"{subject}_BASELINE_STIMULUS_MUSE_cleaned.json")
        ]
        
        for baseline_path in baseline_patterns:
            if os.path.exists(baseline_path):
                baseline_signal = load_single_baseline(baseline_path)
                if baseline_signal is not None:
                    baseline_dict[subject] = baseline_signal
                break
    
    print(f"   ✅ Loaded {len(baseline_dict)} baseline recordings")
    return baseline_dict


# ==================================================
# SIGNAL PROCESSING
# ==================================================

def extract_channels_from_json(data):
    """
    Extract and interpolate EEG channels from JSON data.
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        channels: List of numpy arrays, one per channel
    """
    channels = []
    for ch_name in MUSE_CHANNELS:
        ch_data = _interp_nan(_to_num(data.get(ch_name, [])))
        channels.append(ch_data)
    return channels


def apply_quality_filtering(channels, data, L):
    """
    Apply quality filtering using HSI and HeadBandOn indicators.
    
    Args:
        channels: List of channel arrays
        data: Parsed JSON dictionary
        L: Length to trim channels to
        
    Returns:
        filtered_channels: List of filtered channel arrays
        final_length: Length after filtering
    """
    # Extract quality indicators
    hsi_channels = []
    for hsi_name in QUALITY_CHANNELS:
        hsi_data = _to_num(data.get(hsi_name, []))[:L]
        hsi_channels.append(hsi_data)
    
    head_on = _to_num(data.get("HeadBandOn", []))[:L]
    
    # Create validity mask
    mask = np.ones(L, dtype=bool)
    for ch in channels:
        mask &= np.isfinite(ch[:L])
    
    # Apply quality mask if available
    if len(head_on) == L and all(len(hsi) == L for hsi in hsi_channels):
        quality_mask = (head_on == 1)
        for hsi in hsi_channels:
            quality_mask &= np.isfinite(hsi) & (hsi <= HSI_THRESHOLD)
        mask &= quality_mask
    
    # Filter channels
    filtered_channels = [ch[:L][mask] for ch in channels]
    final_length = len(filtered_channels[0]) if filtered_channels else 0
    
    return filtered_channels, final_length


def create_signal_array(channels):
    """
    Stack channels into a signal array and demean.
    
    Args:
        channels: List of channel arrays
        
    Returns:
        signal: (T, C) array
    """
    signal = np.stack(channels, axis=1)
    signal = signal - np.nanmean(signal, axis=0, keepdims=True)
    return signal


def create_windows(signal, win_samples, step_samples):
    """
    Create overlapping windows from a signal.
    
    Args:
        signal: (T, C) signal array
        win_samples: Window size in samples
        step_samples: Step size in samples
        
    Returns:
        windows: List of (win_samples, C) arrays
    """
    windows = []
    L = len(signal)
    
    for start in range(0, L - win_samples + 1, step_samples):
        window = signal[start:start + win_samples]
        if len(window) == win_samples:
            windows.append(window)
    
    return windows


# ==================================================
# SINGLE FILE PROCESSING
# ==================================================

def process_single_file(fpath, config, baseline_dict, skipped_reasons):
    """
    Process a single MUSE JSON file and extract windows.
    
    Args:
        fpath: Path to JSON file
        config: Configuration object
        baseline_dict: Dictionary of baseline signals
        skipped_reasons: Dictionary to track skipped files
        
    Returns:
        Tuple of (windows, subject, emotion, clip_id, was_reduced) or None if skipped
    """
    fname = os.path.basename(fpath)
    subject, emotion = parse_filename(fname)
    
    # Validation
    if not subject or not emotion:
        skipped_reasons['parse_error'] += 1
        return None
    
    if "BASELINE" in fname:
        skipped_reasons['baseline_file'] += 1
        return None
    
    if emotion not in config.SUPERCLASS_MAP:
        skipped_reasons['unknown_emotion'] += 1
        return None
    
    # Load JSON
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
    except Exception as e:
        skipped_reasons['parse_error'] += 1
        return None
    
    # Extract channels
    channels = extract_channels_from_json(data)
    L = min(len(ch) for ch in channels)
    
    if L == 0:
        skipped_reasons['no_data'] += 1
        return None
    
    # Apply quality filtering
    channels, L = apply_quality_filtering(channels, data, L)
    
    # Check minimum length
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    if L < win_samples:
        skipped_reasons['insufficient_length'] += 1
        return None
    
    # Create signal array
    signal = create_signal_array(channels)
    
    # Apply baseline reduction if available
    was_reduced = False
    if config.USE_BASELINE_REDUCTION and subject in baseline_dict:
        baseline_signal = baseline_dict[subject]
        common_len = min(len(signal), len(baseline_signal))
        signal = apply_baseline_reduction(signal[:common_len], baseline_signal[:common_len])
        was_reduced = True
    
    # Create windows
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    windows = create_windows(signal, win_samples, step_samples)
    
    if not windows:
        skipped_reasons['insufficient_length'] += 1
        return None
    
    clip_id = f"{subject}_{emotion}"
    
    return windows, subject, emotion, clip_id, was_reduced


# ==================================================
# MAIN DATA LOADING
# ==================================================

def load_eeg_data(data_root, config):
    """
    Load EEG data from MUSE files with optional baseline reduction.
    
    Args:
        data_root: Root directory containing MUSE JSON files
        config: Configuration object with parameters
    
    Returns:
        X_raw: (N, T, C) - Raw EEG windows
        y_labels: (N,) - Class labels as integers
        subject_ids: (N,) - Subject IDs for each window
        label_to_id: Dictionary mapping label names to integers
        clip_ids: (N,) - Clip IDs for each window (for leak-free splitting)
    """
    print("\n" + "="*80)
    print("LOADING EEG DATA (MUSE) - RAW DATASET")
    print("="*80)
    
    # Find files
    files = find_muse_files(data_root)
    print(f"Found {len(files)} MUSE files")
    
    if len(files) == 0:
        print("\n❌ ERROR: No MUSE files found!")
        print(f"   Searched in: {data_root}")
        if os.path.exists(data_root):
            print(f"   Directory contents: {os.listdir(data_root)[:10]}")
        raise ValueError("No MUSE files found. Check DATA_ROOT path.")
    
    print(f"\n📁 Sample files:")
    for f in files[:3]:
        print(f"   {os.path.basename(f)}")
    
    # Load baseline files if needed
    baseline_dict = {}
    if config.USE_BASELINE_REDUCTION:
        print(f"\n🔧 Baseline Reduction: ENABLED")
        baseline_dict = load_baseline_files(files, data_root)
    else:
        print(f"\n🔧 Baseline Reduction: DISABLED")
    
    # Track statistics
    all_windows, all_labels, all_subjects, all_clip_ids = [], [], [], []
    reduced_count = 0
    not_reduced_count = 0
    skipped_reasons = {
        'baseline_file': 0,
        'unknown_emotion': 0,
        'no_data': 0,
        'insufficient_length': 0,
        'parse_error': 0
    }
    
    # Process each file
    print(f"\n⚙️  Processing files...")
    for fpath in files:
        result = process_single_file(fpath, config, baseline_dict, skipped_reasons)
        
        if result is None:
            continue
        
        windows, subject, emotion, clip_id, was_reduced = result
        superclass = config.SUPERCLASS_MAP[emotion]
        
        # Track statistics
        if was_reduced:
            reduced_count += 1
        else:
            not_reduced_count += 1
        
        # Add all windows from this recording
        for window in windows:
            all_windows.append(window)
            all_labels.append(superclass)
            all_subjects.append(subject)
            all_clip_ids.append(clip_id)
    
    # Print statistics
    print(f"\n📊 File Processing Summary:")
    print(f"   Total files found: {len(files)}")
    print(f"   Successfully processed: {len(all_windows)} windows")
    print(f"\n   Skipped files:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"      {reason}: {count}")
    
    if len(all_windows) == 0:
        print("\n❌ ERROR: No valid EEG windows extracted!")
        raise ValueError("No valid EEG data extracted.")
    
    # Convert to arrays
    X_raw = np.stack(all_windows).astype(np.float32)
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    y_labels = np.array([label_to_id[lab] for lab in all_labels], dtype=np.int64)
    subject_ids = np.array(all_subjects)
    clip_ids = np.array(all_clip_ids)
    
    print(f"\n✅ EEG data loaded: {X_raw.shape}")
    print(f"   Unique recordings: {len(np.unique(clip_ids))}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    if config.USE_BASELINE_REDUCTION:
        total_files = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction Statistics:")
        print(f"   ✅ Files with baseline reduction: {reduced_count}")
        print(f"   ⚠️  Files without baseline: {not_reduced_count}")
        if total_files > 0:
            print(f"   📈 Reduction rate: {100*reduced_count/total_files:.1f}%")
    
    return X_raw, y_labels, subject_ids, label_to_id, clip_ids


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_eeg_features(X_raw, config, fs=256.0, eps=1e-12):
    """
    Extract 26 features per channel from EEG windows.
    
    Features:
    - Differential Entropy (5 bands)
    - Log Power Spectral Density (5 bands)
    - Temporal statistics (mean, std, skewness, kurtosis)
    - DE asymmetry (left-right hemisphere)
    - Bandpower ratios (theta/alpha, beta/alpha, gamma/beta)
    - Hjorth parameters (mobility, complexity)
    - Time-domain extras (log variance, zero crossing rate)
    
    Args:
        X_raw: (N, T, C) - Raw EEG windows
        config: Configuration object with BANDS definition
        fs: Sampling frequency (Hz)
        eps: Small constant for numerical stability
    
    Returns:
        features: (N, C, 26) - Extracted features
    """
    print("Extracting EEG features (26 per channel)...")
    N, T, C = X_raw.shape
    
    # Compute power spectral density
    P = (np.abs(np.fft.rfft(X_raw, axis=1))**2) / T
    freqs = np.fft.rfftfreq(T, d=1/fs)
    
    feature_list = []
    
    # 1) Differential Entropy (5 features)
    de_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * (bp + eps))
        de_feats.append(de[..., None])
    de_all = np.concatenate(de_feats, axis=2)
    feature_list.append(de_all)
    
    # 2) Log-PSD (5 features)
    psd_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        log_psd = np.log(bp + eps)
        psd_feats.append(log_psd[..., None])
    psd_all = np.concatenate(psd_feats, axis=2)
    feature_list.append(psd_all)
    
    # 3) Temporal statistics (4 features)
    temp_mean = X_raw.mean(axis=1)[..., None]
    temp_std = X_raw.std(axis=1)[..., None]
    temp_skew = skew(X_raw, axis=1)[..., None]
    temp_kurt = kurtosis(X_raw, axis=1)[..., None]
    temp_all = np.concatenate([temp_mean, temp_std, temp_skew, temp_kurt], axis=2)
    feature_list.append(temp_all)
    
    # 4) DE asymmetry (5 features) - Left vs Right hemisphere
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2  # TP9, AF7
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2  # AF8, TP10
    de_asym = de_left - de_right
    de_asym_full = np.tile(de_asym[:, None, :], (1, C, 1))
    feature_list.append(de_asym_full)
    
    # 5) Bandpower ratios (3 features)
    band_bp = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        band_bp.append(bp)
    _, theta_bp, alpha_bp, beta_bp, gamma_bp = band_bp
    
    ratio_theta_alpha = (theta_bp + eps) / (alpha_bp + eps)
    ratio_beta_alpha = (beta_bp + eps) / (alpha_bp + eps)
    ratio_gamma_beta = (gamma_bp + eps) / (beta_bp + eps)
    ratio_all = np.stack([ratio_theta_alpha, ratio_beta_alpha, ratio_gamma_beta], axis=2)
    feature_list.append(ratio_all)
    
    # 6) Hjorth parameters (2 features)
    Xc = X_raw - X_raw.mean(axis=1, keepdims=True)
    dx = np.diff(Xc, axis=1)
    var_x = (Xc**2).mean(axis=1) + eps
    var_dx = (dx**2).mean(axis=1) + eps
    mobility = np.sqrt(var_dx / var_x)
    ddx = np.diff(dx, axis=1)
    var_ddx = (ddx**2).mean(axis=1) + eps
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / (mobility + eps)
    hjorth_all = np.stack([mobility, complexity], axis=2)
    feature_list.append(hjorth_all)
    
    # 7) Time-domain extras (2 features)
    log_var = np.log(var_x + eps)
    sign_x = np.sign(Xc)
    zc = (np.diff(sign_x, axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    td_extras = np.stack([log_var, zc], axis=2)
    feature_list.append(td_extras)
    
    # Concatenate all features
    features = np.concatenate(feature_list, axis=2)
    print(f"EEG features extracted: {features.shape}")
    
    return features.astype(np.float32)


# ==================================================
# DATA SPLITTING (LEAK-FREE)
# ==================================================

def create_data_splits(y_labels, subject_ids, clip_ids, config, 
                      train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits preventing data leakage.
    
    CRITICAL: Ensures windows from the same recording NEVER appear in 
    different splits by splitting recordings FIRST, then using their windows.
    
    Args:
        y_labels: Array of class labels
        subject_ids: Array of subject IDs
        clip_ids: Array of clip/recording IDs (unique per recording)
        config: Configuration object
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        split_indices: Dictionary with 'train', 'val', 'test' index arrays
    """
    print("\n" + "="*80)
    print("CREATING DATA SPLIT (LEAK-FREE)")
    print("="*80)
    
    n_samples = len(y_labels)
    
    if config.SUBJECT_INDEPENDENT:
        # Split by subjects (cross-subject generalization)
        print("  Strategy: SUBJECT-INDEPENDENT (split by subjects)")
        unique_subjects = np.unique(subject_ids)
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * test_ratio)
        n_val = int(len(unique_subjects) * val_ratio)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test+n_val]
        train_subjects = unique_subjects[n_test+n_val:]
        
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"  Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"  Test subjects ({len(test_subjects)}): {test_subjects}")
    else:
        # Split by clips/recordings (within-subject generalization)
        print("  Strategy: SUBJECT-DEPENDENT (split by clips/recordings)")
        print("  ⚠️  ENFORCING CLIP_INDEPENDENT to prevent data leakage")
        
        unique_clips = np.unique(clip_ids)
        np.random.shuffle(unique_clips)
        
        n_test_clips = int(len(unique_clips) * test_ratio)
        n_val_clips = int(len(unique_clips) * val_ratio)
        
        test_clips = unique_clips[:n_test_clips]
        val_clips = unique_clips[n_test_clips:n_test_clips+n_val_clips]
        train_clips = unique_clips[n_test_clips+n_val_clips:]
        
        train_mask = np.isin(clip_ids, train_clips)
        val_mask = np.isin(clip_ids, val_clips)
        test_mask = np.isin(clip_ids, test_clips)
        
        print(f"  Total unique recordings: {len(unique_clips)}")
        print(f"  Train recordings: {len(train_clips)}")
        print(f"  Val recordings: {len(val_clips)}")
        print(f"  Test recordings: {len(test_clips)}")
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train windows: {len(split_indices['train'])}")
    print(f"   Val windows: {len(split_indices['val'])}")
    print(f"   Test windows: {len(split_indices['test'])}")
    
    # Print class distribution for each split
    for split_name, indices in split_indices.items():
        labels_split = y_labels[indices]
        dist = Counter(labels_split)
        print(f"   {split_name.capitalize()} class distribution: {dict(dist)}")
    
    return split_indices
