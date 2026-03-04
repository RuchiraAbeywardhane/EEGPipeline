"""
EEG Data Loader Module
======================

This module handles all data loading, preprocessing, and feature extraction
for the EEG emotion recognition pipeline.

Features:
- MUSE headband EEG data loading from JSON files
- Baseline reduction (InvBase method)
- Quality filtering (HSI, HeadBandOn)
- Windowing with configurable overlap
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
# UTILITY FUNCTIONS
# ==================================================

def check_json_structure(data_root, num_samples=3):
    """
    Check and print the structure of JSON files in the dataset.
    Useful for debugging and understanding new dataset formats.
    
    Args:
        data_root: Root directory containing JSON files
        num_samples: Number of sample files to examine
    """
    print("\n" + "="*80)
    print("CHECKING JSON STRUCTURE")
    print("="*80)
    
    # Search for JSON files
    patterns = [
        os.path.join(data_root, "*.json"),
        os.path.join(data_root, "*", "*.json"),
        os.path.join(data_root, "*", "*", "*.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    
    print(f"Found {len(files)} JSON files")
    
    if len(files) == 0:
        print(f"\n❌ No JSON files found in: {data_root}")
        return
    
    # Examine first few files
    for i, fpath in enumerate(files[:num_samples]):
        print(f"\n{'='*80}")
        print(f"File {i+1}: {os.path.basename(fpath)}")
        print(f"Full path: {fpath}")
        print(f"{'='*80}")
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Print top-level keys
            print(f"\n📋 Top-level keys ({len(data)} total):")
            for key in sorted(data.keys()):
                value = data[key]
                
                # Determine type and length
                if isinstance(value, list):
                    if len(value) > 0:
                        sample_val = value[0]
                        print(f"   {key:20s} -> list[{len(value)}] (first element: {type(sample_val).__name__})")
                    else:
                        print(f"   {key:20s} -> list[0] (empty)")
                elif isinstance(value, dict):
                    print(f"   {key:20s} -> dict with {len(value)} keys")
                else:
                    print(f"   {key:20s} -> {type(value).__name__}: {value}")
            
            # Sample data from arrays
            print(f"\n📊 Sample data from arrays:")
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, list) and len(value) > 0:
                    sample_vals = value[:5]
                    print(f"   {key:20s}: {sample_vals}")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n{'='*80}")
    print(f"Showing {min(num_samples, len(files))} of {len(files)} total files")
    print(f"{'='*80}\n")

def _to_num(x):
    """
    Convert various input types to numeric numpy array.
    
    Args:
        x: Input (list, scalar, etc.)
    
    Returns:
        numpy array of float64
    """
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    """
    Interpolate NaN values in a 1D array using linear interpolation.
    
    Args:
        a: Input array with potential NaN values
    
    Returns:
        Array with NaN values interpolated
    """
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
# BASELINE REDUCTION
# ==================================================

def apply_baseline_reduction(signal, baseline, eps=1e-12):
    """
    Apply InvBase method for baseline reduction.
    
    This method divides the trial FFT by the baseline FFT to reduce
    inter-subject variability by normalizing against each subject's
    resting state baseline.
    
    Args:
        signal: (T, C) - trial signal array
        baseline: (T, C) - baseline signal array (same length as signal)
        eps: small constant to prevent division by zero
    
    Returns:
        reduced_signal: (T, C) - baseline-reduced signal in time domain
    
    Reference:
        InvBase method for EEG baseline reduction
    """
    # Compute FFT for each channel
    FFT_trial = np.fft.rfft(signal, axis=0)
    FFT_baseline = np.fft.rfft(baseline, axis=0)
    
    # InvBase: divide trial by baseline (element-wise per channel)
    FFT_reduced = FFT_trial / (np.abs(FFT_baseline) + eps)
    
    # Convert back to time domain
    signal_reduced = np.fft.irfft(FFT_reduced, n=len(signal), axis=0)
    
    return signal_reduced.astype(np.float32)


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
        parts = fname.split("_")
        
        if len(parts) < 2:
            continue
        
        subject = parts[0]
        
        # Skip if already loaded or if this IS a baseline file
        if subject in baseline_dict or "BASELINE" in fname:
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
                try:
                    with open(baseline_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract baseline channels
                    tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
                    af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
                    af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
                    tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
                    
                    L = min(len(tp9), len(af7), len(af8), len(tp10))
                    if L > 0:
                        baseline_signal = np.stack([tp9[:L], af7[:L], af8[:L], tp10[:L]], axis=1)
                        baseline_signal = baseline_signal - np.nanmean(baseline_signal, axis=0, keepdims=True)
                        baseline_dict[subject] = baseline_signal
                        
                except Exception as e:
                    print(f"   ⚠️  Failed to load baseline for {subject}: {e}")
                break
    
    print(f"   ✅ Loaded {len(baseline_dict)} baseline recordings")
    return baseline_dict


# ==================================================
# DATA LOADING
# ==================================================

def load_eeg_data(data_root, config):
    """
    Load EEG data from MUSE files with optional baseline reduction.
    
    Args:
        data_root: Root directory containing MUSE JSON files
        config: Configuration object with parameters (window size, overlap, etc.)
    
    Returns:
        X_raw: (N, T, C) - Raw EEG windows
        y_labels: (N,) - Class labels as integers
        subject_ids: (N,) - Subject IDs for each window
        label_to_id: Dictionary mapping label names to integers
    """
    print("\n" + "="*80)
    print("LOADING EEG DATA (MUSE) - RAW DATASET")
    print("="*80)
    
    # Search for RAW JSON files (without "_cleaned" suffix)
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*", "*_STIMULUS_MUSE.json")
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
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
    
    # Prepare windowing parameters
    all_windows, all_labels, all_subjects = [], [], []
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    
    # Track statistics
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
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        
        if len(parts) < 2:
            skipped_reasons['parse_error'] += 1
            continue
        
        # Skip baseline files themselves
        if "BASELINE" in fname:
            skipped_reasons['baseline_file'] += 1
            continue
            
        subject = parts[0]
        emotion = parts[1].upper()
        
        if emotion not in config.SUPERCLASS_MAP:
            skipped_reasons['unknown_emotion'] += 1
            continue
        
        superclass = config.SUPERCLASS_MAP[emotion]
        
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            # Extract 4 EEG channels (MUSE: TP9, AF7, AF8, TP10)
            tp9 = _interp_nan(_to_num(data.get("RAW_TP9", [])))
            af7 = _interp_nan(_to_num(data.get("RAW_AF7", [])))
            af8 = _interp_nan(_to_num(data.get("RAW_AF8", [])))
            tp10 = _interp_nan(_to_num(data.get("RAW_TP10", [])))
            
            L = min(len(tp9), len(af7), len(af8), len(tp10))
            if L == 0:
                skipped_reasons['no_data'] += 1
                continue
            
            # Quality filtering using HSI and HeadBandOn
            hsi_tp9 = _to_num(data.get("HSI_TP9", []))[:L]
            hsi_af7 = _to_num(data.get("HSI_AF7", []))[:L]
            hsi_af8 = _to_num(data.get("HSI_AF8", []))[:L]
            hsi_tp10 = _to_num(data.get("HSI_TP10", []))[:L]
            head_on = _to_num(data.get("HeadBandOn", []))[:L]
            
            mask = np.isfinite(tp9[:L]) & np.isfinite(af7[:L]) & np.isfinite(af8[:L]) & np.isfinite(tp10[:L])
            if len(head_on) == L and len(hsi_tp9) == L:
                quality_mask = (
                    (head_on == 1) &
                    np.isfinite(hsi_tp9) & (hsi_tp9 <= 2) &
                    np.isfinite(hsi_af7) & (hsi_af7 <= 2) &
                    np.isfinite(hsi_af8) & (hsi_af8 <= 2) &
                    np.isfinite(hsi_tp10) & (hsi_tp10 <= 2)
                )
                mask = mask & quality_mask
            
            tp9, af7, af8, tp10 = tp9[:L][mask], af7[:L][mask], af8[:L][mask], tp10[:L][mask]
            L = len(tp9)
            if L < win_samples:
                skipped_reasons['insufficient_length'] += 1
                continue
            
            # Stack channels: (T, 4)
            signal = np.stack([tp9, af7, af8, tp10], axis=1)
            signal = signal - np.nanmean(signal, axis=0, keepdims=True)
            
            # Apply baseline reduction if available
            if config.USE_BASELINE_REDUCTION and subject in baseline_dict:
                baseline_signal = baseline_dict[subject]
                
                # Match lengths
                common_len = min(len(signal), len(baseline_signal))
                signal_trim = signal[:common_len]
                baseline_trim = baseline_signal[:common_len]
                
                # Apply InvBase method
                signal = apply_baseline_reduction(signal_trim, baseline_trim)
                L = len(signal)
                
                reduced_count += 1
            else:
                not_reduced_count += 1
            
            # Create windows with overlap
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(superclass)
                    all_subjects.append(subject)
        
        except Exception as e:
            skipped_reasons['parse_error'] += 1
            continue
    
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
    
    print(f"\n✅ EEG data loaded: {X_raw.shape}")
    print(f"   Label distribution: {Counter(all_labels)}")
    
    if config.USE_BASELINE_REDUCTION:
        total_files = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction Statistics:")
        print(f"   ✅ Files with baseline reduction: {reduced_count}")
        print(f"   ⚠️  Files without baseline: {not_reduced_count}")
        if total_files > 0:
            print(f"   📈 Reduction rate: {100*reduced_count/total_files:.1f}%")
    
    return X_raw, y_labels, subject_ids, label_to_id


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_eeg_features(X_raw, config, fs=256.0, eps=1e-12):
    """
    Extract 26 features per channel from EEG windows.
    
    Features include:
    1. Differential Entropy (5 bands)
    2. Log Power Spectral Density (5 bands)
    3. Temporal statistics (mean, std, skewness, kurtosis)
    4. DE asymmetry (left-right hemisphere)
    5. Bandpower ratios (theta/alpha, beta/alpha, gamma/beta)
    6. Hjorth parameters (mobility, complexity)
    7. Time-domain extras (log variance, zero crossing rate)
    
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
    
    # 4) DE asymmetry (5 features)
    # Left hemisphere: TP9 (idx 0), AF7 (idx 1)
    # Right hemisphere: AF8 (idx 2), TP10 (idx 3)
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2
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
# DATA SPLITTING
# ==================================================

def create_data_splits(y_labels, subject_ids, config, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits with subject-independent or random strategy.
    
    Args:
        y_labels: Array of class labels
        subject_ids: Array of subject IDs
        config: Configuration object
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        split_indices: Dictionary with 'train', 'val', 'test' index arrays
    """
    print("\n" + "="*80)
    print("CREATING DATA SPLIT")
    print("="*80)
    
    n_samples = len(y_labels)
    
    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT split")
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
        
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
    else:
        print("  Strategy: RANDOM split")
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_test = int(n_samples * test_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        
        test_mask[indices[:n_test]] = True
        val_mask[indices[n_test:n_test+n_val]] = True
        train_mask[indices[n_test+n_val:]] = True
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📋 Split Summary:")
    print(f"   Train samples: {len(split_indices['train'])}")
    print(f"   Val samples: {len(split_indices['val'])}")
    print(f"   Test samples: {len(split_indices['test'])}")
    
    # Print class distribution for each split
    for split_name, indices in split_indices.items():
        labels_split = y_labels[indices]
        dist = Counter(labels_split)
        print(f"   {split_name.capitalize()} class distribution: {dict(dist)}")
    
    return split_indices
