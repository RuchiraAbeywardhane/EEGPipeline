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

# Import feature extraction from separate module
from eeg_feature_extractor import extract_eeg_features


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

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
    Load EEG recordings from MUSE files with optional baseline reduction.
    Returns full recordings (not windowed yet) to enable proper clip-independent splitting.
    
    Args:
        data_root: Root directory containing MUSE JSON files
        config: Configuration object with parameters
    
    Returns:
        recordings: List of recording dictionaries with 'signal', 'label', 'subject', 'emotion', 'filepath'
        label_to_id: Dictionary mapping label names to integers
    """
    print("\n" + "="*80)
    print("LOADING EEG DATA (MUSE)")
    print("="*80)
    
    # Search for preprocessed JSON files
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned", "*_STIMULUS_MUSE_cleaned.json")
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
    
    # Store recordings without windowing
    all_recordings = []
    
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
    
    # Minimum samples needed (will be used later during windowing)
    min_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    
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
            if L < min_samples:
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
                
                reduced_count += 1
            else:
                not_reduced_count += 1
            
            # Store the full recording (not windowed yet)
            recording_data = {
                'signal': signal.astype(np.float32),
                'label': superclass,
                'subject': subject,
                'emotion': emotion,
                'filepath': fpath
            }
            all_recordings.append(recording_data)
        
        except Exception as e:
            skipped_reasons['parse_error'] += 1
            continue
    
    # Print statistics
    print(f"\n📊 File Processing Summary:")
    print(f"   Total files found: {len(files)}")
    print(f"   Successfully processed: {len(all_recordings)} recordings")
    print(f"\n   Skipped files:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"      {reason}: {count}")
    
    if len(all_recordings) == 0:
        print("\n❌ ERROR: No valid EEG recordings loaded!")
        raise ValueError("No valid EEG data extracted.")
    
    # Create label mapping
    unique_labels = sorted(list(set(rec['label'] for rec in all_recordings)))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    
    # Print label distribution
    label_counts = Counter(rec['label'] for rec in all_recordings)
    print(f"\n✅ Loaded {len(all_recordings)} recordings")
    print(f"   Label distribution: {dict(label_counts)}")
    
    if config.USE_BASELINE_REDUCTION:
        total_files = reduced_count + not_reduced_count
        print(f"\n📊 Baseline Reduction Statistics:")
        print(f"   ✅ Files with baseline reduction: {reduced_count}")
        print(f"   ⚠️  Files without baseline: {not_reduced_count}")
        if total_files > 0:
            print(f"   📈 Reduction rate: {100*reduced_count/total_files:.1f}%")
    
    return all_recordings, label_to_id


# ==================================================
# DATA SPLITTING AND WINDOWING
# ==================================================

def create_data_splits_and_window(recordings, label_to_id, config, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split recordings first, then apply windowing to prevent clip leakage.
    This ensures that when CLIP_INDEPENDENT=True, all windows from the same recording
    stay in the same split (train/val/test).
    
    Args:
        recordings: List of recording dictionaries from load_eeg_data
        label_to_id: Dictionary mapping label names to integers
        config: Configuration object
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        X_raw: Windowed data (N, T, C)
        y_labels: Label array (N,)
        subject_ids: Subject ID array (N,)
        split_indices: Dictionary with 'train', 'val', 'test' index arrays
    """
    print("\n" + "="*80)
    print("SPLITTING AND WINDOWING DATA")
    print("="*80)
    
    n_recordings = len(recordings)
    
    # Extract metadata from recordings
    recording_subjects = np.array([r['subject'] for r in recordings])
    recording_labels = np.array([r['label'] for r in recordings])
    
    # STEP 1: Split recordings (NOT windows)
    if config.SUBJECT_INDEPENDENT:
        print("  Strategy: SUBJECT-INDEPENDENT split")
        print("  (All windows from same subject stay in same split)")
        
        unique_subjects = np.unique(recording_subjects)
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * test_ratio)
        n_val = int(len(unique_subjects) * val_ratio)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test+n_val]
        train_subjects = unique_subjects[n_test+n_val:]
        
        train_rec_mask = np.isin(recording_subjects, train_subjects)
        val_rec_mask = np.isin(recording_subjects, val_subjects)
        test_rec_mask = np.isin(recording_subjects, test_subjects)
        
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
        
    elif config.CLIP_INDEPENDENT:
        print("  Strategy: CLIP-INDEPENDENT split")
        print("  (Split recordings first, then window - NO CLIP LEAKAGE)")
        
        indices = np.arange(n_recordings)
        np.random.shuffle(indices)
        
        n_test = int(n_recordings * test_ratio)
        n_val = int(n_recordings * val_ratio)
        
        train_rec_mask = np.zeros(n_recordings, dtype=bool)
        val_rec_mask = np.zeros(n_recordings, dtype=bool)
        test_rec_mask = np.zeros(n_recordings, dtype=bool)
        
        test_rec_mask[indices[:n_test]] = True
        val_rec_mask[indices[n_test:n_test+n_val]] = True
        train_rec_mask[indices[n_test+n_val:]] = True
        
        print(f"  Train recordings: {np.sum(train_rec_mask)}")
        print(f"  Val recordings: {np.sum(val_rec_mask)}")
        print(f"  Test recordings: {np.sum(test_rec_mask)}")
    else:
        # Subject-dependent WITHOUT clip independence
        # We'll window ALL recordings first, then split randomly
        print("  Strategy: RANDOM WINDOW split")
        print("  (Window first, then split - overlapping windows may leak)")
        train_rec_mask = val_rec_mask = test_rec_mask = None
    
    # STEP 2: Apply windowing to each split separately
    win_samples = int(config.EEG_WINDOW_SEC * config.EEG_FS)
    step_samples = int(win_samples * (1.0 - config.EEG_OVERLAP))
    
    print(f"\n  Windowing parameters:")
    print(f"     Window size: {config.EEG_WINDOW_SEC}s ({win_samples} samples)")
    print(f"     Overlap: {config.EEG_OVERLAP*100:.0f}%")
    print(f"     Step size: {step_samples} samples")
    
    all_windows = []
    all_labels = []
    all_subjects = []
    window_split_ids = []  # Track which split each window belongs to
    
    split_map = {'train': 0, 'val': 1, 'test': 2}
    
    if train_rec_mask is not None:  # CLIP_INDEPENDENT or SUBJECT_INDEPENDENT
        for split_name, rec_mask in [('train', train_rec_mask), 
                                      ('val', val_rec_mask), 
                                      ('test', test_rec_mask)]:
            split_recordings = [recordings[i] for i in range(n_recordings) if rec_mask[i]]
            
            split_window_count = 0
            for rec in split_recordings:
                signal = rec['signal']
                L = len(signal)
                
                # Window this recording
                for start in range(0, L - win_samples + 1, step_samples):
                    window = signal[start:start + win_samples]
                    if len(window) == win_samples:
                        all_windows.append(window)
                        all_labels.append(label_to_id[rec['label']])
                        all_subjects.append(rec['subject'])
                        window_split_ids.append(split_map[split_name])
                        split_window_count += 1
            
            print(f"  {split_name.capitalize()}: {len([r for i, r in enumerate(recordings) if rec_mask[i]])} recordings → {split_window_count} windows")
        
        # Create split indices based on pre-assigned splits
        window_split_ids = np.array(window_split_ids)
        split_indices = {
            'train': np.where(window_split_ids == 0)[0],
            'val': np.where(window_split_ids == 1)[0],
            'test': np.where(window_split_ids == 2)[0]
        }
        
    else:  # Subject-dependent WITHOUT clip independence
        # Window ALL recordings first
        print(f"  Windowing all {n_recordings} recordings...")
        for rec in recordings:
            signal = rec['signal']
            L = len(signal)
            
            for start in range(0, L - win_samples + 1, step_samples):
                window = signal[start:start + win_samples]
                if len(window) == win_samples:
                    all_windows.append(window)
                    all_labels.append(label_to_id[rec['label']])
                    all_subjects.append(rec['subject'])
        
        # Then split windows randomly
        n_windows = len(all_windows)
        indices = np.arange(n_windows)
        np.random.shuffle(indices)
        
        n_test = int(n_windows * test_ratio)
        n_val = int(n_windows * val_ratio)
        
        split_indices = {
            'test': indices[:n_test],
            'val': indices[n_test:n_test+n_val],
            'train': indices[n_test+n_val:]
        }
        
        print(f"  Total windows: {n_windows}")
    
    # Convert to arrays
    X_raw = np.stack(all_windows).astype(np.float32)
    y_labels = np.array(all_labels, dtype=np.int64)
    subject_ids = np.array(all_subjects)
    
    print(f"\n✅ Windowed data shape: {X_raw.shape}")
    print(f"\n📋 Split Summary:")
    print(f"   Train windows: {len(split_indices['train'])}")
    print(f"   Val windows: {len(split_indices['val'])}")
    print(f"   Test windows: {len(split_indices['test'])}")
    
    # Print class distribution for each split
    for split_name, indices in split_indices.items():
        labels_split = y_labels[indices]
        dist = Counter(labels_split)
        print(f"   {split_name.capitalize()} class distribution: {dict(dist)}")
    
    return X_raw, y_labels, subject_ids, split_indices


# ==================================================
# DATA SPLITTING (Legacy - kept for backward compatibility)
# ==================================================

def create_data_splits(y_labels, subject_ids, config, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Legacy function: Create train/val/test splits with subject-independent or random strategy.
    
    WARNING: This function is called AFTER windowing, which can cause data leakage when
    CLIP_INDEPENDENT=True. Use create_data_splits_and_window() instead for proper splitting.
    
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
    print("CREATING DATA SPLIT (LEGACY)")
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
