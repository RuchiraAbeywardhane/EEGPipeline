"""
    EEG-Only Emotion Recognition Pipeline
    ======================================
    
    This script contains a complete EEG emotion recognition pipeline with:
    - Preprocessed EEG data loading (MUSE headband)
    - Baseline reduction (InvBase method)
    - Feature extraction (26 features per channel)
    - BiLSTM classifier with attention
    - Subject-independent or subject-dependent splits
    
    Author: Final Year Project
    Date: 2026
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report

# Import configuration
from eeg_config import Config

# Import model architecture
from eeg_bilstm_model import SimpleBiLSTMClassifier

# Import deep learning feature extractor
from eeg_deep_feature_extractor import create_feature_extractor

# Import training functions from separate module
from eeg_trainer import train_eeg_model


# Global config instance
config = Config()

# Set random seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

print(f"Device: {config.DEVICE}")


# ==================================================
# DATASET LOADER REGISTRY
# ==================================================

def _get_loader(dataset: str):
    """
    Return the (load_eeg_data, extract_eeg_features, create_data_splits)
    triple for the requested dataset.

    Supported values
    ----------------
    'emognition' : Original EmOgnition MUSE JSON dataset (256 Hz)
    'muse_csv'   : New MUSE CSV wearable dataset (128 Hz)
    """
    if dataset == 'emognition':
        from eeg_data_loader_emognitionRaw import (
            load_eeg_data,
            extract_eeg_features,
            create_data_splits,
        )
        return load_eeg_data, extract_eeg_features, create_data_splits

    elif dataset == 'muse_csv':
        from eeg_data_loader_museCSV import (
            load_eeg_data,
            create_data_splits,
        )
        # Both datasets use the same handcrafted feature extractor
        from eeg_feature_extractor import extract_eeg_features
        return load_eeg_data, extract_eeg_features, create_data_splits

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Set config.DATASET to 'emognition' or 'muse_csv'."
        )


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline."""
    # Validate configuration (also sets DATA_ROOT and EEG_FS)
    config.validate_config()

    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print(f"Dataset       : {config.DATASET}")
    print(f"Data root     : {config.DATA_ROOT}")
    print(f"Sampling rate : {config.EEG_FS} Hz")
    print(f"Mode          : {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Clip-Independent    : {config.CLIP_INDEPENDENT}")
    print(f"Baseline Reduction  : {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)

    # ── Step 1: Pick the right loader ──────────────────────────────────────────
    load_eeg_data, extract_eeg_features, create_data_splits = _get_loader(config.DATASET)

    # ── Step 2: Load raw EEG data ──────────────────────────────────────────────
    # Both loaders return the same 5-tuple:
    #   X_raw (N,T,C), y_labels (N,), subject_ids (N,), label_to_id, clip_ids (N,)
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(
        config.DATA_ROOT, config
    )

    # ── Step 3: Extract handcrafted features ───────────────────────────────────
    # feature extractor uses config.EEG_FS which was set by validate_config()
    eeg_X_features = extract_eeg_features(eeg_X_raw, config, fs=config.EEG_FS)

    # ── Step 4: Dynamically set NUM_CLASSES from the loaded data ───────────────
    # (muse_csv has 3 active quadrants; emognition has 4)
    n_classes_actual = len(label_to_id)
    if n_classes_actual != config.NUM_CLASSES:
        print(f"\n⚠️  Adjusting NUM_CLASSES: config={config.NUM_CLASSES} "
              f"→ actual={n_classes_actual} (based on loaded labels)")
        config.NUM_CLASSES = n_classes_actual

    # ── Step 5: Create leak-free data splits ───────────────────────────────────
    split_indices = create_data_splits(eeg_y, eeg_subjects, eeg_clip_ids, config)

    # ── Step 6: Train ──────────────────────────────────────────────────────────
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(
        eeg_X_features, eeg_y, split_indices, label_to_id, config
    )

    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Dataset used  : {config.DATASET}")
    print(f"✅ Model saved   : {config.EEG_CHECKPOINT}")
    print("=" * 80)


if __name__ == "__main__":
    main()