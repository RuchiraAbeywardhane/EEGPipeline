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

# Import data loading functions from RAW emognition dataset
from eeg_data_loader_emognitionRaw import (
    load_eeg_data,
    extract_eeg_features,
    create_data_splits
)

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
# MAIN EXECUTION
# ==================================================

def main():
    """EEG-only emotion recognition pipeline."""
    # Validate configuration to prevent data leakage
    config.validate_config()
    
    print("=" * 80)
    print("EEG-ONLY EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Mode: {'Subject-Independent' if config.SUBJECT_INDEPENDENT else 'Subject-Dependent'}")
    print(f"Clip-Independent: {config.CLIP_INDEPENDENT}")
    print(f"Baseline Reduction: {config.USE_BASELINE_REDUCTION}")
    print("=" * 80)
    
    # Step 1: Load EEG data (RAW dataset returns: X_raw, y_labels, subject_ids, label_to_id, clip_ids)
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(config.DATA_ROOT, config)
    
    # Step 2: Extract features
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Step 3: Create data splits (now requires clip_ids to prevent leakage)
    split_indices = create_data_splits(eeg_y, eeg_subjects, eeg_clip_ids, config)
    
    # Step 4: Train EEG model
    eeg_model, eeg_mu, eeg_sd = train_eeg_model(eeg_X_features, eeg_y, split_indices, label_to_id, config)
    
    print("\n" + "=" * 80)
    print("🎉 EEG PIPELINE COMPLETE! 🎉")
    print("=" * 80)
    print(f"✅ Model saved: {config.EEG_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    main()