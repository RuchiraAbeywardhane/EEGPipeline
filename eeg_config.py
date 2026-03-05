"""
EEG Pipeline Configuration
==========================

This module contains all configuration parameters for the EEG emotion recognition pipeline.

Author: Final Year Project
Date: 2026
"""

import torch

class Config:
    """EEG-specific configuration."""
    # Paths
    # DATA_ROOT = "/kaggle/input/datasets/nethmitb/emognition-processed/Output_KNN_ASR"
    DATA_ROOT = "/kaggle/input/datasets/ruchiabey/emognition"
    # DATA_ROOT = "/kaggle/input/datasets/ruchiabey/asr-outputv2-0/ASR_output"
    
    # Common parameters
    NUM_CLASSES = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True  # For EEG
    
    # Data split mode
    SUBJECT_INDEPENDENT = True
    CLIP_INDEPENDENT = True  # ⚠️ MUST ALWAYS BE TRUE to prevent data leakage!
    
    # Domain Adaptation Mode (only used when training with domain adaptation)
    # 'A': No adaptation (baseline)
    # 'B': Associative adaptation (walker + visit loss)
    # 'C': Adversarial adaptation (DANN with gradient reversal)
    # 'D': Combined (associative + adversarial) - BEST performance
    ADAPTATION_MODE = 'D'
    
    # Associative loss hyperparameters (for modes B and D)
    WALKER_WEIGHT = 1.0      # Weight for walker loss
    VISIT_WEIGHT = 0.6       # Weight for visit loss
    TEMPERATURE = 1.0        # Temperature for similarity softmax
    
    # Validation: Raise error if misconfigured
    @classmethod
    def validate_config(cls):
        """Validate configuration to prevent data leakage."""
        if not cls.CLIP_INDEPENDENT:
            raise ValueError(
                "CLIP_INDEPENDENT must always be True to prevent data leakage! "
                "Setting it to False causes windows from the same recording to appear "
                "in both train and test sets, leading to inflated accuracy."
            )
        
        if cls.ADAPTATION_MODE not in ['A', 'B', 'C', 'D']:
            raise ValueError(
                f"ADAPTATION_MODE must be 'A', 'B', 'C', or 'D', got '{cls.ADAPTATION_MODE}'"
            )
    
    # LOSO (Leave-One-Subject-Out) training
    USE_LOSO = False  # Set to True to enable LOSO cross-validation
    LOSO_SUBJECT = None  # Specify subject ID to leave out (None = iterate through all subjects)
    
    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10
    
    # Label mappings (4-class emotion quadrants)
    SUPERCLASS_MAP = {
        "ENTHUSIASM": "Q1",  # Positive + High Arousal
        "FEAR": "Q2",         # Negative + High Arousal
        "SADNESS": "Q3",      # Negative + Low Arousal
        "NEUTRAL": "Q4",      # Positive + Low Arousal
    }
    
    SUPERCLASS_ID = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = ["Q1_Positive_Active", "Q2_Negative_Active", 
                    "Q3_Negative_Calm", "Q4_Positive_Calm"]
    
    # EEG parameters
    EEG_FS = 256.0
    EEG_CHANNELS = 4  # TP9, AF7, AF8, TP10
    EEG_FEATURES = 26
    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP = 0.5 if CLIP_INDEPENDENT else 0.0
    EEG_BATCH_SIZE = 32 if CLIP_INDEPENDENT else 64
    EEG_EPOCHS = 200 if CLIP_INDEPENDENT else 150
    EEG_LR = 5e-4 if CLIP_INDEPENDENT else 1e-3
    EEG_PATIENCE = 30 if CLIP_INDEPENDENT else 20
    EEG_CHECKPOINT = "best_eeg_model.pt"
    
    # Augmentation settings
    USE_MIXUP = True  # Set to True to enable Mixup data augmentation
    MIXUP_ALPHA = 0.2  # Mixup interpolation strength (only used if USE_MIXUP=True)
    LABEL_SMOOTHING = 0.1 if CLIP_INDEPENDENT else 0.0
    
    # Frequency bands for feature extraction
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)), 
             ("beta", (14, 30)), ("gamma", (31, 45))]
