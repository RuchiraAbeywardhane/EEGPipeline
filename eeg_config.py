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
    # ------------------------------------------------------------------
    # DATASET SELECTOR
    # ------------------------------------------------------------------
    # 'emognition' : Original EmOgnition dataset (MUSE JSON, 256 Hz)
    #                Emotions: AMUSEMENT, ANGER, SADNESS, NEUTRAL
    # 'muse_csv'   : New MUSE CSV dataset (128 Hz, preprocessed CSVs)
    #                Emotions: ANGER, FEAR, HAPPINESS, SADNESS
    DATASET = 'muse_csv'

    # Paths — EmOgnition (original)
    DATA_ROOT_EMOGNITION = "/kaggle/input/datasets/ruchiabey/emognition"

    # Paths — New MUSE CSV dataset (EmoKey / EKM-ED)
    # DATA_ROOT_MUSE_CSV must point to the folder that contains the numbered
    # subject sub-directories (1/, 2/, 103/ etc.)
    # i.e. the full path ending in .../clean-signals/0.0078125S
    DATA_ROOT_MUSE_CSV = "/kaggle/input/datasets/ruchiabey/emoky-dataset/EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/preprocessed/clean-signals/0.0078125S"

    # Active DATA_ROOT (auto-selected below, or override manually)
    DATA_ROOT = DATA_ROOT_EMOGNITION   # will be overridden by validate_config()

    # ------------------------------------------------------------------
    # Common parameters
    # ------------------------------------------------------------------
    NUM_CLASSES = 4
    SEED = 84
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline reduction (InvBase method)
    USE_BASELINE_REDUCTION = True

    # Data split mode
    SUBJECT_INDEPENDENT = False
    CLIP_INDEPENDENT = True  # ⚠️ RECOMMENDED: Keep True to prevent data leakage!

    # Domain Adaptation Mode (only used when training with domain adaptation)
    # 'A': No adaptation (baseline)
    # 'B': Associative adaptation (walker + visit loss)
    # 'C': Adversarial adaptation (DANN with gradient reversal)
    # 'D': Combined (associative + adversarial) - BEST performance
    ADAPTATION_MODE = 'D'

    # Associative loss hyperparameters (for modes B and D)
    WALKER_WEIGHT = 1.0
    VISIT_WEIGHT = 0.6
    TEMPERATURE = 1.0

    # LOSO (Leave-One-Subject-Out) training
    USE_LOSO = False
    LOSO_SUBJECT = None

    # Stratified split parameters
    USE_STRATIFIED_GROUP_SPLIT = True
    MIN_SAMPLES_PER_CLASS = 10

    # ------------------------------------------------------------------
    # Label mappings — EmOgnition dataset (4-class emotion quadrants)
    # ------------------------------------------------------------------
    SUPERCLASS_MAP = {
        "AMUSEMENT": "Q1",  # Positive + High Arousal
        "ANGER":     "Q2",  # Negative + High Arousal
        "SADNESS":   "Q3",  # Negative + Low Arousal
        "NEUTRAL":   "Q4",  # Positive + Low Arousal
    }

    # ------------------------------------------------------------------
    # Label mappings — New MUSE CSV dataset (Russell's circumplex model)
    # ------------------------------------------------------------------
    # HAPPINESS → Q1 (Positive Valence, High Arousal)
    # ANGER     → Q2 (Negative Valence, High Arousal)
    # FEAR      → Q2 (Negative Valence, High Arousal)  — same quadrant as ANGER
    # SADNESS   → Q3 (Negative Valence, Low Arousal)
    # NOTE: Only 3 quadrants are populated (Q4 absent in this dataset).
    #       The pipeline handles this gracefully via label_to_id.
    MUSE_CSV_SUPERCLASS_MAP = {
        "HAPPINESS": "Q1",
        "ANGER":     "Q2",
        "FEAR":      "Q2",
        "SADNESS":   "Q3",
    }

    SUPERCLASS_ID  = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL   = ["Q1_Positive_Active", "Q2_Negative_Active",
                      "Q3_Negative_Calm",   "Q4_Positive_Calm"]

    # ------------------------------------------------------------------
    # EEG signal parameters
    # ------------------------------------------------------------------
    EEG_FS       = 256.0   # EmOgnition native FS (overridden to 128 for muse_csv)
    EEG_CHANNELS = 4       # TP9, AF7, AF8, TP10
    EEG_FEATURES = 26

    EEG_WINDOW_SEC = 10.0
    EEG_OVERLAP    = 0.75

    EEG_BATCH_SIZE = 16
    EEG_EPOCHS     = 300
    EEG_LR         = 1e-3
    EEG_PATIENCE   = 50
    EEG_CHECKPOINT = "best_eeg_model.pt"

    # Feature Extraction Mode
    # 'handcrafted'         : 26 handcrafted features (DE, PSD, etc.)
    # 'deep_cnn'            : CNN learns features from raw EEG
    # 'deep_transformer'    : Transformer learns features from raw EEG
    # 'hybrid_cnn'          : handcrafted + CNN
    # 'hybrid_transformer'  : handcrafted + Transformer
    FEATURE_EXTRACTION_MODE = 'handcrafted'

    # Deep learning feature extractor parameters
    DEEP_FEATURE_DIM  = 128
    CNN_FILTERS       = [32, 64, 128]
    CNN_KERNEL_SIZE   = 7
    TRANSFORMER_HEADS  = 4
    TRANSFORMER_LAYERS = 2

    # Hybrid feature fusion
    HYBRID_FUSION_MODE = 'concat'  # 'concat' | 'attention'

    # Augmentation settings
    USE_MIXUP     = True
    MIXUP_ALPHA   = 0.4
    LABEL_SMOOTHING = 0.1

    # Frequency bands for feature extraction
    BANDS = [("delta", (1, 3)), ("theta", (4, 7)), ("alpha", (8, 13)),
             ("beta",  (14, 30)), ("gamma", (31, 45))]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @classmethod
    def validate_config(cls):
        """Validate configuration and set DATA_ROOT based on DATASET."""
        # Check DATASET value
        valid_datasets = ('emognition', 'muse_csv')
        if cls.DATASET not in valid_datasets:
            raise ValueError(
                f"DATASET must be one of {valid_datasets}, got '{cls.DATASET}'"
            )

        # Auto-select DATA_ROOT
        if cls.DATASET == 'emognition':
            cls.DATA_ROOT = cls.DATA_ROOT_EMOGNITION
            cls.EEG_FS    = 256.0
        else:  # muse_csv
            cls.DATA_ROOT = cls.DATA_ROOT_MUSE_CSV
            cls.EEG_FS    = 128.0   # this dataset is at 128 Hz

        # Check ADAPTATION_MODE
        if cls.ADAPTATION_MODE not in ('A', 'B', 'C', 'D'):
            raise ValueError(
                f"ADAPTATION_MODE must be 'A', 'B', 'C', or 'D', "
                f"got '{cls.ADAPTATION_MODE}'"
            )

        print(f"✅ Config validated  |  DATASET='{cls.DATASET}'  "
              f"|  DATA_ROOT='{cls.DATA_ROOT}'  |  EEG_FS={cls.EEG_FS} Hz")
