"""
Group-Specific EEG Training Script (JDA-style)
===============================================

This script trains a model using ONLY a specific subset of subjects.
Implements JDA-style domain adaptation for cross-subject generalization
within the subject group.

Based on the JDA paper approach:
- Uses all subjects in group as source domain (combined)
- Tests on one held-out subject from the group (LOSO within group)
- Or trains on all subjects and tests on new subjects

Usage:
    # Train on specific group with LOSO
    python train_group_model.py --subjects 57 58 63 64 48 60 --mode loso
    
    # Train on all subjects in group, evaluate on group average
    python train_group_model.py --subjects 57 58 63 64 48 60 --mode group

Author: Final Year Project
Date: 2026
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data, extract_eeg_features


# ==================================================
# GROUP-SPECIFIC MODEL (JDA-INSPIRED)
# ==================================================

class GroupSpecificModel(nn.Module):
    """
    Model for specific subject groups (based on JDA architecture).
    
    Architecture matches JDA paper:
    - Feature extractor: input_dim -> 128
    - Label predictor: 128 -> n_classes
    - (Optional) Domain predictor for within-group adaptation
    """
    
    def __init__(self, input_dim=104, n_classes=4, dropout=0.3):
        super().__init__()
        
        # Feature extractor (310 -> 128 in JDA paper, but we have 104 features)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Label predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(128, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, input_dim) - Input features
            return_features: If True, return intermediate features
        
        Returns:
            logits: (B, n_classes)
            features: (B, 128) if return_features=True
        """
        features = self.feature_extractor(x)
        logits = self.label_predictor(features)
        
        if return_features:
            return logits, features
        return logits


# ==================================================
# DATA LOADING FOR SUBJECT GROUP
# ==================================================

def load_group_data(subject_list, config):
    """
    Load data for a specific group of subjects with NON-OVERLAPPING windows.
    
    Args:
        subject_list: List of subject IDs (e.g., [57, 58, 63, 64, 48, 60])
        config: Configuration object
    
    Returns:
        X_features: (N, input_dim) - Features for all subjects in group
        y_labels: (N,) - Emotion labels
        subject_ids: (N,) - Subject IDs (for LOSO splitting)
        clip_ids: (N,) - Recording IDs
        label_to_id: Label mapping
    """
    print("\n" + "="*80)
    print(f"LOADING DATA FOR SUBJECT GROUP: {subject_list}")
    print("="*80)
    
    # Temporarily set overlap to 0 for non-overlapping windows
    original_overlap = config.EEG_OVERLAP
    config.EEG_OVERLAP = 0.0  # NON-OVERLAPPING (like JDA paper)
    
    # Load all data
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(config.DATA_ROOT, config)
    
    # Restore original overlap
    config.EEG_OVERLAP = original_overlap
    
    # Filter for subject group
    subject_list_str = [str(s) for s in subject_list]
    group_mask = np.isin(eeg_subjects, subject_list_str)
    
    if not group_mask.any():
        available_subjects = np.unique(eeg_subjects)
        raise ValueError(
            f"No subjects from group {subject_list} found in dataset!\n"
            f"Available subjects: {available_subjects}"
        )
    
    X_raw_group = eeg_X_raw[group_mask]
    y_group = eeg_y[group_mask]
    subjects_group = eeg_subjects[group_mask]
    clip_ids_group = eeg_clip_ids[group_mask]
    
    # Check which subjects are present
    present_subjects = np.unique(subjects_group)
    missing_subjects = set(subject_list_str) - set(present_subjects)
    
    if missing_subjects:
        print(f"\n⚠️  WARNING: These subjects are missing from dataset: {missing_subjects}")
    
    print(f"\n✅ Group data loaded:")
    print(f"   Subjects present: {present_subjects}")
    print(f"   Total windows: {len(y_group)} (NON-OVERLAPPING)")
    print(f"   Unique recordings: {len(np.unique(clip_ids_group))}")
    print(f"   Emotion distribution: {np.bincount(y_group, minlength=config.NUM_CLASSES)}")
    
    # Extract features
    print("\n🔧 Extracting features...")
    X_features = extract_eeg_features(X_raw_group, config)
    
    # Flatten features: (N, C, F) -> (N, C*F)
    N, C, F = X_features.shape
    X_features_flat = X_features.reshape(N, C * F)
    
    print(f"   Feature shape: {X_features_flat.shape}")
    
    return X_features_flat, y_group, subjects_group, clip_ids_group, label_to_id


def create_loso_splits(y_labels, subjects, clip_ids, test_subject, config):
    """
    Create LOSO (Leave-One-Subject-Out) splits for within-group evaluation.
    
    Similar to JDA paper where they use 13 subjects for training, 1 for testing.
    
    Args:
        y_labels: (N,) emotion labels
        subjects: (N,) subject IDs
        clip_ids: (N,) recording IDs
        test_subject: Subject to hold out for testing
        config: Configuration object
    
    Returns:
        split_indices: Dict with 'train', 'val', 'test' indices
    """
    print("\n" + "="*80)
    print(f"CREATING LOSO SPLIT (Test Subject: {test_subject})")
    print("="*80)
    
    # Test set: all data from test_subject
    test_mask = (subjects == str(test_subject))
    
    # Train+Val set: all other subjects
    train_val_mask = ~test_mask
    
    if not test_mask.any():
        raise ValueError(f"Test subject {test_subject} not found in data!")
    
    # Split train+val into train and val (90/10 split)
    train_val_subjects = np.unique(subjects[train_val_mask])
    n_val_subjects = max(1, int(0.1 * len(train_val_subjects)))
    
    np.random.shuffle(train_val_subjects)
    val_subjects = train_val_subjects[:n_val_subjects]
    train_subjects = train_val_subjects[n_val_subjects:]
    
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"   Train subjects: {train_subjects}")
    print(f"   Val subjects: {val_subjects}")
    print(f"   Test subject: {test_subject}")
    
    print(f"\n📊 Window Distribution:")
    print(f"   Train: {len(split_indices['train'])} windows")
    print(f"   Val:   {len(split_indices['val'])} windows")
    print(f"   Test:  {len(split_indices['test'])} windows")
    
    # Class distribution per split
    for split_name, indices in split_indices.items():
        if len(indices) == 0:
            continue
        labels_split = y_labels[indices]
        dist = np.bincount(labels_split, minlength=config.NUM_CLASSES)
        print(f"   {split_name.capitalize()} class dist: {dist}")
    
    return split_indices


def create_group_splits(y_labels, subjects, clip_ids, config, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits using all subjects in group (by windows).
    
    This is for when you want to train on the group and test generalization
    within the group (not LOSO).
    
    Args:
        y_labels: (N,) emotion labels
        subjects: (N,) subject IDs
        clip_ids: (N,) recording IDs
        config: Configuration object
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        split_indices: Dict with 'train', 'val', 'test' indices
    """
    print("\n" + "="*80)
    print("CREATING GROUP SPLIT (ALL SUBJECTS, STRATIFIED BY CLASS)")
    print("="*80)
    
    n_samples = len(y_labels)
    
    # Stratified split by class (window-level)
    from collections import defaultdict
    windows_by_class = defaultdict(list)
    for idx in range(n_samples):
        windows_by_class[y_labels[idx]].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for class_id in range(config.NUM_CLASSES):
        class_windows = np.array(windows_by_class[class_id])
        n_class = len(class_windows)
        
        if n_class == 0:
            print(f"   ⚠️  No windows for class {class_id}")
            continue
        
        np.random.shuffle(class_windows)
        
        n_test = max(1, int(n_class * test_ratio))
        n_val = max(1, int(n_class * val_ratio))
        
        test_indices.extend(class_windows[:n_test])
        val_indices.extend(class_windows[n_test:n_test+n_val])
        train_indices.extend(class_windows[n_test+n_val:])
        
        print(f"   Class {class_id}: {n_class} windows → Train:{len(class_windows[n_test+n_val:])}, Val:{n_val}, Test:{n_test}")
    
    # Convert to masks
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    print(f"\n📊 Window Distribution:")
    print(f"   Train: {len(split_indices['train'])} windows")
    print(f"   Val:   {len(split_indices['val'])} windows")
    print(f"   Test:  {len(split_indices['test'])} windows")
    
    return split_indices


def create_group_splits_clip_independent(y_labels, subjects, clip_ids, config, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits using all subjects in group (by CLIPS).
    
    CLIP-INDEPENDENT: Ensures entire clips stay together in one split.
    No data leakage - if a recording is in test, ALL its windows are in test.
    
    Args:
        y_labels: (N,) emotion labels
        subjects: (N,) subject IDs
        clip_ids: (N,) recording IDs
        config: Configuration object
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        split_indices: Dict with 'train', 'val', 'test' indices
    """
    print("\n" + "="*80)
    print("CREATING GROUP SPLIT (CLIP-INDEPENDENT, NO DATA LEAKAGE)")
    print("="*80)
    
    from collections import defaultdict
    
    # Get unique clips and their label distributions
    unique_clips = np.unique(clip_ids)
    n_clips = len(unique_clips)
    
    print(f"   Total unique clips: {n_clips}")
    
    # For each clip, determine its dominant label and collect all window indices
    clip_info = {}  # clip_id -> {'label': dominant_label, 'indices': [window_indices]}
    
    for clip_id in unique_clips:
        clip_mask = (clip_ids == clip_id)
        clip_indices = np.where(clip_mask)[0]
        clip_labels = y_labels[clip_mask]
        
        # Dominant label (most common emotion in this clip)
        dominant_label = np.bincount(clip_labels).argmax()
        
        clip_info[clip_id] = {
            'label': dominant_label,
            'indices': clip_indices,
            'n_windows': len(clip_indices)
        }
    
    # Group clips by their dominant label
    clips_by_class = defaultdict(list)
    for clip_id, info in clip_info.items():
        clips_by_class[info['label']].append(clip_id)
    
    # Split clips by class (stratified)
    train_clips, val_clips, test_clips = [], [], []
    
    for class_id in range(config.NUM_CLASSES):
        class_clip_list = clips_by_class[class_id]
        n_class_clips = len(class_clip_list)
        
        if n_class_clips == 0:
            print(f"   ⚠️  No clips for class {class_id}")
            continue
        
        # Shuffle clips for random split
        np.random.shuffle(class_clip_list)
        
        # Calculate split points
        n_test = max(1, int(n_class_clips * test_ratio))
        n_val = max(1, int(n_class_clips * val_ratio))
        
        test_clips.extend(class_clip_list[:n_test])
        val_clips.extend(class_clip_list[n_test:n_test+n_val])
        train_clips.extend(class_clip_list[n_test+n_val:])
        
        print(f"   Class {class_id}: {n_class_clips} clips → Train:{len(class_clip_list[n_test+n_val:])}, Val:{n_val}, Test:{n_test}")
    
    # Convert clip assignments to window indices
    def clips_to_indices(clip_list):
        indices = []
        for clip_id in clip_list:
            indices.extend(clip_info[clip_id]['indices'])
        return np.array(indices, dtype=int)
    
    split_indices = {
        'train': clips_to_indices(train_clips),
        'val': clips_to_indices(val_clips),
        'test': clips_to_indices(test_clips)
    }
    
    print(f"\n📊 Distribution:")
    print(f"   Train: {len(train_clips)} clips, {len(split_indices['train'])} windows")
    print(f"   Val:   {len(val_clips)} clips, {len(split_indices['val'])} windows")
    print(f"   Test:  {len(test_clips)} clips, {len(split_indices['test'])} windows")
    
    # Verify no clip overlap
    train_clips_set = set(train_clips)
    val_clips_set = set(val_clips)
    test_clips_set = set(test_clips)
    
    assert len(train_clips_set & val_clips_set) == 0, "Train/Val clip overlap!"
    assert len(train_clips_set & test_clips_set) == 0, "Train/Test clip overlap!"
    assert len(val_clips_set & test_clips_set) == 0, "Val/Test clip overlap!"
    
    print(f"\n✅ Verified: No clip overlap between splits (clip-independent)")
    
    # Show class distribution per split
    for split_name, indices in split_indices.items():
        if len(indices) == 0:
            continue
        labels_split = y_labels[indices]
        dist = np.bincount(labels_split, minlength=config.NUM_CLASSES)
        print(f"   {split_name.capitalize()} class dist: {dist}")
    
    return split_indices


# ==================================================
# VISUALIZATION FUNCTIONS (JDA-STYLE)
# ==================================================

def plot_tsne_features(model, X_features, y_labels, subjects, label_to_id, config, save_path, title="t-SNE Feature Visualization"):
    """
    Create t-SNE visualization of learned features (like JDA paper).
    
    Args:
        model: Trained model
        X_features: (N, input_dim) features
        y_labels: (N,) emotion labels
        subjects: (N,) subject IDs
        label_to_id: Label mapping
        config: Configuration object
        save_path: Path to save visualization
        title: Plot title
    """
    print(f"\n🎨 Creating t-SNE visualization: {title}")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Extract intermediate features (128-dim from feature_extractor)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_features).float().to(device)
        
        # Get features from feature extractor
        features = model.feature_extractor(X_tensor)
        features_np = features.cpu().numpy()
    
    print(f"   Feature shape: {features_np.shape}")
    
    # Sample for t-SNE (max 2000 points for speed)
    if len(features_np) > 2000:
        sample_indices = np.random.choice(len(features_np), 2000, replace=False)
        features_sample = features_np[sample_indices]
        labels_sample = y_labels[sample_indices]
        subjects_sample = subjects[sample_indices]
    else:
        features_sample = features_np
        labels_sample = y_labels
        subjects_sample = subjects
    
    print(f"   Running t-SNE on {len(features_sample)} samples...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=config.SEED, init='pca')
    tsne_result = tsne.fit_transform(features_sample)
    
    # Normalize to [0, 1] for consistent plotting
    x_min, x_max = tsne_result.min(0), tsne_result.max(0)
    tsne_norm = (tsne_result - x_min) / (x_max - x_min)
    
    # Create plots
    id2lab = {v: k for k, v in label_to_id.items()}
    colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Color by emotion class
    ax = axes[0]
    for class_id in range(config.NUM_CLASSES):
        mask = (labels_sample == class_id)
        if mask.any():
            ax.scatter(tsne_norm[mask, 0], tsne_norm[mask, 1],
                      c=[colors[class_id]], label=id2lab[class_id],
                      s=20, alpha=0.6, edgecolors='none')
    
    ax.set_title(f'{title}\n(Colored by Emotion)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by subject (domain)
    ax = axes[1]
    unique_subjects = np.unique(subjects_sample)
    subject_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_subjects)))
    
    for idx, subject_id in enumerate(unique_subjects):
        mask = (subjects_sample == subject_id)
        if mask.any():
            ax.scatter(tsne_norm[mask, 0], tsne_norm[mask, 1],
                      c=[subject_colors[idx]], label=f'Subject {subject_id}',
                      s=20, alpha=0.6, edgecolors='none')
    
    ax.set_title(f'{title}\n(Colored by Subject/Domain)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_training_curves(history, save_path, title="Training Curves"):
    """
    Plot training loss and validation metrics over epochs.
    
    Args:
        history: Dict with 'train_loss', 'val_acc', 'val_f1' lists
        save_path: Path to save plot
        title: Plot title
    """
    print(f"\n📈 Creating training curves: {title}")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax = axes[1]
    ax.plot(epochs, history['val_acc'], 'g-', linewidth=2, label='Validation Accuracy')
    ax.axhline(y=max(history['val_acc']), color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Best: {max(history["val_acc"]):.3f}')
    ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: Validation F1 Score
    ax = axes[2]
    ax.plot(epochs, history['val_f1'], 'orange', linewidth=2, label='Validation F1-Score')
    ax.axhline(y=max(history['val_f1']), color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Best: {max(history["val_f1"]):.3f}')
    ax.set_title('Validation F1-Score (Macro)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_per_class_performance(y_true, y_pred, label_to_id, save_path, title="Per-Class Performance"):
    """
    Create detailed per-class performance visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_to_id: Label mapping
        save_path: Path to save plot
        title: Plot title
    """
    print(f"\n📊 Creating per-class performance plot: {title}")
    
    from sklearn.metrics import precision_recall_fscore_support
    
    id2lab = {v: k for k, v in label_to_id.items()}
    class_names = [id2lab[i] for i in range(len(label_to_id))]
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Create bar plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_pos = np.arange(len(class_names))
    width = 0.6
    
    # Plot 1: Precision
    ax = axes[0, 0]
    bars = ax.bar(x_pos, precision, width, color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.set_title('Precision per Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=precision.mean(), color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {precision.mean():.3f}')
    ax.legend()
    for i, (bar, val) in enumerate(zip(bars, precision)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Recall
    ax = axes[0, 1]
    bars = ax.bar(x_pos, recall, width, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    ax.set_title('Recall per Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=recall.mean(), color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {recall.mean():.3f}')
    ax.legend()
    for i, (bar, val) in enumerate(zip(bars, recall)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: F1-Score
    ax = axes[1, 0]
    bars = ax.bar(x_pos, f1, width, color='lightcoral', edgecolor='darkred', linewidth=1.5)
    ax.set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=f1.mean(), color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {f1.mean():.3f}')
    ax.legend()
    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Support (sample count)
    ax = axes[1, 1]
    bars = ax.bar(x_pos, support, width, color='plum', edgecolor='purple', linewidth=1.5)
    ax.set_title('Support (Sample Count) per Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, support)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def visualize_loso_results(results, save_path, title="LOSO Cross-Validation Results"):
    """
    Visualize LOSO (Leave-One-Subject-Out) results across all test subjects.
    
    Args:
        results: List of tuples (test_subject, accuracy, f1_score)
        save_path: Path to save plot
        title: Plot title
    """
    print(f"\n📊 Creating LOSO results visualization: {title}")
    
    subjects = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    f1_scores = [r[2] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(subjects))
    width = 0.35
    
    # Plot 1: Accuracy per test subject
    ax = axes[0]
    bars = ax.bar(x_pos, accuracies, width, color='skyblue', edgecolor='navy', linewidth=1.5, label='Accuracy')
    ax.axhline(y=np.mean(accuracies), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax.set_title('Accuracy per Test Subject (LOSO)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Test Subject', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in subjects], rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: F1-Score per test subject
    ax = axes[1]
    bars = ax.bar(x_pos, f1_scores, width, color='lightcoral', edgecolor='darkred', linewidth=1.5, label='F1-Score')
    ax.axhline(y=np.mean(f1_scores), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
    ax.set_title('F1-Score per Test Subject (LOSO)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_xlabel('Test Subject', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in subjects], rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


# ==================================================
# TRAINING FUNCTION
# ==================================================

def train_group_model(X_features, y_labels, split_indices, label_to_id, config, group_name):
    """
    Train group-specific model.
    
    Args:
        X_features: (N, input_dim) feature array
        y_labels: (N,) label array
        split_indices: Dict with train/val/test indices
        label_to_id: Label mapping
        config: Configuration object
        group_name: Group identifier (e.g., "subjects_57_58_63_64_48_60")
    
    Returns:
        model: Trained model
        history: Training history
        test_results: (test_acc, test_f1)
    """
    print("\n" + "="*80)
    print(f"TRAINING GROUP MODEL: {group_name}")
    print("="*80)
    
    # Split data
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    Xtr, Xva, Xte = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    ytr, yva, yte = y_labels[train_idx], y_labels[val_idx], y_labels[test_idx]
    
    # Standardization (like JDA: MinMaxScaler to [-1, 1])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)
    
    print(f"   Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
    sample_weights = class_weights[ytr]
    
    # Data loaders
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    va_ds = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())
    te_ds = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long())
    
    batch_size = 96  # Match JDA paper
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    
    # Model
    input_dim = Xtr.shape[1]
    model = GroupSpecificModel(
        input_dim=input_dim,
        n_classes=config.NUM_CLASSES,
        dropout=0.3
    ).to(config.DEVICE)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss (match JDA: Adam with 1e-4 learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # L2 regularization (JDA uses 0.001)
    criterion = nn.CrossEntropyLoss()
    l2_lambda = 0.001
    
    # Training loop
    best_f1, best_state = 0.0, None
    patience_counter = 0
    max_patience = 30
    n_epochs = 10000  # JDA uses 10000 iterations
    
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("\n🚀 Starting training (JDA-style)...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(xb)
            
            # Classification loss
            loss = criterion(logits, yb)
            
            # L2 regularization
            l2_reg = torch.tensor(0., device=config.DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = loss + l2_lambda * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(tr_loader)
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = (all_preds == all_targets).mean()
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if epoch % 100 == 0 or epoch < 10:
            print(f"   Epoch {epoch:04d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n   ⏸️  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Test evaluation
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80)
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_acc = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    print(f"\n✅ {group_name} Results:")
    print(f"   Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"   Test Macro-F1: {test_f1:.3f}")
    
    # Classification report
    id2lab = {v: k for k, v in label_to_id.items()}
    print("\n📊 Classification Report:")
    print(classification_report(
        all_targets, all_preds,
        target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
        digits=3, zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[id2lab[i] for i in range(config.NUM_CLASSES)],
                yticklabels=[id2lab[i] for i in range(config.NUM_CLASSES)])
    plt.title(f'Confusion Matrix - {group_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{group_name}.png', dpi=300)
    print(f"\n💾 Saved confusion matrix: confusion_matrix_{group_name}.png")
    plt.close()
    
    # Visualizations
    plot_tsne_features(model, X_features, y_labels, subjects, label_to_id, config, f'tsne_{group_name}.png')
    plot_training_curves(history, f'training_curves_{group_name}.png')
    plot_per_class_performance(all_targets, all_preds, label_to_id, f'per_class_performance_{group_name}.png')
    
    return model, history, (test_acc, test_f1)


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    parser = argparse.ArgumentParser(description='Train group-specific EEG model (JDA-style)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[57, 58, 63, 64, 48, 60],
                       help='List of subject IDs in the group')
    parser.add_argument('--mode', type=str, default='loso',
                       choices=['loso', 'group'],
                       help='Training mode: loso (leave-one-subject-out) or group (all subjects)')
    parser.add_argument('--clip_independent', action='store_true',
                       help='Use clip-independent splits (no data leakage). Only for group mode.')
    parser.add_argument('--test_subject', type=int, default=None,
                       help='Subject to hold out for testing (only for loso mode)')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.FEATURE_EXTRACTION_MODE = 'handcrafted'
    config.validate_config()
    
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    group_name = f"group_{'_'.join(map(str, args.subjects))}"
    
    print("="*80)
    print("GROUP-SPECIFIC EEG TRAINING (JDA-STYLE)")
    print("="*80)
    print(f"Subject group: {args.subjects}")
    print(f"Mode: {args.mode}")
    if args.mode == 'group':
        print(f"Split type: {'Clip-Independent' if args.clip_independent else 'Window-Based'}")
    print(f"Device: {config.DEVICE}")
    print("="*80)
    
    # Load data for subject group
    X_features, y_labels, subjects, clip_ids, label_to_id = load_group_data(args.subjects, config)
    
    present_subjects = [int(s) for s in np.unique(subjects)]
    
    if args.mode == 'loso':
        # LOSO mode: iterate through all subjects in group
        if args.test_subject is not None:
            # Single LOSO run
            test_subjects = [args.test_subject]
        else:
            # All LOSO runs
            test_subjects = present_subjects
        
        results = []
        
        for test_subject in test_subjects:
            print("\n" + "="*80)
            print(f"LOSO RUN: Test Subject = {test_subject}")
            print("="*80)
            
            # Create LOSO splits
            split_indices = create_loso_splits(y_labels, subjects, clip_ids, test_subject, config)
            
            # Train model
            model, history, (test_acc, test_f1) = train_group_model(
                X_features, y_labels, split_indices, label_to_id, config,
                f"{group_name}_loso_test{test_subject}"
            )
            
            results.append((test_subject, test_acc, test_f1))
        
        # Summary
        print("\n" + "="*80)
        print("LOSO SUMMARY")
        print("="*80)
        for test_subj, acc, f1 in results:
            print(f"   Test Subject {test_subj}: Acc={acc:.3f}, F1={f1:.3f}")
        
        avg_acc = np.mean([r[1] for r in results])
        avg_f1 = np.mean([r[2] for r in results])
        print(f"\n   Average Accuracy: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
        print(f"   Average Macro-F1: {avg_f1:.3f}")
        
        # Visualize LOSO results
        visualize_loso_results(results, f'loso_results_{group_name}.png')
    
    else:
        # Group mode: train on all subjects
        if args.clip_independent:
            # Clip-independent splits (no data leakage)
            split_indices = create_group_splits_clip_independent(y_labels, subjects, clip_ids, config)
            group_name += "_clip_independent"
        else:
            # Window-based splits (original method)
            split_indices = create_group_splits(y_labels, subjects, clip_ids, config)
        
        model, history, (test_acc, test_f1) = train_group_model(
            X_features, y_labels, split_indices, label_to_id, config, group_name
        )
        
        # Save model if requested
        if args.save_model:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'subject_group': args.subjects,
                'clip_independent': args.clip_independent if args.mode == 'group' else None,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'label_to_id': label_to_id
            }, args.save_model)
            print(f"\n💾 Model saved to: {args.save_model}")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
