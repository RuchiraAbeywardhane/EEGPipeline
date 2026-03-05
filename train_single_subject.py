"""
Single-Subject EEG Training Script
===================================

This script trains a subject-dependent model using data from ONLY ONE SUBJECT
with NON-OVERLAPPING windows to prevent data leakage.

Key Features:
- Uses data from a single subject only
- Non-overlapping windows (0% overlap)
- Simpler model (no domain adaptation needed)
- Proper train/val/test split by recordings

Usage:
    python train_single_subject.py --subject P001
    python train_single_subject.py --subject 39 --feature_mode handcrafted

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
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data, extract_eeg_features
from eeg_deep_feature_extractor import create_feature_extractor


# ==================================================
# SINGLE-SUBJECT MODEL (SIMPLER ARCHITECTURE)
# ==================================================

class SingleSubjectModel(nn.Module):
    """
    Simpler model for single-subject training.
    No domain adaptation needed since all data is from same subject.
    """
    
    def __init__(self, input_dim=104, hidden=128, n_classes=4, dropout=0.4):
        """
        Args:
            input_dim: Input feature dimension (26*4=104 for handcrafted)
            hidden: Hidden layer size
            n_classes: Number of emotion classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - Flattened features
        
        Returns:
            logits: (B, n_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


# ==================================================
# DATA LOADING FOR SINGLE SUBJECT
# ==================================================

def load_single_subject_data(subject_id, config):
    """
    Load data for a single subject with NON-OVERLAPPING windows.
    
    Args:
        subject_id: Subject ID (e.g., "P001" or "39")
        config: Configuration object
    
    Returns:
        X_features: (N, 104) - Flattened features
        y_labels: (N,) - Emotion labels
        clip_ids: (N,) - Recording IDs
        label_to_id: Label mapping dictionary
    """
    print("\n" + "="*80)
    print(f"LOADING DATA FOR SUBJECT: {subject_id}")
    print("="*80)
    
    # Temporarily set overlap to 0 for non-overlapping windows
    original_overlap = config.EEG_OVERLAP
    config.EEG_OVERLAP = 0.0  # NON-OVERLAPPING
    
    # Load all data
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(config.DATA_ROOT, config)
    
    # Restore original overlap setting
    config.EEG_OVERLAP = original_overlap
    
    # Filter for single subject
    subject_mask = (eeg_subjects == str(subject_id))
    
    if not subject_mask.any():
        available_subjects = np.unique(eeg_subjects)
        raise ValueError(
            f"Subject '{subject_id}' not found in dataset!\n"
            f"Available subjects: {available_subjects}"
        )
    
    X_raw_subject = eeg_X_raw[subject_mask]
    y_subject = eeg_y[subject_mask]
    clip_ids_subject = eeg_clip_ids[subject_mask]
    
    print(f"\n✅ Subject {subject_id} data:")
    print(f"   Total windows: {len(y_subject)} (NON-OVERLAPPING)")
    print(f"   Unique recordings: {len(np.unique(clip_ids_subject))}")
    print(f"   Emotion distribution: {np.bincount(y_subject, minlength=config.NUM_CLASSES)}")
    
    # Extract features
    print("\n🔧 Extracting features...")
    X_features = extract_eeg_features(X_raw_subject, config)
    
    # Flatten features: (N, C, F) -> (N, C*F)
    N, C, F = X_features.shape
    X_features_flat = X_features.reshape(N, C * F)
    
    print(f"   Feature shape: {X_features_flat.shape}")
    
    return X_features_flat, y_subject, clip_ids_subject, label_to_id


def create_single_subject_splits(y_labels, clip_ids, config, train_ratio=0.70, val_ratio=0.15):
    """
    Split data by recordings (NOT windows) to prevent leakage.
    
    Args:
        y_labels: (N,) emotion labels
        clip_ids: (N,) recording IDs
        config: Configuration object
        train_ratio: Train split ratio
        val_ratio: Validation split ratio
    
    Returns:
        split_indices: Dict with 'train', 'val', 'test' indices
    """
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL/TEST SPLIT (BY RECORDINGS)")
    print("="*80)
    
    unique_clips = np.unique(clip_ids)
    n_clips = len(unique_clips)
    
    print(f"   Total recordings: {n_clips}")
    
    # Get label for each clip (majority vote)
    clip_labels = {}
    for clip_id in unique_clips:
        clip_mask = (clip_ids == clip_id)
        clip_label = np.bincount(y_labels[clip_mask]).argmax()
        clip_labels[clip_id] = clip_label
    
    # Stratified split by class
    from collections import defaultdict
    clips_by_class = defaultdict(list)
    for clip_id in unique_clips:
        clips_by_class[clip_labels[clip_id]].append(clip_id)
    
    train_clips, val_clips, test_clips = [], [], []
    
    for class_id in range(config.NUM_CLASSES):
        class_clips = np.array(clips_by_class[class_id])
        n_class = len(class_clips)
        
        if n_class == 0:
            print(f"   ⚠️  No recordings for class {class_id}")
            continue
        
        np.random.shuffle(class_clips)
        
        n_test = max(1, int(n_class * (1 - train_ratio - val_ratio)))
        n_val = max(1, int(n_class * val_ratio))
        
        test_clips.extend(class_clips[:n_test])
        val_clips.extend(class_clips[n_test:n_test+n_val])
        train_clips.extend(class_clips[n_test+n_val:])
        
        print(f"   Class {class_id}: {n_class} recordings → Train:{len(class_clips[n_test+n_val:])}, Val:{n_val}, Test:{n_test}")
    
    # Convert to window indices
    train_mask = np.isin(clip_ids, train_clips)
    val_mask = np.isin(clip_ids, val_clips)
    test_mask = np.isin(clip_ids, test_clips)
    
    split_indices = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
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


# ==================================================
# TRAINING FUNCTION
# ==================================================

def train_single_subject_model(X_features, y_labels, split_indices, label_to_id, config, subject_id):
    """
    Train single-subject model.
    
    Args:
        X_features: (N, input_dim) feature array
        y_labels: (N,) label array
        split_indices: Dict with train/val/test indices
        label_to_id: Label mapping
        config: Configuration object
        subject_id: Subject identifier
    
    Returns:
        model: Trained model
        history: Training history
    """
    print("\n" + "="*80)
    print(f"TRAINING MODEL FOR SUBJECT {subject_id}")
    print("="*80)
    
    # Split data
    train_idx = split_indices['train']
    val_idx = split_indices['val']
    test_idx = split_indices['test']
    
    Xtr, Xva, Xte = X_features[train_idx], X_features[val_idx], X_features[test_idx]
    ytr, yva, yte = y_labels[train_idx], y_labels[val_idx], y_labels[test_idx]
    
    # Standardization
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd
    
    print(f"   Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    
    # Data loaders
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    va_ds = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())
    te_ds = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long())
    
    tr_loader = DataLoader(tr_ds, batch_size=16, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)
    
    # Model
    input_dim = Xtr.shape[1]
    model = SingleSubjectModel(
        input_dim=input_dim,
        hidden=128,
        n_classes=config.NUM_CLASSES,
        dropout=0.4
    ).to(config.DEVICE)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    class_weights_tensor = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training loop
    best_f1, best_state = 0.0, None
    patience_counter = 0
    max_patience = 50
    n_epochs = 300
    
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("\n🚀 Starting training...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        
        scheduler.step(val_f1)
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if epoch % 10 == 0 or epoch < 10:
            print(f"   Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
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
    
    print(f"\n✅ Subject {subject_id} Results:")
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
    plt.title(f'Confusion Matrix - Subject {subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_subject_{subject_id}.png', dpi=300)
    print(f"\n💾 Saved confusion matrix: confusion_matrix_subject_{subject_id}.png")
    plt.close()
    
    return model, history, (test_acc, test_f1)


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    parser = argparse.ArgumentParser(description='Train single-subject EEG emotion recognition model')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., P001 or 39)')
    parser.add_argument('--feature_mode', type=str, default='handcrafted',
                       choices=['handcrafted', 'deep_cnn', 'deep_transformer'],
                       help='Feature extraction mode')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.FEATURE_EXTRACTION_MODE = args.feature_mode
    config.validate_config()
    
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    print("="*80)
    print("SINGLE-SUBJECT EEG TRAINING")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Device: {config.DEVICE}")
    print(f"Window overlap: 0% (NON-OVERLAPPING)")
    print("="*80)
    
    # Load data for single subject
    X_features, y_labels, clip_ids, label_to_id = load_single_subject_data(args.subject, config)
    
    # Create splits by recordings
    split_indices = create_single_subject_splits(y_labels, clip_ids, config)
    
    # Train model
    model, history, (test_acc, test_f1) = train_single_subject_model(
        X_features, y_labels, split_indices, label_to_id, config, args.subject
    )
    
    # Save model if requested
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'subject_id': args.subject,
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
