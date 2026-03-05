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
    test_mask = np.zeros(n_samples, dtype=bool)
    
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
        
    else:
        # Group mode: train on all subjects
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
