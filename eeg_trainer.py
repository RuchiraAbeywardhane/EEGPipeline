"""
EEG Model Training Module
=========================

This module contains all training-related functions for the EEG emotion recognition model.

Functions:
- mixup_data: Data augmentation using mixup
- mixup_criterion: Loss calculation for mixup
- train_eeg_model: Complete training loop with validation and testing

Author: Final Year Project
Date: 2026
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import f1_score, classification_report

from eeg_bilstm_model import SimpleBiLSTMClassifier


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_eeg_model(X_features, y_labels, split_indices, label_mapping, config):
    """Train EEG BiLSTM model with optional domain adaptation."""
    
    # Domain adaptation is MORE CRITICAL for subject-independent mode!
    # - Subject-independent: Different people = LARGE domain shift → need DA
    # - Subject-dependent: Same person, different clips = SMALL domain shift → optional DA
    use_domain_adaptation = config.SUBJECT_INDEPENDENT  # ✅ CORRECTED!
    
    if use_domain_adaptation:
        print("\n" + "="*80)
        print("⚠️  SUBJECT-INDEPENDENT MODE: Using Domain Adaptation")
        print("="*80)
        print("Domain adaptation helps generalize across DIFFERENT SUBJECTS")
        print("by treating each subject as a different domain.")
        print("This addresses the domain shift caused by inter-subject variability.")
        print("="*80)
        return train_eeg_model_with_domain_adaptation(
            X_features, y_labels, split_indices, label_mapping, config
        )
    else:
        print("\n" + "="*80)
        print("✅ SUBJECT-DEPENDENT MODE: Standard Training")
        print("="*80)
        print("Training and testing on same subject's data.")
        print("Minimal domain shift - standard training is sufficient.")
        print("="*80)
        return train_eeg_model_standard(
            X_features, y_labels, split_indices, label_mapping, config
        )


def train_eeg_model_standard(X_features, y_labels, split_indices, label_mapping, config):
    """Standard training without domain adaptation (for subject-independent mode)."""
    print("Mixup Augmentation: {'ENABLED' if config.USE_MIXUP else 'DISABLED'}")
    if config.USE_MIXUP:
        print(f"Mixup Alpha: {config.MIXUP_ALPHA}")
    print("="*80)
    
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
    
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    
    # Balanced sampling
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    # Data loaders
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    # Model WITHOUT domain adaptation
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4,
        use_domain_adaptation=False  # Standard mode
    ).to(config.DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_f1, best_state, wait = 0.0, None, 0
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            # Apply Mixup augmentation based on config flag
            if config.USE_MIXUP and np.random.rand() < 0.5:
                xb_mix, ya, yb_m, lam = mixup_data(xb, yb, alpha=config.MIXUP_ALPHA)
                optimizer.zero_grad()
                logits = model(xb_mix)
                loss = mixup_criterion(criterion, logits, ya, yb_m, lam)
            else:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
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
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(model.state_dict(), config.EEG_CHECKPOINT)
        else:
            wait += 1
            if wait >= config.EEG_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Test evaluation
    if best_state:
        model.load_state_dict(best_state)
    
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
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print("\n" + "="*80)
    print("EEG TEST RESULTS")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, mu, sd


def train_eeg_model_with_domain_adaptation(X_features, y_labels, split_indices, label_mapping, config):
    """Training with domain adaptation for subject-dependent mode (clip-to-clip transfer)."""
    print("🔬 Domain Adaptation Strategy: Adversarial Training")
    print("Treating train clips and test clips as different domains")
    print("="*80)
    
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
    
    print(f"Train (source): {Xtr.shape}, Val (target): {Xva.shape}, Test (target): {Xte.shape}")
    
    # Balanced sampling for source (train)
    class_counts = np.bincount(ytr, minlength=config.NUM_CLASSES).astype(np.float32)
    class_sample_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_sample_weights[ytr]
    sample_weights_tensor = torch.from_numpy(sample_weights.astype(np.float32))
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    # Data loaders
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    
    tr_loader = DataLoader(tr_ds, batch_size=config.EEG_BATCH_SIZE, sampler=train_sampler)
    va_loader = DataLoader(va_ds, batch_size=config.EEG_BATCH_SIZE, shuffle=True)  # Shuffle target
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    # Model WITH domain adaptation
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4,
        use_domain_adaptation=True  # Enable domain adaptation
    ).to(config.DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()
    
    # Training loop with domain adaptation
    best_f1, best_state, wait = 0.0, None, 0
    
    # Create target data iterator
    va_iter = iter(va_loader)
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        train_loss, class_loss_sum, domain_loss_sum = 0.0, 0.0, 0.0
        n_batches = 0
        
        # Dynamic lambda for gradient reversal (from DANN paper)
        p = float(epoch) / config.EEG_EPOCHS
        lambda_da = 2. / (1. + np.exp(-10. * p)) - 1
        
        for xb_source, yb_source in tr_loader:
            # Get target batch
            try:
                xb_target, yb_target = next(va_iter)
            except StopIteration:
                va_iter = iter(va_loader)
                xb_target, yb_target = next(va_iter)
            
            # Ensure same batch size
            min_size = min(len(xb_source), len(xb_target))
            xb_source = xb_source[:min_size].to(config.DEVICE)
            yb_source = yb_source[:min_size].to(config.DEVICE)
            xb_target = xb_target[:min_size].to(config.DEVICE)
            
            # Combine source and target
            xb_combined = torch.cat([xb_source, xb_target], dim=0)
            
            # Domain labels: 0 for source (train), 1 for target (val/test)
            domain_labels = torch.cat([
                torch.zeros(min_size, dtype=torch.long),
                torch.ones(min_size, dtype=torch.long)
            ]).to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with domain adaptation
            class_logits, features, domain_logits = model(xb_combined, lambda_=lambda_da, return_features=True)
            
            # Classification loss (only on source data)
            class_loss = criterion_class(class_logits[:min_size], yb_source)
            
            # Domain adversarial loss (on both source and target)
            domain_loss = criterion_domain(domain_logits, domain_labels)
            
            # Combined loss
            total_loss = class_loss + 0.5 * domain_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            class_loss_sum += class_loss.item()
            domain_loss_sum += domain_loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        class_loss_avg = class_loss_sum / n_batches
        domain_loss_avg = domain_loss_sum / n_batches
        
        # Validation (on target domain)
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                logits = model(xb, return_features=False)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = (all_preds == all_targets).mean()
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} (Class: {class_loss_avg:.4f}, Domain: {domain_loss_avg:.4f}) | "
                  f"λ: {lambda_da:.3f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(model.state_dict(), config.EEG_CHECKPOINT)
        else:
            wait += 1
            if wait >= config.EEG_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Test evaluation
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            logits = model(xb, return_features=False)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_acc = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    print("\n" + "="*80)
    print("EEG TEST RESULTS (with Domain Adaptation)")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, mu, sd
