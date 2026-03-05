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

from eeg_bilstm_model import SimpleBiLSTMClassifier, compute_associative_loss


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
    
    # Determine domain adaptation mode
    use_domain_adaptation = config.SUBJECT_INDEPENDENT
    adaptation_mode = config.ADAPTATION_MODE if use_domain_adaptation else 'A'
    
    mode_descriptions = {
        'A': 'No Adaptation (Baseline)',
        'B': 'Associative Adaptation (Walker + Visit Loss)',
        'C': 'Adversarial Adaptation (DANN)',
        'D': 'Combined Adaptation (Associative + Adversarial)'
    }
    
    if use_domain_adaptation:
        print("\n" + "="*80)
        print(f"⚠️  SUBJECT-INDEPENDENT MODE: Using Domain Adaptation Mode {adaptation_mode}")
        print("="*80)
        print(f"Strategy: {mode_descriptions[adaptation_mode]}")
        print("Domain adaptation helps generalize across DIFFERENT SUBJECTS")
        print("by treating each subject as a different domain.")
        print("="*80)
        return train_eeg_model_with_domain_adaptation(
            X_features, y_labels, split_indices, label_mapping, config, adaptation_mode
        )
    else:
        print("\n" + "="*80)
        print("✅ SUBJECT-DEPENDENT MODE: Standard Training (Mode A)")
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


def train_eeg_model_with_domain_adaptation(X_features, y_labels, split_indices, label_mapping, config, adaptation_mode):
    """
    Training with domain adaptation for subject-independent mode.
    
    Supports 4 modes:
    - Mode A: No adaptation (baseline)
    - Mode B: Associative (walker + visit loss)
    - Mode C: Adversarial (DANN)
    - Mode D: Combined (associative + adversarial) - BEST
    """
    mode_desc = {
        'A': 'No Adaptation (Baseline)',
        'B': 'Associative (Walker + Visit Loss)',
        'C': 'Adversarial (DANN)',
        'D': 'Combined (Associative + Adversarial)'
    }
    print(f"🔬 Domain Adaptation Mode: {adaptation_mode} - {mode_desc[adaptation_mode]}")
    print("Treating train subjects and test subjects as different domains")
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
    
    print(f"Source (train): {Xtr.shape}, Target (val): {Xva.shape}, Target (test): {Xte.shape}")
    
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
    va_loader = DataLoader(va_ds, batch_size=config.EEG_BATCH_SIZE, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    
    # Model with domain adaptation
    use_da = (adaptation_mode != 'A')
    model = SimpleBiLSTMClassifier(
        dx=26, n_channels=4, hidden=256, layers=3,
        n_classes=config.NUM_CLASSES, p_drop=0.4,
        use_domain_adaptation=use_da,
        adaptation_mode=adaptation_mode
    ).to(config.DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.EEG_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    class_weights = torch.from_numpy(class_sample_weights).float().to(config.DEVICE)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss() if adaptation_mode in ['C', 'D'] else None
    
    # Training loop
    best_f1, best_state, wait = 0.0, None, 0
    va_iter = iter(va_loader)
    
    for epoch in range(1, config.EEG_EPOCHS + 1):
        model.train()
        total_loss_sum = 0.0
        class_loss_sum = 0.0
        assoc_loss_sum = 0.0
        walker_loss_sum = 0.0
        visit_loss_sum = 0.0
        domain_loss_sum = 0.0
        n_batches = 0
        
        # Dynamic lambda for gradient reversal
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
            
            optimizer.zero_grad()
            
            # MODE A: No adaptation (baseline)
            if adaptation_mode == 'A':
                logits = model(xb_source)
                class_loss = criterion_class(logits, yb_source)
                total_loss = class_loss
            
            # MODE B: Associative adaptation only
            elif adaptation_mode == 'B':
                xb_combined = torch.cat([xb_source, xb_target], dim=0)
                logits, features = model(xb_combined, return_features=True)
                
                # Classification loss (only on source)
                class_loss = criterion_class(logits[:min_size], yb_source)
                
                # Associative loss (walker + visit)
                source_feats = features[:min_size]
                target_feats = features[min_size:]
                assoc_loss, walker_loss, visit_loss = compute_associative_loss(
                    source_feats, target_feats, yb_source,
                    walker_weight=config.WALKER_WEIGHT,
                    visit_weight=config.VISIT_WEIGHT,
                    temperature=config.TEMPERATURE
                )
                
                total_loss = class_loss + assoc_loss
                assoc_loss_sum += assoc_loss.item()
                walker_loss_sum += walker_loss.item()
                visit_loss_sum += visit_loss.item()
            
            # MODE C: Adversarial adaptation only
            elif adaptation_mode == 'C':
                xb_combined = torch.cat([xb_source, xb_target], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(min_size, dtype=torch.long),
                    torch.ones(min_size, dtype=torch.long)
                ]).to(config.DEVICE)
                
                logits, features, domain_logits = model(xb_combined, lambda_=lambda_da, return_features=True)
                
                # Classification loss (only on source)
                class_loss = criterion_class(logits[:min_size], yb_source)
                
                # Domain adversarial loss
                domain_loss = criterion_domain(domain_logits, domain_labels)
                
                total_loss = class_loss + 0.5 * domain_loss
                domain_loss_sum += domain_loss.item()
            
            # MODE D: Combined (associative + adversarial)
            else:  # adaptation_mode == 'D'
                xb_combined = torch.cat([xb_source, xb_target], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(min_size, dtype=torch.long),
                    torch.ones(min_size, dtype=torch.long)
                ]).to(config.DEVICE)
                
                logits, features, domain_logits = model(xb_combined, lambda_=lambda_da, return_features=True)
                
                # Classification loss (only on source)
                class_loss = criterion_class(logits[:min_size], yb_source)
                
                # Associative loss
                source_feats = features[:min_size]
                target_feats = features[min_size:]
                assoc_loss, walker_loss, visit_loss = compute_associative_loss(
                    source_feats, target_feats, yb_source,
                    walker_weight=config.WALKER_WEIGHT,
                    visit_weight=config.VISIT_WEIGHT,
                    temperature=config.TEMPERATURE
                )
                
                # Domain adversarial loss
                domain_loss = criterion_domain(domain_logits, domain_labels)
                
                total_loss = class_loss + assoc_loss + 0.5 * domain_loss
                assoc_loss_sum += assoc_loss.item()
                walker_loss_sum += walker_loss.item()
                visit_loss_sum += visit_loss.item()
                domain_loss_sum += domain_loss.item()
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss_sum += total_loss.item()
            class_loss_sum += class_loss.item()
            n_batches += 1
        
        # Average losses
        total_loss_avg = total_loss_sum / n_batches
        class_loss_avg = class_loss_sum / n_batches
        
        # Validation
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
        
        # Print progress
        if epoch % 5 == 0 or epoch < 10:
            loss_str = f"Epoch {epoch:03d} | Total: {total_loss_avg:.4f} (Class: {class_loss_avg:.4f}"
            
            if adaptation_mode in ['B', 'D']:
                walker_avg = walker_loss_sum / n_batches
                visit_avg = visit_loss_sum / n_batches
                assoc_avg = assoc_loss_sum / n_batches
                loss_str += f", Assoc: {assoc_avg:.4f} [W:{walker_avg:.4f}, V:{visit_avg:.4f}]"
            
            if adaptation_mode in ['C', 'D']:
                domain_avg = domain_loss_sum / n_batches
                loss_str += f", Domain: {domain_avg:.4f}, λ: {lambda_da:.3f}"
            
            loss_str += f") | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}"
            print(loss_str)
        
        # Early stopping
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
    print(f"EEG TEST RESULTS (Mode {adaptation_mode}: {mode_desc[adaptation_mode]})")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"Test Macro-F1: {test_f1:.3f}")
    id2lab = {v: k for k, v in label_mapping.items()}
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[id2lab[i] for i in range(config.NUM_CLASSES)],
                                digits=3, zero_division=0))
    
    return model, mu, sd
