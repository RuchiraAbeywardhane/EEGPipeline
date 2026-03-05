"""
EEG BiLSTM Model Architecture
==============================

BiLSTM-based classifier with attention mechanism for EEG emotion recognition.
Supports multiple domain adaptation strategies:
- Mode A: No adaptation (baseline)
- Mode B: Associative adaptation (walker + visit loss)
- Mode C: Adversarial adaptation (DANN with gradient reversal)
- Mode D: Combined (associative + adversarial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


class SimpleBiLSTMClassifier(nn.Module):
    """3-layer BiLSTM with attention for EEG emotion recognition."""
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4, 
                 use_domain_adaptation=False, adaptation_mode='C', use_deep_features=False):
        """
        Args:
            dx: Number of features per channel (26 for handcrafted, ignored for deep)
            n_channels: Number of EEG channels (ignored for deep features)
            hidden: Hidden size for LSTM
            layers: Number of LSTM layers
            n_classes: Number of emotion classes
            p_drop: Dropout probability
            use_domain_adaptation: Whether to use domain adaptation
            adaptation_mode: 'A' (none), 'B' (associative), 'C' (adversarial), 'D' (combined)
            use_deep_features: If True, input is (B, feature_dim) instead of (B, C, dx)
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        self.use_domain_adaptation = use_domain_adaptation
        self.adaptation_mode = adaptation_mode
        self.use_deep_features = use_deep_features
        
        if not use_deep_features:
            # Standard handcrafted features: (B, C, dx)
            self.input_proj = nn.Sequential(
                nn.Linear(dx, hidden),
                nn.BatchNorm1d(n_channels),
                nn.ReLU(),
                nn.Dropout(p_drop * 0.5)
            )
            
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=p_drop if layers > 1 else 0
            )
            
            d_lstm = 2 * hidden
            self.norm = nn.LayerNorm(d_lstm)
            self.drop = nn.Dropout(p_drop)

            self.attn = nn.Sequential(
                nn.Linear(d_lstm, d_lstm // 2),
                nn.Tanh(),
                nn.Linear(d_lstm // 2, 1)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(d_lstm, d_lstm),
                nn.BatchNorm1d(d_lstm),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_lstm, hidden),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, n_classes)
            )
        else:
            # Deep learning features: (B, feature_dim) - direct classification
            d_lstm = hidden  # Use hidden as feature dimension
            self.classifier = nn.Sequential(
                nn.Linear(d_lstm, d_lstm),
                nn.BatchNorm1d(d_lstm),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_lstm, hidden),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, n_classes)
            )
        
        # Domain adaptation components
        if use_domain_adaptation:
            d_lstm = 2 * hidden if not use_deep_features else hidden
            if adaptation_mode in ['C', 'D']:
                # Adversarial: gradient reversal + domain discriminator
                self.grl = GradientReversalLayer()
                self.domain_classifier = nn.Sequential(
                    nn.Linear(d_lstm, d_lstm // 2),
                    nn.BatchNorm1d(d_lstm // 2),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                    nn.Linear(d_lstm // 2, 2)  # Binary: source vs target
                )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, lambda_=1.0, return_features=False):
        if not self.use_deep_features:
            # Handcrafted features path: (B, C, dx)
            B, C, dx = x.shape
            x = self.input_proj(x)
            h, _ = self.lstm(x)
            h = self.drop(self.norm(h))

            scores = self.attn(h)
            alpha = torch.softmax(scores, dim=1)
            h_pooled = (alpha * h).sum(dim=1)  # Feature representation
        else:
            # Deep features path: (B, feature_dim) - already extracted
            h_pooled = x

        logits = self.classifier(h_pooled)
        
        # For domain adaptation training
        if return_features:
            outputs = [logits, h_pooled]
            
            if self.use_domain_adaptation and self.adaptation_mode in ['C', 'D']:
                # Apply gradient reversal for adversarial training
                reversed_features = self.grl(h_pooled, lambda_)
                domain_logits = self.domain_classifier(reversed_features)
                outputs.append(domain_logits)
            
            return tuple(outputs)
        
        return logits


# ==================================================
# ASSOCIATIVE DOMAIN ADAPTATION LOSS FUNCTIONS
# ==================================================

def compute_walker_loss(source_features, target_features, source_labels, temperature=1.0, eps=1e-8):
    """
    Compute walker loss for associative domain adaptation.
    
    Walker loss enforces that:
    - Walking from source → target → source should return to samples of same class
    - This creates semantic alignment between domains
    
    Args:
        source_features: (N, D) - Source domain features
        target_features: (M, D) - Target domain features
        source_labels: (N,) - Source labels (class indices)
        temperature: Temperature for softmax (controls sharpness)
        eps: Small constant for numerical stability
        
    Returns:
        walker_loss: Scalar loss value
    """
    # Create equality matrix (which source samples have same label)
    y_sparse = source_labels
    equality_matrix = (y_sparse.unsqueeze(1) == y_sparse.unsqueeze(0)).float()
    p_target = equality_matrix / (equality_matrix.sum(dim=1, keepdim=True) + eps)
    
    # Compute similarity: source → target
    match_ab = torch.matmul(source_features, target_features.t()) / temperature
    p_ab = F.softmax(match_ab, dim=1)  # (N, M)
    
    # Compute similarity: target → source
    p_ba = F.softmax(match_ab.t(), dim=1)  # (M, N)
    
    # Round-trip probability: source → target → source
    p_aba = torch.matmul(p_ab, p_ba)  # (N, N)
    
    # Walker loss: KL divergence between target and actual round-trip distribution
    walker_loss = F.kl_div(
        torch.log(p_aba + eps),
        p_target,
        reduction='batchmean'
    )
    
    return walker_loss


def compute_visit_loss(source_features, target_features, temperature=1.0, eps=1e-8):
    """
    Compute visit loss for associative domain adaptation.
    
    Visit loss encourages:
    - Uniform visitation of target samples when walking from source
    - Prevents collapsing to a few target samples
    
    Args:
        source_features: (N, D) - Source domain features
        target_features: (M, D) - Target domain features
        temperature: Temperature for softmax
        eps: Small constant for numerical stability
        
    Returns:
        visit_loss: Scalar loss value
    """
    # Compute similarity: source → target
    match_ab = torch.matmul(source_features, target_features.t()) / temperature
    p_ab = F.softmax(match_ab, dim=1)  # (N, M)
    
    # Average visit probability for each target sample
    visit_prob = p_ab.mean(dim=0, keepdim=True)  # (1, M)
    
    # Target: uniform distribution
    M = target_features.size(0)
    uniform_prob = torch.ones_like(visit_prob) / M
    
    # Visit loss: KL divergence from uniform
    visit_loss = F.kl_div(
        torch.log(visit_prob + eps),
        uniform_prob,
        reduction='batchmean'
    )
    
    return visit_loss


def compute_associative_loss(source_features, target_features, source_labels, 
                             walker_weight=1.0, visit_weight=0.6, temperature=1.0):
    """
    Compute combined associative domain adaptation loss.
    
    Args:
        source_features: (N, D) - Source domain features
        target_features: (M, D) - Target domain features
        source_labels: (N,) - Source labels
        walker_weight: Weight for walker loss
        visit_weight: Weight for visit loss
        temperature: Temperature for softmax
        
    Returns:
        total_loss: Combined loss
        walker_loss: Walker loss component
        visit_loss: Visit loss component
    """
    # Normalize features for better similarity computation
    source_features = F.normalize(source_features, p=2, dim=1)
    target_features = F.normalize(target_features, p=2, dim=1)
    
    walker_loss = compute_walker_loss(source_features, target_features, source_labels, temperature)
    visit_loss = compute_visit_loss(source_features, target_features, temperature)
    
    total_loss = walker_weight * walker_loss + visit_weight * visit_loss
    
    return total_loss, walker_loss, visit_loss