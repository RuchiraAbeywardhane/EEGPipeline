"""
EEG BiLSTM Model Architecture
==============================

BiLSTM-based classifier with attention mechanism for EEG emotion recognition.
"""

import torch
import torch.nn as nn
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
    """3-layer BiLSTM with attention for EEG."""
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4, 
                 use_domain_adaptation=False):
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        self.use_domain_adaptation = use_domain_adaptation
        
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
        
        # Domain adaptation components (for subject-dependent mode)
        if use_domain_adaptation:
            self.grl = GradientReversalLayer()
            
            # Domain discriminator (binary: source vs target)
            self.domain_classifier = nn.Sequential(
                nn.Linear(d_lstm, d_lstm // 2),
                nn.BatchNorm1d(d_lstm // 2),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_lstm // 2, 2)  # Binary: train clips vs test clips
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, lambda_=1.0, return_features=False):
        B, C, dx = x.shape
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))

        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)  # Feature representation

        logits = self.classifier(h_pooled)
        
        # For domain adaptation training
        if return_features:
            if self.use_domain_adaptation:
                # Apply gradient reversal for adversarial training
                reversed_features = self.grl(h_pooled, lambda_)
                domain_logits = self.domain_classifier(reversed_features)
                return logits, h_pooled, domain_logits
            else:
                return logits, h_pooled
        
        return logits