"""
EEG BiLSTM Model Architecture
==============================

BiLSTM-based classifier with attention mechanism for EEG emotion recognition.
"""

import torch
import torch.nn as nn


class SimpleBiLSTMClassifier(nn.Module):
    """3-layer BiLSTM with attention for EEG."""
    
    def __init__(self, dx=26, n_channels=4, hidden=256, layers=3, n_classes=4, p_drop=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.hidden = hidden
        
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B, C, dx = x.shape
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        h = self.drop(self.norm(h))

        scores = self.attn(h)
        alpha = torch.softmax(scores, dim=1)
        h_pooled = (alpha * h).sum(dim=1)

        return self.classifier(h_pooled)