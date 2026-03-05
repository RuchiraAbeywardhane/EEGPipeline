"""
Deep Learning Feature Extractors for EEG
=========================================

This module provides CNN and Transformer-based feature extractors
as alternatives to handcrafted features.

Author: Final Year Project
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================================================
# CNN-BASED FEATURE EXTRACTOR
# ==================================================

class TemporalConvBlock(nn.Module):
    """Temporal convolutional block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for raw EEG signals.
    
    Architecture:
    - Spatial convolution across channels
    - Multiple temporal convolution blocks with residual connections
    - Global average pooling
    - Feature projection
    
    Args:
        n_channels: Number of EEG channels (default: 4)
        filters: List of filter sizes for each conv block
        kernel_size: Kernel size for temporal convolutions
        feature_dim: Output feature dimension
        dropout: Dropout rate
    """
    
    def __init__(self, n_channels=4, filters=[32, 64, 128], 
                 kernel_size=7, feature_dim=128, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.feature_dim = feature_dim
        
        # Spatial convolution: mix information across channels
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(n_channels, filters[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # Temporal convolution blocks with residual connections
        self.temporal_blocks = nn.ModuleList()
        in_channels = filters[0]
        for out_channels in filters:
            stride = 2 if out_channels != in_channels else 1
            self.temporal_blocks.append(
                TemporalConvBlock(in_channels, out_channels, kernel_size, stride, dropout)
            )
            in_channels = out_channels
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_proj = nn.Sequential(
            nn.Linear(filters[-1], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) - Batch of raw EEG signals
        
        Returns:
            features: (B, feature_dim) - Extracted features
        """
        # Transpose to (B, C, T) for Conv1d
        x = x.transpose(1, 2)  # (B, C, T)
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, filters[0], T)
        
        # Temporal convolutions with residual connections
        for block in self.temporal_blocks:
            x = block(x)  # (B, filters[-1], T//stride)
        
        # Global pooling
        x = self.global_pool(x)  # (B, filters[-1], 1)
        x = x.squeeze(-1)  # (B, filters[-1])
        
        # Feature projection
        features = self.feature_proj(x)  # (B, feature_dim)
        
        return features


# ==================================================
# TRANSFORMER-BASED FEATURE EXTRACTOR
# ==================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeatureExtractor(nn.Module):
    """
    Transformer-based feature extractor for raw EEG signals.
    
    Architecture:
    - Linear projection of channels
    - Positional encoding
    - Multi-head self-attention layers
    - Feed-forward networks
    - Global pooling (CLS token or mean pooling)
    
    Args:
        n_channels: Number of EEG channels (default: 4)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        feature_dim: Output feature dimension
        dropout: Dropout rate
    """
    
    def __init__(self, n_channels=4, d_model=128, nhead=4, num_layers=2,
                 feature_dim=128, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Channel embedding: project channels to d_model dimension
        self.channel_embedding = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) - Batch of raw EEG signals
        
        Returns:
            features: (B, feature_dim) - Extracted features
        """
        B, T, C = x.shape
        
        # Channel embedding
        x = self.channel_embedding(x)  # (B, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T+1, d_model)
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # Feature projection
        features = self.feature_proj(cls_output)  # (B, feature_dim)
        
        return features


# ==================================================
# UNIFIED FEATURE EXTRACTOR WRAPPER
# ==================================================

def create_feature_extractor(config):
    """
    Factory function to create feature extractor based on config.
    
    Args:
        config: Configuration object
    
    Returns:
        feature_extractor: Feature extraction module or None for handcrafted
    """
    if config.FEATURE_EXTRACTION_MODE == 'handcrafted':
        return None  # Use handcrafted features in data loader
    
    elif config.FEATURE_EXTRACTION_MODE == 'deep_cnn':
        return CNNFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            filters=config.CNN_FILTERS,
            kernel_size=config.CNN_KERNEL_SIZE,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
    
    elif config.FEATURE_EXTRACTION_MODE == 'deep_transformer':
        return TransformerFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            d_model=config.DEEP_FEATURE_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
    
    else:
        raise ValueError(
            f"Unknown FEATURE_EXTRACTION_MODE: {config.FEATURE_EXTRACTION_MODE}. "
            f"Must be 'handcrafted', 'deep_cnn', or 'deep_transformer'."
        )
