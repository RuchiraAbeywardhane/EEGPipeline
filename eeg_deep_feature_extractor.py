"""
Deep Learning Feature Extractors for EEG
=========================================

This module provides CNN and Transformer-based feature extractors
as alternatives to handcrafted features, plus hybrid fusion methods.

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
# HYBRID FEATURE FUSION
# ==================================================

class ConcatFusion(nn.Module):
    """Simple concatenation of handcrafted and deep features."""
    
    def __init__(self, handcrafted_dim, deep_dim, output_dim, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(handcrafted_dim + deep_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, handcrafted_feats, deep_feats):
        """
        Args:
            handcrafted_feats: (B, handcrafted_dim)
            deep_feats: (B, deep_dim)
        
        Returns:
            fused_feats: (B, output_dim)
        """
        combined = torch.cat([handcrafted_feats, deep_feats], dim=1)
        return self.projection(combined)


class AttentionFusion(nn.Module):
    """Attention-based fusion that learns to weight handcrafted vs deep features."""
    
    def __init__(self, handcrafted_dim, deep_dim, output_dim, dropout=0.3):
        super().__init__()
        
        # Project both to same dimension
        self.handcrafted_proj = nn.Linear(handcrafted_dim, output_dim)
        self.deep_proj = nn.Linear(deep_dim, output_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),  # 2 weights: handcrafted vs deep
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, handcrafted_feats, deep_feats):
        """
        Args:
            handcrafted_feats: (B, handcrafted_dim)
            deep_feats: (B, deep_dim)
        
        Returns:
            fused_feats: (B, output_dim)
        """
        # Project to same dimension
        h_proj = self.handcrafted_proj(handcrafted_feats)  # (B, output_dim)
        d_proj = self.deep_proj(deep_feats)  # (B, output_dim)
        
        # Compute attention weights
        combined = torch.cat([h_proj, d_proj], dim=1)  # (B, output_dim*2)
        weights = self.attention(combined)  # (B, 2)
        
        # Weighted fusion
        w_h = weights[:, 0].unsqueeze(1)  # (B, 1)
        w_d = weights[:, 1].unsqueeze(1)  # (B, 1)
        
        fused = w_h * h_proj + w_d * d_proj  # (B, output_dim)
        
        return self.output_proj(fused)


class HybridFeatureExtractor(nn.Module):
    """
    Hybrid feature extractor combining handcrafted and deep learning features.
    
    Pipeline:
    1. Extract handcrafted features from processed signal (26 features × 4 channels)
    2. Extract deep features from raw signal (CNN or Transformer)
    3. Fuse both feature types (concatenation or attention-based)
    
    Args:
        deep_extractor: CNN or Transformer feature extractor
        fusion_mode: 'concat' or 'attention'
        handcrafted_dim: Dimension of handcrafted features (default: 104 = 26×4)
        deep_dim: Dimension of deep features (default: 128)
        output_dim: Output dimension after fusion
        dropout: Dropout rate
    """
    
    def __init__(self, deep_extractor, fusion_mode='concat', 
                 handcrafted_dim=104, deep_dim=128, output_dim=256, dropout=0.3):
        super().__init__()
        self.deep_extractor = deep_extractor
        self.fusion_mode = fusion_mode
        
        # Fusion layer
        if fusion_mode == 'concat':
            self.fusion = ConcatFusion(handcrafted_dim, deep_dim, output_dim, dropout)
        elif fusion_mode == 'attention':
            self.fusion = AttentionFusion(handcrafted_dim, deep_dim, output_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def forward(self, raw_eeg, handcrafted_feats):
        """
        Args:
            raw_eeg: (B, T, C) - Raw EEG signals
            handcrafted_feats: (B, C, 26) - Handcrafted features
        
        Returns:
            fused_features: (B, output_dim) - Fused features
        """
        # Extract deep features from raw EEG
        deep_feats = self.deep_extractor(raw_eeg)  # (B, deep_dim)
        
        # Flatten handcrafted features
        B, C, F = handcrafted_feats.shape
        handcrafted_flat = handcrafted_feats.reshape(B, C * F)  # (B, 104)
        
        # Fuse features
        fused = self.fusion(handcrafted_flat, deep_feats)  # (B, output_dim)
        
        return fused


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
        is_hybrid: Boolean indicating if hybrid mode is used
    """
    if config.FEATURE_EXTRACTION_MODE == 'handcrafted':
        return None, False  # Use handcrafted features in data loader
    
    elif config.FEATURE_EXTRACTION_MODE == 'deep_cnn':
        extractor = CNNFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            filters=config.CNN_FILTERS,
            kernel_size=config.CNN_KERNEL_SIZE,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
        return extractor, False
    
    elif config.FEATURE_EXTRACTION_MODE == 'deep_transformer':
        extractor = TransformerFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            d_model=config.DEEP_FEATURE_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
        return extractor, False
    
    elif config.FEATURE_EXTRACTION_MODE == 'hybrid_cnn':
        # Create CNN extractor
        cnn_extractor = CNNFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            filters=config.CNN_FILTERS,
            kernel_size=config.CNN_KERNEL_SIZE,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
        # Wrap in hybrid extractor
        hybrid = HybridFeatureExtractor(
            deep_extractor=cnn_extractor,
            fusion_mode=config.HYBRID_FUSION_MODE,
            handcrafted_dim=config.EEG_CHANNELS * config.EEG_FEATURES,  # 4 × 26 = 104
            deep_dim=config.DEEP_FEATURE_DIM,
            output_dim=config.DEEP_FEATURE_DIM * 2,  # 256 for richer representation
            dropout=0.3
        )
        return hybrid, True
    
    elif config.FEATURE_EXTRACTION_MODE == 'hybrid_transformer':
        # Create Transformer extractor
        transformer_extractor = TransformerFeatureExtractor(
            n_channels=config.EEG_CHANNELS,
            d_model=config.DEEP_FEATURE_DIM,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            feature_dim=config.DEEP_FEATURE_DIM,
            dropout=0.3
        )
        # Wrap in hybrid extractor
        hybrid = HybridFeatureExtractor(
            deep_extractor=transformer_extractor,
            fusion_mode=config.HYBRID_FUSION_MODE,
            handcrafted_dim=config.EEG_CHANNELS * config.EEG_FEATURES,  # 4 × 26 = 104
            deep_dim=config.DEEP_FEATURE_DIM,
            output_dim=config.DEEP_FEATURE_DIM * 2,  # 256 for richer representation
            dropout=0.3
        )
        return hybrid, True
    
    else:
        raise ValueError(
            f"Unknown FEATURE_EXTRACTION_MODE: {config.FEATURE_EXTRACTION_MODE}. "
            f"Must be 'handcrafted', 'deep_cnn', 'deep_transformer', 'hybrid_cnn', or 'hybrid_transformer'."
        )
