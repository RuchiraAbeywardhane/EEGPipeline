"""
EEG Feature Extraction Module
==============================

This module contains feature extraction functions for EEG signals.

Features extracted (26 per channel):
1. Differential Entropy (5 bands)
2. Log Power Spectral Density (5 bands)
3. Temporal statistics (mean, std, skewness, kurtosis)
4. DE asymmetry (left-right hemisphere)
5. Bandpower ratios (theta/alpha, beta/alpha, gamma/beta)
6. Hjorth parameters (mobility, complexity)
7. Time-domain extras (log variance, zero crossing rate)

Author: Final Year Project
Date: 2026
"""

import numpy as np
from scipy.stats import skew, kurtosis


def extract_eeg_features(X_raw, config, fs=256.0, eps=1e-12):
    """
    Extract 26 features per channel from EEG windows.
    
    Features include:
    1. Differential Entropy (5 bands)
    2. Log Power Spectral Density (5 bands)
    3. Temporal statistics (mean, std, skewness, kurtosis)
    4. DE asymmetry (left-right hemisphere)
    5. Bandpower ratios (theta/alpha, beta/alpha, gamma/beta)
    6. Hjorth parameters (mobility, complexity)
    7. Time-domain extras (log variance, zero crossing rate)
    
    Args:
        X_raw: (N, T, C) - Raw EEG windows
        config: Configuration object with BANDS definition
        fs: Sampling frequency (Hz)
        eps: Small constant for numerical stability
    
    Returns:
        features: (N, C, 26) - Extracted features
    """
    print("Extracting EEG features (26 per channel)...")
    N, T, C = X_raw.shape
    
    # Compute power spectral density
    P = (np.abs(np.fft.rfft(X_raw, axis=1))**2) / T
    freqs = np.fft.rfftfreq(T, d=1/fs)
    
    feature_list = []
    
    # 1) Differential Entropy (5 features)
    de_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * (bp + eps))
        de_feats.append(de[..., None])
    de_all = np.concatenate(de_feats, axis=2)
    feature_list.append(de_all)
    
    # 2) Log-PSD (5 features)
    psd_feats = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        log_psd = np.log(bp + eps)
        psd_feats.append(log_psd[..., None])
    psd_all = np.concatenate(psd_feats, axis=2)
    feature_list.append(psd_all)
    
    # 3) Temporal statistics (4 features)
    temp_mean = X_raw.mean(axis=1)[..., None]
    temp_std = X_raw.std(axis=1)[..., None]
    temp_skew = skew(X_raw, axis=1)[..., None]
    temp_kurt = kurtosis(X_raw, axis=1)[..., None]
    temp_all = np.concatenate([temp_mean, temp_std, temp_skew, temp_kurt], axis=2)
    feature_list.append(temp_all)
    
    # 4) DE asymmetry (5 features)
    # Left hemisphere: TP9 (idx 0), AF7 (idx 1)
    # Right hemisphere: AF8 (idx 2), TP10 (idx 3)
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2
    de_asym = de_left - de_right
    de_asym_full = np.tile(de_asym[:, None, :], (1, C, 1))
    feature_list.append(de_asym_full)
    
    # 5) Bandpower ratios (3 features)
    band_bp = []
    for _, (lo, hi) in config.BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)
        band_bp.append(bp)
    _, theta_bp, alpha_bp, beta_bp, gamma_bp = band_bp
    
    ratio_theta_alpha = (theta_bp + eps) / (alpha_bp + eps)
    ratio_beta_alpha = (beta_bp + eps) / (alpha_bp + eps)
    ratio_gamma_beta = (gamma_bp + eps) / (beta_bp + eps)
    ratio_all = np.stack([ratio_theta_alpha, ratio_beta_alpha, ratio_gamma_beta], axis=2)
    feature_list.append(ratio_all)
    
    # 6) Hjorth parameters (2 features)
    Xc = X_raw - X_raw.mean(axis=1, keepdims=True)
    dx = np.diff(Xc, axis=1)
    var_x = (Xc**2).mean(axis=1) + eps
    var_dx = (dx**2).mean(axis=1) + eps
    mobility = np.sqrt(var_dx / var_x)
    ddx = np.diff(dx, axis=1)
    var_ddx = (ddx**2).mean(axis=1) + eps
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / (mobility + eps)
    hjorth_all = np.stack([mobility, complexity], axis=2)
    feature_list.append(hjorth_all)
    
    # 7) Time-domain extras (2 features)
    log_var = np.log(var_x + eps)
    sign_x = np.sign(Xc)
    zc = (np.diff(sign_x, axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    td_extras = np.stack([log_var, zc], axis=2)
    feature_list.append(td_extras)
    
    # Concatenate all features
    features = np.concatenate(feature_list, axis=2)
    print(f"EEG features extracted: {features.shape}")
    
    return features.astype(np.float32)
