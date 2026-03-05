"""
EEG Emotion Visualization using t-SNE
======================================

This script visualizes EEG features in 2D space using t-SNE for:
1. Each subject separately (to see within-subject emotion clusters)
2. All emotions together (to see cross-subject emotion patterns)
3. Subject-wise coloring (to see subject differences)

Usage:
    python visualize_eeg_tsne.py

Author: Final Year Project
Date: 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

# Import your pipeline modules
from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data, extract_eeg_features


# ==================================================
# CONFIGURATION
# ==================================================

config = Config()
np.random.seed(config.SEED)

# Output directory for plots
OUTPUT_DIR = "./tsne_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# t-SNE parameters
PERPLEXITY = 30
N_ITER = 1000
LEARNING_RATE = 200

# Plot parameters
FIGSIZE = (10, 8)
DPI = 300
POINT_SIZE = 50
ALPHA = 0.7

# Color schemes
EMOTION_COLORS = {
    0: '#FF6B6B',  # Q1 - Red (Positive + High Arousal)
    1: '#4ECDC4',  # Q2 - Cyan (Negative + High Arousal)
    2: '#45B7D1',  # Q3 - Blue (Negative + Low Arousal)
    3: '#FFA07A',  # Q4 - Orange (Positive + Low Arousal)
}

EMOTION_LABELS = {
    0: 'Q1: Enthusiasm',
    1: 'Q2: Fear',
    2: 'Q3: Sadness',
    3: 'Q4: Neutral'
}


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def apply_tsne(features, perplexity=30, n_iter=1000, learning_rate=200):
    """
    Apply t-SNE dimensionality reduction.
    
    Args:
        features: (N, D) feature array
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        learning_rate: Learning rate
    
    Returns:
        embedding: (N, 2) 2D embedding
    """
    print(f"   Applying t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    # Optional: Use PCA for pre-processing if features are high-dimensional
    if features.shape[1] > 50:
        print(f"   Pre-processing with PCA: {features.shape[1]} -> 50 dimensions")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=config.SEED,
        verbose=0
    )
    
    embedding = tsne.fit_transform(features)
    print(f"   t-SNE complete! Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return embedding


def plot_tsne_by_emotion(embedding, labels, title, save_path):
    """
    Plot t-SNE embedding colored by emotion.
    
    Args:
        embedding: (N, 2) 2D coordinates
        labels: (N,) emotion labels (0-3)
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=FIGSIZE)
    
    for emotion_id in range(config.NUM_CLASSES):
        mask = (labels == emotion_id)
        if mask.sum() == 0:
            continue
        
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=EMOTION_COLORS[emotion_id],
            label=EMOTION_LABELS[emotion_id],
            s=POINT_SIZE,
            alpha=ALPHA,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_tsne_by_subject(embedding, subject_ids, title, save_path):
    """
    Plot t-SNE embedding colored by subject.
    
    Args:
        embedding: (N, 2) 2D coordinates
        subject_ids: (N,) subject IDs
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=FIGSIZE)
    
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    # Generate colors using a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))
    
    for idx, subject in enumerate(unique_subjects):
        mask = (subject_ids == subject)
        if mask.sum() == 0:
            continue
        
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[idx]],
            label=f'Subject {subject}',
            s=POINT_SIZE,
            alpha=ALPHA,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_tsne_subject_emotion_grid(embedding, labels, subject_ids, save_path):
    """
    Create a grid of subplots: one for each subject showing their emotions.
    
    Args:
        embedding: (N, 2) 2D coordinates
        labels: (N,) emotion labels
        subject_ids: (N,) subject IDs
        save_path: Path to save the plot
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    # Calculate grid size
    n_cols = min(4, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten() if n_subjects > 1 else [axes]
    
    for idx, subject in enumerate(unique_subjects):
        ax = axes[idx]
        mask = (subject_ids == subject)
        
        if mask.sum() == 0:
            ax.axis('off')
            continue
        
        subject_embedding = embedding[mask]
        subject_labels = labels[mask]
        
        # Plot each emotion for this subject
        for emotion_id in range(config.NUM_CLASSES):
            emotion_mask = (subject_labels == emotion_id)
            if emotion_mask.sum() == 0:
                continue
            
            ax.scatter(
                subject_embedding[emotion_mask, 0],
                subject_embedding[emotion_mask, 1],
                c=EMOTION_COLORS[emotion_id],
                label=EMOTION_LABELS[emotion_id],
                s=30,
                alpha=ALPHA,
                edgecolors='white',
                linewidth=0.5
            )
        
        ax.set_title(f'Subject {subject}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('t-SNE Visualization: Each Subject', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


# ==================================================
# MAIN VISUALIZATION PIPELINE
# ==================================================

def main():
    """Main visualization pipeline."""
    print("="*80)
    print("EEG EMOTION VISUALIZATION WITH t-SNE")
    print("="*80)
    
    # Step 1: Load EEG data
    print("\n📂 Loading EEG data...")
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(config.DATA_ROOT, config)
    
    # Step 2: Extract handcrafted features
    print("\n🔧 Extracting features...")
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Flatten features: (N, C, F) -> (N, C*F)
    N, C, F = eeg_X_features.shape
    features_flat = eeg_X_features.reshape(N, C * F)
    
    print(f"   Feature shape: {features_flat.shape}")
    print(f"   Number of subjects: {len(np.unique(eeg_subjects))}")
    print(f"   Emotion distribution: {np.bincount(eeg_y)}")
    
    # Step 3: Overall t-SNE visualization (all data)
    print("\n" + "="*80)
    print("1️⃣  OVERALL VISUALIZATION (ALL SUBJECTS)")
    print("="*80)
    
    embedding_all = apply_tsne(features_flat, PERPLEXITY, N_ITER, LEARNING_RATE)
    
    # Plot 1: Color by emotion
    plot_tsne_by_emotion(
        embedding_all, 
        eeg_y,
        "t-SNE: All Subjects (Colored by Emotion)",
        os.path.join(OUTPUT_DIR, "tsne_all_by_emotion.png")
    )
    
    # Plot 2: Color by subject
    plot_tsne_by_subject(
        embedding_all,
        eeg_subjects,
        "t-SNE: All Subjects (Colored by Subject)",
        os.path.join(OUTPUT_DIR, "tsne_all_by_subject.png")
    )
    
    # Step 4: Per-subject t-SNE visualizations
    print("\n" + "="*80)
    print("2️⃣  PER-SUBJECT VISUALIZATION")
    print("="*80)
    
    unique_subjects = np.unique(eeg_subjects)
    
    for subject in unique_subjects:
        print(f"\n📊 Processing Subject: {subject}")
        
        # Get data for this subject
        subject_mask = (eeg_subjects == subject)
        subject_features = features_flat[subject_mask]
        subject_labels = eeg_y[subject_mask]
        
        n_samples = subject_features.shape[0]
        print(f"   Samples: {n_samples}")
        print(f"   Emotion distribution: {np.bincount(subject_labels, minlength=config.NUM_CLASSES)}")
        
        if n_samples < 10:
            print(f"   ⚠️  Skipping (too few samples)")
            continue
        
        # Apply t-SNE for this subject
        perplexity_subject = min(PERPLEXITY, n_samples // 3)
        embedding_subject = apply_tsne(subject_features, perplexity_subject, N_ITER, LEARNING_RATE)
        
        # Plot for this subject
        plot_tsne_by_emotion(
            embedding_subject,
            subject_labels,
            f"t-SNE: Subject {subject} (Emotions)",
            os.path.join(OUTPUT_DIR, f"tsne_subject_{subject}.png")
        )
    
    # Step 5: Grid visualization (all subjects in one figure)
    print("\n" + "="*80)
    print("3️⃣  GRID VISUALIZATION (ALL SUBJECTS)")
    print("="*80)
    
    plot_tsne_subject_emotion_grid(
        embedding_all,
        eeg_y,
        eeg_subjects,
        os.path.join(OUTPUT_DIR, "tsne_subject_grid.png")
    )
    
    # Step 6: Emotion-wise visualization (each emotion separately)
    print("\n" + "="*80)
    print("4️⃣  EMOTION-WISE VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for emotion_id in range(config.NUM_CLASSES):
        ax = axes[emotion_id]
        emotion_mask = (eeg_y == emotion_id)
        
        if emotion_mask.sum() == 0:
            ax.axis('off')
            continue
        
        emotion_embedding = embedding_all[emotion_mask]
        emotion_subjects = eeg_subjects[emotion_mask]
        
        # Color by subject within this emotion
        unique_emotion_subjects = np.unique(emotion_subjects)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_emotion_subjects)))
        
        for idx, subject in enumerate(unique_emotion_subjects):
            subject_mask = (emotion_subjects == subject)
            ax.scatter(
                emotion_embedding[subject_mask, 0],
                emotion_embedding[subject_mask, 1],
                c=[colors[idx]],
                label=f'S{subject}',
                s=40,
                alpha=ALPHA,
                edgecolors='white',
                linewidth=0.5
            )
        
        ax.set_title(f'{EMOTION_LABELS[emotion_id]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('t-SNE Visualization: Each Emotion (Colored by Subject)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "tsne_emotion_wise.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()
    
    # Step 7: Summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n✅ Total samples: {N}")
    print(f"✅ Feature dimension: {C * F} (channels: {C}, features per channel: {F})")
    print(f"✅ Number of subjects: {len(unique_subjects)}")
    print(f"✅ Subjects: {unique_subjects}")
    
    print(f"\n📈 Emotion distribution:")
    for emotion_id in range(config.NUM_CLASSES):
        count = np.sum(eeg_y == emotion_id)
        pct = 100.0 * count / N
        print(f"   {EMOTION_LABELS[emotion_id]}: {count} ({pct:.1f}%)")
    
    print(f"\n💾 All plots saved to: {OUTPUT_DIR}/")
    print("="*80)
    print("🎉 VISUALIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
