"""
Subject Similarity Analysis for EEG Emotions
=============================================

This script finds groups of subjects who have similar emotion patterns.
Uses multiple methods:
1. Feature-based similarity (cluster subjects by their average features)
2. Prediction-based similarity (how similarly a model confuses their data)
3. Embedding-based similarity (t-SNE distance between subjects' data)

Output:
- Subject similarity matrix
- Hierarchical clustering dendrogram
- Subject groups with similar emotion patterns

Usage:
    python find_similar_subjects.py --method all
    python find_similar_subjects.py --method feature --n_groups 3

Author: Final Year Project
Date: 2026
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd

# Import existing modules
from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data, extract_eeg_features


# ==================================================
# CONFIGURATION
# ==================================================

config = Config()
np.random.seed(config.SEED)

OUTPUT_DIR = "./subject_similarity"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# METHOD 1: FEATURE-BASED SIMILARITY
# ==================================================

def compute_feature_similarity(features, subjects, labels):
    """
    Compute subject similarity based on average feature vectors.
    
    For each subject, compute:
    - Mean feature vector per emotion
    - Overall mean feature vector
    - Feature distribution statistics
    
    Then compute pairwise similarity between subjects.
    
    Args:
        features: (N, feature_dim) - Feature array
        subjects: (N,) - Subject IDs
        labels: (N,) - Emotion labels
    
    Returns:
        similarity_matrix: (n_subjects, n_subjects) - Cosine similarity
        subject_profiles: Dict of subject feature profiles
    """
    print("\n" + "="*80)
    print("METHOD 1: FEATURE-BASED SIMILARITY")
    print("="*80)
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_emotions = config.NUM_CLASSES
    feature_dim = features.shape[1]
    
    # Compute subject profiles
    subject_profiles = {}
    
    for subject in unique_subjects:
        subject_mask = (subjects == subject)
        subject_features = features[subject_mask]
        subject_labels = labels[subject_mask]
        
        # Mean feature per emotion
        emotion_means = []
        for emotion_id in range(n_emotions):
            emotion_mask = (subject_labels == emotion_id)
            if emotion_mask.sum() > 0:
                emotion_mean = subject_features[emotion_mask].mean(axis=0)
            else:
                emotion_mean = np.zeros(feature_dim)
            emotion_means.append(emotion_mean)
        
        # Overall mean
        overall_mean = subject_features.mean(axis=0)
        
        # Concatenate: [overall_mean, emotion1_mean, emotion2_mean, ...]
        profile = np.concatenate([overall_mean] + emotion_means)
        
        subject_profiles[subject] = profile
    
    # Create profile matrix
    profile_matrix = np.array([subject_profiles[s] for s in unique_subjects])
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(profile_matrix)
    
    print(f"   Computed similarity for {n_subjects} subjects")
    print(f"   Profile dimension: {profile_matrix.shape[1]}")
    print(f"   Average similarity: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.3f}")
    
    return similarity_matrix, subject_profiles


# ==================================================
# METHOD 2: EMBEDDING-BASED SIMILARITY
# ==================================================

def compute_embedding_similarity(features, subjects, labels):
    """
    Compute subject similarity based on t-SNE embedding overlap.
    
    For each subject:
    1. Get their data points in t-SNE space
    2. Compute centroid per emotion
    3. Measure distance between subjects' emotion centroids
    
    Args:
        features: (N, feature_dim) - Feature array
        subjects: (N,) - Subject IDs
        labels: (N,) - Emotion labels
    
    Returns:
        similarity_matrix: (n_subjects, n_subjects) - Distance-based similarity
        embedding: (N, 2) - t-SNE embedding
    """
    print("\n" + "="*80)
    print("METHOD 2: EMBEDDING-BASED SIMILARITY")
    print("="*80)
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_emotions = config.NUM_CLASSES
    
    # Apply t-SNE
    print("   Applying t-SNE...")
    n_samples = features.shape[0]
    
    if features.shape[1] > 50:
        pca = PCA(n_components=min(50, n_samples - 1))
        features_pca = pca.fit_transform(features)
    else:
        features_pca = features
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=config.SEED)
    embedding = tsne.fit_transform(features_pca)
    
    print(f"   t-SNE complete! KL divergence: {tsne.kl_divergence_:.4f}")
    
    # Compute emotion centroids per subject
    subject_centroids = {}
    
    for subject in unique_subjects:
        subject_mask = (subjects == subject)
        subject_embedding = embedding[subject_mask]
        subject_labels = labels[subject_mask]
        
        centroids = []
        for emotion_id in range(n_emotions):
            emotion_mask = (subject_labels == emotion_id)
            if emotion_mask.sum() > 0:
                centroid = subject_embedding[emotion_mask].mean(axis=0)
            else:
                centroid = np.array([np.nan, np.nan])
            centroids.append(centroid)
        
        subject_centroids[subject] = np.array(centroids)  # (n_emotions, 2)
    
    # Compute pairwise centroid distances
    distance_matrix = np.zeros((n_subjects, n_subjects))
    
    for i, subj_i in enumerate(unique_subjects):
        for j, subj_j in enumerate(unique_subjects):
            if i == j:
                distance_matrix[i, j] = 0
                continue
            
            centroids_i = subject_centroids[subj_i]
            centroids_j = subject_centroids[subj_j]
            
            # Compute average Euclidean distance between matching emotion centroids
            distances = []
            for emotion_id in range(n_emotions):
                c_i = centroids_i[emotion_id]
                c_j = centroids_j[emotion_id]
                if not (np.isnan(c_i).any() or np.isnan(c_j).any()):
                    dist = np.linalg.norm(c_i - c_j)
                    distances.append(dist)
            
            if len(distances) > 0:
                distance_matrix[i, j] = np.mean(distances)
            else:
                distance_matrix[i, j] = np.inf
    
    # Convert distance to similarity (inverse + normalize)
    max_dist = distance_matrix[distance_matrix != np.inf].max()
    similarity_matrix = 1 - (distance_matrix / max_dist)
    similarity_matrix[distance_matrix == np.inf] = 0
    
    print(f"   Average embedding distance: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.3f}")
    
    return similarity_matrix, embedding


# ==================================================
# METHOD 3: CONFUSION-BASED SIMILARITY
# ==================================================

def compute_confusion_similarity(features, subjects, labels):
    """
    Compute subject similarity based on confusion patterns.
    
    Train a simple classifier and see which subjects get confused similarly.
    Subjects with similar confusion matrices are similar.
    
    Args:
        features: (N, feature_dim) - Feature array
        subjects: (N,) - Subject IDs
        labels: (N,) - Emotion labels
    
    Returns:
        similarity_matrix: (n_subjects, n_subjects) - Confusion-based similarity
        confusion_matrices: Dict of confusion matrices per subject
    """
    print("\n" + "="*80)
    print("METHOD 3: CONFUSION-BASED SIMILARITY")
    print("="*80)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    # Train a simple classifier per subject
    confusion_matrices = {}
    
    for subject in unique_subjects:
        subject_mask = (subjects == subject)
        subject_features = features[subject_mask]
        subject_labels = labels[subject_mask]
        
        if len(subject_features) < 20:
            print(f"   Skipping subject {subject} (too few samples)")
            confusion_matrices[subject] = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
            continue
        
        # Train-test split
        n_train = int(0.7 * len(subject_features))
        indices = np.random.permutation(len(subject_features))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, y_train = subject_features[train_idx], subject_labels[train_idx]
        X_test, y_test = subject_features[test_idx], subject_labels[test_idx]
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=config.SEED)
        clf.fit(X_train, y_train)
        
        # Get predictions
        y_pred = clf.predict(X_test)
        
        # Compute confusion matrix (normalized)
        cm = confusion_matrix(y_test, y_pred, labels=range(config.NUM_CLASSES))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        confusion_matrices[subject] = cm_norm.flatten()  # Flatten to 1D
    
    # Compute similarity based on confusion pattern similarity
    confusion_vectors = np.array([confusion_matrices[s] for s in unique_subjects])
    similarity_matrix = cosine_similarity(confusion_vectors)
    
    print(f"   Computed confusion matrices for {n_subjects} subjects")
    print(f"   Average confusion similarity: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.3f}")
    
    return similarity_matrix, confusion_matrices


# ==================================================
# VISUALIZATION & CLUSTERING
# ==================================================

def plot_similarity_matrix(similarity_matrix, subjects, method_name, save_path):
    """Plot heatmap of subject similarity matrix."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=subjects,
        yticklabels=subjects,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Similarity'}
    )
    
    plt.title(f'Subject Similarity Matrix ({method_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Subject', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_dendrogram(similarity_matrix, subjects, method_name, save_path):
    """Plot hierarchical clustering dendrogram."""
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    plt.figure(figsize=(12, 6))
    
    dendrogram(
        linkage_matrix,
        labels=subjects,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    
    plt.title(f'Subject Clustering Dendrogram ({method_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.axhline(y=0.7 * max(linkage_matrix[:, 2]), c='red', linestyle='--', label='Cut threshold')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()
    
    return linkage_matrix


def find_subject_groups(similarity_matrix, subjects, n_groups=3, method='ward'):
    """
    Cluster subjects into groups.
    
    Args:
        similarity_matrix: (n_subjects, n_subjects) similarity matrix
        subjects: List of subject IDs
        n_groups: Number of groups to find
        method: Clustering method ('ward', 'average', 'complete')
    
    Returns:
        groups: Dict mapping group_id -> list of subjects
    """
    distance_matrix = 1 - similarity_matrix
    
    # Agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_groups,
        affinity='precomputed',
        linkage=method
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Group subjects
    groups = {}
    for group_id in range(n_groups):
        group_mask = (labels == group_id)
        groups[group_id] = subjects[group_mask].tolist()
    
    return groups, labels


# ==================================================
# MAIN EXECUTION
# ==================================================

def main():
    parser = argparse.ArgumentParser(description='Find similar subject groups for EEG emotions')
    parser.add_argument('--method', type=str, default='all',
                       choices=['feature', 'embedding', 'confusion', 'all'],
                       help='Similarity computation method')
    parser.add_argument('--n_groups', type=int, default=3,
                       help='Number of subject groups to find')
    parser.add_argument('--remove_overlap', action='store_true',
                       help='Remove overlapping windows before analysis')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SUBJECT SIMILARITY ANALYSIS")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Number of groups: {args.n_groups}")
    print("="*80)
    
    # Load data
    print("\n📂 Loading EEG data...")
    eeg_X_raw, eeg_y, eeg_subjects, label_to_id, eeg_clip_ids = load_eeg_data(config.DATA_ROOT, config)
    
    # Extract features
    print("\n🔧 Extracting features...")
    eeg_X_features = extract_eeg_features(eeg_X_raw, config)
    
    # Flatten features
    N, C, F = eeg_X_features.shape
    features_flat = eeg_X_features.reshape(N, C * F)
    
    # Remove overlapping windows if requested
    if args.remove_overlap:
        print("\n🔧 Removing overlapping windows...")
        unique_clips = np.unique(eeg_clip_ids)
        non_overlap_indices = []
        
        overlap_ratio = config.EEG_OVERLAP
        stride = int(1.0 / (1.0 - overlap_ratio)) if overlap_ratio > 0 else 1
        
        for clip_id in unique_clips:
            clip_mask = (eeg_clip_ids == clip_id)
            clip_indices = np.where(clip_mask)[0]
            selected = clip_indices[::stride]
            non_overlap_indices.extend(selected)
        
        non_overlap_indices = np.array(non_overlap_indices)
        features_flat = features_flat[non_overlap_indices]
        eeg_y = eeg_y[non_overlap_indices]
        eeg_subjects = eeg_subjects[non_overlap_indices]
        
        print(f"   Kept {len(non_overlap_indices)} non-overlapping windows")
    
    unique_subjects = np.unique(eeg_subjects)
    print(f"\n✅ Dataset loaded:")
    print(f"   Samples: {len(eeg_y)}")
    print(f"   Subjects: {len(unique_subjects)}")
    print(f"   Features: {features_flat.shape[1]}")
    
    # Compute similarity matrices
    similarity_matrices = {}
    
    if args.method in ['feature', 'all']:
        sim_matrix, profiles = compute_feature_similarity(features_flat, eeg_subjects, eeg_y)
        similarity_matrices['Feature-based'] = sim_matrix
        
        plot_similarity_matrix(sim_matrix, unique_subjects, 'Feature-based',
                              os.path.join(OUTPUT_DIR, 'similarity_matrix_feature.png'))
        linkage_mat = plot_dendrogram(sim_matrix, unique_subjects, 'Feature-based',
                                      os.path.join(OUTPUT_DIR, 'dendrogram_feature.png'))
    
    if args.method in ['embedding', 'all']:
        sim_matrix, embedding = compute_embedding_similarity(features_flat, eeg_subjects, eeg_y)
        similarity_matrices['Embedding-based'] = sim_matrix
        
        plot_similarity_matrix(sim_matrix, unique_subjects, 'Embedding-based',
                              os.path.join(OUTPUT_DIR, 'similarity_matrix_embedding.png'))
        linkage_mat = plot_dendrogram(sim_matrix, unique_subjects, 'Embedding-based',
                                      os.path.join(OUTPUT_DIR, 'dendrogram_embedding.png'))
    
    if args.method in ['confusion', 'all']:
        sim_matrix, confusion_mats = compute_confusion_similarity(features_flat, eeg_subjects, eeg_y)
        similarity_matrices['Confusion-based'] = sim_matrix
        
        plot_similarity_matrix(sim_matrix, unique_subjects, 'Confusion-based',
                              os.path.join(OUTPUT_DIR, 'similarity_matrix_confusion.png'))
        linkage_mat = plot_dendrogram(sim_matrix, unique_subjects, 'Confusion-based',
                                      os.path.join(OUTPUT_DIR, 'dendrogram_confusion.png'))
    
    # Combine methods if using 'all'
    if args.method == 'all' and len(similarity_matrices) > 1:
        print("\n" + "="*80)
        print("COMBINING ALL METHODS (ENSEMBLE)")
        print("="*80)
        
        combined_matrix = np.mean(list(similarity_matrices.values()), axis=0)
        similarity_matrices['Combined'] = combined_matrix
        
        plot_similarity_matrix(combined_matrix, unique_subjects, 'Combined',
                              os.path.join(OUTPUT_DIR, 'similarity_matrix_combined.png'))
        linkage_mat = plot_dendrogram(combined_matrix, unique_subjects, 'Combined',
                                      os.path.join(OUTPUT_DIR, 'dendrogram_combined.png'))
    
    # Find subject groups
    print("\n" + "="*80)
    print(f"CLUSTERING SUBJECTS INTO {args.n_groups} GROUPS")
    print("="*80)
    
    for method_name, sim_matrix in similarity_matrices.items():
        print(f"\n📊 {method_name} Groups:")
        groups, labels = find_subject_groups(sim_matrix, unique_subjects, args.n_groups)
        
        for group_id, subject_list in groups.items():
            print(f"   Group {group_id + 1}: {subject_list} ({len(subject_list)} subjects)")
            
            # Print average within-group similarity
            group_indices = [i for i, s in enumerate(unique_subjects) if s in subject_list]
            if len(group_indices) > 1:
                within_sim = []
                for i in group_indices:
                    for j in group_indices:
                        if i < j:
                            within_sim.append(sim_matrix[i, j])
                avg_sim = np.mean(within_sim) if within_sim else 0
                print(f"              Average within-group similarity: {avg_sim:.3f}")
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'subject_groups.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUBJECT SIMILARITY ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for method_name, sim_matrix in similarity_matrices.items():
            f.write(f"\n{method_name} Groups:\n")
            f.write("-"*80 + "\n")
            groups, labels = find_subject_groups(sim_matrix, unique_subjects, args.n_groups)
            
            for group_id, subject_list in groups.items():
                f.write(f"Group {group_id + 1}: {subject_list}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
