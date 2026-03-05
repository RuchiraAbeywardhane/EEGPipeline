"""
ERP-Based SVM Emotion Recognition Pipeline
===========================================

Recreates the methodology described in the paper:

1. DATA LOADING      - Uses existing MUSE dataloader (TP9, AF7, AF8, TP10)
2. ERP EXTRACTION    - Averages signal amplitude in 7 post-stimulus time windows:
                         N100 (100–170 ms), P200 (170–240 ms), N200 (240–300 ms),
                         P300 (300–400 ms), early LPP (400–700 ms),
                         middle LPP (700–1000 ms), late LPP (1000–1300 ms)
                       → 4 channels × 7 components = 28 features per window
3. FEATURE SELECTION - Sequential Forward Selection (SFS) wrapper
4. CLASSIFICATION    - Linear SVM (C ≈ 0.1), pairwise emotion tasks
5. EVALUATION        - 10-fold GroupKFold cross-validation (clip-independent):
                       Accuracy, Confusion Matrix, ROC-AUC

Leakage prevention
------------------
  • Clip-independent  : GroupKFold groups by clip_id so all windows from the
                        same recording always stay in the same fold.
  • Subject-dependent : Windows from different subjects CAN appear together in
                        the same fold — the model sees all subjects during
                        training (within-subject generalisation).

Author: Final Year Project
Date: 2026
"""

import os
import itertools
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import Counter

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, cross_val_predict, cross_validate
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.feature_selection import SequentialFeatureSelector

from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Pairwise tasks: each entry is (label_A, label_B, task_name)
# Labels must match Config.SUPERCLASS_MAP values
PAIRWISE_TASKS = [
    ("Q1", "Q2", "Positive_vs_Negative"),       # e.g. Amusement vs Anger
    ("Q1", "Q4", "Positive_vs_Neutral"),         # e.g. Amusement vs Neutral
    ("Q2", "Q4", "Negative_vs_Neutral"),         # e.g. Anger vs Neutral
]

# ERP time windows in milliseconds (post-stimulus onset)
# Format: (name, start_ms, end_ms)
ERP_WINDOWS = [
    ("N100",       100,  170),
    ("P200",       170,  240),
    ("N200",       240,  300),
    ("P300",       300,  400),
    ("earlyLPP",   400,  700),
    ("middleLPP",  700, 1000),
    ("lateLPP",   1000, 1300),
]

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]   # MUSE channel order

SVM_C         = 0.1       # Regularisation parameter (paper: C ≈ 0.1)
CV_FOLDS      = 10        # GroupKFold folds (one clip per fold where possible)
SFS_DIRECTION = "forward" # Sequential Forward Selection
SFS_CV        = 5         # Inner CV folds for SFS (also GroupKFold)
RESULTS_DIR   = "erp_svm_results"

np.random.seed(Config.SEED)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – ERP FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def ms_to_samples(ms: float, fs: float = 256.0) -> int:
    """Convert milliseconds to sample index."""
    return int(round(ms * fs / 1000.0))


def extract_erp_features(X_raw: np.ndarray,
                         fs: float = 256.0) -> tuple[np.ndarray, list[str]]:
    """
    Compute ERP component amplitudes by averaging within each time window.

    Parameters
    ----------
    X_raw : (N, T, C)  – raw EEG windows (samples × channels)
    fs    : sampling frequency in Hz

    Returns
    -------
    features    : (N, n_channels × n_erp_windows)  float32 array
    feat_names  : list of feature name strings
    """
    N, T, C = X_raw.shape
    print(f"\n{'='*70}")
    print("ERP FEATURE EXTRACTION")
    print(f"{'='*70}")
    print(f"  Input shape   : {X_raw.shape}  (windows × samples × channels)")
    print(f"  Sampling rate : {fs} Hz")
    print(f"  Window length : {T/fs*1000:.0f} ms  ({T} samples)")

    feature_cols = []
    feat_names   = []

    for erp_name, t_start, t_end in ERP_WINDOWS:
        s_start = ms_to_samples(t_start, fs)
        s_end   = ms_to_samples(t_end,   fs)

        # Clamp to actual window length
        s_start = min(s_start, T)
        s_end   = min(s_end,   T)

        if s_end <= s_start:
            print(f"  ⚠️  {erp_name}: window [{t_start}–{t_end} ms] outside signal — skipped")
            continue

        # Mean amplitude within the time window for every channel → (N, C)
        erp_amp = X_raw[:, s_start:s_end, :].mean(axis=1)
        feature_cols.append(erp_amp)

        for ch in CHANNEL_NAMES[:C]:
            feat_names.append(f"{erp_name}_{ch}")

        print(f"  ✅ {erp_name:12s}: [{t_start:4d}–{t_end:4d} ms]  "
              f"samples [{s_start:4d}–{s_end:4d}]  → {C} features")

    features = np.concatenate(feature_cols, axis=1).astype(np.float32)
    print(f"\n  Total features: {features.shape[1]}  "
          f"({C} channels × {len(feature_cols)} ERP components)")
    return features, feat_names


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – PAIRWISE DATASET PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_pairwise(X: np.ndarray, y_str: np.ndarray,
                     groups: np.ndarray,
                     label_a: str, label_b: str):
    """
    Filter dataset to only the two classes in a pairwise task.

    Parameters
    ----------
    X       : (N, F) feature matrix
    y_str   : (N,) string labels (e.g. "Q1", "Q2", ...)
    groups  : (N,) group labels (e.g. clip_ids)
    label_a : first class string
    label_b : second class string

    Returns
    -------
    X_pair  : (M, F)
    y_pair  : (M,)  binary {0, 1}
    g_pair  : (M,)  group labels
    """
    mask = np.isin(y_str, [label_a, label_b])
    X_pair = X[mask]
    y_raw  = y_str[mask]
    y_pair = (y_raw == label_b).astype(int)   # label_a → 0, label_b → 1
    g_pair = groups[mask]
    return X_pair, y_pair, g_pair


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – SEQUENTIAL FORWARD SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def run_sfs(X: np.ndarray, y: np.ndarray,
            groups: np.ndarray,
            feat_names: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Apply Sequential Forward Selection using a Linear SVM scorer.

    Iteratively adds the feature that most improves cross-validated accuracy
    until performance no longer increases (tol=1e-4).

    Parameters
    ----------
    X          : (N, F) full feature matrix (already scaled)
    y          : (N,)   binary labels
    groups     : (N,)   group labels (e.g. clip_ids)
    feat_names : list of F feature name strings

    Returns
    -------
    X_sel       : (N, K) selected feature matrix
    sel_names   : list of K selected feature names
    sel_indices : list of K selected feature indices
    """
    print(f"\n  Running Sequential Forward Selection  (inner CV={SFS_CV} folds) …")

    base_svm = SVC(kernel="linear", C=SVM_C, probability=False, random_state=Config.SEED)

    sfs = SequentialFeatureSelector(
        estimator  = base_svm,
        n_features_to_select = "auto",   # stops when score no longer improves
        tol        = 1e-4,
        direction  = SFS_DIRECTION,
        scoring    = "accuracy",
        cv         = GroupKFold(n_splits=SFS_CV),
        n_jobs     = -1,
    )
    sfs.fit(X, y, groups=groups)

    support     = sfs.get_support()
    sel_indices = list(np.where(support)[0])
    sel_names   = [feat_names[i] for i in sel_indices]
    X_sel       = X[:, support]

    print(f"  ✅ SFS selected {len(sel_indices)} / {X.shape[1]} features:")
    for name in sel_names:
        print(f"     • {name}")

    return X_sel, sel_names, sel_indices


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 – 10-FOLD CROSS-VALIDATION WITH LINEAR SVM
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray,
                         task_name: str) -> dict:
    """
    Train and evaluate a Linear SVM with 10-fold stratified CV.

    Reports: Accuracy, Confusion Matrix, ROC-AUC.

    Parameters
    ----------
    X         : (N, K)  selected & scaled features
    y         : (N,)    binary labels {0, 1}
    groups    : (N,)    group labels (e.g. clip_ids)
    task_name : string for logging

    Returns
    -------
    results dict with keys:
        accuracy, std, confusion_matrix, roc_auc,
        y_true, y_pred, y_prob
    """
    print(f"\n  10-Fold Cross-Validation …")

    svm_clf = SVC(kernel="linear", C=SVM_C,
                  probability=True, random_state=Config.SEED)

    cv = GroupKFold(n_splits=CV_FOLDS)

    # Collect per-fold scores
    cv_results = cross_validate(
        svm_clf, X, y,
        groups=groups,
        cv=cv,
        scoring=["accuracy"],
        return_train_score=False,
        n_jobs=-1
    )

    # Predictions & probabilities across all folds (for confusion matrix / ROC)
    y_pred = cross_val_predict(svm_clf, X, y, groups=groups, cv=cv, method="predict",  n_jobs=-1)
    y_prob = cross_val_predict(svm_clf, X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    acc      = cv_results["test_accuracy"].mean()
    acc_std  = cv_results["test_accuracy"].std()
    cm       = confusion_matrix(y, y_pred)
    roc_auc  = roc_auc_score(y, y_prob)

    print(f"  ✅ Accuracy : {acc*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"     ROC-AUC  : {roc_auc:.4f}")
    print(f"     Confusion matrix:\n{cm}")
    print(f"\n{classification_report(y, y_pred, digits=4)}")

    return {
        "accuracy"        : acc,
        "accuracy_std"    : acc_std,
        "fold_accuracies" : cv_results["test_accuracy"],
        "confusion_matrix": cm,
        "roc_auc"         : roc_auc,
        "y_true"          : y,
        "y_pred"          : y_pred,
        "y_prob"          : y_prob,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 – VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, task_name: str,
                          label_a: str, label_b: str,
                          save_dir: str) -> None:
    """Save a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label_a, label_b]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{task_name.replace('_', ' ')}", fontsize=11)
    plt.tight_layout()
    fpath = os.path.join(save_dir, f"cm_{task_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix saved → {fpath}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   roc_auc: float, task_name: str,
                   save_dir: str) -> None:
    """Save an ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve\n{task_name.replace('_', ' ')}", fontsize=11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fpath = os.path.join(save_dir, f"roc_{task_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  📈 ROC curve saved      → {fpath}")


def plot_feature_importance(sel_names: list[str],
                            task_name: str,
                            X_sel: np.ndarray,
                            y: np.ndarray,
                            save_dir: str) -> None:
    """
    Fit a single Linear SVM on the full selected set and plot |weights|
    as a proxy for feature importance.
    """
    if len(sel_names) == 0:
        return

    svm = SVC(kernel="linear", C=SVM_C,
              probability=False, random_state=Config.SEED)
    svm.fit(X_sel, y)
    weights = np.abs(svm.coef_[0])

    sorted_idx = np.argsort(weights)[::-1]
    sorted_w   = weights[sorted_idx]
    sorted_n   = [sel_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(sel_names) * 0.7), 4))
    ax.bar(range(len(sorted_n)), sorted_w, color="steelblue")
    ax.set_xticks(range(len(sorted_n)))
    ax.set_xticklabels(sorted_n, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|SVM weight|")
    ax.set_title(f"Feature Importance (Linear SVM)\n{task_name.replace('_', ' ')}")
    plt.tight_layout()
    fpath = os.path.join(save_dir, f"importance_{task_name}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  🔍 Feature importance saved → {fpath}")


def plot_summary_bar(task_results: dict, save_dir: str) -> None:
    """
    Bar chart comparing accuracy & AUC across all pairwise tasks.
    """
    tasks  = list(task_results.keys())
    accs   = [task_results[t]["accuracy"] * 100  for t in tasks]
    aucs   = [task_results[t]["roc_auc"]          for t in tasks]
    stds   = [task_results[t]["accuracy_std"] * 100 for t in tasks]

    x = np.arange(len(tasks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, accs, w, yerr=stds, capsize=4,
                   label="Accuracy (%)", color="steelblue", alpha=0.85)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "summary_all_tasks.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\n  📊 Summary chart saved → {fpath}")


def plot_erp_component_heatmap(all_task_selected: dict,
                                feat_names: list[str],
                                save_dir: str) -> None:
    """
    Heatmap showing which ERP-component × channel combinations were selected
    by SFS across pairwise tasks.
    """
    erp_names = [e[0] for e in ERP_WINDOWS]
    n_erp = len(erp_names)
    n_ch  = len(CHANNEL_NAMES)

    # Count selections per task across ERP × channel grid
    fig, axes = plt.subplots(1, len(all_task_selected),
                              figsize=(5 * len(all_task_selected), 4),
                              squeeze=False)

    for ax, (task_name, sel_names) in zip(axes[0], all_task_selected.items()):
        grid = np.zeros((n_erp, n_ch), dtype=int)
        for name in sel_names:
            parts = name.split("_")
            erp_part = parts[0]
            ch_part  = parts[-1]
            if erp_part in erp_names and ch_part in CHANNEL_NAMES:
                ri = erp_names.index(erp_part)
                ci = CHANNEL_NAMES.index(ch_part)
                grid[ri, ci] = 1

        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(CHANNEL_NAMES, fontsize=9)
        ax.set_yticks(range(n_erp))
        ax.set_yticklabels(erp_names, fontsize=9)
        ax.set_title(task_name.replace("_", "\n"), fontsize=9)

        # Annotate cells
        for r in range(n_erp):
            for c in range(n_ch):
                ax.text(c, r, "✓" if grid[r, c] else "",
                        ha="center", va="center", fontsize=11,
                        color="white" if grid[r, c] else "lightgrey")

    plt.suptitle("SFS-Selected ERP Features (ERP × Channel)", fontsize=12, y=1.02)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "erp_selection_heatmap.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🗺️  ERP selection heatmap saved → {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ERP-BASED SVM EMOTION RECOGNITION PIPELINE")
    print("=" * 70)
    print(f"  Dataset         : {config.DATA_ROOT}")
    print(f"  ERP windows     : {len(ERP_WINDOWS)}")
    print(f"  Channels        : {CHANNEL_NAMES}")
    print(f"  SVM C           : {SVM_C}")
    print(f"  CV folds        : {CV_FOLDS}")
    print(f"  SFS direction   : {SFS_DIRECTION}")
    print(f"  Results dir     : {RESULTS_DIR}")
    print("=" * 70)

    # ── 1. Load raw EEG windows ───────────────────────────────────────────────
    X_raw, y_int, subject_ids, label_to_id, clip_ids = load_eeg_data(
        config.DATA_ROOT, config
    )
    # Map integer labels back to string labels (e.g. "Q1", "Q2", …)
    id_to_label = {v: k for k, v in label_to_id.items()}
    y_str = np.array([id_to_label[i] for i in y_int])

    print(f"\n✅ Loaded data: {X_raw.shape}  "
          f"(windows × samples × channels)")
    print(f"   Label distribution: {Counter(y_str)}")

    fs  = config.EEG_FS
    T   = X_raw.shape[1]
    win_ms = T / fs * 1000
    print(f"\n   Window duration : {win_ms:.0f} ms  ({T} samples @ {fs} Hz)")
    print(f"   Max ERP window  : {ERP_WINDOWS[-1][2]} ms")
    if win_ms < ERP_WINDOWS[-1][2]:
        print(f"   ⚠️  Window ({win_ms:.0f} ms) is shorter than the latest ERP "
              f"component ({ERP_WINDOWS[-1][2]} ms).  Late LPP features will be "
              f"partially clipped.  Consider increasing EEG_WINDOW_SEC in config.")

    # ── 2. Extract ERP features ───────────────────────────────────────────────
    X_erp, feat_names = extract_erp_features(X_raw, fs=fs)
    print(f"\n  Feature matrix: {X_erp.shape}  "
          f"({X_erp.shape[1]} features per window)")
    print(f"  Feature names : {feat_names}")

    # ── 3. Run pairwise tasks ─────────────────────────────────────────────────
    task_results      = {}
    all_task_selected = {}

    for label_a, label_b, task_name in PAIRWISE_TASKS:
        print(f"\n{'='*70}")
        print(f"TASK: {task_name}  ({label_a} vs {label_b})")
        print(f"{'='*70}")

        # Filter to pairwise subset
        X_pair, y_pair = prepare_pairwise(X_erp, y_str, label_a, label_b)

        if len(np.unique(y_pair)) < 2:
            print(f"  ⚠️  Not enough samples for both classes — skipping task.")
            continue

        n_a = (y_pair == 0).sum()
        n_b = (y_pair == 1).sum()
        print(f"  Samples: {label_a}={n_a}, {label_b}={n_b}  (total={len(y_pair)})")

        # Standardise features
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_pair)

        # Sequential Forward Selection
        X_sel, sel_names, sel_indices = run_sfs(X_scaled, y_pair, feat_names)
        all_task_selected[task_name] = sel_names

        # 10-fold cross-validation on selected features
        results = run_cross_validation(X_sel, y_pair, task_name)
        task_results[task_name] = results

        # ── Visualisations ────────────────────────────────────────────────────
        plot_confusion_matrix(
            results["confusion_matrix"], task_name, label_a, label_b, RESULTS_DIR
        )
        plot_roc_curve(
            results["y_true"], results["y_prob"],
            results["roc_auc"], task_name, RESULTS_DIR
        )
        plot_feature_importance(
            sel_names, task_name, X_sel, y_pair, RESULTS_DIR
        )

    # ── 4. Cross-task summary ─────────────────────────────────────────────────
    if task_results:
        plot_summary_bar(task_results, RESULTS_DIR)
        plot_erp_component_heatmap(all_task_selected, feat_names, RESULTS_DIR)

    # ── 5. Print final table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"{'Task':<30}  {'Accuracy':>10}  {'±Std':>8}  {'ROC-AUC':>9}  {'#Feats':>7}"
    print(header)
    print("-" * len(header))
    for task_name, res in task_results.items():
        n_sel = len(all_task_selected.get(task_name, []))
        print(f"{task_name:<30}  "
              f"{res['accuracy']*100:>9.2f}%  "
              f"{res['accuracy_std']*100:>7.2f}%  "
              f"{res['roc_auc']:>9.4f}  "
              f"{n_sel:>7}")

    print(f"\n{'='*70}")
    print("✅ ERP-SVM PIPELINE COMPLETE")
    print(f"   All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'='*70}")

    return task_results, all_task_selected


if __name__ == "__main__":
    main()
