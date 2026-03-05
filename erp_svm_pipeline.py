"""
ERP-Based SVM Emotion Recognition Pipeline  (4-Class)
======================================================

1. DATA LOADING      - MUSE dataloader (TP9, AF7, AF8, TP10)
2. ERP EXTRACTION    - Mean amplitude in 7 post-stimulus windows:
                         N100 (100–170 ms), P200 (170–240 ms), N200 (240–300 ms),
                         P300 (300–400 ms), earlyLPP (400–700 ms),
                         middleLPP (700–1000 ms), lateLPP (1000–1300 ms)
                       → 4 channels × 7 components = 28 features per window
3. FEATURE SELECTION - Sequential Forward Selection (SFS) with GroupKFold
4. CLASSIFICATION    - Linear SVM (C ≈ 0.1), one-vs-rest multiclass, 4 emotions:
                         Q1 = Amusement  (Positive / High Arousal)
                         Q2 = Anger      (Negative / High Arousal)
                         Q3 = Sadness    (Negative / Low Arousal)
                         Q4 = Neutral    (Positive / Low Arousal)
5. EVALUATION        - 10-fold GroupKFold CV (clip-independent, subject-dependent):
                       Accuracy, 4×4 Confusion Matrix, per-class & macro ROC-AUC

Leakage prevention
------------------
  • Clip-independent  : GroupKFold on clip_ids — windows from the same
                        recording never span train/test folds.
  • Subject-dependent : All subjects seen during training
                        (within-subject generalisation).

Author: Final Year Project
Date: 2026
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate, cross_val_score
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc,
    classification_report
)

from eeg_config import Config
from eeg_data_loader_emognitionRaw import load_eeg_data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# 4-class labels and display names (must match Config.SUPERCLASS_MAP values)
CLASS_LABELS   = ["Q1", "Q2", "Q3", "Q4"]
CLASS_NAMES    = {
    "Q1": "Amusement\n(Pos/High)",
    "Q2": "Anger\n(Neg/High)",
    "Q3": "Sadness\n(Neg/Low)",
    "Q4": "Neutral\n(Pos/Low)",
}

ERP_WINDOWS = [
    ("N100",      100,  170),
    ("P200",      170,  240),
    ("N200",      240,  300),
    ("P300",      300,  400),
    ("earlyLPP",  400,  700),
    ("middleLPP", 700, 1000),
    ("lateLPP",  1000, 1300),
]

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

SVM_C         = 0.1
CV_FOLDS      = 10
SFS_CV        = 5
SFS_DIRECTION = "forward"
RESULTS_DIR   = "erp_svm_results"

np.random.seed(Config.SEED)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – ERP FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def ms_to_samples(ms: float, fs: float = 256.0) -> int:
    return int(round(ms * fs / 1000.0))


def extract_erp_features(X_raw: np.ndarray,
                         fs: float = 256.0) -> tuple[np.ndarray, list[str]]:
    """
    Average signal amplitude within each ERP time window per channel.
    Returns (N, C × n_erp_windows) feature matrix and feature name list.
    """
    N, T, C = X_raw.shape
    print(f"\n{'='*70}")
    print("ERP FEATURE EXTRACTION")
    print(f"{'='*70}")
    print(f"  Input shape   : {X_raw.shape}  (windows × samples × channels)")
    print(f"  Sampling rate : {fs} Hz")
    print(f"  Window length : {T/fs*1000:.0f} ms  ({T} samples)")

    feature_cols, feat_names = [], []

    for erp_name, t_start, t_end in ERP_WINDOWS:
        s_start = min(ms_to_samples(t_start, fs), T)
        s_end   = min(ms_to_samples(t_end,   fs), T)

        if s_end <= s_start:
            print(f"  ⚠️  {erp_name}: [{t_start}–{t_end} ms] outside signal — skipped")
            continue

        erp_amp = X_raw[:, s_start:s_end, :].mean(axis=1)   # (N, C)
        feature_cols.append(erp_amp)
        for ch in CHANNEL_NAMES[:C]:
            feat_names.append(f"{erp_name}_{ch}")

        print(f"  ✅ {erp_name:12s}: [{t_start:4d}–{t_end:4d} ms]  "
              f"samples [{s_start:4d}–{s_end:4d}]  → {C} features")

    features = np.concatenate(feature_cols, axis=1).astype(np.float32)
    print(f"\n  Total features : {features.shape[1]}  "
          f"({C} channels × {len(feature_cols)} ERP components)")
    return features, feat_names


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – SEQUENTIAL FORWARD SELECTION  (clip-independent inner CV)
# ──────────────────────────────────────────────────────────────────────────────

def run_sfs(X: np.ndarray, y: np.ndarray,
            groups: np.ndarray,
            feat_names: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Manual Sequential Forward Selection using cross_val_score with GroupKFold.
    Iteratively adds the feature that most improves CV accuracy.
    Stops when no candidate improves accuracy by more than tol=1e-4.
    """
    n_clips     = len(np.unique(groups))
    inner_folds = min(SFS_CV, n_clips)
    print(f"\n  Running SFS  (GroupKFold, {inner_folds} inner folds, "
          f"{n_clips} clips) …")

    base_svm = SVC(kernel="linear", C=SVM_C,
                   decision_function_shape="ovr",
                   probability=False, random_state=Config.SEED)
    cv = GroupKFold(n_splits=inner_folds)

    n_features      = X.shape[1]
    selected        = []          # indices of selected features
    remaining       = list(range(n_features))
    best_score      = 0.0
    tol             = 1e-4

    while remaining:
        candidate_scores = {}
        for feat_idx in remaining:
            candidate = selected + [feat_idx]
            score = cross_val_score(
                base_svm, X[:, candidate], y,
                groups=groups, cv=cv,
                scoring="accuracy", n_jobs=-1
            ).mean()
            candidate_scores[feat_idx] = score

        best_candidate = max(candidate_scores, key=candidate_scores.get)
        best_candidate_score = candidate_scores[best_candidate]

        # Stop if improvement is below tolerance
        if best_candidate_score <= best_score + tol:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score
        print(f"     + {feat_names[best_candidate]:25s}  "
              f"→ CV acc = {best_score*100:.2f}%  "
              f"(total selected: {len(selected)})")

    sel_names = [feat_names[i] for i in selected]
    X_sel     = X[:, selected]

    print(f"  ✅ SFS selected {len(selected)} / {n_features} features:")
    for name in sel_names:
        print(f"     • {name}")

    return X_sel, sel_names, selected


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – 4-CLASS 10-FOLD GROUP CV
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray,
                         class_labels: list) -> dict:
    """
    10-fold GroupKFold CV with a Linear SVM (one-vs-rest multiclass).
    Groups = clip_ids → no clip leaks across folds.

    Returns accuracy, 4×4 confusion matrix, per-class & macro ROC-AUC.
    """
    n_clips     = len(np.unique(groups))
    outer_folds = min(CV_FOLDS, n_clips)
    print(f"\n  10-Fold GroupKFold CV  ({outer_folds} folds, {n_clips} clips)")
    print(f"  ℹ️  Clip-independent: YES  |  Subject-dependent: YES")

    svm_clf = SVC(kernel="linear", C=SVM_C,
                  decision_function_shape="ovr",
                  probability=True, random_state=Config.SEED)

    cv = GroupKFold(n_splits=outer_folds)

    cv_results = cross_validate(
        svm_clf, X, y,
        groups=groups, cv=cv,
        scoring=["accuracy"],
        return_train_score=False,
        n_jobs=-1,
    )

    y_pred = cross_val_predict(svm_clf, X, y,
                               groups=groups, cv=cv,
                               method="predict", n_jobs=-1)
    y_prob = cross_val_predict(svm_clf, X, y,
                               groups=groups, cv=cv,
                               method="predict_proba", n_jobs=-1)  # (N, 4)

    acc     = cv_results["test_accuracy"].mean()
    acc_std = cv_results["test_accuracy"].std()
    cm      = confusion_matrix(y, y_pred, labels=class_labels)

    # Binarise labels for OvR ROC-AUC
    y_bin       = label_binarize(y, classes=class_labels)
    roc_auc_per = {}
    for i, lbl in enumerate(class_labels):
        if y_bin[:, i].sum() > 0:
            roc_auc_per[lbl] = roc_auc_score(y_bin[:, i], y_prob[:, i])
    roc_auc_macro = roc_auc_score(y_bin, y_prob,
                                   multi_class="ovr", average="macro")

    print(f"\n  ✅ Accuracy   : {acc*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"     Macro AUC  : {roc_auc_macro:.4f}")
    for lbl, val in roc_auc_per.items():
        print(f"     AUC {lbl} ({CLASS_NAMES[lbl].split(chr(10))[0]:12s}): {val:.4f}")
    print(f"\n{classification_report(y, y_pred, labels=class_labels, digits=4,\
                                    target_names=[CLASS_NAMES[l].split(chr(10))[0] for l in class_labels])}")

    return {
        "accuracy"        : acc,
        "accuracy_std"    : acc_std,
        "fold_accuracies" : cv_results["test_accuracy"],
        "confusion_matrix": cm,
        "roc_auc_macro"   : roc_auc_macro,
        "roc_auc_per"     : roc_auc_per,
        "y_true"          : y,
        "y_pred"          : y_pred,
        "y_prob"          : y_prob,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_labels: list,
                          save_dir: str) -> None:
    """Save a 4×4 confusion matrix."""
    display_names = [CLASS_NAMES[l].replace("\n", "\n") for l in class_labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=display_names
                           ).plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("4-Class Confusion Matrix\n"
                 "(ERP + SFS + Linear SVM, GroupKFold CV)", fontsize=11)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "cm_4class.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix saved → {fpath}")


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_labels: list, save_dir: str) -> None:
    """Save one-vs-rest ROC curves for all 4 classes on one plot."""
    y_bin = label_binarize(y_true, classes=class_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["steelblue", "firebrick", "seagreen", "darkorange"]

    for i, (lbl, col) in enumerate(zip(class_labels, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        name        = CLASS_NAMES[lbl].split("\n")[0]
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves\n(4-Class ERP + SFS + Linear SVM)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(save_dir, "roc_4class.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  📈 ROC curves saved → {fpath}")


def plot_feature_importance(sel_names: list, X_sel: np.ndarray,
                            y: np.ndarray, class_labels: list,
                            save_dir: str) -> None:
    """
    Fit a Linear SVM on the full selected set.
    For multiclass OvR each class has its own weight vector;
    plot the mean |weight| across all classes per feature.
    """
    if not sel_names:
        return

    svm = SVC(kernel="linear", C=SVM_C,
              decision_function_shape="ovr",
              probability=False, random_state=Config.SEED)
    svm.fit(X_sel, y)

    # coef_ shape: (n_classes, n_features)  for OvR with >2 classes
    mean_weights = np.abs(svm.coef_).mean(axis=0)
    sorted_idx   = np.argsort(mean_weights)[::-1]

    fig, ax = plt.subplots(figsize=(max(6, len(sel_names) * 0.8), 4))
    ax.bar(range(len(sel_names)), mean_weights[sorted_idx], color="steelblue")
    ax.set_xticks(range(len(sel_names)))
    ax.set_xticklabels([sel_names[i] for i in sorted_idx],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SVM weight| across classes")
    ax.set_title("Feature Importance (Linear SVM, OvR mean)\n4-Class ERP Classification")
    plt.tight_layout()
    fpath = os.path.join(save_dir, "importance_4class.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  🔍 Feature importance saved → {fpath}")


def plot_erp_channel_heatmap(sel_names: list, save_dir: str) -> None:
    """
    Heatmap of selected ERP component × channel combinations.
    """
    erp_names = [e[0] for e in ERP_WINDOWS]
    n_erp, n_ch = len(erp_names), len(CHANNEL_NAMES)
    grid = np.zeros((n_erp, n_ch), dtype=int)

    for name in sel_names:
        parts    = name.split("_")
        erp_part = parts[0]
        ch_part  = parts[-1]
        if erp_part in erp_names and ch_part in CHANNEL_NAMES:
            grid[erp_names.index(erp_part)][CHANNEL_NAMES.index(ch_part)] = 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_ch));  ax.set_xticklabels(CHANNEL_NAMES, fontsize=10)
    ax.set_yticks(range(n_erp)); ax.set_yticklabels(erp_names, fontsize=10)
    ax.set_title("SFS-Selected ERP × Channel\n(4-Class Task)", fontsize=11)
    for r in range(n_erp):
        for c in range(n_ch):
            ax.text(c, r, "✓" if grid[r, c] else "·",
                    ha="center", va="center", fontsize=12,
                    color="white" if grid[r, c] else "lightgrey")
    plt.tight_layout()
    fpath = os.path.join(save_dir, "erp_selection_heatmap_4class.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  🗺️  ERP heatmap saved → {fpath}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ERP-BASED SVM  –  4-CLASS EMOTION RECOGNITION")
    print("=" * 70)
    print(f"  Classes         : {CLASS_LABELS}  "
          f"(Amusement, Anger, Sadness, Neutral)")
    print(f"  Dataset         : {config.DATA_ROOT}")
    print(f"  Leakage mode    : Clip-Independent (GroupKFold on clip_ids)")
    print(f"  Subject mode    : Subject-Dependent")
    print(f"  ERP windows     : {len(ERP_WINDOWS)}")
    print(f"  Channels        : {CHANNEL_NAMES}")
    print(f"  SVM C           : {SVM_C}  (one-vs-rest)")
    print(f"  CV folds        : {CV_FOLDS}  (outer GroupKFold)")
    print(f"  SFS inner folds : {SFS_CV}")
    print(f"  Results dir     : {RESULTS_DIR}")
    print("=" * 70)

    # ── 1. Load raw EEG windows ───────────────────────────────────────────────
    X_raw, y_int, subject_ids, label_to_id, clip_ids = load_eeg_data(
        config.DATA_ROOT, config
    )
    id_to_label = {v: k for k, v in label_to_id.items()}
    y_str = np.array([id_to_label[i] for i in y_int])

    # Keep only the four target classes
    valid_mask = np.isin(y_str, CLASS_LABELS)
    X_raw, y_str, subject_ids, clip_ids = (
        X_raw[valid_mask], y_str[valid_mask],
        subject_ids[valid_mask], clip_ids[valid_mask]
    )

    print(f"\n✅ Loaded : {X_raw.shape}  (windows × samples × channels)")
    print(f"   Labels : {Counter(y_str)}")
    print(f"   Clips  : {len(np.unique(clip_ids))} unique  "
          f"| Subjects: {len(np.unique(subject_ids))} unique")

    fs, T  = config.EEG_FS, X_raw.shape[1]
    win_ms = T / fs * 1000
    print(f"\n   Window : {win_ms:.0f} ms  ({T} samples @ {fs} Hz)")
    if win_ms < ERP_WINDOWS[-1][2]:
        print(f"   ⚠️  Window ({win_ms:.0f} ms) shorter than lateLPP end "
              f"({ERP_WINDOWS[-1][2]} ms) — late features will be clipped.")

    # ── 2. Extract ERP features ───────────────────────────────────────────────
    X_erp, feat_names = extract_erp_features(X_raw, fs=fs)
    print(f"\n  Feature matrix : {X_erp.shape}")
    print(f"  Feature names  : {feat_names}")

    # ── 3. Standardise ───────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_erp)

    # ── 4. Sequential Forward Selection ──────────────────────────────────────
    # Use integer-encoded labels for SFS (SVC requirement)
    label_enc   = {l: i for i, l in enumerate(CLASS_LABELS)}
    y_int_clean = np.array([label_enc[l] for l in y_str])

    X_sel, sel_names, _ = run_sfs(X_scaled, y_int_clean, clip_ids, feat_names)

    # ── 5. 10-Fold GroupKFold cross-validation ────────────────────────────────
    print(f"\n{'='*70}")
    print("4-CLASS CROSS-VALIDATION")
    print(f"{'='*70}")
    results = run_cross_validation(X_sel, y_int_clean, clip_ids,
                                   list(range(len(CLASS_LABELS))))

    # ── 6. Visualisations ─────────────────────────────────────────────────────
    plot_confusion_matrix(results["confusion_matrix"],
                          list(range(len(CLASS_LABELS))), RESULTS_DIR)
    plot_roc_curves(results["y_true"], results["y_prob"],
                    list(range(len(CLASS_LABELS))), RESULTS_DIR)
    plot_feature_importance(sel_names, X_sel,
                            y_int_clean, list(range(len(CLASS_LABELS))),
                            RESULTS_DIR)
    plot_erp_channel_heatmap(sel_names, RESULTS_DIR)

    # ── 7. Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy (10-fold CV) : {results['accuracy']*100:.2f}% "
          f"± {results['accuracy_std']*100:.2f}%")
    print(f"  Macro ROC-AUC        : {results['roc_auc_macro']:.4f}")
    print(f"  Per-class ROC-AUC    :")
    for i, lbl in enumerate(CLASS_LABELS):
        auc_val = results["roc_auc_per"].get(i, float("nan"))
        print(f"     {lbl} ({CLASS_NAMES[lbl].split(chr(10))[0]:12s}): {auc_val:.4f}")
    print(f"  Selected features ({len(sel_names)}) : {sel_names}")
    print(f"\n{'='*70}")
    print("✅ ERP-SVM 4-CLASS PIPELINE COMPLETE")
    print(f"   Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'='*70}")

    return results, sel_names


if __name__ == "__main__":
    main()
