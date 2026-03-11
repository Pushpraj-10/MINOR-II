"""
Subject-level depression detection evaluation.

Segment-level predictions from the CNN are aggregated per subject by
averaging, then a threshold is applied to produce a single binary
prediction per subject.  This mirrors how real clinical screening tools
work: a subject is flagged only when a majority of their speech segments
score above threshold.

Usage:
    python main.py evaluate-subjects           # uses default threshold 0.5
    python main.py evaluate-subjects --threshold 0.4
    python main.py evaluate-subjects --sweep   # sweep thresholds, report best
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score,
)

from src.config import MODEL_DIR
from src.utils.focal_loss import focal_loss

PROCESSED_DIR = "data/processed/EATD"


def _load_and_stack(split_name: str):
    """Load a split .npz and stack all features along the frequency axis."""
    path = os.path.join(PROCESSED_DIR, f"{split_name}_features.npz")
    data = np.load(path)
    X = np.concatenate([
        data["X_mfcc"],
        data["X_delta_mfcc"],
        data["X_chroma"],
        data["X_spectral_contrast"],
        data["X_zcr"],
    ], axis=1)
    y = data["y"]
    subject_ids = data["subject_ids"]
    return X, y, subject_ids


def aggregate_by_subject(
    y_true_seg: np.ndarray,
    y_pred_seg: np.ndarray,
    subject_ids: np.ndarray,
    threshold: float = 0.5,
):
    """
    Average segment-level scores per subject and threshold to get
    subject-level binary predictions.

    Returns:
        y_true_subj, y_pred_score_subj, y_pred_binary_subj, unique_subject_ids
    """
    unique_ids = np.unique(subject_ids)
    y_true_subj, y_score_subj = [], []
    for sid in unique_ids:
        mask = subject_ids == sid
        # Subject label is the majority label of its segments (should be uniform)
        label = int(np.round(y_true_seg[mask].mean()))
        score = float(y_pred_seg[mask].mean())
        y_true_subj.append(label)
        y_score_subj.append(score)

    y_true_subj = np.array(y_true_subj)
    y_score_subj = np.array(y_score_subj)
    y_pred_binary = (y_score_subj >= threshold).astype(int)
    return y_true_subj, y_score_subj, y_pred_binary, unique_ids


def print_subject_summary(
    split_name: str,
    y_true: np.ndarray,
    y_pred_score: np.ndarray,
    y_pred_binary: np.ndarray,
):
    n_dep = int(y_true.sum())
    n_norm = len(y_true) - n_dep
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    print(f"\n{'='*60}")
    print(f"Subject-Level Results — {split_name} ({len(y_true)} subjects)")
    print(f"{'='*60}")
    print(f"  Subjects: dep={n_dep}  norm={n_norm}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    if n_dep > 0 and n_dep < len(y_true):
        auc = roc_auc_score(y_true, y_pred_score)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        accuracy = (tp + tn) / len(y_true)
        print(f"  AUC:         {auc:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}  ({accuracy*100:.1f}%)")
        print(f"  Sensitivity: {sensitivity:.4f}  (dep recall)")
        print(f"  Specificity: {specificity:.4f}  (norm recall)")
        print(f"  PPV:         {ppv:.4f}  (dep precision)")
        print(f"  F1:          {f1:.4f}")
    else:
        print("  (cannot compute AUC — only one class present)")


def main():
    parser = argparse.ArgumentParser(
        description="Subject-level evaluation with prediction aggregation"
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for subject-level classification")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds 0.1–0.9 and report best on val, apply to test")
    parser.add_argument("--model", type=str,
                        default=os.path.join(MODEL_DIR, "multi_feature_eatd", "multi_feature_best.keras"),
                        help="Path to trained .keras model file")
    args = parser.parse_args()

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = tf.keras.models.load_model(
            args.model,
            custom_objects={"focal_loss": focal_loss()},
        )
    except Exception:
        model = tf.keras.models.load_model(args.model)
    print("  Model loaded.")

    # Load features
    print("\nLoading processed features...")
    X_val, y_val_seg, ids_val = _load_and_stack("val")
    X_test, y_test_seg, ids_test = _load_and_stack("test")
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print(f"  Val  segments: {len(y_val_seg)}  (dep={int(y_val_seg.sum())})")
    print(f"  Test segments: {len(y_test_seg)}  (dep={int(y_test_seg.sum())})")

    # Inference
    print("\nRunning inference...")
    val_preds = model.predict(X_val, verbose=0).squeeze()
    test_preds = model.predict(X_test, verbose=0).squeeze()

    # Threshold selection
    if args.sweep:
        print("\n--- Threshold Sweep (on Validation subjects) ---")
        best_thresh, best_f1 = 0.5, -1.0
        for t in np.arange(0.1, 0.95, 0.05):
            y_tv, y_sv, y_bv, _ = aggregate_by_subject(y_val_seg, val_preds, ids_val, t)
            if len(np.unique(y_bv)) < 2:
                continue
            f = f1_score(y_tv, y_bv, zero_division=0)
            n_dep = int(y_tv.sum())
            tp = int(((y_bv == 1) & (y_tv == 1)).sum())
            fp = int(((y_bv == 1) & (y_tv == 0)).sum())
            print(f"  t={t:.2f}  val_f1={f:.3f}  dep_tp={tp}/{n_dep}  fp={fp}")
            if f > best_f1:
                best_f1, best_thresh = f, t
        print(f"\nBest threshold on val: {best_thresh:.2f}  (f1={best_f1:.3f})")
        threshold = best_thresh
    else:
        threshold = args.threshold

    # Subject-level evaluation
    y_true_val, y_score_val, y_pred_val, _ = aggregate_by_subject(
        y_val_seg, val_preds, ids_val, threshold
    )
    y_true_test, y_score_test, y_pred_test, _ = aggregate_by_subject(
        y_test_seg, test_preds, ids_test, threshold
    )

    print(f"\nUsing threshold: {threshold:.2f}")
    print_subject_summary("Validation", y_true_val, y_score_val, y_pred_val)
    print_subject_summary("Test (unseen speakers)", y_true_test, y_score_test, y_pred_test)

    print("\n--- Classification Report (Test, subject-level) ---")
    print(classification_report(
        y_true_test, y_pred_test,
        target_names=["Normal", "Depression"], zero_division=0,
    ))


if __name__ == "__main__":
    main()
