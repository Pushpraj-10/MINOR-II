"""
Evaluate the exported GRU / BiLSTM TFLite models on EATD processed features.

Both segment-level and subject-level metrics are reported.

Usage:
    python main.py test-recurrent                  # test both GRU and BiLSTM
    python main.py test-recurrent --cell gru       # only GRU
    python main.py test-recurrent --cell bilstm    # only BiLSTM
    python main.py test-recurrent --threshold 0.4  # custom decision threshold
    python main.py test-recurrent --sweep          # find best threshold on val
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, classification_report,
)

PROCESSED_DIR = "data/processed/EATD"
MODEL_ROOTS = {
    "gru": "artifacts/models/recurrent_eatd_gru/gru_best.tflite",
    "bilstm": "artifacts/models/recurrent_eatd_bilstm/bilstm_best.tflite",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_split(split_name: str):
    """
    Load a processed split and return X (N,313,46), y, subject_ids.
    Features are stored as (N,46,313) → transposed to (N,313,46) for
    the recurrent input format.
    """
    path = os.path.join(PROCESSED_DIR, f"{split_name}_features.npz")
    data = np.load(path)
    X = np.concatenate([
        data["X_mfcc"],
        data["X_delta_mfcc"],
        data["X_chroma"],
        data["X_spectral_contrast"],
        data["X_zcr"],
    ], axis=1)                       # (N, 46, 313)
    X = np.transpose(X, (0, 2, 1))  # (N, 313, 46)
    X = X.astype(np.float32)
    y = data["y"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(np.int32)
    return X, y, subject_ids


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def _run_tflite(tflite_path: str, X: np.ndarray) -> np.ndarray:
    """Run batch inference through a TFLite model (flex delegate aware)."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    preds = np.empty(len(X), dtype=np.float32)
    for i, sample in enumerate(X):
        interp.set_tensor(in_idx, sample[np.newaxis])  # (1,313,46)
        interp.invoke()
        preds[i] = interp.get_tensor(out_idx)[0, 0]
    return preds


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _seg_metrics(y_true, y_pred, threshold):
    y_bin = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float((y_bin == y_true.astype(int)).mean())
    return auc, acc, y_bin


def _subject_metrics(y_true_seg, y_pred_seg, subject_ids, threshold):
    unique_ids = np.unique(subject_ids)
    y_true_subj, y_score_subj = [], []
    for sid in unique_ids:
        mask = subject_ids == sid
        y_true_subj.append(int(np.round(y_true_seg[mask].mean())))
        y_score_subj.append(float(y_pred_seg[mask].mean()))
    y_true_subj = np.array(y_true_subj)
    y_score_subj = np.array(y_score_subj)
    y_bin_subj = (y_score_subj >= threshold).astype(int)

    auc = roc_auc_score(y_true_subj, y_score_subj) if len(np.unique(y_true_subj)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true_subj, y_bin_subj, labels=[0, 1]).ravel()
    n = len(y_true_subj)
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true_subj, y_bin_subj, zero_division=0)
    return {
        "n": n, "dep": int(y_true_subj.sum()), "norm": n - int(y_true_subj.sum()),
        "auc": auc, "accuracy": accuracy,
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _print_subject_report(split_name, m):
    print(f"\n{'='*58}")
    print(f"  Subject-level — {split_name}  ({m['n']} subjects: dep={m['dep']} norm={m['norm']})")
    print(f"{'='*58}")
    print(f"  Confusion:   TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  TP={m['tp']}")
    print(f"  AUC:         {m['auc']:.4f}")
    print(f"  Accuracy:    {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Sensitivity: {m['sensitivity']:.4f}  (depression recall)")
    print(f"  Specificity: {m['specificity']:.4f}  (normal recall)")
    print(f"  PPV:         {m['ppv']:.4f}  (depression precision)")
    print(f"  F1:          {m['f1']:.4f}")


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(cell: str, tflite_path: str, threshold: float, sweep: bool):
    print(f"\n{'#'*60}")
    print(f"  Model: {cell.upper()}  —  {tflite_path}")
    print(f"{'#'*60}")

    if not os.path.exists(tflite_path):
        print(f"  [SKIP] TFLite file not found: {tflite_path}")
        return

    print("\nLoading features...")
    X_val, y_val, ids_val = _load_split("val")
    X_test, y_test, ids_test = _load_split("test")
    print(f"  Val  : {len(y_val)} segs  (dep={int(y_val.sum())})")
    print(f"  Test : {len(y_test)} segs  (dep={int(y_test.sum())})")

    print("\nRunning TFLite inference (val)...")
    val_preds = _run_tflite(tflite_path, X_val)
    print("Running TFLite inference (test)...")
    test_preds = _run_tflite(tflite_path, X_test)

    # Optional threshold sweep on val (subject level)
    if sweep:
        print("\n--- Threshold Sweep (validation subjects) ---")
        best_thresh, best_f1 = 0.5, -1.0
        for t in np.arange(0.10, 0.95, 0.05):
            m = _subject_metrics(y_val, val_preds, ids_val, t)
            dep_recall = m["sensitivity"]
            print(f"  t={t:.2f}  AUC={m['auc']:.3f}  F1={m['f1']:.3f}"
                  f"  Sens={dep_recall:.3f}  Spec={m['specificity']:.3f}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thresh = t
        threshold = best_thresh
        print(f"\n  Best threshold (by val F1): {threshold:.2f}")

    print(f"\nThreshold applied: {threshold:.2f}")

    # --- Segment-level ---
    val_seg_auc, val_seg_acc, _ = _seg_metrics(y_val, val_preds, threshold)
    test_seg_auc, test_seg_acc, _ = _seg_metrics(y_test, test_preds, threshold)

    print(f"\n{'='*58}")
    print(f"  Segment-level")
    print(f"{'='*58}")
    print(f"  {'Split':<8}  {'AUC':>7}  {'Acc':>7}")
    print(f"  {'val':<8}  {val_seg_auc:>7.4f}  {val_seg_acc:>7.4f}")
    print(f"  {'test':<8}  {test_seg_auc:>7.4f}  {test_seg_acc:>7.4f}")

    # --- Subject-level ---
    val_m = _subject_metrics(y_val, val_preds, ids_val, threshold)
    test_m = _subject_metrics(y_test, test_preds, ids_test, threshold)
    _print_subject_report("val", val_m)
    _print_subject_report("test", test_m)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test recurrent TFLite models on EATD data")
    parser.add_argument("--cell", choices=["gru", "bilstm", "both"], default="both",
                        help="Which model to evaluate (default: both)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds 0.10–0.90 on val set to find best")
    args = parser.parse_args()

    cells = ["gru", "bilstm"] if args.cell == "both" else [args.cell]
    for cell in cells:
        evaluate_model(cell, MODEL_ROOTS[cell], args.threshold, args.sweep)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
