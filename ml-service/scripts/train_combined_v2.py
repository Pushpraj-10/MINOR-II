"""
Train v2 generalized depression-detection model.

Improvements over train_combined.py:
  - Loads from data/processed/combined_v2/  (CMVN + speaker-disjoint val)
  - Subject-level evaluation on EATD test speakers (aggregate per-segment
    predictions → true speaker-level AUC, the fairest metric for EATD)
  - Optional Mixup feature-space augmentation  (--mixup flag)
    Creates interpolated virtual training samples between pairs of examples,
    reducing over-reliance on individual speaker patterns.

Architecture: same 3-block CNN as multi_feature_combined (101K params).
Output model: artifacts/models/multi_feature_v2/
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import evaluate_model
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE, MODEL_DIR

logger = logging.getLogger(__name__)

COMBINED_DIR = os.path.join("data", "processed", "combined_v2")
MODEL_NAME   = "multi_feature_v2"


# ──────────────────────────────────────────────────────────────────────────────
# Mixup augmentation
# ──────────────────────────────────────────────────────────────────────────────

def apply_mixup(X: np.ndarray, y: np.ndarray,
                alpha: float = 0.3, frac: float = 0.25,
                rng=None) -> tuple:
    """
    Offline Mixup: interpolate pairs of training samples.

    Creates frac*N synthetic samples by linearly mixing feature maps and
    labels with a Beta(alpha, alpha) weight.  Mixed examples are appended
    to the original data and the combined array is shuffled.

    Soft labels from mixing are fully compatible with BinaryCrossentropy.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(X)
    n_mix = int(n * frac)
    idx1 = rng.integers(0, n, n_mix)
    idx2 = rng.integers(0, n, n_mix)
    lam  = rng.beta(alpha, alpha, n_mix).astype(np.float32)
    lam_x = lam[:, np.newaxis, np.newaxis, np.newaxis]   # broadcast over H,W,C
    X_mix = lam_x * X[idx1] + (1 - lam_x) * X[idx2]
    y_mix = lam  * y[idx1]  + (1 - lam)   * y[idx2]
    perm = rng.permutation(n + n_mix)
    return np.concatenate([X, X_mix])[perm], np.concatenate([y, y_mix])[perm]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_shape, dropout=DROPOUT_RATE) -> keras.Model:
    """
    3-block CNN identical to multi_feature_combined (~101K params).

    Input shape: (46, 313, 1)
      Block 1: Conv2D(32) + BN + MaxPool + Dropout(0.4)
      Block 2: Conv2D(64) + BN + MaxPool + Dropout(0.5)
      Block 3: Conv2D(128) + BN + GlobalAveragePool
      Head:    Dense(64) + Dropout(0.6) → Dense(1, sigmoid)
    """
    l2 = keras.regularizers.l2(1e-4)
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.6)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, out, name=MODEL_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_split(name: str):
    data = np.load(os.path.join(COMBINED_DIR, f"{name}_features.npz"))
    return data["X"], data["y"]


def evaluate_subject_level(y_true: np.ndarray, y_proba: np.ndarray,
                            subject_ids: np.ndarray):
    """
    Aggregate per-segment sigmoid scores to subject level and compute AUC.

    For each unique EATD subject ID, average all segment probabilities.
    DS1 samples (tagged "DS1") are excluded.

    Returns:
        (subject_auc, n_subj_dep, n_subj_norm, subj_y, subj_p)
    """
    eatd_mask  = subject_ids != "DS1"
    unique_sids = np.unique(subject_ids[eatd_mask])

    subj_y, subj_p = [], []
    for sid in unique_sids:
        mask = subject_ids == sid
        subj_y.append(float(y_true[mask][0]))          # all segments same label
        subj_p.append(float(y_proba[mask].mean()))     # mean segment probability
    subj_y = np.array(subj_y)
    subj_p = np.array(subj_p)

    if len(np.unique(subj_y)) < 2:
        return None, int((subj_y==1).sum()), int((subj_y==0).sum()), subj_y, subj_p

    auc = roc_auc_score(subj_y, subj_p)
    return auc, int((subj_y==1).sum()), int((subj_y==0).sum()), subj_y, subj_p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train v2 model (CMVN + speaker-disjoint val)"
    )
    parser.add_argument("--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--dropout",         type=float, default=DROPOUT_RATE)
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing factor (0 = no smoothing)")
    parser.add_argument("--mixup",           action="store_true",
                        help="Apply offline Mixup augmentation to training set")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    arch_dir = os.path.join(MODEL_DIR, MODEL_NAME)
    os.makedirs(arch_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading combined_v2 data...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    rav  = np.load(os.path.join(COMBINED_DIR, "ravdess_features.npz"))
    X_rav, y_rav = rav["X"], rav["y"]

    sid_data  = np.load(os.path.join(COMBINED_DIR, "test_subject_ids.npz"),
                        allow_pickle=True)
    test_sids = sid_data["subject_ids"]

    # Add channel dim: (N, 46, 313) → (N, 46, 313, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[...,   np.newaxis]
    X_test  = X_test[...,  np.newaxis]
    X_rav   = X_rav[...,   np.newaxis]

    print(f"  Train:   {X_train.shape}  dep={int(y_train.sum())} "
          f"norm={int((y_train==0).sum())}")
    print(f"  Val:     {X_val.shape}   dep={int(y_val.sum())} "
          f"norm={int((y_val==0).sum())}")
    print(f"  Test:    {X_test.shape}  dep={int(y_test.sum())} "
          f"norm={int((y_test==0).sum())}")
    print(f"  RAVDESS: {X_rav.shape}")

    # ── 2. Optional Mixup ────────────────────────────────────────────────────
    if args.mixup:
        print("\n  Applying offline Mixup augmentation (alpha=0.3, frac=0.25)...")
        X_train, y_train = apply_mixup(X_train, y_train, alpha=0.3, frac=0.25)
        print(f"  Train after Mixup: {X_train.shape}")

    # ── 3. Build model ────────────────────────────────────────────────────────
    meta = np.load(os.path.join(COMBINED_DIR, "metadata.npz"), allow_pickle=True)
    pre_dep  = int(meta["pre_aug_train_dep"])
    pre_norm = int(meta["pre_aug_train_norm"])
    pre_n    = pre_dep + pre_norm
    class_weights = {0: pre_n / (2 * pre_norm), 1: pre_n / (2 * pre_dep)}
    print(f"\n  Class weights (pre-aug): {class_weights}")

    input_shape = X_train.shape[1:]
    print(f"\n[2/5] Building model (input={input_shape})...")
    model = build_model(input_shape, dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    best_path = os.path.join(arch_dir, "best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=20, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=7,
            min_lr=1e-6, mode="max", verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    print("\n[3/5] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )
    model.save(os.path.join(arch_dir, "final.keras"))

    # ── 5. Threshold optimisation ────────────────────────────────────────────
    print("\n[4/5] Optimising decision threshold (Youden's J on val)...")
    val_proba = model.predict(X_val, verbose=0).flatten()
    fpr, tpr, thresholds = roc_curve(y_val, val_proba)
    best_idx = int(np.argmax(tpr - fpr))
    opt_thr  = float(thresholds[best_idx])
    print(f"  Youden's J threshold: {opt_thr:.4f} "
          f"(TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})")

    # ── 6. Evaluate all datasets ─────────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    datasets = [
        ("Train",              X_train, y_train, None),
        ("Validation",         X_val,   y_val,   None),
        ("Test (EATD+DS1)",    X_test,  y_test,  test_sids),
        ("RAVDESS",            X_rav,   y_rav,   None),
    ]

    results = {}
    for thr_name, thr in [("default_0.5", 0.5), ("optimized", opt_thr)]:
        print(f"\n{'#'*60}")
        print(f"  Results at threshold={thr:.4f}  ({thr_name})")
        print(f"{'#'*60}")
        results[thr_name] = {}

        for name, X, y, sids in datasets:
            proba   = model.predict(X, verbose=0).flatten()
            metrics = evaluate_model(y, proba.reshape(-1, 1), threshold=thr)
            cm      = confusion_matrix(y.flatten(), (proba >= thr).astype(int))

            print(f"\n{'='*50}")
            print(f"  {name} ({len(y)} samples)")
            print(f"{'='*50}")
            print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"  AUC:       {metrics['roc_auc']*100:.2f}%")
            print(f"  F1:        {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

            results[thr_name][name] = {
                "accuracy":  round(float(metrics["accuracy"]),  4),
                "auc":       round(float(metrics["roc_auc"]),    4),
                "f1":        round(float(metrics["f1_score"]),   4),
                "precision": round(float(metrics["precision"]),  4),
                "recall":    round(float(metrics["recall"]),     4),
            }

            # Subject-level evaluation (EATD test only, at threshold 0.5)
            if sids is not None and thr_name == "default_0.5":
                subj_auc, n_dep, n_norm, subj_y, subj_p = \
                    evaluate_subject_level(y, proba, sids)
                n_subj = n_dep + n_norm
                print(f"\n  ── EATD Subject-level Evaluation ({n_subj} speakers: "
                      f"dep={n_dep}, norm={n_norm}) ──")
                if subj_auc is not None:
                    print(f"  Subject-level AUC:      {subj_auc*100:.2f}%")
                    fpr_s, tpr_s, thr_s = roc_curve(subj_y, subj_p)
                    j_idx = int(np.argmax(tpr_s - fpr_s))
                    subj_thr = float(thr_s[j_idx])
                    subj_preds = (subj_p >= 0.5).astype(int)
                    subj_acc = float((subj_preds == subj_y).mean())
                    print(f"  Subject-level Acc @0.5: {subj_acc*100:.2f}%")
                    print(f"  (Youden's J subj_thr:   {subj_thr:.4f})")
                    results["default_0.5"]["EATD_subject_level"] = {
                        "n_subjects": n_subj,
                        "auc": round(subj_auc, 4),
                        "accuracy_at_0.5": round(subj_acc, 4),
                    }
                else:
                    print("  Only one class in subject-level labels, AUC undefined.")

    # ── TFLite export ─────────────────────────────────────────────────────────
    print("\n  Exporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path  = os.path.join(arch_dir, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  Saved: {tflite_path}  ({size_kb:.1f} KB)")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "model_name":        MODEL_NAME,
        "input_shape":       list(X_train.shape[1:]),
        "optimal_threshold": opt_thr,
        "class_weights":     class_weights,
        "training": {
            "epochs_run":    len(history.history["loss"]),
            "best_val_auc":  float(max(history.history["val_auc"])),
        },
        "results":      results,
        "tflite_path":  tflite_path,
        "tflite_size_kb": round(size_kb, 1),
    }
    summary_path = os.path.join(arch_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")
    print("\n=== v2 training complete ===\n")


if __name__ == "__main__":
    main()
