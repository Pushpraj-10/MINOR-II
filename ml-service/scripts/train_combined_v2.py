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

    Mixup is a data augmentation trick:
    instead of feeding the model a real sample, we blend two samples together
    (e.g. 70% of sample A + 30% of sample B) and give it a blended label too.
    This forces the model to behave smoothly between examples, reducing
    overconfidence on specific training voices.

    Creates frac*N synthetic samples by linearly mixing feature maps and
    labels with a Beta(alpha, alpha) weight.  Mixed examples are appended
    to the original data and the combined array is shuffled.

    Soft labels from mixing are fully compatible with BinaryCrossentropy.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(X)
    n_mix = int(n * frac)                            # how many blended samples to create
    idx1 = rng.integers(0, n, n_mix)                 # pick n_mix random sample indices
    idx2 = rng.integers(0, n, n_mix)                 # pick another n_mix random indices to blend with
    lam  = rng.beta(alpha, alpha, n_mix).astype(np.float32)  # blend ratio per pair, drawn from Beta distribution
    lam_x = lam[:, np.newaxis, np.newaxis, np.newaxis]   # reshape so it can multiply (H, W, C) tensors
    X_mix = lam_x * X[idx1] + (1 - lam_x) * X[idx2]    # blended feature grid
    y_mix = lam  * y[idx1]  + (1 - lam)   * y[idx2]     # blended label (e.g. 0.7 depressed)
    perm = rng.permutation(n + n_mix)                    # shuffle real + blended together
    return np.concatenate([X, X_mix])[perm], np.concatenate([y, y_mix])[perm]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_shape, dropout=DROPOUT_RATE) -> keras.Model:
    """
    3-block CNN identical to multi_feature_combined (~101K params).

    The model treats the 46×313 feature grid like an image.
    Convolutional layers detect local patterns (e.g. a specific MFCC shape
    at a specific time), deeper layers combine those into higher-level
    patterns (e.g. sustained monotone speech across the full 5 seconds).

    Input shape: (46, 313, 1)
      Block 1: Conv2D(32) + BN + MaxPool + Dropout(0.4)
      Block 2: Conv2D(64) + BN + MaxPool + Dropout(0.5)
      Block 3: Conv2D(128) + BN + GlobalAveragePool
      Head:    Dense(64) + Dropout(0.6) → Dense(1, sigmoid)
    """
    # L2 regularization penalizes large weights — prevents the model from
    # memorizing training speakers instead of learning generalizable patterns
    l2 = keras.regularizers.l2(1e-4)
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs

    # ── Block 1: Detect simple local patterns (edges in the feature map) ──
    # Conv2D scans the grid with 32 different 3×3 filters
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)   # normalize activations → stable training
    x = keras.layers.MaxPooling2D((2, 2))(x)   # halve spatial size → (23, 156, 32)
    x = keras.layers.Dropout(0.4)(x)           # randomly zero 40% of neurons each step

    # ── Block 2: Detect more complex patterns (combinations of Block 1 features) ──
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)   # halve again → (11, 78, 64)
    x = keras.layers.Dropout(0.5)(x)

    # ── Block 3: High-level abstraction across the full feature map ──
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                            kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    # GlobalAveragePooling: collapses the entire spatial map into a single
    # 128-number vector — one number summarizing each filter's average response
    x = keras.layers.GlobalAveragePooling2D()(x)   # → (128,)

    # ── Classification head: maps 128 features → single depression probability ──
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.6)(x)  # heavy dropout here — most overfitting happens at the head
    # sigmoid outputs a number between 0 and 1
    # values > 0.5 → depressed, values <= 0.5 → normal
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
    # These .npz files were created by process_combined_v2.py
    # Each file contains X (features) and y (labels: 1=depressed, 0=normal)
    # X shape: (N_samples, 46_feature_rows, 313_time_columns)
    print("\n[1/5] Loading combined_v2 data...")
    X_train, y_train = load_split("train")  # data the model learns from
    X_val,   y_val   = load_split("val")    # data used to check progress during training (not learned from)
    X_test,  y_test  = load_split("test")   # data held back until the very end for final score

    rav  = np.load(os.path.join(COMBINED_DIR, "ravdess_features.npz"))
    X_rav, y_rav = rav["X"], rav["y"]       # RAVDESS: completely different dataset, tests generalization

    sid_data  = np.load(os.path.join(COMBINED_DIR, "test_subject_ids.npz"),
                        allow_pickle=True)
    test_sids = sid_data["subject_ids"]     # which EATD speaker each test segment belongs to

    # CNNs expect a channel dimension — same as color images have RGB channels
    # Our features are grayscale so we just add 1 channel: (N, 46, 313) → (N, 46, 313, 1)
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
    # Class weights tell the model to care more about depressed samples,
    # because there are far fewer of them in the dataset (~16% depressed vs ~84% normal).
    # Without this, the model could just predict "normal" for everything and still
    # get 84% accuracy — which is useless for us.
    # We use pre-augmentation counts so augmented copies don't get double-counted.
    meta = np.load(os.path.join(COMBINED_DIR, "metadata.npz"), allow_pickle=True)
    pre_dep  = int(meta["pre_aug_train_dep"])   # original depressed sample count (before augmentation)
    pre_norm = int(meta["pre_aug_train_norm"])  # original normal sample count
    pre_n    = pre_dep + pre_norm
    # Weight formula: total / (2 × class_count) — rarer class gets higher weight
    class_weights = {0: pre_n / (2 * pre_norm), 1: pre_n / (2 * pre_dep)}
    print(f"\n  Class weights (pre-aug): {class_weights}")

    input_shape = X_train.shape[1:]
    print(f"\n[2/5] Building model (input={input_shape})...")
    model = build_model(input_shape, dropout=args.dropout)
    model.compile(
        # Adam adjusts the learning rate automatically per parameter (smarter than plain SGD)
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        # BinaryCrossentropy is the standard loss for binary classification.
        # label_smoothing=0.05: instead of hard labels {0, 1}, use {0.025, 0.975}.
        # This prevents the model from becoming overconfident on training samples.
        loss=keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        # Track accuracy and AUC during training so we can see progress each epoch.
        # AUC is a better metric than accuracy here because of class imbalance.
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    best_path = os.path.join(arch_dir, "best.keras")
    callbacks = [
        # EarlyStopping: watch val_auc (AUC on validation set).
        # If it doesn't improve for 20 epochs, stop — and reload the best weights seen.
        # This prevents overfitting: the model stops when it starts memorizing training data.
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=20, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        # ReduceLROnPlateau: if val_auc is stuck for 7 epochs, cut learning rate in half.
        # Smaller steps help the model escape local plateaus near the end of training.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=7,
            min_lr=1e-6, mode="max", verbose=1,
        ),
        # ModelCheckpoint: every time val_auc reaches a new best, save the model.
        # This means best.keras always holds the best model, not the last one.
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
    # The model always outputs a probability (0–1) rather than a hard label.
    # We decide: above what probability do we call it "depressed"?
    # Default = 0.5, but that may not be the best cut-off for this specific model.
    #
    # Youden's J = TPR - FPR  (maximized when we correctly catch the most
    # depressed cases while minimizing false alarms on healthy people).
    # We find the best threshold on the validation set.
    print("\n[4/5] Optimising decision threshold (Youden's J on val)...")
    val_proba = model.predict(X_val, verbose=0).flatten()  # raw probabilities for all val samples
    fpr, tpr, thresholds = roc_curve(y_val, val_proba)     # sweep all possible thresholds
    best_idx = int(np.argmax(tpr - fpr))                   # index where Youden's J is maximum
    opt_thr  = float(thresholds[best_idx])                 # the optimal threshold value
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
    # TFLite converts the Keras model to a format runnable on Android/iOS.
    # Optimize.DEFAULT applies dynamic range quantization:
    # it converts float32 weights to int8, reducing file size ~4× with minimal accuracy loss.
    # Note: this model exports the CNN weights only — the feature extraction
    # (STFT, MFCC etc.) is NOT baked in here.  The combined model with preprocessing
    # is built separately in src/export/tflite_converter.py.
    print("\n  Exporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enable quantization
    tflite_model = converter.convert()                     # do the conversion
    tflite_path  = os.path.join(arch_dir, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)                              # write binary to disk
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
