"""
Train a recurrent (GRU / BiLSTM) model on pre-processed multi-feature EATD data.

Architecture:
    Audio → Feature extraction (46 features × 313 steps)
         → GRU or BiLSTM
         → Dense layer
         → Depression probability (sigmoid)

Usage:
    python main.py train-recurrent                   # default: BiLSTM
    python main.py train-recurrent --cell gru         # GRU variant
    python main.py train-recurrent --cell bilstm      # BiLSTM variant
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.evaluation.evaluator import print_split_summary
from src.export.tflite_converter import convert_to_tflite
from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    RANDOM_STATE, MODEL_DIR,
)

logger = logging.getLogger(__name__)
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
    ], axis=1)  # (N, 46, 313)
    y = data["y"]
    return X, y


class _Attention(keras.layers.Layer):
    """Simple additive attention over time steps → weighted context vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(int(input_shape[-1]), 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(int(input_shape[1]), 1),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        # x: (batch, time, features)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, time, 1)
        a = tf.nn.softmax(e, axis=1)                     # (batch, time, 1)
        return tf.reduce_sum(x * a, axis=1)               # (batch, features)


def _build_model(input_shape, cell_type="bilstm", dropout_rate=DROPOUT_RATE) -> keras.Model:
    """Lightweight Conv1D + recurrent model for depression detection.

    Conv1D compresses 313 steps → ~79 steps, then a single small RNN layer
    with attention pooling. Strongly regularized to prevent overfitting
    on ~1500 training samples.

    Input: (time_steps=313, features=46)
    """
    l2 = keras.regularizers.l2(1e-3)
    inp = keras.layers.Input(shape=input_shape)

    # Normalize input features
    x = keras.layers.LayerNormalization()(inp)

    # Conv1D front-end: 313 → ~79 steps
    x = keras.layers.Conv1D(32, kernel_size=5, strides=4, activation="relu",
                            padding="same", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.4)(x)

    if cell_type == "bilstm":
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(24, return_sequences=True, kernel_regularizer=l2)
        )(x)
    elif cell_type == "gru":
        x = keras.layers.GRU(32, return_sequences=True, kernel_regularizer=l2)(x)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")

    x = keras.layers.Dropout(0.5)(x)

    # Attention pooling over time
    x = _Attention(name="attention")(x)

    x = keras.layers.Dense(16, activation="relu", kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)

    name = f"depression_{cell_type}"
    return keras.Model(inputs=inp, outputs=out, name=name)


def main():
    parser = argparse.ArgumentParser(
        description="Train recurrent (GRU/BiLSTM) model on EATD data"
    )
    parser.add_argument("--cell", type=str, default="bilstm",
                        choices=["gru", "bilstm"],
                        help="Recurrent cell type (default: bilstm)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    arch_dir = os.path.join(MODEL_DIR, f"recurrent_eatd_{args.cell}")
    os.makedirs(arch_dir, exist_ok=True)

    # ── 1. Load processed features ──────────────────────────────────
    print(f"\n[1/4] Loading processed multi-feature data...")
    X_train, y_train = _load_and_stack("train")
    X_val, y_val = _load_and_stack("val")
    X_test, y_test = _load_and_stack("test")

    # Transpose: (N, 46, 313) → (N, 313, 46) — sequence of 313 steps, 46 features each
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print(f"  Train: {X_train.shape}  (dep={int(y_train.sum())}  norm={int((y_train==0).sum())})")
    print(f"  Val:   {X_val.shape}  (dep={int(y_val.sum())}  norm={int((y_val==0).sum())})")
    print(f"  Test:  {X_test.shape}  (dep={int(y_test.sum())}  norm={int((y_test==0).sum())})")

    # Pre-augmentation class weights
    meta_path = os.path.join(PROCESSED_DIR, "feature_metadata.npz")
    meta = np.load(meta_path, allow_pickle=True)
    pre_dep = int(meta["pre_aug_train_dep"])
    pre_norm = int(meta["pre_aug_train_norm"])
    pre_n = pre_dep + pre_norm
    class_weights = {0: pre_n / (2 * pre_norm), 1: pre_n / (2 * pre_dep)}
    print(f"  Class weights (pre-augmentation): {class_weights}")

    # ── 2. Build model ──────────────────────────────────────────────
    print(f"\n[2/4] Building {args.cell.upper()} model (input={X_train.shape[1:]})...")
    model = _build_model(X_train.shape[1:], cell_type=args.cell, dropout_rate=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # ── 3. Train ────────────────────────────────────────────────────
    best_path = os.path.join(arch_dir, f"{args.cell}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            mode="min", restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, mode="min", verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_loss",
            mode="min", save_best_only=True, verbose=1,
        ),
    ]

    print(f"\n[3/4] Training {args.cell.upper()} (BCE + pre-aug class weights)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )

    final_path = os.path.join(arch_dir, f"{args.cell}_final.keras")
    model.save(final_path)
    print(f"\n  Best model:  {best_path}")
    print(f"  Final model: {final_path}")

    # ── 4. Evaluate ─────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating {args.cell.upper()}...")
    print_split_summary(y_train, model.predict(X_train, verbose=0), "Train")
    print_split_summary(y_val, model.predict(X_val, verbose=0), "Validation")
    print_split_summary(y_test, model.predict(X_test, verbose=0), "Test (unseen speakers)")

    # ── 5. Export TFLite ────────────────────────────────────────────
    print(f"\n[5/5] Exporting TFLite...")
    # Reload best checkpoint for export
    best_model = keras.models.load_model(
        best_path,
        custom_objects={"_Attention": _Attention},
    )
    tflite_path = os.path.join(arch_dir, f"{args.cell}_best.tflite")
    convert_to_tflite(best_model, tflite_path)
    print(f"  TFLite saved: {tflite_path}")
    _verify_tflite(tflite_path, X_test[:1])

    print(f"\nModel saved to {arch_dir}/")


def _verify_tflite(tflite_path: str, sample_input: np.ndarray):
    """Run one sample through the TFLite model to confirm it works."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    interpreter.set_tensor(inp_det[0]["index"], sample_input.astype(np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(out_det[0]["index"])
    print(f"  TFLite test inference: {result[0][0]:.4f} (expected 0–1)  ✓")


if __name__ == "__main__":
    main()
