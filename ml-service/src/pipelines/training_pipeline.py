"""
Training pipeline for all depression detection model architectures.

Encapsulates the full load → extract features → split → train → evaluate
→ TFLite export workflow. scripts/train_model.py is a thin CLI wrapper
that calls run_training_pipeline().
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset
from src.features.tf_audio import (
    compute_mel_weights,
    extract_features_batch,
    extract_dual_features_batch,
)
from src.models.architectures import MODEL_REGISTRY, get_model
from src.export.tflite_converter import (
    build_combined_cnn,
    build_combined_lstm,
    build_combined_dual_branch,
    convert_to_tflite,
)
from src.evaluation.evaluator import print_split_summary, evaluate_tflite_on_splits
from src.config import (
    SAMPLE_RATE, DURATION, EXPECTED_TIME_STEPS, AUDIO_LENGTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
    DATA_DIR, DEPRESSION_DIR, NORMAL_DIR, MODEL_DIR,
)

logger = logging.getLogger(__name__)

_COMBINED_BUILDERS = {
    "cnn": build_combined_cnn,
    "lstm": build_combined_lstm,
    "dual_branch": build_combined_dual_branch,
}


# ── Private helpers ───────────────────────────────────────────────────

def _load_audio():
    """Load raw audio waveforms from the configured dataset directory."""
    loader = AudioDataLoader(
        data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, duration=DURATION, mono=True
    )
    audio_list, labels, _ = loader.load_dataset(
        depression_dir=DEPRESSION_DIR, normal_dir=NORMAL_DIR
    )
    logger.info(
        "Loaded %d samples  (dep=%d  norm=%d)",
        len(audio_list), labels.count(1), labels.count(0),
    )
    return audio_list, np.array(labels)


def _build_callbacks(arch_name, timestamp):
    """Return standard Keras training callbacks and the checkpoint path."""
    best_path = os.path.join(MODEL_DIR, f"{arch_name}_{timestamp}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=EARLY_STOP_PATIENCE,
            mode="max", restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_auc",
            mode="max", save_best_only=True, verbose=1,
        ),
    ]
    return callbacks, best_path


def _eval_single(model, X, y, split_name):
    return print_split_summary(y, model.predict(X, verbose=0), split_name)


def _eval_dual(model, X_mel, X_mfcc, y, split_name):
    return print_split_summary(y, model.predict([X_mel, X_mfcc], verbose=0), split_name)


# ── Pipeline implementations ──────────────────────────────────────────

def _train_single_input(arch_name, arch_info, epochs, batch_size, lr, dropout):
    """Training pipeline for single-input architectures (CNN, LSTM variants)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/6] Loading audio data...")
    audio_list, y = _load_audio()
    print(f"Loaded {len(audio_list)} samples  "
          f"(dep={int(y.sum())}  norm={int((y == 0).sum())})")

    input_type = arch_info["input_type"]
    output_format = "lstm" if input_type == "lstm" else "cnn"

    print(f"\n[2/6] Extracting mel features (format={output_format})...")
    mel_weights = compute_mel_weights()
    X = extract_features_batch(
        audio_list, mel_weights, EXPECTED_TIME_STEPS,
        feature_type="mel", output_format=output_format,
    )
    print(f"Feature shape: {X.shape}")

    print("\n[3/6] Splitting data (70/15/15)...")
    splits = split_dataset(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    audio_splits = split_dataset(
        np.array(audio_list), y,
        test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE,
    )

    print(f"\n[4/6] Building {arch_name}...")
    model = get_model(arch_name, input_shape=X_train.shape[1:], dropout_rate=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    callbacks, best_path = _build_callbacks(arch_name, timestamp)
    print("\nTraining...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )

    final_path = os.path.join(MODEL_DIR, f"{arch_name}_{timestamp}_final.keras")
    model.save(final_path)
    print(f"\nBest model:  {best_path}")
    print(f"Final model: {final_path}")

    print("\n[5/6] Evaluating Keras model...")
    _eval_single(model, X_train, y_train, "Train")
    _eval_single(model, X_val,   y_val,   "Validation")
    _eval_single(model, X_test,  y_test,  "Test")

    print("\n[6/6] Creating combined TFLite model...")
    build_fn = _COMBINED_BUILDERS[arch_info["combined_builder"]]
    combined = build_fn(model, model_name=f"{arch_name}_depression_detector")
    combined.summary()

    tflite_path = os.path.join(MODEL_DIR, arch_info["tflite_name"])
    convert_to_tflite(combined, tflite_path)

    print("\n--- TFLite Split Evaluation ---")
    evaluate_tflite_on_splits(tflite_path, audio_splits, audio_length=AUDIO_LENGTH)

    return tflite_path


def _train_dual_input(arch_name, arch_info, epochs, batch_size, lr, dropout):
    """Training pipeline for the dual-branch (mel + MFCC) architecture."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/6] Loading audio data...")
    audio_list, y = _load_audio()
    print(f"Loaded {len(audio_list)} samples  "
          f"(dep={int(y.sum())}  norm={int((y == 0).sum())})")

    print("\n[2/6] Extracting mel + MFCC features...")
    mel_weights = compute_mel_weights()
    X_mel, X_mfcc = extract_dual_features_batch(audio_list, mel_weights, EXPECTED_TIME_STEPS)
    print(f"Mel shape: {X_mel.shape}  MFCC shape: {X_mfcc.shape}")

    print("\n[3/6] Splitting data (70/15/15)...")
    indices = np.arange(len(y))
    idx_temp, idx_test, y_temp, _ = train_test_split(
        indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    adj_val = VAL_SIZE / (1 - TEST_SIZE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, y_temp, test_size=adj_val, random_state=RANDOM_STATE, stratify=y_temp
    )

    X_mel_train,  X_mel_val,  X_mel_test  = X_mel[idx_train],  X_mel[idx_val],  X_mel[idx_test]
    X_mfcc_train, X_mfcc_val, X_mfcc_test = X_mfcc[idx_train], X_mfcc[idx_val], X_mfcc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    audio_splits = split_dataset(
        np.array(audio_list), y,
        test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE,
    )

    print(f"\n[4/6] Building {arch_name}...")
    model = get_model(
        arch_name,
        mel_shape=X_mel_train.shape[1:],
        mfcc_shape=X_mfcc_train.shape[1:],
        dropout_rate=dropout,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    callbacks, best_path = _build_callbacks(arch_name, timestamp)
    print("\nTraining...")
    model.fit(
        [X_mel_train, X_mfcc_train], y_train,
        validation_data=([X_mel_val, X_mfcc_val], y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )

    final_path = os.path.join(MODEL_DIR, f"{arch_name}_{timestamp}_final.keras")
    model.save(final_path)
    print(f"\nBest model:  {best_path}")
    print(f"Final model: {final_path}")

    print("\n[5/6] Evaluating Keras model...")
    _eval_dual(model, X_mel_train, X_mfcc_train, y_train, "Train")
    _eval_dual(model, X_mel_val,   X_mfcc_val,   y_val,   "Validation")
    _eval_dual(model, X_mel_test,  X_mfcc_test,  y_test,  "Test")

    print("\n[6/6] Creating combined TFLite model...")
    combined = build_combined_dual_branch(model, model_name=f"{arch_name}_detector")
    combined.summary()

    tflite_path = os.path.join(MODEL_DIR, arch_info["tflite_name"])
    convert_to_tflite(combined, tflite_path)

    print("\n--- TFLite Split Evaluation ---")
    evaluate_tflite_on_splits(tflite_path, audio_splits, audio_length=AUDIO_LENGTH)

    return tflite_path


# ── Public API ────────────────────────────────────────────────────────

def run_training_pipeline(
    arch_name,
    *,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    dropout=DROPOUT_RATE,
):
    """
    Run the full training pipeline for the given architecture.

    Dispatches to single-input or dual-input path based on MODEL_REGISTRY.
    Returns the path to the exported TFLite model.
    """
    if arch_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    arch_info = MODEL_REGISTRY[arch_name]
    if arch_info["input_type"] == "dual":
        return _train_dual_input(arch_name, arch_info, epochs, batch_size, lr, dropout)
    return _train_single_input(arch_name, arch_info, epochs, batch_size, lr, dropout)
