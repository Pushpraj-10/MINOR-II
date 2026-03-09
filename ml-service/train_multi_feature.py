"""
Model 3: Multi-Feature Fusion CNN (MFCC + Mel Spectrogram).

Approach: Use BOTH MFCC (13 bands, captures timbral quality) and mel spectrogram
(128 bands, captures spectral detail) as separate input branches. Each branch has
its own CNN, and features are fused before the classifier. This leverages
complementary information from different audio representations.

Usage:
    python train_multi_feature.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "data/raw/voice_data"
SAMPLE_RATE = 16000
DURATION = 5.0
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)

# Mel spectrogram params
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2
MEL_TIME_STEPS = 313

# MFCC params
N_MFCC = 13
MFCC_TIME_STEPS = 313  # same STFT params → same time steps

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOP_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = "artifacts/models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Model Architecture
# ============================================================
def create_multi_feature_cnn(
    mel_shape=(128, 313, 1),
    mfcc_shape=(13, 313, 1),
    dropout_rate=0.3,
) -> keras.Model:
    """
    Dual-branch CNN fusing mel spectrogram and MFCC features.
    
    Branch A (Mel): 3 Conv2D blocks → GlobalAvgPool → 64-dim
    Branch B (MFCC): 2 Conv2D blocks → GlobalAvgPool → 32-dim
    Fusion: Concatenate(64 + 32) → Dense(64) → Dense(1)
    """
    # --- Branch A: Mel Spectrogram ---
    mel_input = layers.Input(shape=mel_shape, name="mel_input")
    a = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(mel_input)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Dropout(dropout_rate)(a)

    a = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Dropout(dropout_rate)(a)

    a = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a)
    a = layers.BatchNormalization()(a)
    a = layers.GlobalAveragePooling2D()(a)  # -> 64-dim

    # --- Branch B: MFCC ---
    mfcc_input = layers.Input(shape=mfcc_shape, name="mfcc_input")
    b = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(mfcc_input)
    b = layers.BatchNormalization()(b)
    b = layers.MaxPooling2D((2, 2))(b)
    b = layers.Dropout(dropout_rate)(b)

    b = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(b)
    b = layers.BatchNormalization()(b)
    b = layers.GlobalAveragePooling2D()(b)  # -> 64-dim

    # --- Fusion ---
    fused = layers.Concatenate()([a, b])  # 64 + 64 = 128-dim
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(dropout_rate + 0.2)(fused)
    output = layers.Dense(1, activation="sigmoid")(fused)

    model = keras.Model(inputs=[mel_input, mfcc_input], outputs=output, name="multi_feature_cnn")
    return model


# ============================================================
# Feature extraction helpers
# ============================================================
def extract_mel_tf(audio_np, mel_weights):
    """Mel spectrogram -> (n_mels, time) shape."""
    audio_tf = tf.constant(audio_np[np.newaxis], dtype=tf.float32)
    stft = tf.signal.stft(audio_tf, frame_length=N_FFT, frame_step=HOP_LENGTH,
                          fft_length=N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.math.square(tf.abs(stft))
    mel = tf.matmul(power, tf.constant(mel_weights, dtype=tf.float32))
    amin = 1e-10
    ref = tf.reduce_max(mel, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    mel = tf.maximum(mel, amin)
    log10 = tf.math.log(10.0)
    mel_db = 10.0 * (tf.math.log(mel) - tf.math.log(ref)) / log10
    mel_db = tf.maximum(mel_db, -80.0)
    mel_db = tf.squeeze(mel_db, axis=0)  # (time, n_mels)
    mel_db = tf.transpose(mel_db)  # (n_mels, time)
    return mel_db.numpy()


def extract_mfcc_tf(audio_np, mel_weights):
    """MFCC via tf.signal: mel spectrogram -> DCT -> first N_MFCC coefficients."""
    audio_tf = tf.constant(audio_np[np.newaxis], dtype=tf.float32)
    stft = tf.signal.stft(audio_tf, frame_length=N_FFT, frame_step=HOP_LENGTH,
                          fft_length=N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.math.square(tf.abs(stft))
    mel = tf.matmul(power, tf.constant(mel_weights, dtype=tf.float32))

    # Log mel (for MFCC, use simple log not power_to_db)
    mel = tf.maximum(mel, 1e-10)
    log_mel = tf.math.log(mel)

    # DCT-II to get MFCCs
    mfccs = tf.signal.dct(log_mel, type=2, norm="ortho")
    mfccs = mfccs[:, :, :N_MFCC]  # keep first 13 coefficients

    mfccs = tf.squeeze(mfccs, axis=0)  # (time, n_mfcc)
    mfccs = tf.transpose(mfccs)  # (n_mfcc, time)
    return mfccs.numpy()


# ============================================================
# Evaluation & TFLite helpers
# ============================================================
def evaluate_split(model, X_mel, X_mfcc, y, split_name="Test"):
    y_pred_prob = model.predict([X_mel, X_mfcc], verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)
    cm = confusion_matrix(y, y_pred)
    print(f"\n{'='*50}")
    print(f"{split_name} Results ({len(y)} samples)")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC:      {auc:.4f} ({auc*100:.2f}%)")
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y, y_pred, target_names=['Normal', 'Depression'])}")
    return acc, auc


def create_combined_tflite(classification_model):
    """Combined: audio -> [mel_spec, mfcc] -> dual-branch CNN -> prediction."""
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")

    # Shared STFT
    stft = tf.signal.stft(audio_input, frame_length=N_FFT, frame_step=HOP_LENGTH,
                           fft_length=N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.math.square(tf.abs(stft))
    mel_w = tf.constant(mel_weights, dtype=tf.float32)
    mel_raw = tf.matmul(power, mel_w)

    # --- Mel spectrogram branch ---
    log10 = tf.math.log(10.0)
    amin = 1e-10
    ref = tf.reduce_max(mel_raw, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    mel_safe = tf.maximum(mel_raw, amin)
    mel_db = 10.0 * (tf.math.log(mel_safe) - tf.math.log(ref)) / log10
    mel_db = tf.maximum(mel_db, -80.0)
    mel_db = tf.transpose(mel_db, perm=[0, 2, 1])  # (batch, n_mels, time)
    mel_db = mel_db[:, :, :MEL_TIME_STEPS]
    pad_mel = MEL_TIME_STEPS - tf.shape(mel_db)[2]
    mel_db = tf.pad(mel_db, tf.stack([
        tf.constant([0, 0]), tf.constant([0, 0]),
        tf.stack([tf.constant(0), pad_mel])
    ]))
    mel_db = tf.reshape(mel_db, [-1, N_MELS, MEL_TIME_STEPS])
    mel_4d = tf.expand_dims(mel_db, axis=-1)  # (batch, 128, 313, 1)

    # --- MFCC branch ---
    mel_log = tf.maximum(mel_raw, 1e-10)
    mel_log = tf.math.log(mel_log)
    mfccs = tf.signal.dct(mel_log, type=2, norm="ortho")
    mfccs = mfccs[:, :, :N_MFCC]  # (batch, time, 13)
    mfccs = tf.transpose(mfccs, perm=[0, 2, 1])  # (batch, 13, time)
    mfccs = mfccs[:, :, :MFCC_TIME_STEPS]
    pad_mfcc = MFCC_TIME_STEPS - tf.shape(mfccs)[2]
    mfccs = tf.pad(mfccs, tf.stack([
        tf.constant([0, 0]), tf.constant([0, 0]),
        tf.stack([tf.constant(0), pad_mfcc])
    ]))
    mfccs = tf.reshape(mfccs, [-1, N_MFCC, MFCC_TIME_STEPS])
    mfcc_4d = tf.expand_dims(mfccs, axis=-1)  # (batch, 13, 313, 1)

    output = classification_model([mel_4d, mfcc_4d], training=False)
    return keras.Model(inputs=audio_input, outputs=output, name="multi_feature_depression_detector")


def convert_to_tflite(combined_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    try:
        print("Trying builtins-only conversion...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        needs_flex = False
        print("SUCCESS: Converted with TFLite builtins only!")
    except Exception as e:
        print(f"Builtins-only failed: {e}")
        print("\nRetrying with SELECT_TF_OPS...")
        converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        needs_flex = True
        print("SUCCESS: Converted with SELECT_TF_OPS")

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Saved: {output_path} ({size_kb:.1f} KB)")
    print(f"Needs flex delegate: {needs_flex}")
    return tflite_model


def evaluate_tflite_splits(tflite_path, audio_splits):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    for name in ["train", "val", "test"]:
        audio_list, y_true = audio_splits[name]
        preds = []
        for audio in audio_list:
            audio_in = audio.astype(np.float32)
            if len(audio_in) < AUDIO_LENGTH:
                audio_in = np.pad(audio_in, (0, AUDIO_LENGTH - len(audio_in)))
            else:
                audio_in = audio_in[:AUDIO_LENGTH]
            interpreter.set_tensor(inp[0]["index"], audio_in[np.newaxis])
            interpreter.invoke()
            preds.append(interpreter.get_tensor(out[0]["index"])[0][0])
        preds = np.array(preds)
        y_pred = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, preds)
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nTFLite {name.capitalize()} ({len(y_true)} samples): "
              f"Acc={acc:.4f}, AUC={auc:.4f} | TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Model 3: Multi-Feature Fusion CNN (Mel + MFCC)")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading audio data...")
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, duration=DURATION, mono=True)
    audio_list, labels, file_paths = loader.load_dataset(depression_dir="depression1", normal_dir="normal1")
    print(f"Loaded {len(audio_list)} samples (dep={labels.count(1)}, norm={labels.count(0)})")

    # 2. Extract BOTH feature types
    print("\n[2/6] Extracting mel spectrogram + MFCC features (tf.signal)...")
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    mel_features = []
    mfcc_features = []
    for i, audio in enumerate(audio_list):
        # Mel spectrogram
        mel = extract_mel_tf(audio, mel_weights)  # (n_mels, time)
        if mel.shape[1] < MEL_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, MEL_TIME_STEPS - mel.shape[1])))
        else:
            mel = mel[:, :MEL_TIME_STEPS]
        mel_features.append(mel)

        # MFCC
        mfcc = extract_mfcc_tf(audio, mel_weights)  # (n_mfcc, time)
        if mfcc.shape[1] < MFCC_TIME_STEPS:
            mfcc = np.pad(mfcc, ((0, 0), (0, MFCC_TIME_STEPS - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :MFCC_TIME_STEPS]
        mfcc_features.append(mfcc)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(audio_list)}")

    X_mel = np.array(mel_features)[..., np.newaxis]   # (N, 128, 313, 1)
    X_mfcc = np.array(mfcc_features)[..., np.newaxis]  # (N, 13, 313, 1)
    y = np.array(labels)
    print(f"Mel shape: {X_mel.shape}, MFCC shape: {X_mfcc.shape}")

    # 3. Split — need to split both feature arrays identically
    print("\n[3/6] Splitting data...")
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    idx_temp, idx_test, y_temp, y_test_labels = train_test_split(
        indices, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    adj_val = VAL_SIZE / (1 - TEST_SIZE)
    idx_train, idx_val, y_train_labels, y_val_labels = train_test_split(
        idx_temp, y_temp, test_size=adj_val, random_state=RANDOM_STATE, stratify=y_temp
    )

    X_mel_train, X_mel_val, X_mel_test = X_mel[idx_train], X_mel[idx_val], X_mel[idx_test]
    X_mfcc_train, X_mfcc_val, X_mfcc_test = X_mfcc[idx_train], X_mfcc[idx_val], X_mfcc[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    audio_arr = np.array(audio_list)
    audio_splits = split_dataset(audio_arr, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)

    # 4. Build & train
    print("\n[4/6] Building Multi-Feature Fusion CNN...")
    model = create_multi_feature_cnn(
        mel_shape=X_mel_train.shape[1:],
        mfcc_shape=X_mfcc_train.shape[1:],
        dropout_rate=DROPOUT_RATE,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    best_model_path = os.path.join(MODEL_DIR, f"multi_feature_{TIMESTAMP}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=EARLY_STOP_PATIENCE,
                                       mode="max", restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_auc",
                                         mode="max", save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(
        [X_mel_train, X_mfcc_train], y_train,
        validation_data=([X_mel_val, X_mfcc_val], y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1,
    )

    final_path = os.path.join(MODEL_DIR, f"multi_feature_{TIMESTAMP}_final.keras")
    model.save(final_path)
    print(f"\nBest model:  {best_model_path}")
    print(f"Final model: {final_path}")

    # 5. Evaluate
    print("\n[5/6] Evaluating Keras model...")
    evaluate_split(model, X_mel_train, X_mfcc_train, y_train, "Train")
    evaluate_split(model, X_mel_val, X_mfcc_val, y_val, "Validation")
    evaluate_split(model, X_mel_test, X_mfcc_test, y_test, "Test")

    # 6. TFLite
    print("\n[6/6] Creating combined TFLite model...")
    combined = create_combined_tflite(model)
    combined.summary()
    tflite_path = os.path.join(MODEL_DIR, "multi_feature_depression_combined.tflite")
    convert_to_tflite(combined, tflite_path)

    print("\n--- TFLite Split Evaluation ---")
    evaluate_tflite_splits(tflite_path, audio_splits)

    print("\n" + "=" * 60)
    print("  DONE! Multi-Feature Fusion CNN Training Complete")
    print("=" * 60)
    print(f"  Keras: {best_model_path}")
    print(f"  TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
