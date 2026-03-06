"""
Train a CNN model on mel spectrogram features for depression detection.
Also creates a combined TFLite model with preprocessing baked in.

Usage:
    python train_mel_cnn.py
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

# Add project root to path
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
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)  # 80000

# Mel spectrogram params
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2  # 8000 Hz

# Expected time steps: ceil((AUDIO_LENGTH - N_FFT) / HOP_LENGTH) + 1
EXPECTED_TIME_STEPS = 313

# Training params
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOP_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Split config
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = "artifacts/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def create_mel_cnn(
    input_shape=(128, 313, 1), dropout_rate=0.3
) -> keras.Model:
    """
    CNN for mel spectrogram features.
    
    4 Conv blocks to handle the larger 128-band input (vs 13 for MFCCs).
    Uses BatchNorm + GlobalAvgPool for efficient mobile deployment.
    """
    model = keras.Sequential(
        [
            # Block 1: (128, 313, 1) -> (64, 156, 32)
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),

            # Block 2: (64, 156, 32) -> (32, 78, 64)
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate + 0.1),

            # Block 3: (32, 78, 64) -> (16, 39, 128)
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate + 0.1),

            # Block 4: (16, 39, 128) -> global avg pool -> 128
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            # Classifier
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate + 0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mel_cnn",
    )
    return model


def evaluate_split(model, X, y, split_name="Test"):
    """Evaluate model on a data split and print metrics."""
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'='*50}")
    print(f"{split_name} Results ({len(y)} samples)")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC:      {auc:.4f} ({auc*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y, y_pred, target_names=['Normal', 'Depression'])}")

    return acc, auc


def create_combined_tflite(classification_model):
    """
    Create a combined model: raw audio -> mel spectrogram preprocessing -> CNN -> prediction.
    
    Preprocessing chain:
        STFT -> power spectrum -> mel filterbank -> log scale -> normalize -> CNN
    
    No DCT step needed (unlike MFCC). This is simpler and maps directly to mel spectrogram.
    """
    # Precompute mel filterbank weights
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=F_MIN,
        upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")

    # STFT
    x = tf.signal.stft(
        audio_input,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )

    # Power spectrogram |S|^2
    x = tf.abs(x)
    x = tf.math.square(x)

    # Mel filterbank: (batch, time, n_mels)
    mel_w = tf.constant(mel_weights, dtype=tf.float32)
    x = tf.matmul(x, mel_w)

    # Power to dB: matching librosa.power_to_db(mel_spec, ref=np.max)
    # Formula: 10 * log10(S / ref) where ref = max of each spectrogram
    # This gives values in [-80, 0] range (with amin=1e-10 default)
    log10 = tf.math.log(10.0)
    amin = 1e-10
    ref = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    x = tf.maximum(x, amin)
    x = 10.0 * (tf.math.log(x) - tf.math.log(ref)) / log10
    # Clip to top_db=80.0 (librosa default)
    x = tf.maximum(x, -80.0)

    # Transpose to (batch, n_mels, time) to match training feature shape
    x = tf.transpose(x, perm=[0, 2, 1])

    # Truncate/pad time dim to EXPECTED_TIME_STEPS
    x = x[:, :, :EXPECTED_TIME_STEPS]
    pad_size = EXPECTED_TIME_STEPS - tf.shape(x)[2]
    paddings = tf.stack([
        tf.constant([0, 0]),
        tf.constant([0, 0]),
        tf.stack([tf.constant(0), pad_size])
    ])
    x = tf.pad(x, paddings)
    x = tf.reshape(x, [-1, N_MELS, EXPECTED_TIME_STEPS])

    # No per-band normalization — training pipeline (extract_mel_spectrogram)
    # does NOT normalize, just returns dB-scaled values

    # Add channel dimension: (batch, n_mels, time, 1)
    x = tf.expand_dims(x, axis=-1)

    # Classification
    output = classification_model(x, training=False)

    combined = keras.Model(inputs=audio_input, outputs=output, name="mel_depression_detector")
    return combined


def convert_to_tflite(combined_model, output_path):
    """Convert combined model to TFLite, trying builtins-only first."""
    converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
    # Float32 only — no quantization. DEFAULT quantization was squashing
    # the mel spectrogram model outputs to zero due to wide dynamic range.

    needs_flex = False
    try:
        print("Trying builtins-only conversion...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        print("SUCCESS: Converted with TFLite builtins only!")
    except Exception as e:
        print(f"Builtins-only failed: {e}")
        print("\nRetrying with SELECT_TF_OPS...")
        converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
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


def verify_tflite(tflite_path, data_dir):
    """Verify TFLite model on a few samples."""
    import librosa

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nTFLite Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"TFLite Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    dep_dir = os.path.join(data_dir, "depression1")
    norm_dir = os.path.join(data_dir, "normal1")
    dep_files = sorted([f for f in os.listdir(dep_dir) if f.endswith(".wav")])[:5]
    norm_files = sorted([f for f in os.listdir(norm_dir) if f.endswith(".wav")])[:5]

    print("\nTFLite Depression samples:")
    for f in dep_files:
        audio, sr = librosa.load(os.path.join(dep_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        interpreter.set_tensor(input_details[0]["index"], audio[np.newaxis].astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        label = "DEP" if out[0][0] >= 0.5 else "NORM"
        print(f"  {f}: {out[0][0]:.4f} [{label}]")

    print("\nTFLite Normal samples:")
    for f in norm_files:
        audio, sr = librosa.load(os.path.join(norm_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        interpreter.set_tensor(input_details[0]["index"], audio[np.newaxis].astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        label = "DEP" if out[0][0] >= 0.5 else "NORM"
        print(f"  {f}: {out[0][0]:.4f} [{label}]")


def evaluate_tflite_splits(tflite_path, audio_list_train, y_train, audio_list_val, y_val, audio_list_test, y_test):
    """Evaluate TFLite model on train/val/test splits."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for split_name, audio_list, y_true in [
        ("Train", audio_list_train, y_train),
        ("Validation", audio_list_val, y_val),
        ("Test", audio_list_test, y_test),
    ]:
        preds = []
        for audio in audio_list:
            audio_in = audio.astype(np.float32)
            if len(audio_in) < AUDIO_LENGTH:
                audio_in = np.pad(audio_in, (0, AUDIO_LENGTH - len(audio_in)))
            else:
                audio_in = audio_in[:AUDIO_LENGTH]
            interpreter.set_tensor(input_details[0]["index"], audio_in[np.newaxis])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            preds.append(out[0][0])

        preds = np.array(preds)
        y_pred = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, preds)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f"TFLite {split_name} Results ({len(y_true)} samples)")
        print(f"{'='*50}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"AUC:      {auc:.4f} ({auc*100:.2f}%)")
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Depression Detection - Mel Spectrogram CNN Training")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load audio data
    # ----------------------------------------------------------
    print("\n[1/6] Loading audio data...")
    loader = AudioDataLoader(
        data_dir=DATA_DIR,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        mono=True,
    )
    audio_list, labels, file_paths = loader.load_dataset(
        depression_dir="depression1",
        normal_dir="normal1",
    )
    print(f"Loaded {len(audio_list)} samples (dep={labels.count(1)}, norm={labels.count(0)})")

    # ----------------------------------------------------------
    # 2. Extract mel spectrogram features using tf.signal
    #    (matches the combined TFLite model preprocessing exactly)
    # ----------------------------------------------------------
    print("\n[2/6] Extracting mel spectrogram features (tf.signal)...")

    # Precompute mel filterbank weights
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=F_MIN,
        upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    def extract_mel_tf(audio_np):
        """Extract mel spectrogram using tf.signal ops (matching TFLite pipeline)."""
        audio_tf = tf.constant(audio_np[np.newaxis], dtype=tf.float32)
        stft = tf.signal.stft(audio_tf, frame_length=N_FFT, frame_step=HOP_LENGTH,
                              fft_length=N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
        power = tf.math.square(tf.abs(stft))
        mel = tf.matmul(power, tf.constant(mel_weights, dtype=tf.float32))

        # power_to_db with ref=max (matching librosa)
        amin = 1e-10
        ref = tf.reduce_max(mel, axis=[1, 2], keepdims=True)
        ref = tf.maximum(ref, amin)
        mel = tf.maximum(mel, amin)
        log10 = tf.math.log(10.0)
        mel_db = 10.0 * (tf.math.log(mel) - tf.math.log(ref)) / log10
        mel_db = tf.maximum(mel_db, -80.0)

        # (1, time, n_mels) -> (n_mels, time, 1)
        mel_db = tf.squeeze(mel_db, axis=0)  # (time, n_mels)
        mel_db = tf.transpose(mel_db)  # (n_mels, time)
        return mel_db.numpy()

    features = []
    for i, audio in enumerate(audio_list):
        mel = extract_mel_tf(audio)
        # Pad/truncate time axis
        if mel.shape[1] < EXPECTED_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, EXPECTED_TIME_STEPS - mel.shape[1])))
        else:
            mel = mel[:, :EXPECTED_TIME_STEPS]
        features.append(mel)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(audio_list)}")

    X = np.array(features)[..., np.newaxis]  # (N, 128, 313, 1)
    y = np.array(labels)
    print(f"Feature shape: {X.shape}")  # Expected: (800, 128, 313, 1)

    # ----------------------------------------------------------
    # 3. Split data
    # ----------------------------------------------------------
    print("\n[3/6] Splitting data (70/15/15)...")
    splits = split_dataset(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Also split the raw audio for TFLite evaluation later
    audio_arr = np.array(audio_list)
    audio_splits = split_dataset(audio_arr, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)

    # ----------------------------------------------------------
    # 4. Build and train model
    # ----------------------------------------------------------
    print("\n[4/6] Building mel spectrogram CNN...")
    input_shape = X_train.shape[1:]  # (128, 313, 1)
    model = create_mel_cnn(input_shape=input_shape, dropout_rate=DROPOUT_RATE)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # Callbacks
    best_model_path = os.path.join(MODEL_DIR, f"mel_cnn_{TIMESTAMP}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOP_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, f"mel_cnn_{TIMESTAMP}_final.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    print(f"Best model saved:  {best_model_path}")

    # ----------------------------------------------------------
    # 5. Evaluate on all splits
    # ----------------------------------------------------------
    print("\n[5/6] Evaluating Keras model on all splits...")
    evaluate_split(model, X_train, y_train, "Train")
    evaluate_split(model, X_val, y_val, "Validation")
    evaluate_split(model, X_test, y_test, "Test")

    # ----------------------------------------------------------
    # 6. Create combined TFLite model
    # ----------------------------------------------------------
    print("\n[6/6] Creating combined TFLite model (audio -> prediction)...")
    combined = create_combined_tflite(model)
    combined.summary()

    tflite_path = os.path.join(MODEL_DIR, "mel_depression_combined.tflite")
    convert_to_tflite(combined, tflite_path)

    # Verify TFLite on a few samples
    print("\n--- Verifying TFLite model ---")
    verify_tflite(tflite_path, DATA_DIR)

    # Evaluate TFLite on all splits
    print("\n--- TFLite Split Evaluation ---")
    a_train, ay_train = audio_splits["train"]
    a_val, ay_val = audio_splits["val"]
    a_test, ay_test = audio_splits["test"]
    evaluate_tflite_splits(
        tflite_path,
        a_train, ay_train,
        a_val, ay_val,
        a_test, ay_test,
    )

    print("\n" + "=" * 60)
    print("  DONE! Mel Spectrogram CNN Training Complete")
    print("=" * 60)
    print(f"  Keras model: {best_model_path}")
    print(f"  TFLite model: {tflite_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
