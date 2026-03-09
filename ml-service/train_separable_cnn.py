"""
Model 5: Depthwise Separable CNN (MobileNet-style) on mel spectrograms.

Approach: Uses depthwise separable convolutions instead of standard Conv2D.
This factorizes convolution into a depthwise spatial filter + pointwise 1x1
filter, dramatically reducing parameters and computation while maintaining
accuracy. Ideal for mobile deployment.

Usage:
    python train_separable_cnn.py
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

N_FFT = 512
HOP_LENGTH = 256
N_MELS = 128
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2
EXPECTED_TIME_STEPS = 313

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
def create_separable_cnn(input_shape=(128, 313, 1), dropout_rate=0.3) -> keras.Model:
    """
    MobileNet-style depthwise separable CNN.
    
    - First layer is standard Conv2D (to learn initial feature maps).
    - Remaining 4 blocks use SeparableConv2D for efficiency.
    - Each separable block: SepConv -> BN -> ReLU -> MaxPool -> Dropout.
    - Much fewer parameters than standard CNN (~60% reduction).
    """
    inputs = layers.Input(shape=input_shape)

    # Initial standard Conv to project from 1 channel
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    # Separable Block 1: (128, 313, 16) -> (64, 156, 32)
    x = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable Block 2: (64, 156, 32) -> (32, 78, 64)
    x = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable Block 3: (32, 78, 64) -> (16, 39, 128)
    x = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)

    # Separable Block 4: (16, 39, 128) -> (8, 19, 128)
    x = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="separable_cnn")
    return model


# ============================================================
# Shared helpers
# ============================================================
def extract_mel_tf(audio_np, mel_weights):
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
    mel_db = tf.squeeze(mel_db, axis=0)
    mel_db = tf.transpose(mel_db)  # (n_mels, time)
    return mel_db.numpy()


def evaluate_split(model, X, y, split_name="Test"):
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
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y, y_pred, target_names=['Normal', 'Depression'])}")
    return acc, auc


def create_combined_tflite(classification_model):
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")
    x = tf.signal.stft(audio_input, frame_length=N_FFT, frame_step=HOP_LENGTH,
                        fft_length=N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    x = tf.math.square(tf.abs(x))
    mel_w = tf.constant(mel_weights, dtype=tf.float32)
    x = tf.matmul(x, mel_w)

    log10 = tf.math.log(10.0)
    amin = 1e-10
    ref = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    x = tf.maximum(x, amin)
    x = 10.0 * (tf.math.log(x) - tf.math.log(ref)) / log10
    x = tf.maximum(x, -80.0)

    x = tf.transpose(x, perm=[0, 2, 1])
    x = x[:, :, :EXPECTED_TIME_STEPS]
    pad_size = EXPECTED_TIME_STEPS - tf.shape(x)[2]
    paddings = tf.stack([
        tf.constant([0, 0]),
        tf.constant([0, 0]),
        tf.stack([tf.constant(0), pad_size]),
    ])
    x = tf.pad(x, paddings)
    x = tf.reshape(x, [-1, N_MELS, EXPECTED_TIME_STEPS])
    x = tf.expand_dims(x, axis=-1)

    output = classification_model(x, training=False)
    return keras.Model(inputs=audio_input, outputs=output, name="separable_cnn_depression_detector")


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
    print("  Model 5: Depthwise Separable CNN (MobileNet-style)")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading audio data...")
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, duration=DURATION, mono=True)
    audio_list, labels, file_paths = loader.load_dataset(depression_dir="depression1", normal_dir="normal1")
    print(f"Loaded {len(audio_list)} samples (dep={labels.count(1)}, norm={labels.count(0)})")

    # 2. Extract mel features
    print("\n[2/6] Extracting mel spectrogram features (tf.signal)...")
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()

    features = []
    for i, audio in enumerate(audio_list):
        mel = extract_mel_tf(audio, mel_weights)
        if mel.shape[1] < EXPECTED_TIME_STEPS:
            mel = np.pad(mel, ((0, 0), (0, EXPECTED_TIME_STEPS - mel.shape[1])))
        else:
            mel = mel[:, :EXPECTED_TIME_STEPS]
        features.append(mel)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(audio_list)}")

    X = np.array(features)[..., np.newaxis]
    y = np.array(labels)
    print(f"Feature shape: {X.shape}")

    # 3. Split
    print("\n[3/6] Splitting data...")
    splits = split_dataset(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    audio_arr = np.array(audio_list)
    audio_splits = split_dataset(audio_arr, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)

    # 4. Build & train
    print("\n[4/6] Building Depthwise Separable CNN...")
    model = create_separable_cnn(input_shape=X_train.shape[1:], dropout_rate=DROPOUT_RATE)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    best_model_path = os.path.join(MODEL_DIR, f"separable_cnn_{TIMESTAMP}_best.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=EARLY_STOP_PATIENCE,
                                       mode="max", restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_auc",
                                         mode="max", save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    final_path = os.path.join(MODEL_DIR, f"separable_cnn_{TIMESTAMP}_final.keras")
    model.save(final_path)
    print(f"\nBest model:  {best_model_path}")
    print(f"Final model: {final_path}")

    # 5. Evaluate
    print("\n[5/6] Evaluating Keras model...")
    evaluate_split(model, X_train, y_train, "Train")
    evaluate_split(model, X_val, y_val, "Validation")
    evaluate_split(model, X_test, y_test, "Test")

    # 6. TFLite
    print("\n[6/6] Creating combined TFLite model...")
    combined = create_combined_tflite(model)
    combined.summary()
    tflite_path = os.path.join(MODEL_DIR, "separable_cnn_depression_combined.tflite")
    convert_to_tflite(combined, tflite_path)

    print("\n--- TFLite Split Evaluation ---")
    evaluate_tflite_splits(tflite_path, audio_splits)

    print("\n" + "=" * 60)
    print("  DONE! Separable CNN Training Complete")
    print("=" * 60)
    print(f"  Keras: {best_model_path}")
    print(f"  TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
