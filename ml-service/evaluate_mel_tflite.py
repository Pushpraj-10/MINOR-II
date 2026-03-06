"""Full evaluation of the mel spectrogram combined TFLite model on train/val/test splits."""
import os, sys, numpy as np, tensorflow as tf
from tensorflow import keras
import librosa
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

sys.path.insert(0, '.')
from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset

SAMPLE_RATE = 16000
AUDIO_LENGTH = 80000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 128
EXPECTED_TIME_STEPS = 313
F_MIN = 0.0
F_MAX = 8000.0

MODEL_PATH = 'artifacts/models/mel_cnn_20260228_133811_best.keras'
TFLITE_PATH = 'artifacts/models/mel_depression_combined.tflite'

def build_and_convert():
    """Build combined model and convert to TFLite (float32)."""
    model = keras.models.load_model(MODEL_PATH)

    mel_weights_np = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT//2+1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX, dtype=tf.float32
    ).numpy()

    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name='audio_input')
    x = tf.signal.stft(audio_input, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT,
                        window_fn=tf.signal.hann_window, pad_end=False)
    x = tf.abs(x)
    x = tf.math.square(x)
    mel_w = tf.constant(mel_weights_np, dtype=tf.float32)
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
        tf.stack([tf.constant(0), pad_size])
    ])
    x = tf.pad(x, paddings)
    x = tf.reshape(x, [-1, N_MELS, EXPECTED_TIME_STEPS])
    x = tf.expand_dims(x, axis=-1)
    output = model(x, training=False)
    combined = keras.Model(inputs=audio_input, outputs=output, name='mel_depression_detector')

    # Convert float32 (no quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(combined)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite: {TFLITE_PATH} ({len(tflite_model)/1024:.1f} KB)")
    return TFLITE_PATH


def evaluate_tflite_split(interpreter, input_details, output_details, audio_list, y_true, split_name):
    preds = []
    for audio in audio_list:
        audio_in = audio.astype(np.float32)
        if len(audio_in) < AUDIO_LENGTH:
            audio_in = np.pad(audio_in, (0, AUDIO_LENGTH - len(audio_in)))
        else:
            audio_in = audio_in[:AUDIO_LENGTH]
        interpreter.set_tensor(input_details[0]['index'], audio_in[np.newaxis])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
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
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal', 'Depression'])}")
    return acc, auc


def main():
    print("=" * 60)
    print("  Mel Spectrogram TFLite - Full Evaluation")
    print("=" * 60)

    # Build and convert
    print("\n[1] Building combined model and converting to TFLite...")
    build_and_convert()

    # Load data
    print("\n[2] Loading audio data...")
    loader = AudioDataLoader(data_dir="data/raw/voice_data", sample_rate=SAMPLE_RATE,
                             duration=5.0, mono=True)
    audio_list, labels, file_paths = loader.load_dataset()
    y = np.array(labels)
    audio_arr = np.array(audio_list)
    print(f"Loaded {len(audio_list)} samples")

    # Split (same as training)
    print("[3] Splitting data...")
    splits = split_dataset(audio_arr, y, test_size=0.15, val_size=0.15, random_state=42)
    a_train, y_train = splits["train"]
    a_val, y_val = splits["val"]
    a_test, y_test = splits["test"]

    # Evaluate
    print("\n[4] Evaluating TFLite model on all splits...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print(f"Input: {inp[0]['shape']}, Output: {out[0]['shape']}")

    evaluate_tflite_split(interpreter, inp, out, a_train, y_train, "Train")
    evaluate_tflite_split(interpreter, inp, out, a_val, y_val, "Validation")
    evaluate_tflite_split(interpreter, inp, out, a_test, y_test, "Test")

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
