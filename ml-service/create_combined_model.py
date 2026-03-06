"""
Create a combined TFLite model that includes audio preprocessing (MFCC extraction)
using TensorFlow ops, so the Flutter app only needs to feed raw audio samples.

Tries builtins-only first for best mobile compatibility.
Falls back to SELECT_TF_OPS if needed.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

# Configuration matching the training pipeline
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 13
AUDIO_LENGTH = 80000  # 5 seconds at 16kHz
EXPECTED_TIME_STEPS = 313
F_MIN = 0.0
F_MAX = 8000.0  # sr / 2


def create_dct_matrix(n_mfcc, n_mels):
    """Create DCT-II matrix matching librosa/scipy's ortho normalization."""
    basis = np.zeros((n_mels, n_mfcc), dtype=np.float32)
    for k in range(n_mfcc):
        for n in range(n_mels):
            basis[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    basis[:, 0] *= np.sqrt(1.0 / n_mels)
    basis[:, 1:] *= np.sqrt(2.0 / n_mels)
    return basis


def create_combined_model(classification_model):
    """Create a Keras functional model that does preprocessing + classification.
    
    Uses tf.signal for STFT and keras.layers for operations to maximize 
    TFLite builtin op compatibility.
    """
    # Precompute constant matrices
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=F_MIN,
        upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()
    
    dct_matrix = create_dct_matrix(N_MFCC, N_MELS)
    
    # Build as functional model
    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name='audio_input')
    
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
    
    # Mel filterbank
    mel_w = tf.constant(mel_weights, dtype=tf.float32)
    x = tf.matmul(x, mel_w)
    
    # Log mel spectrogram
    x = tf.math.log(tf.maximum(x, 1e-10))
    
    # DCT -> MFCCs
    dct_w = tf.constant(dct_matrix, dtype=tf.float32)
    x = tf.matmul(x, dct_w)
    # shape: [batch, time_steps, N_MFCC]
    
    # Transpose to [batch, N_MFCC, time_steps]
    x = tf.transpose(x, perm=[0, 2, 1])
    
    # Truncate/pad time dim to EXPECTED_TIME_STEPS
    x = x[:, :, :EXPECTED_TIME_STEPS]
    pad_size = EXPECTED_TIME_STEPS - tf.shape(x)[2]
    x = tf.pad(x, [[0, 0], [0, 0], [0, pad_size]])
    # Use tf.reshape instead of tf.ensure_shape to avoid flex ops requirement
    x = tf.reshape(x, [-1, N_MFCC, EXPECTED_TIME_STEPS])
    
    # Per-coefficient normalization
    mean = tf.reduce_mean(x, axis=2, keepdims=True)
    variance = tf.reduce_mean(tf.math.square(x - mean), axis=2, keepdims=True)
    std = tf.math.sqrt(variance + 1e-8)
    x = (x - mean) / std
    
    # Reshape for CNN: [batch, N_MFCC, EXPECTED_TIME_STEPS, 1]
    x = tf.expand_dims(x, axis=-1)
    
    # Classification
    output = classification_model(x, training=False)
    
    combined = keras.Model(inputs=audio_input, outputs=output, name='depression_detector')
    return combined


def main():
    print("=" * 60)
    print("Creating Combined TFLite Model (Audio -> Prediction)")
    print("=" * 60)
    
    # Load the trained classification model
    model_path = 'artifacts/models/depression_detection_20260219_195034_best.keras'
    print(f"\nLoading trained model from: {model_path}")
    classification_model = keras.models.load_model(model_path)
    
    # Create the combined model
    print("\nCreating combined preprocessing + classification model...")
    combined = create_combined_model(classification_model)
    combined.summary()
    
    # Test with zeros
    test = tf.zeros([1, AUDIO_LENGTH], dtype=tf.float32)
    print(f"Test output (zeros): {combined(test).numpy()}")
    
    # Test with actual audio
    import librosa
    dep_dir = 'data/raw/voice_data/depression1'
    norm_dir = 'data/raw/voice_data/normal1'
    dep_files = sorted([f for f in os.listdir(dep_dir) if f.endswith('.wav')])[:3]
    norm_files = sorted([f for f in os.listdir(norm_dir) if f.endswith('.wav')])[:3]
    
    print("\n--- TF model predictions ---")
    for f in dep_files:
        audio, sr = librosa.load(os.path.join(dep_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        pred = combined(tf.constant(audio[np.newaxis], dtype=tf.float32))
        print(f"  Dep [{f}]: {pred.numpy()[0][0]:.6f}")
    
    for f in norm_files:
        audio, sr = librosa.load(os.path.join(norm_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        pred = combined(tf.constant(audio[np.newaxis], dtype=tf.float32))
        print(f"  Norm [{f}]: {pred.numpy()[0][0]:.6f}")
    
    # Convert to TFLite — try builtins-only first
    print("\n--- Converting to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(combined)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    needs_flex = False
    try:
        print("Trying builtins-only conversion...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        print("SUCCESS: Converted with TFLite builtins only!")
    except Exception as e:
        print(f"Builtins-only failed: {e}")
        print("\nRetrying with SELECT_TF_OPS (flex delegate)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(combined)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        needs_flex = True
        print("SUCCESS: Converted with SELECT_TF_OPS")
    
    output_path = 'artifacts/models/depression_detection_combined.tflite'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\nSaved: {output_path} ({size_kb:.1f} KB)")
    print(f"Needs flex delegate: {needs_flex}")
    
    # Verify TFLite model
    print("\n--- Verifying TFLite model ---")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")
    
    print("\nTFLite depression samples:")
    for f in dep_files[:5]:
        audio, sr = librosa.load(os.path.join(dep_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        interpreter.set_tensor(input_details[0]['index'], audio[np.newaxis].astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        label = "DEP" if out[0][0] >= 0.5 else "NORM"
        print(f"  {f}: {out[0][0]:.4f} [{label}]")
    
    print("\nTFLite normal samples:")
    for f in norm_files[:5]:
        audio, sr = librosa.load(os.path.join(norm_dir, f), sr=SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))))[:AUDIO_LENGTH]
        interpreter.set_tensor(input_details[0]['index'], audio[np.newaxis].astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        label = "DEP" if out[0][0] >= 0.5 else "NORM"
        print(f"  {f}: {out[0][0]:.4f} [{label}]")
    
    print(f"\nDone! Model at: {output_path}")


if __name__ == '__main__':
    main()
