"""
TFLite model export: combined preprocessing + classification models.

Single Responsibility: Build combined models (raw audio -> prediction)
and convert them to TFLite format for mobile deployment.

Each builder handles one model topology's preprocessing graph.
The convert_to_tflite() function handles the actual conversion with
automatic fallback from builtins-only to flex delegate.
"""

import tensorflow as tf
from tensorflow import keras
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from src.config import (
    SAMPLE_RATE, DURATION, AUDIO_LENGTH,
    N_FFT, HOP_LENGTH, N_MELS, N_MFCC,
    F_MIN, F_MAX, EXPECTED_TIME_STEPS,
)


def _mel_preprocessing_graph(audio_input, mel_weights_np, transpose=True):
    """
    Shared mel spectrogram preprocessing sub-graph.

    Returns tensor of shape:
        transpose=True:  (batch, n_mels, time)
        transpose=False: (batch, time, n_mels)
    """
    x = tf.signal.stft(
        audio_input,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    x = tf.math.square(tf.abs(x))
    mel_w = tf.constant(mel_weights_np, dtype=tf.float32)
    x = tf.matmul(x, mel_w)

    log10 = tf.math.log(10.0)
    amin = 1e-10
    ref = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    x = tf.maximum(x, amin)
    x = 10.0 * (tf.math.log(x) - tf.math.log(ref)) / log10
    x = tf.maximum(x, -80.0)

    if transpose:
        x = tf.transpose(x, perm=[0, 2, 1])  # (batch, n_mels, time)

    return x


def _pad_time_axis(x, expected_steps, time_axis):
    """Truncate or pad the time dimension to expected_steps."""
    if time_axis == 2:
        x = x[:, :, :expected_steps]
        pad_size = expected_steps - tf.shape(x)[2]
        paddings = tf.stack([
            tf.constant([0, 0]),
            tf.constant([0, 0]),
            tf.stack([tf.constant(0), pad_size]),
        ])
    else:  # time_axis == 1
        x = x[:, :expected_steps, :]
        pad_size = expected_steps - tf.shape(x)[1]
        paddings = tf.stack([
            tf.constant([0, 0]),
            tf.stack([tf.constant(0), pad_size]),
            tf.constant([0, 0]),
        ])
    return tf.pad(x, paddings)


def _get_mel_weights():
    """Compute mel filterbank weights."""
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=F_MIN,
        upper_edge_hertz=F_MAX,
        dtype=tf.float32,
    ).numpy()


# ── Combined model builders (one per input topology) ─────────────────

def build_combined_cnn(classification_model, model_name="depression_detector"):
    """
    Combined model for CNN-based architectures (mel_cnn, cnn_lstm,
    cnn_attention, separable_cnn).

    audio (80000,) -> mel (128, 313, 1) -> CNN -> prediction
    """
    mel_weights_np = _get_mel_weights()
    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")

    x = _mel_preprocessing_graph(audio_input, mel_weights_np, transpose=True)
    x = _pad_time_axis(x, EXPECTED_TIME_STEPS, time_axis=2)
    x = tf.reshape(x, [-1, N_MELS, EXPECTED_TIME_STEPS])
    x = tf.expand_dims(x, axis=-1)

    output = classification_model(x, training=False)
    return keras.Model(inputs=audio_input, outputs=output, name=model_name)


def build_combined_lstm(classification_model, model_name="lstm_depression_detector"):
    """
    Combined model for LSTM-based architectures.

    audio (80000,) -> mel (313, 128) -> LSTM -> prediction
    """
    mel_weights_np = _get_mel_weights()
    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")

    x = _mel_preprocessing_graph(audio_input, mel_weights_np, transpose=False)
    x = _pad_time_axis(x, EXPECTED_TIME_STEPS, time_axis=1)
    x = tf.reshape(x, [-1, EXPECTED_TIME_STEPS, N_MELS])

    output = classification_model(x, training=False)
    return keras.Model(inputs=audio_input, outputs=output, name=model_name)


def build_combined_dual_branch(classification_model, model_name="multi_feature_detector"):
    """
    Combined model for dual-branch (mel + MFCC) architectures.

    audio (80000,) -> [mel (128, 313, 1), mfcc (13, 313, 1)] -> CNN -> prediction
    """
    mel_weights_np = _get_mel_weights()
    audio_input = keras.Input(shape=(AUDIO_LENGTH,), dtype=tf.float32, name="audio_input")

    # Shared STFT
    stft = tf.signal.stft(
        audio_input,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power = tf.math.square(tf.abs(stft))
    mel_w = tf.constant(mel_weights_np, dtype=tf.float32)
    mel_raw = tf.matmul(power, mel_w)

    # Mel spectrogram branch
    log10 = tf.math.log(10.0)
    amin = 1e-10
    ref = tf.reduce_max(mel_raw, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    mel_safe = tf.maximum(mel_raw, amin)
    mel_db = 10.0 * (tf.math.log(mel_safe) - tf.math.log(ref)) / log10
    mel_db = tf.maximum(mel_db, -80.0)
    mel_db = tf.transpose(mel_db, perm=[0, 2, 1])
    mel_db = _pad_time_axis(mel_db, EXPECTED_TIME_STEPS, time_axis=2)
    mel_db = tf.reshape(mel_db, [-1, N_MELS, EXPECTED_TIME_STEPS])
    mel_4d = tf.expand_dims(mel_db, axis=-1)

    # MFCC branch
    mel_log = tf.maximum(mel_raw, 1e-10)
    mel_log = tf.math.log(mel_log)
    mfccs = tf.signal.dct(mel_log, type=2, norm="ortho")
    mfccs = mfccs[:, :, :N_MFCC]
    mfccs = tf.transpose(mfccs, perm=[0, 2, 1])
    mfccs = _pad_time_axis(mfccs, EXPECTED_TIME_STEPS, time_axis=2)
    mfccs = tf.reshape(mfccs, [-1, N_MFCC, EXPECTED_TIME_STEPS])
    mfcc_4d = tf.expand_dims(mfccs, axis=-1)

    output = classification_model([mel_4d, mfcc_4d], training=False)
    return keras.Model(inputs=audio_input, outputs=output, name=model_name)


# ── TFLite conversion ────────────────────────────────────────────────

def convert_to_tflite(model, output_path: str) -> bytes:
    """
    Convert a Keras model to TFLite.

    Tries builtins-only first for maximum mobile compatibility.
    Falls back to SELECT_TF_OPS (flex delegate) if needed.

    Returns:
        The raw tflite model bytes.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    needs_flex = False
    try:
        logger.info("Trying builtins-only TFLite conversion...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        logger.info("Converted with TFLite builtins only.")
    except Exception as e:
        logger.warning("Builtins-only failed: %s", e)
        logger.info("Retrying with SELECT_TF_OPS...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        needs_flex = True
        logger.info("Converted with SELECT_TF_OPS (flex delegate required).")

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    logger.info("Saved: %s (%.1f KB, flex=%s)", output_path, size_kb, needs_flex)
    return tflite_model
