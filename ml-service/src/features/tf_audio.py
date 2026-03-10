"""
Mel spectrogram feature extraction using tf.signal.

This module provides a single source of truth for the mel spectrogram
pipeline used by all model architectures. Using tf.signal (not librosa)
ensures the features match exactly what the TFLite combined model will
compute on-device.

Feature shapes produced:
    - CNN-style:  (n_mels, time_steps)        → add channel dim → (n_mels, time, 1)
    - LSTM-style: (time_steps, n_mels)         → feed directly to BiLSTM
    - MFCC:       (n_mfcc, time_steps)          → add channel dim → (n_mfcc, time, 1)
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple


def compute_mel_weights(
    n_mels: int = 128,
    n_fft: int = 512,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Precompute mel filterbank weight matrix."""
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=f_min,
        upper_edge_hertz=f_max,
        dtype=tf.float32,
    ).numpy()


def extract_mel_spectrogram(
    audio: np.ndarray,
    mel_weights: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    transpose: bool = True,
) -> np.ndarray:
    """
    Extract log-mel spectrogram using tf.signal ops.

    Matches the TFLite combined model preprocessing exactly:
        STFT → power spectrum → mel filterbank → power_to_dB (ref=max)

    Args:
        audio: 1-D audio array (already padded to target length).
        mel_weights: Precomputed mel filterbank matrix.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        transpose: If True, return (n_mels, time) for CNN input.
                   If False, return (time, n_mels) for LSTM input.

    Returns:
        Log-mel spectrogram as numpy array.
    """
    audio_tf = tf.constant(audio[np.newaxis], dtype=tf.float32)

    stft = tf.signal.stft(
        audio_tf,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )

    power = tf.math.square(tf.abs(stft))
    mel = tf.matmul(power, tf.constant(mel_weights, dtype=tf.float32))

    # power_to_dB with ref=max (matching librosa.power_to_db)
    amin = 1e-10
    ref = tf.reduce_max(mel, axis=[1, 2], keepdims=True)
    ref = tf.maximum(ref, amin)
    mel = tf.maximum(mel, amin)
    log10 = tf.math.log(10.0)
    mel_db = 10.0 * (tf.math.log(mel) - tf.math.log(ref)) / log10
    mel_db = tf.maximum(mel_db, -80.0)

    mel_db = tf.squeeze(mel_db, axis=0)  # (time, n_mels)

    if transpose:
        mel_db = tf.transpose(mel_db)  # (n_mels, time)

    return mel_db.numpy()


def extract_mfcc(
    audio: np.ndarray,
    mel_weights: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Extract MFCCs via tf.signal: mel spectrogram → log → DCT.

    Args:
        audio: 1-D audio array.
        mel_weights: Precomputed mel filterbank matrix.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_mfcc: Number of MFCC coefficients to keep.

    Returns:
        MFCCs with shape (n_mfcc, time_steps).
    """
    audio_tf = tf.constant(audio[np.newaxis], dtype=tf.float32)

    stft = tf.signal.stft(
        audio_tf,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power = tf.math.square(tf.abs(stft))
    mel = tf.matmul(power, tf.constant(mel_weights, dtype=tf.float32))

    mel = tf.maximum(mel, 1e-10)
    log_mel = tf.math.log(mel)

    mfccs = tf.signal.dct(log_mel, type=2, norm="ortho")
    mfccs = mfccs[:, :, :n_mfcc]

    mfccs = tf.squeeze(mfccs, axis=0)  # (time, n_mfcc)
    mfccs = tf.transpose(mfccs)  # (n_mfcc, time)
    return mfccs.numpy()


def extract_features_batch(
    audio_list: List[np.ndarray],
    mel_weights: np.ndarray,
    expected_time_steps: int = 313,
    feature_type: str = "mel",
    n_fft: int = 512,
    hop_length: int = 256,
    n_mfcc: int = 13,
    output_format: str = "cnn",
) -> np.ndarray:
    """
    Extract features for an entire dataset.

    Args:
        audio_list: List of 1-D audio arrays.
        mel_weights: Precomputed mel filterbank.
        expected_time_steps: Pad/truncate time axis to this length.
        feature_type: "mel" for mel spectrogram, "mfcc" for MFCC.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_mfcc: Number of MFCCs (only used when feature_type="mfcc").
        output_format: "cnn" → (N, freq, time, 1), "lstm" → (N, time, freq).

    Returns:
        Feature array ready for model input.
    """
    features = []
    use_transpose = (output_format == "cnn")

    for i, audio in enumerate(audio_list):
        if feature_type == "mfcc":
            feat = extract_mfcc(audio, mel_weights, n_fft, hop_length, n_mfcc)
            # feat shape: (n_mfcc, time) — always transposed
        else:
            feat = extract_mel_spectrogram(
                audio, mel_weights, n_fft, hop_length, transpose=use_transpose
            )

        # Pad/truncate time axis
        time_axis = 1 if use_transpose or feature_type == "mfcc" else 0
        time_len = feat.shape[time_axis]

        if time_len < expected_time_steps:
            pad_width = [(0, 0)] * feat.ndim
            pad_width[time_axis] = (0, expected_time_steps - time_len)
            feat = np.pad(feat, pad_width)
        else:
            slices = [slice(None)] * feat.ndim
            slices[time_axis] = slice(0, expected_time_steps)
            feat = feat[tuple(slices)]

        features.append(feat)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(audio_list)}")

    X = np.array(features)

    if output_format == "cnn":
        X = X[..., np.newaxis]  # add channel dim

    return X


def extract_dual_features_batch(
    audio_list: List[np.ndarray],
    mel_weights: np.ndarray,
    expected_time_steps: int = 313,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract both mel spectrogram and MFCC features for dual-branch architectures.

    Calls extract_features_batch twice so both arrays share the same padding
    and time-axis length.

    Returns:
        X_mel:  shape (N, n_mels, time, 1) — ready for mel CNN branch.
        X_mfcc: shape (N, n_mfcc, time, 1) — ready for MFCC CNN branch.
    """
    X_mel = extract_features_batch(
        audio_list, mel_weights, expected_time_steps,
        feature_type="mel", n_fft=n_fft, hop_length=hop_length,
        output_format="cnn",
    )
    X_mfcc = extract_features_batch(
        audio_list, mel_weights, expected_time_steps,
        feature_type="mfcc", n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc,
        output_format="cnn",
    )
    return X_mel, X_mfcc
