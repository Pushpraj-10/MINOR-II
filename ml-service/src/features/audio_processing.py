"""Audio feature extraction for depression detection."""

import librosa
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default feature extraction configuration
DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_fft": 512,
    "hop_length": 256,
    "n_mels": 128,
}


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features from audio signal.

    Args:
        audio: Audio time series array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        normalize: Whether to normalize features (zero mean, unit variance)

    Returns:
        MFCC features with shape (n_mfcc, time_steps)

    Raises:
        ValueError: If audio is empty or invalid
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    if normalize:
        mean = np.mean(mfccs, axis=1, keepdims=True)
        std = np.std(mfccs, axis=1, keepdims=True) + 1e-8
        mfccs = (mfccs - mean) / std

    return mfccs


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Extract mel-spectrogram features from audio signal.

    Args:
        audio: Audio time series array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands

    Returns:
        Mel-spectrogram in dB with shape (n_mels, time_steps)
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def extract_features_from_dataset(
    audio_list: List[np.ndarray],
    feature_type: str = "mfcc",
    config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Extract features from a list of audio arrays.

    Args:
        audio_list: List of audio numpy arrays
        feature_type: Type of features ('mfcc' or 'mel_spectrogram')
        config: Feature extraction config (uses DEFAULT_CONFIG if None)

    Returns:
        Feature array with shape (n_samples, n_features, time_steps, 1)
        Ready for CNN input.
    """
    if config is None:
        config = DEFAULT_CONFIG

    features = []
    failed = 0

    for i, audio in enumerate(audio_list):
        try:
            if feature_type == "mfcc":
                feat = extract_mfcc(
                    audio,
                    sr=config.get("sample_rate", 16000),
                    n_mfcc=config.get("n_mfcc", 13),
                    n_fft=config.get("n_fft", 512),
                    hop_length=config.get("hop_length", 256),
                    normalize=True,
                )
            elif feature_type == "mel_spectrogram":
                feat = extract_mel_spectrogram(
                    audio,
                    sr=config.get("sample_rate", 16000),
                    n_fft=config.get("n_fft", 512),
                    hop_length=config.get("hop_length", 256),
                    n_mels=config.get("n_mels", 128),
                )
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            features.append(feat)

        except Exception as e:
            logger.warning(f"Failed to extract features for sample {i}: {e}")
            failed += 1

    if not features:
        raise RuntimeError("No features could be extracted from the dataset")

    # Ensure all features have the same shape (pad shorter ones)
    max_time = max(f.shape[1] for f in features)
    padded_features = []
    for feat in features:
        if feat.shape[1] < max_time:
            pad_width = max_time - feat.shape[1]
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")
        padded_features.append(feat)

    # Stack and add channel dimension for CNN: (samples, features, time, 1)
    X = np.array(padded_features)[..., np.newaxis]

    logger.info(
        f"Extracted {feature_type} features: shape={X.shape}, failed={failed}"
    )

    return X
