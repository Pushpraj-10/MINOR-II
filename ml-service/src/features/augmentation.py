"""
Audio data augmentation for depression detection.

WHY AUGMENTATION?
The depressed class is heavily underrepresented (~16% of samples).
If we train as-is, the model learns to just predict "normal" for everything
and still scores 84% accuracy — useless for our purpose.

Augmentation artificially creates more depressed samples by slightly
modifying existing ones.  The modifications are small enough that the
new samples still sound like genuine depressed speech, but different
enough that the model can't just memorize the originals.

Provides waveform-level augmentations (pitch shift, time stretch,
noise injection) applied to raw audio segments *before* feature
extraction.  Augmentations are applied only to minority-class
(depression) segments to help balance the dataset.
"""

import numpy as np
from typing import List, Tuple, Optional


def pitch_shift(audio: np.ndarray, sr: int = 16000, n_steps: float = 1.0) -> np.ndarray:
    """
    Shift pitch by resampling.

    Changing pitch slightly (e.g. +1 semitone) creates a new sample that
    sounds like a slightly higher-pitched version of the same person.
    The model must learn features that are robust to small pitch differences,
    which improves generalization to new speakers.

    A positive n_steps raises pitch; negative lowers it.
    Uses simple linear-interpolation resampling (no external deps).
    """
    factor = 2.0 ** (n_steps / 12.0)  # semitone to frequency ratio conversion
    indices = np.arange(0, len(audio), factor)
    indices = indices[indices < len(audio) - 1]
    int_part = indices.astype(int)
    frac_part = indices - int_part
    shifted = audio[int_part] * (1 - frac_part) + audio[int_part + 1] * frac_part
    # Resize back to original length so input/output shapes match
    target_indices = np.linspace(0, len(shifted) - 1, len(audio))
    int_t = target_indices.astype(int)
    int_t = np.clip(int_t, 0, len(shifted) - 2)
    frac_t = target_indices - int_t
    return (shifted[int_t] * (1 - frac_t) + shifted[int_t + 1] * frac_t).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float = 0.9) -> np.ndarray:
    """
    Stretch/compress audio in time via linear interpolation.

    Makes the same speech slightly slower or faster without changing pitch.
    Simulates the natural variation in speech rate between people.
    Depressed speech is typically slower — but training only on slow speech
    would overfit to that exact speed.  This makes the model learn the
    *pattern* of slow speech, not just the literal length of pauses.

    rate < 1.0 → slower (longer), rate > 1.0 → faster (shorter).
    Output is resized to original length.
    """
    stretched_len = int(len(audio) / rate)  # temporary stretched length
    indices = np.linspace(0, len(audio) - 1, stretched_len)
    int_part = indices.astype(int)
    int_part = np.clip(int_part, 0, len(audio) - 2)
    frac_part = indices - int_part
    stretched = audio[int_part] * (1 - frac_part) + audio[int_part + 1] * frac_part
    # Resize back to original length
    target_indices = np.linspace(0, len(stretched) - 1, len(audio))
    int_t = target_indices.astype(int)
    int_t = np.clip(int_t, 0, len(stretched) - 2)
    frac_t = target_indices - int_t
    return (stretched[int_t] * (1 - frac_t) + stretched[int_t + 1] * frac_t).astype(np.float32)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to the audio signal.

    Simulates real-world recording conditions: background hum, mic noise, etc.
    Without this, the model might overfit to the clean studio-quality clips
    in EATD-Corpus and fail on noisier phone recordings.
    noise_factor=0.005 is subtle — barely audible to a human ear.
    """
    rng = np.random.default_rng()
    noise = rng.normal(0, noise_factor, size=len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)  # clip to valid range [-1, 1]


def augment_segment(
    audio: np.ndarray,
    sr: int = 16000,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate augmented copies of a single audio segment.

    Returns a list of 3 augmented versions:
      1. Pitch-shifted (random ±1–2 semitones)
      2. Time-stretched (random 0.85–1.15)
      3. Noise-injected (random 0.003–0.008)
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = []

    # Pitch shift
    steps = rng.choice([-2, -1, 1, 2])
    augmented.append(pitch_shift(audio, sr=sr, n_steps=steps))

    # Time stretch
    rate = rng.uniform(0.85, 1.15)
    augmented.append(time_stretch(audio, rate=rate))

    # Noise injection
    noise_level = rng.uniform(0.003, 0.008)
    augmented.append(add_noise(audio, noise_factor=noise_level))

    return augmented


def augment_minority_class(
    X: np.ndarray,
    y: np.ndarray,
    sr: int = 16000,
    random_state: int = 42,
    max_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment minority-class (depression, label=1) segments to reduce imbalance.

    Each depression segment gets up to 3 augmented copies (pitch, stretch, noise).
    The augmented data is appended to the original arrays.

    Args:
        X: Audio segments array, shape (N, audio_length).
        y: Binary labels array, shape (N,).
        sr: Sample rate.
        random_state: Random seed for reproducibility.
        max_ratio: If set (0 < max_ratio < 1), cap the minority class at this
            fraction of the final training set.  For example, max_ratio=0.40
            ensures depression never exceeds 40% of training samples, preventing
            the distribution mismatch that causes over-prediction at test time.

    Returns:
        (X_augmented, y_augmented) with augmented minority samples appended.
    """
    rng = np.random.default_rng(random_state)
    minority_mask = y == 1
    minority_X = X[minority_mask]
    n_orig_minority = int(minority_mask.sum())
    majority_count = int((y == 0).sum())

    aug_segments = []
    for seg in minority_X:
        aug_segments.extend(augment_segment(seg, sr=sr, rng=rng))

    if not aug_segments:
        return X, y

    # Cap augmented copies so minority class stays <= max_ratio of training data
    if max_ratio is not None and 0.0 < max_ratio < 1.0:
        max_minority = int(max_ratio / (1.0 - max_ratio) * majority_count)
        max_aug_copies = max(0, max_minority - n_orig_minority)
        if max_aug_copies < len(aug_segments):
            indices = rng.permutation(len(aug_segments))[:max_aug_copies]
            aug_segments = [aug_segments[i] for i in indices]

    if not aug_segments:
        return X, y

    X_aug = np.array(aug_segments, dtype=np.float32)
    y_aug = np.ones(len(aug_segments), dtype=np.float32)

    X_out = np.concatenate([X, X_aug], axis=0)
    y_out = np.concatenate([y, y_aug], axis=0)

    # Shuffle
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]
