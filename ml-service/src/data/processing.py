"""
Multi-feature extraction and processing for the EATD-Corpus.

Extracts five acoustic feature sets from each audio segment and saves
them as compressed .npz files under data/processed/:

    Feature            Shape per segment         Purpose
    ─────────────────  ────────────────────────   ──────────────────────────
    MFCC               (n_mfcc, time_steps)       Vocal tract characteristics
    Delta MFCC         (n_mfcc, time_steps)       Speech dynamics
    Chroma             (12, time_steps)            Pitch class distribution
    Spectral contrast  (7, time_steps)             Frequency energy variations
    Zero crossing rate (1, time_steps)             Voice activity

All features share the same time axis (EXPECTED_TIME_STEPS) so they can
be stacked for multi-feature model input.

Saved files:
    data/processed/train_features.npz  (X_mfcc, X_delta, X_chroma, X_contrast, X_zcr, y)
    data/processed/val_features.npz
    data/processed/test_features.npz
    data/processed/feature_metadata.npz  (feature shapes, config snapshot)
"""

import os
import logging
import numpy as np
import librosa

from src.data.eatd_loader import (
    load_eatd_corpus,
    build_speaker_independent_splits,
)
from src.features.augmentation import augment_minority_class
from src.config import (
    SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH,
    N_MFCC, EXPECTED_TIME_STEPS,
    VAL_SIZE, RANDOM_STATE,
    EATD_CORPUS_DIR,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed/EATD"


def _pad_or_truncate(feat: np.ndarray, target_time: int) -> np.ndarray:
    """Pad or truncate the last axis to target_time."""
    t = feat.shape[-1]
    if t < target_time:
        pad_width = [(0, 0)] * (feat.ndim - 1) + [(0, target_time - t)]
        return np.pad(feat, pad_width)
    return feat[..., :target_time]


def extract_multi_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mfcc: int = N_MFCC,
    target_time: int = EXPECTED_TIME_STEPS,
) -> dict:
    """
    Extract all five feature types from a single audio segment.

    Returns dict with keys: mfcc, delta_mfcc, chroma, spectral_contrast, zcr.
    Each value has shape (freq_bins, target_time).
    """
    # MFCC (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    mfcc = _pad_or_truncate(mfcc, target_time)

    # Delta MFCC (n_mfcc, T)
    delta_mfcc = librosa.feature.delta(mfcc)

    # Chroma (12, T)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr,
                                          n_fft=n_fft, hop_length=hop_length)
    chroma = _pad_or_truncate(chroma, target_time)

    # Spectral contrast (7, T)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                  n_fft=n_fft, hop_length=hop_length)
    contrast = _pad_or_truncate(contrast, target_time)

    # Zero crossing rate (1, T)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft,
                                              hop_length=hop_length)
    zcr = _pad_or_truncate(zcr, target_time)

    return {
        "mfcc": mfcc.astype(np.float32),
        "delta_mfcc": delta_mfcc.astype(np.float32),
        "chroma": chroma.astype(np.float32),
        "spectral_contrast": contrast.astype(np.float32),
        "zcr": zcr.astype(np.float32),
    }


def extract_multi_features_batch(
    audio_list,
    sr: int = SAMPLE_RATE,
    target_time: int = EXPECTED_TIME_STEPS,
) -> dict:
    """
    Extract multi-features for an entire list of audio segments.

    Returns dict of arrays, each with shape (N, freq_bins, target_time).
    """
    all_feats = {k: [] for k in ["mfcc", "delta_mfcc", "chroma", "spectral_contrast", "zcr"]}
    n = len(audio_list)

    for i, audio in enumerate(audio_list):
        feats = extract_multi_features(audio, sr=sr, target_time=target_time)
        for k in all_feats:
            all_feats[k].append(feats[k])
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  Extracted features for {i + 1}/{n} segments")

    return {k: np.array(v) for k, v in all_feats.items()}


def process_and_save(augment: bool = True, overlap: float = 0.5):
    """
    Full processing pipeline: load EATD audio → split → augment → extract
    multi-features → save to data/processed/.

    Args:
        augment: Whether to augment minority class in training set.
        overlap: Segment overlap fraction.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── 1. Load corpus and split ─────────────────────────────────────
    print("\n[1/4] Loading EATD-Corpus...")
    subjects = load_eatd_corpus(
        EATD_CORPUS_DIR,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        overlap=overlap,
    )
    splits = build_speaker_independent_splits(
        subjects, val_size=VAL_SIZE, random_state=RANDOM_STATE,
        return_subject_ids=True,
    )

    audio_train, y_train, ids_train = splits["train"]
    audio_val, y_val, ids_val = splits["val"]
    audio_test, y_test, ids_test = splits["test"]
    subject_names = splits["subject_names"]

    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # Save pre-augmentation counts for correct class weight computation
    pre_aug_dep = int(y_train.sum())
    pre_aug_norm = int((y_train == 0).sum())

    # ── 2. Augment training minority class ───────────────────────────
    if augment:
        print("\n[2/4] Augmenting minority class (depression)...")
        n_before_aug = len(y_train)
        dep_mask = y_train == 1

        audio_train_aug, y_train_aug = augment_minority_class(
            audio_train, y_train, sr=SAMPLE_RATE, random_state=RANDOM_STATE,
        )
        # Augmented copies carry the same subject ID as their source.
        # augment_minority_class appends 3 copies per dep segment then shuffles
        # with the same seed — replay that permutation to align ids.
        aug_ids_extra = np.tile(ids_train[dep_mask], 3)
        ids_train_combined = np.concatenate([ids_train, aug_ids_extra])
        rng_perm = np.random.default_rng(RANDOM_STATE)
        perm = rng_perm.permutation(len(y_train_aug))
        audio_train = audio_train_aug
        y_train = y_train_aug
        ids_train = ids_train_combined[perm]
        print(f"  Train segments: {n_before_aug} -> {len(y_train)}  "
              f"(dep={int(y_train.sum())}  norm={int((y_train == 0).sum())})")
    else:
        print("\n[2/4] Skipping augmentation.")

    # ── 3. Extract multi-features for each split ─────────────────────
    print("\n[3/4] Extracting multi-features (MFCC, Delta MFCC, Chroma, "
          "Spectral Contrast, ZCR)...")

    split_data = {
        "train": (audio_train, y_train, ids_train),
        "val": (audio_val, y_val, ids_val),
        "test": (audio_test, y_test, ids_test),
    }

    for split_name, (audio_arr, labels, subj_ids) in split_data.items():
        print(f"\n  --- {split_name} ({len(labels)} segments) ---")
        feats = extract_multi_features_batch(list(audio_arr))

        out_path = os.path.join(PROCESSED_DIR, f"{split_name}_features.npz")
        np.savez_compressed(
            out_path,
            X_mfcc=feats["mfcc"],
            X_delta_mfcc=feats["delta_mfcc"],
            X_chroma=feats["chroma"],
            X_spectral_contrast=feats["spectral_contrast"],
            X_zcr=feats["zcr"],
            y=labels,
            subject_ids=subj_ids,
        )
        print(f"  Saved {out_path}")
        for k, v in feats.items():
            print(f"    {k}: {v.shape}")

    # ── 4. Save metadata ─────────────────────────────────────────────
    print("\n[4/4] Saving metadata...")
    meta_path = os.path.join(PROCESSED_DIR, "feature_metadata.npz")
    np.savez(
        meta_path,
        feature_names=np.array(["mfcc", "delta_mfcc", "chroma", "spectral_contrast", "zcr"]),
        feature_shapes=np.array([
            (N_MFCC, EXPECTED_TIME_STEPS),          # mfcc
            (N_MFCC, EXPECTED_TIME_STEPS),          # delta_mfcc
            (12, EXPECTED_TIME_STEPS),               # chroma
            (7, EXPECTED_TIME_STEPS),                # spectral contrast
            (1, EXPECTED_TIME_STEPS),                # zcr
        ]),
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC,
        target_time_steps=EXPECTED_TIME_STEPS,
        augmented=augment,
        pre_aug_train_dep=pre_aug_dep,
        pre_aug_train_norm=pre_aug_norm,
        subject_names=subject_names,
    )
    print(f"  Saved {meta_path}")
    print("\nDone! Processed features saved to data/processed/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process EATD-Corpus into multi-feature arrays")
    parser.add_argument("--no-augment", action="store_true", help="Skip minority class augmentation")
    parser.add_argument("--overlap", type=float, default=0.5, help="Segment overlap (0.0–0.9)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_and_save(augment=not args.no_augment, overlap=args.overlap)