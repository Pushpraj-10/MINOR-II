"""
v2 Processing Pipeline — CMVN + Speaker-disjoint Validation

Key changes vs process_combined.py:

  1. Speaker-disjoint validation:
       All t_* EATD subjects → train (no speaker leakage into val).
       v_* subjects split: 25% → val, 75% → test.
       This ensures early stopping in training monitors on truly unseen speakers,
       giving a more honest gradient and stopping criterion.

  2. CMVN (Cepstral Mean/Variance Normalisation):
       Per-utterance mean/variance normalisation applied to MFCC (bins 0–12)
       and delta-MFCC (bins 13–25) before global z-score normalisation.
       Removes speaker-level channel offsets so the model learns relative
       spectral patterns rather than absolute speaker characteristics.

  3. Subject ID tracking:
       Test set retains EATD subject IDs for subject-level evaluation
       (aggregate segment predictions per speaker).

Output: data/processed/combined_v2/
"""

import os
import sys
import logging
import argparse
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.eatd_loader import load_eatd_corpus
from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset
from src.features.augmentation import augment_minority_class
from src.data.processing import extract_multi_features_batch
from src.config import (
    SAMPLE_RATE, DURATION, AUDIO_LENGTH,
    N_FFT, HOP_LENGTH, N_MFCC, EXPECTED_TIME_STEPS,
    VAL_SIZE, RANDOM_STATE, EATD_CORPUS_DIR,
)

logger = logging.getLogger(__name__)

DATASET_1_DIR = os.path.join("data", "raw", "DATASET_1")
RAVDESS_DIR   = os.path.join("data", "raw", "The Ryerson Audio-Visual Dataset")
OUTPUT_DIR    = os.path.join("data", "processed", "combined_v2")

DEPRESSION_EMOTIONS = {"04"}   # sad
NORMAL_EMOTIONS     = {"01"}   # neutral


# ──────────────────────────────────────────────────────────────────────────────
# CMVN
# ──────────────────────────────────────────────────────────────────────────────

def apply_cmvn(X: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Per-utterance Cepstral Mean/Variance Normalisation (CMVN).

    Problem: different microphones, rooms, and people have a constant
    offset in their MFCC values.  Person A might always have higher
    MFCC values than Person B just because of their microphone —
    not because they are depressed.

    CMVN fixes this by subtracting each clip's own mean and dividing
    by its own standard deviation (per MFCC bin, across time).
    After CMVN, the model only sees *relative* patterns in the voice,
    not absolute microphone-level offsets — so it generalizes better.

    Normalises MFCC (bins 0..n_mfcc-1) and delta-MFCC (bins n_mfcc..2*n_mfcc-1)
    independently per utterance over the time axis.  Leaves chroma, spectral
    contrast, and ZCR bins untouched.

    Args:
        X:      Feature array of shape (N, 46, 313).
        n_mfcc: Number of MFCC coefficients (default 13 → acts on bins 0–25).

    Returns:
        Copy of X with CMVN applied to the first 2*n_mfcc frequency bins.
    """
    X_out = X.copy()
    for i in range(len(X_out)):
        for dim in range(n_mfcc * 2):   # MFCC (0-12) + delta-MFCC (13-25)
            row = X_out[i, dim, :]      # one frequency bin across all 313 time steps
            m = row.mean()              # average value of this bin for this clip
            s = row.std()               # spread of values for this bin
            X_out[i, dim, :] = (row - m) / (s + 1e-8)  # normalize: zero mean, unit variance
    return X_out


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_1():
    """Load DATASET_1 and produce train / val / test splits."""
    print("\n  Loading DATASET_1 (depression1/normal1)...")
    loader = AudioDataLoader(
        data_dir=DATASET_1_DIR, sample_rate=SAMPLE_RATE,
        duration=DURATION, mono=True,
    )
    audio_list, labels, _ = loader.load_dataset(
        depression_dir="depression1", normal_dir="normal1",
    )
    X = np.array(audio_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    splits = split_dataset(X, y, test_size=0.15, val_size=0.15,
                           random_state=RANDOM_STATE)
    for name in ["train", "val", "test"]:
        _, ys = splits[name]
        print(f"    DS1 {name}: {len(ys)} "
              f"(dep={int(ys.sum())}, norm={int((ys==0).sum())})")
    return splits


def build_v2_eatd_splits(subjects: dict,
                          v_val_frac: float = 0.25,
                          random_state: int = RANDOM_STATE):
    """
    Speaker-disjoint splits for EATD-Corpus:
      - All t_* subjects → train  (no speaker leakage)
      - v_* subjects: v_val_frac  → val  (speaker-disjoint)
                      remainder   → test
    """
    t_ids = sorted(k for k, v in subjects.items() if v["split_prefix"] == "t")
    v_ids = sorted(k for k, v in subjects.items() if v["split_prefix"] == "v")

    v_labels = np.array([subjects[k]["label"] for k in v_ids])
    v_val_ids, v_test_ids = train_test_split(
        v_ids,
        test_size=(1.0 - v_val_frac),
        stratify=v_labels,
        random_state=random_state,
    )

    print(f"\n  EATD speaker-disjoint split (v2):")
    print(f"    t_* ({len(t_ids)} subjects) → train  [ALL]")
    print(f"    v_* val  ({len(v_val_ids)} subjects) → val")
    print(f"    v_* test ({len(v_test_ids)} subjects) → test")

    def collect(id_list, tag, track_ids=False):
        segs, labs, sids = [], [], []
        for sid in id_list:
            s = subjects[sid]
            segs.extend(s["segments"])
            labs.extend([s["label"]] * len(s["segments"]))
            if track_ids:
                sids.extend([sid] * len(s["segments"]))
        X = np.array(segs, dtype=np.float32)
        y = np.array(labs, dtype=np.float32)
        print(f"    {tag}: {len(y)} segs "
              f"(dep={int(y.sum())}, norm={int((y==0).sum())})")
        return X, y, np.array(sids)

    train_X, train_y, _         = collect(t_ids,      "EATD train")
    val_X,   val_y,   _         = collect(v_val_ids,  "EATD val  (speaker-disjoint)")
    test_X,  test_y,  test_sids = collect(v_test_ids, "EATD test (speaker-disjoint)",
                                          track_ids=True)
    return {
        "train": (train_X, train_y),
        "val":   (val_X, val_y),
        "test":  (test_X, test_y, test_sids),
    }


def load_ravdess():
    """Load RAVDESS as a held-out cross-domain evaluation set."""
    print("\n  Loading RAVDESS (hold-out cross-domain)...")
    audio_list, labels = [], []
    ravdess_path = os.path.join(PROJECT_ROOT, RAVDESS_DIR)

    for actor_dir in sorted(Path(ravdess_path).glob("Actor_*")):
        for wav_file in sorted(actor_dir.glob("*.wav")):
            parts = wav_file.stem.split("-")
            if len(parts) != 7:
                continue
            emotion = parts[2]
            if emotion in DEPRESSION_EMOTIONS:
                label = 1
            elif emotion in NORMAL_EMOTIONS:
                label = 0
            else:
                continue
            try:
                audio, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                if len(audio) < AUDIO_LENGTH:
                    audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
                else:
                    audio = audio[:AUDIO_LENGTH]
                audio_list.append(audio.astype(np.float32))
                labels.append(label)
            except Exception as e:
                print(f"    Failed: {wav_file}: {e}")

    X = np.array(audio_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"    RAVDESS: {len(y)} "
          f"(dep/sad={int(y.sum())}, norm/neutral={int((y==0).sum())})")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_scaler(X: np.ndarray):
    """Per-frequency-bin mean and std over all training samples and time.

    This is the global Z-score scaler.  Unlike CMVN (which is per-clip),
    this computes statistics across the ENTIRE training set.
    It ensures that across all 46 feature rows, each row has approximately
    zero mean and unit variance — so no single feature dominates the CNN
    just because it happens to have larger numbers.

    CRITICAL: these stats are computed ONLY from training data.
    They are then applied to validation, test, and live audio too.
    If we recomputed them on val/test, the model would see information
    from those splits during preprocessing — that's called data leakage.
    """
    mean = X.mean(axis=(0, 2), keepdims=True)   # mean per frequency bin (across all clips and time)
    std  = X.std(axis=(0, 2),  keepdims=True)   # std per frequency bin
    std  = np.where(std < 1e-8, 1.0, std)        # avoid division by zero for silent/flat features
    return mean.squeeze(), std.squeeze()   # (46,), (46,)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # Apply the saved scaler: shift and scale each of the 46 feature rows
    return (X - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v2 processing: CMVN + speaker-disjoint validation"
    )
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip minority-class augmentation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("  Combined Multi-Feature Processing Pipeline  v2")
    print("  CMVN + Speaker-disjoint Validation")
    print("=" * 70)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets...")
    subjects   = load_eatd_corpus(EATD_CORPUS_DIR, sample_rate=SAMPLE_RATE,
                                   duration=DURATION, overlap=0.5)
    eatd_splits = build_v2_eatd_splits(subjects)
    ds1_splits  = load_dataset_1()
    rav_X, rav_y = load_ravdess()

    # ── 2. Combine EATD + DS1 for train / val ────────────────────────────────
    print("\n[2/6] Combining EATD + DATASET_1...")
    combined = {}
    for split_name in ["train", "val"]:
        eatd_X, eatd_y  = eatd_splits[split_name]
        ds1_X,  ds1_y   = ds1_splits[split_name]
        X = np.concatenate([eatd_X, ds1_X], axis=0)
        y = np.concatenate([eatd_y, ds1_y], axis=0)
        rng  = np.random.default_rng(RANDOM_STATE + hash(split_name) % 1000)
        perm = rng.permutation(len(y))
        combined[split_name] = (X[perm], y[perm])
        print(f"    {split_name}: {len(y)} "
              f"(dep={int(y.sum())}, norm={int((y==0).sum())})")

    # Test: EATD (with subject IDs) + DS1
    eatd_test_X, eatd_test_y, eatd_test_sids = eatd_splits["test"]
    ds1_test_X,  ds1_test_y                  = ds1_splits["test"]
    ds1_sids = np.array(["DS1"] * len(ds1_test_y))

    test_X    = np.concatenate([eatd_test_X,    ds1_test_X],    axis=0)
    test_y    = np.concatenate([eatd_test_y,    ds1_test_y],    axis=0)
    test_sids = np.concatenate([eatd_test_sids, ds1_sids],      axis=0)

    rng  = np.random.default_rng(RANDOM_STATE + 999)
    perm = rng.permutation(len(test_y))
    combined["test"]      = (test_X[perm], test_y[perm])
    test_sids_shuffled    = test_sids[perm]
    print(f"    test: {len(test_y)} "
          f"(dep={int(test_y.sum())}, norm={int((test_y==0).sum())})")

    # ── 3. Augment minority class in training set ────────────────────────────
    X_tr, y_tr = combined["train"]
    pre_dep  = int(y_tr.sum())
    pre_norm = int((y_tr == 0).sum())
    if not args.no_augment:
        print("\n[3/6] Augmenting minority class (depression) in training set...")
        X_tr, y_tr = augment_minority_class(
            X_tr, y_tr, sr=SAMPLE_RATE,
            random_state=RANDOM_STATE, max_ratio=0.45,
        )
        combined["train"] = (X_tr, y_tr)
        print(f"    After aug: {len(y_tr)} "
              f"(dep={int(y_tr.sum())}, norm={int((y_tr==0).sum())})")
    else:
        print("\n[3/6] Skipping augmentation.")

    # ── 4. Extract multi-features ────────────────────────────────────────────
    print("\n[4/6] Extracting multi-features...")

    def extract_and_stack(X_audio, tag):
        print(f"  [{tag}] {len(X_audio)} segments...")
        feats = extract_multi_features_batch(
            list(X_audio), sr=SAMPLE_RATE, target_time=EXPECTED_TIME_STEPS
        )
        X_stacked = np.concatenate([
            feats["mfcc"],               # (N, 13, 313)
            feats["delta_mfcc"],         # (N, 13, 313)
            feats["chroma"],             # (N, 12, 313)
            feats["spectral_contrast"],  # (N,  7, 313)
            feats["zcr"],                # (N,  1, 313)
        ], axis=1)                       # → (N, 46, 313)
        print(f"    shape: {X_stacked.shape}")
        return X_stacked

    combined["ravdess"] = (rav_X, rav_y)
    feat_sets = {}
    for name in ["train", "val", "test", "ravdess"]:
        X_audio, y_split = combined[name]
        feat_sets[name] = (extract_and_stack(X_audio, name), y_split)

    # ── 5. CMVN → global z-score ─────────────────────────────────────────────
    print("\n[5/6] Normalising: CMVN → global z-score...")
    print("  Applying per-utterance CMVN to MFCC/delta-MFCC channels (bins 0–25)...")
    for name in ["train", "val", "test", "ravdess"]:
        X_f, y_f = feat_sets[name]
        feat_sets[name] = (apply_cmvn(X_f), y_f)

    X_train_feat = feat_sets["train"][0]
    mean, std = compute_scaler(X_train_feat)
    print("  Applying global z-score normalisation...")
    for name in ["train", "val", "test", "ravdess"]:
        X_f, y_f = feat_sets[name]
        feat_sets[name] = (normalize(X_f, mean, std), y_f)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f"\n[6/6] Saving to {OUTPUT_DIR}/")
    for name in ["train", "val", "test", "ravdess"]:
        X_f, y_f = feat_sets[name]
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"{name}_features.npz"), X=X_f, y=y_f)
        print(f"  {name}: {X_f.shape}")
    np.savez_compressed(os.path.join(OUTPUT_DIR, "scaler.npz"),
                        mean=mean, std=std)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test_subject_ids.npz"),
                        subject_ids=test_sids_shuffled)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "metadata.npz"),
                        pre_aug_train_dep=pre_dep,
                        pre_aug_train_norm=pre_norm)

    print("\nDone.")
    print(f"  CMVN applied to MFCC/delta-MFCC dims (0-{N_MFCC*2-1})")
    print(f"  Speaker-disjoint val: v_* only")
    print(f"  Subject IDs saved for test set subject-level evaluation")


if __name__ == "__main__":
    main()
