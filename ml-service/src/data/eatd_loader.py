"""
EATD-Corpus data loader for depression detection.

The EATD-Corpus contains audio interviews from 162 subjects, each with
SDS (Zung Self-Rating Depression Scale) scores.  Subjects with SDS >= 53
are labelled as depressed.

Each subject has 3 VAD-processed audio files (*_out.wav): positive,
negative, and neutral interview responses.  These are segmented into
fixed-length windows to match the existing feature pipeline.

Speaker-independent splitting is enforced: all segments from one subject
stay in the same split.
"""

import os
import wave
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

SDS_THRESHOLD = 53  # standard clinical cut-off


def _read_sds_score(subject_dir: str) -> float:
    """Read SDS score from new_label.txt (preferred) or label.txt."""
    new_label = os.path.join(subject_dir, "new_label.txt")
    if os.path.exists(new_label):
        with open(new_label) as f:
            return float(f.read().strip())
    with open(os.path.join(subject_dir, "label.txt")) as f:
        return float(f.read().strip())


def _load_wav_mono(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Load a WAV file and convert to mono float32 normalised to [-1, 1]."""
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        if sr != target_sr:
            # Simple resampling via linear interpolation (already 16 kHz in practice)
            ratio = target_sr / sr
            indices = np.arange(0, len(audio), 1 / ratio)
            indices = indices[indices < len(audio) - 1].astype(int)
            audio = audio[indices]
        return audio
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _segment_audio(audio: np.ndarray, window_length: int, hop_length: int) -> List[np.ndarray]:
    """Split audio into fixed-length overlapping windows, padding the last if needed."""
    segments = []
    start = 0
    while start < len(audio):
        end = start + window_length
        segment = audio[start:end]
        if len(segment) < window_length:
            segment = np.pad(segment, (0, window_length - len(segment)))
        segments.append(segment)
        start += hop_length
    return segments


def load_eatd_corpus(
    corpus_dir: str,
    sample_rate: int = 16000,
    duration: float = 5.0,
    overlap: float = 0.5,
    audio_files: Tuple[str, ...] = ("positive_out.wav", "negative_out.wav", "neutral_out.wav"),
) -> Dict[str, dict]:
    """
    Load the EATD-Corpus and return per-subject data.

    Args:
        corpus_dir: Path to the EATD-Corpus root containing t_*/v_* dirs.
        sample_rate: Target sample rate.
        duration: Segment window length in seconds.
        overlap: Overlap fraction between consecutive windows (0.0–0.9).
        audio_files: Which WAV files to load per subject.

    Returns:
        Dict mapping subject_id -> {
            "segments": list of np.ndarray (each of length window_length),
            "label": 0 or 1,
            "sds_score": float,
            "split_prefix": "t" or "v",
        }
    """
    window_length = int(sample_rate * duration)
    hop = int(window_length * (1 - overlap))

    subjects = {}
    for entry in sorted(os.listdir(corpus_dir)):
        if not (entry.startswith("t_") or entry.startswith("v_")):
            continue
        subj_dir = os.path.join(corpus_dir, entry)
        if not os.path.isdir(subj_dir):
            continue

        sds = _read_sds_score(subj_dir)
        label = 1 if sds >= SDS_THRESHOLD else 0
        split_prefix = entry.split("_")[0]  # "t" or "v"

        segments = []
        for wav_name in audio_files:
            wav_path = os.path.join(subj_dir, wav_name)
            if not os.path.exists(wav_path):
                continue
            audio = _load_wav_mono(wav_path, target_sr=sample_rate)
            if audio is None or len(audio) == 0:
                continue
            segments.extend(_segment_audio(audio, window_length, hop))

        if not segments:
            logger.warning("No audio segments for subject %s, skipping.", entry)
            continue

        subjects[entry] = {
            "segments": segments,
            "label": label,
            "sds_score": sds,
            "split_prefix": split_prefix,
        }

    dep = sum(1 for s in subjects.values() if s["label"] == 1)
    norm = len(subjects) - dep
    total_segs = sum(len(s["segments"]) for s in subjects.values())
    logger.info(
        "EATD-Corpus: %d subjects (dep=%d, norm=%d), %d total segments",
        len(subjects), dep, norm, total_segs,
    )
    return subjects


def build_speaker_independent_splits(
    subjects: Dict[str, dict],
    val_size: float = 0.15,
    random_state: int = 42,
    return_subject_ids: bool = False,
) -> Dict[str, Tuple]:
    """
    Create speaker-independent train/val/test splits.

    - t_* subjects → train + val (stratified by label)
    - v_* subjects → test (held-out speakers)

    Args:
        subjects: Output of load_eatd_corpus().
        val_size: Fraction of t_* subjects used for validation.
        random_state: Random seed.
        return_subject_ids: If True, each split value is (X, y, subject_ids)
            where subject_ids is an array of integer indices identifying which
            subject each segment came from.  A companion ``subject_names`` array
            is added under the "subject_names" key of the returned dict.

    Returns:
        Dict with "train", "val", "test" keys, each (X, y) tuple (or
        (X, y, subject_ids) when return_subject_ids=True), plus
        "subject_names" when return_subject_ids=True.
    """
    rng = np.random.RandomState(random_state)

    # Separate by dataset prefix
    t_ids = [k for k, v in subjects.items() if v["split_prefix"] == "t"]
    v_ids = [k for k, v in subjects.items() if v["split_prefix"] == "v"]

    # Stratified val split from t_* subjects
    t_labels = np.array([subjects[k]["label"] for k in t_ids])
    t_ids = np.array(t_ids)

    # Shuffle maintaining stratification
    dep_ids = t_ids[t_labels == 1]
    norm_ids = t_ids[t_labels == 0]
    rng.shuffle(dep_ids)
    rng.shuffle(norm_ids)

    n_val_dep = max(1, int(len(dep_ids) * val_size))
    n_val_norm = max(1, int(len(norm_ids) * val_size))

    val_ids = list(dep_ids[:n_val_dep]) + list(norm_ids[:n_val_norm])
    train_ids = list(dep_ids[n_val_dep:]) + list(norm_ids[n_val_norm:])

    # Map subject name → integer index (across all subjects, stable order)
    all_ids = sorted(subjects.keys())
    subj_to_idx = {sid: i for i, sid in enumerate(all_ids)}

    def _collect(id_list):
        segs, labels, sidxs = [], [], []
        for sid in id_list:
            s = subjects[sid]
            segs.extend(s["segments"])
            labels.extend([s["label"]] * len(s["segments"]))
            sidxs.extend([subj_to_idx[sid]] * len(s["segments"]))
        X = np.array(segs, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        ids = np.array(sidxs, dtype=np.int32)
        return (X, y, ids) if return_subject_ids else (X, y)

    train_out = _collect(train_ids)
    val_out = _collect(val_ids)
    test_out = _collect(v_ids)

    y_train = train_out[1]
    y_val = val_out[1]
    y_test = test_out[1]

    logger.info(
        "Speaker-independent splits:  train=%d (%d subj)  val=%d (%d subj)  test=%d (%d subj)",
        len(y_train), len(train_ids), len(y_val), len(val_ids), len(y_test), len(v_ids),
    )
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        logger.info("  %s: dep=%d  norm=%d", name, int(y.sum()), int((y == 0).sum()))

    result: Dict[str, Any] = {
        "train": train_out,
        "val": val_out,
        "test": test_out,
    }
    if return_subject_ids:
        result["subject_names"] = np.array(all_ids)
    return result


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced EATD data."""
    n = len(y)
    n_pos = y.sum()
    n_neg = n - n_pos
    return {
        0: n / (2 * n_neg),
        1: n / (2 * n_pos),
    }
