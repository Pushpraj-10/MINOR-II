"""
EATD-Corpus loader for HLG-Net.

The EATD-Corpus contains audio interviews from subjects, each with
SDS (Zung Self-Rating Depression Scale) scores.
Subjects with SDS >= 53 are labelled as depressed.

Each subject has 3 VAD-processed audio files (*_out.wav):
positive, negative, and neutral interview responses.

This loader yields each of these audio files as a separate item,
paired with the subject's continuous SDS score.
"""

import os
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

SDS_DEPRESSION_THRESHOLD = 53


def _read_sds_score(subject_dir: str) -> float:
    """Read SDS score from new_label.txt (preferred) or label.txt."""
    new_label = os.path.join(subject_dir, "new_label.txt")
    if os.path.exists(new_label):
        with open(new_label) as f:
            return float(f.read().strip())
    with open(os.path.join(subject_dir, "label.txt")) as f:
        return float(f.read().strip())


def load_eatd_splits(
    corpus_dir: str,
    audio_files: Tuple[str, ...] = ("positive_out.wav", "negative_out.wav", "neutral_out.wav"),
) -> Dict[str, List[Tuple[str, float]]]:
    """Load EATD-Corpus keeping predefined train/val/test prefixes.

    - t_* subjects: split into train/val (random 85/15)
    - v_* subjects: test

    Args:
        corpus_dir: Path to the EATD-Corpus root.
        audio_files: Which WAV files to load per subject.

    Returns:
        Dict mapping split names to lists of (audio_path, sds_score).
    """
    import random
    random.seed(42)

    categories = {"train": [], "val": [], "test": []}
    
    # Collect subjects
    t_subjects = []
    v_subjects = []

    for entry in sorted(os.listdir(corpus_dir)):
        if not (entry.startswith("t_") or entry.startswith("v_")):
            continue
        subj_dir = os.path.join(corpus_dir, entry)
        if not os.path.isdir(subj_dir):
            continue

        sds = _read_sds_score(subj_dir)
        
        # Collect existing audio files for this subject
        subj_files = []
        for wav_name in audio_files:
            wav_path = os.path.join(subj_dir, wav_name)
            if os.path.exists(wav_path):
                subj_files.append((wav_path, sds))
                
        if not subj_files:
            continue
            
        if entry.startswith("v_"):
            v_subjects.append(subj_files)
        else:
            t_subjects.append(subj_files)

    # Shuffle t_subjects and split 85% train, 15% val
    random.shuffle(t_subjects)
    n_val = max(1, int(len(t_subjects) * 0.15))
    
    val_subjects = t_subjects[:n_val]
    train_subjects = t_subjects[n_val:]
    
    for subj_files in train_subjects:
        categories["train"].extend(subj_files)
    for subj_files in val_subjects:
        categories["val"].extend(subj_files)
    for subj_files in v_subjects:
        categories["test"].extend(subj_files)

    for split_name, entries in categories.items():
        dep_count = sum(1 for _, sds in entries if sds >= SDS_DEPRESSION_THRESHOLD)
        logger.info(
            "EATD-Corpus %s: %d files (dep=%d, normal=%d)",
            split_name, len(entries), dep_count, len(entries) - dep_count
        )

    return categories
