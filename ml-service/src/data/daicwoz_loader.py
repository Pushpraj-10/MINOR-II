"""
DAIC-WOZ (AVEC 2017) dataset loader for HLG-Net.

Loads participant audio files and PHQ-8 depression scores from the DAIC-WOZ
corpus.  The dataset has pre-defined train/dev/test splits via CSV files.

Each participant has an audio file at:
    {DAICWOZ_DIR}/{participant_id}_P/{participant_id}_AUDIO.wav

PHQ-8 scores range 0–24.  Binary depression label: PHQ8_Score >= 10.
"""

import os
import csv
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

PHQ8_DEPRESSION_THRESHOLD = 10  # standard clinical cut-off


def _parse_train_dev_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    """Parse train/dev CSV with columns: Participant_ID, PHQ8_Binary, PHQ8_Score, ...

    Returns:
        List of (participant_id, binary_label, phq8_score) tuples.
    """
    entries = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["Participant_ID"])
            binary = int(row["PHQ8_Binary"])
            score = int(row["PHQ8_Score"])
            entries.append((pid, binary, score))
    return entries


def _parse_test_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    """Parse test CSV with columns: Participant_ID, PHQ_Binary, PHQ_Score, Gender.

    Note: Test CSV uses 'PHQ_Binary'/'PHQ_Score' (without '8'), different from
    the train/dev CSVs which use 'PHQ8_Binary'/'PHQ8_Score'.

    Returns:
        List of (participant_id, binary_label, phq_score) tuples.
    """
    entries = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["Participant_ID"].strip())
            binary = int(row["PHQ_Binary"].strip())
            score = int(row["PHQ_Score"].strip())
            entries.append((pid, binary, score))
    return entries


def _get_audio_path(daicwoz_dir: str, participant_id: int) -> str:
    """Construct path to participant audio file."""
    return os.path.join(daicwoz_dir, f"{participant_id}_P", f"{participant_id}_AUDIO.wav")


def load_daicwoz_splits(
    daicwoz_dir: str,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Load DAIC-WOZ dataset with pre-defined train/dev/test splits.

    Args:
        daicwoz_dir: Path to the DAICWOZ root directory containing CSV files
                     and participant folders.

    Returns:
        Dict with keys "train", "val", "test", each mapping to a list of
        (audio_path, binary_label, phq8_score) tuples.
        Only includes entries where the audio file exists on disk.
    """
    train_csv = os.path.join(daicwoz_dir, "train_split_Depression_AVEC2017.csv")
    dev_csv = os.path.join(daicwoz_dir, "dev_split_Depression_AVEC2017.csv")
    test_csv = os.path.join(daicwoz_dir, "full_test_split.csv")

    splits = {}
    for split_name, csv_path, parser in [
        ("train", train_csv, _parse_train_dev_csv),
        ("val", dev_csv, _parse_train_dev_csv),
        ("test", test_csv, _parse_test_csv),
    ]:
        if not os.path.exists(csv_path):
            logger.warning("CSV not found: %s — skipping %s split", csv_path, split_name)
            splits[split_name] = []
            continue

        raw_entries = parser(csv_path)
        valid = []
        skipped = 0
        for pid, binary, score in raw_entries:
            audio_path = _get_audio_path(daicwoz_dir, pid)
            if os.path.exists(audio_path):
                valid.append((audio_path, binary, score))
            else:
                skipped += 1

        splits[split_name] = valid
        dep_count = sum(1 for _, b, _ in valid if b == 1)
        logger.info(
            "DAICWOZ %s: %d participants loaded (%d depressed, %d normal, %d skipped — no audio)",
            split_name, len(valid), dep_count, len(valid) - dep_count, skipped,
        )

    return splits
