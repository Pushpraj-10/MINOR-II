"""
Full RAVDESS dataset loader for HLG-Net regression.

Loads from "The Ryerson Audio-Visual Dataset" containing Actor_* directories.
RAVDESS filename identifiers:
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    e.g., 03-01-04-01-01-01-01.wav

Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
For depression detection, we use:
    - 01 (neutral) -> mapped to Normal (Proxy Score = 0)
    - 04 (sad)     -> mapped to Depressed (Proxy Score = 20)
Other emotions are ignored.
"""

import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

PROXY_SCORE_DEPRESSED = 20
PROXY_SCORE_NORMAL = 0


def load_ravdess_dataset(
    dataset_dir: str,
) -> List[Tuple[str, float]]:
    """Load raw RAVDESS dataset.

    Args:
        dataset_dir: Path to the root directory containing Actor_XX folders.

    Returns:
        List of (audio_path, proxy_score) tuples.
    """
    entries = []

    if not os.path.exists(dataset_dir):
        logger.warning(f"RAVDESS directory not found: {dataset_dir}")
        return entries

    for actor_folder in os.listdir(dataset_dir):
        if not actor_folder.startswith("Actor_"):
            continue
            
        actor_path = os.path.join(dataset_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
            
        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
                
            # Parse emotion from filename (e.g., 03-01-04-...)
            parts = file.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue
                
            emotion = parts[2]
            
            if emotion == "01":  # neutral -> normal
                score = PROXY_SCORE_NORMAL
            elif emotion == "04":  # sad -> depressed
                score = PROXY_SCORE_DEPRESSED
            else:
                continue  # skip angry/happy/fear/etc.
                
            full_path = os.path.join(actor_path, file)
            entries.append((full_path, float(score)))

    dep_count = sum(1 for _, s in entries if s == PROXY_SCORE_DEPRESSED)
    logger.info(
        "RAVDESS total: %d files loaded (dep/sad=%d, normal/neutral=%d)",
        len(entries), dep_count, len(entries) - dep_count,
    )

    return entries
