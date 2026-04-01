"""
dataset-depression loader for HLG-Net.

Loads the RAVDESS-based acted emotion dataset which has audio clips organised
into depression{1,2} and normal{1,2} folders.

Since this dataset has no continuous depression scores, we assign proxy scores:
  - depression folders → proxy_score = 20  (above typical threshold)
  - normal folders     → proxy_score = 0

These proxies enable training the HLG-Net regression model, though they are
not clinically meaningful severity measures.
"""

import os
import logging
from typing import List, Tuple

from src.utils.file_utils import get_all_files

logger = logging.getLogger(__name__)

# Proxy depression scores for binary-only datasets
PROXY_SCORE_DEPRESSED = 20
PROXY_SCORE_NORMAL = 0

# Folder → (label, proxy_score) mapping
FOLDER_MAP = {
    "depression1": (1, PROXY_SCORE_DEPRESSED),
    "depression2": (1, PROXY_SCORE_DEPRESSED),
    "normal1": (0, PROXY_SCORE_NORMAL),
    "normal2": (0, PROXY_SCORE_NORMAL),
}


def load_depression_dataset(
    dataset_dir: str,
    extensions: List[str] = None,
) -> List[Tuple[str, int, int]]:
    """Load the dataset-depression corpus.

    Args:
        dataset_dir: Path to the dataset-depression root directory containing
                     depression{1,2} and normal{1,2} subdirectories.
        extensions: Audio file extensions to include (default: [".wav"]).

    Returns:
        List of (audio_path, binary_label, proxy_score) tuples.
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac"]

    all_entries = []

    for folder_name, (label, proxy_score) in FOLDER_MAP.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            logger.info("Folder not found: %s — skipping", folder_path)
            continue

        files = get_all_files(folder_path, extensions)
        for f in files:
            all_entries.append((str(f), label, proxy_score))

        logger.info(
            "dataset-depression/%s: %d files (label=%d, proxy_score=%d)",
            folder_name, len(files), label, proxy_score,
        )

    dep_count = sum(1 for _, l, _ in all_entries if l == 1)
    logger.info(
        "dataset-depression total: %d files (dep=%d, normal=%d)",
        len(all_entries), dep_count, len(all_entries) - dep_count,
    )

    return all_entries
