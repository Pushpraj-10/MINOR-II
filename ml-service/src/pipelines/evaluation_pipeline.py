"""
Evaluation pipeline for depression detection TFLite models.

Provides the shared audio-loading + splitting step used by both
scripts/evaluate.py and scripts/train_all.py, and wraps the
per-model TFLite evaluation logic.
"""

import os
import numpy as np

from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset
from src.evaluation.evaluator import evaluate_tflite_on_splits
from src.config import (
    SAMPLE_RATE, DURATION, AUDIO_LENGTH,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
    DATA_DIR, DEPRESSION_DIR, NORMAL_DIR,
)


def load_audio_splits():
    """
    Load all audio waveforms and return stratified train/val/test splits.

    Returns a dict with keys "train", "val", "test", each a tuple (X, y).
    """
    loader = AudioDataLoader(
        data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, duration=DURATION, mono=True
    )
    audio_list, labels, _ = loader.load_dataset(
        depression_dir=DEPRESSION_DIR, normal_dir=NORMAL_DIR
    )
    audio_arr = np.array(audio_list)
    y = np.array(labels)
    return split_dataset(
        audio_arr, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )


def run_evaluation_pipeline(tflite_path, splits, *, split_names=None):
    """
    Evaluate a TFLite model on pre-loaded audio splits.

    Args:
        tflite_path:  Path to the .tflite model file.
        splits:       Output of load_audio_splits() — dict of (X, y) tuples.
        split_names:  Subset of splits to evaluate, e.g. ["val", "test"].
                      If None, all splits are evaluated.

    Returns:
        Dict of per-split metric dicts as returned by evaluate_tflite_on_splits().
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
    return evaluate_tflite_on_splits(
        tflite_path, splits,
        audio_length=AUDIO_LENGTH,
        split_names=split_names,
    )