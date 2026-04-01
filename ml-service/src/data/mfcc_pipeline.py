"""
40-D MFCC extraction pipeline and PyTorch Dataset for HLG-Net.

Provides:
  - extract_mfcc_40d(): Extract 40-D MFCC features from a single audio file
  - HLGNetDataset: PyTorch Dataset that lazily extracts MFCC from audio paths
  - create_dataloaders(): Factory to build train/val/test DataLoaders
"""

import os
import logging
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from src.config import (
    SAMPLE_RATE,
    HLGNET_N_MFCC,
    HLGNET_FRAME_SIZE,
    HLGNET_HOP_LENGTH,
    HLGNET_N_FFT,
    HLGNET_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def extract_mfcc_40d(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = HLGNET_N_MFCC,
    n_fft: int = HLGNET_N_FFT,
    hop_length: int = HLGNET_HOP_LENGTH,
    frame_size: int = HLGNET_FRAME_SIZE,
) -> np.ndarray:
    """Extract 40-D MFCC features from an audio file.

    Follows the HLG-Net specification:
      - 16kHz sample rate
      - 25ms window (n_fft=400)
      - 10ms hop (hop_length=160) → 100 Hz frame rate
      - 40 MFCC coefficients
      - Pad/truncate to exactly frame_size frames

    Args:
        audio_path: Path to audio file.
        sr: Target sample rate.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        frame_size: Fixed output length in frames.

    Returns:
        MFCC array of shape (frame_size, n_mfcc) = (4687, 40).
    """
    y, _ = librosa.load(audio_path, sr=sr)
    y = librosa.util.normalize(y)  # Normalize waveform

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # mfcc shape: (n_mfcc, time_steps)

    # Pad or truncate to fixed frame_size
    if mfcc.shape[1] < frame_size:
        pad_width = frame_size - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :frame_size]

    mfcc_T = mfcc.T.astype(np.float32)  # → (frame_size, n_mfcc)
    
    # Per-instance standardization (zero mean, unit variance)
    mean = np.mean(mfcc_T, axis=0, keepdims=True)
    std = np.std(mfcc_T, axis=0, keepdims=True)
    mfcc_T = (mfcc_T - mean) / (std + 1e-6)
    
    return mfcc_T


class HLGNetDataset(Dataset):
    """PyTorch Dataset for HLG-Net that extracts 40-D MFCC on the fly.

    Each item returns:
        mfcc: FloatTensor of shape (frame_size, n_mfcc)
        score: FloatTensor scalar — depression severity score

    Supports optional caching to avoid re-extracting MFCC each epoch.
    """

    def __init__(
        self,
        entries: List[Tuple[str, float]],
        cache_dir: Optional[str] = None,
        sr: int = SAMPLE_RATE,
        n_mfcc: int = HLGNET_N_MFCC,
        frame_size: int = HLGNET_FRAME_SIZE,
    ):
        """
        Args:
            entries: List of (audio_path, score) tuples.
            cache_dir: If set, cache extracted MFCC as .npy files here.
            sr: Target sample rate.
            n_mfcc: Number of MFCC coefficients.
            frame_size: Fixed output length in frames.
        """
        self.entries = entries
        self.cache_dir = cache_dir
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_size = frame_size

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path, score = self.entries[idx]

        # Try loading from cache
        mfcc = None
        if self.cache_dir:
            cache_key = os.path.basename(audio_path).replace(".", "_") + f"_{idx}.npy"
            cache_path = os.path.join(self.cache_dir, cache_key)
            if os.path.exists(cache_path):
                mfcc = np.load(cache_path)

        # Extract if not cached
        if mfcc is None:
            try:
                mfcc = extract_mfcc_40d(
                    audio_path,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    frame_size=self.frame_size,
                )
            except Exception as e:
                logger.warning("Failed to extract MFCC from %s: %s", audio_path, e)
                # Return zero-padded features on failure
                mfcc = np.zeros((self.frame_size, self.n_mfcc), dtype=np.float32)

            # Save to cache
            if self.cache_dir:
                np.save(cache_path, mfcc)

        return (
            torch.FloatTensor(mfcc),
            torch.FloatTensor([score]),
        )


def create_dataloaders(
    splits: Dict[str, List[Tuple[str, float]]],
    batch_size: int = HLGNET_BATCH_SIZE,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test splits.

    Args:
        splits: Dict mapping split name → list of (audio_path, score) tuples.
        batch_size: Batch size.
        cache_dir: Base directory for MFCC caching.
        num_workers: Number of data loading workers.

    Returns:
        Dict mapping split name → DataLoader.
    """
    loaders = {}
    for split_name, entries in splits.items():
        if not entries:
            logger.warning("Empty split: %s — skipping", split_name)
            continue

        split_cache = os.path.join(cache_dir, split_name) if cache_dir else None
        dataset = HLGNetDataset(entries, cache_dir=split_cache)

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        logger.info(
            "DataLoader %s: %d samples, %d batches",
            split_name, len(dataset), len(loaders[split_name]),
        )

    return loaders
