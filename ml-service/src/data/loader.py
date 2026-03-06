"""Audio data loading utilities for depression detection."""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from src.utils.file_utils import get_all_files

logger = logging.getLogger(__name__)


class AudioDataLoader:
    """Load and manage audio datasets for depression detection."""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        duration: float = 5.0,
        mono: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            sample_rate: Target sample rate for loading
            duration: Target duration in seconds
            mono: Convert to mono
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        self.target_length = int(sample_rate * duration)

    def load_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load a single audio file, normalize and pad/truncate to fixed length.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio array of shape (target_length,) or None if failed
        """
        try:
            y, sr = librosa.load(
                audio_path, sr=self.sample_rate, mono=self.mono, duration=self.duration
            )

            # Normalize amplitude
            y = librosa.util.normalize(y)

            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) > 0:
                y = y_trimmed

            # Pad or truncate to fixed length
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)))
            else:
                y = y[: self.target_length]

            return y

        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            return None

    def load_dataset(
        self,
        depression_dir: str = "depression1",
        normal_dir: str = "normal1",
        extensions: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load all audio files from depression and normal directories.

        Args:
            depression_dir: Name of depression subdirectory
            normal_dir: Name of normal subdirectory
            extensions: Audio file extensions to load

        Returns:
            Tuple of (audio_list, labels, file_paths)
        """
        if extensions is None:
            extensions = [".wav", ".mp3", ".flac"]

        audio_list = []
        labels = []
        file_paths = []

        # Load depression samples (label=1)
        dep_path = self.data_dir / depression_dir
        dep_files = get_all_files(str(dep_path), extensions)
        logger.info(f"Found {len(dep_files)} depression files in {dep_path}")

        for audio_file in dep_files:
            y = self.load_audio_file(str(audio_file))
            if y is not None:
                audio_list.append(y)
                labels.append(1)
                file_paths.append(str(audio_file))

        # Load normal samples (label=0)
        norm_path = self.data_dir / normal_dir
        norm_files = get_all_files(str(norm_path), extensions)
        logger.info(f"Found {len(norm_files)} normal files in {norm_path}")

        for audio_file in norm_files:
            y = self.load_audio_file(str(audio_file))
            if y is not None:
                audio_list.append(y)
                labels.append(0)
                file_paths.append(str(audio_file))

        logger.info(
            f"Loaded {len(audio_list)} total samples "
            f"(depression={labels.count(1)}, normal={labels.count(0)})"
        )

        return audio_list, labels, file_paths
