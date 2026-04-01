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

        Every audio file in the world has different length, volume, and sample rate.
        This function makes them all identical so the model always sees the same input shape.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio array of shape (target_length,) or None if failed
        """
        try:
            # librosa.load: reads the audio file and resamples it to our target rate (16kHz).
            # sr=self.sample_rate forces resampling so all clips have the same frequency resolution.
            y, sr = librosa.load(
                audio_path, sr=self.sample_rate, mono=self.mono, duration=self.duration
            )

            # Scale the waveform so the loudest peak = 1.0
            # This removes volume differences between speakers (a quiet voice vs a loud one
            # should look the same to the model \u2014 volume isn't a depression marker we want).
            y = librosa.util.normalize(y)

            # Remove leading/trailing silence (anything >20dB below peak is trimmed).
            # This avoids the model learning from silence at the start/end of recordings.
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) > 0:
                y = y_trimmed

            # All clips must be exactly 80,000 samples (5s \u00d7 16kHz).
            # Shorter clips get zero-padded at the end; longer clips get cut off.
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
