# `src/data/` Directory

## Purpose
This module handles all data-related operations including loading, validation, cleaning, and preparing datasets for feature engineering.

## What Belongs Here
- **Data loaders**: Functions to read audio files, CSVs, metadata
- **Data validators**: Schema validation, quality checks
- **Data cleaning**: Handle missing values, corrupted files, outliers
- **Data splitting**: Train/validation/test split logic
- **Dataset classes**: PyTorch Dataset or TensorFlow data pipeline implementations
- **Data augmentation**: Audio augmentation techniques (if applicable)

## What Should NOT Be Here
- ❌ Feature extraction (belongs in `src/features/`)
- ❌ Model training code (belongs in `src/models/` or `src/pipelines/`)
- ❌ Raw data files (belongs in `data/raw/`)
- ❌ Visualization code (can be in notebooks or `src/evaluation/`)

## Architectural Responsibilities
- **Data abstraction**: Provide clean interface to access data
- **Data validation**: Ensure data quality before processing
- **Reproducibility**: Consistent data loading across experiments
- **Error handling**: Gracefully handle corrupted or missing files

## Typical Modules

### `loader.py`
Load audio files and metadata from disk
```python
def load_audio_dataset(data_dir: str, metadata_csv: str) -> Tuple[List, List]:
    """Load audio files and labels from directory."""
    pass

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load dataset metadata from CSV."""
    pass
```

### `validator.py`
Validate data quality and schema
```python
def validate_audio_file(audio_path: str) -> bool:
    """Check if audio file is valid and readable."""
    pass

def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Ensure DataFrame has required columns."""
    pass
```

### `preprocessor.py`
Clean and preprocess audio data
```python
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    pass

def remove_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence from audio."""
    pass

def pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate audio to target length."""
    pass
```

### `splitter.py`
Split data into train/val/test sets
```python
def split_dataset(
    file_paths: List[str],
    labels: List[int],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Tuple]:
    """Split dataset with stratification."""
    pass
```

### `augmentation.py` (Optional)
Data augmentation for training
```python
def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add random noise to audio."""
    pass

def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Randomly shift audio in time."""
    pass

def pitch_shift(audio: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
    """Shift audio pitch."""
    pass
```

## Interactions with Other Modules
- **`src/features/`**: Provides cleaned data for feature extraction
- **`src/models/`**: Supplies data for model training
- **`src/pipelines/`**: Called by training pipeline
- **`config/`**: Loads data paths and processing parameters
- **`data/`**: Reads from raw/, writes to interim/

## Best Practices
1. **Separation of concerns**: Keep data loading separate from feature engineering
2. **Lazy loading**: Don't load all data into memory at once for large datasets
3. **Caching**: Cache processed data to avoid redundant computation
4. **Error handling**: Validate files exist and are readable before processing
5. **Logging**: Log data statistics (num files, class distribution, etc.)
6. **Reproducibility**: Use fixed random seeds for splitting
7. **Generators**: Use Python generators for memory-efficient iteration
8. **Path handling**: Use `pathlib.Path` for cross-platform compatibility

## Example Implementation
```python
# src/data/loader.py
"""Data loading utilities for depression detection project."""

import os
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AudioDataLoader:
    """Load and manage audio datasets."""
    
    def __init__(self, data_dir: str, sample_rate: int = 16000):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing audio files
            sample_rate: Target sample rate for audio loading
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        
    def load_dataset(
        self,
        metadata_csv: str,
        duration: float = 5.0
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load entire dataset into memory.
        
        Args:
            metadata_csv: Path to CSV with columns ['file_path', 'label']
            duration: Audio duration in seconds
            
        Returns:
            Tuple of (audio_arrays, labels)
        """
        metadata = pd.read_csv(metadata_csv)
        logger.info(f"Loading {len(metadata)} audio files")
        
        audio_data = []
        labels = []
        
        for idx, row in metadata.iterrows():
            try:
                audio_path = self.data_dir / row['file_path']
                audio, _ = librosa.load(
                    audio_path,
                    sr=self.sample_rate,
                    duration=duration
                )
                audio_data.append(audio)
                labels.append(row['label'])
                
            except Exception as e:
                logger.warning(f"Failed to load {row['file_path']}: {e}")
                continue
                
        logger.info(f"Successfully loaded {len(audio_data)} files")
        return audio_data, labels
    
    def load_generator(
        self,
        metadata_csv: str,
        batch_size: int = 32
    ):
        """
        Generator for memory-efficient loading.
        
        Args:
            metadata_csv: Path to metadata CSV
            batch_size: Number of samples per batch
            
        Yields:
            Batches of (audio_arrays, labels)
        """
        metadata = pd.read_csv(metadata_csv)
        
        for i in range(0, len(metadata), batch_size):
            batch = metadata.iloc[i:i+batch_size]
            # Load batch
            yield batch_audio, batch_labels
```

## Data Validation Example
```python
# src/data/validator.py
"""Data validation utilities."""

import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that audio file exists and is readable.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(audio_path)
    
    # Check existence
    if not path.exists():
        logger.error(f"File not found: {audio_path}")
        return False
    
    # Check extension
    if path.suffix.lower() not in ['.wav', '.mp3', '.flac']:
        logger.error(f"Unsupported format: {path.suffix}")
        return False
    
    # Try loading
    try:
        librosa.load(audio_path, sr=None, duration=1.0)
        return True
    except Exception as e:
        logger.error(f"Cannot load {audio_path}: {e}")
        return False


def validate_dataset(data_dir: str, metadata_csv: str) -> Dict[str, int]:
    """
    Validate entire dataset and return statistics.
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'missing_files': 0
    }
    
    # Implementation
    return stats
```

## Testing Data Module
```python
# tests/test_data_loader.py
import pytest
from src.data.loader import AudioDataLoader

def test_audio_loader_initialization():
    loader = AudioDataLoader('data/raw/', sample_rate=16000)
    assert loader.sample_rate == 16000

def test_load_single_audio(sample_audio_path):
    loader = AudioDataLoader('data/raw/')
    audio, label = loader.load_single(sample_audio_path)
    assert audio.shape[0] > 0
```
