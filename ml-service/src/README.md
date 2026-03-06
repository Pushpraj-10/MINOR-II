# `src/` Directory

## Purpose
This directory contains all production-ready source code for the ML project, organized into modular, reusable, and testable components.

## What Belongs Here
- **Production Python modules**: Clean, refactored code from notebooks
- **Reusable functions and classes**: Battle-tested implementations
- **Pipeline orchestration**: End-to-end workflow automation
- **Utility libraries**: Helper functions used across the project
- **Package initialization**: `__init__.py` files for proper imports

## What Should NOT Be Here
- ❌ Jupyter notebooks (belongs in `notebooks/`)
- ❌ Experimental/prototype code (test in notebooks first)
- ❌ Configuration files (belongs in `config/`)
- ❌ Model artifacts (belongs in `artifacts/`)
- ❌ Test files (belongs in `tests/`)

## Architectural Responsibilities
- **Code organization**: Maintain clean separation of concerns
- **Reusability**: DRY (Don't Repeat Yourself) principle
- **Modularity**: Each module has a single, well-defined purpose
- **Production-ready**: Code is tested, documented, and optimized

## Module Structure

### `data/`
Data loading, validation, and preprocessing scripts

### `features/`
Feature engineering and extraction (MFCC, spectrograms, etc.)

### `models/`
Model architecture definitions, custom layers, model builders

### `evaluation/`
Model evaluation, metrics computation, performance analysis

### `pipelines/`
End-to-end workflows for training, evaluation, and deployment

### `utils/`
Shared utilities (logging, config loading, file I/O, etc.)

## Interactions with Other Directories
- **`config/`**: Load configurations for parameterized behavior
- **`data/`**: Process data stored in data directory
- **`artifacts/`**: Save model outputs and metrics
- **`notebooks/`**: Import functions for experimentation
- **`tests/`**: Validated by test suite
- **`main.py`**: Orchestrated by main entry point

## Best Practices
1. **Type hints**: Use Python type annotations for clarity
   ```python
   def extract_mfcc(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
   ```

2. **Docstrings**: Document all public functions/classes
   ```python
   def train_model(config: dict) -> keras.Model:
       """
       Train a depression detection model.
       
       Args:
           config: Training configuration dictionary
           
       Returns:
           Trained Keras model
       """
   ```

3. **Error handling**: Use proper exception handling
   ```python
   try:
       audio, sr = librosa.load(path)
   except Exception as e:
       logger.error(f"Failed to load audio: {e}")
       raise
   ```

4. **Logging**: Use structured logging instead of print
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info(f"Processing {len(files)} files")
   ```

5. **Imports**: Use absolute imports from project root
   ```python
   from src.features.audio_processing import extract_mfcc
   from src.utils.config_loader import load_config
   ```

6. **Constants**: Define magic numbers as named constants
   ```python
   SAMPLE_RATE = 16000
   DURATION = 5.0
   N_MFCC = 13
   ```

7. **Single Responsibility**: Each function does one thing well

8. **Avoid hardcoding**: Use config files or environment variables

## Code Quality Standards
- **PEP 8 compliance**: Follow Python style guide
- **Maximum line length**: 88 characters (Black formatter)
- **Function complexity**: Keep cyclomatic complexity low
- **Test coverage**: >80% for critical modules
- **No circular imports**: Maintain clean dependency graph

## Example Module Structure
```python
# src/features/audio_processing.py
"""
Audio preprocessing and feature extraction module.

This module provides functions for loading, preprocessing, and extracting
features from audio files for depression detection.
"""

import librosa
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
DURATION = 5.0
N_MFCC = 13


def load_audio(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    duration: Optional[float] = DURATION
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        duration: Audio duration in seconds (None for full length)
        
    Returns:
        Tuple of (audio_array, sample_rate)
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        librosa.LibrosaError: If audio loading fails
    """
    try:
        audio, sample_rate = librosa.load(audio_path, sr=sr, duration=duration)
        logger.debug(f"Loaded audio: {audio_path} ({len(audio)} samples)")
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        raise


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC
) -> np.ndarray:
    """Extract MFCC features from audio."""
    # Implementation
    pass
```

## Dependency Management
- Keep dependencies minimal and well-justified
- Pin versions for reproducibility
- Separate dev dependencies from production

## Refactoring Checklist
When moving code from notebooks to `src/`:
- [ ] Remove exploratory code and debugging statements
- [ ] Extract magic numbers to constants/config
- [ ] Add type hints and docstrings
- [ ] Add error handling
- [ ] Replace print() with logging
- [ ] Write unit tests
- [ ] Update imports to use src modules
- [ ] Remove notebook-specific code (%matplotlib, etc.)
