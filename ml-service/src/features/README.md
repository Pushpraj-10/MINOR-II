# `src/features/` Directory

## Purpose
This module transforms raw preprocessed data into feature representations suitable for machine learning models (e.g., MFCC, spectrograms, mel-spectrograms).

## What Belongs Here
- **Feature extraction**: MFCC, spectrogram, mel-spectrogram computation
- **Feature engineering**: Creating derived features, feature combinations
- **Feature normalization**: Standardization, min-max scaling
- **Feature selection**: Selecting most informative features
- **Feature transformations**: Dimensionality reduction (PCA, etc.)
- **Feature utilities**: Helper functions for feature processing

## What Should NOT Be Here
- ❌ Data loading/cleaning (belongs in `src/data/`)
- ❌ Model architecture (belongs in `src/models/`)
- ❌ Raw audio preprocessing (belongs in `src/data/`)
- ❌ Training logic (belongs in `src/pipelines/`)

## Architectural Responsibilities
- **Feature abstraction**: Transform data into ML-ready representations
- **Consistency**: Ensure features are computed identically in train/test
- **Reusability**: Share feature extraction across training and inference
- **Efficiency**: Optimize feature computation for speed and memory

## Typical Modules

### `audio_processing.py`
Core audio feature extraction
```python
def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """Extract MFCC features from audio signal."""
    pass

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128
) -> np.ndarray:
    """Extract mel-spectrogram features."""
    pass

def extract_spectral_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract spectral centroid, rolloff, contrast, etc."""
    pass
```

### `feature_engineering.py`
Create derived features
```python
def compute_feature_statistics(features: np.ndarray) -> np.ndarray:
    """Compute mean, std, min, max across time dimension."""
    pass

def create_delta_features(mfcc: np.ndarray) -> np.ndarray:
    """Compute delta and delta-delta features."""
    pass
```

### `normalization.py`
Feature scaling and normalization
```python
class FeatureNormalizer:
    """Normalize features using training statistics."""
    
    def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        """Compute normalization parameters from training data."""
        pass
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization using stored parameters."""
        pass
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass
```

### `feature_builder.py`
High-level feature pipeline orchestration
```python
class AudioFeatureBuilder:
    """End-to-end feature extraction pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def build_features(self, audio_path: str) -> np.ndarray:
        """Extract all features from audio file."""
        pass
    
    def build_dataset_features(
        self,
        audio_list: List[str],
        output_path: str
    ) -> None:
        """Extract features for entire dataset and save."""
        pass
```

## Interactions with Other Modules
- **`src/data/`**: Receives cleaned audio data
- **`src/models/`**: Provides features for model training/inference
- **`src/pipelines/`**: Called during training and prediction pipelines
- **`src/utils/`**: May use utility functions for file I/O
- **`config/`**: Loads feature extraction parameters
- **`data/processed/`**: Saves extracted features

## Best Practices
1. **Deterministic**: Same input should always produce same features
2. **Train/test consistency**: Use same parameters for training and inference
3. **Save normalization params**: Persist mean/std from training for test time
4. **Vectorization**: Use NumPy vectorized operations for speed
5. **Memory efficiency**: Process in batches for large datasets
6. **Documentation**: Document feature dimensions and expected input/output shapes
7. **Configuration**: Parameterize feature extraction settings via config
8. **Validation**: Check for NaN/Inf values in extracted features

## Example Implementation
```python
# src/features/audio_processing.py
"""Audio feature extraction for depression detection."""

import librosa
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 256,
    'n_mels': 128
}


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio time series array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        normalize: Whether to normalize features
        
    Returns:
        MFCC features with shape (n_mfcc, time_steps)
        
    Raises:
        ValueError: If audio is empty or invalid
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty")
    
    try:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Normalize if requested
        if normalize:
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        logger.debug(f"Extracted MFCC features with shape: {mfccs.shape}")
        return mfccs
        
    except Exception as e:
        logger.error(f"MFCC extraction failed: {e}")
        raise


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract mel-spectrogram features.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        
    Returns:
        Mel-spectrogram with shape (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_all_features(
    audio: np.ndarray,
    sr: int = 16000,
    config: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all audio features.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        config: Feature extraction configuration
        
    Returns:
        Dictionary of feature arrays
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    features = {}
    
    # MFCCs
    features['mfcc'] = extract_mfcc(
        audio, sr,
        n_mfcc=config['n_mfcc'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length']
    )
    
    # Mel-spectrogram
    features['mel_spectrogram'] = extract_mel_spectrogram(
        audio, sr,
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']
    )
    
    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=audio, sr=sr
    )
    
    # Zero-crossing rate
    features['zcr'] = librosa.feature.zero_crossing_rate(audio)
    
    return features
```

## Feature Normalization Example
```python
# src/features/normalization.py
"""Feature normalization utilities."""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalize features using training set statistics.
    
    This ensures consistent normalization between training and inference.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        """
        Compute normalization parameters from training data.
        
        Args:
            features: Training features with shape (n_samples, ...)
            
        Returns:
            Self for method chaining
        """
        self.mean_ = np.mean(features, axis=0)
        self.std_ = np.std(features, axis=0)
        self.is_fitted = True
        
        logger.info("Normalizer fitted to training data")
        logger.debug(f"Mean shape: {self.mean_.shape}, Std shape: {self.std_.shape}")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using stored parameters.
        
        Args:
            features: Features to normalize
            
        Returns:
            Normalized features
            
        Raises:
            RuntimeError: If normalizer not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        return (features - self.mean_) / (self.std_ + 1e-8)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)
    
    def save(self, path: str) -> None:
        """Save normalizer parameters to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean_, 'std': self.std_}, f)
        logger.info(f"Normalizer saved to {path}")
    
    def load(self, path: str) -> 'FeatureNormalizer':
        """Load normalizer parameters from disk."""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.mean_ = params['mean']
        self.std_ = params['std']
        self.is_fitted = True
        logger.info(f"Normalizer loaded from {path}")
        return self
```

## Testing Feature Extraction
```python
# tests/test_feature_extraction.py
import pytest
import numpy as np
from src.features.audio_processing import extract_mfcc

def test_mfcc_shape():
    """Test MFCC output has correct shape."""
    audio = np.random.randn(16000 * 5)  # 5 seconds
    mfccs = extract_mfcc(audio, sr=16000, n_mfcc=13)
    
    assert mfccs.shape[0] == 13, "Should have 13 MFCC coefficients"
    assert mfccs.shape[1] > 0, "Should have time steps"

def test_mfcc_normalization():
    """Test normalized MFCCs have mean≈0 and std≈1."""
    audio = np.random.randn(16000 * 5)
    mfccs = extract_mfcc(audio, normalize=True)
    
    assert abs(np.mean(mfccs)) < 0.1
    assert abs(np.std(mfccs) - 1.0) < 0.1
```
