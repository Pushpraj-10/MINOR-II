# `data/processed/` Directory

## Purpose
Store final, model-ready feature representations extracted from cleaned data, organized into train/validation/test splits.

## What Belongs Here
- **Extracted features**: MFCC arrays, spectrograms, mel-spectrograms
- **Train/val/test splits**: Properly split and stratified datasets
- **Labels**: Corresponding labels for all features
- **Normalization parameters**: Mean/std used for feature normalization
- **Split indices**: Records of which samples are in which split
- **Feature metadata**: Information about feature extraction parameters

## What Should NOT Be Here
- ❌ Raw audio files (belongs in `raw/` or `interim/`)
- ❌ Trained models (belongs in `artifacts/models/`)
- ❌ Evaluation metrics (belongs in `artifacts/metrics/`)
- ❌ Intermediate preprocessing outputs (belongs in `interim/`)

## Key Principles
- **Model-ready**: Data is in exact format needed for model input
- **Reproducible**: Can be regenerated from `interim/` data
- **Well-organized**: Clear separation of train/val/test
- **Properly split**: No data leakage between splits

## Typical Structure
```
data/processed/
├── mfcc_features_train.npy      # Training features (N, 13, 215)
├── labels_train.npy              # Training labels (N,)
├── mfcc_features_val.npy        # Validation features
├── labels_val.npy                # Validation labels
├── mfcc_features_test.npy       # Test features
├── labels_test.npy               # Test labels
│
├── normalizer.pkl                # Saved normalizer (mean/std)
├── split_indices.json            # Record of train/val/test split
├── feature_config.yaml           # Feature extraction parameters
│
└── metadata.json                 # Dataset statistics and info
```

## Feature File Formats

### NumPy Arrays (.npy)
Best for large numerical arrays (features, labels)
```python
import numpy as np

# Save
np.save('mfcc_features_train.npy', X_train)

# Load
X_train = np.load('mfcc_features_train.npy')
```

### HDF5 (.h5)
Efficient for very large datasets
```python
import h5py

# Save
with h5py.File('features.h5', 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)

# Load
with h5py.File('features.h5', 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]
```

## Expected Feature Shapes

For MFCC features with 5-second audio:
```python
# Input shape for one sample
(13, 215, 1)  # (n_mfcc, time_steps, channels)

# Training set shape
X_train.shape  # (num_samples, 13, 215, 1)
y_train.shape  # (num_samples,)
```

## Data Splits
- **Training set**: 70% of data (model learning)
- **Validation set**: 15% of data (hyperparameter tuning)
- **Test set**: 15% of data (final evaluation, never used in training)

## Best Practices
1. **Stratified splitting**: Maintain class balance in all splits
2. **Fixed random seed**: Ensure reproducible splits
3. **No data leakage**: Features from same source should be in same split
4. **Save split info**: Record which files went into which split
5. **Normalize after splitting**: Fit normalizer only on training data
6. **Document parameters**: Save feature extraction config
7. **Check shapes**: Validate all arrays have expected dimensions
8. **Memory efficiency**: Use appropriate data types (float32 vs float64)

## Feature Extraction Pipeline
```python
# src/features/build_features.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.features.audio_processing import extract_mfcc
from src.features.normalization import FeatureNormalizer

def build_dataset(interim_dir: str, output_dir: str):
    """
    Extract features and create train/val/test splits.
    
    Steps:
        1. Load cleaned audio from interim/
        2. Extract MFCC features
        3. Split into train/val/test (70/15/15)
        4. Fit normalizer on training data only
        5. Normalize all splits
        6. Save features and normalizer
    """
    # Extract features from all audio files
    features = []
    labels = []
    file_paths = []
    
    for audio_file in Path(interim_dir).glob("**/*.wav"):
        mfcc = extract_mfcc(str(audio_file))
        features.append(mfcc)
        
        # Label from directory name
        label = 1 if 'depressed' in str(audio_file) else 0
        labels.append(label)
        file_paths.append(str(audio_file))
    
    X = np.array(features)
    y = np.array(labels)
    
    # Split dataset (stratified)
    X_train, X_temp, y_train, y_temp, files_train, files_temp = train_test_split(
        X, y, file_paths, test_size=0.3, stratify=y, random_state=42
    )
    
    X_val, X_test, y_val, y_test, files_val, files_test = train_test_split(
        X_temp, y_temp, files_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Normalize (fit on training data only!)
    normalizer = FeatureNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    
    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Save everything
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'mfcc_features_train.npy', X_train)
    np.save(output_path / 'labels_train.npy', y_train)
    np.save(output_path / 'mfcc_features_val.npy', X_val)
    np.save(output_path / 'labels_val.npy', y_val)
    np.save(output_path / 'mfcc_features_test.npy', X_test)
    np.save(output_path / 'labels_test.npy', y_test)
    
    # Save normalizer
    normalizer.save(output_path / 'normalizer.pkl')
    
    # Save split information
    import json
    split_info = {
        'train_files': files_train,
        'val_files': files_val,
        'test_files': files_test,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    with open(output_path / 'split_indices.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Dataset created:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
```

## Data Validation
Before training, validate processed data:
```python
import numpy as np

# Load data
X_train = np.load('data/processed/mfcc_features_train.npy')
y_train = np.load('data/processed/labels_train.npy')

# Check shapes
assert X_train.shape[0] == y_train.shape[0], "Mismatched samples"
assert X_train.shape[1:] == (13, 215, 1), "Unexpected feature shape"

# Check labels
assert set(y_train) == {0, 1}, "Invalid labels"

# Check for NaN/Inf
assert not np.any(np.isnan(X_train)), "NaN values found"
assert not np.any(np.isinf(X_train)), "Inf values found"

print("✓ Data validation passed")
```

## Metadata Example
```json
{
  "creation_date": "2026-02-19",
  "feature_type": "mfcc",
  "sample_rate": 16000,
  "duration": 5.0,
  "n_mfcc": 13,
  "n_fft": 512,
  "hop_length": 256,
  "train_samples": 850,
  "val_samples": 182,
  "test_samples": 182,
  "class_distribution": {
    "train": {"normal": 425, "depressed": 425},
    "val": {"normal": 91, "depressed": 91},
    "test": {"normal": 91, "depressed": 91}
  }
}
```

## Notes
- Never use test set during development (only for final evaluation)
- Regenerate if feature extraction parameters change
- Monitor file sizes (large arrays can fill disk quickly)
- Consider compression for long-term storage
- Keep this directory in `.gitignore` (too large for git)
