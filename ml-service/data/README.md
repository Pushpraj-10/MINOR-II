# `data/` Directory

## Purpose
This directory manages all data assets across different stages of the ML pipeline, from raw unprocessed audio files to fully prepared training datasets.

## What Belongs Here
- **Audio files**: WAV, MP3, FLAC files for depression detection
- **Dataset metadata**: CSV files with labels, splits, file paths
- **Data version tracking**: DVC files or manifest files
- **Data documentation**: Dataset descriptions, data dictionaries

## What Should NOT Be Here
- ❌ Large files directly in git (use Git LFS or DVC)
- ❌ Temporary files from processing (use `/tmp` or clean up after)
- ❌ Model artifacts (belongs in `artifacts/`)
- ❌ Processed features for model training (keep in `processed/` subdirectory)

## Architectural Responsibilities
- **Data lifecycle management**: Track data from acquisition to model-ready format
- **Reproducibility**: Enable exact recreation of training datasets
- **Data versioning**: Maintain different versions of datasets
- **Storage optimization**: Efficient organization of potentially large audio files

## Subdirectories

### `raw/`
- Original, immutable source data
- Never modify files here
- Keep backups of this data

### `interim/`
- Intermediate processing stages
- Cleaned audio, normalized files
- Can be regenerated from `raw/`

### `processed/`
- Final feature-engineered data ready for model training
- MFCC features, spectrograms
- Train/validation/test splits

## Interactions with Other Directories
- **`src/data/`**: Contains scripts that process data in this directory
- **`src/features/`**: Reads from `interim/`, writes to `processed/`
- **`notebooks/`**: Exploratory analysis of data stored here
- **`config/`**: Data paths and processing parameters defined there

## Best Practices
1. **Immutable raw data**: Never modify files in `raw/`, always copy to process
2. **Use DVC or Git LFS**: Track large files without bloating git repository
3. **Document lineage**: Maintain clear records of data sources and transformations
4. **Consistent naming**: `{dataset}_{split}_{version}.{ext}` (e.g., `voice_train_v2.csv`)
5. **Separate by purpose**: Keep train/val/test splits clearly organized
6. **Add `.gitignore`**: Exclude large binary files from direct git tracking
7. **Checksum validation**: Verify data integrity with MD5/SHA256 hashes
8. **Privacy compliance**: Ensure no PII or sensitive data leaks

## Example Structure
```
data/
├── raw/
│   ├── kaggle_voice_dataset/
│   └── ravdess_dataset/
├── interim/
│   ├── cleaned_audio/
│   └── normalized_16khz/
└── processed/
    ├── mfcc_features_train.npy
    ├── mfcc_features_val.npy
    ├── mfcc_features_test.npy
    ├── labels_train.npy
    ├── labels_val.npy
    └── labels_test.npy
```

## Data Versioning
Consider using DVC (Data Version Control):
```bash
# Initialize DVC
dvc init

# Track large data files
dvc add data/raw/kaggle_voice_dataset
dvc add data/processed/mfcc_features_train.npy

# Commit DVC files to git
git add data/raw/kaggle_voice_dataset.dvc
git commit -m "Track raw voice dataset"
```
