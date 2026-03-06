# `data/raw/` Directory

## Purpose
Store original, immutable source data exactly as received from external sources, without any modifications or preprocessing.

## What Belongs Here
- **Original voice recordings**: Depression and normal voice samples
- **Raw audio files**: Unprocessed WAV, MP3, FLAC files
- **Original metadata**: CSV files with labels, speaker info, recording details
- **Dataset documentation**: README files, data dictionaries
- **License files**: Dataset licenses and terms of use

## What Should NOT Be Here
- ❌ Modified or processed audio files (use `interim/` or `processed/`)
- ❌ Extracted features (belongs in `processed/`)
- ❌ Filtered or cleaned data (belongs in `interim/`)
- ❌ Temporary or intermediate files

## Key Principles
- **Immutability**: NEVER modify files in this directory
- **Archive**: Treat as read-only source of truth
- **Backup**: Keep backups of this data (most critical tier)
- **Version control**: Use DVC or Git LFS to track data versions

## Current Structure
```
data/raw/
└── voice_data/
    ├── depression1/          # Voice samples from depressed individuals
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    │
    ├── normal1/              # Voice samples from normal/healthy individuals
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    │
    ├── metadata.csv          # (Optional) Labels, demographics, recording info
    └── README.md             # Dataset description and provenance
```

**Note**: Place all depression-labeled voice recordings in `depression1/` and normal/healthy recordings in `normal1/`. This binary classification structure simplifies data loading and labeling.

## Best Practices
1. **Never delete**: Disk space is cheaper than re-downloading/re-collecting data
2. **Document provenance**: Keep records of where data came from
3. **Organize by label**: Keep depression/normal samples in separate directories
4. **Include metadata**: Store original metadata files with the data
5. **Checksums**: Maintain MD5/SHA256 hashes to verify data integrity
6. **Read-only permissions**: Consider making files read-only on disk
7. **Version control**: Use DVC to track data without bloating git
8. **Naming convention**: Use consistent file naming (e.g., `depression_001.wav`, `normal_001.wav`)

## Populating the Dataset
If your folders are currently empty, here's how to add data:

### From Existing Datasets
```python
# Example: Copy and organize files from external datasets
import shutil
from pathlib import Path

def organize_voice_data(source_dir: str, label: str):
    """
    Copy voice files from source to appropriate raw directory.
    
    Args:
        source_dir: Path to source dataset
        label: 'depression1' or 'normal1'
    """
    source = Path(source_dir)
    target = Path(f'data/raw/voice_data/{label}')
    target.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for audio_file in source.glob('**/*.wav'):
        # Copy with sequential naming
        dest = target / f"{label}_{count:04d}.wav"
        shutil.copy2(audio_file, dest)
        count += 1
    
    print(f"Copied {count} files to {target}")

# Usage
organize_voice_data('/path/to/depression/dataset', 'depression1')
organize_voice_data('/path/to/normal/dataset', 'normal1')
```

### Creating Metadata CSV
After populating, create a metadata file:
```python
import pandas as pd
from pathlib import Path

def create_metadata():
    """Generate metadata CSV for all voice files."""
    records = []
    
    for label_dir in ['depression1', 'normal1']:
        label = 0 if label_dir == 'normal1' else 1
        dir_path = Path(f'data/raw/voice_data/{label_dir}')
        
        for audio_file in dir_path.glob('*.wav'):
            records.append({
                'file_path': str(audio_file.relative_to('data/raw')),
                'filename': audio_file.name,
                'label': label,
                'label_name': 'depressed' if label == 1 else 'normal',
                'category': label_dir
            })
    
    df = pd.DataFrame(records)
    df.to_csv('data/raw/voice_data/metadata.csv', index=False)
    print(f"Created metadata for {len(df)} files")
    return df

# Usage
metadata = create_metadata()
print(metadata.head())
```

## Data Integrity
```bash
# Generate checksums for raw data (Linux/Mac)
find voice_data -type f -name "*.wav" -exec md5sum {} \; > checksums.md5

# For Windows PowerShell
Get-ChildItem -Path voice_data -Recurse -Filter *.wav | 
    Get-FileHash -Algorithm MD5 | 
    Select-Object Hash, Path | 
    Export-Csv checksums.csv

# Verify data integrity later
md5sum -c checksums.md5  # Linux/Mac
```

## DVC Usage
```bash
# Track raw data with DVC
dvc add data/raw/voice_data

# Commit DVC files to git
git add data/raw/voice_data.dvc .gitignore
git commit -m "Track raw voice dataset"

# Push to remote storage (configure remote first)
dvc push
```

## Volume Requirements
- **Expected size**: Each voice sample is typically 5 seconds at 16kHz = ~160KB
- **Total storage**: For 1,000 samples ≈ 160MB
- **Backup strategy**: Keep offline backup (external drive or cloud storage)

## Dataset Validation
Before processing, validate your raw data:

```python
# scripts/validate_raw_data.py
from pathlib import Path
import librosa

def validate_raw_dataset():
    """Check raw dataset integrity and report statistics."""
    stats = {
        'depression1': {'count': 0, 'corrupted': 0, 'total_duration': 0},
        'normal1': {'count': 0, 'corrupted': 0, 'total_duration': 0}
    }
    
    for label in ['depression1', 'normal1']:
        dir_path = Path(f'data/raw/voice_data/{label}')
        
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            continue
        
        for audio_file in dir_path.glob('*.wav'):
            try:
                y, sr = librosa.load(audio_file, sr=None, duration=None)
                duration = len(y) / sr
                stats[label]['count'] += 1
                stats[label]['total_duration'] += duration
            except Exception as e:
                print(f"❌ Corrupted: {audio_file.name} - {e}")
                stats[label]['corrupted'] += 1
    
    # Print report
    print("\n" + "="*50)
    print("Raw Dataset Validation Report")
    print("="*50)
    
    for label, data in stats.items():
        print(f"\n{label}:")
        print(f"  ✓ Valid files: {data['count']}")
        print(f"  ❌ Corrupted: {data['corrupted']}")
        if data['count'] > 0:
            avg_duration = data['total_duration'] / data['count']
            print(f"  ⏱  Avg duration: {avg_duration:.2f}s")
            print(f"  📊 Total duration: {data['total_duration']/60:.2f} min")
    
    total_valid = sum(s['count'] for s in stats.values())
    print(f"\n{'='*50}")
    print(f"Total valid samples: {total_valid}")
    
    # Check balance
    if stats['depression1']['count'] > 0 and stats['normal1']['count'] > 0:
        ratio = stats['depression1']['count'] / stats['normal1']['count']
        if 0.8 <= ratio <= 1.2:
            print("✓ Dataset is balanced")
        else:
            print(f"⚠️  Dataset imbalanced (ratio: {ratio:.2f})")
    
    return stats

# Run validation
if __name__ == '__main__':
    validate_raw_dataset()
```

Run this validation after populating your dataset:
```bash
python scripts/validate_raw_data.py
```

## Important Notes
- All data processing should read from `voice_data/` and write to `interim/` or `processed/`
- Keep a separate backup of this directory outside version control
- Document the source of your voice data (e.g., Kaggle, DAIC-WOZ, RAVDESS, custom recordings)
- If you discover data quality issues (corrupted files, noise), document them but don't fix them here
- Maintain class balance: aim for similar number of samples in `depression1/` and `normal1/`
- **File formats**: Prefer WAV (uncompressed) for highest quality, convert MP3/other formats during preprocessing
- **Sample rate**: Original files can vary, standardization happens in the interim stage
