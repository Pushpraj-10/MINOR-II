# `data/interim/` Directory

## Purpose
Store intermediate data that has been cleaned and preprocessed but not yet transformed into final features for model training.

## What Belongs Here
- **Cleaned audio**: Normalized volume, trimmed silence
- **Standardized format**: All audio converted to 16kHz mono WAV
- **Filtered data**: Removed corrupted or invalid files
- **Merged datasets**: Combined data from multiple sources
- **Quality-controlled data**: Data that passed validation checks

## What Should NOT Be Here
- ❌ Original raw data (belongs in `raw/`)
- ❌ Extracted features/MFCCs (belongs in `processed/`)
- ❌ Train/val/test splits (belongs in `processed/`)
- ❌ Model-ready data (belongs in `processed/`)

## Key Principles
- **Reproducible**: Can be regenerated from `raw/` using scripts
- **Intermediate state**: Between raw and model-ready
- **Clean but not featured**: Cleaned data, not extracted features
- **Documented transformations**: Clear record of what was done

## Typical Structure
```
data/interim/
├── cleaned_audio/
│   ├── depressed/
│   │   ├── audio_001_cleaned.wav
│   │   └── ...
│   └── normal/
│       ├── audio_001_cleaned.wav
│       └── ...
│
├── normalized_16khz/
│   ├── depressed/
│   └── normal/
│
├── merged_dataset/
│   ├── all_audio/
│   └── metadata_combined.csv
│
└── processing_log.txt  # Record of preprocessing steps
```

## Common Preprocessing Steps
1. **Audio normalization**: Consistent volume levels
2. **Resampling**: Convert all to 16kHz sample rate
3. **Channel conversion**: Stereo → Mono
4. **Duration standardization**: Trim or pad to fixed length
5. **Silence removal**: Trim leading/trailing silence
6. **Format conversion**: All to WAV format
7. **Quality filtering**: Remove corrupted/low-quality files

## Interactions
- **Input**: Read from `data/raw/`
- **Output**: Write to `data/interim/`
- **Next step**: Process from here to `data/processed/`
- **Scripts**: Processed by scripts in `src/data/`

## Best Practices
1. **Keep processing scripts**: Save preprocessing code in `src/data/`
2. **Log transformations**: Document what was done to each file
3. **Validate output**: Check that preprocessing worked correctly
4. **Don't commit large files**: Use `.gitignore` for audio files
5. **Reproducibility**: Should be regenerable from raw data
6. **Incremental processing**: Process in batches if dataset is large

## Example Processing Pipeline
```python
# src/data/preprocess_audio.py
import librosa
import soundfile as sf
from pathlib import Path

def clean_audio(raw_dir: str, interim_dir: str):
    """
    Clean and normalize audio files from raw to interim.
    
    Steps:
        1. Load audio from raw/
        2. Resample to 16kHz
        3. Convert to mono
        4. Normalize volume
        5. Trim silence
        6. Save to interim/
    """
    for audio_file in Path(raw_dir).glob("**/*.wav"):
        # Load
        y, sr = librosa.load(audio_file, sr=None)
        
        # Resample to 16kHz
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to mono if stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Save to interim
        output_path = Path(interim_dir) / audio_file.relative_to(raw_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, y, sr)
```

## Verification
After preprocessing, verify data quality:
- Check sample rate is consistent (16kHz)
- Verify all files are mono
- Check for any remaining corrupted files
- Ensure volume levels are normalized
- Verify file counts match expectations

## Notes
- This directory can be deleted and regenerated from `raw/` if needed
- Keep processing time reasonable (optimize for large datasets)
- Consider parallel processing for large datasets
- Document any files that failed preprocessing
