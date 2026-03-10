# Depression Detection ML Service

Voice-based depression detection using deep learning. Trains multiple model architectures on audio data and exports combined TFLite models (raw audio → prediction) for mobile deployment via Flutter.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a single model
python scripts/train_model.py --arch mel_cnn

# Train all architectures
python scripts/train_all.py

# List available architectures
python scripts/train_model.py --list

# Evaluate TFLite models
python scripts/evaluate.py
```

## Project Structure

```
ml-service/
├── data/
│   └── raw/voice_data/            # Audio files (depression1/, normal1/)
├── artifacts/
│   ├── models/                    # Trained .keras and .tflite models
│   ├── metrics/                   # Evaluation metrics JSON
│   ├── plots/                     # Confusion matrices, ROC curves
│   └── logs/                      # Training logs
├── src/                           # Core library
│   ├── config.py                  # Central constants (single source of truth)
│   ├── data/
│   │   ├── loader.py              # AudioDataLoader (WAV/MP3/FLAC → numpy)
│   │   └── splitter.py            # Stratified train/val/test splits
│   ├── features/
│   │   └── tf_audio.py            # tf.signal mel/MFCC extraction (TFLite-compatible)
│   ├── models/
│   │   └── architectures.py       # All 6 model architectures + MODEL_REGISTRY
│   ├── export/
│   │   └── tflite_converter.py    # Combined model builders + TFLite conversion
│   ├── evaluation/
│   │   └── evaluator.py           # Metrics, plots, confusion matrices
│   └── utils/
│       └── file_utils.py          # Directory and file utilities
├── scripts/                       # CLI entry points
│   ├── train_model.py             # Train any architecture (--arch flag)
│   ├── train_all.py               # Train all + comparison table
│   ├── evaluate.py                # Evaluate TFLite on internal splits
│   └── evaluate_ravdess.py        # Cross-dataset RAVDESS evaluation
├── main.py                        # Unified CLI dispatcher
└── requirements.txt
```

## Model Architectures

| Architecture | Input | Description |
|---|---|---|
| `mel_cnn` | Mel (128, 313, 1) | 4-block CNN on mel spectrograms |
| `bilstm` | Mel (313, 128) | Bidirectional LSTM on mel sequences |
| `cnn_lstm` | Mel (128, 313, 1) | CNN feature extraction → LSTM temporal modeling |
| `cnn_attention` | Mel (128, 313, 1) | CNN + Multi-Head Self-Attention |
| `separable_cnn` | Mel (128, 313, 1) | MobileNet-style depthwise separable CNN |
| `multi_feature` | Mel + MFCC | Dual-branch fusion (mel spectrogram + MFCC) |

## Audio Pipeline

- **Sample Rate**: 16 kHz
- **Duration**: 5.0 seconds (80,000 samples)
- **Features**: Mel spectrogram (128 bands, 313 time steps) via `tf.signal`
- **Classification**: Binary (Depression vs Normal), sigmoid output
- **TFLite Export**: Combined models embed preprocessing (raw audio → prediction, no external DSP needed)

## Training a Custom Model

```python
from src.models.architectures import get_model, MODEL_REGISTRY
from src.features.tf_audio import compute_mel_weights, extract_features_batch
from src.export.tflite_converter import build_combined_cnn, convert_to_tflite

# Build model
model = get_model("mel_cnn", input_shape=(128, 313, 1), dropout_rate=0.3)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ... train ...

# Export to TFLite
combined = build_combined_cnn(model)
convert_to_tflite(combined, "artifacts/models/my_model.tflite")
```
