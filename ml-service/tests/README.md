# Tests

Automated tests for the depression detection ML pipeline.

## Modules to Test

| Module | Key functions |
|---|---|
| `src.data.loader` | `AudioDataLoader` — audio loading, normalization, padding |
| `src.data.splitter` | `split_dataset` — stratified train/val/test splits |
| `src.features.tf_audio` | `extract_mel_spectrogram`, `extract_mfcc`, `extract_features_batch` |
| `src.models.architectures` | `get_model`, `list_architectures`, MODEL_REGISTRY |
| `src.export.tflite_converter` | `build_combined_cnn`, `convert_to_tflite` |
| `src.evaluation.evaluator` | `evaluate_model`, `evaluate_tflite_on_splits` |
| `src.utils.file_utils` | `ensure_dir`, `get_all_files` |


## Running Tests

```bash
pytest
pytest --cov=src --cov-report=html
pytest -k "test_mfcc" -v
```
