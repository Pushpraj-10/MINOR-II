# `artifacts/models/` Directory

## Purpose
Store all trained models, model checkpoints, and exported models in various formats for deployment.

## What Belongs Here
- **Trained models**: Saved Keras/PyTorch models (.h5, .pt)
- **Model checkpoints**: Intermediate training snapshots
- **Exported models**: TensorFlow Lite (.tflite), ONNX (.onnx), Core ML (.mlmodel)
- **Model metadata**: Training configuration, architecture info
- **Model cards**: Documentation of model performance and limitations

## What Should NOT Be Here
- ❌ Source code (belongs in `src/models/`)
- ❌ Training scripts (belongs in `src/pipelines/`)
- ❌ Evaluation metrics (belongs in `../metrics/`)
- ❌ Training plots (belongs in `../plots/`)

## Typical Structure
```
artifacts/models/
├── 20260219_1400_baseline_cnn/
│   ├── best_model.h5              # Best model during training
│   ├── final_model.h5             # Model after all epochs
│   ├── model.tflite               # TFLite for mobile
│   ├── model_quantized.tflite     # Quantized TFLite
│   ├── config_snapshot.yaml        # Training configuration
│   ├── model_card.md              # Model documentation
│   ├── architecture.txt           # model.summary() output
│   └── training_history.json      # Loss/accuracy per epoch
│
├── 20260220_0930_mobilenet_v1/
│   └── ...
│
├── production/
│   ├── depression_model_v1.0.0.h5
│   ├── depression_model_v1.0.0.tflite
│   └── model_metadata.json
│
└── best_model.h5  # Symlink to current best model
```

## Model Naming Convention
```
{YYYYMMDD}_{HHMM}_{experiment_name}/
    best_model.h5
    
# Or for production releases
{model_name}_v{major}.{minor}.{patch}.{format}
depression_model_v1.0.0.h5
depression_model_v1.1.0.tflite
```

## Model Formats

### Keras (.h5)
Full Keras model with architecture and weights
```python
# Save
model.save('artifacts/models/baseline_cnn/model.h5')

# Load
from tensorflow import keras
model = keras.models.load_model('artifacts/models/baseline_cnn/model.h5')
```

### TensorFlow Lite (.tflite)
Optimized for mobile deployment
```python
import tensorflow as tf

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### ONNX (.onnx)
Cross-platform model format
```python
import tf2onnx

# Convert
model_proto, _ = tf2onnx.convert.from_keras(model)

# Save
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

## Model Metadata
Store important information with each model:
```json
{
  "model_id": "20260219_1400_baseline_cnn",
  "version": "1.0.0",
  "created_date": "2026-02-19T14:00:00",
  "framework": "TensorFlow 2.13.0",
  "architecture": "Lightweight CNN",
  "parameters": 247856,
  "model_size_mb": 3.2,
  "input_shape": [13, 215, 1],
  "output_shape": [1],
  
  "training": {
    "dataset": "data/processed/v1",
    "train_samples": 850,
    "val_samples": 182,
    "epochs_trained": 45,
    "early_stopped": true,
    "best_epoch": 38
  },
  
  "performance": {
    "val_accuracy": 0.847,
    "val_auc": 0.891,
    "test_accuracy": 0.839,
    "test_auc": 0.885
  },
  
  "deployment": {
    "tflite_size_mb": 0.8,
    "inference_time_ms": 120,
    "target_platform": "Android/iOS"
  },
  
  "git_info": {
    "commit_hash": "a3f5b2c",
    "branch": "experiment/baseline-cnn"
  }
}
```

## Model Card Template
```markdown
# Model Card: Depression Detection Baseline CNN

## Model Details
- **Developer**: Your Name
- **Model date**: February 19, 2026
- **Model version**: 1.0.0
- **Model type**: Convolutional Neural Network (CNN)
- **Training algorithm**: Adam optimizer with early stopping

## Intended Use
- **Primary use**: Voice-based depression screening (not diagnosis)
- **Deployment**: Mobile application (Android/iOS)
- **Users**: General public for self-assessment
- **Out of scope**: Medical diagnosis, treatment decisions

## Factors
- **Groups**: Adults (18+) with voice capability
- **Language**: English speakers
- **Audio quality**: Clear audio, minimal background noise

## Metrics
- **Accuracy**: 83.9% on test set
- **AUC-ROC**: 0.885
- **Precision**: 81.2%
- **Recall**: 87.4%

## Training Data
- **Dataset**: Kaggle voice + RAVDESS (1,214 samples)
- **Class balance**: 50% depressed, 50% normal
- **Audio length**: 5 seconds per sample
- **Sample rate**: 16kHz mono

## Limitations
- Works only with English speech
- Requires good audio quality
- Not validated for clinical diagnosis
- May have reduced accuracy with accents
- Trained on limited demographic diversity

## Trade-offs
- Optimized for mobile deployment (small size)
- Accuracy vs. model size trade-off
- Privacy (on-device) vs. accuracy (cloud-based)
```

## Best Practices
1. **Version everything**: Use semantic versioning for production models
2. **Document thoroughly**: Include model cards and metadata
3. **Tag best models**: Use symlinks or tags for current best model
4. **Organize by experiment**: Keep related artifacts together
5. **Include config snapshot**: Save exact training configuration
6. **Track git commit**: Link model to code version
7. **Validate before deployment**: Test converted models match original
8. **Compression**: Compress old models for archival

## Model Lifecycle

### Development
- Experiment models saved with timestamp
- Keep checkpoints during training
- Save best model based on validation metric

### Staging
- Validate model performance
- Test converted formats
- Benchmark inference speed

### Production
- Semantic versioning (v1.0.0)
- Comprehensive testing
- Model card required
- Code freeze on associated commit

### Retirement
- Archive old models
- Document why model was retired
- Keep for reproducibility

## Storage Management
```bash
# Compress old experiment models
tar -czf 20260219_experiments.tar.gz 20260219_*

# Move to archive
mv 20260219_experiments.tar.gz ../archives/

# Keep only recent and production models
```

## Model Comparison
Track multiple models for comparison:
```json
{
  "models": [
    {
      "name": "baseline_cnn_v1",
      "accuracy": 0.839,
      "auc": 0.885,
      "size_mb": 3.2,
      "inference_ms": 120
    },
    {
      "name": "mobilenet_v1",
      "accuracy": 0.852,
      "auc": 0.897,
      "size_mb": 2.1,
      "inference_ms": 95
    }
  ]
}
```

## Notes
- Exclude from git (add to `.gitignore`)
- Use cloud storage or model registry for team collaboration
- Consider MLflow or similar for model tracking
- Backup production models separately
