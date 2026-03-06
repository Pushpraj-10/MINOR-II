# `src/pipelines/` Directory

## Purpose
This module orchestrates end-to-end ML workflows, connecting data loading, feature extraction, model training, evaluation, and deployment steps into cohesive pipelines.

## What Belongs Here
- **Training pipelines**: Complete training workflows from data to model
- **Inference pipelines**: Prediction workflows for new data
- **Evaluation pipelines**: Systematic model testing procedures
- **Preprocessing pipelines**: Data preparation workflows
- **Deployment pipelines**: Model export and conversion workflows
- **Pipeline utilities**: Helper functions for pipeline orchestration

## What Should NOT Be Here
- ❌ Individual component implementations (delegate to other modules)
- ❌ Model architectures (belongs in `src/models/`)
- ❌ Feature extraction logic (belongs in `src/features/`)
- ❌ Evaluation metrics (belongs in `src/evaluation/`)

## Architectural Responsibilities
- **Orchestration**: Coordinate multiple pipeline stages
- **Workflow management**: Define clear execution order and dependencies
- **Error handling**: Graceful failure handling and recovery
- **Logging**: Comprehensive pipeline execution logging
- **Reproducibility**: Ensure consistent pipeline execution

## Typical Modules

### `training_pipeline.py`
End-to-end model training workflow
```python
def train_model(config: dict) -> Tuple[keras.Model, Dict]:
    """
    Complete training pipeline.
    
    Steps:
        1. Load and validate data
        2. Extract features
        3. Build model
        4. Train model
        5. Evaluate model
        6. Save artifacts
    """
    pass

class TrainingPipeline:
    """Configurable training pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def run(self) -> keras.Model:
        """Execute full training pipeline."""
        pass
```

### `inference_pipeline.py`
Prediction workflow for new data
```python
class InferencePipeline:
    """Production inference pipeline."""
    
    def __init__(self, model_path: str, config: dict):
        self.model = load_model(model_path)
        self.config = config
    
    def predict(self, audio_path: str) -> Dict:
        """Predict depression probability from audio file."""
        pass
    
    def batch_predict(
        self,
        audio_paths: List[str]
    ) -> List[Dict]:
        """Predict for multiple audio files."""
        pass
```

### `preprocessing_pipeline.py`
Data preparation workflow
```python
def preprocess_dataset(
    raw_data_dir: str,
    output_dir: str,
    config: dict
) -> None:
    """
    Preprocess entire dataset.
    
    Steps:
        1. Load raw audio files
        2. Clean and normalize
        3. Extract features
        4. Save processed data
    """
    pass
```

### `deployment_pipeline.py`
Model deployment preparation
```python
def prepare_for_deployment(
    model: keras.Model,
    output_dir: str,
    target: str = 'tflite'
) -> None:
    """
    Prepare model for deployment.
    
    Steps:
        1. Optimize model (quantization, pruning)
        2. Convert to target format
        3. Validate converted model
        4. Package with metadata
    """
    pass
```

## Interactions with Other Modules
- **`src/data/`**: Load and validate datasets
- **`src/features/`**: Extract features from data
- **`src/models/`**: Build and compile models
- **`src/evaluation/`**: Evaluate trained models
- **`src/utils/`**: Use logging, config utilities
- **`config/`**: Load pipeline configurations
- **`artifacts/`**: Save trained models and metrics

## Best Practices
1. **Modular design**: Each pipeline stage is a separate function
2. **Configuration-driven**: Parameterize all pipeline settings
3. **Logging**: Log progress at each pipeline stage
4. **Error handling**: Catch and log errors with context
5. **Checkpointing**: Save intermediate results for recovery
6. **Validation**: Validate inputs/outputs at each stage
7. **Idempotency**: Re-running pipeline should be safe
8. **Documentation**: Clear docstrings for each pipeline

## Example Implementation
```python
# src/pipelines/training_pipeline.py
"""End-to-end training pipeline for depression detection."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
from datetime import datetime

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from src.data.loader import AudioDataLoader
from src.features.audio_processing import extract_mfcc
from src.models.model_builder import ModelBuilder
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for depression detection model.
    
    Orchestrates:
        - Data loading
        - Feature extraction
        - Model building
        - Training with callbacks
        - Evaluation
        - Artifact saving
    """
    
    def __init__(self, config_path: str):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.timestamp}_{self.config.get('experiment_name', 'experiment')}"
        
        # Setup output directories
        self.output_dir = Path(self.config['output_dir']) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized training pipeline: {self.experiment_name}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and validation data.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        logger.info("Loading training data...")
        
        # Load processed features (pre-extracted)
        data_dir = Path(self.config['data_dir'])
        
        X_train = np.load(data_dir / 'mfcc_features_train.npy')
        y_train = np.load(data_dir / 'labels_train.npy')
        X_val = np.load(data_dir / 'mfcc_features_val.npy')
        y_val = np.load(data_dir / 'labels_val.npy')
        
        logger.info(f"Loaded {len(X_train)} training samples, "
                   f"{len(X_val)} validation samples")
        
        # Add channel dimension if needed
        if len(X_train.shape) == 3:
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]
        
        return X_train, y_train, X_val, y_val
    
    def build_model(self, input_shape: Tuple) -> keras.Model:
        """
        Build model from configuration.
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building model...")
        
        # Update config with input shape
        model_config = self.config['model'].copy()
        model_config['input_shape'] = input_shape
        
        # Build model
        builder = ModelBuilder(model_config)
        model = builder.build()
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info(f"Model built: {model.count_params():,} parameters")
        
        return model
    
    def setup_callbacks(self) -> list:
        """
        Setup training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=self.config['training'].get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ))
        
        # Model checkpoint
        checkpoint_path = self.output_dir / 'best_model.h5'
        callbacks.append(ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        logger.info(f"Setup {len(callbacks)} training callbacks")
        
        return callbacks
    
    def train(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> keras.callbacks.History:
        """
        Train model.
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        callbacks = self.setup_callbacks()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
        return history
    
    def evaluate(
        self,
        model: keras.Model,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            model: Trained model
            X_val, y_val: Validation data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        evaluator = ModelEvaluator(model, self.experiment_name)
        metrics = evaluator.evaluate(X_val, y_val)
        
        # Generate report
        evaluator.generate_report(
            X_val, y_val,
            save_dir=str(self.output_dir / 'evaluation')
        )
        
        evaluator.print_summary()
        
        return metrics
    
    def save_artifacts(
        self,
        model: keras.Model,
        history: keras.callbacks.History,
        metrics: Dict[str, float]
    ) -> None:
        """
        Save training artifacts.
        
        Args:
            model: Trained model
            history: Training history
            metrics: Evaluation metrics
        """
        logger.info("Saving artifacts...")
        
        # Save final model
        model.save(self.output_dir / 'final_model.h5')
        
        # Save training history
        import json
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }, f, indent=2)
        
        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save config snapshot
        config_path = self.output_dir / 'config_snapshot.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Artifacts saved to {self.output_dir}")
    
    def run(self) -> keras.Model:
        """
        Execute full training pipeline.
        
        Returns:
            Trained Keras model
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting Training Pipeline: {self.experiment_name}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. Load data
            X_train, y_train, X_val, y_val = self.load_data()
            
            # 2. Build model
            input_shape = X_train.shape[1:]
            model = self.build_model(input_shape)
            
            # 3. Train model
            history = self.train(model, X_train, y_train, X_val, y_val)
            
            # 4. Evaluate model
            metrics = self.evaluate(model, X_val, y_val)
            
            # 5. Save artifacts
            self.save_artifacts(model, history, metrics)
            
            logger.info(f"{'='*60}")
            logger.info("Training pipeline completed successfully!")
            logger.info(f"{'='*60}")
            
            return model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise


def train_model(config_path: str) -> keras.Model:
    """
    Convenience function to run training pipeline.
    
    Args:
        config_path: Path to training configuration
        
    Returns:
        Trained model
    """
    pipeline = TrainingPipeline(config_path)
    return pipeline.run()
```

## Inference Pipeline Example
```python
# src/pipelines/inference_pipeline.py
"""Inference pipeline for production predictions."""

import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

from tensorflow import keras
import librosa

from src.features.audio_processing import extract_mfcc
from src.features.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production inference pipeline."""
    
    def __init__(
        self,
        model_path: str,
        normalizer_path: str = None,
        config: dict = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model
            normalizer_path: Path to saved normalizer
            config: Feature extraction configuration
        """
        self.model = keras.models.load_model(model_path)
        self.config = config or {}
        
        # Load normalizer if provided
        self.normalizer = None
        if normalizer_path:
            self.normalizer = FeatureNormalizer()
            self.normalizer.load(normalizer_path)
        
        logger.info(f"Inference pipeline initialized with model: {model_path}")
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict depression probability from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.config.get('sample_rate', 16000),
            duration=self.config.get('duration', 5.0)
        )
        
        # Extract features
        features = extract_mfcc(audio, sr=sr)
        
        # Normalize if normalizer available
        if self.normalizer:
            features = self.normalizer.transform(features[np.newaxis, ...])[0]
        
        # Add batch and channel dimensions
        features = features[np.newaxis, ..., np.newaxis]
        
        # Predict
        probability = self.model.predict(features, verbose=0)[0][0]
        
        return {
            'audio_path': audio_path,
            'depression_probability': float(probability),
            'prediction': 'depressed' if probability >= 0.5 else 'normal',
            'confidence': float(max(probability, 1 - probability))
        }
```

## Testing Pipelines
```python
# tests/test_training_pipeline.py
import pytest
from src.pipelines.training_pipeline import TrainingPipeline

def test_training_pipeline_initialization(tmp_path, sample_config):
    """Test pipeline can be initialized."""
    sample_config['output_dir'] = str(tmp_path)
    pipeline = TrainingPipeline(sample_config)
    assert pipeline is not None

def test_full_training_pipeline(tmp_path, sample_config, sample_data):
    """Test full pipeline execution (with small data)."""
    # This is integration test - may be slow
    pipeline = TrainingPipeline(sample_config)
    model = pipeline.run()
    assert model is not None
```
