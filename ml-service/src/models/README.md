# `src/models/` Directory

## Purpose
This module contains model architecture definitions, model building utilities, custom layers, and model management functionality.

## What Belongs Here
- **Model architectures**: CNN, 1D-CNN, MobileNet-inspired models
- **Custom layers**: Any custom TensorFlow/PyTorch layers
- **Model builders**: Factory functions to create models from config
- **Model utilities**: Model loading, saving, conversion helpers
- **Transfer learning**: Pre-trained model adaptation
- **Model optimization**: Quantization, pruning utilities

## What Should NOT Be Here
- ❌ Training loops (belongs in `src/pipelines/`)
- ❌ Evaluation metrics (belongs in `src/evaluation/`)
- ❌ Feature extraction (belongs in `src/features/`)
- ❌ Trained model files (belongs in `artifacts/models/`)

## Architectural Responsibilities
- **Model definitions**: Define all neural network architectures
- **Abstraction**: Provide clean interfaces for model creation
- **Configurability**: Build models from configuration parameters
- **Modularity**: Reusable components and building blocks

## Typical Modules

### `cnn_models.py`
CNN architecture definitions
```python
def create_lightweight_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int = 1
) -> keras.Model:
    """Create lightweight 2D CNN for MFCC features."""
    pass

def create_1d_cnn(input_shape: Tuple[int, int]) -> keras.Model:
    """Create 1D CNN for temporal features."""
    pass
```

### `mobile_models.py`
Mobile-optimized architectures
```python
def create_mobilenet_audio(
    input_shape: Tuple[int, int, int],
    alpha: float = 1.0
) -> keras.Model:
    """MobileNet-inspired architecture for audio."""
    pass

def depthwise_separable_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int]
) -> tf.Tensor:
    """Depthwise separable convolution block."""
    pass
```

### `model_builder.py`
Factory for creating models from config
```python
class ModelBuilder:
    """Build models from configuration."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def build(self) -> keras.Model:
        """Build model based on config."""
        pass
    
    @staticmethod
    def from_config(config_path: str) -> keras.Model:
        """Load config and build model."""
        pass
```

### `model_utils.py`
Model loading, saving, conversion utilities
```python
def save_model(model: keras.Model, save_path: str) -> None:
    """Save Keras model to disk."""
    pass

def load_model(model_path: str) -> keras.Model:
    """Load Keras model from disk."""
    pass

def convert_to_tflite(
    model: keras.Model,
    output_path: str,
    quantize: bool = True
) -> None:
    """Convert Keras model to TensorFlow Lite."""
    pass
```

### `custom_layers.py`
Custom layer implementations
```python
class AttentionLayer(keras.layers.Layer):
    """Custom attention mechanism for audio features."""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Initialize weights
        pass
    
    def call(self, inputs):
        # Forward pass
        pass
```

## Interactions with Other Modules
- **`src/pipelines/`**: Training pipeline uses models defined here
- **`src/evaluation/`**: Evaluation loads models for testing
- **`config/`**: Model hyperparameters loaded from config
- **`artifacts/models/`**: Trained models saved there
- **`src/utils/`**: May use logging, config utilities

## Best Practices
1. **Functional API**: Prefer Keras Functional API over Sequential for flexibility
2. **Type hints**: Annotate input/output shapes clearly
3. **Docstrings**: Document model architecture, input requirements, outputs
4. **Configuration**: Parameterize all hyperparameters
5. **Model summary**: Add method to print model architecture
6. **Input validation**: Validate input shapes in model builders
7. **Naming**: Use descriptive layer names for debugging
8. **Reproducibility**: Set random seeds for weight initialization

## Example Implementation
```python
# src/models/cnn_models.py
"""CNN model architectures for depression detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def create_lightweight_cnn(
    input_shape: Tuple[int, int, int] = (13, 215, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Create lightweight CNN optimized for mobile deployment.
    
    Architecture:
        - 3 Conv2D blocks with BatchNorm and MaxPooling
        - GlobalAveragePooling2D for spatial reduction
        - Dense layers for classification
    
    Args:
        input_shape: Input feature shape (n_mfcc, time_steps, channels)
        num_classes: Number of output classes (1 for binary)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
        
    Example:
        >>> model = create_lightweight_cnn(input_shape=(13, 215, 1))
        >>> model.summary()
    """
    inputs = keras.Input(shape=input_shape, name='mfcc_input')
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', 
                     padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', 
                     padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)
    
    # Block 3
    x = layers.Conv2D(64, (3, 3), activation='relu', 
                     padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Classifier
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate + 0.1, name='dropout3')(x)
    
    # Output layer
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Dense(num_classes, activation=activation, 
                          name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, 
                       name='lightweight_cnn')
    
    logger.info(f"Created lightweight CNN with {model.count_params():,} parameters")
    
    return model


def create_1d_cnn(
    input_shape: Tuple[int, int] = (13, 215),
    num_classes: int = 1,
    filters: list = [64, 128, 128]
) -> keras.Model:
    """
    Create 1D CNN for temporal audio features.
    
    Args:
        input_shape: Input shape (n_features, time_steps)
        num_classes: Number of output classes
        filters: List of filter sizes for each conv block
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # Convolutional blocks
    for i, num_filters in enumerate(filters):
        x = layers.Conv1D(
            num_filters, 3, activation='relu',
            padding='same', name=f'conv1d_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = layers.MaxPooling1D(2, name=f'pool_{i+1}')(x)
        x = layers.Dropout(0.2, name=f'dropout_{i+1}')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.4, name='dropout_fc')(x)
    
    # Output
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Dense(num_classes, activation=activation, 
                          name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_1d')
    
    logger.info(f"Created 1D CNN with {model.count_params():,} parameters")
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    metrics: list = None
) -> keras.Model:
    """
    Compile model with optimizer and loss.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    logger.info(f"Model compiled with lr={learning_rate}")
    
    return model
```

## Model Builder Pattern
```python
# src/models/model_builder.py
"""Model factory for creating models from configuration."""

from typing import Dict
import yaml
from tensorflow import keras
from src.models.cnn_models import create_lightweight_cnn, create_1d_cnn
import logging

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build models from configuration."""
    
    MODEL_REGISTRY = {
        'lightweight_cnn': create_lightweight_cnn,
        '1d_cnn': create_1d_cnn,
    }
    
    def __init__(self, config: Dict):
        """
        Initialize model builder.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
    
    def build(self) -> keras.Model:
        """
        Build model based on configuration.
        
        Returns:
            Keras model
            
        Raises:
            ValueError: If model type not recognized
        """
        model_type = self.config.get('model_type', 'lightweight_cnn')
        
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get model creation function
        model_fn = self.MODEL_REGISTRY[model_type]
        
        # Build model with config parameters
        model = model_fn(
            input_shape=tuple(self.config.get('input_shape', [13, 215, 1])),
            num_classes=self.config.get('num_classes', 1),
            **self.config.get('model_params', {})
        )
        
        logger.info(f"Built model: {model_type}")
        
        return model
    
    @classmethod
    def from_config_file(cls, config_path: str) -> keras.Model:
        """
        Load configuration from file and build model.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Keras model
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        builder = cls(config)
        return builder.build()
```

## TensorFlow Lite Conversion
```python
# src/models/model_utils.py
"""Model utility functions."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def convert_to_tflite(
    model: keras.Model,
    output_path: str,
    quantize: bool = True,
    representative_dataset = None
) -> None:
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model: Keras model to convert
        output_path: Path to save .tflite file
        quantize: Whether to apply quantization
        representative_dataset: Generator for int8 quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            # Full integer quantization
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save
    Path(output_path).write_bytes(tflite_model)
    
    # Log size comparison
    original_size = model.count_params() * 4 / 1024  # Rough estimate in KB
    tflite_size = len(tflite_model) / 1024
    
    logger.info(f"TFLite model saved to {output_path}")
    logger.info(f"Size: {tflite_size:.2f} KB (compression: {original_size/tflite_size:.2f}x)")
```

## Testing Models
```python
# tests/test_models.py
import pytest
from src.models.cnn_models import create_lightweight_cnn
import numpy as np

def test_model_creation():
    """Test model can be created."""
    model = create_lightweight_cnn(input_shape=(13, 215, 1))
    assert model is not None

def test_model_output_shape():
    """Test model output has correct shape."""
    model = create_lightweight_cnn()
    sample_input = np.random.rand(1, 13, 215, 1).astype(np.float32)
    output = model.predict(sample_input)
    assert output.shape == (1, 1)

def test_model_output_range():
    """Test sigmoid output is in [0, 1]."""
    model = create_lightweight_cnn()
    sample_input = np.random.rand(10, 13, 215, 1).astype(np.float32)
    predictions = model.predict(sample_input)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
```
