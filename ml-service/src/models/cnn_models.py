"""CNN model architectures for depression detection.

All architectures are optimized for mobile deployment (small size, fast inference).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_lightweight_cnn(
    input_shape: Tuple[int, int, int] = (13, 215, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Lightweight 2D CNN optimized for mobile deployment.

    Architecture: 3 Conv blocks with BatchNorm + GlobalAvgPool + Dense
    Model size: ~100K-300K parameters
    Inference: 50-200ms on mobile

    Args:
        input_shape: (n_mfcc, time_steps, channels)
        num_classes: 1 for binary classification (sigmoid output)
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled Keras model
    """
    
    model = keras.Sequential(
        [
            # Block 1
            layers.Input(shape=input_shape),
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate + 0.1),
            # Block 3
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            # Classifier
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate + 0.1),
            layers.Dense(num_classes, activation="sigmoid"),
        ],
        name="lightweight_cnn",
    )

    logger.info(f"Created lightweight CNN: {model.count_params():,} parameters")
    return model


def create_1d_cnn(
    input_shape: Tuple[int, int] = (13, 215),
    num_classes: int = 1,
    dropout_rate: float = 0.3,
    filters: Optional[list] = None,
) -> keras.Model:
    """
    1D CNN for temporal audio features.

    Even smaller model suitable for very resource-constrained devices.
    Model size: ~50K-150K parameters

    Args:
        input_shape: (n_features, time_steps)
        num_classes: 1 for binary classification
        dropout_rate: Dropout rate
        filters: List of filter counts per block

    Returns:
        Compiled Keras model
    """
    if filters is None:
        filters = [64, 128, 128]

    model = keras.Sequential(name="1d_cnn")
    model.add(layers.Input(shape=input_shape))

    for i, n_filters in enumerate(filters):
        model.add(layers.Conv1D(n_filters, 3, activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout_rate + i * 0.05))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout_rate + 0.1))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    logger.info(f"Created 1D CNN: {model.count_params():,} parameters")
    return model


def create_mobilenet_audio(
    input_shape: Tuple[int, int, int] = (13, 215, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.5,
) -> keras.Model:
    """
    MobileNet-inspired architecture using depthwise separable convolutions.

    Better accuracy with efficient computation.
    Model size: ~200K-500K parameters

    Args:
        input_shape: (n_mfcc, time_steps, channels)
        num_classes: 1 for binary classification
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Depthwise separable blocks
    x = layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="mobilenet_audio")
    logger.info(f"Created MobileNet audio: {model.count_params():,} parameters")
    return model


def build_model(
    architecture: str,
    input_shape: Tuple,
    num_classes: int = 1,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Factory function to build a model by architecture name.

    Args:
        architecture: One of 'lightweight_cnn', '1d_cnn', 'mobilenet'
        input_shape: Input shape for the model
        num_classes: Number of output classes
        dropout_rate: Dropout rate

    Returns:
        Keras model instance
    """
    builders = {
        "lightweight_cnn": create_lightweight_cnn,
        "1d_cnn": create_1d_cnn,
        "mobilenet": create_mobilenet_audio,
    }

    if architecture not in builders:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: {list(builders.keys())}"
        )

    return builders[architecture](
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    loss: str = "binary_crossentropy",
    metrics: Optional[list] = None,
) -> keras.Model:
    """
    Compile model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        loss: Loss function name
        metrics: List of metric names

    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ["accuracy", keras.metrics.AUC(name="auc")]
    else:
        parsed_metrics = []
        for m in metrics:
            if m.lower() == "auc":
                parsed_metrics.append(keras.metrics.AUC(name="auc"))
            else:
                parsed_metrics.append(m)
        metrics = parsed_metrics

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    logger.info(f"Model compiled: lr={learning_rate}, loss={loss}")
    return model
