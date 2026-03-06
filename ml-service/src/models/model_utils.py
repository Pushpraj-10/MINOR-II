"""Model conversion and optimization utilities for mobile deployment."""

from typing import Optional

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
    quantization_type: str = "dynamic",
    representative_data: Optional[np.ndarray] = None,
) -> str:
    
    """
    Convert Keras model to TensorFlow Lite format.

    Args:
        model: Trained Keras model
        output_path: Path to save .tflite file
        quantize: Whether to apply quantization
        quantization_type: 'dynamic' or 'full_integer'
        representative_data: Sample data for full integer quantization

    Returns:
        Path to saved TFLite model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization_type == "full_integer" and representative_data is not None:

            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i : i + 1].astype(np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            logger.info("Using full integer quantization")
        else:
            logger.info("Using dynamic range quantization")

    tflite_model = converter.convert()

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(tflite_model)

    # Report sizes
    original_size = model.count_params() * 4 / 1024  # rough estimate in KB
    tflite_size = len(tflite_model) / 1024
    logger.info(f"TFLite model saved: {output}")
    logger.info(f"TFLite size: {tflite_size:.2f} KB")

    return str(output)
