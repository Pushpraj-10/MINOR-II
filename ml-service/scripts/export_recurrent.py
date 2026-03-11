"""
Export a trained recurrent (GRU / BiLSTM) model to TFLite.

The TFLite model takes pre-extracted multi-feature input (1, 313, 46) and
outputs a depression probability (1, 1). Feature extraction (MFCC, Delta MFCC,
Chroma, Spectral Contrast, ZCR) is performed outside this model.

Usage:
    python main.py export-recurrent --cell gru
    python main.py export-recurrent --cell bilstm
    python main.py export-recurrent --model path/to/custom_best.keras --cell gru
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.export.tflite_converter import convert_to_tflite
from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

# Must match the custom layer defined in train_recurrent.py
class _Attention(keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(int(input_shape[-1]), 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(int(input_shape[1]), 1),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


def _verify_tflite(tflite_path: str):
    """Run a random sample through the TFLite model to confirm it runs."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    dummy = np.random.rand(1, 313, 46).astype(np.float32)
    interpreter.set_tensor(inp_det[0]["index"], dummy)
    interpreter.invoke()
    result = interpreter.get_tensor(out_det[0]["index"])
    print(f"  TFLite test inference: {result[0][0]:.4f} (expected 0–1)  ✓")
    print(f"  Input  shape: {inp_det[0]['shape']}  dtype: {inp_det[0]['dtype']}")
    print(f"  Output shape: {out_det[0]['shape']}  dtype: {out_det[0]['dtype']}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained recurrent model to TFLite"
    )
    parser.add_argument("--cell", type=str, default="bilstm",
                        choices=["gru", "bilstm"],
                        help="Recurrent cell type (default: bilstm)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override path to .keras model (default: auto-detect best)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override path for output .tflite file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    arch_dir = os.path.join(MODEL_DIR, f"recurrent_eatd_{args.cell}")
    keras_path = args.model or os.path.join(arch_dir, f"{args.cell}_best.keras")
    tflite_path = args.output or os.path.join(arch_dir, f"{args.cell}_best.tflite")

    if not os.path.exists(keras_path):
        print(f"Error: Model not found at {keras_path}")
        print(f"  Train first with: python main.py train-recurrent --cell {args.cell}")
        sys.exit(1)

    print(f"\nLoading model: {keras_path}")
    model = keras.models.load_model(
        keras_path,
        custom_objects={"_Attention": _Attention},
    )
    model.summary()

    print(f"\nExporting TFLite → {tflite_path}")
    convert_to_tflite(model, tflite_path)

    print(f"\nVerifying TFLite...")
    _verify_tflite(tflite_path)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"\nDone. TFLite model: {tflite_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
