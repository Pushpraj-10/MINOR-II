"""
Evaluate TFLite models on internal train/val/test splits.

Usage:
    python scripts/evaluate.py                          # Evaluate all models
    python scripts/evaluate.py --model mel_cnn bilstm   # Specific models
    python scripts/evaluate.py --tflite path/to/model.tflite  # Custom path
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.architectures import MODEL_REGISTRY
from src.pipelines.evaluation_pipeline import load_audio_splits, run_evaluation_pipeline
from src.config import MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFLite models.")
    parser.add_argument("--model", nargs="*", help="Architecture names to evaluate.")
    parser.add_argument("--tflite", type=str, help="Path to a specific TFLite model.")
    args = parser.parse_args()

    print("Loading audio data...")
    splits = load_audio_splits()

    if args.tflite:
        run_evaluation_pipeline(args.tflite, splits)
    else:
        model_names = args.model if args.model else list(MODEL_REGISTRY.keys())
        for name in model_names:
            if name not in MODEL_REGISTRY:
                print(f"\nSKIP: Unknown architecture '{name}'")
                continue
            tflite_path = os.path.join(MODEL_DIR, MODEL_REGISTRY[name]["tflite_name"])
            if not os.path.exists(tflite_path):
                print(f"\nSKIP: {name} — {tflite_path} not found")
                continue
            run_evaluation_pipeline(tflite_path, splits)


if __name__ == "__main__":
    main()
