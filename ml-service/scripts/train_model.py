"""
Train a depression detection model.

Usage:
    python scripts/train_model.py --arch mel_cnn
    python scripts/train_model.py --arch bilstm --epochs 50
    python scripts/train_model.py --arch multi_feature --dropout 0.4
    python scripts/train_model.py --list
"""

import os
import sys
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.architectures import MODEL_REGISTRY
from src.pipelines.training_pipeline import run_training_pipeline
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train a depression detection model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--arch", type=str, default="mel_cnn",
        help="Architecture name. Use --list to see options.",
    )
    parser.add_argument("--list", action="store_true", help="List available architectures.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    args = parser.parse_args()

    if args.list:
        print("Available architectures:")
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:20s}  {info['description']}")
        return

    if args.arch not in MODEL_REGISTRY:
        print(f"Error: Unknown architecture '{args.arch}'. "
              f"Available: {', '.join(MODEL_REGISTRY)}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Depression Detection — {MODEL_REGISTRY[args.arch]['description']}")
    print("=" * 60)

    tflite_path = run_training_pipeline(
        args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
    )

    print("\n" + "=" * 60)
    print(f"  DONE! {args.arch} training complete")
    print("=" * 60)
    print(f"  TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
