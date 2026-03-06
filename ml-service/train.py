"""
Depression Detection Model - Training Entry Point
===================================================

Usage:
    python train.py                              # Use default config
    python train.py --config config/custom.yaml  # Use custom config

This script orchestrates the full training pipeline:
    1. Load raw audio from data/raw/voice_data/
    2. Extract MFCC features
    3. Split into train/val/test
    4. Train lightweight CNN
    5. Evaluate (accuracy, F1, AUC, confusion matrix, ROC)
    6. Export to Keras + TFLite for mobile deployment
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.pipelines.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train depression detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    log_cfg = config.get("logging", {})
    logger = setup_logger(
        name="ml_service",
        log_dir=log_cfg.get("log_dir", "artifacts/logs"),
        level=getattr(
            __import__("logging"), log_cfg.get("level", "INFO")
        ),
    )

    logger.info("Depression Detection Model Training")
    logger.info(f"Config: {args.config}")

    # Run training pipeline
    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Experiment:  {result['experiment']}")
    print(f"Samples:     {result['data']['total_samples']} total")
    print(f"Input Shape: {result['data']['input_shape']}")
    print(f"Accuracy:    {result['metrics']['accuracy']:.4f}")
    print(f"F1 Score:    {result['metrics']['f1_score']:.4f}")
    print(f"ROC AUC:     {result['metrics']['roc_auc']:.4f}")
    print(f"Models:      {result['exports']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
