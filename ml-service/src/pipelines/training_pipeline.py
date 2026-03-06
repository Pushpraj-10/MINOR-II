"""End-to-end training pipeline for depression detection model."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import json
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset
from src.features.audio_processing import extract_features_from_dataset
from src.models.cnn_models import build_model, compile_model
from src.models.model_utils import convert_to_tflite
from src.evaluation.evaluator import (
    evaluate_model,
    save_metrics,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
)
from src.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for depression detection model.

    Orchestrates:
        - Data loading from raw audio files
        - Feature extraction (MFCC)
        - Train/val/test splitting
        - Model building and compilation
        - Training with callbacks
        - Evaluation and metrics
        - Model export (Keras + TFLite)
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = (
            f"{config['artifacts']['experiment_name']}_{self.timestamp}"
        )

        # Ensure output directories exist
        self.models_dir = ensure_dir(config["artifacts"]["models_dir"])
        self.metrics_dir = ensure_dir(config["artifacts"]["metrics_dir"])
        self.plots_dir = ensure_dir(config["artifacts"]["plots_dir"])

        logger.info(f"Initialized training pipeline: {self.experiment_name}")

    def load_data(self) -> Tuple[list, np.ndarray]:
        """
        Load audio data from raw directory.

        Returns:
            Tuple of (audio_list, labels_array)
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data")
        logger.info("=" * 60)

        data_cfg = self.config["data"]
        audio_cfg = self.config["audio"]

        loader = AudioDataLoader(
            data_dir=data_cfg["raw_dir"],
            sample_rate=audio_cfg["sample_rate"],
            duration=audio_cfg["duration"],
            mono=audio_cfg["mono"],
        )

        audio_list, labels, file_paths = loader.load_dataset(
            depression_dir=data_cfg["depression_dir"],
            normal_dir=data_cfg["normal_dir"],
            extensions=data_cfg["audio_extensions"],
        )

        if len(audio_list) == 0:
            raise RuntimeError(
                f"No audio files found in {data_cfg['raw_dir']}. "
                f"Please add .wav files to {data_cfg['depression_dir']}/ "
                f"and {data_cfg['normal_dir']}/ directories."
            )

        labels = np.array(labels)
        logger.info(f"Loaded {len(audio_list)} audio samples")

        return audio_list, labels

    def extract_features(self, audio_list: list) -> np.ndarray:
        """
        Extract features from audio data.

        Args:
            audio_list: List of audio numpy arrays

        Returns:
            Feature array ready for model input
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting features")
        logger.info("=" * 60)

        feat_cfg = self.config["features"]

        X = extract_features_from_dataset(
            audio_list=audio_list,
            feature_type=feat_cfg["type"],
            config={
                "sample_rate": self.config["audio"]["sample_rate"],
                "n_mfcc": feat_cfg["n_mfcc"],
                "n_fft": feat_cfg["n_fft"],
                "hop_length": feat_cfg["hop_length"],
                "n_mels": feat_cfg["n_mels"],
            },
        )

        logger.info(f"Feature shape: {X.shape}")
        return X

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets.

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Splitting dataset")
        logger.info("=" * 60)

        data_cfg = self.config["data"]

        splits = split_dataset(
            X,
            y,
            test_size=data_cfg["test_size"],
            val_size=data_cfg["val_size"],
            random_state=data_cfg["random_state"],
        )

        return splits

    def build_model(self, input_shape: Tuple) -> keras.Model:
        """
        Build and compile the model.

        Args:
            input_shape: Shape of input features (without batch dimension)

        Returns:
            Compiled Keras model
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Building model")
        logger.info("=" * 60)

        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        model = build_model(
            architecture=model_cfg["architecture"],
            input_shape=input_shape,
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )

        model = compile_model(
            model,
            learning_rate=train_cfg["learning_rate"],
            loss=train_cfg["loss"],
            metrics=train_cfg["metrics"],
        )

        model.summary(print_fn=logger.info)
        return model

    def _get_callbacks(self) -> list:
        """Create training callbacks."""
        train_cfg = self.config["training"]

        checkpoint_path = str(
            self.models_dir / f"{self.experiment_name}_best.keras"
        )

        callbacks = [
            EarlyStopping(
                monitor=train_cfg["early_stopping"]["monitor"],
                patience=train_cfg["early_stopping"]["patience"],
                restore_best_weights=train_cfg["early_stopping"][
                    "restore_best_weights"
                ],
                verbose=1,
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor=train_cfg["checkpoint"]["monitor"],
                save_best_only=train_cfg["checkpoint"]["save_best_only"],
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor=train_cfg["reduce_lr"]["monitor"],
                factor=train_cfg["reduce_lr"]["factor"],
                patience=train_cfg["reduce_lr"]["patience"],
                min_lr=train_cfg["reduce_lr"]["min_lr"],
                verbose=1,
            ),
        ]

        return callbacks

    def train(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Train the model.

        Returns:
            Training history object
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Training model")
        logger.info("=" * 60)

        train_cfg = self.config["training"]
        callbacks = self._get_callbacks()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(f"Training completed after {len(history.history['loss'])} epochs")
        return history

    def evaluate(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history=None,
    ) -> Dict:
        """
        Evaluate model and generate reports.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Evaluating model")
        logger.info("=" * 60)

        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)

        # Compute metrics
        metrics = evaluate_model(y_test, y_pred_proba)

        # Save metrics
        metrics_path = str(self.metrics_dir / f"{self.experiment_name}_metrics.json")
        save_metrics(metrics, metrics_path)

        # Print classification report
        print_classification_report(y_test, y_pred_proba)

        # Generate plots
        plots_dir = str(self.plots_dir / self.experiment_name)
        if history:
            plot_training_history(history, plots_dir)
        plot_confusion_matrix(y_test, y_pred_proba, plots_dir)
        plot_roc_curve(y_test, y_pred_proba, plots_dir)

        return metrics

    def export_model(
        self, model: keras.Model, X_train: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Export model in Keras and TFLite formats.

        Returns:
            Dictionary with paths to exported models
        """
        logger.info("=" * 60)
        logger.info("STEP 7: Exporting model")
        logger.info("=" * 60)

        exports = {}

        # Save Keras model
        keras_path = str(self.models_dir / f"{self.experiment_name}_final.keras")
        model.save(keras_path)
        exports["keras"] = keras_path
        logger.info(f"Keras model saved: {keras_path}")

        # Convert to TFLite
        opt_cfg = self.config.get("optimization", {})
        if opt_cfg.get("quantize", True):
            tflite_path = str(
                self.models_dir / f"{self.experiment_name}.tflite"
            )
            convert_to_tflite(
                model,
                tflite_path,
                quantize=True,
                quantization_type=opt_cfg.get("quantization_type", "dynamic"),
                representative_data=X_train,
            )
            exports["tflite"] = tflite_path

        return exports

    def run(self) -> Dict:
        """
        Execute the full training pipeline.

        Returns:
            Dictionary with metrics, model paths, and experiment info
        """
        logger.info("=" * 60)
        logger.info(f"STARTING TRAINING PIPELINE: {self.experiment_name}")
        logger.info("=" * 60)

        # Step 1: Load data
        audio_list, labels = self.load_data()

        # Step 2: Extract features
        X = self.extract_features(audio_list)

        # Step 3: Split data
        splits = self.split_data(X, labels)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Step 4: Build model
        input_shape = X_train.shape[1:]  # (n_mfcc, time_steps, 1)
        model = self.build_model(input_shape)

        # Step 5: Train
        history = self.train(model, X_train, y_train, X_val, y_val)

        # Step 6: Evaluate
        metrics = self.evaluate(model, X_test, y_test, history)

        # Step 7: Export
        exports = self.export_model(model, X_train)

        # Save final summary
        result = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {
                "architecture": self.config["model"]["architecture"],
                "feature_type": self.config["features"]["type"],
                "epochs_trained": len(history.history["loss"]),
                "batch_size": self.config["training"]["batch_size"],
            },
            "data": {
                "total_samples": len(labels),
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test),
                "input_shape": list(input_shape),
            },
            "metrics": metrics,
            "exports": exports,
        }

        summary_path = self.metrics_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:  {metrics['roc_auc']:.4f}")
        logger.info(f"Models:   {exports}")
        logger.info("=" * 60)

        return result
