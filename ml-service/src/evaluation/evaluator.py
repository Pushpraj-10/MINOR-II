"""Model evaluation utilities for depression detection."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from pathlib import Path
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metric name -> value
    """
    y_pred = (y_pred_proba >= threshold).astype(int).flatten()
    y_true = y_true.flatten()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "threshold": threshold,
    }

    logger.info(
        f"Evaluation: accuracy={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}, auc={metrics['roc_auc']:.4f}"
    )

    return metrics


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save metrics dictionary to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def plot_training_history(
    history,
    output_dir: str,
) -> None:
    """
    Plot training/validation loss and accuracy curves.

    Args:
        history: Keras training history object
        output_dir: Directory to save plots
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved to {out / 'training_history.png'}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str,
    threshold: float = 0.5,
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plot
        threshold: Classification threshold
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_pred = (y_pred_proba >= threshold).astype(int).flatten()
    y_true = y_true.flatten()

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Depression"],
        yticklabels=["Normal", "Depression"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {out / 'confusion_matrix.png'}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str,
) -> None:
    """
    Plot and save ROC curve.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plot
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred_proba.flatten())
    auc = roc_auc_score(y_true.flatten(), y_pred_proba.flatten())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {out / 'roc_curve.png'}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Print and return sklearn classification report."""
    y_pred = (y_pred_proba >= threshold).astype(int).flatten()
    y_true = y_true.flatten()

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Depression"]
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    return report
