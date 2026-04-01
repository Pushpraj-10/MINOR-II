"""
HLG-Net evaluation script for depression detection (Binary Classification).

Computes 2-class accuracy, precision, recall, F1, and AUC on test data.
Supports cross-dataset evaluation.

Usage:
    python scripts/evaluate_hlgnet.py --model-path artifacts/models/hlgnet/best_model.pt --dataset daicwoz
    python scripts/evaluate_hlgnet.py --model-path artifacts/models/hlgnet/best_model.pt --dataset depression
    python scripts/evaluate_hlgnet.py --model-path artifacts/models/hlgnet/best_model.pt --dataset all
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hlg_net import HLGNet
from src.data.mfcc_pipeline import create_dataloaders
from src.data.daicwoz_loader import load_daicwoz_splits
from src.data.depression_dataset_loader import load_depression_dataset
from src.data.eatd_hlgnet_loader import load_eatd_splits, SDS_DEPRESSION_THRESHOLD
from src.data.ravdess_loader import load_ravdess_dataset
from src.config import (
    DAICWOZ_DIR,
    DEPRESSION_DATASET_DIR,
    EATD_CORPUS_DIR,
    RAW_RAVDESS_DIR,
    HLGNET_N_MFCC,
    HLGNET_D_MODEL,
    HLGNET_NUM_HEADS,
    HLGNET_BATCH_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "artifacts/evaluation/hlgnet"


def _load_model(model_path: str, device: torch.device) -> HLGNet:
    """Load a trained HLG-Net model from checkpoint."""
    model = HLGNet(
        input_dim=HLGNET_N_MFCC,
        d_model=HLGNET_D_MODEL,
        num_heads=HLGNET_NUM_HEADS,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded model from %s (epoch %d, val_f1=%.3f)",
                model_path, checkpoint.get("epoch", -1), checkpoint.get("val_f1", -1))
    return model


def _collect_predictions(
    model: HLGNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """Run inference on a DataLoader and collect classification probabilities + targets."""
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for mfcc, target in loader:
            mfcc = mfcc.to(device)
            logit = model(mfcc)
            prob = torch.sigmoid(logit)
            all_probs.extend(prob.cpu().numpy())
            all_targets.extend(target.squeeze(-1).numpy())

    return np.array(all_probs), np.array(all_targets)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    dataset_name: str = "test",
) -> dict:
    """Compute classification metrics metrics.

    Args:
        probs: Predicted probability scores ([0, 1]).
        targets: Ground truth binary labels (0 or 1).
        dataset_name: Name for logging.

    Returns:
        Dict of computed metrics.
    """
    # Binary classification metrics (prob > 0.5 = depressed)
    pred_class = (probs > 0.5).astype(int)
    true_class = targets.astype(int)

    acc = float(accuracy_score(true_class, pred_class))
    prec = float(precision_score(true_class, pred_class, zero_division=0))
    rec = float(recall_score(true_class, pred_class, zero_division=0))
    f1 = float(f1_score(true_class, pred_class, zero_division=0))
    cm = confusion_matrix(true_class, pred_class, labels=[0, 1]).tolist()
    
    unique_classes = np.unique(true_class)
    try:
        if len(unique_classes) > 1:
            roc_auc = float(roc_auc_score(true_class, probs))
            pr_auc = float(average_precision_score(true_class, probs))
        else:
            roc_auc = 0.0
            pr_auc = 0.0
    except Exception:
        roc_auc = 0.0
        pr_auc = 0.0

    metrics = {
        "dataset": dataset_name,
        "n_samples": len(targets),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }

    # Print formatted report
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name} Evaluation Results ({len(targets)} samples)")
    print(f"{'=' * 60}")
    print(f"  Classification (threshold=0.5):")
    print(f"    Recall (Depressed): {rec:.4f}  <-- MAIN METRIC")
    print(f"    F1-Score:           {f1:.4f}")
    print(f"    Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"    Precision:          {prec:.4f}")
    print(f"    ROC-AUC:            {roc_auc:.4f}")
    print(f"    PR-AUC:             {pr_auc:.4f}")
    print(f"  Confusion Matrix:")
    if len(cm) >= 2 and isinstance(cm[0], list) and len(cm[0]) >= 2:
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    else:
        print(f"    {cm}")
    print()

    # Detailed classification report
    report = classification_report(
        true_class, pred_class,
        labels=[0, 1],
        target_names=["Normal", "Depressed"],
        zero_division=0,
    )
    print(report)

    return metrics


def evaluate(
    model_path: str,
    dataset: str = "daicwoz",
    device: str = "auto",
    batch_size: int = HLGNET_BATCH_SIZE,
):
    """Evaluate HLG-Net on specified dataset(s).

    Args:
        model_path: Path to trained model checkpoint.
        dataset: One of "daicwoz", "depression", "eatd", "ravdess", "all".
        device: "auto", "cuda", or "cpu".
        batch_size: Batch size for inference.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = _load_model(model_path, device)
    all_results = {}

    datasets_to_eval = []
    if dataset in ("daicwoz", "all"):
        datasets_to_eval.append("daicwoz")
    if dataset in ("depression", "all"):
        datasets_to_eval.append("depression")
    if dataset in ("eatd", "all"):
        datasets_to_eval.append("eatd")
    if dataset in ("ravdess", "all"):
        datasets_to_eval.append("ravdess")

    for ds_name in datasets_to_eval:
        logger.info("Evaluating on: %s", ds_name)

        if ds_name == "daicwoz":
            splits = load_daicwoz_splits(DAICWOZ_DIR)
            eval_entries = {
                split: [(p, float(b)) for p, b, _ in entries]
                for split, entries in splits.items()
            }
        elif ds_name == "depression":
            entries = load_depression_dataset(DEPRESSION_DATASET_DIR)
            eval_entries = {
                "test": [(p, float(b)) for p, b, _ in entries],
            }
        elif ds_name == "eatd":
            splits = load_eatd_splits(EATD_CORPUS_DIR)
            eval_entries = {
                split: [(p, float(s >= SDS_DEPRESSION_THRESHOLD)) for p, s in entries]
                for split, entries in splits.items()
            }
        elif ds_name == "ravdess":
            entries = load_ravdess_dataset(RAW_RAVDESS_DIR)
            eval_entries = {
                # Proxy score 20 = Depressed (1), 0 = Normal (0)
                "test": [(p, float(s > 10)) for p, s in entries],
            }

        loaders = create_dataloaders(
            eval_entries,
            batch_size=batch_size,
            num_workers=0,
        )

        for split_name, loader in loaders.items():
            probs, targets = _collect_predictions(model, loader, device)
            metrics = compute_metrics(
                probs, targets,
                dataset_name=f"{ds_name}/{split_name}",
            )
            all_results[f"{ds_name}_{split_name}"] = metrics

    # Save all results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"results_{dataset}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate HLG-Net for depression detection")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="daicwoz",
                        choices=["daicwoz", "depression", "eatd", "ravdess", "all"],
                        help="Dataset(s) to evaluate on")
    parser.add_argument("--batch-size", type=int, default=HLGNET_BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
