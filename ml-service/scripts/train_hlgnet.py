"""
HLG-Net training script for depression severity prediction.

Usage:
    python scripts/train_hlgnet.py --dataset daicwoz
    python scripts/train_hlgnet.py --dataset depression
    python scripts/train_hlgnet.py --dataset combined
    python scripts/train_hlgnet.py --dataset daicwoz --epochs 2 --batch-size 4  # quick test
"""

import os
import sys
import time
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hlg_net import HLGNet
from src.data.mfcc_pipeline import create_dataloaders
from src.data.daicwoz_loader import load_daicwoz_splits
from src.data.depression_dataset_loader import load_depression_dataset
from src.data.eatd_hlgnet_loader import load_eatd_splits, SDS_DEPRESSION_THRESHOLD
from src.config import (
    DAICWOZ_DIR,
    DEPRESSION_DATASET_DIR,
    EATD_CORPUS_DIR,
    HLGNET_EPOCHS,
    HLGNET_LR,
    HLGNET_BATCH_SIZE,
    HLGNET_N_MFCC,
    HLGNET_D_MODEL,
    HLGNET_NUM_HEADS,
)
from sklearn.metrics import f1_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = "artifacts/models/hlgnet"
CACHE_DIR = "data/interim/hlgnet_cache"


def _prepare_daicwoz_entries(daicwoz_dir: str):
    """Load DAICWOZ and convert to (audio_path, binary_label)."""
    splits = load_daicwoz_splits(daicwoz_dir)
    result = {}
    for split_name, entries in splits.items():
        # entries: List[(audio_path, binary_label, phq8_score)]
        result[split_name] = [(path, float(binary)) for path, binary, _ in entries]
    return result


def _prepare_eatd_entries(eatd_dir: str):
    """Load EATD-Corpus and convert to (audio_path, binary_label)."""
    splits = load_eatd_splits(eatd_dir)
    result = {}
    for split_name, entries in splits.items():
        # entries: List[(audio_path, sds_score)]
        result[split_name] = [(path, float(score >= SDS_DEPRESSION_THRESHOLD)) for path, score in entries]
    return result


def _prepare_depression_entries(dataset_dir: str):
    """Load dataset-depression and convert to (audio_path, binary_label)."""
    entries = load_depression_dataset(dataset_dir)
    # entries: List[(audio_path, binary_label, proxy_score)]

    # Simple 80/10/10 split (no speaker independence needed — acted data)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(entries))
    n_train = int(0.8 * len(entries))
    n_val = int(0.1 * len(entries))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": [(entries[i][0], float(entries[i][1])) for i in train_idx],
        "val": [(entries[i][0], float(entries[i][1])) for i in val_idx],
        "test": [(entries[i][0], float(entries[i][1])) for i in test_idx],
    }


def _prepare_combined_entries(daicwoz_dir: str, eatd_dir: str, depression_dir: str):
    """Combine DAICWOZ, EATD, and dataset-depression entries."""
    daicwoz = _prepare_daicwoz_entries(daicwoz_dir)
    eatd = _prepare_eatd_entries(eatd_dir)
    depression = _prepare_depression_entries(depression_dir)

    combined = {}
    for split in ["train", "val", "test"]:
        combined[split] = daicwoz.get(split, []) + eatd.get(split, []) + depression.get(split, [])
    return combined


def train(
    dataset: str = "daicwoz",
    epochs: int = HLGNET_EPOCHS,
    batch_size: int = HLGNET_BATCH_SIZE,
    lr: float = HLGNET_LR,
    device: str = "auto",
):
    """Train HLG-Net model.

    Args:
        dataset: One of "daicwoz", "depression", "combined".
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: "auto", "cuda", or "cpu".
    """
    # ── Device setup ──────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Using device: %s", device)

    # ── Data preparation ──────────────────────────────────────────────
    logger.info("Loading dataset: %s", dataset)
    if dataset == "daicwoz":
        entries = _prepare_daicwoz_entries(DAICWOZ_DIR)
    elif dataset == "eatd":
        entries = _prepare_eatd_entries(EATD_CORPUS_DIR)
    elif dataset == "depression":
        entries = _prepare_depression_entries(DEPRESSION_DATASET_DIR)
    elif dataset == "combined":
        entries = _prepare_combined_entries(DAICWOZ_DIR, EATD_CORPUS_DIR, DEPRESSION_DATASET_DIR)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use: daicwoz, eatd, depression, combined")

    num_pos = 0
    num_neg = 0
    for split, e in entries.items():
        labels = [l for _, l in e]
        pos = int(sum(labels))
        neg = len(labels) - pos
        logger.info("  %s: %d samples, (Depressed: %d, Normal: %d)", split, len(e), pos, neg)
        if split == "train":
            num_pos += pos
            num_neg += neg

    loaders = create_dataloaders(
        entries,
        batch_size=batch_size,
        cache_dir=os.path.join(CACHE_DIR, dataset),
        num_workers=0,
    )

    if "train" not in loaders:
        logger.error("No training data available. Aborting.")
        return

    # ── Model setup ───────────────────────────────────────────────────
    model = HLGNet(
        input_dim=HLGNET_N_MFCC,
        d_model=HLGNET_D_MODEL,
        num_heads=HLGNET_NUM_HEADS,
    ).to(device)

    logger.info("HLG-Net parameters: %d (%.3fM)", model.count_parameters(), model.count_parameters() / 1e6)

    # BCEWithLogitsLoss with dynamic class weighting to heavily penalize missing depression cases
    # Due to severe imbalance, pos_weight emphasizes Recall on the Depressed minority class
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32).to(device)
    logger.info("Calculated BCE positive class weight: %.2f (Normal: %d / Depressed: %d)", pos_weight.item(), num_neg, num_pos)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ─────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_f1 = -1.0  # Track max F1 score instead of MAE
    history = {"train_loss": [], "val_f1": [], "val_recall": []}

    for epoch in range(epochs):
        t0 = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        n_batches = 0
        for mfcc, score in loaders["train"]:
            mfcc = mfcc.to(device)
            score = score.squeeze(-1).to(device) # now contains 0 or 1 binary label

            optimizer.zero_grad()
            pred_logit = model(mfcc)
            loss = criterion(pred_logit, score)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        val_f1 = 0.0
        val_recall = 0.0
        if "val" in loaders:
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for mfcc, score in loaders["val"]:
                    mfcc = mfcc.to(device)
                    score = score.squeeze(-1)
                    
                    pred_logit = model(mfcc)
                    pred_prob = torch.sigmoid(pred_logit)
                    
                    all_preds.extend((pred_prob > 0.5).cpu().numpy().astype(int))
                    all_targets.extend(score.numpy().astype(int))
            
            if all_targets:
                val_f1 = float(f1_score(all_targets, all_preds, zero_division=0))
                val_recall = float(recall_score(all_targets, all_preds, zero_division=0))

        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %3d/%d  |  Train Loss: %.4f  |  Val F1: %.3f  |  Val Recall: %.3f  |  Time: %.1fs",
            epoch + 1, epochs, avg_train_loss, val_f1, val_recall, elapsed,
        )

        # Save best model (optimizing for F1 instead of MAE)
        if val_f1 > best_val_f1 or epoch == 0:
            best_val_f1 = val_f1
            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "dataset": dataset,
            }, save_path)
            logger.info("  → Saved best model (F1: %.3f, Recall: %.3f)", val_f1, val_recall)

    # Save final model
    final_path = os.path.join(MODEL_DIR, "final_model.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": history["val_f1"][-1] if history["val_f1"] else None,
        "dataset": dataset,
    }, final_path)

    # Save training history
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\nTraining complete!")
    logger.info("  Best Val F1:  %.3f", best_val_f1)
    logger.info("  Best model:   %s", os.path.join(MODEL_DIR, "best_model.pt"))
    logger.info("  Final model:  %s", final_path)
    logger.info("  History:      %s", history_path)


def main():
    parser = argparse.ArgumentParser(description="Train HLG-Net with Binary Classification for depression detection")
    parser.add_argument("--dataset", type=str, default="combined",
                        choices=["daicwoz", "eatd", "depression", "combined"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=HLGNET_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=HLGNET_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=HLGNET_LR, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
