"""
Train all model architectures sequentially and produce a comparison table.

Usage:
    python scripts/train_all.py                # Train all + compare
    python scripts/train_all.py --skip-training  # Compare existing TFLite models only
    python scripts/train_all.py --arch mel_cnn bilstm  # Train specific ones
"""

import os
import sys
import time
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.architectures import MODEL_REGISTRY
from src.pipelines.evaluation_pipeline import load_audio_splits, run_evaluation_pipeline
from src.config import MODEL_DIR


def run_training(arch_name):
    """Run the unified training script for one architecture."""
    script = os.path.join(PROJECT_ROOT, "scripts", "train_model.py")
    print(f"\n{'=' * 70}")
    print(f"  Training: {arch_name}")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script, "--arch", arch_name],
        capture_output=False, text=True,
    )
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {arch_name} — {elapsed:.1f}s")
    return result.returncode == 0, elapsed


def print_comparison_table(results):
    """Print a formatted comparison table sorted by test AUC."""
    print("\n" + "=" * 100)
    print("  MODEL COMPARISON — Depression Detection")
    print("=" * 100)

    header = (
        f"{'Model':<28} {'Test Acc':>9} {'Test AUC':>9} "
        f"{'Val Acc':>9} {'Val AUC':>9} {'Size(KB)':>9} {'ms/sample':>10}"
    )
    print(header)
    print("-" * 100)

    sorted_results = sorted(results, key=lambda r: r.get("test_auc", 0), reverse=True)
    for r in sorted_results:
        t_acc = f"{r['test_acc'] * 100:.2f}%" if r.get("test_acc") else "N/A"
        t_auc = f"{r['test_auc'] * 100:.2f}%" if r.get("test_auc") else "N/A"
        v_acc = f"{r['val_acc'] * 100:.2f}%" if r.get("val_acc") else "N/A"
        v_auc = f"{r['val_auc'] * 100:.2f}%" if r.get("val_auc") else "N/A"
        size = f"{r.get('size_kb', 0):.1f}" if r.get("size_kb") else "N/A"
        ms = f"{r.get('avg_ms', 0):.1f}" if r.get("avg_ms") else "N/A"
        print(f"{r['name']:<28} {t_acc:>9} {t_auc:>9} {v_acc:>9} {v_auc:>9} {size:>9} {ms:>10}")

    print("=" * 100)

    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST MODEL (by Test AUC): {best['name']}")
        print(f"  Test Accuracy: {best['test_acc'] * 100:.2f}%")
        print(f"  Test AUC:      {best['test_auc'] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train all models and compare.")
    parser.add_argument("--skip-training", action="store_true", help="Only compare existing TFLite models.")
    parser.add_argument("--arch", nargs="*", help="Train specific architectures only.")
    args = parser.parse_args()

    arch_names = args.arch if args.arch else list(MODEL_REGISTRY.keys())

    # Training phase
    if not args.skip_training:
        print("=" * 70)
        print("  TRAINING ALL MODELS")
        print("=" * 70)

        training_results = []
        for name in arch_names:
            if name not in MODEL_REGISTRY:
                print(f"\n  SKIP: Unknown architecture '{name}'")
                continue
            success, elapsed = run_training(name)
            training_results.append((name, success, elapsed))

        print("\n\n" + "=" * 70)
        print("  TRAINING SUMMARY")
        print("=" * 70)
        for name, success, elapsed in training_results:
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {name:<30} {elapsed:.1f}s")

    # Evaluation phase
    print("\n\n" + "=" * 70)
    print("  EVALUATING ALL TFLITE MODELS")
    print("=" * 70)

    print("\nLoading audio data...")
    splits = load_audio_splits()

    results = []
    for name in arch_names:
        if name not in MODEL_REGISTRY:
            continue
        tflite_path = os.path.join(MODEL_DIR, MODEL_REGISTRY[name]["tflite_name"])
        if not os.path.exists(tflite_path):
            print(f"\n  SKIP: {name} — {tflite_path} not found")
            continue

        print(f"\n  Evaluating: {name}...")
        size_kb = os.path.getsize(tflite_path) / 1024

        try:
            eval_results = run_evaluation_pipeline(
                tflite_path, splits, split_names=["val", "test"]
            )
            results.append({
                "name": name,
                "test_acc": eval_results["test"]["accuracy"],
                "test_auc": eval_results["test"]["roc_auc"],
                "val_acc": eval_results["val"]["accuracy"],
                "val_auc": eval_results["val"]["roc_auc"],
                "size_kb": size_kb,
                "avg_ms": eval_results["test"]["avg_ms"],
            })
        except Exception as e:
            print(f"    ERROR: {e}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
