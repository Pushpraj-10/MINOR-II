"""
Run all model training scripts sequentially and produce a comparison table.

Usage:
    python run_all_models.py           # Train all 5 new models + existing 2
    python run_all_models.py --skip-training  # Just compare existing TFLite models
"""

import os
import sys
import subprocess
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "data/raw/voice_data"
SAMPLE_RATE = 16000
DURATION = 5.0
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
MODEL_DIR = "artifacts/models"

# All model training scripts
TRAINING_SCRIPTS = [
    ("Mel CNN (4-block)", "train_mel_cnn.py"),
    ("BiLSTM", "train_lstm.py"),
    ("CNN-LSTM Hybrid", "train_cnn_lstm.py"),
    ("Multi-Feature Fusion", "train_multi_feature.py"),
    ("CNN + Attention", "train_attention.py"),
    ("Separable CNN", "train_separable_cnn.py"),
]

# TFLite model paths for comparison
TFLITE_MODELS = {
    "MFCC CNN (original)":          "artifacts/models/depression_detection_combined.tflite",
    "Mel CNN (4-block)":            "artifacts/models/mel_depression_combined.tflite",
    "BiLSTM":                       "artifacts/models/lstm_depression_combined.tflite",
    "CNN-LSTM Hybrid":              "artifacts/models/cnn_lstm_depression_combined.tflite",
    "Multi-Feature Fusion":         "artifacts/models/multi_feature_depression_combined.tflite",
    "CNN + Attention":              "artifacts/models/attention_depression_combined.tflite",
    "Separable CNN":                "artifacts/models/separable_cnn_depression_combined.tflite",
}


def load_audio_data():
    """Load audio data and create splits for evaluation."""
    loader = AudioDataLoader(data_dir=DATA_DIR, sample_rate=SAMPLE_RATE, duration=DURATION, mono=True)
    audio_list, labels, _ = loader.load_dataset(depression_dir="depression1", normal_dir="normal1")
    audio_arr = np.array(audio_list)
    y = np.array(labels)
    splits = split_dataset(audio_arr, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)
    return splits


def evaluate_tflite(tflite_path, audio_list, y_true):
    """Run TFLite inference and return metrics."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    preds = []
    times = []
    for audio in audio_list:
        audio_in = audio.astype(np.float32)
        if len(audio_in) < AUDIO_LENGTH:
            audio_in = np.pad(audio_in, (0, AUDIO_LENGTH - len(audio_in)))
        else:
            audio_in = audio_in[:AUDIO_LENGTH]

        t0 = time.perf_counter()
        interpreter.set_tensor(inp[0]["index"], audio_in[np.newaxis])
        interpreter.invoke()
        t1 = time.perf_counter()

        preds.append(interpreter.get_tensor(out[0]["index"])[0][0])
        times.append(t1 - t0)

    preds = np.array(preds)
    y_pred = (preds >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, preds) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)
    avg_time_ms = np.mean(times) * 1000

    return {
        "accuracy": acc,
        "auc": auc,
        "tn": int(cm[0][0]), "fp": int(cm[0][1]),
        "fn": int(cm[1][0]), "tp": int(cm[1][1]),
        "avg_inference_ms": avg_time_ms,
    }


def get_model_size_kb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / 1024
    return 0


def run_training(script_name, model_name):
    """Run a training script and capture result."""
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"  Script:   {script_name}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {model_name} — {elapsed:.1f}s")
    return result.returncode == 0, elapsed


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 110)
    print("  MODEL COMPARISON — Depression Detection")
    print("=" * 110)

    # Header
    header = f"{'Model':<28} {'Test Acc':>9} {'Test AUC':>9} {'Val Acc':>9} {'Val AUC':>9} {'Size(KB)':>9} {'ms/sample':>10} {'Flex?':>6}"
    print(header)
    print("-" * 110)

    # Sort by test AUC descending
    sorted_results = sorted(results, key=lambda r: r.get("test_auc", 0), reverse=True)

    for r in sorted_results:
        name = r["name"]
        test_acc = f"{r.get('test_acc', 0)*100:.2f}%" if r.get("test_acc") else "N/A"
        test_auc = f"{r.get('test_auc', 0)*100:.2f}%" if r.get("test_auc") else "N/A"
        val_acc = f"{r.get('val_acc', 0)*100:.2f}%" if r.get("val_acc") else "N/A"
        val_auc = f"{r.get('val_auc', 0)*100:.2f}%" if r.get("val_auc") else "N/A"
        size = f"{r.get('size_kb', 0):.1f}" if r.get("size_kb") else "N/A"
        ms = f"{r.get('avg_ms', 0):.1f}" if r.get("avg_ms") else "N/A"
        flex = r.get("needs_flex", "?")

        print(f"{name:<28} {test_acc:>9} {test_auc:>9} {val_acc:>9} {val_auc:>9} {size:>9} {ms:>10} {flex:>6}")

    print("=" * 110)

    # Best model
    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST MODEL (by Test AUC): {best['name']}")
        print(f"  Test Accuracy: {best.get('test_acc', 0)*100:.2f}%")
        print(f"  Test AUC:      {best.get('test_auc', 0)*100:.2f}%")
        print(f"  Model Size:    {best.get('size_kb', 0):.1f} KB")


def main():
    skip_training = "--skip-training" in sys.argv

    if not skip_training:
        print("=" * 70)
        print("  TRAINING ALL MODELS")
        print("=" * 70)
        training_results = []
        for name, script in TRAINING_SCRIPTS:
            if os.path.exists(script):
                success, elapsed = run_training(script, name)
                training_results.append((name, success, elapsed))
            else:
                print(f"\n  SKIP: {script} not found")
                training_results.append((name, False, 0))

        print("\n\n" + "=" * 70)
        print("  TRAINING SUMMARY")
        print("=" * 70)
        for name, success, elapsed in training_results:
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {name:<30} {elapsed:.1f}s")

    # ============================================================
    # Evaluate all TFLite models
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  EVALUATING ALL TFLITE MODELS")
    print("=" * 70)

    print("\nLoading audio data...")
    splits = load_audio_data()
    a_val, y_val = splits["val"]
    a_test, y_test = splits["test"]

    results = []
    for name, path in TFLITE_MODELS.items():
        if not os.path.exists(path):
            print(f"\n  SKIP: {name} — {path} not found")
            continue

        print(f"\n  Evaluating: {name}...")
        size_kb = get_model_size_kb(path)

        # Check if needs flex delegate
        try:
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            needs_flex = "No"
        except Exception:
            needs_flex = "Yes"

        try:
            val_metrics = evaluate_tflite(path, a_val, y_val)
            test_metrics = evaluate_tflite(path, a_test, y_test)

            results.append({
                "name": name,
                "test_acc": test_metrics["accuracy"],
                "test_auc": test_metrics["auc"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "size_kb": size_kb,
                "avg_ms": test_metrics["avg_inference_ms"],
                "needs_flex": needs_flex,
            })
            print(f"    Test: Acc={test_metrics['accuracy']*100:.2f}%, AUC={test_metrics['auc']*100:.2f}%")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "name": name,
                "test_acc": 0, "test_auc": 0,
                "val_acc": 0, "val_auc": 0,
                "size_kb": size_kb, "avg_ms": 0,
                "needs_flex": needs_flex,
            })

    print_comparison_table(results)

    # Save results to JSON
    results_path = os.path.join(MODEL_DIR, "model_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
