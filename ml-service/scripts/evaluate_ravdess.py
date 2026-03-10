"""
Evaluate TFLite models on the RAVDESS dataset for cross-dataset validation.

RAVDESS filename convention: XX-XX-EMOTION-XX-XX-XX-ACTOR.wav
    Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

Mapping:
    Sad (04)     -> Depression (1)
    Neutral (01) -> Normal (0)

Usage:
    python scripts/evaluate_ravdess.py
    python scripts/evaluate_ravdess.py --ravdess-dir /path/to/ravdess
    python scripts/evaluate_ravdess.py --model mel_cnn bilstm
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
import librosa
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.architectures import MODEL_REGISTRY
from src.config import SAMPLE_RATE, AUDIO_LENGTH, MODEL_DIR

# Emotion mapping
DEPRESSION_EMOTIONS = {"04"}  # sad
NORMAL_EMOTIONS = {"01"}       # neutral
EMOTION_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


def parse_ravdess_filename(filepath):
    """Parse RAVDESS filename and return emotion code, or None."""
    parts = Path(filepath).stem.split("-")
    if len(parts) != 7:
        return None
    return {"emotion": parts[2], "actor": parts[6], "intensity": parts[3]}


def load_ravdess_data(ravdess_dir):
    """Load RAVDESS audio files mapped to depression/normal labels."""
    audio_list, labels, file_info = [], [], []

    for actor_dir in sorted(Path(ravdess_dir).glob("Actor_*")):
        for wav_file in sorted(actor_dir.glob("*.wav")):
            info = parse_ravdess_filename(wav_file)
            if info is None:
                continue

            emotion = info["emotion"]
            if emotion in DEPRESSION_EMOTIONS:
                label = 1
            elif emotion in NORMAL_EMOTIONS:
                label = 0
            else:
                continue

            try:
                audio, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                if len(audio) < AUDIO_LENGTH:
                    audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
                else:
                    audio = audio[:AUDIO_LENGTH]

                audio_list.append(audio.astype(np.float32))
                labels.append(label)
                file_info.append({
                    "file": wav_file.name,
                    "actor": info["actor"],
                    "emotion": EMOTION_NAMES.get(emotion, emotion),
                    "intensity": "normal" if info["intensity"] == "01" else "strong",
                })
            except Exception as e:
                print(f"  Failed to load {wav_file}: {e}")

    return audio_list, np.array(labels), file_info


def evaluate_model(tflite_path, audio_list, labels):
    """Evaluate a TFLite model and return metrics dict."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    preds = []
    t0 = time.time()
    for audio in audio_list:
        interpreter.set_tensor(inp[0]["index"], audio[np.newaxis])
        interpreter.invoke()
        preds.append(interpreter.get_tensor(out[0]["index"])[0][0])
    elapsed = time.time() - t0

    preds = np.array(preds)
    y_pred = (preds >= 0.5).astype(int)
    cm = confusion_matrix(labels, y_pred)

    return {
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred, zero_division=0),
        "recall": recall_score(labels, y_pred, zero_division=0),
        "f1": f1_score(labels, y_pred, zero_division=0),
        "auc": roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
        "cm": cm,
        "preds": preds,
        "y_pred": y_pred,
        "ms_per_sample": elapsed / len(audio_list) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on RAVDESS dataset.")
    parser.add_argument(
        "--ravdess-dir", type=str,
        required=True,
        help="Path to RAVDESS dataset directory.",
    )
    parser.add_argument("--model", nargs="*", help="Architecture names to evaluate.")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAVDESS Dataset Evaluation — Depression Detection")
    print("=" * 60)
    print(f"\n  Sad (04)     -> Depression (1)")
    print(f"  Neutral (01) -> Normal (0)")

    print("\nLoading RAVDESS audio files...")
    audio_list, labels, file_info = load_ravdess_data(args.ravdess_dir)
    n_dep = np.sum(labels == 1)
    n_norm = np.sum(labels == 0)
    print(f"Loaded {len(audio_list)} samples (depression/sad={n_dep}, normal/neutral={n_norm})")

    if len(audio_list) == 0:
        print("Error: No audio files found. Check --ravdess-dir path.")
        sys.exit(1)

    # Determine which models to evaluate
    model_names = args.model if args.model else list(MODEL_REGISTRY.keys())

    all_results = []
    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"\nSKIP: Unknown architecture '{name}'")
            continue
        tflite_path = os.path.join(MODEL_DIR, MODEL_REGISTRY[name]["tflite_name"])
        if not os.path.exists(tflite_path):
            print(f"\nSKIP: {name} — {tflite_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {name}")
        print(f"{'=' * 60}")
        result = evaluate_model(tflite_path, audio_list, labels)
        result["name"] = name
        all_results.append(result)

        cm = result["cm"]
        print(f"  Accuracy:  {result['accuracy'] * 100:.2f}%")
        print(f"  Precision: {result['precision'] * 100:.2f}%")
        print(f"  Recall:    {result['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {result['f1'] * 100:.2f}%")
        print(f"  AUC-ROC:   {result['auc'] * 100:.2f}%")
        print(f"  TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # Summary table
    if all_results:
        print(f"\n\n{'=' * 90}")
        print(f"  RAVDESS COMPARISON TABLE")
        print(f"{'=' * 90}")
        header = f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'ms':>7}"
        print(header)
        print("-" * 90)
        sorted_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
        for r in sorted_results:
            print(
                f"{r['name']:<28} "
                f"{r['accuracy'] * 100:>6.2f}% "
                f"{r['precision'] * 100:>6.2f}% "
                f"{r['recall'] * 100:>6.2f}% "
                f"{r['f1'] * 100:>6.2f}% "
                f"{r['auc'] * 100:>6.2f}% "
                f"{r['ms_per_sample']:>6.1f}"
            )
        print("=" * 90)

        best = sorted_results[0]
        print(f"\n  BEST MODEL (by Accuracy): {best['name']}")
        print(f"  Accuracy: {best['accuracy'] * 100:.2f}%  |  AUC: {best['auc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
