"""
Evaluate the combined TFLite model on the RAVDESS dataset.

RAVDESS filename convention: XX-XX-EMOTION-XX-XX-XX-ACTOR.wav
  Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

Mapping for depression detection:
  - Sad (04)     → Depression (1)   -- closest match to depressive speech
  - Neutral (01) → Normal (0)       -- baseline normal speech
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from pathlib import Path
import librosa
import time

RAVDESS_DIR = Path(r"d:\MINOR\Voice data and codes\The Ryerson Audio-Visual Dataset\The Ryerson Audio-Visual Dataset")
SAMPLE_RATE = 16000
AUDIO_LEN = 80000  # 5 seconds

ALL_MODELS = {
    "MFCC CNN (baseline)":    "artifacts/models/depression_detection_combined.tflite",
    "Mel CNN (4-block)":      "artifacts/models/mel_depression_combined.tflite",
    "BiLSTM":                 "artifacts/models/lstm_depression_combined.tflite",
    "CNN-LSTM Hybrid":        "artifacts/models/cnn_lstm_depression_combined.tflite",
    "Multi-Feature Fusion":   "artifacts/models/multi_feature_depression_combined.tflite",
    "CNN + Attention":        "artifacts/models/attention_depression_combined.tflite",
    "Separable CNN":          "artifacts/models/separable_cnn_depression_combined.tflite",
}

# Emotion code mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Which emotions to use for testing
DEPRESSION_EMOTIONS = ["04"]  # sad → depression
NORMAL_EMOTIONS = ["01"]      # neutral → normal


def parse_ravdess_filename(filename):
    """Parse RAVDESS filename and return emotion code."""
    parts = filename.stem.split("-")
    if len(parts) != 7:
        return None
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def load_ravdess_data():
    """Load RAVDESS audio files with mapped labels."""
    audio_list = []
    labels = []
    file_info = []

    for actor_dir in sorted(RAVDESS_DIR.glob("Actor_*")):
        for wav_file in sorted(actor_dir.glob("*.wav")):
            info = parse_ravdess_filename(wav_file)
            if info is None:
                continue

            emotion = info["emotion"]

            if emotion in DEPRESSION_EMOTIONS:
                label = 1  # depression
            elif emotion in NORMAL_EMOTIONS:
                label = 0  # normal
            else:
                continue  # skip other emotions

            try:
                audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
                # Normalize
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                # Pad/truncate
                if len(audio) < AUDIO_LEN:
                    audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
                else:
                    audio = audio[:AUDIO_LEN]

                audio_list.append(audio.astype(np.float32))
                labels.append(label)
                file_info.append({
                    "file": wav_file.name,
                    "actor": info["actor"],
                    "emotion": EMOTION_MAP[emotion],
                    "intensity": "normal" if info["intensity"] == "01" else "strong",
                })
            except Exception as e:
                print(f"  Failed to load {wav_file}: {e}")

    return audio_list, np.array(labels), file_info


def evaluate_model(model_name, model_path, audio_list, labels, file_info):
    """Evaluate a single TFLite model and return metrics dict."""
    import os
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_name} — {model_path} not found")
        return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    start = time.time()
    for audio in audio_list:
        interpreter.set_tensor(input_details[0]['index'], audio[np.newaxis])
        interpreter.invoke()
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        preds.append(prob)
    elapsed = time.time() - start

    preds = np.array(preds)
    y_pred = (preds >= 0.5).astype(int)

    acc = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred, zero_division=0)
    rec = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    cm = confusion_matrix(labels, y_pred)

    return {
        "name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "preds": preds,
        "y_pred": y_pred,
        "elapsed": elapsed,
        "ms_per_sample": elapsed / len(audio_list) * 1000,
    }


def main():
    print("=" * 60)
    print("  RAVDESS Dataset Evaluation — All Models")
    print("=" * 60)
    print(f"\nLabel mapping:")
    print(f"  Sad (emotion=04)     → Depression (1)")
    print(f"  Neutral (emotion=01) → Normal (0)")

    # Load data
    print("\nLoading RAVDESS audio files...")
    audio_list, labels, file_info = load_ravdess_data()
    n_dep = np.sum(labels == 1)
    n_norm = np.sum(labels == 0)
    print(f"Loaded {len(audio_list)} samples (depression/sad={n_dep}, normal/neutral={n_norm})")

    # Evaluate all models
    all_results = []
    for model_name, model_path in ALL_MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")
        result = evaluate_model(model_name, model_path, audio_list, labels, file_info)
        if result is not None:
            all_results.append(result)
            cm = result["cm"]
            print(f"  Accuracy:  {result['accuracy']*100:.2f}%")
            print(f"  Precision: {result['precision']*100:.2f}%")
            print(f"  Recall:    {result['recall']*100:.2f}%")
            print(f"  F1 Score:  {result['f1']*100:.2f}%")
            print(f"  AUC-ROC:   {result['auc']*100:.2f}%")
            print(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # Summary comparison table
    print(f"\n\n{'='*90}")
    print(f"  RAVDESS COMPARISON TABLE — All Models (Sad vs Neutral)")
    print(f"{'='*90}")
    header = f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} {'ms':>7}"
    print(header)
    print("-" * 90)
    sorted_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
    for r in sorted_results:
        print(f"{r['name']:<28} {r['accuracy']*100:>6.2f}% {r['precision']*100:>6.2f}% {r['recall']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['auc']*100:>6.2f}% {r['ms_per_sample']:>6.1f}")
    print("=" * 90)

    # Detailed per-model output for best model
    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST MODEL (by Accuracy): {best['name']}")
        print(f"  Accuracy: {best['accuracy']*100:.2f}%  |  AUC: {best['auc']*100:.2f}%")

        # Per-actor breakdown for best model
        print(f"\n{'='*60}")
        print(f"  PER-ACTOR BREAKDOWN — {best['name']}")
        print(f"{'='*60}")
        actors = sorted(set(info["actor"] for info in file_info))
        print(f"{'Actor':>8} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Prob':>10}")
        print("-" * 50)
        for actor in actors:
            idxs = [i for i, info in enumerate(file_info) if info["actor"] == actor]
            actor_labels = labels[idxs]
            actor_preds_bin = best["y_pred"][idxs]
            actor_probs = best["preds"][idxs]
            correct = np.sum(actor_labels == actor_preds_bin)
            actor_acc = correct / len(idxs)
            avg_prob = np.mean(actor_probs)
            print(f"  {actor:>6} {len(idxs):>8d} {correct:>8d} {actor_acc*100:>9.1f}% {avg_prob:>9.3f}")

        # Misclassified samples for best model
        misclassified = np.where(labels != best["y_pred"])[0]
        if len(misclassified) > 0:
            print(f"\n{'='*60}")
            print(f"  MISCLASSIFIED SAMPLES — {best['name']} ({len(misclassified)} total)")
            print(f"{'='*60}")
            for idx in misclassified[:20]:
                info = file_info[idx]
                true_label = "Dep/Sad" if labels[idx] == 1 else "Normal"
                pred_label = "Dep/Sad" if best["y_pred"][idx] == 1 else "Normal"
                print(f"  {info['file']}: true={true_label}, pred={pred_label}, prob={best['preds'][idx]:.4f} (Actor {info['actor']}, {info['intensity']})")


if __name__ == "__main__":
    main()
