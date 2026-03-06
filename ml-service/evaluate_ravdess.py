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
MODEL_PATH = "artifacts/models/mel_depression_combined.tflite"
SAMPLE_RATE = 16000
AUDIO_LEN = 80000  # 5 seconds

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


def main():
    print("=" * 60)
    print("  RAVDESS Dataset Evaluation")
    print("  Model: Combined TFLite (depression_detection_combined)")
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

    # Load TFLite model
    print("\nLoading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    print("\nRunning inference...")
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

    # Metrics
    acc = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred, zero_division=0)
    rec = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    auc = roc_auc_score(labels, preds)
    cm = confusion_matrix(labels, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — RAVDESS (Sad vs Neutral)")
    print(f"{'=' * 60}")
    print(f"  Samples:   {len(audio_list)} (dep={n_dep}, norm={n_norm})")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc*100:.2f}%")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(audio_list)*1000:.1f}ms/sample)")
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred Normal  Pred Dep")
    print(f"  True Normal       {cm[0][0]:>5d}      {cm[0][1]:>5d}")
    print(f"  True Dep(Sad)     {cm[1][0]:>5d}      {cm[1][1]:>5d}")
    print(f"\n{classification_report(labels, y_pred, target_names=['Normal/Neutral', 'Depression/Sad'])}")

    # Per-actor breakdown
    print(f"\n{'=' * 60}")
    print(f"  PER-ACTOR BREAKDOWN")
    print(f"{'=' * 60}")
    actors = sorted(set(info["actor"] for info in file_info))
    print(f"{'Actor':>8} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Prob':>10}")
    print("-" * 50)
    for actor in actors:
        idxs = [i for i, info in enumerate(file_info) if info["actor"] == actor]
        actor_labels = labels[idxs]
        actor_preds_bin = y_pred[idxs]
        actor_probs = preds[idxs]
        correct = np.sum(actor_labels == actor_preds_bin)
        actor_acc = correct / len(idxs)
        avg_prob = np.mean(actor_probs)
        print(f"  {actor:>6} {len(idxs):>8d} {correct:>8d} {actor_acc*100:>9.1f}% {avg_prob:>9.3f}")

    # Show some misclassified samples
    misclassified = np.where(labels != y_pred)[0]
    if len(misclassified) > 0:
        print(f"\n{'=' * 60}")
        print(f"  MISCLASSIFIED SAMPLES ({len(misclassified)} total)")
        print(f"{'=' * 60}")
        for idx in misclassified[:20]:
            info = file_info[idx]
            true_label = "Dep/Sad" if labels[idx] == 1 else "Normal"
            pred_label = "Dep/Sad" if y_pred[idx] == 1 else "Normal"
            print(f"  {info['file']}: true={true_label}, pred={pred_label}, prob={preds[idx]:.4f} (Actor {info['actor']}, {info['intensity']})")


if __name__ == "__main__":
    main()
