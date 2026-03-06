"""
Evaluate the combined TFLite model on train/val/test splits.
Replicates the exact data loading and splitting from the training pipeline.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from src.data.loader import AudioDataLoader
from src.data.splitter import split_dataset
import logging, time

logging.basicConfig(level=logging.INFO)

# --- 1. Load data exactly like training pipeline ---
print("=" * 60)
print("Loading data (same as training pipeline)...")
print("=" * 60)

loader = AudioDataLoader(
    data_dir="data/raw/voice_data",
    sample_rate=16000,
    duration=5.0,
    mono=True,
)
audio_list, labels, file_paths = loader.load_dataset(
    depression_dir="depression1",
    normal_dir="normal1",
    extensions=[".wav", ".mp3", ".flac"],
)
labels = np.array(labels)
print(f"Total samples: {len(audio_list)} (dep={np.sum(labels==1)}, norm={np.sum(labels==0)})")

# --- 2. Split exactly like training (random_state=42, 70/15/15) ---
# The splitter operates on feature arrays, so stack audio into array first
X_audio = np.array(audio_list)  # shape: (N, 80000)

splits = split_dataset(
    X_audio, labels,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- 3. Load TFLite model ---
print("\nLoading TFLite model...")
MODEL_PATH = "artifacts/models/depression_detection_combined.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

AUDIO_LEN = 80000

def predict_batch(X, y, split_name):
    """Run TFLite inference on a split and report metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {split_name} SET ({len(X)} samples)")
    print(f"{'=' * 60}")
    
    preds = []
    start = time.time()
    for i in range(len(X)):
        audio = X[i].astype(np.float32)
        # Normalize to [-1, 1]
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx
        # Pad/truncate
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
        else:
            audio = audio[:AUDIO_LEN]
        
        interpreter.set_tensor(input_details[0]['index'], audio[np.newaxis])
        interpreter.invoke()
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        preds.append(prob)
    
    elapsed = time.time() - start
    preds = np.array(preds)
    y_pred = (preds >= 0.5).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, preds)
    cm = confusion_matrix(y, y_pred)
    
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc*100:.2f}%")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(X)*1000:.1f}ms/sample)")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred Normal  Pred Dep")
    print(f"  True Normal    {cm[0][0]:>5d}      {cm[0][1]:>5d}")
    print(f"  True Dep       {cm[1][0]:>5d}      {cm[1][1]:>5d}")
    print(f"\n{classification_report(y, y_pred, target_names=['Normal', 'Depression'])}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

# --- 4. Evaluate all splits ---
train_metrics = predict_batch(X_train, y_train, "TRAIN")
val_metrics = predict_batch(X_val, y_val, "VALIDATION")
test_metrics = predict_batch(X_test, y_test, "TEST")

# --- 5. Summary ---
print("\n" + "=" * 60)
print("  SUMMARY — Combined TFLite Model")
print("=" * 60)
header = f"{'Metric':<12} {'Train':>10} {'Val':>10} {'Test':>10}"
print(header)
print("-" * len(header))
for m in ["accuracy", "precision", "recall", "f1", "auc"]:
    print(f"{m:<12} {train_metrics[m]*100:>9.2f}% {val_metrics[m]*100:>9.2f}% {test_metrics[m]*100:>9.2f}%")
