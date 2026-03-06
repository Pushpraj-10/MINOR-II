"""
Test TFLite model with proper preprocessing matching the training pipeline.
Also compare librosa MFCCs with what the Dart implementation would produce.
"""
import tensorflow as tf
import numpy as np
import librosa
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='artifacts/models/depression_detection_20260219_195034.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_and_predict(audio_path, label_name):
    """Exactly replicate training pipeline preprocessing."""
    # Step 1: Load audio at 16kHz (same as training)
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"\n--- {label_name}: {os.path.basename(audio_path)} ---")
    print(f"  Raw audio: {len(audio)} samples ({len(audio)/sr:.2f}s), range=[{audio.min():.4f}, {audio.max():.4f}]")
    
    # Step 2: Normalize (same as loader.py)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Step 3: Pad/truncate to 5 seconds = 80000 samples (same as loader.py)
    target_length = 80000
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    print(f"  After pad/trunc: {len(audio)} samples, range=[{audio.min():.4f}, {audio.max():.4f}]")
    
    # Step 4: Extract MFCCs (same as audio_processing.py)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    print(f"  MFCC shape: {mfccs.shape}")
    print(f"  MFCC raw range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    
    # Step 5: Per-coefficient normalization
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True) + 1e-8
    mfccs = (mfccs - mean) / std
    print(f"  MFCC norm range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    print(f"  MFCC means per coeff: {np.mean(mfccs, axis=1)[:5].round(4)}")
    
    # Step 6: Pad time steps if needed (should be 313 already for 80000 samples)
    if mfccs.shape[1] < 313:
        mfccs = np.pad(mfccs, ((0,0),(0,313-mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :313]
    
    # Step 7: Reshape for model [1, 13, 313, 1]
    mfccs = mfccs[np.newaxis, ..., np.newaxis].astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prob = output[0][0]
    print(f"  >>> Prediction: {prob:.6f} ({'DEPRESSION' if prob >= 0.5 else 'NORMAL'})")
    return prob

# Test with depression samples
dep_dir = 'data/raw/voice_data/depression1'
dep_files = sorted([f for f in os.listdir(dep_dir) if f.endswith('.wav')])[:5]
print("=" * 60)
print("DEPRESSION SAMPLES")
print("=" * 60)
dep_probs = []
for f in dep_files:
    p = preprocess_and_predict(os.path.join(dep_dir, f), "Depression")
    dep_probs.append(p)

# Test with normal samples
norm_dir = 'data/raw/voice_data/normal1'
norm_files = sorted([f for f in os.listdir(norm_dir) if f.endswith('.wav')])[:5]
print("\n" + "=" * 60)
print("NORMAL SAMPLES")
print("=" * 60)
norm_probs = []
for f in norm_files:
    p = preprocess_and_predict(os.path.join(norm_dir, f), "Normal")
    norm_probs.append(p)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Depression avg prob: {np.mean(dep_probs):.4f}")
print(f"Normal avg prob: {np.mean(norm_probs):.4f}")

# Now show what the librosa MFCCs look like (first 3 values of each coefficient)
print("\n" + "=" * 60)
print("LIBROSA MFCC SAMPLE VALUES (first depression file)")
print("=" * 60)
audio, sr = librosa.load(os.path.join(dep_dir, dep_files[0]), sr=16000)
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))
audio = np.pad(audio, (0, max(0, 80000 - len(audio))))[:80000]
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
print(f"Raw MFCCs [coeff 0, first 10 time steps]: {mfccs[0, :10].round(3)}")
print(f"Raw MFCCs [coeff 1, first 10 time steps]: {mfccs[1, :10].round(3)}")
# After normalization
mean = np.mean(mfccs, axis=1, keepdims=True)
std = np.std(mfccs, axis=1, keepdims=True) + 1e-8
mfccs_n = (mfccs - mean) / std
print(f"Norm MFCCs [coeff 0, first 10]: {mfccs_n[0, :10].round(4)}")
print(f"Norm MFCCs [coeff 1, first 10]: {mfccs_n[1, :10].round(4)}")
