import tensorflow as tf
import numpy as np
import librosa
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='artifacts/models/depression_detection_20260219_195034.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== INPUT DETAILS ===")
for d in input_details:
    print(f"  Name: {d['name']}")
    print(f"  Shape: {d['shape']}")
    print(f"  Dtype: {d['dtype']}")

print("=== OUTPUT DETAILS ===")
for d in output_details:
    print(f"  Name: {d['name']}")
    print(f"  Shape: {d['shape']}")
    print(f"  Dtype: {d['dtype']}")

# Test with zeros
input_shape = input_details[0]['shape']
test_input = np.zeros(input_shape, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"\n=== Zero input => Output: {output} ===")

# Test with random normal features (simulating normalized MFCCs)
np.random.seed(42)
test_input = np.random.randn(*input_shape).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"=== Random normal input => Output: {output} ===")

# Test with large positive values (out of distribution)
test_input = np.ones(input_shape, dtype=np.float32) * 10.0
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"=== Large positive input => Output: {output} ===")

# Test with actual depression sample
dep_dir = 'data/raw/voice_data/depression1'
files = [f for f in os.listdir(dep_dir) if f.endswith('.wav')][:1]
if files:
    audio, sr = librosa.load(os.path.join(dep_dir, files[0]), sr=16000, duration=5.0)
    print(f"\nDep audio: len={len(audio)}, range=[{audio.min():.4f}, {audio.max():.4f}]")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    print(f"Raw MFCC shape: {mfccs.shape}, range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    # Normalize per-coefficient
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True) + 1e-8
    mfccs = (mfccs - mean) / std
    print(f"Norm MFCC range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    # Pad to 313
    if mfccs.shape[1] < 313:
        mfccs = np.pad(mfccs, ((0,0),(0,313-mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :313]
    mfccs = mfccs[np.newaxis, ..., np.newaxis].astype(np.float32)
    print(f"Final shape: {mfccs.shape}")
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"=== Depression sample => Output: {output} ===")

# Test with actual normal sample
norm_dir = 'data/raw/voice_data/normal1'
files = [f for f in os.listdir(norm_dir) if f.endswith('.wav')][:1]
if files:
    audio, sr = librosa.load(os.path.join(norm_dir, files[0]), sr=16000, duration=5.0)
    print(f"\nNormal audio: len={len(audio)}, range=[{audio.min():.4f}, {audio.max():.4f}]")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    print(f"Raw MFCC shape: {mfccs.shape}, range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True) + 1e-8
    mfccs = (mfccs - mean) / std
    print(f"Norm MFCC range: [{mfccs.min():.3f}, {mfccs.max():.3f}]")
    if mfccs.shape[1] < 313:
        mfccs = np.pad(mfccs, ((0,0),(0,313-mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :313]
    mfccs = mfccs[np.newaxis, ..., np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"=== Normal sample => Output: {output} ===")
