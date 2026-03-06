# 📱 Voice-Based Depression Detection Mobile App

## 🎯 Project Goal

Create a **local, privacy-first mobile app** that detects depression using voice analysis, running entirely on-device without requiring cloud connectivity.

## 📋 Training Strategy for Mobile Deployment

### Phase 1: Data Preparation

#### 1.1 Dataset Organization
```
voice_training_data/
├── train/
│   ├── depressed/     # Depression voice samples
│   └── normal/        # Normal voice samples
├── validation/
│   ├── depressed/
│   └── normal/
└── test/
    ├── depressed/
    └── normal/
```

**Action Items:**
- Combine voice data from:
  - `EEG_data sets/data_set_from kaggle_voice/`
  - `EEG_data sets/The Ryerson Audio-Visual Dataset/`
- Split: 70% train, 15% validation, 15% test
- Ensure balanced classes (equal depressed/normal samples)

#### 1.2 Audio Preprocessing for Mobile
```python
# Mobile-optimized settings
TARGET_SR = 16000      # Lower sample rate for mobile
TARGET_DURATION = 5.0  # 5 seconds = good accuracy + mobile-friendly
N_MFCC = 13           # Fewer coefficients = smaller model
```

**Key Considerations:**
- Optimal audio clips (5 seconds) for balance of accuracy and speed
- Lower sample rate (16kHz instead of 44.1kHz)
- Mono audio only (saves memory)
- Normalize all audio to consistent volume

---

## 🧠 Model Architecture Options

### Option 1: Lightweight CNN (RECOMMENDED for Mobile)
**Best for:** Real-time inference, low latency  
**Model Size:** 1-5 MB  
**Inference Time:** 50-200ms on mobile

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_mobile_cnn(input_shape=(13, 215, 1)):
    """
    Lightweight CNN optimized for mobile deployment
    Input: MFCC features (n_mfcc=13, time_steps≈215 for 5s audio)
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', 
                     padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # Better than Flatten for mobile
        
        # Classifier
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')  # Binary: depressed vs normal
    ])
    
    return model

# Compile with mobile-friendly settings
model = create_mobile_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
```

**Why this architecture?**
- Small number of parameters (~100K-300K)
- No recurrent layers (faster on mobile CPUs)
- Global Average Pooling reduces overfitting
- Batch normalization speeds up convergence

---

### Option 2: 1D CNN (Alternative)
**Best for:** Simpler features, even smaller model  
**Model Size:** 0.5-2 MB

```python
def create_1d_cnn(input_shape=(13, 215)):
    """
    1D CNN for temporal audio features
    Works directly on MFCC time series
    """
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model
```

---

### Option 3: MobileNet-Inspired (Advanced)
**Best for:** Transfer learning, better accuracy  
**Model Size:** 3-10 MB

```python
def depthwise_separable_conv(x, filters, kernel_size):
    """Mobile-optimized convolution"""
    x = layers.SeparableConv2D(filters, kernel_size, 
                               padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def create_mobilenet_audio(input_shape=(13, 215, 1)):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = depthwise_separable_conv(x, 64, (3, 3))
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = depthwise_separable_conv(x, 128, (3, 3))
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)
```

---

## 🔧 Complete Training Pipeline

### Step 1: Feature Extraction Script
```python
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    'sample_rate': 16000,
    'duration': 5.0,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 256
}

def extract_features(audio_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=CONFIG['sample_rate'], 
                            duration=CONFIG['duration'])
        
        # Normalize
        y = librosa.util.normalize(y)
        
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Pad/truncate to fixed length
        target_length = int(CONFIG['sample_rate'] * CONFIG['duration'])
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr,
            n_mfcc=CONFIG['n_mfcc'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length']
        )
        
        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        return mfccs
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_dataset(data_dir):
    """Load and prepare dataset"""
    X = []
    y = []
    
    # Load depressed samples
    depressed_dir = os.path.join(data_dir, 'depressed')
    for file in os.listdir(depressed_dir):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(depressed_dir, file))
            if features is not None:
                X.append(features)
                y.append(1)  # 1 = depressed
    
    # Load normal samples
    normal_dir = os.path.join(data_dir, 'normal')
    for file in os.listdir(normal_dir):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(normal_dir, file))
            if features is not None:
                X.append(features)
                y.append(0)  # 0 = normal
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN (add channel dimension)
    X = X[..., np.newaxis]  # Shape: (samples, n_mfcc, time_steps, 1)
    
    return X, y

# Load data
X, y = load_dataset('path/to/voice_data')

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
```

---

### Step 2: Training Script
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Create model
model = create_mobile_cnn(input_shape=X_train.shape[1:])

# Callbacks for better training
callbacks = [
    # Stop if no improvement
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Data augmentation (optional but recommended)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    width_shift_range=0.1,   # Time shift
    fill_mode='nearest'
)

# Train model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
```

---

### Step 3: Model Optimization for Mobile

#### 3.1 Quantization (Reduce model size by 4x)
```python
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Option A: Dynamic range quantization (easiest, 4x smaller)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Option B: Full integer quantization (8x smaller, slower)
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model_int8 = converter.convert()

# Save TFLite model
with open('depression_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Original model size: {os.path.getsize('best_model.h5') / 1024:.2f} KB")
print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
```

#### 3.2 Pruning (Remove unnecessary weights)
```python
import tensorflow_model_optimization as tfmot

# Prune model during training
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # Remove 50% of weights
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train pruned model
model_for_pruning.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks
)

# Remove pruning wrappers and convert
model_pruned = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

---

## 📲 Mobile Deployment Options

### Option 1: TensorFlow Lite (Android/iOS)
**Best for:** Cross-platform, Google ecosystem

**Android Integration:**
```java
// 1. Add to build.gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}

// 2. Load model
import org.tensorflow.lite.Interpreter;

Interpreter tflite = new Interpreter(loadModelFile());

// 3. Run inference
float[][] input = preprocessAudio(audioFile);  // Your preprocessing
float[][] output = new float[1][1];
tflite.run(input, output);
float probability = output[0][0];
```

**iOS Integration (Swift):**
```swift
import TensorFlowLite

// 1. Load model
guard let modelPath = Bundle.main.path(forResource: "depression_model", 
                                       ofType: "tflite") else { return }
let interpreter = try Interpreter(modelPath: modelPath)

// 2. Allocate tensors
try interpreter.allocateTensors()

// 3. Run inference
let inputData = preprocessAudio(audioFile)
try interpreter.copy(Data(bytes: inputData), toInputAt: 0)
try interpreter.invoke()
let outputTensor = try interpreter.output(at: 0)
```

---

### Option 2: Core ML (iOS only)
**Best for:** Native iOS performance

```python
# Convert to Core ML
import coremltools as ct

# Convert TF model
coreml_model = ct.convert(
    model,
    inputs=[ct.TensorType(name="mfcc_input", shape=X_train.shape[1:])],
    classifier_config=ct.ClassifierConfig(['normal', 'depressed'])
)

# Save
coreml_model.save('DepressionDetector.mlmodel')
```

**iOS Usage:**
```swift
import CoreML

let model = try DepressionDetector()
let prediction = try model.prediction(mfcc_input: mfccArray)
print("Prediction: \(prediction.classLabel)")
print("Confidence: \(prediction.classLabelProbs)")
```

---

### Option 3: ONNX (Cross-platform)
**Best for:** Maximum flexibility

```python
import tf2onnx

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(model)

with open("depression_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

---

## 🎯 Implementation Roadmap

### Week 1-2: Data Preparation
- [ ] Organize voice datasets from Kaggle and RAVDESS
- [ ] Clean and standardize audio files
- [ ] Implement audio preprocessing pipeline
- [ ] Create train/val/test splits
- [ ] Extract and save MFCC features

### Week 3: Model Development
- [ ] Implement lightweight CNN architecture
- [ ] Set up training pipeline with callbacks
- [ ] Train baseline model
- [ ] Evaluate performance (target: >75% accuracy)
- [ ] Fine-tune hyperparameters

### Week 4: Optimization
- [ ] Apply quantization (TFLite conversion)
- [ ] Test pruning if model too large
- [ ] Benchmark inference speed on mobile device
- [ ] Optimize for <100MB app size

### Week 5-6: Mobile Integration
- [ ] Choose platform (Android/iOS/both)
- [ ] Implement audio recording in app
- [ ] Integrate model inference
- [ ] Build UI for results display
- [ ] Add privacy features (local-only processing)

### Week 7: Testing & Refinement
- [ ] Test on real devices (various models)
- [ ] Measure battery impact
- [ ] Optimize memory usage
- [ ] Add error handling
- [ ] User experience testing

---

## 📊 Expected Performance Metrics

### Model Performance
- **Accuracy:** 75-88% (5-second voice clips, realistic for voice-only)
- **Model Size:** 2-4 MB (after quantization)
- **Inference Time:** 100-200ms on mobile
- **Battery Impact:** <5% per hour of use

### Mobile Requirements
- **Minimum RAM:** 2GB
- **Storage:** <50MB app size
- **Android:** API Level 21+ (Android 5.0+)
- **iOS:** iOS 12.0+
- **Permissions:** Microphone only

---

## ⚠️ Important Considerations

### 1. Privacy & Security
✅ **All processing happens on-device**  
✅ No audio data sent to servers  
✅ No user data collection  
✅ Clear privacy policy in app  

### 2. Medical Disclaimer
**CRITICAL:** Include clear disclaimers:
- "This app is NOT a medical diagnosis tool"
- "Results are for informational purposes only"
- "Consult healthcare professionals for actual diagnosis"
- "Do not use for self-diagnosis or treatment decisions"

### 3. Accuracy Limitations
- Voice-only detection is less accurate than multimodal
- Background noise affects performance
- Accent/language variations may impact results
- Should be used as screening tool, not diagnostic

### 4. Ethical Considerations
- Avoid stigmatizing language in UI
- Provide mental health resources in app
- Include crisis hotline numbers
- Design with compassion and empathy

---

## 🛠️ Recommended Tech Stack

### Backend (Model Training)
- **Python 3.8+**
- **TensorFlow 2.13+**
- **librosa** for audio processing
- **scikit-learn** for data splitting
- **numpy, pandas** for data handling

### Mobile Development

**Android:**
- Kotlin/Java
- TensorFlow Lite Android Support Library
- Android AudioRecord API
- Material Design UI

**iOS:**
- Swift/SwiftUI
- Core ML or TensorFlow Lite
- AVFoundation for audio
- Native iOS UI components

**Cross-Platform (Alternative):**
- Flutter + TFLite plugin
- React Native + TensorFlow.js

---

## 📝 Next Steps - Action Plan

1. **Start Here:** Run the audio processing notebook to understand current pipeline
2. **Gather Data:** Combine voice datasets into structured folders
3. **Build Pipeline:** Implement the feature extraction script above
4. **Train Model:** Start with lightweight CNN (Option 1)
5. **Convert:** Create TFLite model for mobile
6. **Test:** Deploy to test device and measure performance
7. **Iterate:** Refine based on real-world testing

---

## 📚 Additional Resources

### Learning Materials
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [librosa Tutorial](https://librosa.org/doc/main/tutorial.html)
- [Audio Classification with TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio)

### Datasets to Consider
- DAIC-WOZ (Depression Interview Dataset)
- AVEC Challenges (Audio-Visual Emotion Challenge)
- Your current Kaggle voice dataset
- RAVDESS (already have)

### Tools
- **Netron:** Visualize model architecture
- **TensorBoard:** Track training metrics
- **Android Studio:** Android development
- **Xcode:** iOS development

---

**Good luck with your mobile app! Focus on creating a privacy-first, ethical tool that helps people while being transparent about limitations.** 🚀
