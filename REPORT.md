# Depression Detection from Voice — Model Training & Evaluation Report

**Date:** March 9, 2026  
**Project:** Voice-Based Depression Detection (Minor Project)  
**Task:** Binary classification — Depression vs Normal from 5-second audio clips

---

## 1. Dataset Overview

| Property | Value |
| :--- | :--- |
| Total Samples | 800 (400 depression + 400 normal) |
| Audio Format | WAV, mono, 16 kHz sample rate |
| Duration | 5.0 seconds (80,000 samples per clip) |
| Preprocessing | Normalize amplitude, trim silence (top_db=20) |
| Train / Val / Test Split | 70% / 15% / 15% (560 / 120 / 120) |
| Stratification | Yes (equal class balance in all splits) |
| Random State | 42 |

### Feature Extraction (via `tf.signal` for TFLite compatibility)

| Feature | Parameters |
| :--- | :--- |
| **Mel Spectrogram** | n_fft=512, hop_length=256, n_mels=128, f_min=0, f_max=8000 |
| **MFCC** | n_mfcc=13, derived from Mel via DCT-II |
| **Time Steps** | 313 (from 80,000 samples with hop=256) |

---

## 2. Models Trained

Seven models were trained and evaluated. All models use the same dataset split and produce a combined TFLite model with audio preprocessing baked in (raw audio → prediction in a single forward pass).

### Model 1: MFCC CNN (Baseline)
- **Architecture:** 4-block CNN on MFCC features (13 × 313 × 1)
- **Parameters:** ~58K
- **Training:** Previously trained (epoch details from prior session)
- **TFLite:** 118.0 KB, quantized (int8), no flex delegate needed

### Model 2: Mel Spectrogram CNN (4-Block)
- **Architecture:** 4-block CNN on Mel spectrogram (128 × 313 × 1)
  - Conv2D(32) → BN → MaxPool → Conv2D(64) → BN → MaxPool → Conv2D(128) → BN → MaxPool → Conv2D(256) → BN → GAP → Dense(128) → Dense(1)
- **Parameters:** ~249K
- **Training:** Previously trained (100% on all splits from prior session)
- **TFLite:** 1124.6 KB, float32, no flex delegate needed

### Model 3: Bidirectional LSTM (BiLSTM)
- **Architecture:** BiLSTM on Mel spectrogram in sequence format (313 × 128)
  - BiLSTM(64, return_sequences=True) → BN → BiLSTM(32) → BN → Dense(64) → Dropout(0.4) → Dense(1, sigmoid)
- **Parameters:** 145,025 (566.5 KB)
- **Training:** 29 epochs (early stopping at patience=10, best epoch=19)
- **Learning Rate:** Started at 0.001, reduced to 0.000125
- **TFLite:** 741.0 KB, float32, **requires flex delegate** (TensorListReserve/SetItem/Stack ops)

### Model 4: CNN-LSTM Hybrid
- **Architecture:** 2 CNN blocks + LSTM on Mel spectrogram (128 × 313 × 1)
  - Conv2D(32) → BN → MaxPool(2,2) → Conv2D(64) → BN → MaxPool(2,2) → Permute → Reshape(78, 2048) → LSTM(64) → BN → Dense(32) → Dense(1, sigmoid)
- **Parameters:** 562,497 (2.15 MB)
- **Training:** 22 epochs (early stopping, best epoch=12)
- **Learning Rate:** 0.001
- **TFLite:** 2354.8 KB, float32, **requires flex delegate** (TensorListReserve/SetItem/Stack ops)

### Model 5: Multi-Feature Fusion CNN (Mel + MFCC)
- **Architecture:** Dual-branch CNN with feature fusion
  - **Branch A (Mel):** Conv2D(32) → BN → MaxPool → Conv2D(64) → BN → MaxPool → Conv2D(64) → BN → GAP → 64-dim
  - **Branch B (MFCC):** Conv2D(32) → BN → MaxPool → Conv2D(64) → BN → GAP → 64-dim
  - **Fusion:** Concatenate(128) → Dense(64) → Dropout(0.4) → Dense(1, sigmoid)
- **Parameters:** 83,905 (327.75 KB)
- **Training:** 28 epochs (early stopping, best epoch=17)
- **Learning Rate:** Started at 0.001, reduced to 0.00025
- **TFLite:** 486.5 KB, float32, **requires flex delegate** (StridedSlice op from MFCC extraction)

### Model 6: CNN + Multi-Head Self-Attention
- **Architecture:** 2 CNN blocks + Multi-Head Attention on Mel spectrogram (128 × 313 × 1)
  - Conv2D(32) → BN → MaxPool(2,2) → Conv2D(64) → BN → MaxPool(2,2) → Permute → Reshape(78, 2048) → Dense(128) → MultiHeadAttention(4 heads, key_dim=32) + Residual → LayerNorm → GlobalAvgPool1D → Dense(64) → Dense(1, sigmoid)
- **Parameters:** 356,097 (1.36 MB)
- **Training:** 15 epochs (early stopping at patience=12, best epoch=3)
- **Learning Rate:** 0.0005
- **TFLite:** 1556.3 KB, float32, **no flex delegate needed**

### Model 7: Depthwise Separable CNN (MobileNet-style)
- **Architecture:** Lightweight separable convolutions on Mel spectrogram (128 × 313 × 1)
  - Conv2D(16) → BN → 4× [SeparableConv2D(32→64→128→128) → BN → MaxPool] → GAP → Dense(64) → Dense(1, sigmoid)
- **Parameters:** 39,601 (154.69 KB) — **smallest model**
- **Training:** 26 epochs (early stopping, best epoch=15)
- **Learning Rate:** Started at 0.001, reduced to 0.0000625
- **TFLite:** 309.6 KB, float32, **no flex delegate needed**

---

## 3. Results Summary

### 3.1 Keras Model Evaluation (with correct decision threshold)

| # | Model | Train Acc | Train AUC | Val Acc | Val AUC | Test Acc | Test AUC | Params |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | MFCC CNN (baseline) | 99.29% | 99.61% | 95.83% | 99.31% | 96.67% | 99.61% | ~58K |
| 2 | Mel CNN (4-block) | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | ~249K |
| 3 | BiLSTM | 67.14% | 99.96% | 64.17% | 100.00% | 64.17% | 99.78% | 145K |
| 4 | CNN-LSTM Hybrid | 99.82% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 562K |
| 5 | Multi-Feature Fusion | 98.57% | 100.00% | 98.35% | 100.00% | 98.33% | 100.00% | 84K |
| 6 | CNN + Attention | 52.14% | 99.92% | 51.67% | 100.00% | 50.83% | 100.00% | 356K |
| 7 | Separable CNN | 50.00% | 100.00% | 50.00% | 100.00% | 50.00% | 100.00% | 40K |

### 3.2 TFLite Combined Model Evaluation (raw audio input → prediction)

| # | Model | Test Acc | Test AUC | Val Acc | Val AUC | Size (KB) | Inference (ms) | Flex Delegate |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | MFCC CNN (baseline) | 96.67% | 99.61% | 95.83% | 99.31% | 118.0 | 15.0 | No |
| 2 | Mel CNN (4-block) | 100.00% | 100.00% | 100.00% | 100.00% | 1124.6 | 25.7 | No |
| 3 | BiLSTM | 64.17% | 99.78% | 64.17% | 100.00% | 741.0 | 59.8 | **Yes** |
| 4 | **CNN-LSTM Hybrid** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 2354.8 | 29.8 | **Yes** |
| 5 | Multi-Feature Fusion | 98.33% | 100.00% | 98.33% | 100.00% | 486.5 | 27.0 | **Yes** |
| 6 | CNN + Attention | 50.83% | 100.00% | 51.67% | 100.00% | 1556.3 | 24.7 | No |
| 7 | Separable CNN | 50.00% | 100.00% | 50.00% | 100.00% | 309.6 | 43.0 | No |

---

## 4. Detailed Confusion Matrices (Test Set — 120 samples)

### Model 1: MFCC CNN (baseline)

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 56 (TN) | 4 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 93.75% | Recall: 100.00% | F1: 96.77%

### Model 2: Mel CNN (4-block)

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 60 (TN) | 0 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 100.00% | Recall: 100.00% | F1: 100.00%

### Model 3: BiLSTM

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 17 (TN) | 43 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 58.25% | Recall: 100.00% | F1: 73.62%

> **Issue:** High false positive rate — predicts most samples as depression. AUC is 99.78% indicating the ranking is correct but the 0.5 threshold is suboptimal.

### Model 4: CNN-LSTM Hybrid

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 60 (TN) | 0 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 100.00% | Recall: 100.00% | F1: 100.00%

### Model 5: Multi-Feature Fusion (Mel + MFCC)

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 58 (TN) | 2 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 96.77% | Recall: 100.00% | F1: 98.36%

### Model 6: CNN + Attention

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 60 (TN) | 0 (FP) |
| **Actual Depression** | 59 (FN) | 1 (TP) |

> Precision: 100.00% | Recall: 1.67% | F1: 3.28%

> **Issue:** Predicts almost everything as normal. Despite perfect AUC (ranking is correct), the sigmoid output distribution is heavily skewed — most predictions are near 0. The 0.5 threshold misses nearly all depression cases.

### Model 7: Separable CNN

|  | Predicted Normal | Predicted Depression |
| :--- | :---: | :---: |
| **Actual Normal** | 0 (TN) | 60 (FP) |
| **Actual Depression** | 0 (FN) | 60 (TP) |

> Precision: 50.00% | Recall: 100.00% | F1: 66.67%

> **Issue:** Predicts everything as depression. Similar threshold problem — the model has learned discriminative features (AUC=100%) but all sigmoid outputs are above 0.5.

---

## 5. Analysis & Key Observations

### 5.1 Top Performers (by Test Accuracy)

| Rank | Model | Test Acc | Test AUC | Size | Flex Delegate |
| :---: | :--- | ---: | ---: | ---: | :---: |
| 1 | Mel CNN (4-block) | 100.00% | 100.00% | 1124.6 KB | No |
| 1 | CNN-LSTM Hybrid | 100.00% | 100.00% | 2354.8 KB | Yes |
| 3 | Multi-Feature Fusion | 98.33% | 100.00% | 486.5 KB | Yes |
| 4 | MFCC CNN (baseline) | 96.67% | 99.61% | 118.0 KB | No |

### 5.2 CNN-based approaches dominate
- All four models with the highest accuracy are CNN-based or CNN-hybrid architectures.
- Pure sequence models (BiLSTM) and attention-only approaches struggled with calibration despite strong ranking ability (high AUC).

### 5.3 AUC vs Accuracy disconnect
- Three models (BiLSTM, CNN + Attention, Separable CNN) achieved near-perfect AUC (~100%) but poor accuracy (50-64%).
- **Root cause:** These models learned discriminative features but their sigmoid output distributions are not calibrated around the 0.5 threshold. The models have learned to rank depression > normal correctly, but the raw output values are shifted.
- **Potential fix:** Using a calibrated threshold (e.g., via Youden's J statistic on the validation set) instead of the default 0.5 would significantly improve their usable accuracy.

### 5.4 Model Size vs Performance
- **Best efficiency:** MFCC CNN at 118 KB achieves 96.67% accuracy — best size-to-performance ratio.
- **Best overall accuracy:** Mel CNN and CNN-LSTM Hybrid both achieve 100%, but CNN-LSTM is 2x larger and requires flex delegate.
- **Smallest mel-based model:** Separable CNN at 309.6 KB (39K params) — but needs threshold calibration.

### 5.5 Flex Delegate Requirements
- Models using LSTM layers (BiLSTM, CNN-LSTM) require the TensorFlow Lite Flex delegate due to `TensorListReserve`, `TensorListSetItem`, and `TensorListStack` ops.
- Multi-Feature Fusion requires flex for complex `StridedSlice` from MFCC extraction.
- This means these models need additional dependencies on mobile (larger app size).
- Models **without** flex delegate: MFCC CNN, Mel CNN, CNN + Attention, Separable CNN.

### 5.6 Inference Speed
| Model | Inference Time | Notes |
| :--- | ---: | :--- |
| MFCC CNN | 15.0 ms | Fastest — smaller feature space (13 vs 128 mel bins) |
| CNN + Attention | 24.7 ms | Surprisingly fast despite complexity |
| Mel CNN | 25.7 ms | Good balance of speed and accuracy |
| Multi-Feature | 27.0 ms | Dual-branch adds modest overhead |
| CNN-LSTM | 29.8 ms | LSTM adds latency vs pure CNN |
| Separable CNN | 43.0 ms | Depth-wise ops less optimized on CPU |
| BiLSTM | 59.8 ms | Slowest — sequential LSTM processing |

---

## 6. Training Dynamics

| Model | Best Epoch | Total Epochs | Early Stop Patience | Val Loss at Best |
| :--- | ---: | ---: | ---: | ---: |
| Mel CNN | ~20 | 40 | 10 | ~0.001 |
| BiLSTM | 19 | 29 | 10 | 0.879 |
| CNN-LSTM | 12 | 22 | 10 | 0.086 |
| Multi-Feature | 17 | 28 | 10 | 0.048 |
| CNN + Attention | 3 | 15 | 12 | 0.588 |
| Separable CNN | 15 | 26 | 10 | 0.715 |

**Notable observations:**
- CNN + Attention converged extremely fast (best at epoch 3), suggesting the attention mechanism quickly captures the global patterns.
- Separable CNN's validation loss remained high (~0.7, near random) for the first 12 epochs before suddenly improving — indicating a phase transition in learning.
- BiLSTM showed clear overfitting: training accuracy reached 99.6% while validation accuracy plateaued around 64%.

---

## 7. Recommendations

### For Mobile Deployment (Flutter App)

1. **Best Choice: Mel CNN (4-block)**
   - 100% test accuracy, 100% AUC
   - No flex delegate needed (standard TFLite builtins only)
   - 1124.6 KB model size, 25.7 ms inference
   - Already deployed and validated

2. **Alternative: MFCC CNN (baseline)**
   - 96.67% test accuracy
   - Smallest model (118 KB), fastest inference (15 ms)
   - Best for resource-constrained devices

3. **Not recommended for mobile:**
   - CNN-LSTM Hybrid — despite 100% accuracy, requires flex delegate (adds ~8 MB to app)
   - BiLSTM, Attention, Separable CNN — need threshold calibration before deployment

### For Further Improvement
- Apply **threshold calibration** to BiLSTM, Attention, and Separable CNN models (they have discriminative power but poor default thresholds)
- Test on **external datasets** (e.g., RAVDESS) for cross-domain generalization
- Consider **ensemble** of Mel CNN + Multi-Feature Fusion for improved robustness
- Explore **data augmentation** (pitch shift, time stretch, noise injection) to address potential overfitting on the small dataset

---

## 8. File Artifacts

| File | Description | Size |
| :--- | :--- | ---: |
| `artifacts/models/depression_detection_combined.tflite` | MFCC CNN combined model | 118.0 KB |
| `artifacts/models/mel_depression_combined.tflite` | Mel CNN combined model | 1124.6 KB |
| `artifacts/models/lstm_depression_combined.tflite` | BiLSTM combined model | 741.0 KB |
| `artifacts/models/cnn_lstm_depression_combined.tflite` | CNN-LSTM combined model | 2354.8 KB |
| `artifacts/models/multi_feature_depression_combined.tflite` | Multi-Feature Fusion combined model | 486.5 KB |
| `artifacts/models/attention_depression_combined.tflite` | CNN + Attention combined model | 1556.3 KB |
| `artifacts/models/separable_cnn_depression_combined.tflite` | Separable CNN combined model | 309.6 KB |

---

## 9. Cross-Domain Evaluation — RAVDESS Dataset

To assess generalization beyond the training corpus, all 7 TFLite models were evaluated on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. RAVDESS contains professional actor recordings across 8 emotions; we use **Sad → Depression (1)** and **Neutral → Normal (0)** as a proxy mapping.

### 9.1 Dataset Details

| Property | Value |
| :--- | :--- |
| Source | RAVDESS (24 actors, speech modality) |
| Total samples | 288 (192 sad, 96 neutral) |
| Class ratio | 2:1 (sad : neutral) |
| Audio format | 16 kHz, mono, 5 s padded/truncated |

### 9.2 Results Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | ms/sample |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mel CNN (4-block)** | **67.01%** | 66.90% | 100.00% | 80.17% | **72.32%** | 25.5 |
| Multi-Feature Fusion | 66.67% | 66.67% | 100.00% | 80.00% | **77.36%** | 25.1 |
| BiLSTM | 66.67% | 66.67% | 100.00% | 80.00% | 46.47% | 57.2 |
| Separable CNN | 66.67% | 66.67% | 100.00% | 80.00% | 65.30% | 41.2 |
| CNN-LSTM Hybrid | 64.93% | 66.08% | 97.40% | 78.74% | 51.61% | 29.9 |
| MFCC CNN (baseline) | 64.24% | 71.09% | 78.12% | 74.44% | 55.70% | 12.4 |
| CNN + Attention | 40.62% | 67.21% | 21.35% | 32.41% | 47.11% | 21.7 |

### 9.3 Confusion Matrix — Best Model (Mel CNN)

|  | Pred Normal | Pred Depression |
| :--- | :---: | :---: |
| **True Normal** | 1 | 95 |
| **True Sad** | 0 | 192 |

### 9.4 Per-Actor Breakdown (Mel CNN)

All 24 actors scored **66.7% accuracy** (8/12 correct), except **Actor 18** who reached **75.0%** (9/12). The model assigns very high depression probability (avg ≥ 0.92) to nearly all samples regardless of emotion, causing it to misclassify almost all neutral samples as depressed.

### 9.5 Analysis

1. **High recall, low specificity**: Most models achieve ~100% recall (detecting all sad samples) but near-zero specificity (classifying almost every neutral sample as depressed too). The models lean heavily toward the positive class.

2. **Accuracy ceiling at ~67%**: With a 2:1 class imbalance (192 sad vs 96 neutral), a model predicting "depressed" for all samples achieves exactly 66.67%. Most models are at or near this baseline, indicating they do not meaningfully discriminate on RAVDESS.

3. **AUC tells a different story**: Multi-Feature Fusion (77.36%) and Mel CNN (72.32%) show moderate ranking ability (scores for sad samples are higher than for neutral on average), suggesting the underlying representations capture *some* signal, even though the 0.5 decision threshold is poorly calibrated for this domain.

4. **MFCC CNN is the most balanced**: Although its accuracy is lower (64.24%), it is the only model that correctly identifies a meaningful number of neutral samples (TN=35), and it achieves the highest precision (71.09%) on this dataset.

5. **CNN + Attention collapses in the opposite direction**: With only 21% recall, this model predicts "normal" for most samples — the inverse failure mode.

6. **Domain gap**: The training data consists of real/semi-natural speech while RAVDESS uses acted emotions from professional actors. This fundamental domain mismatch limits cross-dataset transfer. These results are expected for a model trained on a different distribution.

### 9.6 Recommendations

- **Threshold tuning**: For deployment, calibrate the decision threshold per-model using a held-out validation set from the target domain rather than the default 0.5.
- **Domain adaptation**: Fine-tuning on a small labelled subset of the target domain (even 50–100 samples) would likely improve cross-domain performance significantly.
- **Multi-Feature Fusion** has the highest AUC on RAVDESS (77.36%), making it the best candidate if threshold calibration is applied.

---

*Report generated on March 9, 2026. All models trained on the same 800-sample dataset with identical preprocessing and evaluation splits. RAVDESS cross-domain evaluation added on March 9, 2026.*
