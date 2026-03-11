# Training Pipeline — `multi_feature_v2`

**Script:** `ml-service/scripts/train_combined_v2.py`  
**Processes:** `data/processed/combined_v2/`  
**Output:** `artifacts/models/multi_feature_v2/`

---

## Overview

The v2 training pipeline is designed to fix the core failure mode of v1 — **speaker leakage into validation**. The entire pipeline from data processing through threshold selection is built around the principle that the model should never see the same speaker in both training and evaluation.

```
combined_v2 .npz files
        │
        ▼ [Step 1] Load splits
  Train (3152) + Val (417) + Test (1343) + RAVDESS (288)
        │
        ▼ [Step 2] Optional Mixup augmentation (--mixup flag)
  Interpolated virtual samples appended to train
        │
        ▼ [Step 3] Build 3-block CNN  (101K params)
  input (46, 313, 1) → Conv×3 → GAP → Dense → sigmoid
        │
        ▼ [Step 4] Train with EarlyStopping on val_AUC
  class-weighted BCE + label smoothing + ReduceLR
        │
        ▼ [Step 5] Threshold optimisation (Youden's J on val)
  optimal_threshold = 0.7932
        │
        ▼ [Step 6] Evaluate all datasets + subject-level AUC
  TFLite export → artifacts/models/multi_feature_v2/
```

---

## Input Features

Each audio segment is represented as a **(46 × 313)** stacked feature map (plus a channel dimension → **46 × 313 × 1**). The 46 rows are five distinct acoustic features concatenated along the frequency axis:

| Rows | Feature | Bins | What it captures |
|------|---------|------|-----------------|
| 0 – 12 | **MFCC** | 13 | Vocal tract shape / spectral envelope |
| 13 – 25 | **Delta-MFCC** | 13 | First-order temporal dynamics (prosodic changes) |
| 26 – 37 | **Chroma** | 12 | Pitch-class energy (harmonic content) |
| 38 – 44 | **Spectral Contrast** | 7 | Peak-vs-valley energy ratio per sub-band |
| 45 | **ZCR** | 1 | Zero-crossing rate (breathiness / voice activity) |
| | **Total** | **46** | |

The 313 columns are time frames produced by a 256-sample hop over 80,000 samples (5 s at 16 kHz).

### Why these features?

Each row targets a known acoustic marker of depression:

| Depression symptom | Captured by |
|--------------------|------------|
| Reduced prosodic variation / monotone speech | Delta-MFCC — flatter over time |
| Lowered pitch and slowed speech rate | Chroma — energy shifts to lower pitch classes |
| Breathiness, reduced vocal effort | ZCR — higher rate for breathy/whispered voice |
| Changed vocal tract shape | MFCC — altered formant structure |
| Flat spectral energy distribution | Spectral Contrast — reduced peak-valley gap |

### Normalisation (applied during processing, before training loads the data)

1. **CMVN** — per utterance, on MFCC + delta-MFCC bins (0–25) over the 313 time frames. Cancels speaker-level channel effects (microphone/room offset).
2. **Global z-score** — per frequency bin, statistics computed from training set only, applied to all splits. Scales all 46 rows to approximately zero mean and unit variance.

---

## Data Splits

| Split | Samples | Depressed | Normal | Source |
|-------|---------|-----------|--------|--------|
| Train | 3,152 | 1,418 | 1,734 | EATD t_* + DATASET_1 (post-augmentation) |
| Validation | 417 | 119 | 298 | EATD v_* (25%) — speaker-disjoint |
| Test | 1,343 | 206 | 1,137 | EATD v_* (75%) + DATASET_1 — speaker-disjoint |
| RAVDESS | 288 | 192 | 96 | Held-out cross-domain (sad→dep, neutral→norm) |

**Speaker-disjoint rule:** All 83 `t_*` EATD subjects go exclusively to train. The 79 `v_*` subjects are split 25/75% into val/test. No speaker appears in more than one split.

---

## Model Architecture

**3-block CNN — ~101K parameters**  
Input shape: `(46, 313, 1)`

```
Input (46, 313, 1)
│
├─ Block 1: Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.4)
│           Output: (23, 156, 32)
│
├─ Block 2: Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.5)
│           Output: (11, 78, 64)
│
├─ Block 3: Conv2D(128, 3×3, relu) → BatchNorm → GlobalAveragePooling2D
│           Output: (128,)
│
├─ Dense(64, relu) → Dropout(0.6)
│
└─ Dense(1, sigmoid)  →  depression probability
```

All Conv2D and Dense layers use **L2 regularisation (λ = 1e-4)**.

The CNN treats the 46 × 313 map like an image: it learns spatial patterns across both the feature axis (which features co-activate) and the time axis (how patterns evolve over the 5-second window).

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 (default) |
| Loss | Binary cross-entropy + label smoothing = 0.05 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | `val_auc`, patience = 20, restores best weights |
| LR reduction | ReduceLROnPlateau on `val_auc`, factor 0.5, patience 7, min 1e-6 |
| Best checkpoint | saved to `best.keras` when `val_auc` improves |

### Class Weights

Computed from **pre-augmentation** counts to avoid double-counting the synthetic samples:

$$w_{\text{dep}} = \frac{N}{2 \times N_{\text{dep}}}, \quad w_{\text{norm}} = \frac{N}{2 \times N_{\text{norm}}}$$

where $N$ = total pre-aug train samples, $N_{\text{dep}}$ = 810, $N_{\text{norm}}$ = 1,733 → $w_{\text{dep}} \approx 1.57$, $w_{\text{norm}} \approx 0.73$.

### Label Smoothing

The target labels are softened from hard {0,1} to {0.025, 0.975} using `label_smoothing=0.05`. This prevents the model from becoming overconfident on augmented training segments and improves calibration.

### Optional Mixup (`--mixup` flag)

When enabled, 25% × N synthetic samples are created per epoch by linearly blending pairs of training examples:

$$X_{\text{mix}} = \lambda X_i + (1-\lambda) X_j, \quad y_{\text{mix}} = \lambda y_i + (1-\lambda) y_j$$

where $\lambda \sim \text{Beta}(0.3, 0.3)$. The soft labels are fully compatible with binary cross-entropy.

---

## Threshold Optimisation

After training, the model is evaluated on the **validation set** to find the optimal decision threshold using **Youden's J statistic**:

$$J = \text{TPR} - \text{FPR}, \quad \hat{t} = \arg\max_t J(t)$$

The resulting threshold (`optimal_threshold = 0.7932`) is higher than 0.5, meaning the model is conservative — it only predicts depression when it is quite confident. This trades some recall for precision.

| Threshold | Use case |
|-----------|---------|
| **0.50** | Clinical screening — prefer high recall (catch more cases, accept false alarms) |
| **0.793** | Balanced precision/recall — better F1 and accuracy |

---

## Evaluation

All four datasets are evaluated at both thresholds. Additionally, EATD test speakers receive **subject-level evaluation**:

1. Compute sigmoid probability for every segment of that speaker.
2. Average all segment probabilities → one score per speaker.
3. Apply threshold → binary prediction per speaker.
4. Compute AUC and accuracy across all test speakers.

This mirrors real clinical deployment: a patient produces multiple speech segments, and a single diagnosis is produced per patient.

### Results Summary

| Metric | Train | Val | Test | RAVDESS |
|--------|-------|-----|------|---------|
| AUC | ~99% | 84.15% | 63.81% | 52.95% |
| Accuracy (@0.5) | ~95% | 65.5% | 59.2% | 49.7% |
| F1 (@0.5) | ~0.93 | 0.41 | 0.30 | 0.50 |

Subject-level AUC (EATD test): **43.03%** — below chance, indicating the model has not yet generalised at the speaker level.

---

## Output Artefacts

| File | Contents |
|------|----------|
| `best.keras` | Best model checkpoint (by val_AUC), used for inference |
| `final.keras` | Model state at end of training |
| `multi_feature_v2.tflite` | Quantised TFLite model (110.5 KB) for mobile deployment |
| `training_summary.json` | Hyperparameters, per-split metrics, optimal threshold |

---

## Running the Pipeline

```bash
cd ml-service

# Standard run
.\venv\Scripts\python.exe scripts/train_combined_v2.py --epochs 100 --lr 0.001 --label-smoothing 0.05

# With Mixup augmentation
.\venv\Scripts\python.exe scripts/train_combined_v2.py --epochs 100 --mixup

# Must process data first (if combined_v2/ doesn't exist)
.\venv\Scripts\python.exe scripts/process_combined_v2.py
```
