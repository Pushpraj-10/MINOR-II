# Data Processing Pipeline — Speech-Based Depression Detection

**Model:** `multi_feature_v2`  
**Date:** 2026-03-11  
**Script:** `ml-service/scripts/process_combined_v2.py`

---

## Overview

The pipeline converts raw audio recordings into normalised, stacked feature tensors ready for CNN training. It runs in **six sequential stages**:

```
Raw audio (WAV)
    │
    ▼ [Stage 1] Load & segment
Overlapping 5-second clips  (sr=16 kHz, 80,000 samples)
    │
    ▼ [Stage 2] Speaker-disjoint splits
 t_* subjects → Train
 v_* subjects → Val (25%) / Test (75%)
 DS1 random split (70 / 15 / 15%)
    │
    ▼ [Stage 3] Minority-class augmentation
 pitch shift · time stretch · noise injection
 (training set only, max_ratio = 0.45)
    │
    ▼ [Stage 4] Multi-feature extraction
 5 features → stacked (46, 313) tensor
    │
    ▼ [Stage 5] Normalisation
 CMVN per utterance → global z-score per feature bin
    │
    ▼ [Stage 6] Save
 data/processed/combined_v2/  (.npz)
```

---

## Stage 1 — Audio Loading & Segmentation

### Configuration

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Segment length | 5 seconds (80,000 samples) |
| Overlap | 50% between consecutive segments |
| Padding | zero-pad if shorter than 5 s |
| Truncation | trim if longer |
| Amplitude normalisation | peak normalise to ±1.0 |

### Datasets

| Source | Subjects / Files | Depression criterion |
|--------|-----------------|----------------------|
| EATD-Corpus | 162 subjects (t_*, v_*) | SDS score ≥ 53 |
| DATASET_1 | 800 recordings (depression1 / normal1) | folder label |
| RAVDESS | 24 actors × various | emotion code `04` (sad) → dep; `01` (neutral) → norm |

---

## Stage 2 — Speaker-Disjoint Splits

The critical change from v1: **no speaker identity is shared between train and validation**.

### EATD-Corpus split logic

```
All 83 t_* subjects ──────────────────────────────► Train
All 79 v_* subjects ─── stratified split ────────►  Val  (25% = 19 subjects)
                                                     Test (75% = 60 subjects)
```

Stratification is on subject-level label (dep/norm) using `sklearn.model_selection.train_test_split` with `random_state=42`.

### DATASET_1 split

Random 70 / 15 / 15% split (no speaker metadata available), combined with EATD segments in the corresponding split.

### Final segment counts

| Split | Total | Depressed | Normal |
|-------|-------|-----------|--------|
| Train (pre-aug) | 2,543 | 810 | 1,733 |
| Train (post-aug) | 3,152 | 1,418 | 1,734 |
| Validation | 417 | 119 | 298 |
| Test | 1,343 | 206 | 1,137 |
| RAVDESS (held-out) | 288 | 192 | 96 |

> **Class imbalance:** Depression is a minority class (~34% in train, ~29% in val, ~15% in test). Handled via augmentation (train) and class-weighted loss (all).

![Class distribution](ml-service/artifacts/plots/5_class_distribution.png)

---

## Stage 3 — Minority-Class Augmentation

Applied **only to training depression segments** to partially close the class imbalance gap.

### Augmentation techniques

| Technique | Parameter | Description |
|-----------|-----------|-------------|
| Pitch shift | ±2 semitones (random) | `librosa.effects.pitch_shift` |
| Time stretch | ×0.9–1.1 (random) | `librosa.effects.time_stretch`, re-padded/trimmed |
| Additive Gaussian noise | σ = 0.005 | simulates real recording noise |

Each depressed segment is augmented up to a `max_ratio = 0.45` relative to the normal class count (i.e., depression segments are boosted to at most 45% of the normal count). Augmented samples are **only** added; the originals are kept.

---

## Stage 4 — Multi-Feature Extraction

Each 5-second segment produces a **stacked (46, 313) feature map**, where 46 is the total frequency/coefficient axis and 313 is the number of time frames.

### STFT parameters

| Parameter | Value |
|-----------|-------|
| FFT size (n_fft) | 512 |
| Hop length | 256 samples (16 ms) |
| Time frames | ⌊(80000 − 512) / 256⌋ + 1 = **313** |

### Feature blocks

| Feature | Bins | Rows in tensor | Purpose |
|---------|------|----------------|---------|
| **MFCC** | 13 | 0–12 | Vocal tract shape (spectral envelope) |
| **Delta-MFCC** | 13 | 13–25 | First-order temporal dynamics |
| **Chroma** | 12 | 26–37 | Pitch-class energy (harmonic content) |
| **Spectral Contrast** | 7 | 38–44 | Peak-valley energy ratio per sub-band |
| **ZCR** | 1 | 45 | Voice activity / breathiness proxy |
| **Total** | **46** | 0–45 | — |

All five features share the same time axis. Shorter features are zero-padded; longer are truncated to exactly 313 frames.

Feature map examples (one depressed, one normal sample):

![Feature spectrograms](ml-service/artifacts/plots/1_feature_spectrograms.png)

### Why these five features?

| Symptom of depression | Captured by |
|-----------------------|------------|
| Reduced prosodic variation | Delta-MFCC (flattened dynamics) |
| Lowered fundamental frequency | Chroma (energy shifts to lower pitch classes) |
| Reduced voice energy / breathiness | ZCR (higher for breathy/whispered speech) |
| Changed vocal tract tension | MFCC (formant structure) |
| Monotone energy distribution | Spectral Contrast (less contrast between peaks and valleys) |

---

## Stage 5 — Normalisation Pipeline

Normalisation happens in two steps, in this order:

### Step 5a — CMVN (Cepstral Mean and Variance Normalisation)

Applied **per utterance** to MFCC and delta-MFCC bins (rows 0–25):

$$\hat{x}_{i,t} = \frac{x_{i,t} - \mu_i}{\sigma_i + \epsilon}$$

where $\mu_i, \sigma_i$ are the mean and standard deviation of bin $i$ computed **over the 313 time frames of that utterance**, and $\epsilon = 10^{-8}$ avoids division by zero.

**Why CMVN?** Different microphones and recording environments introduce additive offsets in cepstral coefficients (channel effects). CMVN cancels them by centring each utterance around zero — the model then sees relative spectral patterns rather than absolute values that may differ by speaker/recording setup.

Chroma, Spectral Contrast, and ZCR are not CMVN-normalised because they carry useful absolute information (e.g., ZCR magnitude is meaningful).

### Step 5b — Global Z-score Normalisation

After CMVN, a per-frequency-bin mean and standard deviation is computed **from the training set only** and applied to all splits:

$$\hat{X}_{n,f,t} = \frac{X_{n,f,t} - \bar{\mu}_f}{\bar{\sigma}_f + \epsilon}$$

This scales each of the 46 feature rows to approximately zero mean and unit variance across the entire corpus, helping gradient flow in the CNN.

> The scaler statistics (`mean`, `std`) are saved in `data/processed/combined_v2/scaler.npz` and must be loaded at inference time to normalise new audio consistently.

---

## Stage 6 — Saved Artefacts

Everything is saved to `data/processed/combined_v2/`:

| File | Contents |
|------|----------|
| `train_features.npz` | `X` (3152, 46, 313), `y` (3152,) |
| `val_features.npz` | `X` (417, 46, 313), `y` (417,) |
| `test_features.npz` | `X` (1343, 46, 313), `y` (1343,) |
| `ravdess_features.npz` | `X` (288, 46, 313), `y` (288,) |
| `scaler.npz` | `mean` (46,), `std` (46,) — computed on train only |
| `test_subject_ids.npz` | `subject_ids` (1343,) — EATD subject identifiers for test samples |
| `metadata.npz` | pre-augmentation depression/normal counts |

---

## Feature Analysis

### Correlation Matrix

The heatmap below shows Pearson correlations between all 46 feature dimensions (temporal mean per sample) on the training set. Each cell shows how much two feature bins co-vary across all 3,152 training segments.

![Correlation matrix](ml-service/artifacts/plots/2_feature_correlation_matrix.png)

**Key observations:**

- **MFCC ↔ Delta-MFCC:** Moderate positive correlation within the same coefficient index (lower-order coefficients more correlated than higher-order), which is expected — the delta of a slowly varying coefficient will itself be small.
- **Chroma ↔ Chroma:** Strong internal correlations (especially adjacent pitch classes), because energy in one pitch class often leaks into neighbouring ones.
- **Spectral Contrast:** Low correlation with MFCC/Chroma — it captures a different aspect (energy ratio between peaks and valleys) than envelope shape.
- **ZCR:** Nearly uncorrelated with all other features — confirms it carries independent information about voice periodicity.

### Feature Distributions: Depressed vs. Normal

![Feature distributions (train)](ml-service/artifacts/plots/4_feature_distributions_train.png)

![Feature distributions (test)](ml-service/artifacts/plots/4_feature_distributions_test.png)

The box plots show temporal-mean distributions split by class. Observations:
- **MFCC:** Slight depression shift toward lower mean values (reduced vocal energy).
- **Delta-MFCC:** Distribution is tighter for depressed speech (flatter prosody, less variation).
- **Chroma:** Distributions overlap substantially — pitch content alone is not discriminative.
- **Spectral Contrast:** Depressed speech shows marginally lower contrast (flatter spectral energy).
- **ZCR:** Small but consistent difference; depressed speech tends to have slightly higher ZCR (breathiness).

---

## Prediction Score Distributions

The histograms show the model's output probability for each class, per split:

![Score distributions](ml-service/artifacts/plots/6_score_distributions.png)

- On training data: well-separated distributions (model has learned the training pattern).
- On validation/test: distributions overlap significantly — the depression probability for true-depressed samples is spread across the full [0, 1] range.
- The optimal threshold (0.793, Youden's J) is higher than 0.5 because the model is conservative — most outputs are pushed toward 0.

---

## Confusion Matrices

### Default threshold (0.5)

![Confusion matrices thr=0.5](ml-service/artifacts/plots/3a_confusion_matrices_thr0.5.png)

### Optimised threshold (0.793)

![Confusion matrices optimised](ml-service/artifacts/plots/3b_confusion_matrices_optimised_thr.png)

### Test set: threshold comparison

![Test threshold comparison](ml-service/artifacts/plots/3c_test_threshold_comparison.png)

**Trade-off summary:**

| Threshold | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-----------|--------------|----------------|-------------|---------|
| 0.50 | 59.2% | 20.1% | **55.8%** | 0.296 |
| 0.793 | 71.4% | 25.0% | 43.2% | 0.317 |

- At **0.5**: higher recall (catches more true depressed), but many false alarms (FP = 457).
- At **0.793**: better precision and accuracy, but misses more depressed cases (lower recall).
- For a clinical screening tool where **missing depression is the worst error**, threshold **0.5** is preferred.

---

## Inference Pipeline (new audio)

To process new audio at inference time, apply the same pipeline in this order:

1. Load WAV at 16 kHz, mono.
2. Segment into 5-second clips (80,000 samples), zero-pad last clip if needed.
3. Extract the 5 features with the same STFT parameters (`n_fft=512`, `hop_length=256`).
4. Apply CMVN to MFCC+delta-MFCC bins (rows 0–25) per clip.
5. Load `scaler.npz` and apply global z-score normalisation.
6. Add channel dimension: `X[..., np.newaxis]` → shape `(N, 46, 313, 1)`.
7. Run through `multi_feature_v2.tflite` (110.5 KB) or `best.keras`.
8. Average segment probabilities per speaker for subject-level diagnosis.
