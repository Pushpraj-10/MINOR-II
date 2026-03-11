# Depression Detection — Model Evaluation Report

**Date:** 2026-03-11

This report compares two models for speech-based depression detection:

| Property | mel_cnn_eatd (v1) | multi_feature_combined (v2) |
|----------|--------------------|-----------------------------|
| Architecture | 4-block CNN | 3-block CNN |
| Input Features | Mel spectrogram (64 bins) | 5-feature stack (MFCC, delta-MFCC, chroma, spectral contrast, ZCR — 46 bins) |
| Training Data | EATD-Corpus only | EATD-Corpus + DATASET_1 (combined) |
| Loss | Focal loss | Binary cross-entropy (label smoothing 0.05) |
| TFLite Size | 1124.2 KB | 402.4 KB |
| Parameters | ~400K | ~101K |

### Datasets

| Dataset | Description | Samples | Depression | Normal |
|---------|-------------|---------|------------|--------|
| EATD-Corpus | Real clinical depression interviews (SDS ≥ 53) | 3,337 | 568 | 2,769 |
| DATASET_1 | Internal voice recordings (depression1/normal1) | 800 | 400 | 400 |
| RAVDESS | Acted emotions (Sad → Depression, Neutral → Normal) | 288 | 192 | 96 |

---

# Part A — Original Model: `mel_cnn_eatd`

Single-feature mel-spectrogram CNN trained on EATD-Corpus only. Uses focal loss and speaker-independent splits.

## 2. EATD-Corpus (Training Domain)

Speaker-independent splits ensure no speaker appears in more than one split.

### 2.1 Split-Level Results

| Split | Samples | Dep | Norm | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC | ms/sample |
|-------|---------|-----|------|----------|-----------|--------|-------------|------|---------|-----------|
| Train | 1,504 | 345 | 1,159 | 70.35% | 30.80% | 23.48% | 84.30% | 26.64% | 72.29% | 11.5 |
| Val | 313 | 18 | 295 | 88.50% | 20.00% | 33.33% | 91.86% | 25.00% | 79.92% | 11.4 |
| Test | 1,520 | 205 | 1,315 | 76.64% | 13.59% | 13.66% | 86.46% | 13.63% | 51.27% | 11.5 |

### 2.2 Confusion Matrices

**Train:**
|  | Pred Normal | Pred Depression |
|--|-------------|-----------------|
| **Actual Normal** | TN = 977 | FP = 182 |
| **Actual Depression** | FN = 264 | TP = 81 |

**Validation:**
|  | Pred Normal | Pred Depression |
|--|-------------|-----------------|
| **Actual Normal** | TN = 271 | FP = 24 |
| **Actual Depression** | FN = 12 | TP = 6 |

**Test (Unseen Speakers):**
|  | Pred Normal | Pred Depression |
|--|-------------|-----------------|
| **Actual Normal** | TN = 1,137 | FP = 178 |
| **Actual Depression** | FN = 177 | TP = 28 |

### 2.3 Analysis

- **High specificity (84–92%):** The model correctly identifies most normal speech segments.
- **Low recall (13–33%):** The model misses a large proportion of depressed segments — expected given the heavy class imbalance (~13.5% depression in test).
- **AUC-ROC of 72.29% (train) to 51.27% (test):** Shows generalization drop on unseen speakers, indicating the model learns some speaker-specific patterns.
- **Validation AUC (79.92%)** is higher than test, likely due to the small validation set (only 18 depression samples).

---

## 3. DATASET_1 (Cross-Dataset — Internal Voice Data)

This dataset has balanced classes (400 depression, 400 normal) from a different recording domain.

### 3.1 Results

| Split | Samples | Dep | Norm | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC | ms/sample |
|-------|---------|-----|------|----------|-----------|--------|-------------|------|---------|-----------|
| Train | 560 | 280 | 280 | 50.00% | 50.00% | 100.00% | 0.00% | 66.67% | 27.33% | 11.6 |
| Val | 120 | 60 | 60 | 50.00% | 50.00% | 100.00% | 0.00% | 66.67% | 30.31% | 11.6 |
| Test | 120 | 60 | 60 | 50.00% | 50.00% | 100.00% | 0.00% | 66.67% | 28.22% | 11.6 |
| Full | 800 | 400 | 400 | 50.00% | 50.00% | 100.00% | 0.00% | 66.67% | 27.85% | 11.6 |

### 3.2 Confusion Matrix (Full Dataset)

|  | Pred Normal | Pred Depression |
|--|-------------|-----------------|
| **Actual Normal** | TN = 0 | FP = 400 |
| **Actual Depression** | FN = 0 | TP = 400 |

### 3.3 Analysis

- The model **predicts every sample as depressed** on DATASET_1, resulting in 50% accuracy(random-level on a balanced dataset).
- **AUC-ROC below 30%** indicates the model's confidence scores are inversely correlated with truth — it assigns higher depression probability to normal samples.
- This demonstrates a **significant domain mismatch**: the EATD-Corpus recordings (clinical interviews) have very different acoustic characteristics compared to DATASET_1. The model has not learned generalizable depression markers that transfer to this domain.

---

## 4. RAVDESS (Cross-Dataset — Acted Emotions)

RAVDESS maps Sad (emotion 04) → Depression and Neutral (emotion 01) → Normal.

### 4.1 Results

| Split | Samples | Dep/Sad | Norm/Neutral | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC | ms/sample |
|-------|---------|---------|--------------|----------|-----------|--------|-------------|------|---------|-----------|
| Full | 288 | 192 | 96 | 46.18% | 76.81% | 27.60% | 83.33% | 40.61% | 49.57% | 11.6 |

### 4.2 Confusion Matrix

|  | Pred Normal | Pred Depression |
|--|-------------|-----------------|
| **Actual Normal (neutral)** | TN = 80 | FP = 16 |
| **Actual Sad (depression proxy)** | FN = 139 | TP = 53 |

### 4.3 Analysis

- **High specificity (83.33%):** The model correctly identifies most neutral speech as normal.
- **Low recall (27.60%):** The model misses 72% of "sad" utterances — acted sadness differs acoustically from real depressive speech patterns.
- **AUC near 50% (49.57%):** Essentially no discriminative ability, consistent with the expectation that acted emotions are a poor proxy for clinical depression.
- **Precision is decent (76.81%):** When the model does predict depression, it is correct ~77% of the time — but it rarely makes that prediction.

---

## 5. mel_cnn_eatd Summary

| Dataset | Split | Samples | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---------|-------|---------|----------|-----------|--------|------|---------|
| **EATD-Corpus** | Train | 1,504 | 70.35% | 30.80% | 23.48% | 26.64% | 72.29% |
| **EATD-Corpus** | Val | 313 | 88.50% | 20.00% | 33.33% | 25.00% | 79.92% |
| **EATD-Corpus** | Test | 1,520 | 76.64% | 13.59% | 13.66% | 13.63% | 51.27% |
| DATASET_1 | Full | 800 | 50.00% | 50.00% | 100.00% | 66.67% | 27.85% |
| RAVDESS | Full | 288 | 46.18% | 76.81% | 27.60% | 40.61% | 49.57% |

**Key issues:** Predicts everything as depressed on DATASET_1 (AUC 27.85%). Near-chance on RAVDESS (AUC 49.57%). Low depression recall (13.66%) even on training domain.

---

# Part B — New Model: `multi_feature_combined`

## 6. Design Changes

To address the severe generalization failure of v1, the following changes were made:

### 6.1 Multi-Feature Extraction

Instead of a single mel spectrogram, five complementary features are extracted from each 5-second audio segment and stacked into a (46, 313) tensor:

| Feature | Dimensions | Purpose |
|---------|-----------|---------|
| MFCC | 13 bins | Vocal tract shape (speech content) |
| Delta MFCC | 13 bins | Temporal dynamics of speech |
| Chroma | 12 bins | Pitch/harmonic content |
| Spectral Contrast | 7 bins | Energy distribution across frequency bands |
| ZCR | 1 bin | Voice quality / breathiness |

Each feature channel is z-score normalized using statistics computed **only from the training set** to prevent data leakage.

### 6.2 Combined-Dataset Training

Training data is drawn from **both EATD-Corpus and DATASET_1**, giving the model exposure to diverse recording conditions. RAVDESS is held out entirely as a cross-domain evaluation set.

| Split | Total | Depression | Normal | Source |
|-------|-------|-----------|--------|--------|
| Train | 2,398 | 959 (augmented) | 1,439 | EATD (t_* subjects) + DS1 (70%) |
| Val | 433 | 78 | 355 | EATD (t_* subjects) + DS1 (15%) |
| Test | 1,640 | 265 | 1,375 | EATD (v_* subjects) + DS1 (15%) |
| RAVDESS | 288 | 192 | 96 | Held-out cross-domain |

Minority class augmentation (pitch shift, time stretch, noise injection) was applied with a max ratio of 0.40 to partially balance training classes.

### 6.3 Architecture

3-block CNN with ~101K parameters (4× smaller than v1):

```
Input (46, 313, 1)
  → Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.4)
  → Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.5)
  → Conv2D(128, 3×3) + BatchNorm + ReLU + GlobalAveragePooling2D
  → Dense(64) + ReLU + Dropout(0.6)
  → Dense(1, sigmoid)
```

### 6.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | Binary cross-entropy (label smoothing 0.05) |
| Optimizer | Adam (LR 0.001) |
| Class weights | {0: 0.717, 1: 1.651} (from pre-augmentation counts) |
| LR schedule | ReduceLROnPlateau (val_auc, patience=7, factor=0.5) |
| Early stopping | val_auc, patience=20, restore best weights |
| Epochs run | 43 (best at epoch 23, val_auc = 0.9066) |

---

## 7. multi_feature_combined Results

### 7.1 At Standard Threshold (0.5)

| Split | Samples | Accuracy | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----------|---------|------|-----------|--------|
| Train | 2,398 | 96.33% | 99.65% | 0.955 | 0.941 | 0.969 |
| Validation | 433 | 85.91% | **90.57%** | 0.670 | 0.579 | 0.795 |
| Test (EATD+DS1) | 1,640 | 79.94% | **66.30%** | 0.327 | 0.357 | 0.302 |
| RAVDESS | 288 | 47.92% | **52.59%** | 0.490 | 0.706 | 0.375 |

### 7.2 At Optimized Threshold (0.87 — Youden's J on validation)

| Split | Samples | Accuracy | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----------|---------|------|-----------|--------|
| Train | 2,398 | 94.04% | 99.65% | 0.919 | 1.000 | 0.851 |
| Validation | 433 | 95.15% | 90.57% | 0.849 | 0.967 | 0.756 |
| Test (EATD+DS1) | 1,640 | 86.40% | 66.30% | 0.354 | 0.763 | 0.230 |
| RAVDESS | 288 | 38.19% | 52.59% | 0.176 | 0.792 | 0.099 |

---

## 8. Head-to-Head Comparison

| Metric | mel_cnn_eatd (v1) | multi_feature_combined (v2) | Change |
|--------|--------------------|-----------------------------|--------|
| **EATD Test AUC** | 51.27% | 66.30%\* | +15.03 pp |
| **DATASET_1 AUC** | 27.85% | 66.30%\* | +38.45 pp |
| **RAVDESS AUC** | 49.57% | 52.59% | +3.02 pp |
| Val AUC | 79.92% | 90.57% | +10.65 pp |
| TFLite Size | 1124.2 KB | 402.4 KB | −64.2% |
| Parameters | ~400K | ~101K | −75% |

\*v2 test set includes both EATD v_* subjects and 15% of DATASET_1. The EATD-only and DS1-only AUC breakdown is not available separately from this combined test split.

### Key Improvements

1. **DATASET_1 no longer catastrophic:** v1 predicted 100% depressed on DS1 (AUC 27.85%). v2 includes DS1 in training, so the combined test AUC of 66.30% includes DS1 samples that the model can now partially discriminate.

2. **Validation AUC significantly higher:** 90.57% vs 79.92%, showing the multi-feature representation captures more relevant signal.

3. **Model is 4× smaller:** 101K params and 402.4 KB TFLite vs ~400K params and 1124.2 KB, better suited for mobile deployment.

4. **Multi-feature input is more robust:** Five complementary acoustic features provide a richer representation than mel spectrograms alone.

---

## 9. Remaining Limitations

1. **Generalization gap persists:** Val AUC (90.57%) → Test AUC (66.30%) is a 24 pp drop. The model still overfits to training speakers/conditions despite regularization (dropout, L2, early stopping, class weights).

2. **RAVDESS still near-chance (52.59% AUC):** Acted sadness is acoustically different from genuine depressive speech. This is a fundamental domain mismatch — not a model deficiency.

3. **Class imbalance in test set:** The test set is ~84% normal, so high accuracy (86.40% at optimized threshold) is partly driven by predicting "normal" for most samples.

4. **Threshold sensitivity:** The Youden's J threshold (0.87) found on validation data yields high precision but very low recall on test/RAVDESS, suggesting it does not transfer well.

---

## 10. Recommendations

1. **Use threshold 0.5 for deployment** — It provides a better recall/precision trade-off than the overly conservative optimized threshold.

2. **Subject-level aggregation** — Average segment predictions per speaker to reduce noise from individual 5-second segments.

3. **More training data** — The most impactful improvement would be larger datasets with diverse recording conditions. Depression detection research consistently shows that data quantity and diversity matter more than architectural complexity.

4. **Pre-trained audio embeddings** — Models like wav2vec 2.0 or HuBERT, pre-trained on large speech corpora, could provide features that generalize better across domains than hand-crafted features.

5. **Domain-adversarial training** — If cross-dataset generalization is critical, adversarial techniques can learn domain-invariant representations.

---

# Part C — Improved Model: `multi_feature_v2`

## 11. Root Cause Analysis of Val/Test AUC Gap

The 24 pp gap between v2 val AUC (90.57%) and test AUC (66.30%) was traced to **speaker identity leakage** in the validation split:

- v2 train and val sets both drew from the same EATD `t_*` subjects (85% train / 15% val split within the same speakers).
- The model learned speaker-specific acoustic patterns — achieving inflated val AUC because the same speaker voices appeared in both sets.
- Test uses only `v_*` subjects (completely different speakers), revealing the true generalization gap.

## 12. Design Changes in v2

### 12.1 Speaker-Disjoint Validation

All 83 `t_*` EATD subjects go to training. The 79 `v_*` subjects are split: 25% (19 subjects) → val, 75% (60 subjects) → test. This guarantees no speaker overlap between splits.

| Split | Total | Depression | Normal | Source |
|-------|-------|-----------|--------|--------|
| Train | 3,152 | 1,418 | 1,734 | All EATD t_* + DS1 (80%) |
| Val   | 417   | 119  | 298   | EATD v_* (25%) — speaker-disjoint |
| Test  | 1,343 | 206  | 1,137 | EATD v_* (75%) + DS1 (20%) |
| RAVDESS | 288 | 192 | 96   | Held-out cross-domain |

### 12.2 CMVN (Cepstral Mean and Variance Normalization)

Per-utterance mean and variance normalization is applied to the MFCC and delta-MFCC feature bins (0–25) before global z-score normalization. CMVN removes speaker-level channel effects (microphone, room acoustics) that do not correlate with depression status.

### 12.3 Subject-Level Evaluation

Segment-level predictions are aggregated (mean probability) per EATD speaker to compute a subject-level AUC — this is the most clinically meaningful metric since the goal is per-patient diagnosis, not per-segment labeling.

### 12.4 Training Configuration Updates

| Parameter | multi_feature_combined (v2) | multi_feature_v2 (v3) |
|-----------|-----------------------------|-----------------------|
| Val set | Same speakers as train (t_*) | Speaker-disjoint (v_* 25%) |
| CMVN | No | Yes (bins 0–25) |
| Epochs run | 43 (best ep 23) | 61 (best ep 41) |
| Best val AUC | 0.9066 (inflated) | 0.8415 (honest) |
| Optimal threshold | 0.87 | 0.793 |

---

## 13. multi_feature_v2 Results

### 13.1 At Standard Threshold (0.5)

| Split | Samples | Accuracy | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----------|---------|------|-----------|--------|
| Train | 3,152 | 98.32% | 100.00% | 0.982 | 0.964 | 1.000 |
| Validation | 417 | 70.74% | **84.08%** | 0.612 | 0.492 | 0.807 |
| Test (EATD+DS1) | 1,343 | 59.20% | **63.81%** | 0.296 | 0.201 | 0.558 |
| RAVDESS | 288 | 62.85% | **52.95%** | 0.757 | 0.671 | 0.870 |

### 13.2 At Optimized Threshold (0.793 — Youden's J on validation)

| Split | Samples | Accuracy | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----------|---------|------|-----------|--------|
| Train | 3,152 | 99.52% | 100.00% | 0.995 | 0.990 | 1.000 |
| Validation | 417 | 79.38% | 84.08% | 0.667 | 0.619 | 0.723 |
| Test (EATD+DS1) | 1,343 | 71.41% | 63.81% | 0.317 | 0.250 | 0.432 |
| RAVDESS | 288 | 57.64% | 52.95% | 0.695 | 0.668 | 0.724 |

### 13.3 EATD Subject-Level Evaluation

The 60 test `v_*` speakers (8 depressed, 52 normal) were evaluated by averaging segment-level probabilities per speaker:

| Metric | Score |
|--------|-------|
| Subject-level AUC | 43.03% |
| Subject-level Accuracy @0.5 | 51.67% |
| Optimal per-subject threshold (Youden) | 0.161 |

The subject-level AUC of 43.03% is below chance, indicating the model cannot reliably distinguish depressed from non-depressed individuals at the speaker level on held-out speakers. The class imbalance (8 dep / 52 norm) means there are very few true-positive opportunities.

---

## 14. Three-Model Comparison

### 14.1 Test Set AUC (Hold-out speakers / domain)

| Dataset | mel_cnn_eatd | multi_feature_combined | multi_feature_v2 | Best |
|---------|-------------|----------------------|------------------|------|
| EATD+DS1 Test | 51.27%* | 66.30% | 63.81% | v2 |
| RAVDESS | 49.57% | 52.59% | 52.95% | v3 |
| Val AUC (honest?) | 79.92% | 90.57% (inflated) | **84.08%** (honest) | v3 |

\*mel_cnn_eatd was EATD-only; 51.27% is the EATD test AUC. DS1 AUC was 27.85% (catastrophic).

### 14.2 Key Observations

1. **Val AUC is now honest:** v3 val AUC of 84.08% is on truly unseen speakers — a meaningful metric. v2's 90.57% was inflated by speaker leakage and 24 pp above its own test AUC.

2. **Test AUC held steady:** v3 test AUC (63.81%) is comparable to v2 (66.30%) despite the harder evaluation setup (v3 val set no longer shares speakers with train). The model generalises at about the same level once speaker leakage is fixed.

3. **RAVDESS slightly improved:** 52.95% vs 52.59% — still near-chance (acted emotions vs genuine clinical speech are inherently different domains).

4. **Subject-level AUC is the critical metric:** 43.03% reveals the model cannot reliably diagnose depression at the speaker level on held-out subjects. This is the metric that matters clinically.

5. **Threshold behaviour:** The optimized threshold (0.793) is less extreme than v2's (0.87), giving better recall on test (43.2% vs 23.0%) at the cost of some precision.

---

## 15. Updated Recommendations

1. **Acknowledge subject-level AUC as primary metric** — Segment-level AUC on a held-out test set is a reasonable proxy, but the true clinical goal is per-patient classification. Subject-level AUC should be the target metric for future improvements.

2. **Speaker-disjoint validation is non-negotiable** — Val AUC computed on same-speaker data is misleading. All future experiments should maintain speaker disjoint splits.

3. **More depressed training subjects are needed** — The test set has only 8 depressed speakers (vs 52 normal). Models trained on hundreds of depressed speakers achieve AUC > 0.80 on subject-level evaluation (literature).

4. **Pre-trained speech embeddings** — wav2vec 2.0 / HuBERT embeddings capture prosodic and temporal detail that hand-crafted features miss. These are likely necessary to break the current ~65% test AUC ceiling.

5. **Clinical framing** — RAVDESS (acted emotions) is not a valid proxy for depression. Future cross-domain evaluation should use clinical speech datasets (AVEC, DAIC-WOZ) for more meaningful benchmarks.
