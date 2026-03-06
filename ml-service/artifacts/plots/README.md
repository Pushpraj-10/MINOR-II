# `artifacts/plots/` Directory

## Purpose
Store all visualization outputs from model training, evaluation, and analysis including charts, graphs, and diagnostic plots.

## What Belongs Here
- **Training curves**: Loss and accuracy over epochs
- **Confusion matrices**: Heatmaps of classification results
- **ROC curves**: ROC/AUC visualizations
- **Feature visualizations**: MFCC spectrograms, audio waveforms
- **Error analysis plots**: Misclassification examples
- **Comparison charts**: Multi-model performance comparisons
- **Distribution plots**: Class distributions, prediction histograms

## What Should NOT Be Here
- ❌ Raw metrics data (belongs in `../metrics/`)
- ❌ Model files (belongs in `../models/`)
- ❌ Source code (belongs in `src/evaluation/`)
- ❌ Temporary/draft plots (clean up before committing)

## Typical Structure
```
artifacts/plots/
├── training/
│   ├── 20260219_1400_baseline_cnn_loss_curve.png
│   ├── 20260219_1400_baseline_cnn_accuracy_curve.png
│   └── 20260220_0930_mobilenet_training_curves.png
│
├── evaluation/
│   ├── confusion_matrix_baseline_cnn.png
│   ├── roc_curve_baseline_cnn.png
│   ├── precision_recall_curve.png
│   └── calibration_plot.png
│
├── features/
│   ├── mfcc_spectrogram_depressed.png
│   ├── mfcc_spectrogram_normal.png
│   └── feature_distribution.png
│
├── comparisons/
│   ├── model_accuracy_comparison.png
│   ├── model_size_vs_accuracy.png
│   └── inference_time_comparison.png
│
└── error_analysis/
    ├── high_confidence_errors.png
    └── prediction_distribution.png
```

## Plot Specifications

### Image Format
- **Format**: PNG (for quality and transparency)
- **DPI**: 300 (publication quality)
- **Size**: 8x6 inches (standard)
- **Font**: Clear, readable (10-12pt minimum)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
# ... plotting code ...
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Common Plot Types

### 1. Training Loss Curves
```python
import matplotlib.pyplot as plt

def plot_training_curves(history, save_path):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 2. Confusion Matrix Heatmap
```python
import seaborn as sns

def plot_confusion_matrix(cm, labels, save_path):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 3. ROC Curve
```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores, save_path):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, 
             linestyle='--', label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 4. Model Comparison
```python
def plot_model_comparison(metrics_df, save_path):
    """Compare multiple models on key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    axes[0, 0].barh(metrics_df['model_type'], metrics_df['accuracy'])
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # AUC comparison
    axes[0, 1].barh(metrics_df['model_type'], metrics_df['auc_roc'])
    axes[0, 1].set_xlabel('AUC-ROC')
    axes[0, 1].set_title('Model AUC Comparison')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Model size
    axes[1, 0].barh(metrics_df['model_type'], metrics_df['model_size_mb'])
    axes[1, 0].set_xlabel('Model Size (MB)')
    axes[1, 0].set_title('Model Size Comparison')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Inference time
    axes[1, 1].barh(metrics_df['model_type'], metrics_df['inference_ms'])
    axes[1, 1].set_xlabel('Inference Time (ms)')
    axes[1, 1].set_title('Inference Speed Comparison')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 5. MFCC Visualization
```python
import librosa.display

def plot_mfcc_spectrogram(mfcc, sr, save_path):
    """Visualize MFCC features."""
    plt.figure(figsize=(10, 6))
    
    librosa.display.specshow(
        mfcc, sr=sr, x_axis='time',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC Spectrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('MFCC Coefficients', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

## Best Practices
1. **Consistent styling**: Use same color scheme across plots
2. **High quality**: 300 DPI for publication/presentation
3. **Clear labels**: All axes, titles, and legends labeled
4. **Readable fonts**: Minimum 10-12pt font size
5. **Grid lines**: Add subtle grid for readability
6. **Color accessibility**: Use colorblind-friendly palettes
7. **Tight layout**: Use `bbox_inches='tight'` to avoid clipping
8. **Close figures**: Call `plt.close()` to free memory

## Style Configuration
```python
# Set global matplotlib style
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Colorblind-friendly palette
colors = sns.color_palette('colorblind')
```

## Naming Convention
```
{experiment_id}_{plot_type}_{optional_detail}.png

Examples:
20260219_1400_training_curves.png
baseline_cnn_confusion_matrix.png
model_comparison_accuracy_vs_size.png
mfcc_spectrogram_depressed_sample_01.png
```

## File Size Management
```python
# For plots that might be large, optimize
plt.savefig('plot.png', dpi=300, bbox_inches='tight', 
            optimize=True)  # For PNG optimization

# Or use JPEG for very large images (with slight quality loss)
plt.savefig('plot.jpg', dpi=300, bbox_inches='tight', quality=95)
```

## Version Control
- **Do commit**: Final, publication-ready plots
- **Don't commit**: Draft plots, experimental visualizations
- Add to `.gitignore` if plots are auto-generated and can be reproduced

## Plot Generation Automation
```python
# scripts/generate_all_plots.py
def generate_experiment_plots(experiment_dir: str):
    """Generate all plots for an experiment."""
    
    # Load results
    metrics = load_metrics(f"{experiment_dir}/metrics.json")
    history = load_history(f"{experiment_dir}/training_history.json")
    
    # Create plots directory
    plots_dir = Path(experiment_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_training_curves(history, plots_dir / 'training_curves.png')
    plot_roc_curve(y_true, y_scores, plots_dir / 'roc_curve.png')
    plot_confusion_matrix(cm, labels, plots_dir / 'confusion_matrix.png')
    
    print(f"Generated plots in {plots_dir}")
```

## Notes
- Always close figures after saving to free memory
- Use vector formats (PDF, SVG) for presentations if needed
- Consider dark mode variants for presentations
- Compress PNGs if file size is an issue
- Include timestamp or experiment ID in filenames
