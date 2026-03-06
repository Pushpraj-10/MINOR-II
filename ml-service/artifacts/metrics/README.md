# `artifacts/metrics/` Directory

## Purpose
Store quantitative evaluation metrics, performance reports, and benchmark results from model training and testing.

## What Belongs Here
- **Evaluation metrics**: JSON/CSV files with accuracy, precision, recall, F1, AUC
- **Confusion matrices**: Numerical confusion matrix data
- **Performance reports**: Comprehensive evaluation summaries
- **Benchmark results**: Speed/memory benchmarks
- **Comparison reports**: Multi-model comparison metrics
- **Cross-validation results**: K-fold CV metrics

## What Should NOT Be Here
- ❌ Visualizations (belongs in `../plots/`)
- ❌ Model files (belongs in `../models/`)
- ❌ Training logs (can be in model directory)
- ❌ Source code (belongs in `src/evaluation/`)

## Typical Structure
```
artifacts/metrics/
├── 20260219_1400_baseline_cnn_metrics.json
├── 20260220_0930_mobilenet_metrics.json
│
├── experiments/
│   ├── experiment_001_results.csv
│   ├── experiment_002_results.csv
│   └── comparison_summary.csv
│
├── benchmarks/
│   ├── inference_speed_benchmark.json
│   ├── memory_usage_benchmark.json
│   └── mobile_device_tests.csv
│
├── cross_validation/
│   ├── 5fold_cv_results.json
│   └── fold_wise_metrics.csv
│
└── test_results_summary.csv  # Overall summary of all experiments
```

## Metrics File Format

### Individual Experiment Metrics (JSON)
```json
{
  "experiment_id": "20260219_1400_baseline_cnn",
  "model_type": "lightweight_cnn",
  "timestamp": "2026-02-19T14:00:00",
  
  "dataset": {
    "test_samples": 182,
    "positive_samples": 91,
    "negative_samples": 91
  },
  
  "classification_metrics": {
    "accuracy": 0.8396,
    "precision": 0.8122,
    "recall": 0.8736,
    "f1_score": 0.8418,
    "auc_roc": 0.8852,
    "threshold": 0.5
  },
  
  "confusion_matrix": {
    "true_positives": 79,
    "false_positives": 18,
    "false_negatives": 11,
    "true_negatives": 74,
    "sensitivity": 0.8778,
    "specificity": 0.8044
  },
  
  "per_class_metrics": {
    "normal": {
      "precision": 0.8705,
      "recall": 0.8044,
      "f1_score": 0.8361
    },
    "depressed": {
      "precision": 0.8122,
      "recall": 0.8736,
      "f1_score": 0.8418
    }
  },
  
  "computational": {
    "inference_time_mean_ms": 118.5,
    "inference_time_std_ms": 12.3,
    "model_size_mb": 3.2,
    "parameters": 247856
  }
}
```

### Comparison CSV
```csv
experiment_id,model_type,accuracy,precision,recall,f1_score,auc_roc,model_size_mb,inference_ms
20260219_1400_baseline_cnn,lightweight_cnn,0.8396,0.8122,0.8736,0.8418,0.8852,3.2,118.5
20260220_0930_mobilenet,mobilenet_inspired,0.8516,0.8345,0.8791,0.8562,0.8974,2.1,95.2
20260221_1500_1d_cnn,1d_cnn,0.8242,0.7989,0.8571,0.8270,0.8723,1.8,87.3
```

## Benchmark Metrics

### Inference Speed Benchmark
```json
{
  "benchmark_date": "2026-02-19",
  "model": "depression_model_v1.0.0",
  "device_tests": [
    {
      "device": "Pixel 6 Pro",
      "os": "Android 13",
      "cpu": "Google Tensor",
      "memory_gb": 12,
      "results": {
        "mean_inference_ms": 95.3,
        "std_inference_ms": 8.2,
        "min_inference_ms": 82.1,
        "max_inference_ms": 124.5,
        "memory_usage_mb": 45.2,
        "battery_drain_percent_per_hour": 3.2
      }
    },
    {
      "device": "iPhone 13 Pro",
      "os": "iOS 16",
      "cpu": "A15 Bionic",
      "memory_gb": 6,
      "results": {
        "mean_inference_ms": 78.1,
        "std_inference_ms": 6.5,
        "min_inference_ms": 68.3,
        "max_inference_ms": 95.7,
        "memory_usage_mb": 38.6,
        "battery_drain_percent_per_hour": 2.8
      }
    }
  ]
}
```

## Cross-Validation Results
```json
{
  "cv_strategy": "stratified_5fold",
  "random_seed": 42,
  "model_config": "config/model_config.yaml",
  
  "fold_results": [
    {
      "fold": 1,
      "train_samples": 971,
      "val_samples": 243,
      "accuracy": 0.8436,
      "auc_roc": 0.8891
    },
    {
      "fold": 2,
      "train_samples": 971,
      "val_samples": 243,
      "accuracy": 0.8272,
      "auc_roc": 0.8754
    },
    {
      "fold": 3,
      "train_samples": 971,
      "val_samples": 243,
      "accuracy": 0.8518,
      "auc_roc": 0.8923
    },
    {
      "fold": 4,
      "train_samples": 971,
      "val_samples": 243,
      "accuracy": 0.8395,
      "auc_roc": 0.8812
    },
    {
      "fold": 5,
      "train_samples": 971,
      "val_samples": 243,
      "accuracy": 0.8354,
      "auc_roc": 0.8769
    }
  ],
  
  "summary": {
    "mean_accuracy": 0.8395,
    "std_accuracy": 0.0089,
    "mean_auc_roc": 0.8830,
    "std_auc_roc": 0.0067,
    "confidence_interval_95": [0.8306, 0.8484]
  }
}
```

## Best Practices
1. **Consistent format**: Use JSON for structured data, CSV for tabular
2. **Include metadata**: Timestamp, model ID, dataset info
3. **Version control**: Track metrics files in git (they're small)
4. **Naming convention**: `{experiment_id}_{metric_type}.json`
5. **Comprehensive**: Include all relevant metrics, not just accuracy
6. **Reproducible**: Include random seeds and configuration
7. **Human-readable**: Format JSON with indentation
8. **Comparison-friendly**: Use consistent keys across experiments

## Analysis Scripts
```python
# scripts/analyze_metrics.py
import json
import pandas as pd
from pathlib import Path

def load_all_metrics(metrics_dir: str) -> pd.DataFrame:
    """Load all experiment metrics into DataFrame for analysis."""
    metrics_list = []
    
    for file in Path(metrics_dir).glob("*_metrics.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            
        metrics_list.append({
            'experiment_id': data['experiment_id'],
            'model_type': data['model_type'],
            'accuracy': data['classification_metrics']['accuracy'],
            'auc_roc': data['classification_metrics']['auc_roc'],
            'f1_score': data['classification_metrics']['f1_score'],
            'model_size_mb': data['computational']['model_size_mb'],
            'inference_ms': data['computational']['inference_time_mean_ms']
        })
    
    return pd.DataFrame(metrics_list)

# Usage
df_metrics = load_all_metrics('artifacts/metrics/')
print(df_metrics.sort_values('auc_roc', ascending=False))

# Find best model by AUC
best_model = df_metrics.loc[df_metrics['auc_roc'].idxmax()]
print(f"\nBest model: {best_model['experiment_id']}")
print(f"AUC: {best_model['auc_roc']:.4f}")
```

## Metrics Summary Example
```python
# Generate summary report
def generate_summary_report(metrics_dir: str, output_file: str):
    """Generate comprehensive summary of all experiments."""
    df = load_all_metrics(metrics_dir)
    
    summary = {
        "total_experiments": len(df),
        "best_accuracy": {
            "experiment": df.loc[df['accuracy'].idxmax(), 'experiment_id'],
            "value": float(df['accuracy'].max())
        },
        "best_auc": {
            "experiment": df.loc[df['auc_roc'].idxmax(), 'experiment_id'],
            "value": float(df['auc_roc'].max())
        },
        "smallest_model": {
            "experiment": df.loc[df['model_size_mb'].idxmin(), 'experiment_id'],
            "size_mb": float(df['model_size_mb'].min())
        },
        "fastest_inference": {
            "experiment": df.loc[df['inference_ms'].idxmin(), 'experiment_id'],
            "time_ms": float(df['inference_ms'].min())
        },
        "statistics": {
            "mean_accuracy": float(df['accuracy'].mean()),
            "std_accuracy": float(df['accuracy'].std()),
            "mean_auc": float(df['auc_roc'].mean()),
            "std_auc": float(df['auc_roc'].std())
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
```

## Notes
- Version control these files (they're small text files)
- Use clear, descriptive keys
- Include units in metric names (e.g., `inference_time_ms`)
- Save raw metrics, compute derived metrics in analysis
- Keep historical metrics for tracking progress over time
