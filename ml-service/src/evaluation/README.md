# `src/evaluation/` Directory

## Purpose
This module handles model evaluation, metrics computation, performance analysis, and generation of evaluation reports and visualizations.

## What Belongs Here
- **Metrics computation**: Accuracy, precision, recall, F1, AUC-ROC
- **Evaluation pipelines**: Systematic model testing procedures
- **Performance analysis**: Error analysis, confusion matrices
- **Visualization**: ROC curves, calibration plots, training curves
- **Model comparison**: Compare multiple models or versions
- **Benchmark utilities**: Test model performance on standard datasets

## What Should NOT Be Here
- ❌ Training logic (belongs in `src/pipelines/`)
- ❌ Model definitions (belongs in `src/models/`)
- ❌ Feature extraction (belongs in `src/features/`)
- ❌ Saved plots/metrics (belongs in `artifacts/metrics/` and `artifacts/plots/`)

## Architectural Responsibilities
- **Performance measurement**: Quantify model quality objectively
- **Diagnostic analysis**: Identify model weaknesses and errors
- **Reporting**: Generate comprehensive evaluation reports
- **Visualization**: Create intuitive performance visualizations

## Typical Modules

### `metrics.py`
Core metrics computation
```python
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, AUC."""
    pass

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """Compute confusion matrix."""
    pass
```

### `evaluator.py`
High-level evaluation orchestration
```python
class ModelEvaluator:
    """Evaluate model performance on test data."""
    
    def __init__(self, model: keras.Model):
        self.model = model
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Run full evaluation and return metrics."""
        pass
    
    def generate_report(self, save_dir: str) -> None:
        """Generate comprehensive evaluation report."""
        pass
```

### `visualization.py`
Evaluation visualizations
```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None
) -> None:
    """Plot and optionally save confusion matrix."""
    pass

def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = None
) -> None:
    """Plot ROC curve with AUC score."""
    pass

def plot_training_history(
    history: Dict,
    save_path: str = None
) -> None:
    """Plot training/validation loss and accuracy curves."""
    pass
```

### `error_analysis.py`
Analyze model errors and failures
```python
def analyze_errors(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """Analyze misclassified samples."""
    pass

def get_high_confidence_errors(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    confidence_threshold: float = 0.9
) -> List[int]:
    """Find high-confidence incorrect predictions."""
    pass
```

## Interactions with Other Modules
- **`src/models/`**: Loads trained models for evaluation
- **`src/features/`**: May need to extract features from test data
- **`artifacts/models/`**: Loads saved models
- **`artifacts/metrics/`**: Saves computed metrics
- **`artifacts/plots/`**: Saves visualization plots
- **`data/processed/`**: Loads test datasets

## Best Practices
1. **Comprehensive metrics**: Report multiple metrics (accuracy alone is insufficient)
2. **Stratified evaluation**: Consider class balance in metrics
3. **Confidence intervals**: Report uncertainty in metrics when possible
4. **Threshold analysis**: Test different classification thresholds
5. **Cross-validation**: Use k-fold CV for robust performance estimates
6. **Reproducibility**: Fix random seeds for consistent evaluation
7. **Documentation**: Clearly document which dataset was used for evaluation
8. **Visualization quality**: Create publication-ready plots

## Example Implementation
```python
# src/evaluation/metrics.py
"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'threshold': threshold
    }
    
    # Add confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
    })
    
    logger.info(f"Metrics computed: Accuracy={metrics['accuracy']:.4f}, "
                f"AUC={metrics['auc_roc']:.4f}")
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} "
                f"({metric}={best_score:.4f})")
    
    return best_threshold, best_score
```

## Evaluation Orchestration
```python
# src/evaluation/evaluator.py
"""Model evaluation orchestrator."""

import numpy as np
from tensorflow import keras
from typing import Dict, Optional
import json
from pathlib import Path
import logging

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_roc_curve
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: keras.Model,
        model_name: str = "model"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            model_name: Name for saving results
        """
        self.model = model
        self.model_name = model_name
        self.metrics = None
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test).flatten()
        
        # Compute metrics
        self.metrics = compute_classification_metrics(
            y_test, y_pred_proba, threshold
        )
        
        # Add sample info
        self.metrics['num_test_samples'] = len(X_test)
        self.metrics['positive_samples'] = int(np.sum(y_test))
        self.metrics['negative_samples'] = int(len(y_test) - np.sum(y_test))
        
        return self.metrics
    
    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: str
    ) -> None:
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation if not done yet
        if self.metrics is None:
            self.evaluate(X_test, y_test)
        
        # Save metrics as JSON
        metrics_path = save_path / f"{self.model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate visualizations
        y_pred_proba = self.model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Confusion matrix
        cm_path = save_path / f"{self.model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, save_path=str(cm_path))
        
        # ROC curve
        roc_path = save_path / f"{self.model_name}_roc_curve.png"
        plot_roc_curve(y_test, y_pred_proba, save_path=str(roc_path))
        
        logger.info(f"Evaluation report generated in {save_dir}")
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        if self.metrics is None:
            logger.warning("No metrics available. Run evaluate() first.")
            return
        
        print(f"\n{'='*50}")
        print(f"Model Evaluation Summary: {self.model_name}")
        print(f"{'='*50}")
        print(f"Test Samples: {self.metrics['num_test_samples']}")
        print(f"  - Depressed: {self.metrics['positive_samples']}")
        print(f"  - Normal: {self.metrics['negative_samples']}")
        print(f"\nPerformance Metrics:")
        print(f"  - Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  - Precision: {self.metrics['precision']:.4f}")
        print(f"  - Recall:    {self.metrics['recall']:.4f}")
        print(f"  - F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"  - AUC-ROC:   {self.metrics['auc_roc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  - TP: {self.metrics['true_positives']}, "
              f"FP: {self.metrics['false_positives']}")
        print(f"  - FN: {self.metrics['false_negatives']}, "
              f"TN: {self.metrics['true_negatives']}")
        print(f"{'='*50}\n")
```

## Visualization Module
```python
# src/evaluation/visualization.py
"""Evaluation visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = ['Normal', 'Depressed'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for display
        save_path: Path to save plot (None to display)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
```
