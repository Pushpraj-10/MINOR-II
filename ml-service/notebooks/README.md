# `notebooks/` Directory

## Purpose
This directory contains Jupyter notebooks for exploratory data analysis (EDA), prototyping, experimentation, and interactive model development before productionizing code.

## What Belongs Here
- **EDA notebooks**: Data exploration, visualization, statistical analysis
- **Prototype experiments**: Quick model iterations and hypothesis testing
- **Data quality checks**: Investigate data issues, outliers, distributions
- **Model analysis**: Feature importance, error analysis, performance investigation
- **Documentation notebooks**: Tutorial-style notebooks explaining the pipeline
- **Presentation notebooks**: Demo notebooks for stakeholders

## What Should NOT Be Here
- ❌ Production code (should be refactored to `src/`)
- ❌ Notebooks with sensitive credentials (use environment variables)
- ❌ Duplicate code across notebooks (extract to `src/utils/`)
- ❌ Very large outputs (clear outputs before committing)
- ❌ Notebooks that haven't been run end-to-end

## Architectural Responsibilities
- **Experimentation sandbox**: Rapid iteration without breaking production code
- **Knowledge sharing**: Document insights, findings, and decision rationale
- **Prototyping**: Test ideas before committing to production implementation
- **Reproducible research**: Enable others to understand and reproduce your analysis

## Interactions with Other Directories
- **`src/`**: Import utility functions and classes from source code
- **`data/`**: Load datasets for analysis (prefer `processed/` for speed)
- **`config/`**: Load configurations for consistency with production
- **`artifacts/`**: Load models and metrics for analysis
- **Root `main.py`**: Production code may be refactored from successful notebooks

## Best Practices
1. **Clear naming**: Use descriptive names like `01_eda_audio_features.ipynb`, `02_baseline_model.ipynb`
2. **Number prefixes**: Order notebooks logically (01, 02, 03...) for workflow clarity
3. **Clear outputs before commit**: `jupyter nbconvert --clear-output --inplace *.ipynb`
4. **Run top-to-bottom**: Ensure notebooks execute in order without errors
5. **Add markdown documentation**: Explain what each section does and why
6. **Modular imports**: Use `%load_ext autoreload` and import from `src/`
7. **Time-stamp versions**: Archive old experiments as `experiment_name_YYYYMMDD.ipynb`
8. **Refactor to production**: Move proven code from notebooks to `src/`

## Notebook Naming Convention
```
notebooks/
├── 01_eda_voice_dataset.ipynb           # Initial data exploration
├── 02_audio_preprocessing_analysis.ipynb # MFCC extraction tests
├── 03_baseline_cnn_model.ipynb          # First model prototype
├── 04_model_optimization.ipynb          # Quantization experiments
├── 05_error_analysis.ipynb              # Investigate predictions
└── demo_depression_detection.ipynb      # Interactive demo
```

## Template Structure for Notebooks
```python
# Cell 1: Notebook metadata and description
"""
# Notebook Title

**Author**: Your Name
**Date**: 2026-02-19
**Purpose**: Brief description of notebook goal
**Status**: [Exploration | Prototype | Production-Ready | Archived]
"""

# Cell 2: Imports and setup
import sys
sys.path.append('..')  # Add project root to path

from src.utils.config_loader import load_config
from src.features.audio_processing import extract_mfcc
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2

# Cell 3: Load configurations
config = load_config('../config/audio_processing.yaml')

# Continue with analysis...
```

## Version Control Tips
```bash
# Install nbstripout to auto-clear outputs
pip install nbstripout
nbstripout --install

# Or manually clear outputs before committing
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Converting Notebooks to Scripts
```bash
# Extract code from notebook to .py script
jupyter nbconvert --to script notebooks/03_baseline_cnn_model.ipynb

# Move refined code to src/
# Refactor into modular functions
```

## Notebook Categories

### 🔍 Exploratory (EDA)
- Understand data distributions
- Identify data quality issues
- Visualize patterns and correlations

### 🧪 Experimental
- Prototype new features
- Test different model architectures
- Hyperparameter tuning

### 📊 Analytical
- Model performance analysis
- Error analysis and debugging
- Feature importance studies

### 📚 Documentation
- Explain pipeline decisions
- Tutorial walkthroughs
- Demo notebooks for presentations
