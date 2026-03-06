# `config/` Directory

## Purpose
This directory stores all configuration files for the ML project, including model hyperparameters, data processing settings, training configurations, and environment-specific parameters.

## What Belongs Here
- **Model configurations**: Hyperparameters, architecture definitions (YAML, JSON, Python)
- **Training configurations**: Learning rates, batch sizes, epochs, optimizer settings
- **Data pipeline configs**: Audio processing parameters (sample rate, duration, MFCC settings)
- **Environment configs**: Paths, logging levels, device settings
- **Deployment configs**: Model serving parameters, API configurations
- **Experiment tracking configs**: MLflow, Weights & Biases settings

## What Should NOT Be Here
- ❌ Credentials or secrets (use environment variables or secret management)
- ❌ Trained model files (belongs in `artifacts/models/`)
- ❌ Data files (belongs in `data/`)
- ❌ Source code (belongs in `src/`)
- ❌ Hard-coded absolute paths (use relative paths or env variables)

## Architectural Responsibilities
- **Centralized configuration management**: Single source of truth for all parameters
- **Version control**: All configs should be tracked in git
- **Environment separation**: Support different configs for dev/staging/prod
- **Reproducibility**: Enable exact experiment reproduction through config versioning

## Interactions with Other Directories
- **`src/`**: Source code imports and loads configurations from here
- **`notebooks/`**: Notebooks reference configs for consistent experimentation
- **`artifacts/`**: Model metadata may reference which config was used
- **`main.py`**: Entry point loads primary configuration files

## Best Practices
1. **Use structured formats**: Prefer YAML or JSON over Python files for better readability
2. **Schema validation**: Validate configs on load to catch errors early
3. **Naming convention**: `{component}_{environment}.yaml` (e.g., `model_production.yaml`)
4. **Document defaults**: Include comments explaining each parameter
5. **Avoid duplication**: Use inheritance or includes for shared configurations
6. **Version with code**: Commit config changes with related code changes
7. **Sensitive data**: Never commit API keys, passwords, or private credentials

## Example Structure
```
config/
├── audio_processing.yaml    # MFCC, sampling rate, duration
├── model_config.yaml         # CNN architecture, layers
├── training_config.yaml      # Epochs, batch size, learning rate
├── data_paths.yaml           # Relative paths to datasets
└── deployment_config.yaml    # TFLite conversion settings
```

## Usage Example
```python
# In src/utils/config_loader.py
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# In training script
from src.utils.config_loader import load_config
config = load_config('config/model_config.yaml')
```
