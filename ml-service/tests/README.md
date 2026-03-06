# `tests/` Directory

## Purpose
This directory contains automated tests to ensure code quality, correctness, and reliability across the ML pipeline.

## What Belongs Here
- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test interactions between components
- **Data validation tests**: Verify data quality and schema compliance
- **Model tests**: Test model inference, performance thresholds
- **Pipeline tests**: End-to-end testing of training/inference pipelines
- **Fixtures and mocks**: Sample data and mock objects for testing

## What Should NOT Be Here
- ❌ Production code (belongs in `src/`)
- ❌ Large test datasets (use small samples or synthetic data)
- ❌ Experiment notebooks (belongs in `notebooks/`)
- ❌ Configuration files (belongs in `config/`)

## Architectural Responsibilities
- **Quality assurance**: Catch bugs before they reach production
- **Regression prevention**: Ensure changes don't break existing functionality
- **Documentation**: Tests serve as executable documentation
- **Confidence**: Enable safe refactoring and feature additions

## Test Structure
```
tests/
├── test_data_processing.py      # Test audio loading, preprocessing
├── test_feature_extraction.py   # Test MFCC extraction
├── test_model_inference.py      # Test model predictions
├── test_utils.py                # Test utility functions
├── test_pipeline.py             # End-to-end pipeline tests
├── conftest.py                  # pytest fixtures and configuration
└── fixtures/
    ├── sample_audio.wav         # Small test audio file
    └── sample_features.npy      # Pre-computed test features
```

## Interactions with Other Directories
- **`src/`**: Tests import and validate code from source modules
- **`data/`**: May use small samples from data for integration tests
- **`config/`**: Load test configurations for reproducible tests
- **`artifacts/`**: May test model loading and inference

## Best Practices
1. **Test naming**: `test_{module}_{function}_{scenario}.py`
2. **Isolation**: Each test should be independent
3. **Fast execution**: Keep unit tests under 1 second each
4. **Fixtures**: Use pytest fixtures for common setup
5. **Coverage**: Aim for >80% code coverage on critical paths
6. **Mock external dependencies**: Don't hit APIs or databases in unit tests
7. **Deterministic**: Tests should produce consistent results
8. **Clear assertions**: One logical assertion per test

## Test Categories

### Unit Tests
Test individual functions in isolation
```python
# tests/test_feature_extraction.py
import pytest
import numpy as np
from src.features.audio_processing import extract_mfcc

def test_extract_mfcc_shape():
    """Test MFCC output shape is correct"""
    # Arrange
    audio_path = "tests/fixtures/sample_audio.wav"
    
    # Act
    mfccs = extract_mfcc(audio_path, n_mfcc=13)
    
    # Assert
    assert mfccs.shape[0] == 13, "Number of MFCC coefficients should be 13"
    assert mfccs.shape[1] > 0, "Should have time steps"

def test_extract_mfcc_normalization():
    """Test MFCC features are normalized"""
    audio_path = "tests/fixtures/sample_audio.wav"
    mfccs = extract_mfcc(audio_path, normalize=True)
    
    assert abs(np.mean(mfccs)) < 0.1, "Mean should be close to 0"
    assert abs(np.std(mfccs) - 1.0) < 0.1, "Std should be close to 1"
```

### Integration Tests
Test component interactions
```python
# tests/test_pipeline.py
from src.pipelines.training_pipeline import train_model

def test_training_pipeline_completes():
    """Test full training pipeline runs without errors"""
    config = {
        'epochs': 2,
        'batch_size': 16,
        'model_type': 'lightweight_cnn'
    }
    
    model, metrics = train_model(config)
    
    assert model is not None
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0.0
```

### Model Tests
Validate model behavior
```python
# tests/test_model_inference.py
import numpy as np
from src.models.model_loader import load_model

def test_model_prediction_shape():
    """Test model output shape is correct"""
    model = load_model("artifacts/models/best_model.h5")
    sample_input = np.random.rand(1, 13, 215, 1)
    
    prediction = model.predict(sample_input)
    
    assert prediction.shape == (1, 1), "Should output single probability"
    assert 0 <= prediction[0][0] <= 1, "Probability should be between 0 and 1"

def test_model_performance_threshold():
    """Test model meets minimum accuracy threshold"""
    from src.evaluation.evaluator import evaluate_model
    
    metrics = evaluate_model("artifacts/models/best_model.h5", "data/processed/test")
    
    assert metrics['accuracy'] >= 0.75, "Model accuracy should be at least 75%"
```

## Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_extraction.py

# Run tests matching pattern
pytest -k "test_mfcc"

# Run with verbose output
pytest -v

# Run and stop at first failure
pytest -x
```

## CI/CD Integration
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test-Driven Development (TDD)
1. Write failing test first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Repeat

This ensures comprehensive test coverage and better design.
