# `src/utils/` Directory

## Purpose
This module contains shared utility functions and helper classes used across the entire project, providing common functionality like logging, configuration loading, file I/O, and data transformations.

## What Belongs Here
- **Configuration utilities**: Config loading, validation, merging
- **Logging utilities**: Logger setup, custom formatters
- **File I/O helpers**: Path handling, file operations
- **Common transformations**: Reusable data transformations
- **Timing utilities**: Performance measurement, profiling
- **Validation helpers**: Input validation, type checking
- **Constants**: Project-wide constants and enumerations

## What Should NOT Be Here
- ❌ Domain-specific logic (belongs in dedicated modules)
- ❌ Model architectures (belongs in `src/models/`)
- ❌ Feature extraction (belongs in `src/features/`)
- ❌ Large, complex classes (consider dedicated module)

## Architectural Responsibilities
- **Code reuse**: Eliminate duplication across modules
- **Consistency**: Standardize common operations
- **Simplification**: Abstract complex operations into simple functions
- **Maintainability**: Centralize utility code for easier updates

## Typical Modules

### `config_loader.py`
Configuration file loading and validation
```python
def load_config(config_path: str) -> Dict:
    """Load YAML/JSON configuration file."""
    pass

def validate_config(config: Dict, schema: Dict) -> None:
    """Validate configuration against schema."""
    pass

def merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge two configurations with override priority."""
    pass
```

### `logger.py`
Logging setup and utilities
```python
def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup configured logger."""
    pass

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""
    pass
```

### `file_utils.py`
File and path operations
```python
def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    pass

def get_all_files(directory: str, extension: str) -> List[Path]:
    """Get all files with given extension."""
    pass

def safe_file_name(name: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    pass
```

### `timer.py`
Performance timing utilities
```python
class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass

def profile_function(func):
    """Decorator to profile function execution time."""
    pass
```

### `validators.py`
Input validation utilities
```python
def validate_audio_path(path: str) -> bool:
    """Validate audio file path exists and has correct extension."""
    pass

def validate_array_shape(
    array: np.ndarray,
    expected_shape: Tuple
) -> None:
    """Validate numpy array has expected shape."""
    pass
```

### `constants.py`
Project-wide constants
```python
# Audio processing constants
SAMPLE_RATE = 16000
DURATION = 5.0
N_MFCC = 13

# Model constants
INPUT_SHAPE = (13, 215, 1)
NUM_CLASSES = 1

# File extensions
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac']
```

## Interactions with Other Modules
- **All modules in `src/`**: Import and use utility functions
- **`config/`**: Load configuration files from here
- **Notebooks**: Import utilities for consistent experimentation

## Best Practices
1. **Pure functions**: Utilities should be stateless when possible
2. **Single responsibility**: Each function does one thing well
3. **Documentation**: Clear docstrings with examples
4. **Error handling**: Validate inputs and provide helpful error messages
5. **Type hints**: Use type annotations for clarity
6. **Testing**: Unit test all utility functions
7. **No side effects**: Avoid modifying global state
8. **Performance**: Keep utilities efficient and lightweight

## Example Implementation
```python
# src/utils/config_loader.py
"""Configuration loading and validation utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format not supported
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on extension
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    
    Values in override take precedence over base.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict, required_keys: list) -> None:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports nested with '.')
        
    Raises:
        ValueError: If required key is missing
    """
    for key in required_keys:
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[k]
        else:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
    
    logger.debug(f"Configuration validated: {len(required_keys)} required keys present")
```

## Logger Utility
```python
# src/utils/logger.py
"""Logging utilities for the ML project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for terminal."""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_colors: bool = True
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        use_colors: Whether to use colored output for console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors:
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
```

## File Utilities
```python
# src/utils/file_utils.py
"""File and path utility functions."""

import os
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return dir_path


def get_all_files(
    directory: str,
    extension: str = None,
    recursive: bool = False
) -> List[Path]:
    """
    Get all files in directory, optionally filtered by extension.
    
    Args:
        directory: Directory to search
        extension: File extension to filter (e.g., '.wav')
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        pattern = f"**/*{extension}" if extension else "**/*"
    else:
        pattern = f"*{extension}" if extension else "*"
    
    files = [f for f in dir_path.glob(pattern) if f.is_file()]
    
    logger.debug(f"Found {len(files)} files in {directory}")
    
    return files


def safe_file_name(name: str) -> str:
    """
    Sanitize filename for cross-platform compatibility.
    
    Args:
        name: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove/replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Trim whitespace
    name = name.strip()
    
    # Ensure not empty
    if not name:
        name = 'unnamed'
    
    return name


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    size_bytes = Path(file_path).stat().st_size
    return size_bytes / (1024 * 1024)
```

## Timer Utility
```python
# src/utils/timer.py
"""Performance timing utilities."""

import time
import logging
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        with Timer("Processing audio"):
            # ... code to time ...
    """
    
    def __init__(self, description: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            description: Description of timed operation
        """
        self.description = description
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        logger.debug(f"{self.description} started")
        return self
    
    def __exit__(self, *args):
        """Stop timer and log duration."""
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {self.elapsed:.2f}s")


def profile_function(func):
    """
    Decorator to profile function execution time.
    
    Example:
        @profile_function
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper
```

## Testing Utilities
```python
# tests/test_utils.py
import pytest
from src.utils.config_loader import load_config, merge_configs
from src.utils.file_utils import safe_file_name

def test_load_config(tmp_path):
    """Test configuration loading."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("model_type: cnn\nepochs: 10")
    
    config = load_config(str(config_file))
    assert config['model_type'] == 'cnn'
    assert config['epochs'] == 10

def test_merge_configs():
    """Test configuration merging."""
    base = {'a': 1, 'b': {'c': 2}}
    override = {'b': {'c': 3, 'd': 4}}
    
    merged = merge_configs(base, override)
    assert merged['a'] == 1
    assert merged['b']['c'] == 3
    assert merged['b']['d'] == 4

def test_safe_file_name():
    """Test filename sanitization."""
    assert safe_file_name("file<>name.txt") == "file__name.txt"
    assert safe_file_name("  trimmed  ") == "trimmed"
```
