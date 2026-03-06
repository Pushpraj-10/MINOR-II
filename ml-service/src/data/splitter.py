"""Data splitting utilities with stratification."""

import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train, validation, and test sets with stratification.

    Args:
        X: Features array
        y: Labels array
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' keys,
        each containing (X, y) tuple
    """
    # First split: separate test set
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    # Second split: separate val from temp
    val_proportion = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion),
        random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Split dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    logger.info(
        f"Train distribution: depression={np.sum(y_train==1)}, normal={np.sum(y_train==0)}"
    )
    logger.info(
        f"Val distribution:   depression={np.sum(y_val==1)}, normal={np.sum(y_val==0)}"
    )
    logger.info(
        f"Test distribution:  depression={np.sum(y_test==1)}, normal={np.sum(y_test==0)}"
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
