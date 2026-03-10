"""Evaluation modules for model assessment."""
from src.evaluation.evaluator import (
    evaluate_model,
    print_split_summary,
    evaluate_tflite_on_splits,
)

__all__ = [
    "evaluate_model",
    "print_split_summary",
    "evaluate_tflite_on_splits",
]
