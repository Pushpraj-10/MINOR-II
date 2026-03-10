"""Pipeline modules for orchestrating ML workflows."""
from src.pipelines.training_pipeline import run_training_pipeline
from src.pipelines.evaluation_pipeline import load_audio_splits, run_evaluation_pipeline

__all__ = [
    "run_training_pipeline",
    "load_audio_splits",
    "run_evaluation_pipeline",
]
