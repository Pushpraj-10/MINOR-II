"""Pipeline modules for orchestrating ML workflows."""
from src.pipelines.evaluation_pipeline import load_audio_splits, run_evaluation_pipeline

__all__ = ["load_audio_splits", "run_evaluation_pipeline"]
