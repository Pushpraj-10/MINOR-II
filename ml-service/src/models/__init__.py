"""Model architecture modules."""
from src.models.architectures import MODEL_REGISTRY, get_model, list_architectures

__all__ = ["MODEL_REGISTRY", "get_model", "list_architectures"]
