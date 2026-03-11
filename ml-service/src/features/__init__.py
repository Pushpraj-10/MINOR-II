"""Feature extraction modules for audio processing."""
from src.features.augmentation import augment_minority_class, augment_segment

__all__ = ["augment_minority_class", "augment_segment"]
