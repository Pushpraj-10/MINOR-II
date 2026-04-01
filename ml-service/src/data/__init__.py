"""Data modules for loading, validating, and preprocessing audio data.

Imports are lazy to allow PyTorch-only (HLG-Net) code to work
without requiring tensorflow.
"""


def __getattr__(name):
    if name == "AudioDataLoader":
        from src.data.loader import AudioDataLoader
        return AudioDataLoader
    if name == "split_dataset":
        from src.data.splitter import split_dataset
        return split_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AudioDataLoader", "split_dataset"]
