"""Model architecture modules.

TF/Keras architectures are imported lazily to avoid requiring tensorflow
when only PyTorch (HLG-Net) is needed.
"""


def get_model(*args, **kwargs):
    from src.models.architectures import get_model as _get_model
    return _get_model(*args, **kwargs)


def list_architectures():
    from src.models.architectures import list_architectures as _list
    return _list()


def get_model_registry():
    from src.models.architectures import MODEL_REGISTRY
    return MODEL_REGISTRY


__all__ = ["get_model", "list_architectures", "get_model_registry"]
