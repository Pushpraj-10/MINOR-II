"""Utility modules for the depression detection project.

Imports are lazy to avoid requiring tensorflow when only file_utils is needed.
"""


def __getattr__(name):
    if name in ("ensure_dir", "get_all_files"):
        from src.utils.file_utils import ensure_dir, get_all_files
        return {"ensure_dir": ensure_dir, "get_all_files": get_all_files}[name]
    if name == "focal_loss":
        from src.utils.focal_loss import focal_loss
        return focal_loss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ensure_dir", "get_all_files", "focal_loss"]
