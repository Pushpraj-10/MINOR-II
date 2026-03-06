"""File and path utility functions."""

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_all_files(
    directory: str,
    extensions: List[str],
    recursive: bool = False,
) -> List[Path]:
    """
    Get all files with given extensions from a directory.

    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.wav', '.mp3'])
        recursive: Whether to search subdirectories

    Returns:
        Sorted list of matching file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = []
    for ext in extensions:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(dir_path.glob(pattern))

    return sorted(files)
