"""Training utilities: seeding, checkpointing, running meters."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: The random seed.
        deterministic: If True, request deterministic cuDNN behavior (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a checkpoint dict, creating parent directories as needed.

    Args:
        state: Serializable dict (model/optimizer state, metadata, ...).
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict from disk.

    Args:
        path: Checkpoint file path.
        map_location: Device to map tensors onto.

    Returns:
        The loaded checkpoint dict.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    logger.info("Loaded checkpoint from %s", path)
    return ckpt


class AverageMeter:
    """Tracks the running average of a scalar value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Add ``val`` observed over ``n`` samples and refresh the average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0
