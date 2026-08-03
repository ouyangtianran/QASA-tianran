"""Checkpoint loading helpers, including compatibility with older environments."""

import sys

import numpy as np
import torch


def _install_numpy_pickle_aliases():
    """Allow NumPy 2.x checkpoints to be read by NumPy 1.x."""
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


def load_checkpoint(path, map_location="cpu"):
    _install_numpy_pickle_aliases()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint
