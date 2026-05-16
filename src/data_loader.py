"""Data loading, cropping, and train/test splitting."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_ct_image(path: str) -> np.ndarray:
    """Load a 16-bit TIF as float64 normalized to [0, 1].

    The CT slice is expected to be a single-channel image. If the source has
    multiple channels, only the first one is kept.
    """
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        denom = 65535.0
    elif arr.dtype == np.uint8:
        denom = 255.0
    else:
        # Fall back to dynamic range; avoids zero-division on flat inputs.
        arr = arr.astype(np.float64)
        amax = float(arr.max()) if arr.size else 1.0
        denom = amax if amax > 0.0 else 1.0
        return np.clip(arr / denom, 0.0, 1.0)
    return (arr.astype(np.float64) / denom).clip(0.0, 1.0)


def load_ink_mask(path: str) -> np.ndarray:
    """Load a binary PNG as uint8 array with values in {0, 1}."""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Source masks may be encoded as either {0, 1} (palette PNG) or {0, 255}
    # (8-bit grayscale). Any strictly positive pixel counts as ink.
    return (arr > 0).astype(np.uint8)


def crop_region(arr: np.ndarray, top: int, left: int, h: int, w: int) -> np.ndarray:
    """Return a (h, w) crop starting at (top, left)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    H, W = arr.shape
    if top < 0 or left < 0 or h <= 0 or w <= 0:
        raise ValueError("Crop indices and sizes must be non-negative and positive.")
    if top + h > H or left + w > W:
        raise ValueError(
            f"Crop ({top}:{top+h}, {left}:{left+w}) exceeds image of shape {arr.shape}."
        )
    return arr[top:top + h, left:left + w].copy()


def spatial_split(crop_h: int) -> tuple[slice, slice]:
    """Return (train_rows, test_rows) as slice objects splitting horizontally.

    Top half is training, bottom half is testing. Odd heights leave the extra
    row in the training set.
    """
    if crop_h <= 1:
        raise ValueError("crop_h must be at least 2 to split into two parts.")
    split = (crop_h + 1) // 2
    return slice(0, split), slice(split, crop_h)
