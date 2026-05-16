"""Lightweight artefact serialization helpers."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable

import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    """Create directory ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save a binary mask as an 8-bit PNG (0 / 255)."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}.")
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def save_posterior_png(posterior: np.ndarray, path: str) -> None:
    """Save a posterior probability map as an 8-bit grayscale PNG."""
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2D, got shape {posterior.shape}.")
    arr = np.clip(np.asarray(posterior, dtype=np.float64), 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_energy_csv(history: Iterable, path: str) -> None:
    """Persist the energy trace as a 3-column CSV: phase, sweep, energy."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phase", "sweep", "energy"])
        for row in history:
            if len(row) != 3:
                raise ValueError(
                    f"energy_history rows must be 3-tuples (phase, sweep, energy); got {row!r}."
                )
            phase, sweep, energy = row
            writer.writerow([phase, int(sweep), float(energy)])


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to plain Python."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data: dict, path: str) -> None:
    """Persist a dict as pretty-printed JSON, converting NumPy types."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, indent=2, sort_keys=True)
        fh.write("\n")
