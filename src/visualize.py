"""Plotting helpers (matplotlib only)."""

from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # No GUI; we only write PNGs.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _flatten_energy_trace(history: Iterable) -> tuple[list[int], list[float], int]:
    """Turn the per-sweep history into x / y arrays plus the burn-in cutoff.

    The history is a list of ``(phase, sweep, energy)`` tuples. Burn-in and
    sampling phases are concatenated; the cutoff is the total burn-in sweep
    count (so the sampling phase starts at ``x = burn_count + 1``).
    """
    burn_x: list[int] = []
    burn_y: list[float] = []
    sample_x: list[int] = []
    sample_y: list[float] = []
    burn_max = 0
    for row in history:
        phase, sweep, energy = row
        if phase == "burn":
            burn_x.append(int(sweep))
            burn_y.append(float(energy))
            if int(sweep) > burn_max:
                burn_max = int(sweep)
        else:
            sample_x.append(int(sweep))
            sample_y.append(float(energy))
    # Offset sampling sweeps so the trace plots as one continuous timeline.
    sample_x_global = [burn_max + s for s in sample_x]
    return (burn_x + sample_x_global, burn_y + sample_y, burn_max)


def plot_energy(history: Iterable, path: str) -> None:
    """Line plot of MRF energy vs sweep, with a marker at end of burn-in."""
    xs, ys, burn_cut = _flatten_energy_trace(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, lw=1.2)
    if burn_cut > 0:
        ax.axvline(burn_cut, color="red", lw=1.0, linestyle="--", label="end of burn-in")
        ax.legend(loc="best")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("MRF energy")
    ax.set_title("Energy trace")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(
    ct: np.ndarray,
    ground_truth: np.ndarray,
    predictions: dict,
    path: str,
) -> None:
    """Grid of CT, ground truth, and one prediction per beta value."""
    if ct.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("ct and ground_truth must be 2D arrays.")
    keys = sorted(predictions.keys(), key=lambda k: float(k))
    n_cols = 2 + len(keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(ct, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    for i, key in enumerate(keys):
        ax = axes[2 + i]
        ax.imshow(predictions[key], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"beta = {key}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
