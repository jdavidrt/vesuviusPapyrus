"""Probability distributions implemented from scratch."""

from __future__ import annotations

import math

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Numerically stable log of Gaussian PDF, evaluated elementwise.

    Implements log N(x; mu, sigma^2) = -0.5 log(2 pi sigma^2) - (x - mu)^2 / (2 sigma^2).
    Working in log space avoids the underflow that linear-space Gaussians hit
    on typical CT intensity scales.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}.")
    var = sigma * sigma
    diff = np.asarray(x, dtype=np.float64) - float(mu)
    return -0.5 * (_LOG_2PI + math.log(var)) - (diff * diff) / (2.0 * var)
