"""Maximum likelihood estimation of per-class Gaussian parameters."""

from __future__ import annotations

import numpy as np

from .distributions import gaussian_log_pdf

_MIN_SIGMA = 1e-6


def _sample_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Sample mean and unbiased standard deviation (Bessel correction)."""
    if values.size < 2:
        raise ValueError(
            "Need at least 2 samples to estimate mean and std with Bessel correction."
        )
    mu = float(values.mean())
    # Unbiased estimator: denominator n - 1.
    diff = values.astype(np.float64) - mu
    var = float((diff * diff).sum()) / (values.size - 1)
    sigma = float(np.sqrt(var))
    if sigma < _MIN_SIGMA:
        sigma = _MIN_SIGMA
    return mu, sigma


def estimate_class_params(intensities: np.ndarray, labels: np.ndarray) -> dict:
    """Estimate per-class Gaussian parameters from training intensities and labels.

    Returns a dict with keys 'mu_ink', 'sigma_ink', 'mu_noink', 'sigma_noink'.
    Uses the sample mean and sample standard deviation with Bessel correction.
    """
    if intensities.shape != labels.shape:
        raise ValueError(
            f"intensities {intensities.shape} and labels {labels.shape} must match."
        )
    flat_x = intensities.ravel().astype(np.float64)
    flat_y = labels.ravel().astype(np.int64)
    ink_mask = flat_y == 1
    noink_mask = flat_y == 0
    if not ink_mask.any():
        raise ValueError("Training set contains no ink pixels (label = 1).")
    if not noink_mask.any():
        raise ValueError("Training set contains no non-ink pixels (label = 0).")
    mu_ink, sigma_ink = _sample_mean_std(flat_x[ink_mask])
    mu_noink, sigma_noink = _sample_mean_std(flat_x[noink_mask])
    return {
        "mu_ink": mu_ink,
        "sigma_ink": sigma_ink,
        "mu_noink": mu_noink,
        "sigma_noink": sigma_noink,
    }


def precompute_log_unaries(
    image: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_unary_noink, log_unary_ink), each shape (H, W).

    Each entry is log P(I_ij | class) using the per-class Gaussian model.
    Pre-computing once amortises the cost across all sweeps.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")
    log_unary_noink = gaussian_log_pdf(image, params["mu_noink"], params["sigma_noink"])
    log_unary_ink = gaussian_log_pdf(image, params["mu_ink"], params["sigma_ink"])
    return log_unary_noink, log_unary_ink
