"""Gibbs sampler over the 2D Ising-like Markov Random Field."""

from __future__ import annotations

import time

import numpy as np

from .energy import mrf_energy
from .learning import precompute_log_unaries


def count_neighbors_equal(state: np.ndarray, value: int) -> np.ndarray:
    """For each pixel, count 4-neighbours equal to ``value``.

    Boundary pixels naturally have fewer than four neighbours; we do not
    pad. The output is an int32 array of the same shape as ``state``.
    """
    if state.ndim != 2:
        raise ValueError(f"state must be 2D, got shape {state.shape}.")
    match = (state == value).astype(np.int32)
    counts = np.zeros_like(match)
    counts[1:, :] += match[:-1, :]   # neighbour above
    counts[:-1, :] += match[1:, :]   # neighbour below
    counts[:, 1:] += match[:, :-1]   # neighbour left
    counts[:, :-1] += match[:, 1:]   # neighbour right
    return counts


def _neighbour_totals(shape: tuple[int, int]) -> np.ndarray:
    """Total number of in-bounds 4-neighbours for each pixel (2..4)."""
    H, W = shape
    n = np.zeros((H, W), dtype=np.int32)
    n[1:, :] += 1
    n[:-1, :] += 1
    n[:, 1:] += 1
    n[:, :-1] += 1
    return n


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arbitrarily large |x|."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def gibbs_sweep(
    state: np.ndarray,
    log_unary_noink: np.ndarray,
    log_unary_ink: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> None:
    """One full sweep updating every pixel. Mutates ``state`` in place.

    Uses a checkerboard schedule: pixels with (i + j) even form one
    conditionally independent set and are updated together, then the
    odd-parity pixels are updated. Both halves use freshly drawn uniforms.
    This preserves the Gibbs invariant distribution because each set is
    conditionally independent given the other.
    """
    if state.shape != log_unary_noink.shape or state.shape != log_unary_ink.shape:
        raise ValueError("state and log-unary arrays must share the same shape.")
    H, W = state.shape
    n_total = _neighbour_totals((H, W))
    delta_log_unary = log_unary_ink - log_unary_noink

    I, J = np.indices((H, W))
    parity = (I + J) % 2
    masks = (parity == 0, parity == 1)

    for active_mask in masks:
        n_ink = count_neighbors_equal(state, 1)
        # delta = log p(s=1 | blanket) - log p(s=0 | blanket)
        #       = log_unary_ink - log_unary_noink + beta * (n_ink - n_noink)
        # with n_noink = n_total - n_ink, so n_ink - n_noink = 2 * n_ink - n_total.
        delta = delta_log_unary + beta * (2.0 * n_ink - n_total)
        p1 = _stable_sigmoid(delta)
        u = rng.random((H, W))
        new_vals = (u < p1).astype(np.uint8)
        state[active_mask] = new_vals[active_mask]


def run_gibbs(
    image: np.ndarray,
    params: dict,
    beta: float,
    n_burn: int,
    n_sample: int,
    seed: int,
    energy_log_every: int = 10,
) -> dict:
    """Run the full Gibbs sampler with burn-in and sampling phases.

    Returns a dict with keys:
        * ``posterior``: (H, W) float64 array of marginal P(HasInk=1).
        * ``final_mask``: (H, W) uint8 array (posterior > 0.5).
        * ``energy_history``: list of ``(phase, sweep_index, energy)`` tuples.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}.")
    if n_burn < 0 or n_sample <= 0:
        raise ValueError("n_burn must be >= 0 and n_sample must be >= 1.")
    if energy_log_every <= 0:
        raise ValueError("energy_log_every must be >= 1.")

    log_unary_noink, log_unary_ink = precompute_log_unaries(image, params)
    H, W = image.shape
    state = np.zeros((H, W), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    history: list[tuple[str, int, float]] = []
    posterior_sum = np.zeros((H, W), dtype=np.float64)

    def _record(phase: str, sweep_idx: int) -> None:
        e = mrf_energy(state, log_unary_noink, log_unary_ink, beta)
        history.append((phase, sweep_idx, e))

    # Burn-in.
    print(f"[gibbs] burn-in: {n_burn} sweeps (beta={beta})")
    t0 = time.time()
    for sweep in range(1, n_burn + 1):
        gibbs_sweep(state, log_unary_noink, log_unary_ink, beta, rng)
        if sweep % energy_log_every == 0 or sweep == n_burn:
            _record("burn", sweep)
        if sweep % 50 == 0:
            print(
                f"[gibbs] burn sweep {sweep}/{n_burn} "
                f"energy={history[-1][2]:.2f} elapsed={time.time() - t0:.1f}s"
            )

    # Sampling.
    print(f"[gibbs] sampling: {n_sample} sweeps")
    for sweep in range(1, n_sample + 1):
        gibbs_sweep(state, log_unary_noink, log_unary_ink, beta, rng)
        posterior_sum += state
        if sweep % energy_log_every == 0 or sweep == n_sample:
            _record("sample", sweep)
        if sweep % 50 == 0:
            print(
                f"[gibbs] sample sweep {sweep}/{n_sample} "
                f"energy={history[-1][2]:.2f} elapsed={time.time() - t0:.1f}s"
            )

    posterior = posterior_sum / float(n_sample)
    final_mask = (posterior > 0.5).astype(np.uint8)
    return {
        "posterior": posterior,
        "final_mask": final_mask,
        "energy_history": history,
    }
