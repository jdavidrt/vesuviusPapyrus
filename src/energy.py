"""MRF energy as a convergence diagnostic."""

from __future__ import annotations

import numpy as np


def mrf_energy(
    state: np.ndarray,
    log_unary_noink: np.ndarray,
    log_unary_ink: np.ndarray,
    beta: float,
) -> float:
    """Total MRF energy for the current state.

    E(s) = -sum_ij log P(I_ij | s_ij) - beta * sum_{<ij,kl>} 1[s_ij = s_kl]

    Adjacent pairs in the 4-neighbourhood are counted exactly once by summing
    only over right-neighbour and bottom-neighbour pairs.
    """
    if state.shape != log_unary_noink.shape or state.shape != log_unary_ink.shape:
        raise ValueError("state and log-unary arrays must share the same shape.")
    s = state.astype(bool)
    # Unary term: pick log-unary by current label per pixel.
    unary = np.where(s, log_unary_ink, log_unary_noink)
    unary_sum = float(unary.sum())

    # Pairwise term: agreements with right neighbour and bottom neighbour.
    right_agree = (state[:, :-1] == state[:, 1:]).sum()
    down_agree = (state[:-1, :] == state[1:, :]).sum()
    pairwise_sum = float(right_agree + down_agree)

    return -unary_sum - beta * pairwise_sum
