# EVOLVE-BLOCK-START
"""Initial construction for the Erdos minimum overlap task."""

import time
import numpy as np


def _project_to_feasible(h_values: np.ndarray) -> np.ndarray:
    """Project h onto [0,1]^n with sum(h)=n/2."""
    h = np.asarray(h_values, dtype=np.float64).copy()
    n_points = h.size
    target_sum = n_points / 2.0

    h = np.clip(h, 0.0, 1.0)
    current_sum = float(np.sum(h))
    if current_sum <= 1e-12:
        return np.full(n_points, 0.5, dtype=np.float64)

    h *= target_sum / current_sum

    # Iteratively fix clipping violations while preserving the target sum.
    for _ in range(12):
        h = np.clip(h, 0.0, 1.0)
        delta = target_sum - float(np.sum(h))
        if abs(delta) < 1e-10:
            break
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if not np.any(free):
            break
        h[free] += delta / float(np.sum(free))

    h = np.clip(h, 0.0, 1.0)
    # Final correction (numerical): distribute tiny residual to non-saturated indices.
    residual = target_sum - float(np.sum(h))
    if abs(residual) > 1e-9:
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += residual / float(np.sum(free))
            h = np.clip(h, 0.0, 1.0)
    return h


def _compute_c5(h_values: np.ndarray) -> float:
    n_points = h_values.size
    dx = 2.0 / n_points
    overlap = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
    return float(np.max(overlap))


def run(seed: int = 42, budget_s: float = 10.0, **kwargs):
    """
    Return (h_values, c5_bound, n_points) for Erdos minimum overlap.

    Lower c5_bound is better.
    """
    del kwargs

    start = time.time()
    rng = np.random.default_rng(seed)
    n_points = 400

    x = np.linspace(0.0, 2.0, n_points, endpoint=False)
    base = 0.5 + 0.18 * np.sin(2.0 * np.pi * x / 2.0) - 0.10 * np.sin(
        6.0 * np.pi * x / 2.0
    )
    best_h = _project_to_feasible(base)
    best_c5 = _compute_c5(best_h)

    # Lightweight randomized local search.
    while time.time() - start < max(0.1, 0.95 * budget_s):
        candidate = best_h.copy()
        block_size = int(rng.integers(8, 40))
        idx = rng.choice(n_points, size=block_size, replace=False)
        candidate[idx] += rng.normal(0.0, 0.06, size=block_size)
        candidate = _project_to_feasible(candidate)
        candidate_c5 = _compute_c5(candidate)
        if candidate_c5 < best_c5:
            best_h = candidate
            best_c5 = candidate_c5

    return best_h, float(best_c5), int(n_points)


# EVOLVE-BLOCK-END
