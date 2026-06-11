# EVOLVE-BLOCK-START
"""Initial search baseline for the AlphaEvolve AC inequality task."""

import time
import numpy as np


def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluate coefficient sequence with safety checks.
    Lower is better.
    """
    if not isinstance(sequence, list):
        return float(np.inf)
    if not sequence:
        return float(np.inf)

    clean: list[float] = []
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return float(np.inf)
        if np.isnan(x) or np.isinf(x):
            return float(np.inf)
        clean.append(float(x))

    clean = [max(0.0, min(1000.0, x)) for x in clean]
    n = len(clean)
    conv = np.convolve(clean, clean)
    max_b = float(np.max(conv))
    sum_a = float(np.sum(clean))
    if sum_a < 0.01:
        return float(np.inf)
    return float(2.0 * n * max_b / (sum_a**2))


def _normalize_scale(seq: np.ndarray) -> np.ndarray:
    """Scale sequence to stable magnitude while keeping nonnegative shape."""
    s = np.asarray(seq, dtype=np.float64)
    s = np.clip(s, 0.0, None)
    total = float(np.sum(s))
    if total <= 1e-12:
        return np.ones_like(s)
    # objective is scale-invariant theoretically, but this keeps numerics stable.
    return s / total * len(s)


def _seed_sequence(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    base = (
        0.9
        + 0.30 * np.sin(2.0 * np.pi * x)
        + 0.12 * np.sin(6.0 * np.pi * x + rng.uniform(0.0, 2.0 * np.pi))
    )
    noise = rng.normal(0.0, 0.08, n)
    return _normalize_scale(base + noise)


def run(seed: int = 42, budget_s: float = 10.0, **kwargs) -> list[float]:
    """
    Search for a low-value nonnegative sequence.

    Returns:
        list[float]: candidate coefficient sequence
    """
    del kwargs

    rng = np.random.default_rng(seed)
    start = time.time()
    deadline = start + max(0.1, budget_s * 0.98)

    # Multi-start initial pool.
    candidate_lengths = [128, 192, 256, 320, 384]
    best_seq = _seed_sequence(rng, n=256)
    best_val = evaluate_sequence(best_seq.tolist())

    for n in candidate_lengths:
        seq = _seed_sequence(rng, n=n)
        val = evaluate_sequence(seq.tolist())
        if val < best_val:
            best_seq, best_val = seq, val

    # Local stochastic search: block perturbation + smoothing.
    while time.time() < deadline:
        seq = best_seq.copy()
        n = len(seq)

        block = int(rng.integers(max(4, n // 64), max(8, n // 10)))
        idx = rng.choice(n, size=block, replace=False)
        seq[idx] += rng.normal(0.0, 0.10, size=block)

        if rng.random() < 0.35:
            # low-pass smooth
            seq = (
                0.25 * np.roll(seq, 1)
                + 0.50 * seq
                + 0.25 * np.roll(seq, -1)
            )

        if rng.random() < 0.15 and n < 800:
            # upsample once in a while to explore larger dimensions
            seq = np.repeat(seq, 2)[: min(800, 2 * n)]

        seq = _normalize_scale(seq)
        val = evaluate_sequence(seq.tolist())
        if val < best_val:
            best_seq, best_val = seq, val

    return [float(x) for x in best_seq.tolist()]


# EVOLVE-BLOCK-END
