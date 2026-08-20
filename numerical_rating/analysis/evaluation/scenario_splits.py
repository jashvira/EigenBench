"""Deterministic held-out scenario splits."""

from __future__ import annotations

import numpy as np


def scenario_folds(count: int, folds: int, seed: int) -> list[np.ndarray]:
    """Create deterministic near-equal held-out scenario folds."""
    if not 2 <= folds <= count:
        raise ValueError("folds must be between 2 and the scenario count")
    permutation = np.random.default_rng(seed).permutation(count)
    return [
        np.asarray(fold, dtype=np.int64) for fold in np.array_split(permutation, folds)
    ]
