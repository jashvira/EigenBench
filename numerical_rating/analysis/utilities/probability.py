"""Probability transforms shared by the comparison models."""

from __future__ import annotations

import numpy as np


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Apply a stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
