"""Numerical-rating standardization and EigenTrust aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrustResult:
    """Intermediate matrices and final trust from a numerical-rating tensor."""

    scenario_count: int
    sigma: np.ndarray
    sigma_used: np.ndarray
    affinity: np.ndarray
    trust_matrix: np.ndarray
    one_step_trust: np.ndarray
    trust: np.ndarray
    iterations: int | None
    stationary_solver: str


def softmax_rows(
    values: np.ndarray, temperature: float, zero_diagonal: bool
) -> np.ndarray:
    """Convert judge-target affinities into a row-stochastic trust matrix."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = values / temperature
    if zero_diagonal:
        logits = logits.copy()
        np.fill_diagonal(logits, -np.inf)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def stationary_trust(
    matrix: np.ndarray, tolerance: float = 1e-12
) -> tuple[np.ndarray, int | None, str]:
    """Return the stationary left eigenvector of a trust matrix."""
    trust = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for iteration in range(1, 10_001):
        next_trust = trust @ matrix
        if np.linalg.norm(next_trust - trust, ord=1) < tolerance:
            return next_trust, iteration, "power_iteration"
        trust = next_trust

    system = matrix.T - np.eye(matrix.shape[0])
    system[-1] = 1.0
    target = np.zeros(matrix.shape[0])
    target[-1] = 1.0
    solved = np.linalg.solve(system, target)
    if np.any(solved < -1e-10):
        raise RuntimeError("Stationary linear solve produced negative trust")
    solved = np.clip(solved, 0.0, None)
    solved /= solved.sum()
    return solved, None, "linear_solve"


def compute_trust(
    ratings: np.ndarray,
    *,
    temperature: float = 1.0,
    zero_diagonal: bool = False,
    standardize: bool = True,
    sigma_floor_ratio: float | None = None,
) -> TrustResult:
    """Standardize judge ratings and propagate the resulting EigenTrust."""
    judges, scenarios, models = ratings.shape
    if judges != models:
        raise ValueError(
            "EigenTrust requires the same judge and target population size"
        )

    if zero_diagonal:
        if models <= 2:
            raise ValueError("Self-score exclusion requires at least three models")
        off_diagonal = ~np.eye(models, dtype=bool)
        mask = off_diagonal[:, None, :]
        off_diagonal_mean = np.where(mask, ratings, 0.0).sum(axis=2, keepdims=True) / (
            models - 1
        )
        centered = np.where(mask, ratings - off_diagonal_mean, 0.0)
        variance_denominator = scenarios * (models - 2)
    else:
        centered = ratings - ratings.mean(axis=2, keepdims=True)
        variance_denominator = scenarios * (models - 1)
    if not np.allclose(centered.sum(axis=2), 0.0, atol=1e-10):
        raise AssertionError("Scenario-centred ratings do not sum to zero")
    sigma = np.sqrt(np.square(centered).sum(axis=(1, 2)) / variance_denominator)
    if np.any(sigma == 0):
        raise ValueError("At least one judge has zero residual score variance")

    sigma_used = sigma.copy() if standardize else np.ones_like(sigma)
    if standardize and sigma_floor_ratio is not None:
        floor = sigma_floor_ratio * float(np.median(sigma))
        sigma_used = np.maximum(sigma_used, floor)

    standardized = centered / sigma_used[:, None, None]
    affinity = standardized.mean(axis=1)
    if zero_diagonal:
        np.fill_diagonal(affinity, 0.0)
    if not np.allclose(affinity.sum(axis=1), 0.0, atol=1e-10):
        raise AssertionError("Judge affinity rows do not sum to zero")
    trust_matrix = softmax_rows(affinity, temperature, zero_diagonal)
    if not np.allclose(trust_matrix.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Trust matrix rows do not sum to one")
    one_step = np.full(judges, 1.0 / judges) @ trust_matrix
    trust, iterations, stationary_solver = stationary_trust(trust_matrix)
    if np.linalg.norm(trust @ trust_matrix - trust, ord=1) > 1e-10:
        raise AssertionError("Stationary trust vector failed its residual check")
    return TrustResult(
        scenario_count=scenarios,
        sigma=sigma,
        sigma_used=sigma_used,
        affinity=affinity,
        trust_matrix=trust_matrix,
        one_step_trust=one_step,
        trust=trust,
        iterations=iterations,
        stationary_solver=stationary_solver,
    )


def eigentrust_elo(trust: np.ndarray) -> np.ndarray:
    """Convert trust shares to EigenBench's display Elo scale."""
    return 1500.0 + 400.0 * np.log10(len(trust) * np.clip(trust, 1e-12, None))
