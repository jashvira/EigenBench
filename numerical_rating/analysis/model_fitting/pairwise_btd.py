"""Deterministic pooled Davidson BTD fitting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    loss_gradient as davidson_loss_gradient,
)
from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    unpack_parameters,
)
from numerical_rating.analysis.utilities.optimization import minimize_lbfgs
from pipeline.trust import compute_trust_matrix_ties_from_logits, eigentrust


@dataclass(frozen=True)
class PairwiseFit:
    """A fitted Davidson model and its propagated trust."""

    parameters: np.ndarray
    trust: np.ndarray
    loss: float
    iterations: int
    attempted_starts: int = 1
    successful_starts: int = 1
    best_start: int = 0
    near_optimal_starts: int = 1
    max_near_optimal_trust_l1: float = 0.0
    stable_near_optima: bool = True
    gradient_l2: float = float("nan")
    parameter_l2: float = float("nan")
    start_diagnostics: tuple[dict, ...] = ()
    failed_starts: tuple[str, ...] = ()


def unpack(parameters: np.ndarray, num_models: int, dimension: int):
    return unpack_parameters(
        parameters,
        num_rows=num_models,
        num_models=num_models,
        dimension=dimension,
        center_models=False,
    )


def btd_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return mean Davidson cross-entropy and its analytic gradient."""
    return davidson_loss_gradient(
        parameters,
        rows,
        num_rows=num_models,
        num_models=num_models,
        dimension=dimension,
        center_models=False,
    )


def pairwise_trust(
    parameters: np.ndarray, *, num_models: int, dimension: int
) -> np.ndarray:
    """Convert fitted SciPy parameters through EigenBench's trust path."""
    judges, targets, log_ties = unpack(parameters, num_models, dimension)
    trust_matrix = compute_trust_matrix_ties_from_logits(
        torch.from_numpy(judges @ targets.T),
        torch.from_numpy(log_ties),
        logit_clip=40.0,
    )
    trust = eigentrust(
        trust_matrix,
        alpha=0.0,
        tol=1e-12,
        max_iter=10_000,
        verbose=False,
        raise_on_nonconvergence=True,
    )
    return trust.numpy()


def fit_pairwise_btd(
    rows: np.ndarray,
    *,
    initial: np.ndarray,
    num_models: int = 8,
    dimension: int = 2,
    max_iterations: int = 1_000,
) -> PairwiseFit:
    """Fit the pooled Davidson likelihood deterministically."""
    objective = partial(
        btd_loss_gradient,
        num_models=num_models,
        dimension=dimension,
    )
    result, _ = minimize_lbfgs(
        objective,
        initial,
        args=(rows,),
        max_iterations=max_iterations,
        record_trace=False,
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Pairwise BTD optimization produced a non-finite result")
    if not result.success:
        raise RuntimeError(
            "Pairwise BTD optimization failed after "
            f"{result.nit} iterations: {result.message}"
        )
    return PairwiseFit(
        parameters=result.x,
        trust=pairwise_trust(
            result.x,
            num_models=num_models,
            dimension=dimension,
        ),
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        parameter_l2=float(np.linalg.norm(result.x)),
    )


def fit_pairwise_btd_multistart(
    rows: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_models: int = 8,
    dimension: int = 2,
    max_iterations: int = 1_000,
    near_loss_tolerance: float = 1e-7,
    trust_l1_tolerance: float = 1e-3,
    reject_unstable: bool = True,
) -> PairwiseFit:
    """Choose the best converged fit and detect incompatible near-optima."""
    if len(initials) < 2:
        raise ValueError("Multi-start BTD requires at least two initializations")
    fits: list[tuple[int, PairwiseFit]] = []
    errors: list[str] = []
    for start_index, initial in enumerate(initials):
        try:
            fit = fit_pairwise_btd(
                rows,
                initial=initial,
                num_models=num_models,
                dimension=dimension,
                max_iterations=max_iterations,
            )
        except RuntimeError as error:
            errors.append(f"start {start_index}: {error}")
        else:
            fits.append((start_index, fit))
    if len(fits) < 2:
        detail = "; ".join(errors) if errors else "fewer than two starts converged"
        raise RuntimeError(f"Pairwise BTD multi-start validation failed: {detail}")

    best_start, best = min(fits, key=lambda item: item[1].loss)
    loss_cutoff = best.loss + max(near_loss_tolerance, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= loss_cutoff]
    max_l1 = max(float(np.abs(fit.trust - best.trust).sum()) for fit in near)
    stable_near_optima = max_l1 <= trust_l1_tolerance
    if reject_unstable and not stable_near_optima:
        raise RuntimeError(
            "Pairwise BTD has near-optimal fits with incompatible trust vectors: "
            f"maximum L1 distance {max_l1:.6g}"
        )
    return PairwiseFit(
        parameters=best.parameters,
        trust=best.trust,
        loss=best.loss,
        iterations=best.iterations,
        attempted_starts=len(initials),
        successful_starts=len(fits),
        best_start=best_start,
        near_optimal_starts=len(near),
        max_near_optimal_trust_l1=max_l1,
        stable_near_optima=stable_near_optima,
        gradient_l2=best.gradient_l2,
        parameter_l2=best.parameter_l2,
        start_diagnostics=tuple(
            {
                "start": start_index,
                "loss": fit.loss,
                "iterations": fit.iterations,
                "gradient_l2": fit.gradient_l2,
                "parameter_l2": fit.parameter_l2,
                "trust": fit.trust.tolist(),
            }
            for start_index, fit in fits
        ),
        failed_starts=tuple(errors),
    )


def initial_btd_parameters(
    seed: int,
    num_models: int = 8,
    dimension: int = 2,
) -> np.ndarray:
    """Create a deterministic small-normal embedding initialization."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0.0, 0.1, 2 * num_models * dimension)
    return np.concatenate((embeddings, np.zeros(num_models)))


def multistart_initials(
    warm_start: np.ndarray,
    *,
    seed: int,
    cold_starts: int = 2,
) -> list[np.ndarray]:
    """Build a deterministic warm-plus-cold initialization bank."""
    if cold_starts < 1:
        raise ValueError("At least one cold BTD start is required")
    return [warm_start] + [
        initial_btd_parameters(seed + offset) for offset in range(cold_starts)
    ]


def fit_diagnostics(fit: PairwiseFit) -> dict[str, object]:
    """Return serializable optimizer and multi-start diagnostics."""
    return {
        "loss": fit.loss,
        "iterations": fit.iterations,
        "attempted_starts": fit.attempted_starts,
        "successful_starts": fit.successful_starts,
        "best_start": fit.best_start,
        "near_optimal_starts": fit.near_optimal_starts,
        "max_near_optimal_trust_l1": fit.max_near_optimal_trust_l1,
        "stable_near_optima": fit.stable_near_optima,
        "gradient_l2": fit.gradient_l2,
        "parameter_l2": fit.parameter_l2,
        "starts": list(fit.start_diagnostics),
        "failed_starts": list(fit.failed_starts),
    }
