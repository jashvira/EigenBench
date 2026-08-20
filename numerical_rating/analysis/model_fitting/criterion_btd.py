"""Fit criterion-conditioned Davidson BTD models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    logits as davidson_logits,
)
from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    loss_gradient as davidson_loss_gradient,
)
from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    unpack_parameters,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    MarginFit,
    score_correlation,
)
from numerical_rating.analysis.utilities.optimization import minimize_lbfgs


@dataclass(frozen=True)
class CriterionBTDFit:
    parameters: np.ndarray
    row_vectors: np.ndarray
    model_vectors: np.ndarray
    log_ties: np.ndarray
    scores: np.ndarray
    loss: float
    iterations: int
    gradient_l2: float
    loss_trace: tuple[dict[str, float | int], ...] = ()
    attempted_starts: int = 1
    successful_starts: int = 1
    best_start: int = 0
    min_near_score_correlation: float = 1.0
    start_diagnostics: tuple[dict[str, object], ...] = ()
    failed_starts: tuple[str, ...] = ()


def _unpack_btd(
    parameters: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return unpack_parameters(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        center_models=True,
    )


def criterion_btd_logits(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> np.ndarray:
    return davidson_logits(
        parameters,
        rows,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        center_models=True,
    )


def criterion_btd_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return Davidson cross-entropy and gradient for criterion rows."""
    return davidson_loss_gradient(
        parameters,
        rows,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        center_models=True,
    )


def _fit_btd_once(
    rows: np.ndarray,
    initial: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> CriterionBTDFit:
    objective = partial(
        criterion_btd_loss_gradient,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    result, trace = minimize_lbfgs(
        objective,
        initial,
        args=(rows,),
        max_iterations=max_iterations,
        record_trace=True,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Criterion BTD optimization failed: {result.message}")
    row_vectors, model_vectors, log_ties = _unpack_btd(
        result.x,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    return CriterionBTDFit(
        parameters=result.x,
        row_vectors=row_vectors,
        model_vectors=model_vectors,
        log_ties=log_ties,
        scores=row_vectors @ model_vectors.T,
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        loss_trace=trace,
    )


def fit_btd_multistart(
    rows: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> CriterionBTDFit:
    fits: list[tuple[int, CriterionBTDFit]] = []
    errors: list[str] = []
    for start, initial in enumerate(initials):
        try:
            fit = _fit_btd_once(
                rows,
                initial,
                num_rows=num_rows,
                num_models=num_models,
                dimension=dimension,
                max_iterations=max_iterations,
            )
        except RuntimeError as error:
            errors.append(f"start {start}: {error}")
        else:
            fits.append((start, fit))
    if not fits:
        raise RuntimeError("All criterion BTD starts failed: " + "; ".join(errors))
    best_start, best = min(fits, key=lambda item: item[1].loss)
    cutoff = best.loss + max(1e-8, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= cutoff]
    min_correlation = min(score_correlation(best.scores, fit.scores) for fit in near)
    return CriterionBTDFit(
        parameters=best.parameters,
        row_vectors=best.row_vectors,
        model_vectors=best.model_vectors,
        log_ties=best.log_ties,
        scores=best.scores,
        loss=best.loss,
        iterations=best.iterations,
        gradient_l2=best.gradient_l2,
        loss_trace=best.loss_trace,
        attempted_starts=len(initials),
        successful_starts=len(fits),
        best_start=best_start,
        min_near_score_correlation=min_correlation,
        start_diagnostics=tuple(
            {
                "start": start,
                "loss": fit.loss,
                "iterations": fit.iterations,
                "gradient_l2": fit.gradient_l2,
                "trace": list(fit.loss_trace),
                "score_correlation_to_best": score_correlation(
                    best.scores,
                    fit.scores,
                ),
            }
            for start, fit in fits
        ),
        failed_starts=tuple(errors),
    )


def btd_initials(
    direct: MarginFit,
    calibration: np.ndarray,
    *,
    starts: int,
    seed: int,
) -> list[np.ndarray]:
    root_scale = float(np.sqrt(np.exp(calibration[0])))
    warm = np.concatenate(
        (
            (root_scale * direct.row_vectors).ravel(),
            (root_scale * direct.model_vectors).ravel(),
            calibration[1:],
        )
    )
    initials = [warm]
    rng = np.random.default_rng(seed)
    factor_count = direct.row_vectors.size + direct.model_vectors.size
    initials.extend(
        np.concatenate(
            (rng.normal(0.0, 0.1, factor_count), np.zeros(len(calibration) - 1))
        )
        for _ in range(starts - 1)
    )
    return initials
