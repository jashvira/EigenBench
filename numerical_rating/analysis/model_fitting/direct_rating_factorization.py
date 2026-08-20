"""Fit low-rank models to direct numerical-rating margins."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.stats import pearsonr

from numerical_rating.analysis.utilities.optimization import minimize_lbfgs


@dataclass(frozen=True)
class SVDFit:
    judges: np.ndarray
    models: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray


def standardize_ratings(
    train: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre each scenario and use the training spread for both tensors."""
    if train.ndim != 3 or evaluation.ndim != 3:
        raise ValueError("Rating tensors must have shape judge x scenario x model")
    if train.shape[0] != evaluation.shape[0] or train.shape[2] != evaluation.shape[2]:
        raise ValueError("Train and evaluation tensors must share judges and models")
    if train.shape[1] < 1 or train.shape[2] < 2:
        raise ValueError("Standardization requires scenarios and at least two models")

    train_centered = train - train.mean(axis=2, keepdims=True)
    denominator = train.shape[1] * (train.shape[2] - 1)
    scale = np.sqrt(np.square(train_centered).sum(axis=(1, 2)) / denominator)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Every judge must have positive finite rating spread")
    evaluation_centered = evaluation - evaluation.mean(axis=2, keepdims=True)
    return (
        train_centered / scale[:, None, None],
        evaluation_centered / scale[:, None, None],
        scale,
    )


def fit_rank_svd(matrix: np.ndarray, rank: int) -> SVDFit:
    """Return the rank-d least-squares factorization of a complete matrix."""
    if matrix.ndim != 2:
        raise ValueError("SVD input must be a matrix")
    max_rank = min(matrix.shape)
    if not 0 <= rank <= max_rank:
        raise ValueError(f"rank must be between 0 and {max_rank}")
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    if rank == 0:
        judges = np.zeros((matrix.shape[0], 0))
        models = np.zeros((matrix.shape[1], 0))
        scores = np.zeros_like(matrix)
    else:
        root = np.sqrt(singular_values[:rank])
        judges = left[:, :rank] * root
        models = right_t[:rank].T * root
        scores = judges @ models.T
    return SVDFit(judges, models, scores, singular_values)


def fit_criterion_svd(
    ratings: np.ndarray, rank: int
) -> tuple[SVDFit, np.ndarray, np.ndarray]:
    """Fit criterion-judge vectors and shared model vectors."""
    if ratings.ndim != 4:
        raise ValueError(
            "Criterion ratings must have shape criterion x judge x scenario x model"
        )
    criterion_count, judge_count, scenario_count, model_count = ratings.shape
    flattened = ratings.reshape(
        criterion_count * judge_count, scenario_count, model_count
    )
    standardized, _, scales = standardize_ratings(flattened, flattened)
    affinity = standardized.mean(axis=1)
    return (
        fit_rank_svd(affinity, rank),
        affinity,
        scales.reshape(criterion_count, judge_count),
    )


@dataclass(frozen=True)
class MarginFit:
    parameters: np.ndarray
    row_vectors: np.ndarray
    model_vectors: np.ndarray
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


def unpack_factors(
    parameters: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_size = num_rows * dimension
    row_vectors = parameters[:row_size].reshape(num_rows, dimension)
    model_vectors = parameters[row_size:].reshape(num_models, dimension)
    model_vectors = model_vectors - model_vectors.mean(axis=0, keepdims=True)
    return row_vectors, model_vectors


def margin_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return MSE and gradient for criterion-conditioned score gaps."""
    row_index, left, right, _ = rows.T
    row_vectors, model_vectors = unpack_factors(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    prediction = np.sum(
        row_vectors[row_index] * (model_vectors[left] - model_vectors[right]),
        axis=1,
    )
    error = prediction - targets
    loss = float(np.mean(np.square(error)))
    coefficient = 2.0 * error / len(rows)

    row_gradient = np.zeros_like(row_vectors)
    model_gradient = np.zeros_like(model_vectors)
    np.add.at(
        row_gradient,
        row_index,
        coefficient[:, None] * (model_vectors[left] - model_vectors[right]),
    )
    contribution = coefficient[:, None] * row_vectors[row_index]
    np.add.at(model_gradient, left, contribution)
    np.add.at(model_gradient, right, -contribution)
    model_gradient -= model_gradient.mean(axis=0, keepdims=True)
    return loss, np.concatenate((row_gradient.ravel(), model_gradient.ravel()))


def _fit_margin_once(
    rows: np.ndarray,
    targets: np.ndarray,
    initial: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> MarginFit:
    objective = partial(
        margin_loss_gradient,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    result, trace = minimize_lbfgs(
        objective,
        initial,
        args=(rows, targets),
        max_iterations=max_iterations,
        record_trace=True,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Margin MSE optimization failed: {result.message}")
    row_vectors, model_vectors = unpack_factors(
        result.x,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    return MarginFit(
        parameters=result.x,
        row_vectors=row_vectors,
        model_vectors=model_vectors,
        scores=row_vectors @ model_vectors.T,
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        loss_trace=trace,
    )


def score_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    return float(pearsonr(left.ravel(), right.ravel()).statistic)


def fit_margin_multistart(
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> MarginFit:
    fits: list[tuple[int, MarginFit]] = []
    errors: list[str] = []
    for start, initial in enumerate(initials):
        try:
            fit = _fit_margin_once(
                rows,
                targets,
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
        raise RuntimeError("All margin MSE starts failed: " + "; ".join(errors))
    best_start, best = min(fits, key=lambda item: item[1].loss)
    cutoff = best.loss + max(1e-8, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= cutoff]
    min_correlation = min(score_correlation(best.scores, fit.scores) for fit in near)
    return MarginFit(
        parameters=best.parameters,
        row_vectors=best.row_vectors,
        model_vectors=best.model_vectors,
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


def factor_initials(
    standardized_ratings: np.ndarray,
    *,
    dimension: int,
    starts: int,
    seed: int,
) -> list[np.ndarray]:
    warm = fit_rank_svd(standardized_ratings.mean(axis=1), dimension)
    initials = [
        np.concatenate(
            (
                warm.judges.ravel(),
                (warm.models - warm.models.mean(axis=0, keepdims=True)).ravel(),
            )
        )
    ]
    rng = np.random.default_rng(seed)
    parameter_count = (
        standardized_ratings.shape[0] + standardized_ratings.shape[2]
    ) * dimension
    initials.extend(rng.normal(0.0, 0.1, parameter_count) for _ in range(starts - 1))
    return initials


def fit_svd_direct(
    standardized_ratings: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    dimension: int,
) -> MarginFit:
    """Fit the direct model by truncated SVD of mean ratings."""
    svd = fit_rank_svd(standardized_ratings.mean(axis=1), dimension)
    model_vectors = svd.models - svd.models.mean(axis=0, keepdims=True)
    scores = svd.judges @ model_vectors.T
    prediction = scores[rows[:, 0], rows[:, 1]] - scores[rows[:, 0], rows[:, 2]]
    loss = float(np.mean(np.square(prediction - targets)))
    parameters = np.concatenate((svd.judges.ravel(), model_vectors.ravel()))
    trace = ({"iteration": 0, "loss": loss, "gradient_l2": 0.0},)
    return MarginFit(
        parameters=parameters,
        row_vectors=svd.judges,
        model_vectors=model_vectors,
        scores=scores,
        loss=loss,
        iterations=0,
        gradient_l2=0.0,
        loss_trace=trace,
        start_diagnostics=(
            {
                "start": 0,
                "loss": loss,
                "iterations": 0,
                "gradient_l2": 0.0,
                "trace": list(trace),
                "solver": "truncated_svd",
                "score_correlation_to_best": 1.0,
            },
        ),
    )
