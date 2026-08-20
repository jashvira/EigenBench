"""Calibration and classification helpers for Davidson comparisons."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    logits as davidson_logits,
)
from numerical_rating.analysis.utilities.probability import softmax_logits


def fixed_score_logits(
    parameters: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Apply a fitted scale and per-judge tie logits to fixed scores."""
    judge, left, right, _ = rows.T
    scale = np.exp(parameters[0])
    left_score = scale * scores[judge, left]
    right_score = scale * scores[judge, right]
    ties = parameters[1:]
    return np.column_stack(
        (ties[judge] + 0.5 * (left_score + right_score), left_score, right_score)
    )


def calibration_loss_gradient(
    parameters: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return calibration cross-entropy and its analytic gradient."""
    judge, left, right, choice = rows.T
    scale = np.exp(parameters[0])
    base_left = scores[judge, left]
    base_right = scores[judge, right]
    logits = fixed_score_logits(parameters, scores, rows)
    probabilities = softmax_logits(logits)
    index = np.arange(len(rows))
    loss = -np.log(np.maximum(probabilities[index, choice], 1e-300)).mean()

    residual = probabilities.copy()
    residual[index, choice] -= 1.0
    residual /= len(rows)
    scale_gradient = np.sum(
        0.5 * residual[:, 0] * (base_left + base_right)
        + residual[:, 1] * base_left
        + residual[:, 2] * base_right
    )
    tie_gradient = np.zeros(scores.shape[0])
    np.add.at(tie_gradient, judge, residual[:, 0])
    gradient = np.concatenate(([scale * scale_gradient], tie_gradient))
    return float(loss), gradient


def fit_direct_calibration(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Fit one positive score scale and one Davidson tie logit per judge."""
    fits = []
    for log_scale in (-1.0, 0.0, 1.0):
        initial = np.concatenate(([log_scale], np.zeros(scores.shape[0])))
        result = minimize(
            calibration_loss_gradient,
            initial,
            args=(scores, rows),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-8.0, 8.0)] + [(None, None)] * scores.shape[0],
            options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if result.success and np.isfinite(result.fun) and np.isfinite(result.x).all():
            fits.append(result)
    if not fits:
        raise RuntimeError("Direct-score calibration failed from all starts")
    return min(fits, key=lambda result: result.fun).x


def pairwise_logits(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_models: int,
    dimension: int,
) -> np.ndarray:
    """Compute Davidson logits from fitted pairwise parameters."""
    return davidson_logits(
        parameters,
        rows,
        num_rows=num_models,
        num_models=num_models,
        dimension=dimension,
        center_models=False,
    )


def classification_metrics(logits: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    """Return trit negative log-likelihood and accuracy."""
    probabilities = softmax_logits(logits)
    index = np.arange(len(choices))
    return {
        "nll": float(-np.log(np.maximum(probabilities[index, choices], 1e-300)).mean()),
        "accuracy": float(np.mean(np.argmax(logits, axis=1) == choices)),
    }
