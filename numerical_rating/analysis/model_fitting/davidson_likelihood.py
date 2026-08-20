"""Shared Davidson logits, cross-entropy, and analytic gradient."""

from __future__ import annotations

import numpy as np

from numerical_rating.analysis.utilities.probability import softmax_logits


def unpack_parameters(
    parameters: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    center_models: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_size = num_rows * dimension
    model_size = num_models * dimension
    row_vectors = parameters[:row_size].reshape(num_rows, dimension)
    model_vectors = parameters[row_size : row_size + model_size].reshape(
        num_models,
        dimension,
    )
    if center_models:
        model_vectors = model_vectors - model_vectors.mean(axis=0, keepdims=True)
    log_ties = parameters[row_size + model_size :]
    return row_vectors, model_vectors, log_ties


def logits(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    center_models: bool,
) -> np.ndarray:
    row_index, left, right, _ = rows.T
    row_vectors, model_vectors, log_ties = unpack_parameters(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        center_models=center_models,
    )
    left_score = np.sum(row_vectors[row_index] * model_vectors[left], axis=1)
    right_score = np.sum(row_vectors[row_index] * model_vectors[right], axis=1)
    return np.column_stack(
        (
            log_ties[row_index] + 0.5 * (left_score + right_score),
            left_score,
            right_score,
        )
    )


def loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    center_models: bool,
) -> tuple[float, np.ndarray]:
    """Return mean Davidson cross-entropy and its analytic gradient."""
    row_index, left, right, choice = rows.T
    row_vectors, model_vectors, _ = unpack_parameters(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        center_models=center_models,
    )
    probabilities = softmax_logits(
        logits(
            parameters,
            rows,
            num_rows=num_rows,
            num_models=num_models,
            dimension=dimension,
            center_models=center_models,
        )
    )
    index = np.arange(len(rows))
    loss = float(-np.log(np.maximum(probabilities[index, choice], 1e-300)).mean())
    residual = probabilities
    residual[index, choice] -= 1.0
    residual /= len(rows)
    left_coefficient = 0.5 * residual[:, 0] + residual[:, 1]
    right_coefficient = 0.5 * residual[:, 0] + residual[:, 2]

    row_gradient = np.zeros_like(row_vectors)
    model_gradient = np.zeros_like(model_vectors)
    tie_gradient = np.zeros(num_rows)
    np.add.at(
        row_gradient,
        row_index,
        left_coefficient[:, None] * model_vectors[left]
        + right_coefficient[:, None] * model_vectors[right],
    )
    np.add.at(
        model_gradient,
        left,
        left_coefficient[:, None] * row_vectors[row_index],
    )
    np.add.at(
        model_gradient,
        right,
        right_coefficient[:, None] * row_vectors[row_index],
    )
    if center_models:
        model_gradient -= model_gradient.mean(axis=0, keepdims=True)
    np.add.at(tie_gradient, row_index, residual[:, 0])
    return loss, np.concatenate(
        (row_gradient.ravel(), model_gradient.ravel(), tie_gradient)
    )
