import numpy as np

from numerical_rating.analysis.evaluation.trit_calibration import (
    calibration_loss_gradient,
    pairwise_logits,
)
from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    logits as davidson_logits,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    fit_criterion_svd,
    fit_rank_svd,
    standardize_ratings,
)


def test_svd_recovers_exact_low_rank_matrix():
    rng = np.random.default_rng(4)
    judges = rng.normal(size=(8, 2))
    models = rng.normal(size=(8, 2))
    matrix = judges @ models.T

    fit = fit_rank_svd(matrix, 2)

    np.testing.assert_allclose(fit.scores, matrix, atol=1e-12)
    np.testing.assert_allclose(fit.judges @ fit.models.T, matrix, atol=1e-12)


def test_criterion_svd_uses_criterion_judges_and_shared_models():
    rng = np.random.default_rng(5)
    criterion_judges = rng.normal(size=(3 * 2, 2))
    models = rng.normal(size=(4, 2))
    score_matrix = criterion_judges @ models.T
    scenario_offsets = rng.normal(size=(5,))
    ratings = (score_matrix[:, None, :] + scenario_offsets[None, :, None]).reshape(
        3, 2, 5, 4
    )

    fit, affinity, scales = fit_criterion_svd(ratings, 2)

    assert fit.judges.shape == (6, 2)
    assert fit.models.shape == (4, 2)
    assert scales.shape == (3, 2)
    np.testing.assert_allclose(fit.scores, affinity, atol=1e-12)


def test_averaged_svd_and_cell_mse_have_same_variable_term():
    rng = np.random.default_rng(7)
    ratings = rng.normal(size=(3, 11, 4))
    standardized, _, _ = standardize_ratings(ratings, ratings)
    affinity = standardized.mean(axis=1)
    prediction = rng.normal(size=(3, 4))

    cell_error = np.square(standardized - prediction[:, None, :]).sum()
    fixed_error = np.square(standardized - affinity[:, None, :]).sum()
    matrix_error = ratings.shape[1] * np.square(affinity - prediction).sum()

    np.testing.assert_allclose(cell_error, fixed_error + matrix_error, atol=1e-12)


def test_standardization_uses_train_scale_for_evaluation():
    train = np.asarray([[[1.0, 3.0], [2.0, 4.0]]])
    evaluation = np.asarray([[[10.0, 14.0]]])

    standardized_train, standardized_evaluation, scale = standardize_ratings(
        train, evaluation
    )

    np.testing.assert_allclose(standardized_train.mean(axis=2), 0.0)
    np.testing.assert_allclose(standardized_evaluation.mean(axis=2), 0.0)
    np.testing.assert_allclose(standardized_evaluation[0, 0], [-2.0, 2.0] / scale[0])


def test_calibration_gradient_matches_finite_difference():
    scores = np.asarray([[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2]])
    rows = np.asarray(
        [[0, 0, 1, 1], [0, 2, 1, 0], [1, 1, 0, 2], [1, 2, 0, 1]],
        dtype=np.int64,
    )
    parameters = np.asarray([0.2, -0.1, 0.3])
    loss, gradient = calibration_loss_gradient(parameters, scores, rows)
    assert np.isfinite(loss)

    epsilon = 1e-6
    finite = np.empty_like(parameters)
    for index in range(len(parameters)):
        offset = np.zeros_like(parameters)
        offset[index] = epsilon
        plus = calibration_loss_gradient(parameters + offset, scores, rows)[0]
        minus = calibration_loss_gradient(parameters - offset, scores, rows)[0]
        finite[index] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-8)


def test_pairwise_logits_use_shared_davidson_implementation():
    rng = np.random.default_rng(11)
    num_models = 3
    dimension = 2
    parameters = rng.normal(size=2 * num_models * dimension + num_models)
    rows = np.asarray(
        [[0, 0, 1, 1], [1, 2, 0, 0], [2, 1, 2, 2]],
        dtype=np.int64,
    )

    actual = pairwise_logits(
        parameters,
        rows,
        num_models=num_models,
        dimension=dimension,
    )
    expected = davidson_logits(
        parameters,
        rows,
        num_rows=num_models,
        num_models=num_models,
        dimension=dimension,
        center_models=False,
    )

    np.testing.assert_array_equal(actual, expected)
