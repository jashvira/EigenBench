import numpy as np
import pytest

from numerical_rating.analysis.data_loading.matched_criterion_data import (
    MatchedCriterionData,
)
from numerical_rating.analysis.data_loading.matched_criterion_data import (
    responses_match as _responses_match,
)
from numerical_rating.analysis.experiments.criterion_direct_vs_btd import (
    heldout_evaluation as evaluation,
)
from numerical_rating.analysis.model_fitting.criterion_btd import (
    criterion_btd_logits,
    criterion_btd_loss_gradient,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    fit_svd_direct,
    margin_loss_gradient,
)
from numerical_rating.collection.data import sha256_text

cross_validate = evaluation.cross_validate
rows_and_targets = evaluation.rows_and_targets
_fit_models = evaluation.fit_models


def _finite_difference(function, parameters, epsilon=1e-6):
    gradient = np.empty_like(parameters)
    for index in range(len(parameters)):
        offset = np.zeros_like(parameters)
        offset[index] = epsilon
        gradient[index] = (
            function(parameters + offset)[0] - function(parameters - offset)[0]
        ) / (2 * epsilon)
    return gradient


def test_margin_gradient_matches_finite_difference():
    rng = np.random.default_rng(11)
    rows = np.asarray(
        [[0, 0, 1, 1], [1, 2, 0, 0], [2, 1, 2, 2], [0, 2, 1, 1]],
        dtype=np.int64,
    )
    targets = np.asarray([0.4, -0.2, 0.8, -0.5])
    parameters = rng.normal(size=(3 + 3) * 2)

    def function(value):
        return margin_loss_gradient(
            value,
            rows,
            targets,
            num_rows=3,
            num_models=3,
            dimension=2,
        )

    loss, gradient = function(parameters)

    assert np.isfinite(loss)
    np.testing.assert_allclose(
        gradient,
        _finite_difference(function, parameters),
        rtol=1e-6,
        atol=1e-8,
    )


def test_criterion_btd_gradient_matches_finite_difference():
    rng = np.random.default_rng(12)
    rows = np.asarray(
        [[0, 0, 1, 1], [1, 2, 0, 0], [2, 1, 2, 2], [0, 2, 1, 1]],
        dtype=np.int64,
    )
    parameters = rng.normal(size=(3 + 3) * 2 + 3)

    def function(value):
        return criterion_btd_loss_gradient(
            value,
            rows,
            num_rows=3,
            num_models=3,
            dimension=2,
        )

    loss, gradient = function(parameters)

    assert np.isfinite(loss)
    np.testing.assert_allclose(
        gradient,
        _finite_difference(function, parameters),
        rtol=1e-6,
        atol=1e-8,
    )


def test_logits_match_repository_criteria_vector_btd():
    torch = pytest.importorskip("torch")
    from pipeline.train.bt_models import CriteriaVectorBTD

    rng = np.random.default_rng(13)
    num_criteria = 2
    num_models = 3
    dimension = 2
    row_vectors = rng.normal(size=(num_criteria * num_models, dimension))
    model_vectors = rng.normal(size=(num_models, dimension))
    model_vectors -= model_vectors.mean(axis=0, keepdims=True)
    ties = rng.normal(size=num_criteria * num_models)
    parameters = np.concatenate((row_vectors.ravel(), model_vectors.ravel(), ties))
    rows = np.asarray([[0, 0, 1, 1], [4, 2, 0, 0], [5, 1, 2, 2]], dtype=np.int64)

    expected = criterion_btd_logits(
        parameters,
        rows,
        num_rows=num_criteria * num_models,
        num_models=num_models,
        dimension=dimension,
    )
    model = CriteriaVectorBTD(num_criteria, num_models, dimension)
    with torch.no_grad():
        model.u.weight.copy_(torch.from_numpy(row_vectors).float())
        model.v.weight.copy_(torch.from_numpy(model_vectors).float())
        model.log_lambda.weight.copy_(torch.from_numpy(ties[:, None]).float())
    row_index, left, right, _ = rows.T
    actual = model(
        torch.from_numpy(row_index // num_models),
        torch.from_numpy(row_index % num_models),
        torch.from_numpy(left),
        torch.from_numpy(right),
    )

    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-6)


def test_rows_and_targets_preserve_pair_orientation():
    blocks = {
        10: np.asarray([[0, 0, 2, 1], [1, 2, 1, 0]], dtype=np.int64),
        11: np.asarray([[1, 0, 1, 2]], dtype=np.int64),
    }
    ratings = np.asarray(
        [
            [[1.0, 2.0, 4.0], [3.0, 2.0, 1.0]],
            [[0.0, 5.0, 2.0], [4.0, 1.0, 3.0]],
        ]
    )

    rows, targets = rows_and_targets(blocks, np.asarray([10, 11]), ratings)

    np.testing.assert_array_equal(rows, np.vstack((blocks[10], blocks[11])))
    np.testing.assert_allclose(targets, [-3.0, -3.0, 3.0])


def test_svd_direct_uses_the_rank_factorization():
    rng = np.random.default_rng(14)
    row_vectors = rng.normal(size=(4, 2))
    model_vectors = rng.normal(size=(3, 2))
    model_vectors -= model_vectors.mean(axis=0, keepdims=True)
    scores = row_vectors @ model_vectors.T
    ratings = np.repeat(scores[:, None, :], 5, axis=1)
    rows = np.asarray([[0, 0, 1, 1], [1, 2, 0, 2], [3, 1, 2, 0]], dtype=np.int64)
    targets = scores[rows[:, 0], rows[:, 1]] - scores[rows[:, 0], rows[:, 2]]

    fit = fit_svd_direct(ratings, rows, targets, dimension=2)

    np.testing.assert_allclose(fit.scores, scores, atol=1e-12)
    assert fit.loss < 1e-24
    assert fit.iterations == 0


def test_response_matching_rejects_changed_text():
    evaluation = {
        "scenario_index": 5,
        "eval1": 1,
        "eval1 response": "left response",
        "eval2": 2,
        "eval2 response": "right response",
    }
    hashes = {
        (5, 1): sha256_text("left response"),
        (5, 2): sha256_text("right response"),
    }

    assert _responses_match(evaluation, hashes)
    evaluation["eval2 response"] = "different response"
    assert not _responses_match(evaluation, hashes)


def test_cross_validation_passes_only_training_scenarios_to_fit(monkeypatch):
    scenario_ids = np.arange(4, dtype=np.int64)
    data = MatchedCriterionData(
        ratings=np.arange(12, dtype=float).reshape(1, 1, 4, 3),
        criterion_ids=["criterion_01"],
        rating_scenario_ids=scenario_ids,
        scenario_ids=scenario_ids,
        judge_names=["judge"],
        model_names=["a", "b", "c"],
        blocks={
            int(scenario): np.asarray([[0, 0, 1, 1]], dtype=np.int64)
            for scenario in scenario_ids
        },
        diagnostics={},
    )

    class FitReached(Exception):
        pass

    def inspect_fit(ratings, rows, targets, **_):
        assert ratings.shape == (1, 2, 3)
        assert rows.shape == (2, 4)
        assert targets.shape == (2,)
        raise FitReached

    monkeypatch.setattr(evaluation, "fit_models", inspect_fit)
    with pytest.raises(FitReached):
        cross_validate(
            data,
            folds=2,
            dimension=1,
            starts=1,
            seed=0,
            max_iterations=10,
            direct_method="svd",
        )


def test_direct_calibration_receives_only_supplied_rows(monkeypatch):
    ratings = np.arange(18, dtype=float).reshape(2, 3, 3)
    rows = np.asarray([[0, 0, 1, 1], [1, 2, 0, 0]], dtype=np.int64)
    targets = np.asarray([-1.0, 2.0])

    class CalibrationReached(Exception):
        pass

    def inspect_calibration(scores, calibration_rows):
        assert scores.shape == (2, 3)
        np.testing.assert_array_equal(calibration_rows, rows)
        raise CalibrationReached

    monkeypatch.setattr(
        evaluation,
        "fit_direct_calibration",
        inspect_calibration,
    )
    with pytest.raises(CalibrationReached):
        _fit_models(
            ratings,
            rows,
            targets,
            dimension=1,
            starts=1,
            seed=0,
            max_iterations=10,
            direct_method="svd",
        )
