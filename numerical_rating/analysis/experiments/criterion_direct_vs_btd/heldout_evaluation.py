"""Held-out evaluation for direct ratings and criterion BTD."""

from __future__ import annotations

import numpy as np

from numerical_rating.analysis.data_loading.matched_criterion_data import (
    MatchedCriterionData,
)
from numerical_rating.analysis.evaluation.embedding_comparison import geometry_metrics
from numerical_rating.analysis.evaluation.scenario_splits import scenario_folds
from numerical_rating.analysis.evaluation.trit_calibration import (
    classification_metrics,
    fit_direct_calibration,
    fixed_score_logits,
)
from numerical_rating.analysis.model_fitting.criterion_btd import (
    CriterionBTDFit,
    btd_initials,
    criterion_btd_logits,
    fit_btd_multistart,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    MarginFit,
    factor_initials,
    fit_margin_multistart,
    fit_rank_svd,
    fit_svd_direct,
    standardize_ratings,
)
from numerical_rating.analysis.utilities.probability import softmax_logits


def rows_and_targets(
    blocks: dict[int, np.ndarray],
    scenario_ids: np.ndarray,
    standardized_ratings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Join pairwise rows to direct rating gaps in the same orientation."""
    row_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for position, scenario in enumerate(scenario_ids):
        rows = blocks[int(scenario)]
        row_index, left, right, _ = rows.T
        target = (
            standardized_ratings[row_index, position, left]
            - standardized_ratings[row_index, position, right]
        )
        row_parts.append(rows)
        target_parts.append(target)
    return np.concatenate(row_parts), np.concatenate(target_parts)


def calibration_metrics(logits: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    metrics = classification_metrics(logits, choices)
    probabilities = softmax_logits(logits)

    def ece(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        total = len(probability)
        value = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (probability >= lower) & (
                probability <= upper if upper == 1.0 else probability < upper
            )
            if np.any(selected):
                value += selected.mean() * abs(
                    probability[selected].mean() - outcome[selected].mean()
                )
        return float(value if total else np.nan)

    metrics["tie_ece"] = ece(probabilities[:, 0], choices == 0)
    strict = choices != 0
    strict_probability = probabilities[strict, 1] / probabilities[strict, 1:].sum(
        axis=1
    )
    metrics["win_loss_ece"] = ece(strict_probability, choices[strict] == 1)
    return metrics


def mean_se(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows])
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def fit_models(
    standardized_ratings: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    dimension: int,
    starts: int,
    seed: int,
    max_iterations: int,
    direct_method: str,
) -> tuple[MarginFit, np.ndarray, CriterionBTDFit]:
    num_rows, _, num_models = standardized_ratings.shape
    if direct_method == "svd":
        direct = fit_svd_direct(
            standardized_ratings,
            rows,
            targets,
            dimension=dimension,
        )
    elif direct_method == "matched_margin":
        direct = fit_margin_multistart(
            rows,
            targets,
            initials=factor_initials(
                standardized_ratings,
                dimension=dimension,
                starts=starts,
                seed=seed,
            ),
            num_rows=num_rows,
            num_models=num_models,
            dimension=dimension,
            max_iterations=max_iterations,
        )
    else:
        raise ValueError(f"Unknown direct method: {direct_method}")
    calibration = fit_direct_calibration(direct.scores, rows)
    btd = fit_btd_multistart(
        rows,
        initials=btd_initials(
            direct,
            calibration,
            starts=starts,
            seed=seed + 1_000,
        ),
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
        max_iterations=max_iterations,
    )
    return direct, calibration, btd


def cross_validate(
    data: MatchedCriterionData,
    *,
    folds: int,
    dimension: int,
    starts: int,
    seed: int,
    max_iterations: int,
    direct_method: str,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    flattened = data.ratings.reshape(
        data.ratings.shape[0] * data.ratings.shape[1],
        data.ratings.shape[2],
        data.ratings.shape[3],
    )
    scenario_position = {
        int(scenario): position
        for position, scenario in enumerate(data.rating_scenario_ids)
    }
    matched_positions = np.asarray(
        [scenario_position[int(scenario)] for scenario in data.scenario_ids]
    )
    matched = flattened[:, matched_positions, :]
    split = scenario_folds(len(data.scenario_ids), folds, seed)
    output: list[dict[str, object]] = []
    choice_parts: list[np.ndarray] = []
    direct_probability_parts: list[np.ndarray] = []
    btd_probability_parts: list[np.ndarray] = []
    all_positions = np.arange(len(data.scenario_ids))
    for fold, test_positions in enumerate(split):
        train_positions = np.setdiff1d(
            all_positions,
            test_positions,
            assume_unique=True,
        )
        train_ratings, test_ratings, _ = standardize_ratings(
            matched[:, train_positions, :],
            matched[:, test_positions, :],
        )
        train_rows, train_targets = rows_and_targets(
            data.blocks,
            data.scenario_ids[train_positions],
            train_ratings,
        )
        test_rows, test_targets = rows_and_targets(
            data.blocks,
            data.scenario_ids[test_positions],
            test_ratings,
        )
        direct, calibration, btd = fit_models(
            train_ratings,
            train_rows,
            train_targets,
            dimension=dimension,
            starts=starts,
            seed=seed + 10_000 * fold,
            max_iterations=max_iterations,
            direct_method=direct_method,
        )
        direct_prediction = (
            direct.scores[test_rows[:, 0], test_rows[:, 1]]
            - direct.scores[test_rows[:, 0], test_rows[:, 2]]
        )
        direct_logits = fixed_score_logits(calibration, direct.scores, test_rows)
        btd_logits = criterion_btd_logits(
            btd.parameters,
            test_rows,
            num_rows=train_ratings.shape[0],
            num_models=train_ratings.shape[2],
            dimension=dimension,
        )
        direct_classification = calibration_metrics(
            direct_logits,
            test_rows[:, 3],
        )
        btd_classification = calibration_metrics(btd_logits, test_rows[:, 3])
        choice_parts.append(test_rows[:, 3])
        direct_probability_parts.append(softmax_logits(direct_logits))
        btd_probability_parts.append(softmax_logits(btd_logits))
        geometry = geometry_metrics(direct.scores, btd.scores)
        test_mse = float(np.mean(np.square(direct_prediction - test_targets)))
        baseline_mse = float(np.mean(np.square(test_targets)))
        output.append(
            {
                "fold": fold,
                "train_scenarios": len(train_positions),
                "test_scenarios": len(test_positions),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "direct_test_mse": test_mse,
                "direct_test_mse_reduction_vs_zero": 1.0 - test_mse / baseline_mse,
                "direct_nll": direct_classification["nll"],
                "direct_accuracy": direct_classification["accuracy"],
                "direct_tie_ece": direct_classification["tie_ece"],
                "direct_win_loss_ece": direct_classification["win_loss_ece"],
                "btd_nll": btd_classification["nll"],
                "btd_accuracy": btd_classification["accuracy"],
                "btd_tie_ece": btd_classification["tie_ece"],
                "btd_win_loss_ece": btd_classification["win_loss_ece"],
                "nll_gap_direct_minus_btd": (
                    direct_classification["nll"] - btd_classification["nll"]
                ),
                "score_pearson": geometry["score_pearson"],
                "margin_pearson": geometry["margin_pearson"],
                "margin_sign_agreement": geometry["margin_sign_agreement"],
                "direct_train_mse": direct.loss,
                "btd_train_nll": btd.loss,
                "direct_iterations": direct.iterations,
                "btd_iterations": btd.iterations,
                "direct_gradient_l2": direct.gradient_l2,
                "btd_gradient_l2": btd.gradient_l2,
                "direct_successful_starts": direct.successful_starts,
                "btd_successful_starts": btd.successful_starts,
                "direct_min_near_score_correlation": (
                    direct.min_near_score_correlation
                ),
                "btd_min_near_score_correlation": btd.min_near_score_correlation,
            }
        )
    return (
        output,
        np.concatenate(choice_parts),
        np.concatenate(direct_probability_parts),
        np.concatenate(btd_probability_parts),
    )


def reliability_rows(
    choices: np.ndarray,
    direct_probabilities: np.ndarray,
    btd_probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, object]]:
    """Bin held-out tie and strict-win probabilities."""
    output: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model_name, probabilities in (
        ("direct", direct_probabilities),
        ("btd", btd_probabilities),
    ):
        series = [("tie", probabilities[:, 0], choices == 0)]
        strict = choices != 0
        strict_win = probabilities[strict, 1] / probabilities[strict, 1:].sum(axis=1)
        series.append(("left_win_given_not_tie", strict_win, choices[strict] == 1))
        for target_name, prediction, outcome in series:
            for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                selected = (prediction >= lower) & (
                    prediction <= upper if upper == 1.0 else prediction < upper
                )
                if np.any(selected):
                    output.append(
                        {
                            "model": model_name,
                            "target": target_name,
                            "bin": index,
                            "lower": lower,
                            "upper": upper,
                            "count": int(selected.sum()),
                            "mean_prediction": float(prediction[selected].mean()),
                            "empirical_rate": float(outcome[selected].mean()),
                        }
                    )
    return output


def svd_rank_curve(
    data: MatchedCriterionData,
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, float | int]]]:
    """Measure held-out matched-margin MSE across SVD ranks."""
    flattened = data.ratings.reshape(
        data.ratings.shape[0] * data.ratings.shape[1],
        data.ratings.shape[2],
        data.ratings.shape[3],
    )
    scenario_position = {
        int(scenario): position
        for position, scenario in enumerate(data.rating_scenario_ids)
    }
    matched_positions = np.asarray(
        [scenario_position[int(scenario)] for scenario in data.scenario_ids]
    )
    matched = flattened[:, matched_positions, :]
    split = scenario_folds(len(data.scenario_ids), folds, seed)
    all_positions = np.arange(len(data.scenario_ids))
    rows: list[dict[str, object]] = []
    for fold, test_positions in enumerate(split):
        train_positions = np.setdiff1d(
            all_positions,
            test_positions,
            assume_unique=True,
        )
        train_ratings, test_ratings, _ = standardize_ratings(
            matched[:, train_positions, :],
            matched[:, test_positions, :],
        )
        test_rows, test_targets = rows_and_targets(
            data.blocks,
            data.scenario_ids[test_positions],
            test_ratings,
        )
        baseline_mse = float(np.mean(np.square(test_targets)))
        for rank in range(matched.shape[2]):
            scores = fit_rank_svd(train_ratings.mean(axis=1), rank).scores
            prediction = (
                scores[test_rows[:, 0], test_rows[:, 1]]
                - scores[test_rows[:, 0], test_rows[:, 2]]
            )
            mse = float(np.mean(np.square(prediction - test_targets)))
            rows.append(
                {
                    "fold": fold,
                    "rank": rank,
                    "test_mse": mse,
                    "mse_reduction_vs_rank_0": 1.0 - mse / baseline_mse,
                }
            )
    summary = []
    for rank in range(matched.shape[2]):
        values = np.asarray(
            [float(row["test_mse"]) for row in rows if row["rank"] == rank]
        )
        reductions = np.asarray(
            [
                float(row["mse_reduction_vs_rank_0"])
                for row in rows
                if row["rank"] == rank
            ]
        )
        summary.append(
            {
                "rank": rank,
                "mean_test_mse": float(values.mean()),
                "fold_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                "mean_mse_reduction_vs_rank_0": float(reductions.mean()),
            }
        )
    return rows, summary
