"""Compare whole and criterion direct-rating embeddings with pairwise BTD.

Each direct model is the global rank-d least-squares factorization of its
scenario-averaged standardized rating matrix. Pairwise comparisons use the
existing deterministic Davidson fitter. Scenario splits are shared throughout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from numerical_rating.scenario_uncertainty import (
    concatenate_blocks,
    fit_pairwise_btd_multistart,
    initial_btd_parameters,
    load_numerical_tensor,
    load_pairwise_data,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = (
    ROOT / "data/output/numerical_rating/kindness_1000_round_robin/ratings.csv"
)
DEFAULT_CRITERION_RATINGS = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_criterion_round_robin/ratings.csv"
)
DEFAULT_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_round_robin/"
    "direct_embedding_comparison"
)


@dataclass(frozen=True)
class SVDFit:
    judges: np.ndarray
    models: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray


def load_criterion_tensor(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[str], list[str]]:
    """Load a complete criterion-judge-scenario-model rating tensor."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("Criterion ratings file is empty")

    criterion_ids = sorted({row["dimension_id"] for row in rows})
    judge_ids = sorted({int(row["judge_id"]) for row in rows})
    scenario_ids = sorted({int(row["scenario_index"]) for row in rows})
    model_ids = sorted({int(row["model_id"]) for row in rows})
    if judge_ids != list(range(len(judge_ids))):
        raise ValueError("Criterion ratings require contiguous judge IDs")
    if model_ids != list(range(len(model_ids))):
        raise ValueError("Criterion ratings require contiguous model IDs")

    expected_rows = (
        len(criterion_ids) * len(judge_ids) * len(scenario_ids) * len(model_ids)
    )
    if len(rows) != expected_rows:
        raise ValueError("Criterion ratings do not form a complete rectangle")

    criterion_position = {
        criterion_id: position for position, criterion_id in enumerate(criterion_ids)
    }
    scenario_position = {
        scenario_id: position for position, scenario_id in enumerate(scenario_ids)
    }
    tensor = np.full(
        (len(criterion_ids), len(judge_ids), len(scenario_ids), len(model_ids)),
        np.nan,
    )
    criterion_hashes: list[str | None] = [None] * len(criterion_ids)
    judge_names: list[str | None] = [None] * len(judge_ids)
    model_names: list[str | None] = [None] * len(model_ids)
    for row in rows:
        criterion = criterion_position[row["dimension_id"]]
        judge = int(row["judge_id"])
        scenario = scenario_position[int(row["scenario_index"])]
        model = int(row["model_id"])
        if np.isfinite(tensor[criterion, judge, scenario, model]):
            raise ValueError("Criterion ratings contain a duplicate cell")
        tensor[criterion, judge, scenario, model] = float(row["score"])

        criterion_hash = row["dimension_hash"]
        if criterion_hashes[criterion] not in (None, criterion_hash):
            raise ValueError("Criterion text hash changed within the ratings file")
        criterion_hashes[criterion] = criterion_hash
        if judge_names[judge] not in (None, row["judge_name"]):
            raise ValueError("Judge name changed within the ratings file")
        judge_names[judge] = row["judge_name"]
        if model_names[model] not in (None, row["model_name"]):
            raise ValueError("Model name changed within the ratings file")
        model_names[model] = row["model_name"]

    if not np.isfinite(tensor).all():
        raise ValueError("Criterion ratings contain a missing or non-finite cell")
    return (
        tensor,
        criterion_ids,
        [str(value) for value in criterion_hashes],
        scenario_ids,
        [str(value) for value in judge_names],
        [str(value) for value in model_names],
    )


def standardize_ratings(
    train: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre ratings within scenario and scale evaluation data by train spread."""
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
    """Return the canonical rank-d MSE factorization of a complete matrix."""
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


def scenario_folds(count: int, folds: int, seed: int) -> list[np.ndarray]:
    """Create deterministic near-equal held-out scenario folds."""
    if not 2 <= folds <= count:
        raise ValueError("folds must be between 2 and the scenario count")
    permutation = np.random.default_rng(seed).permutation(count)
    return [
        np.asarray(fold, dtype=np.int64)
        for fold in np.array_split(permutation, folds)
    ]


def rating_rank_cross_validation(
    ratings: np.ndarray,
    *,
    folds: list[np.ndarray],
    max_rank: int,
) -> list[dict[str, float | int]]:
    """Measure scenario-held-out rating MSE for ranks zero through max_rank."""
    all_positions = np.arange(ratings.shape[1])
    rows: list[dict[str, float | int]] = []
    for fold_index, test_positions in enumerate(folds):
        train_positions = np.setdiff1d(
            all_positions, test_positions, assume_unique=True
        )
        train, test, _ = standardize_ratings(
            ratings[:, train_positions, :], ratings[:, test_positions, :]
        )
        affinity = train.mean(axis=1)
        for rank in range(max_rank + 1):
            prediction = fit_rank_svd(affinity, rank).scores
            mse = float(np.mean(np.square(test - prediction[:, None, :])))
            rows.append({"fold": fold_index, "rank": rank, "test_mse": mse})
    return rows


def summarize_rank_cv(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    """Summarize fold MSE and paired uncertainty relative to the best rank."""
    ranks = sorted({int(row["rank"]) for row in rows})
    by_rank = {
        rank: np.asarray(
            [float(row["test_mse"]) for row in rows if row["rank"] == rank]
        )
        for rank in ranks
    }
    best_rank = min(ranks, key=lambda rank: float(by_rank[rank].mean()))
    best = by_rank[best_rank]
    summary: list[dict[str, float | int]] = []
    for rank in ranks:
        values = by_rank[rank]
        delta = values - best
        summary.append(
            {
                "rank": rank,
                "mean_test_mse": float(values.mean()),
                "fold_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                "delta_from_best": float(delta.mean()),
                "paired_delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                "best_rank": best_rank,
            }
        )
    return summary


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _fixed_score_logits(
    parameters: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    judge, left, right, _ = rows.T
    scale = np.exp(parameters[0])
    left_score = scale * scores[judge, left]
    right_score = scale * scores[judge, right]
    ties = parameters[1:]
    return np.column_stack(
        (ties[judge] + 0.5 * (left_score + right_score), left_score, right_score)
    )


def _calibration_loss_gradient(
    parameters: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> tuple[float, np.ndarray]:
    judge, left, right, choice = rows.T
    scale = np.exp(parameters[0])
    base_left = scores[judge, left]
    base_right = scores[judge, right]
    logits = _fixed_score_logits(parameters, scores, rows)
    probabilities = _softmax_logits(logits)
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
            _calibration_loss_gradient,
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
    best = min(fits, key=lambda result: result.fun)
    return best.x


def _unpack_pairwise(
    parameters: np.ndarray, *, num_models: int, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = num_models * dimension
    judges = parameters[:size].reshape(num_models, dimension)
    models = parameters[size : 2 * size].reshape(num_models, dimension)
    ties = parameters[2 * size :]
    return judges, models, ties


def pairwise_logits(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_models: int,
    dimension: int,
) -> np.ndarray:
    """Compute Davidson logits from fitted pairwise parameters."""
    judge, left, right, _ = rows.T
    judges, models, ties = _unpack_pairwise(
        parameters, num_models=num_models, dimension=dimension
    )
    left_score = np.sum(judges[judge] * models[left], axis=1)
    right_score = np.sum(judges[judge] * models[right], axis=1)
    return np.column_stack(
        (ties[judge] + 0.5 * (left_score + right_score), left_score, right_score)
    )


def classification_metrics(logits: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    probabilities = _softmax_logits(logits)
    index = np.arange(len(choices))
    return {
        "nll": float(
            -np.log(np.maximum(probabilities[index, choices], 1e-300)).mean()
        ),
        "accuracy": float(np.mean(np.argmax(logits, axis=1) == choices)),
    }


def pairwise_prediction_cross_validation(
    ratings: np.ndarray,
    scenario_ids: np.ndarray,
    blocks: dict[int, np.ndarray],
    *,
    folds: list[np.ndarray],
    dimension: int,
    starts: int,
    seed: int,
) -> list[dict[str, float | int | bool]]:
    """Compare direct-score and BTD predictions on held-out scenarios."""
    all_positions = np.arange(len(scenario_ids))
    output: list[dict[str, float | int | bool]] = []
    for fold_index, test_positions in enumerate(folds):
        train_positions = np.setdiff1d(
            all_positions, test_positions, assume_unique=True
        )
        train_ratings, _, _ = standardize_ratings(
            ratings[:, train_positions, :], ratings[:, test_positions, :]
        )
        direct_scores = fit_rank_svd(train_ratings.mean(axis=1), dimension).scores
        train_scenarios = scenario_ids[train_positions]
        test_scenarios = scenario_ids[test_positions]
        train_rows = concatenate_blocks(blocks, train_scenarios)
        test_rows = concatenate_blocks(blocks, test_scenarios)

        calibration_parameters = fit_direct_calibration(direct_scores, train_rows)
        direct_metrics = classification_metrics(
            _fixed_score_logits(calibration_parameters, direct_scores, test_rows),
            test_rows[:, 3],
        )
        pairwise = fit_pairwise_btd_multistart(
            train_rows,
            initials=[
                initial_btd_parameters(
                    seed + 10_000 * fold_index + start,
                    num_models=ratings.shape[0],
                    dimension=dimension,
                )
                for start in range(starts)
            ],
            num_models=ratings.shape[0],
            dimension=dimension,
            reject_unstable=False,
        )
        pairwise_metrics = classification_metrics(
            pairwise_logits(
                pairwise.parameters,
                test_rows,
                num_models=ratings.shape[0],
                dimension=dimension,
            ),
            test_rows[:, 3],
        )
        output.append(
            {
                "fold": fold_index,
                "train_scenarios": len(train_scenarios),
                "test_scenarios": len(test_scenarios),
                "test_rows": len(test_rows),
                "direct_nll": direct_metrics["nll"],
                "direct_accuracy": direct_metrics["accuracy"],
                "pairwise_nll": pairwise_metrics["nll"],
                "pairwise_accuracy": pairwise_metrics["accuracy"],
                "nll_gap_direct_minus_pairwise": (
                    direct_metrics["nll"] - pairwise_metrics["nll"]
                ),
                "accuracy_gap_direct_minus_pairwise": (
                    direct_metrics["accuracy"] - pairwise_metrics["accuracy"]
                ),
                "direct_scale": float(np.exp(calibration_parameters[0])),
                "pairwise_train_nll": pairwise.loss,
                "pairwise_stable_near_optima": pairwise.stable_near_optima,
            }
        )
    return output


def geometry_metrics(direct: np.ndarray, pairwise: np.ndarray) -> dict[str, float]:
    """Compare row-centred score matrices and their implied pairwise margins."""
    direct = direct - direct.mean(axis=1, keepdims=True)
    pairwise = pairwise - pairwise.mean(axis=1, keepdims=True)
    direct_margins = []
    pairwise_margins = []
    for judge in range(direct.shape[0]):
        for left in range(direct.shape[1]):
            for right in range(left + 1, direct.shape[1]):
                direct_margins.append(direct[judge, left] - direct[judge, right])
                pairwise_margins.append(pairwise[judge, left] - pairwise[judge, right])
    direct_margins = np.asarray(direct_margins)
    pairwise_margins = np.asarray(pairwise_margins)
    slope = float(np.sum(direct * pairwise) / np.sum(np.square(direct)))
    return {
        "score_pearson": float(pearsonr(direct.ravel(), pairwise.ravel()).statistic),
        "score_spearman": float(
            spearmanr(direct.ravel(), pairwise.ravel()).statistic
        ),
        "margin_pearson": float(pearsonr(direct_margins, pairwise_margins).statistic),
        "margin_spearman": float(
            spearmanr(direct_margins, pairwise_margins).statistic
        ),
        "margin_sign_agreement": float(
            np.mean(np.sign(direct_margins) == np.sign(pairwise_margins))
        ),
        "best_scale_direct_to_pairwise": slope,
        "scaled_relative_frobenius_error": float(
            np.linalg.norm(slope * direct - pairwise) / np.linalg.norm(pairwise)
        ),
    }


def aligned_embedding_metrics(
    direct_scores: np.ndarray, pairwise_scores: np.ndarray, rank: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize score matrices, align factors, and compare latent distances."""
    direct = direct_scores - direct_scores.mean(axis=1, keepdims=True)
    pairwise = pairwise_scores - pairwise_scores.mean(axis=1, keepdims=True)
    direct /= np.linalg.norm(direct)
    pairwise /= np.linalg.norm(pairwise)
    direct_fit = fit_rank_svd(direct, rank)
    pairwise_fit = fit_rank_svd(pairwise, rank)
    direct_stack = np.vstack((direct_fit.judges, direct_fit.models))
    pairwise_stack = np.vstack((pairwise_fit.judges, pairwise_fit.models))
    rotation, _ = orthogonal_procrustes(direct_stack, pairwise_stack)
    aligned_judges = direct_fit.judges @ rotation
    aligned_models = direct_fit.models @ rotation
    aligned_stack = np.vstack((aligned_judges, aligned_models))
    difference = aligned_stack - pairwise_stack
    metrics = {
        "aligned_relative_error": float(
            np.linalg.norm(difference) / np.linalg.norm(pairwise_stack)
        ),
        "judge_distance_spearman": float(
            spearmanr(pdist(aligned_judges), pdist(pairwise_fit.judges)).statistic
        ),
        "model_distance_spearman": float(
            spearmanr(pdist(aligned_models), pdist(pairwise_fit.models)).statistic
        ),
    }
    return (
        metrics,
        aligned_judges,
        aligned_models,
        pairwise_fit.judges,
        pairwise_fit.models,
    )


def shared_model_metrics(
    reference_scores: np.ndarray, candidate_scores: np.ndarray, rank: int
) -> dict[str, float]:
    """Compare canonical model vectors from score matrices with different rows."""
    reference = reference_scores / np.linalg.norm(reference_scores)
    candidate = candidate_scores / np.linalg.norm(candidate_scores)
    reference_models = fit_rank_svd(reference, rank).models
    candidate_models = fit_rank_svd(candidate, rank).models
    rotation, _ = orthogonal_procrustes(candidate_models, reference_models)
    candidate_aligned = candidate_models @ rotation
    return {
        "aligned_relative_error": float(
            np.linalg.norm(candidate_aligned - reference_models)
            / np.linalg.norm(reference_models)
        ),
        "distance_spearman": float(
            spearmanr(
                pdist(candidate_aligned), pdist(reference_models)
            ).statistic
        ),
    }


def numerical_bootstrap(
    ratings: np.ndarray,
    *,
    point_scores: np.ndarray,
    rank: int,
    samples: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Resample scenarios and measure stability of the direct score matrix."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, float | int]] = []
    for draw in range(samples):
        positions = rng.integers(0, ratings.shape[1], size=ratings.shape[1])
        sampled, _, _ = standardize_ratings(
            ratings[:, positions, :], ratings[:, positions, :]
        )
        scores = fit_rank_svd(sampled.mean(axis=1), rank).scores
        output.append(
            {
                "draw": draw,
                "score_pearson": float(
                    pearsonr(scores.ravel(), point_scores.ravel()).statistic
                ),
                "relative_frobenius_error": float(
                    np.linalg.norm(scores - point_scores) / np.linalg.norm(point_scores)
                ),
            }
        )
    return output


def _mean_se(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows])
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def _rank_mse_reduction(summary: list[dict[str, float | int]], rank: int) -> float:
    mse = {int(row["rank"]): float(row["mean_test_mse"]) for row in summary}
    return 1.0 - mse[rank] / mse[0]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(args: argparse.Namespace) -> dict:
    ratings, all_scenario_ids, model_names = load_numerical_tensor(args.ratings)
    (
        criterion_ratings,
        criterion_ids,
        criterion_hashes,
        criterion_scenario_ids,
        criterion_judge_names,
        criterion_model_names,
    ) = load_criterion_tensor(args.criterion_ratings)
    if criterion_scenario_ids != all_scenario_ids:
        raise ValueError("Whole and criterion ratings use different scenarios")
    if criterion_model_names != model_names:
        raise ValueError(
            "Whole and criterion ratings use different target-model rosters"
        )

    pairwise_data = load_pairwise_data(
        args.evaluations, num_criteria=8, cleaning="corrected"
    )
    matched_scenarios = np.asarray(sorted(pairwise_data.blocks), dtype=np.int64)
    scenario_position = {
        scenario: position for position, scenario in enumerate(all_scenario_ids)
    }
    matched_positions = np.asarray(
        [scenario_position[int(scenario)] for scenario in matched_scenarios]
    )
    matched_ratings = ratings[:, matched_positions, :]
    if len(matched_scenarios) != 871:
        raise ValueError(
            f"Expected 871 matched scenarios, found {len(matched_scenarios)}"
        )

    rank_folds = scenario_folds(ratings.shape[1], args.rank_folds, args.seed)
    rank_rows = rating_rank_cross_validation(
        ratings,
        folds=rank_folds,
        max_rank=min(ratings.shape[0], ratings.shape[2]) - 1,
    )
    rank_summary = summarize_rank_cv(rank_rows)
    criterion_flattened = criterion_ratings.reshape(
        criterion_ratings.shape[0] * criterion_ratings.shape[1],
        criterion_ratings.shape[2],
        criterion_ratings.shape[3],
    )
    criterion_rank_rows = rating_rank_cross_validation(
        criterion_flattened,
        folds=rank_folds,
        max_rank=criterion_ratings.shape[3] - 1,
    )
    criterion_rank_summary = summarize_rank_cv(criterion_rank_rows)

    full_standardized, _, full_judge_scales = standardize_ratings(ratings, ratings)
    full_direct_fit = fit_rank_svd(
        full_standardized.mean(axis=1), args.dimension
    )
    full_explained = float(
        np.square(full_direct_fit.singular_values[: args.dimension]).sum()
        / np.square(full_direct_fit.singular_values).sum()
    )
    criterion_fit, criterion_affinity, criterion_judge_scales = fit_criterion_svd(
        criterion_ratings, args.dimension
    )
    criterion_explained = float(
        np.square(criterion_fit.singular_values[: args.dimension]).sum()
        / np.square(criterion_fit.singular_values).sum()
    )
    criterion_score_tensor = criterion_fit.scores.reshape(
        criterion_ratings.shape[0],
        criterion_ratings.shape[1],
        criterion_ratings.shape[3],
    )
    criterion_explained_by_id = {}
    for criterion, criterion_id in enumerate(criterion_ids):
        row_slice = slice(
            criterion * criterion_ratings.shape[1],
            (criterion + 1) * criterion_ratings.shape[1],
        )
        criterion_explained_by_id[criterion_id] = float(
            1.0
            - np.square(
                criterion_affinity[row_slice] - criterion_fit.scores[row_slice]
            ).sum()
            / np.square(criterion_affinity[row_slice]).sum()
        )
    criterion_mean_scores = criterion_score_tensor.mean(axis=0)
    whole_criterion_geometry = geometry_metrics(
        full_direct_fit.scores, criterion_mean_scores
    )
    whole_criterion_models = shared_model_metrics(
        full_direct_fit.scores, criterion_fit.scores, args.dimension
    )
    standardized, _, matched_judge_scales = standardize_ratings(
        matched_ratings, matched_ratings
    )
    affinity = standardized.mean(axis=1)
    matched_direct_fit = fit_rank_svd(affinity, args.dimension)
    matched_explained = float(
        np.square(matched_direct_fit.singular_values[: args.dimension]).sum()
        / np.square(matched_direct_fit.singular_values).sum()
    )

    pairwise_rows = concatenate_blocks(pairwise_data.blocks, matched_scenarios)
    pairwise_fit = fit_pairwise_btd_multistart(
        pairwise_rows,
        initials=[
            initial_btd_parameters(
                args.seed + start,
                num_models=len(model_names),
                dimension=args.dimension,
            )
            for start in range(max(6, args.pairwise_starts))
        ],
        num_models=len(model_names),
        dimension=args.dimension,
    )
    pairwise_judges, pairwise_models, _ = _unpack_pairwise(
        pairwise_fit.parameters,
        num_models=len(model_names),
        dimension=args.dimension,
    )
    pairwise_scores = pairwise_judges @ pairwise_models.T
    geometry = geometry_metrics(matched_direct_fit.scores, pairwise_scores)
    (
        aligned_metrics,
        direct_judges_aligned,
        direct_models_aligned,
        pairwise_judges_canonical,
        pairwise_models_canonical,
    ) = aligned_embedding_metrics(
        matched_direct_fit.scores, pairwise_scores, args.dimension
    )

    prediction_folds = scenario_folds(
        len(matched_scenarios), args.pairwise_folds, args.seed + 1
    )
    prediction_rows = pairwise_prediction_cross_validation(
        matched_ratings,
        matched_scenarios,
        pairwise_data.blocks,
        folds=prediction_folds,
        dimension=args.dimension,
        starts=args.pairwise_starts,
        seed=args.seed,
    )
    bootstrap_rows = numerical_bootstrap(
        ratings,
        point_scores=full_direct_fit.scores,
        rank=args.dimension,
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )
    criterion_bootstrap_rows = numerical_bootstrap(
        criterion_flattened,
        point_scores=criterion_fit.scores,
        rank=args.dimension,
        samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "rank_cv_folds.csv", rank_rows)
    _write_csv(output_dir / "rank_cv_summary.csv", rank_summary)
    _write_csv(output_dir / "criterion_rank_cv_folds.csv", criterion_rank_rows)
    _write_csv(output_dir / "criterion_rank_cv_summary.csv", criterion_rank_summary)
    _write_csv(output_dir / "pairwise_prediction_cv.csv", prediction_rows)
    _write_csv(output_dir / "numerical_bootstrap.csv", bootstrap_rows)
    _write_csv(
        output_dir / "criterion_numerical_bootstrap.csv",
        criterion_bootstrap_rows,
    )

    score_rows = []
    embedding_rows = []
    for judge, judge_name in enumerate(model_names):
        for model, model_name in enumerate(model_names):
            score_rows.append(
                {
                    "judge_id": judge,
                    "judge_name": judge_name,
                    "model_id": model,
                    "model_name": model_name,
                    "direct_full_score": full_direct_fit.scores[judge, model],
                    "direct_matched_score": matched_direct_fit.scores[judge, model],
                    "pairwise_score": pairwise_scores[judge, model],
                }
            )
        for axis in range(args.dimension):
            embedding_rows.extend(
                (
                    {
                        "method": "direct_full",
                        "kind": "judge",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": full_direct_fit.judges[judge, axis],
                    },
                    {
                        "method": "direct_full",
                        "kind": "model",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": full_direct_fit.models[judge, axis],
                    },
                    {
                        "method": "direct_aligned",
                        "kind": "judge",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": direct_judges_aligned[judge, axis],
                    },
                    {
                        "method": "direct_aligned",
                        "kind": "model",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": direct_models_aligned[judge, axis],
                    },
                    {
                        "method": "pairwise_canonical",
                        "kind": "judge",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": pairwise_judges_canonical[judge, axis],
                    },
                    {
                        "method": "pairwise_canonical",
                        "kind": "model",
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": pairwise_models_canonical[judge, axis],
                    },
                )
            )
    _write_csv(output_dir / "score_matrices.csv", score_rows)
    _write_csv(output_dir / "aligned_embeddings.csv", embedding_rows)

    criterion_score_rows = []
    criterion_embedding_rows = []
    for criterion, (criterion_id, criterion_hash) in enumerate(
        zip(criterion_ids, criterion_hashes)
    ):
        for judge, judge_name in enumerate(criterion_judge_names):
            row = criterion * len(criterion_judge_names) + judge
            for model, model_name in enumerate(model_names):
                criterion_score_rows.append(
                    {
                        "criterion_id": criterion_id,
                        "criterion_hash": criterion_hash,
                        "judge_id": judge,
                        "judge_name": judge_name,
                        "model_id": model,
                        "model_name": model_name,
                        "standardized_mean_rating": criterion_affinity[row, model],
                        "rank_score": criterion_fit.scores[row, model],
                        "residual": (
                            criterion_affinity[row, model]
                            - criterion_fit.scores[row, model]
                        ),
                    }
                )
            for axis in range(args.dimension):
                criterion_embedding_rows.append(
                    {
                        "kind": "criterion_judge",
                        "criterion_id": criterion_id,
                        "criterion_hash": criterion_hash,
                        "id": judge,
                        "name": judge_name,
                        "axis": axis,
                        "value": criterion_fit.judges[row, axis],
                    }
                )
    for model, model_name in enumerate(model_names):
        for axis in range(args.dimension):
            criterion_embedding_rows.append(
                {
                    "kind": "shared_model",
                    "criterion_id": "",
                    "criterion_hash": "",
                    "id": model,
                    "name": model_name,
                    "axis": axis,
                    "value": criterion_fit.models[model, axis],
                }
            )
    _write_csv(output_dir / "criterion_score_matrix.csv", criterion_score_rows)
    _write_csv(output_dir / "criterion_embeddings.csv", criterion_embedding_rows)

    bootstrap_correlations = np.asarray(
        [float(row["score_pearson"]) for row in bootstrap_rows]
    )
    bootstrap_errors = np.asarray(
        [float(row["relative_frobenius_error"]) for row in bootstrap_rows]
    )
    criterion_bootstrap_correlations = np.asarray(
        [float(row["score_pearson"]) for row in criterion_bootstrap_rows]
    )
    criterion_bootstrap_errors = np.asarray(
        [
            float(row["relative_frobenius_error"])
            for row in criterion_bootstrap_rows
        ]
    )
    results = {
        "data": {
            "ratings_path": str(args.ratings),
            "ratings_sha256": _sha256_file(args.ratings),
            "criterion_ratings_path": str(args.criterion_ratings),
            "criterion_ratings_sha256": _sha256_file(args.criterion_ratings),
            "evaluations_path": str(args.evaluations),
            "evaluations_sha256": _sha256_file(args.evaluations),
            "ratings_shape": list(ratings.shape),
            "criterion_ratings_shape": list(criterion_ratings.shape),
            "criterion_ids": criterion_ids,
            "criterion_hashes": criterion_hashes,
            "criterion_judge_names": criterion_judge_names,
            "matched_scenarios": len(matched_scenarios),
            "pairwise_rows": len(pairwise_rows),
            "model_names": model_names,
            "pairwise_diagnostics": pairwise_data.diagnostics,
        },
        "settings": {
            "dimension": args.dimension,
            "dimension_reason": (
                "Matched to the existing rank-two BTD; rank CV is diagnostic."
            ),
            "rank_folds": args.rank_folds,
            "pairwise_folds": args.pairwise_folds,
            "pairwise_starts": args.pairwise_starts,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "direct_factorization": {
            "full_singular_values": full_direct_fit.singular_values.tolist(),
            "full_rank_explained_fraction": full_explained,
            "full_judge_scales": full_judge_scales.tolist(),
            "matched_singular_values": matched_direct_fit.singular_values.tolist(),
            "matched_rank_explained_fraction": matched_explained,
            "matched_judge_scales": matched_judge_scales.tolist(),
            "full_matched_score_pearson": float(
                pearsonr(
                    full_direct_fit.scores.ravel(),
                    matched_direct_fit.scores.ravel(),
                ).statistic
            ),
            "full_matched_relative_frobenius_error": float(
                np.linalg.norm(full_direct_fit.scores - matched_direct_fit.scores)
                / np.linalg.norm(full_direct_fit.scores)
            ),
            "heldout_mse_reduction_vs_rank_0": _rank_mse_reduction(
                rank_summary, args.dimension
            ),
            "rank_cv": rank_summary,
        },
        "criterion_factorization": {
            "architecture": "one vector per criterion-judge row; shared model vectors",
            "singular_values": criterion_fit.singular_values.tolist(),
            "rank_explained_fraction": criterion_explained,
            "rank_explained_fraction_by_criterion": criterion_explained_by_id,
            "criterion_judge_scales": criterion_judge_scales.tolist(),
            "heldout_mse_reduction_vs_rank_0": _rank_mse_reduction(
                criterion_rank_summary, args.dimension
            ),
            "rank_cv": criterion_rank_summary,
            "whole_score_consistency": whole_criterion_geometry,
            "whole_shared_model_consistency": whole_criterion_models,
            "whole_score_consistency_note": (
                "Criterion judge ID 3 is Grok 4.3; whole-rating judge ID 3 is Grok 4."
            ),
        },
        "full_pairwise_fit": {
            "train_nll": pairwise_fit.loss,
            "attempted_starts": pairwise_fit.attempted_starts,
            "successful_starts": pairwise_fit.successful_starts,
            "near_optimal_starts": pairwise_fit.near_optimal_starts,
            "stable_near_optima": pairwise_fit.stable_near_optima,
            "max_near_optimal_trust_l1": pairwise_fit.max_near_optimal_trust_l1,
        },
        "geometry_comparison": {**geometry, **aligned_metrics},
        "heldout_pairwise_prediction": {
            "direct_nll": _mean_se(prediction_rows, "direct_nll"),
            "pairwise_nll": _mean_se(prediction_rows, "pairwise_nll"),
            "nll_gap_direct_minus_pairwise": _mean_se(
                prediction_rows, "nll_gap_direct_minus_pairwise"
            ),
            "direct_accuracy": _mean_se(prediction_rows, "direct_accuracy"),
            "pairwise_accuracy": _mean_se(prediction_rows, "pairwise_accuracy"),
            "accuracy_gap_direct_minus_pairwise": _mean_se(
                prediction_rows, "accuracy_gap_direct_minus_pairwise"
            ),
        },
        "direct_bootstrap_stability": {
            "score_pearson_quantiles": np.quantile(
                bootstrap_correlations, [0.025, 0.5, 0.975]
            ).tolist(),
            "relative_frobenius_error_quantiles": np.quantile(
                bootstrap_errors, [0.025, 0.5, 0.975]
            ).tolist(),
        },
        "criterion_bootstrap_stability": {
            "score_pearson_quantiles": np.quantile(
                criterion_bootstrap_correlations, [0.025, 0.5, 0.975]
            ).tolist(),
            "relative_frobenius_error_quantiles": np.quantile(
                criterion_bootstrap_errors, [0.025, 0.5, 0.975]
            ).tolist(),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument(
        "--criterion-ratings", type=Path, default=DEFAULT_CRITERION_RATINGS
    )
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--rank-folds", type=int, default=10)
    parser.add_argument("--pairwise-folds", type=int, default=5)
    parser.add_argument("--pairwise-starts", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dimension < 1 or args.dimension > 7:
        parser.error("--dimension must be between 1 and 7")
    if args.pairwise_starts < 2:
        parser.error("--pairwise-starts must be at least 2")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


def main() -> None:
    results = run_analysis(parse_args())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
