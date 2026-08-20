"""Compare whole and criterion direct-rating embeddings with pairwise BTD.

Each direct model is the global rank-d least-squares factorization of its
scenario-averaged standardized rating matrix. Pairwise comparisons use the
existing deterministic Davidson fitter. Scenario splits are shared throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from numerical_rating.analysis.data_loading.criterion_ratings import (
    load_criterion_tensor,
)
from numerical_rating.analysis.data_loading.numerical_ratings import (
    load_numerical_tensor,
)
from numerical_rating.analysis.data_loading.pairwise_judgments import (
    concatenate_blocks,
    load_pairwise_data,
)
from numerical_rating.analysis.evaluation.embedding_comparison import (
    aligned_embedding_metrics,
    geometry_metrics,
    shared_model_metrics,
)
from numerical_rating.analysis.evaluation.scenario_splits import scenario_folds
from numerical_rating.analysis.evaluation.trit_calibration import (
    classification_metrics,
    fit_direct_calibration,
    fixed_score_logits,
    pairwise_logits,
)
from numerical_rating.analysis.model_fitting.davidson_likelihood import (
    unpack_parameters,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    fit_criterion_svd,
    fit_rank_svd,
    standardize_ratings,
)
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    fit_pairwise_btd_multistart,
    initial_btd_parameters,
)
from numerical_rating.analysis.utilities.artifact_io import (
    atomic_csv,
    atomic_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[4]
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
    ROOT / "data/output/numerical_rating/kindness_1000_round_robin/"
    "direct_embedding_comparison"
)


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
            fixed_score_logits(calibration_parameters, direct_scores, test_rows),
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
    full_direct_fit = fit_rank_svd(full_standardized.mean(axis=1), args.dimension)
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
    pairwise_judges, pairwise_models, _ = unpack_parameters(
        pairwise_fit.parameters,
        num_rows=len(model_names),
        num_models=len(model_names),
        dimension=args.dimension,
        center_models=False,
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
    atomic_csv(output_dir / "rank_cv_folds.csv", list(rank_rows[0]), rank_rows)
    atomic_csv(output_dir / "rank_cv_summary.csv", list(rank_summary[0]), rank_summary)
    atomic_csv(
        output_dir / "criterion_rank_cv_folds.csv",
        list(criterion_rank_rows[0]),
        criterion_rank_rows,
    )
    atomic_csv(
        output_dir / "criterion_rank_cv_summary.csv",
        list(criterion_rank_summary[0]),
        criterion_rank_summary,
    )
    atomic_csv(
        output_dir / "pairwise_prediction_cv.csv",
        list(prediction_rows[0]),
        prediction_rows,
    )
    atomic_csv(
        output_dir / "numerical_bootstrap.csv",
        list(bootstrap_rows[0]),
        bootstrap_rows,
    )
    atomic_csv(
        output_dir / "criterion_numerical_bootstrap.csv",
        list(criterion_bootstrap_rows[0]),
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
    atomic_csv(output_dir / "score_matrices.csv", list(score_rows[0]), score_rows)
    atomic_csv(
        output_dir / "aligned_embeddings.csv",
        list(embedding_rows[0]),
        embedding_rows,
    )

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
    atomic_csv(
        output_dir / "criterion_score_matrix.csv",
        list(criterion_score_rows[0]),
        criterion_score_rows,
    )
    atomic_csv(
        output_dir / "criterion_embeddings.csv",
        list(criterion_embedding_rows[0]),
        criterion_embedding_rows,
    )

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
        [float(row["relative_frobenius_error"]) for row in criterion_bootstrap_rows]
    )
    results = {
        "data": {
            "ratings_path": str(args.ratings),
            "ratings_sha256": sha256_file(args.ratings),
            "criterion_ratings_path": str(args.criterion_ratings),
            "criterion_ratings_sha256": sha256_file(args.criterion_ratings),
            "evaluations_path": str(args.evaluations),
            "evaluations_sha256": sha256_file(args.evaluations),
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
    atomic_json(output_dir / "results.json", results)
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
