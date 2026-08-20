"""Run the matched direct-rating versus criterion-BTD comparison."""

from __future__ import annotations

import argparse

import numpy as np

from numerical_rating.analysis.data_loading.matched_criterion_data import (
    load_matched_criterion_data,
)
from numerical_rating.analysis.evaluation.embedding_comparison import (
    aligned_embedding_metrics,
    geometry_metrics,
)
from numerical_rating.analysis.model_fitting.direct_rating_factorization import (
    standardize_ratings,
)
from numerical_rating.analysis.utilities.artifact_io import (
    atomic_csv,
    atomic_json,
    sha256_file,
)

from .heldout_evaluation import (
    cross_validate,
    fit_models,
    mean_se,
    reliability_rows,
    rows_and_targets,
    svd_rank_curve,
)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    data = load_matched_criterion_data(args.ratings, args.evaluations)
    if len(data.scenario_ids) != 871:
        raise ValueError(
            f"Expected 871 matched scenarios, found {len(data.scenario_ids)}"
        )
    if data.diagnostics["matched_criterion_rows"] != 102_437:
        raise ValueError(
            "Expected 102437 exact-response criterion rows, found "
            f"{data.diagnostics['matched_criterion_rows']}"
        )

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
    standardized, _, scales = standardize_ratings(matched, matched)
    rows, targets = rows_and_targets(data.blocks, data.scenario_ids, standardized)
    direct, calibration, btd = fit_models(
        standardized,
        rows,
        targets,
        dimension=args.dimension,
        starts=max(args.starts, 4),
        seed=args.seed,
        max_iterations=args.max_iterations,
        direct_method=args.direct_method,
    )
    folds, heldout_choices, direct_probabilities, btd_probabilities = cross_validate(
        data,
        folds=args.folds,
        dimension=args.dimension,
        starts=args.starts,
        seed=args.seed + 1,
        max_iterations=args.max_iterations,
        direct_method=args.direct_method,
    )
    reliability = reliability_rows(
        heldout_choices,
        direct_probabilities,
        btd_probabilities,
    )
    rank_rows, rank_summary = svd_rank_curve(
        data,
        folds=args.folds,
        seed=args.seed + 1,
    )
    geometry = geometry_metrics(direct.scores, btd.scores)
    aligned, _, _, _, _ = aligned_embedding_metrics(
        direct.scores,
        btd.scores,
        args.dimension,
    )
    results: dict[str, object] = {
        "data": {
            "ratings_path": str(args.ratings),
            "ratings_sha256": sha256_file(args.ratings),
            "evaluations_path": str(args.evaluations),
            "evaluations_sha256": sha256_file(args.evaluations),
            "rating_shape": list(data.ratings.shape),
            "criterion_ids": data.criterion_ids,
            "judge_names": data.judge_names,
            "model_names": data.model_names,
            **data.diagnostics,
        },
        "settings": {
            "dimension": args.dimension,
            "folds": args.folds,
            "starts": args.starts,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "direct_method": args.direct_method,
            "judge_slot_note": (
                "Judge slot 3 is Grok 4.3 for criterion ratings and Grok 4 "
                "for pairwise trits."
            ),
        },
        "full_fit": {
            "direct_train_mse": direct.loss,
            "direct_solver": (
                "truncated_svd"
                if args.direct_method == "svd"
                else "matched_margin_lbfgs"
            ),
            "btd_train_nll": btd.loss,
            "direct_scale_for_trits": float(np.exp(calibration[0])),
            "direct_gradient_l2": direct.gradient_l2,
            "btd_gradient_l2": btd.gradient_l2,
            "direct_successful_starts": direct.successful_starts,
            "btd_successful_starts": btd.successful_starts,
            "direct_min_near_score_correlation": (direct.min_near_score_correlation),
            "btd_min_near_score_correlation": btd.min_near_score_correlation,
            "direct_starts": list(direct.start_diagnostics),
            "btd_starts": list(btd.start_diagnostics),
            "direct_failed_starts": list(direct.failed_starts),
            "btd_failed_starts": list(btd.failed_starts),
            "criterion_judge_scales": scales.tolist(),
        },
        "geometry": {**geometry, **aligned},
        "heldout": {
            "direct_mse": mean_se(folds, "direct_test_mse"),
            "direct_mse_reduction_vs_zero": mean_se(
                folds,
                "direct_test_mse_reduction_vs_zero",
            ),
            "direct_nll": mean_se(folds, "direct_nll"),
            "btd_nll": mean_se(folds, "btd_nll"),
            "nll_gap_direct_minus_btd": mean_se(
                folds,
                "nll_gap_direct_minus_btd",
            ),
            "direct_accuracy": mean_se(folds, "direct_accuracy"),
            "btd_accuracy": mean_se(folds, "btd_accuracy"),
            "direct_tie_ece": mean_se(folds, "direct_tie_ece"),
            "btd_tie_ece": mean_se(folds, "btd_tie_ece"),
            "direct_win_loss_ece": mean_se(folds, "direct_win_loss_ece"),
            "btd_win_loss_ece": mean_se(folds, "btd_win_loss_ece"),
            "score_pearson": mean_se(folds, "score_pearson"),
            "margin_pearson": mean_se(folds, "margin_pearson"),
            "margin_sign_agreement": mean_se(folds, "margin_sign_agreement"),
        },
        "svd_rank_curve": rank_summary,
    }

    atomic_json(args.output / "results.json", results)
    atomic_csv(args.output / "folds.csv", list(folds[0]), folds)
    atomic_csv(
        args.output / "reliability.csv",
        list(reliability[0]),
        reliability,
    )
    atomic_csv(
        args.output / "svd_rank_curve.csv",
        list(rank_rows[0]),
        rank_rows,
    )
    optimization_rows = [
        {"model": model_name, "start": start["start"], **point}
        for model_name, diagnostics in (
            ("direct", direct.start_diagnostics),
            ("btd", btd.start_diagnostics),
        )
        for start in diagnostics
        for point in start["trace"]
    ]
    atomic_csv(
        args.output / "optimization_curves.csv",
        list(optimization_rows[0]),
        optimization_rows,
    )
    score_rows = []
    for row_index in range(direct.scores.shape[0]):
        criterion = row_index // len(data.judge_names)
        judge = row_index % len(data.judge_names)
        for model in range(direct.scores.shape[1]):
            score_rows.append(
                {
                    "criterion_id": data.criterion_ids[criterion],
                    "judge_id": judge,
                    "judge_name": data.judge_names[judge],
                    "model_id": model,
                    "model_name": data.model_names[model],
                    "direct_score": direct.scores[row_index, model],
                    "btd_score": btd.scores[row_index, model],
                }
            )
    atomic_csv(
        args.output / "score_matrices.csv",
        list(score_rows[0]),
        score_rows,
    )
    return results
