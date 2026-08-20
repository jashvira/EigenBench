"""Run paired scenario uncertainty analysis for direct and pairwise rankings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from numerical_rating.analysis.aggregation.eigentrust import (
    compute_trust,
    eigentrust_elo,
)
from numerical_rating.analysis.data_loading.numerical_ratings import (
    load_numerical_tensor,
)
from numerical_rating.analysis.data_loading.pairwise_judgments import (
    PAIRWISE_CLEANING,
    concatenate_blocks,
    load_pairwise_data,
)
from numerical_rating.analysis.experiments.scenario_uncertainty.resampling import (
    delete_group_jackknife,
    paired_scenario_bootstrap,
    ranking_metrics,
)
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    fit_diagnostics,
    fit_pairwise_btd_multistart,
    initial_btd_parameters,
)
from numerical_rating.analysis.utilities.artifact_io import (
    atomic_csv,
    atomic_json,
    sha256_file,
)
from numerical_rating.collection.data import provenance_path


def run_analysis(
    *,
    ratings_path: Path,
    evaluations_path: Path,
    meta_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    jackknife_groups: int,
    workers: int,
    seed: int,
) -> dict:
    """Run paired scenario bootstrap and delete-group jackknife analyses."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if jackknife_groups < 2:
        raise ValueError("jackknife_groups must be at least two")
    if workers < 1:
        raise ValueError("workers must be positive")

    ratings, all_scenarios, model_names = load_numerical_tensor(ratings_path)
    pairwise_data = load_pairwise_data(
        evaluations_path,
        num_criteria=8,
        cleaning=PAIRWISE_CLEANING,
    )
    legacy_data = load_pairwise_data(
        evaluations_path,
        num_criteria=8,
        cleaning="published_legacy",
    )
    blocks = pairwise_data.blocks
    legacy_blocks = legacy_data.blocks
    matched_scenarios = np.asarray(sorted(blocks), dtype=np.int64)
    scenario_position = {
        scenario: index for index, scenario in enumerate(all_scenarios)
    }
    matched_positions = np.asarray(
        [scenario_position[int(scenario)] for scenario in matched_scenarios]
    )
    matched_ratings = ratings[:, matched_positions, :]
    if len(matched_scenarios) != 871:
        raise ValueError(
            f"Expected 871 matched scenarios, found {len(matched_scenarios)}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    published_pairwise = np.asarray(meta["eigentrust"], dtype=float)
    published_pairwise /= published_pairwise.sum()
    full_pairwise_rows = concatenate_blocks(blocks, matched_scenarios)
    pairwise_point = fit_pairwise_btd_multistart(
        full_pairwise_rows,
        initials=[initial_btd_parameters(seed + start) for start in range(6)],
    )
    legacy_scenarios = np.asarray(sorted(legacy_blocks), dtype=np.int64)
    if not np.array_equal(legacy_scenarios, matched_scenarios):
        raise ValueError("Legacy cleaning changed the matched scenario support")
    legacy_pairwise_rows = concatenate_blocks(legacy_blocks, legacy_scenarios)
    legacy_pairwise_point = fit_pairwise_btd_multistart(
        legacy_pairwise_rows,
        initials=[pairwise_point.parameters]
        + [initial_btd_parameters(seed + 100 + start) for start in range(5)],
    )
    numerical_point = compute_trust(matched_ratings).trust
    input_hashes = {
        "ratings": sha256_file(ratings_path),
        "evaluations": sha256_file(evaluations_path),
        "meta": sha256_file(meta_path),
    }
    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "inputs": input_hashes,
                "pairwise_cleaning": PAIRWISE_CLEANING,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    bootstrap = paired_scenario_bootstrap(
        matched_ratings=matched_ratings,
        matched_scenarios=matched_scenarios,
        blocks=blocks,
        pairwise_point=pairwise_point,
        output_dir=output_dir,
        bootstrap_samples=bootstrap_samples,
        workers=workers,
        seed=seed,
        input_fingerprint=input_fingerprint,
    )
    numerical_draws = bootstrap.numerical_draws
    pairwise_draws = bootstrap.pairwise_draws
    numerical_lower, numerical_upper = np.quantile(
        numerical_draws,
        [0.025, 0.975],
        axis=0,
    )
    pairwise_lower, pairwise_upper = np.quantile(
        pairwise_draws,
        [0.025, 0.975],
        axis=0,
    )
    numerical_ranks = (-numerical_draws).argsort(axis=1).argsort(axis=1) + 1
    pairwise_ranks = (-pairwise_draws).argsort(axis=1).argsort(axis=1) + 1
    numerical_rank_lower, numerical_rank_upper = np.quantile(
        numerical_ranks,
        [0.025, 0.975],
        axis=0,
    )
    pairwise_rank_lower, pairwise_rank_upper = np.quantile(
        pairwise_ranks,
        [0.025, 0.975],
        axis=0,
    )
    paired_difference = numerical_draws - pairwise_draws
    paired_difference_lower, paired_difference_upper = np.quantile(
        paired_difference,
        [0.025, 0.975],
        axis=0,
    )
    bootstrap_metrics = [
        {
            "draw": index,
            **ranking_metrics(numerical_draws[index], pairwise_draws[index]),
        }
        for index in range(bootstrap_samples)
    ]

    jackknife = delete_group_jackknife(
        matched_ratings=matched_ratings,
        matched_scenarios=matched_scenarios,
        blocks=blocks,
        model_names=model_names,
        numerical_point=numerical_point,
        pairwise_point=pairwise_point,
        groups=jackknife_groups,
        seed=seed,
    )
    numerical_jackknife = jackknife.numerical_interval
    pairwise_jackknife = jackknife.pairwise_interval
    difference_jackknife = jackknife.difference_interval
    difference_point = numerical_point - pairwise_point.trust
    numerical_point_ranks = (-numerical_point).argsort().argsort() + 1
    pairwise_point_ranks = (-pairwise_point.trust).argsort().argsort() + 1
    summary_rows = [
        {
            "model_id": model_id,
            "model_name": model_name,
            "numerical_trust": numerical_point[model_id],
            "numerical_bootstrap_lower": numerical_lower[model_id],
            "numerical_bootstrap_upper": numerical_upper[model_id],
            "numerical_rank": int(numerical_point_ranks[model_id]),
            "numerical_rank_lower": numerical_rank_lower[model_id],
            "numerical_rank_upper": numerical_rank_upper[model_id],
            "numerical_jackknife_estimate": numerical_jackknife[0][model_id],
            "numerical_jackknife_se": numerical_jackknife[1][model_id],
            "pairwise_refit_trust": pairwise_point.trust[model_id],
            "pairwise_bootstrap_lower": pairwise_lower[model_id],
            "pairwise_bootstrap_upper": pairwise_upper[model_id],
            "pairwise_rank": int(pairwise_point_ranks[model_id]),
            "pairwise_rank_lower": pairwise_rank_lower[model_id],
            "pairwise_rank_upper": pairwise_rank_upper[model_id],
            "pairwise_jackknife_estimate": pairwise_jackknife[0][model_id],
            "pairwise_jackknife_se": pairwise_jackknife[1][model_id],
            "paired_trust_difference": difference_point[model_id],
            "paired_difference_bootstrap_lower": paired_difference_lower[model_id],
            "paired_difference_bootstrap_upper": paired_difference_upper[model_id],
            "paired_difference_jackknife_estimate": difference_jackknife[0][model_id],
            "paired_difference_jackknife_se": difference_jackknife[1][model_id],
            "legacy_pairwise_trust": legacy_pairwise_point.trust[model_id],
            "published_pairwise_trust": published_pairwise[model_id],
        }
        for model_id, model_name in enumerate(model_names)
    ]

    metric_names = ("l1", "spearman", "kendall")
    metric_intervals = {
        name: np.quantile(
            [row[name] for row in bootstrap_metrics],
            [0.025, 0.5, 0.975],
        ).tolist()
        for name in metric_names
    }
    result = {
        "method": {
            "resampling_unit": "scenario",
            "bootstrap_samples": bootstrap_samples,
            "jackknife_groups": jackknife_groups,
            "seed": seed,
            "pairwise_model": "rank-2 Davidson BTD",
            "pairwise_optimizer": (
                "deterministic multi-start L-BFGS-B with analytic gradient"
            ),
            "pairwise_cleaning": PAIRWISE_CLEANING,
            "interval": "95% percentile bootstrap",
            "partition_interval": "unequal delete-group jackknife normal interval",
            "jackknife_group_size_note": (
                "Delete-m_j pseudovalues account for one group of 88 scenarios "
                "and nine groups of 87"
            ),
            "uncertainty_scope": (
                "scenario selection with stored target and judge responses held fixed"
            ),
        },
        "inputs": {
            "ratings": provenance_path(ratings_path),
            "ratings_sha256": input_hashes["ratings"],
            "evaluations": provenance_path(evaluations_path),
            "evaluations_sha256": input_hashes["evaluations"],
            "meta": provenance_path(meta_path),
            "meta_sha256": input_hashes["meta"],
        },
        "support": {
            "matched_scenarios": len(matched_scenarios),
            "pairwise_rows": len(full_pairwise_rows),
            "legacy_pairwise_rows": len(legacy_pairwise_rows),
            "ratings_shape": list(matched_ratings.shape),
            "scenario_ids": matched_scenarios.tolist(),
            "pairwise_cleaning": pairwise_data.diagnostics,
            "legacy_cleaning": legacy_data.diagnostics,
        },
        "point": {
            "numerical_trust": numerical_point.tolist(),
            "numerical_elo": eigentrust_elo(numerical_point).tolist(),
            "pairwise_refit_trust": pairwise_point.trust.tolist(),
            "pairwise_refit_elo": eigentrust_elo(pairwise_point.trust).tolist(),
            "published_pairwise_trust": published_pairwise.tolist(),
            "pairwise_refit": {
                **fit_diagnostics(pairwise_point),
                "agreement_with_published": ranking_metrics(
                    pairwise_point.trust,
                    published_pairwise,
                ),
            },
            "legacy_cleaning_sensitivity": {
                "trust": legacy_pairwise_point.trust.tolist(),
                "elo": eigentrust_elo(legacy_pairwise_point.trust).tolist(),
                "fit": fit_diagnostics(legacy_pairwise_point),
                "agreement_with_primary_cleaning": ranking_metrics(
                    legacy_pairwise_point.trust,
                    pairwise_point.trust,
                ),
            },
            "numerical_vs_pairwise_refit": ranking_metrics(
                numerical_point,
                pairwise_point.trust,
            ),
        },
        "bootstrap": {
            "draw_shape": [bootstrap_samples, len(matched_scenarios)],
            "optimizer_unstable_draw_count": len(bootstrap.unstable_draws),
            "optimizer_unstable_draws": bootstrap.unstable_draws,
            "scenario_draws_reproducible_from": {
                "generator": "numpy.default_rng",
                "seed": seed,
                "low": 0,
                "high": len(matched_scenarios),
            },
            "paired_metric_95_interval": metric_intervals,
            "pairwise_fit_diagnostics": bootstrap.diagnostics,
        },
        "jackknife": {
            "groups": [
                matched_scenarios[positions].tolist()
                for positions in jackknife.group_positions
            ],
            "pairwise_fit_diagnostics": jackknife.diagnostics,
            "numerical_bias_corrected_estimate": numerical_jackknife[0].tolist(),
            "numerical_standard_error": numerical_jackknife[1].tolist(),
            "numerical_lower": numerical_jackknife[2].tolist(),
            "numerical_upper": numerical_jackknife[3].tolist(),
            "pairwise_bias_corrected_estimate": pairwise_jackknife[0].tolist(),
            "pairwise_standard_error": pairwise_jackknife[1].tolist(),
            "pairwise_lower": pairwise_jackknife[2].tolist(),
            "pairwise_upper": pairwise_jackknife[3].tolist(),
            "paired_difference_bias_corrected_estimate": (
                difference_jackknife[0].tolist()
            ),
            "paired_difference_standard_error": difference_jackknife[1].tolist(),
            "paired_difference_lower": difference_jackknife[2].tolist(),
            "paired_difference_upper": difference_jackknife[3].tolist(),
        },
    }

    atomic_json(output_dir / "results.json", result)
    atomic_csv(
        output_dir / "bootstrap_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    atomic_csv(
        output_dir / "jackknife_leave_group_out.csv",
        list(jackknife.leave_out_rows[0]),
        jackknife.leave_out_rows,
    )
    atomic_csv(
        output_dir / "scenario_groups.csv",
        ["scenario_index", "group"],
        jackknife.group_rows,
    )
    atomic_csv(
        output_dir / "paired_bootstrap_metrics.csv",
        list(bootstrap_metrics[0]),
        bootstrap_metrics,
    )
    bootstrap_trust_rows = []
    for draw_index in range(bootstrap_samples):
        row: dict[str, int | float] = {"draw": draw_index}
        for model_id in range(len(model_names)):
            row[f"numerical_{model_id}"] = numerical_draws[draw_index, model_id]
            row[f"pairwise_{model_id}"] = pairwise_draws[draw_index, model_id]
        bootstrap_trust_rows.append(row)
    atomic_csv(
        output_dir / "bootstrap_trust_draws.csv",
        list(bootstrap_trust_rows[0]),
        bootstrap_trust_rows,
    )
    stale_partition = output_dir / "partition_results.csv"
    if stale_partition.exists():
        stale_partition.unlink()
    for checkpoint_path in (
        output_dir / "bootstrap_checkpoint.npz",
        output_dir / "bootstrap_checkpoint_diagnostics.json",
    ):
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    return result
