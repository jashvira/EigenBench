"""Compute one EigenTrust result per criterion from Inspect logs."""

from __future__ import annotations

import argparse
import json

import numpy as np

from numerical_rating.analysis.aggregation.eigentrust import (
    compute_trust,
    eigentrust_elo,
)
from numerical_rating.analysis.reports.round_robin_report import (
    Rating,
    completed_logs,
    eval_log_path,
    load_ratings,
    rating_tensor,
    write_output_generation,
)
from numerical_rating.collection.data import (
    RatingConfig,
    load_config,
    load_constitution_criteria,
    provenance_path,
    repo_path,
    sha256_file,
)
from numerical_rating.collection.run_criteria import (
    CRITERION_PROMPT,
    DEFAULT_CONFIG,
    criterion_provenance,
)

DEFAULT_LOG_DIR = repo_path("runs/numerical_rating/kindness_1000_criterion_round_robin")
DEFAULT_OUTPUT_DIR = repo_path(
    "data/output/numerical_rating/kindness_1000_criterion_round_robin"
)


def analyze(
    ratings: np.ndarray,
    *,
    config: RatingConfig,
    scenario_ids: list[int],
    source_logs: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Compute independent trust vectors for each constitution criterion."""
    criteria = load_constitution_criteria(config.constitution)
    model_names = [
        config.response_models[model_id].name
        for model_id in sorted(config.response_models)
    ]
    results = []
    for criterion_index, criterion in enumerate(criteria):
        trust = compute_trust(ratings[criterion_index])
        results.append(
            {
                "criterion_id": criterion.criterion_id,
                "criterion_text": criterion.text,
                "criterion_text_hash": criterion.text_hash,
                "trust": {
                    model_name: float(value)
                    for model_name, value in zip(model_names, trust.trust)
                },
                "elo": {
                    model_name: float(value)
                    for model_name, value in zip(
                        model_names, eigentrust_elo(trust.trust)
                    )
                },
            }
        )
    return {
        "scenario_count": len(scenario_ids),
        "scenario_indices": scenario_ids,
        "judge_count": ratings.shape[1],
        "target_count": ratings.shape[3],
        "criterion_count": ratings.shape[0],
        "aggregation": "independent EigenTrust fit per criterion",
        "source_logs": source_logs or [],
        "criteria": results,
    }


def criterion_tensors(
    ratings: list[Rating],
    *,
    config: RatingConfig,
) -> tuple[np.ndarray, list[int]]:
    """Build one complete judge-scenario-target tensor per criterion."""
    tensors = []
    scenario_ids: list[int] | None = None
    for criterion in load_constitution_criteria(config.constitution):
        tensor, current_scenarios = rating_tensor(
            ratings,
            config,
            dimension_id=criterion.criterion_id,
        )
        if scenario_ids is None:
            scenario_ids = current_scenarios
        elif current_scenarios != scenario_ids:
            raise ValueError("Criterion ratings use different scenario sets")
        tensors.append(tensor)
    return np.stack(tensors), scenario_ids or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=repo_path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generation-max-tokens", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    criteria = load_constitution_criteria(config.constitution)
    generation_max_tokens = (
        args.generation_max_tokens
        if args.generation_max_tokens is not None
        else config.criterion_generation_max_tokens
    )
    logs = completed_logs(
        args.log_dir,
        config,
        config_name=args.config,
        expected_samples=config.expected_response_cells,
        task_name="pointwise_criterion_rating",
        run_id=f"{config.run_id}_criteria",
        prompt_path=CRITERION_PROMPT,
        generation_limits={
            judge.model: generation_max_tokens for judge in config.judges
        },
        required_metadata={
            "rating_mode": "criterion_wise",
            "criteria": criterion_provenance(criteria),
            "criterion_count": len(criteria),
        },
    )
    ratings = load_ratings(
        logs,
        config,
        scorer_name="criterion_scores",
        dimensions=criteria,
    )
    tensors, scenario_ids = criterion_tensors(ratings, config=config)
    source_logs = [
        {
            "judge_model": judge.model,
            "path": provenance_path(eval_log_path(log)),
            "sha256": sha256_file(eval_log_path(log)),
        }
        for judge, log in zip(config.judges, logs)
    ]
    result = analyze(
        tensors,
        config=config,
        scenario_ids=scenario_ids,
        source_logs=source_logs,
    )
    write_output_generation(
        args.output_dir,
        ratings=ratings,
        result=result,
    )
    print(
        json.dumps(
            {
                "rating_cells": len(ratings),
                "criteria": len(criteria),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
