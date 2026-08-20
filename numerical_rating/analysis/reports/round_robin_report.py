"""Aggregate numerical round-robin ratings into EigenTrust scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import numpy as np
from inspect_ai.log import (
    EvalLogInfo,
    list_eval_logs,
    read_eval_log,
    read_eval_log_samples,
)
from scipy.stats import kendalltau, spearmanr

from numerical_rating.analysis.aggregation.eigentrust import (
    TrustResult,
    compute_trust,
    eigentrust_elo,
)
from numerical_rating.collection.data import (
    ConstitutionCriterion,
    RatingConfig,
    load_config,
    load_constitution_text,
    load_response_cells,
    provenance_path,
    repo_path,
    sha256_text,
)
from pipeline.utils.comparisons import (
    extract_comparisons_with_ties_criteria,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_LOG_DIR = ROOT / "runs" / "numerical_rating" / "kindness_1000_round_robin"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "output" / "numerical_rating" / "kindness_1000_round_robin"
)
PUBLISHED_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
PUBLISHED_META = ROOT / "data/output/valuearena/raw/runs/8_models/kindness/meta.json"
SCORER_NAME = "whole_constitution_score"


@dataclass(frozen=True, slots=True)
class Rating:
    judge_id: int
    judge_name: str
    judge_model: str
    scenario_index: int
    model_id: int
    model_name: str
    score: float
    rationale: str
    response_hash: str
    dimension_id: str = "whole_constitution"
    dimension_hash: str = ""


def eval_log_path(info: EvalLogInfo) -> Path:
    """Return the local path represented by an Inspect log reference."""
    parsed = urlparse(info.name)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"Inspect log is not local: {info.name}")
    return Path(info.name)


def completed_logs(
    log_dir: Path,
    config: RatingConfig,
    *,
    config_name: str,
    expected_samples: int,
    task_name: str = "pointwise_constitution_rating",
    run_id: str | None = None,
    prompt_path: Path | None = None,
    generation_limits: Mapping[str, int] | None = None,
    required_metadata: Mapping[str, object] | None = None,
) -> list[EvalLogInfo]:
    """Select the newest complete log for each configured judge."""
    constitution_hash = sha256_text(load_constitution_text(config.constitution))
    prompt = prompt_path or config.prompt
    prompt_hash = sha256_text(prompt.read_text(encoding="utf-8"))
    expected_config = provenance_path(config_name)
    expected_generation_limits = generation_limits or {
        judge.model: (
            judge.max_tokens
            if judge.max_tokens is not None
            else config.generation_max_tokens
        )
        for judge in config.judges
    }
    metadata_requirements = dict(required_metadata or {})
    selected: dict[str, EvalLogInfo] = {}
    for info in list_eval_logs(str(log_dir), formats=["eval"]):
        log = read_eval_log(info, header_only=True)
        metadata = log.eval.metadata or {}
        judge_model = metadata.get("judge_model")
        results = log.results
        if (
            judge_model
            and log.status == "success"
            and not log.invalidated
            and log.eval.task == task_name
            and metadata.get("run_id") == (run_id or config.run_id)
            and metadata.get("config") == expected_config
            and metadata.get("constitution_hash") == constitution_hash
            and metadata.get("prompt_hash") == prompt_hash
            and metadata.get("generation_max_tokens")
            == expected_generation_limits.get(str(judge_model))
            and metadata.get("generation_temperature") == config.generation_temperature
            and results
            and results.total_samples == expected_samples
            and results.completed_samples == expected_samples
            and all(
                metadata.get(key) == value
                for key, value in metadata_requirements.items()
            )
        ):
            judge_model = str(judge_model)
            current = selected.get(judge_model)
            if current is None or float(info.mtime or 0) > float(current.mtime or 0):
                selected[judge_model] = info

    expected = {judge.model for judge in config.judges}
    missing = sorted(expected - selected.keys())
    if missing:
        raise ValueError("Missing complete judge logs: " + ", ".join(missing))
    return [selected[judge.model] for judge in config.judges]


def load_ratings(
    logs: list[EvalLogInfo],
    config: RatingConfig,
    *,
    scorer_name: str = SCORER_NAME,
    dimensions: tuple[ConstitutionCriterion, ...] = (),
) -> list[Rating]:
    """Extract dimensioned numerical ratings from Inspect samples."""
    expected_dimensions = [dimension.criterion_id for dimension in dimensions]
    dimension_by_id = {dimension.criterion_id: dimension for dimension in dimensions}
    response_cells = {
        (cell.scenario_index, cell.model_id): cell
        for cell in load_response_cells(config)
    }
    ratings: list[Rating] = []
    for info in logs:
        for sample in read_eval_log_samples(
            info,
            all_samples_required=True,
            exclude_fields={"events", "events_data", "store", "attachments"},
        ):
            score = (sample.scores or {}).get(scorer_name)
            if score is None:
                raise ValueError(f"Missing numerical score in {info.name}: {sample.id}")
            sample_metadata = sample.metadata or {}
            score_metadata = score.metadata or {}
            for key in (
                "judge_id",
                "judge_name",
                "judge_model",
                "model_id",
                "model_name",
            ):
                if (
                    key in score_metadata
                    and key in sample_metadata
                    and score_metadata[key] != sample_metadata[key]
                ):
                    raise ValueError(
                        f"Conflicting {key} metadata in {info.name}: {sample.id}"
                    )
            metadata = {**sample_metadata, **score_metadata}
            judge_id = int(metadata["judge_id"])
            model_id = int(metadata["model_id"])
            if not 0 <= judge_id < len(config.judges):
                raise ValueError(f"Invalid judge identity in {info.name}: {sample.id}")
            if not 0 <= model_id < config.expected_models:
                raise ValueError(f"Invalid target identity in {info.name}: {sample.id}")
            expected_judge = config.judges[judge_id]
            expected_model = config.response_models[model_id]
            if (
                metadata.get("judge_name") != expected_judge.name
                or metadata.get("judge_model") != expected_judge.model
            ):
                raise ValueError(f"Judge metadata mismatch in {info.name}: {sample.id}")
            if metadata.get("model_name") != expected_model.name:
                raise ValueError(
                    f"Target metadata mismatch in {info.name}: {sample.id}"
                )
            scenario_index = int(metadata["scenario_index"])
            response_cell = response_cells.get((scenario_index, model_id))
            if response_cell is None:
                raise ValueError(f"Unknown response cell in {info.name}: {sample.id}")
            response_hash = str(metadata["response_hash"])
            if response_hash != response_cell.response_hash:
                raise ValueError(f"Stale response in {info.name}: {sample.id}")

            if dimensions:
                if metadata.get("criterion_ids") != expected_dimensions:
                    raise ValueError(
                        f"Criterion IDs do not match in {info.name}: {sample.id}"
                    )
                if not isinstance(score.value, dict):
                    raise ValueError(
                        f"Criterion score is not a mapping in {info.name}: {sample.id}"
                    )
                try:
                    rationales = json.loads(score.explanation or "")
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid criterion rationales in {info.name}: {sample.id}"
                    ) from exc
                if (
                    list(score.value) != expected_dimensions
                    or not isinstance(rationales, dict)
                    or list(rationales) != expected_dimensions
                ):
                    raise ValueError(
                        f"Criterion score shape does not match in {info.name}: "
                        f"{sample.id}"
                    )
                score_items = [
                    (
                        dimension_id,
                        score.value[dimension_id],
                        rationales[dimension_id],
                    )
                    for dimension_id in expected_dimensions
                ]
            else:
                if not isinstance(score.value, (int, float)):
                    raise ValueError(
                        f"Score is not numeric in {info.name}: {sample.id}"
                    )
                score_items = [
                    (
                        "whole_constitution",
                        score.value,
                        score.explanation or "",
                    )
                ]

            for dimension_id, value, rationale in score_items:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(float(value))
                    or not config.score_min <= float(value) <= config.score_max
                ):
                    raise ValueError(
                        f"Invalid {dimension_id} score in {info.name}: {sample.id}"
                    )
                if dimensions and (
                    not isinstance(rationale, str) or not rationale.strip()
                ):
                    raise ValueError(
                        f"Invalid {dimension_id} rationale in {info.name}: {sample.id}"
                    )
                ratings.append(
                    Rating(
                        judge_id=judge_id,
                        judge_name=str(metadata["judge_name"]),
                        judge_model=str(metadata["judge_model"]),
                        scenario_index=scenario_index,
                        model_id=model_id,
                        model_name=str(metadata["model_name"]),
                        score=float(value),
                        rationale=rationale,
                        response_hash=response_hash,
                        dimension_id=dimension_id,
                        dimension_hash=(
                            dimension_by_id[dimension_id].text_hash
                            if dimensions
                            else ""
                        ),
                    )
                )
    return ratings


def rating_tensor(
    ratings: list[Rating],
    config: RatingConfig,
    *,
    dimension_id: str = "whole_constitution",
) -> tuple[np.ndarray, list[int]]:
    """Build and validate the judge-by-scenario-by-target rating tensor."""
    ratings = [rating for rating in ratings if rating.dimension_id == dimension_id]
    if not ratings:
        raise ValueError(f"No ratings found for dimension: {dimension_id}")
    scenario_indices = sorted({rating.scenario_index for rating in ratings})
    scenario_position = {
        scenario_index: position
        for position, scenario_index in enumerate(scenario_indices)
    }
    shape = (len(config.judges), len(scenario_indices), config.expected_models)
    tensor = np.full(shape, np.nan, dtype=float)
    for rating in ratings:
        if not 0 <= rating.judge_id < len(config.judges):
            raise ValueError(f"Invalid judge_id: {rating.judge_id}")
        if not 0 <= rating.model_id < config.expected_models:
            raise ValueError(f"Invalid model_id: {rating.model_id}")
        if not config.score_min <= rating.score <= config.score_max:
            raise ValueError(f"Score outside configured scale: {rating.score}")
        position = (
            rating.judge_id,
            scenario_position[rating.scenario_index],
            rating.model_id,
        )
        if not np.isnan(tensor[position]):
            raise ValueError(f"Duplicate rating cell: {position}")
        tensor[position] = rating.score

    expected_shape = (
        len(config.judges),
        config.expected_scenarios,
        config.expected_models,
    )
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Expected rating tensor {expected_shape}, found {tensor.shape}"
        )
    if np.isnan(tensor).any():
        missing = int(np.isnan(tensor).sum())
        raise ValueError(f"Rating tensor has {missing} missing cells")
    if not np.isfinite(tensor).all():
        raise ValueError("Rating tensor contains non-finite scores")
    judge_rows = {
        judge.target_model_id: index for index, judge in enumerate(config.judges)
    }
    tensor = tensor[
        [judge_rows[index] for index in range(config.expected_models)], :, :
    ]
    return tensor, scenario_indices


def published_scenarios(config: RatingConfig) -> list[int]:
    """Return scenario IDs that contributed to the published pairwise fit."""
    evaluations = [
        json.loads(line)
        for line in PUBLISHED_EVALUATIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    criterion_count = len(json.loads(config.constitution.read_text(encoding="utf-8")))
    comparisons, _ = extract_comparisons_with_ties_criteria(
        evaluations,
        num_criteria=criterion_count,
    )
    # Tie reconciliation changes choices, not which scenarios supplied valid
    # judgments, so support is read before repeated passes are flattened.
    return sorted({int(comparison[1]) for comparison in comparisons})


def published_trust(config: RatingConfig) -> np.ndarray:
    """Load published trust values after checking their model ordering."""
    metadata = json.loads(PUBLISHED_META.read_text(encoding="utf-8"))
    published_models = metadata.get("models")
    if not isinstance(published_models, dict):
        raise ValueError("Published metadata has no ordered model mapping")

    expected = [
        (config.response_models[index].name, config.response_models[index].api_id)
        for index in range(config.expected_models)
    ]
    observed = [
        (name, str(spec.get("id")))
        for name, spec in published_models.items()
        if isinstance(spec, dict)
    ]
    if observed != expected:
        raise ValueError("Published model order or IDs do not match the run manifest")

    values = metadata["eigentrust"]
    trust = np.asarray(values, dtype=float)
    if trust.shape != (config.expected_models,) or not np.isfinite(trust).all():
        raise ValueError("Published EigenTrust vector has an invalid shape or value")
    if np.any(trust < 0) or trust.sum() <= 0:
        raise ValueError("Published EigenTrust vector must be nonnegative and nonzero")
    return trust / trust.sum()


def bootstrap_intervals(
    ratings: np.ndarray, *, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, ratings.shape[0]), dtype=float)
    for index in range(samples):
        scenario_sample = rng.integers(0, ratings.shape[1], size=ratings.shape[1])
        draws[index] = compute_trust(ratings[:, scenario_sample, :]).trust
    pairwise = (draws[:, :, None] > draws[:, None, :]).mean(axis=0)
    return (
        draws.std(axis=0, ddof=1),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
        pairwise,
        draws,
    )


def comparison_intervals(
    draws: np.ndarray, reference: np.ndarray
) -> dict[str, list[float]]:
    """Bootstrap comparison intervals conditional on a fixed reference vector."""
    metrics = {
        "l1": np.abs(draws - reference).sum(axis=1),
        "spearman": np.asarray(
            [spearmanr(draw, reference).statistic for draw in draws], dtype=float
        ),
        "kendall": np.asarray(
            [kendalltau(draw, reference).statistic for draw in draws], dtype=float
        ),
    }
    if any(not np.isfinite(values).all() for values in metrics.values()):
        raise ValueError("Bootstrap comparison produced a non-finite statistic")
    return {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in metrics.items()
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_indices(indices: list[int]) -> str:
    """Hash an ordered scenario set for result provenance."""
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def serialise_result(result: TrustResult) -> dict[str, object]:
    return {
        "scenario_count": result.scenario_count,
        "sigma": result.sigma.tolist(),
        "sigma_used": result.sigma_used.tolist(),
        "affinity": result.affinity.tolist(),
        "trust_matrix": result.trust_matrix.tolist(),
        "one_step_trust": result.one_step_trust.tolist(),
        "recursive_effect_l1": float(
            np.abs(result.trust - result.one_step_trust).sum()
        ),
        "trust": result.trust.tolist(),
        "elo": eigentrust_elo(result.trust).tolist(),
        "iterations": result.iterations,
        "stationary_solver": result.stationary_solver,
    }


def write_ratings(path: Path, ratings: list[Rating]) -> None:
    fields = list(Rating.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: getattr(rating, field) for field in fields} for rating in ratings
        )


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_output_generation(
    output_dir: Path,
    *,
    ratings: list[Rating],
    result: dict[str, object],
    rankings: list[dict[str, object]] | None = None,
) -> None:
    """Write the ratings, results, and optional ranking table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ratings(output_dir / "ratings.csv", ratings)
    write_json(output_dir / "results.json", result)
    if rankings is not None:
        write_rankings(output_dir / "rankings.csv", rankings)


def ranking_rows(
    config: RatingConfig,
    full: TrustResult,
    matched: TrustResult,
    original: np.ndarray,
) -> list[dict[str, object]]:
    model_names = [
        config.response_models[index].name for index in range(config.expected_models)
    ]
    return sorted(
        (
            {
                "model_id": index,
                "model_name": model_names[index],
                "numerical_1000_trust": float(full.trust[index]),
                "numerical_1000_elo": float(eigentrust_elo(full.trust)[index]),
                "numerical_871_trust": float(matched.trust[index]),
                "numerical_871_elo": float(eigentrust_elo(matched.trust)[index]),
                "pairwise_871_trust": float(original[index]),
                "pairwise_871_elo": float(eigentrust_elo(original)[index]),
            }
            for index in range(config.expected_models)
        ),
        key=lambda row: float(row["numerical_1000_trust"]),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=repo_path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstraps < 2:
        parser.error("--bootstraps must be at least 2")

    config = load_config(args.config)
    logs = completed_logs(
        args.log_dir,
        config,
        config_name=args.config,
        expected_samples=config.expected_response_cells,
    )
    ratings = load_ratings(logs, config)
    tensor, scenario_indices = rating_tensor(ratings, config)
    full = compute_trust(tensor)

    matched_indices = published_scenarios(config)
    positions = {scenario: index for index, scenario in enumerate(scenario_indices)}
    matched_tensor = tensor[:, [positions[index] for index in matched_indices], :]
    matched = compute_trust(matched_tensor)
    original = published_trust(config)

    full_se, full_lower, full_upper, full_pairwise, _ = bootstrap_intervals(
        tensor, samples=args.bootstraps, seed=args.seed
    )
    (
        matched_se,
        matched_lower,
        matched_upper,
        matched_pairwise,
        matched_draws,
    ) = bootstrap_intervals(matched_tensor, samples=args.bootstraps, seed=args.seed)
    sensitivities = {
        "temperature_0.5": compute_trust(tensor, temperature=0.5),
        "temperature_2": compute_trust(tensor, temperature=2.0),
        "exclude_self_scores": compute_trust(tensor, zero_diagonal=True),
        "no_standardization": compute_trust(tensor, standardize=False),
        "sigma_floor_half_median": compute_trust(tensor, sigma_floor_ratio=0.5),
    }

    comparison = {
        "l1": float(np.abs(matched.trust - original).sum()),
        "spearman": float(spearmanr(matched.trust, original).statistic),
        "kendall": float(kendalltau(matched.trust, original).statistic),
    }
    comparison_ci = comparison_intervals(matched_draws, original)
    result = {
        "run_id": config.run_id,
        "input": {
            "response_cache": str(config.source.relative_to(ROOT)),
            "response_cache_sha256": sha256_file(config.source),
            "scenario_set_sha256": hash_indices(scenario_indices),
            "matched_scenario_set_sha256": hash_indices(matched_indices),
            "constitution_sha256": sha256_file(config.constitution),
            "prompt_sha256": sha256_file(config.prompt),
        },
        "method": {
            "centering": "within judge and scenario across targets",
            "variance_denominator": "scenario_count * (target_count - 1)",
            "temperature": 1.0,
            "zero_diagonal": False,
            "standardization": "per-judge residual standard deviation",
            "stationary_tolerance_l1": 1e-12,
        },
        "judge_models": [vars(judge) for judge in config.judges],
        "target_models": [
            vars(config.response_models[index])
            for index in range(config.expected_models)
        ],
        "rating_cells": len(ratings),
        "primary_1000": serialise_result(full),
        "matched_871": serialise_result(matched),
        "published_pairwise_871": {
            "trust": original.tolist(),
            "elo": eigentrust_elo(original).tolist(),
        },
        "comparison_to_pairwise": comparison,
        "comparison_to_pairwise_bootstrap_95_ci": {
            "conditioning": "published pairwise trust vector held fixed",
            **comparison_ci,
        },
        "bootstrap": {
            "samples": args.bootstraps,
            "seed": args.seed,
            "primary_1000_standard_error": full_se.tolist(),
            "primary_1000_95_ci": [full_lower.tolist(), full_upper.tolist()],
            "primary_1000_pairwise_win_probability": full_pairwise.tolist(),
            "matched_871_standard_error": matched_se.tolist(),
            "matched_871_95_ci": [matched_lower.tolist(), matched_upper.tolist()],
            "matched_871_pairwise_win_probability": matched_pairwise.tolist(),
        },
        "sensitivities_1000": {
            name: serialise_result(value) for name, value in sensitivities.items()
        },
    }

    rows = ranking_rows(config, full, matched, original)
    write_output_generation(
        args.output_dir,
        ratings=ratings,
        result=result,
        rankings=rows,
    )
    print(
        json.dumps(
            {
                "rating_cells": len(ratings),
                "comparison_to_pairwise": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
