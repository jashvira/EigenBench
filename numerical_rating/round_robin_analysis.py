"""Aggregate numerical round-robin ratings into EigenTrust scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
from inspect_ai.log import (
    EvalLogInfo,
    list_eval_logs,
    read_eval_log,
    read_eval_log_samples,
)
from scipy.stats import kendalltau, spearmanr

from pipeline.utils.comparisons import (
    extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria,
)
from numerical_rating.data import (
    RatingConfig,
    load_config,
    load_constitution_text,
    load_repaired_cell_manifest,
    load_response_cells,
    provenance_path,
    repo_path,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_LOG_DIRS = (
    ROOT / "runs" / "numerical_rating" / "kindness_1000_round_robin",
    ROOT / "runs" / "numerical_rating" / "kindness_1000_round_robin_gemini4096",
)
DEFAULT_REPAIR_LOG_DIRS = (
    ROOT / "runs" / "numerical_rating" / "kindness_1000_round_robin_repairs",
)
DEFAULT_REPAIR_CELLS = (
    ROOT
    / "data/output/valuearena/processed/full8_kindness/"
    "askreddit_1000_repaired_cells.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "output" / "numerical_rating" / "kindness_1000_round_robin"
)
PUBLISHED_EVALUATIONS = (
    ROOT
    / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
PUBLISHED_META = ROOT / "data/output/valuearena/raw/runs/8_models/kindness/meta.json"
SCORER_NAME = "whole_constitution_score"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TrustResult:
    scenario_count: int
    sigma: np.ndarray
    sigma_used: np.ndarray
    affinity: np.ndarray
    trust_matrix: np.ndarray
    one_step_trust: np.ndarray
    trust: np.ndarray
    iterations: int | None
    stationary_solver: str


def parsed_rationale(parsed: object, explanation: str | None) -> str:
    """Read rationale text from current and earlier scorer metadata."""
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return parsed or explanation or ""
    if isinstance(parsed, dict):
        rationale = parsed.get("rationale")
        if isinstance(rationale, str):
            return rationale
    return explanation or ""


def completed_logs(
    log_dirs: list[Path],
    config: RatingConfig,
    *,
    config_name: str,
    expected_samples: int,
    cell_manifest: Path | None = None,
) -> list[EvalLogInfo]:
    """Select the newest complete log for each configured judge."""
    constitution_hash = sha256_text(load_constitution_text(config.constitution))
    prompt_hash = sha256_text(config.prompt.read_text(encoding="utf-8"))
    expected_manifest = (
        str(cell_manifest.relative_to(ROOT)) if cell_manifest is not None else None
    )
    expected_manifest_hash = (
        sha256_text(cell_manifest.read_text(encoding="utf-8"))
        if cell_manifest is not None
        else None
    )
    expected_config = provenance_path(config_name)
    generation_limits = {
        judge.model: (
            judge.max_tokens
            if judge.max_tokens is not None
            else config.generation_max_tokens
        )
        for judge in config.judges
    }
    selected: dict[str, EvalLogInfo] = {}
    for log_dir in log_dirs:
        for info in list_eval_logs(str(log_dir), formats=["eval", "json"]):
            log = read_eval_log(info, header_only=True)
            metadata = log.eval.metadata or {}
            judge_model = metadata.get("judge_model")
            results = log.results
            if (
                judge_model
                and log.status == "success"
                and not log.invalidated
                and log.eval.task == "pointwise_constitution_rating"
                and metadata.get("run_id") == config.run_id
                and metadata.get("config") == expected_config
                and metadata.get("constitution_hash") == constitution_hash
                and metadata.get("prompt_hash") == prompt_hash
                and metadata.get("generation_max_tokens")
                == generation_limits.get(str(judge_model))
                and metadata.get("generation_temperature")
                == config.generation_temperature
                and metadata.get("cell_manifest") == expected_manifest
                and metadata.get("cell_manifest_hash") == expected_manifest_hash
                and results
                and results.total_samples == expected_samples
                and results.completed_samples == expected_samples
            ):
                judge_model = str(judge_model)
                current = selected.get(judge_model)
                if current is None or float(info.mtime or 0) > float(
                    current.mtime or 0
                ):
                    selected[judge_model] = info

    expected = {judge.model for judge in config.judges}
    missing = sorted(expected - selected.keys())
    if missing:
        raise ValueError("Missing complete judge logs: " + ", ".join(missing))
    return [selected[judge.model] for judge in config.judges]


def load_ratings(logs: list[EvalLogInfo], config: RatingConfig) -> list[Rating]:
    """Extract one parsed numerical score from every Inspect sample."""
    ratings: list[Rating] = []
    for info in logs:
        for sample in read_eval_log_samples(
            info,
            all_samples_required=True,
            exclude_fields={"events", "events_data", "store", "attachments"},
        ):
            score = (sample.scores or {}).get(SCORER_NAME)
            if score is None or not isinstance(score.value, (int, float)):
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
            parsed = metadata.get("parsed")
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
            ratings.append(
                Rating(
                    judge_id=judge_id,
                    judge_name=str(metadata["judge_name"]),
                    judge_model=str(metadata["judge_model"]),
                    scenario_index=int(metadata["scenario_index"]),
                    model_id=model_id,
                    model_name=str(metadata["model_name"]),
                    score=float(score.value),
                    rationale=parsed_rationale(parsed, score.explanation),
                    response_hash=str(metadata["response_hash"]),
                )
            )
    return ratings


def repaired_cell_count(path: Path, config: RatingConfig) -> int:
    """Return the number of cells declared by a repair manifest."""
    manifest = load_repaired_cell_manifest(
        path,
        config=config,
        response_cells=load_response_cells(config),
    )
    if not manifest.cells:
        raise ValueError(f"Repair manifest has no cells: {path}")
    return len(manifest.cells)


def reconcile_ratings(
    base_ratings: list[Rating],
    repair_ratings: list[Rating],
    config: RatingConfig,
) -> tuple[list[Rating], int]:
    """Replace ratings whose response hashes no longer match the cache."""
    expected_hashes = {
        (cell.scenario_index, cell.model_id): cell.response_hash
        for cell in load_response_cells(config)
    }
    expected_base_count = len(config.judges) * config.expected_response_cells
    if len(base_ratings) != expected_base_count:
        raise ValueError(
            f"Expected {expected_base_count} base ratings, found {len(base_ratings)}"
        )

    merged: dict[tuple[int, int, int], Rating] = {}
    stale: set[tuple[int, int, int]] = set()
    for rating in base_ratings:
        key = (rating.judge_id, rating.scenario_index, rating.model_id)
        if key in merged:
            raise ValueError(f"Duplicate base rating cell: {key}")
        expected_hash = expected_hashes.get((rating.scenario_index, rating.model_id))
        if expected_hash is None:
            raise ValueError(f"Base rating references an unknown response cell: {key}")
        merged[key] = rating
        if rating.response_hash != expected_hash:
            stale.add(key)

    replacements: dict[tuple[int, int, int], Rating] = {}
    for rating in repair_ratings:
        key = (rating.judge_id, rating.scenario_index, rating.model_id)
        if key in replacements:
            raise ValueError(f"Duplicate repair rating cell: {key}")
        expected_hash = expected_hashes.get((rating.scenario_index, rating.model_id))
        if expected_hash is None or rating.response_hash != expected_hash:
            raise ValueError(f"Repair rating does not match the response cache: {key}")
        replacements[key] = rating

    repair_keys = set(replacements)
    if repair_keys != stale:
        missing = sorted(stale - repair_keys)
        extra = sorted(repair_keys - stale)
        raise ValueError(
            f"Repair ratings do not match stale cells; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    merged.update(replacements)
    return [merged[key] for key in sorted(merged)], len(replacements)


def rating_tensor(
    ratings: list[Rating], config: RatingConfig
) -> tuple[np.ndarray, list[int]]:
    """Build and validate the judge-by-scenario-by-target rating tensor."""
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


def softmax_rows(
    values: np.ndarray, temperature: float, zero_diagonal: bool
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = values / temperature
    if zero_diagonal:
        logits = logits.copy()
        np.fill_diagonal(logits, -np.inf)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def stationary_trust(
    matrix: np.ndarray, tolerance: float = 1e-12
) -> tuple[np.ndarray, int | None, str]:
    trust = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for iteration in range(1, 10_001):
        next_trust = trust @ matrix
        if np.linalg.norm(next_trust - trust, ord=1) < tolerance:
            return next_trust, iteration, "power_iteration"
        trust = next_trust

    system = matrix.T - np.eye(matrix.shape[0])
    system[-1] = 1.0
    target = np.zeros(matrix.shape[0])
    target[-1] = 1.0
    solved = np.linalg.solve(system, target)
    if np.any(solved < -1e-10):
        raise RuntimeError("Stationary linear solve produced negative trust")
    solved = np.clip(solved, 0.0, None)
    solved /= solved.sum()
    return solved, None, "linear_solve"


def compute_trust(
    ratings: np.ndarray,
    *,
    temperature: float = 1.0,
    zero_diagonal: bool = False,
    standardize: bool = True,
    sigma_floor_ratio: float | None = None,
) -> TrustResult:
    """Compute direct judge-target affinities and their EigenTrust vector."""
    judges, scenarios, models = ratings.shape
    if judges != models:
        raise ValueError(
            "EigenTrust requires the same judge and target population size"
        )

    if zero_diagonal:
        if models <= 2:
            raise ValueError("Self-score exclusion requires at least three models")
        off_diagonal = ~np.eye(models, dtype=bool)
        mask = off_diagonal[:, None, :]
        off_diagonal_mean = np.where(mask, ratings, 0.0).sum(
            axis=2, keepdims=True
        ) / (models - 1)
        centered = np.where(mask, ratings - off_diagonal_mean, 0.0)
        variance_denominator = scenarios * (models - 2)
    else:
        centered = ratings - ratings.mean(axis=2, keepdims=True)
        variance_denominator = scenarios * (models - 1)
    if not np.allclose(centered.sum(axis=2), 0.0, atol=1e-10):
        raise AssertionError("Scenario-centred ratings do not sum to zero")
    sigma = np.sqrt(np.square(centered).sum(axis=(1, 2)) / variance_denominator)
    if np.any(sigma == 0):
        raise ValueError("At least one judge has zero residual score variance")

    sigma_used = sigma.copy() if standardize else np.ones_like(sigma)
    if standardize and sigma_floor_ratio is not None:
        floor = sigma_floor_ratio * float(np.median(sigma))
        sigma_used = np.maximum(sigma_used, floor)

    standardized = centered / sigma_used[:, None, None]
    affinity = standardized.mean(axis=1)
    if zero_diagonal:
        np.fill_diagonal(affinity, 0.0)
    if not np.allclose(affinity.sum(axis=1), 0.0, atol=1e-10):
        raise AssertionError("Judge affinity rows do not sum to zero")
    trust_matrix = softmax_rows(affinity, temperature, zero_diagonal)
    if not np.allclose(trust_matrix.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Trust matrix rows do not sum to one")
    one_step = np.full(judges, 1.0 / judges) @ trust_matrix
    trust, iterations, stationary_solver = stationary_trust(trust_matrix)
    if np.linalg.norm(trust @ trust_matrix - trust, ord=1) > 1e-10:
        raise AssertionError("Stationary trust vector failed its residual check")
    return TrustResult(
        scenario_count=scenarios,
        sigma=sigma,
        sigma_used=sigma_used,
        affinity=affinity,
        trust_matrix=trust_matrix,
        one_step_trust=one_step,
        trust=trust,
        iterations=iterations,
        stationary_solver=stationary_solver,
    )


def published_scenarios(config: RatingConfig) -> list[int]:
    """Return scenario IDs that contributed to the published pairwise fit."""
    evaluations = [
        json.loads(line)
        for line in PUBLISHED_EVALUATIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    criterion_count = len(
        json.loads(config.constitution.read_text(encoding="utf-8"))
    )
    comparisons, _ = extract_comparisons_with_ties_criteria(
        evaluations,
        num_criteria=criterion_count,
    )
    comparisons = handle_inconsistencies_with_ties_criteria(comparisons)
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


def eigentrust_elo(trust: np.ndarray) -> np.ndarray:
    return 1500.0 + 400.0 * np.log10(len(trust) * np.clip(trust, 1e-12, None))


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


@contextmanager
def atomic_text_output(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Replace a text artifact only after its complete contents reach disk."""
    output = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline=newline,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(output.name)
    try:
        yield output
        output.flush()
        os.fsync(output.fileno())
        output.close()
        os.replace(temporary, path)
    except BaseException:
        output.close()
        temporary.unlink(missing_ok=True)
        raise


def write_ratings(path: Path, ratings: list[Rating]) -> None:
    fields = list(Rating.__dataclass_fields__)
    with atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(vars(rating) for rating in ratings)


def write_json(path: Path, value: object) -> None:
    """Write strict JSON without exposing partial output after interruption."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with atomic_text_output(path) as output:
        output.write(payload)


def write_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically write the final model ranking table."""
    with atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_output_generation(
    output_dir: Path,
    *,
    ratings: list[Rating],
    result: dict[str, object],
    rankings: list[dict[str, object]],
) -> None:
    """Commit the three analysis artifacts as one recoverable directory swap."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
    ) as transaction:
        transaction_dir = Path(transaction)
        staged = transaction_dir / "staged"
        staged.mkdir()
        write_ratings(staged / "ratings.csv", ratings)
        write_json(staged / "results.json", result)
        write_rankings(staged / "rankings.csv", rankings)

        previous = transaction_dir / "previous"
        if output_dir.exists():
            os.replace(output_dir, previous)
            try:
                os.replace(staged, output_dir)
            except BaseException:
                os.replace(previous, output_dir)
                raise
        else:
            os.replace(staged, output_dir)


def ranking_rows(
    config: RatingConfig,
    full: TrustResult,
    matched: TrustResult,
    raw_full: TrustResult,
    raw_matched: TrustResult,
    original: np.ndarray,
) -> list[dict[str, object]]:
    model_names = [
        config.response_models[index].name
        for index in range(config.expected_models)
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
                "raw_numerical_1000_trust": float(raw_full.trust[index]),
                "raw_numerical_871_trust": float(raw_matched.trust[index]),
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
    parser.add_argument("--log-dir", action="append", type=repo_path, default=[])
    parser.add_argument(
        "--repair-log-dir", action="append", type=repo_path, default=[]
    )
    parser.add_argument(
        "--repair-cells", type=repo_path, default=DEFAULT_REPAIR_CELLS
    )
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstraps < 2:
        parser.error("--bootstraps must be at least 2")

    config = load_config(args.config)
    logs = completed_logs(
        args.log_dir or list(DEFAULT_LOG_DIRS),
        config,
        config_name=args.config,
        expected_samples=config.expected_response_cells,
    )
    base_ratings = load_ratings(logs, config)
    repair_ratings: list[Rating] = []
    repair_count = 0
    if not args.repair_cells.exists():
        raise FileNotFoundError(f"Repair manifest is missing: {args.repair_cells}")
    repair_cell_total = repaired_cell_count(args.repair_cells, config)
    repair_logs = completed_logs(
        args.repair_log_dir or list(DEFAULT_REPAIR_LOG_DIRS),
        config,
        config_name=args.config,
        expected_samples=repair_cell_total,
        cell_manifest=args.repair_cells,
    )
    repair_ratings = load_ratings(repair_logs, config)
    ratings, repair_count = reconcile_ratings(base_ratings, repair_ratings, config)
    tensor, scenario_indices = rating_tensor(ratings, config)
    raw_tensor, raw_scenario_indices = rating_tensor(base_ratings, config)
    if raw_scenario_indices != scenario_indices:
        raise ValueError("Raw and corrected ratings use different scenario sets")
    full = compute_trust(tensor)
    raw_full = compute_trust(raw_tensor)

    matched_indices = published_scenarios(config)
    positions = {scenario: index for index, scenario in enumerate(scenario_indices)}
    matched_tensor = tensor[:, [positions[index] for index in matched_indices], :]
    raw_matched_tensor = raw_tensor[
        :, [positions[index] for index in matched_indices], :
    ]
    matched = compute_trust(matched_tensor)
    raw_matched = compute_trust(raw_matched_tensor)
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
    raw_comparison = {
        "l1": float(np.abs(raw_matched.trust - original).sum()),
        "spearman": float(spearmanr(raw_matched.trust, original).statistic),
        "kendall": float(kendalltau(raw_matched.trust, original).statistic),
    }
    comparison_ci = comparison_intervals(matched_draws, original)
    result = {
        "run_id": config.run_id,
        "input": {
            "response_cache": str(config.source.relative_to(ROOT)),
            "response_cache_sha256": sha256_file(config.source),
            "repair_manifest": (
                str(args.repair_cells.relative_to(ROOT))
                if args.repair_cells.exists()
                else None
            ),
            "repair_manifest_sha256": (
                sha256_file(args.repair_cells) if args.repair_cells.exists() else None
            ),
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
        "replacement_rating_cells": repair_count,
        "primary_1000": serialise_result(full),
        "matched_871": serialise_result(matched),
        "pre_repair_response_cache": {
            "primary_1000": serialise_result(raw_full),
            "matched_871": serialise_result(raw_matched),
            "comparison_to_pairwise": raw_comparison,
        },
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
            name: serialise_result(value)
            for name, value in sensitivities.items()
        },
    }

    rows = ranking_rows(config, full, matched, raw_full, raw_matched, original)
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
                "replacement_rating_cells": repair_count,
                "comparison_to_pairwise": comparison,
                "raw_comparison_to_pairwise": raw_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
