"""Scenario-level uncertainty for numerical and published pairwise rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr

from numerical_rating.data import provenance_path
from numerical_rating.trust import compute_trust, eigentrust_elo
from pipeline.utils.comparisons import (
    extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_round_robin/ratings.csv"
)
DEFAULT_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
DEFAULT_META = ROOT / "data/output/valuearena/raw/runs/8_models/kindness/meta.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_round_robin/"
    "scenario_uncertainty"
)
PAIRWISE_CLEANING = "corrected"


@dataclass(frozen=True)
class PairwiseFit:
    """A fitted rank-two Davidson model and its propagated trust."""

    parameters: np.ndarray
    trust: np.ndarray
    loss: float
    iterations: int
    attempted_starts: int = 1
    successful_starts: int = 1
    best_start: int = 0
    near_optimal_starts: int = 1
    max_near_optimal_trust_l1: float = 0.0
    stable_near_optima: bool = True
    gradient_l2: float = float("nan")
    parameter_l2: float = float("nan")
    start_diagnostics: tuple[dict, ...] = ()
    failed_starts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairwiseData:
    """Scenario comparison blocks and cleaning diagnostics."""

    blocks: dict[int, np.ndarray]
    diagnostics: dict[str, int | str]


def load_numerical_tensor(path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load and validate the complete judge-scenario-target rating rectangle."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    judge_ids = sorted({int(row["judge_id"]) for row in rows})
    scenario_ids = sorted({int(row["scenario_index"]) for row in rows})
    model_ids = sorted({int(row["model_id"]) for row in rows})
    if judge_ids != list(range(8)) or model_ids != list(range(8)):
        raise ValueError("Expected judge and target IDs 0..7")
    if len(scenario_ids) != 1_000 or len(rows) != 64_000:
        raise ValueError("Expected a complete 8 x 1,000 x 8 rating rectangle")

    scenario_position = {scenario: index for index, scenario in enumerate(scenario_ids)}
    tensor = np.full((8, len(scenario_ids), 8), np.nan, dtype=float)
    model_names: list[str | None] = [None] * 8
    for row in rows:
        judge = int(row["judge_id"])
        scenario = scenario_position[int(row["scenario_index"])]
        model = int(row["model_id"])
        if np.isfinite(tensor[judge, scenario, model]):
            raise ValueError("Ratings file contains a duplicate cell")
        tensor[judge, scenario, model] = float(row["score"])
        model_names[model] = row["model_name"]
    if not np.isfinite(tensor).all():
        raise ValueError("Ratings file contains a missing or non-finite cell")
    if any(name is None for name in model_names):
        raise ValueError("Ratings file is missing a model name")
    return tensor, scenario_ids, [str(name) for name in model_names]


def _evaluation_key(evaluation: dict) -> str:
    """Serialize an evaluation deterministically for exact deduplication."""
    serialized = json.dumps(
        evaluation,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _content_hash(value: object) -> str:
    """Hash a JSON value used to identify one collection pass."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _collection_pass_key(evaluation: dict) -> tuple:
    """Identify reversed judgments over the same cached responses and reflections."""
    targets = tuple(
        sorted(
            (
                int(evaluation[index_key]),
                _content_hash(evaluation[response_key]),
                _content_hash(evaluation[reflection_key]),
            )
            for index_key, response_key, reflection_key in (
                ("eval1", "eval1 response", "eval1 reflection"),
                ("eval2", "eval2 response", "eval2 reflection"),
            )
        )
    )
    return (
        int(evaluation["scenario_index"]),
        _content_hash(evaluation["scenario"]),
        int(evaluation["judge"]),
        _content_hash(evaluation["constitution"]),
        targets,
    )


def _reconcile_transpose_pair(left: list[int], right: list[int]) -> list[list[int]]:
    """Apply the published tie rule to one forward/reverse judgment pair."""
    left_choice = left[-1]
    right_choice = right[-1]
    if left_choice == 0 or right_choice == 0 or left_choice != right_choice:
        return [left, right]
    return [left[:-1] + [0], right[:-1] + [0]]


def _reconcile_pass_comparisons(comparisons: list[list[int]]) -> list[list[int]]:
    """Reconcile criterion rows from one identified collection pass."""
    grouped: dict[tuple[int, int, int, int, int], list[list[int]]] = defaultdict(list)
    for row in comparisons:
        criterion, scenario, judge, first, second, _ = row
        key = (criterion, scenario, judge, min(first, second), max(first, second))
        grouped[key].append(row)

    cleaned: list[list[int]] = []
    for rows in grouped.values():
        if len(rows) == 1:
            cleaned.extend(rows)
            continue
        if len(rows) != 2 or rows[0][3:5] != rows[1][4:2:-1]:
            raise ValueError("Ambiguous rows within a corrected collection pass")
        cleaned.extend(_reconcile_transpose_pair(rows[0], rows[1]))
    return cleaned


def _extract_corrected_comparisons(
    evaluations: list[dict], *, num_criteria: int
) -> tuple[list[list[int]], dict[str, int]]:
    """Pair reversed records by response provenance before extracting criteria."""
    passes: dict[tuple, list[dict]] = defaultdict(list)
    for evaluation in evaluations:
        passes[_collection_pass_key(evaluation)].append(evaluation)

    comparisons: list[list[int]] = []
    bidirectional_passes = 0
    incomplete_passes = 0
    for records in passes.values():
        if len(records) == 1:
            incomplete_passes += 1
        elif (
            len(records) == 2
            and records[0]["eval1"] == records[1]["eval2"]
            and records[0]["eval2"] == records[1]["eval1"]
        ):
            bidirectional_passes += 1
        else:
            raise ValueError(
                "Corrected pairwise cleaning found an ambiguous collection pass"
            )
        pass_comparisons, _ = extract_comparisons_with_ties_criteria(
            records,
            num_criteria=num_criteria,
        )
        comparisons.extend(_reconcile_pass_comparisons(pass_comparisons))
    return comparisons, {
        "identified_passes": len(passes),
        "bidirectional_passes": bidirectional_passes,
        "incomplete_passes": incomplete_passes,
    }


def load_pairwise_data(
    evaluations_path: Path,
    *,
    num_criteria: int,
    cleaning: Literal["published_legacy", "corrected"] = PAIRWISE_CLEANING,
) -> PairwiseData:
    """Parse pairwise judgments using published or repeat-preserving cleaning."""
    evaluations = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_evaluations = len(evaluations)
    duplicates_removed = 0
    if cleaning == "corrected":
        unique: list[dict] = []
        seen: set[str] = set()
        for evaluation in evaluations:
            key = _evaluation_key(evaluation)
            if key in seen:
                duplicates_removed += 1
                continue
            seen.add(key)
            unique.append(evaluation)
        evaluations = unique
    elif cleaning != "published_legacy":
        raise ValueError(f"Unknown pairwise cleaning mode: {cleaning}")

    pass_diagnostics: dict[str, int] = {}
    if cleaning == "published_legacy":
        comparisons, _ = extract_comparisons_with_ties_criteria(
            evaluations, num_criteria=num_criteria
        )
    else:
        comparisons, pass_diagnostics = _extract_corrected_comparisons(
            evaluations,
            num_criteria=num_criteria,
        )
    extracted_rows = len(comparisons)
    group_sizes: dict[tuple[int, int, int, int, int], int] = defaultdict(int)
    for criterion, scenario, judge, first, second, _ in comparisons:
        group_sizes[
            (criterion, scenario, judge, min(first, second), max(first, second))
        ] += 1
    repeated_groups = sum(size > 2 for size in group_sizes.values())

    if cleaning == "published_legacy":
        comparisons = handle_inconsistencies_with_ties_criteria(comparisons)

    blocks: dict[int, list[list[int]]] = {}
    for _, scenario, judge, left, right, choice in comparisons:
        blocks.setdefault(int(scenario), []).append(
            [int(judge), int(left), int(right), int(choice)]
        )
    block_arrays = {
        scenario: np.asarray(rows, dtype=np.int64)
        for scenario, rows in blocks.items()
    }
    return PairwiseData(
        blocks=block_arrays,
        diagnostics={
            "cleaning": cleaning,
            "raw_evaluations": raw_evaluations,
            "exact_duplicates_removed": duplicates_removed,
            "extracted_criterion_rows": extracted_rows,
            "repeated_judgment_groups": repeated_groups,
            "retained_criterion_rows": len(comparisons),
            "retained_scenarios": len(block_arrays),
            **pass_diagnostics,
        },
    )


def load_pairwise_blocks(
    evaluations_path: Path, *, num_criteria: int
) -> dict[int, np.ndarray]:
    """Return pairwise blocks with repeated collection passes preserved."""
    return load_pairwise_data(
        evaluations_path,
        num_criteria=num_criteria,
        cleaning=PAIRWISE_CLEANING,
    ).blocks


def _unpack(parameters: np.ndarray, num_models: int, dimension: int):
    embedding_size = num_models * dimension
    judges = parameters[:embedding_size].reshape(num_models, dimension)
    targets = parameters[embedding_size : 2 * embedding_size].reshape(
        num_models, dimension
    )
    log_ties = parameters[2 * embedding_size :]
    return judges, targets, log_ties


def _btd_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return the mean Davidson cross-entropy and its analytic gradient."""
    judge_index, left_index, right_index, choices = rows.T
    judges, targets, log_ties = _unpack(parameters, num_models, dimension)
    left_score = np.sum(judges[judge_index] * targets[left_index], axis=1)
    right_score = np.sum(judges[judge_index] * targets[right_index], axis=1)
    logits = np.column_stack(
        (
            log_ties[judge_index] + 0.5 * (left_score + right_score),
            left_score,
            right_score,
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    row_index = np.arange(len(rows))
    loss = -np.log(np.maximum(probabilities[row_index, choices], 1e-300)).mean()

    residual = probabilities
    residual[row_index, choices] -= 1.0
    residual /= len(rows)
    left_gradient = 0.5 * residual[:, 0] + residual[:, 1]
    right_gradient = 0.5 * residual[:, 0] + residual[:, 2]

    judge_gradient = np.zeros_like(judges)
    target_gradient = np.zeros_like(targets)
    tie_gradient = np.zeros_like(log_ties)
    np.add.at(
        judge_gradient,
        judge_index,
        left_gradient[:, None] * targets[left_index]
        + right_gradient[:, None] * targets[right_index],
    )
    np.add.at(
        target_gradient,
        left_index,
        left_gradient[:, None] * judges[judge_index],
    )
    np.add.at(
        target_gradient,
        right_index,
        right_gradient[:, None] * judges[judge_index],
    )
    np.add.at(tie_gradient, judge_index, residual[:, 0])
    gradient = np.concatenate(
        (judge_gradient.ravel(), target_gradient.ravel(), tie_gradient)
    )
    return float(loss), gradient


def _pairwise_trust(
    parameters: np.ndarray, *, num_models: int, dimension: int
) -> np.ndarray:
    """Apply EigenBench's Davidson tie contribution and EigenTrust propagation."""
    judges, targets, log_ties = _unpack(parameters, num_models, dimension)
    score = np.exp(np.clip(judges @ targets.T, -40.0, 40.0))
    root_score = np.sqrt(score)
    tie_contribution = (
        0.5
        * np.exp(np.clip(log_ties, -40.0, 40.0))[:, None]
        * root_score
        * (root_score.sum(axis=1, keepdims=True) - root_score)
    )
    matrix = score + tie_contribution
    matrix /= matrix.sum(axis=1, keepdims=True)
    trust = np.full(num_models, 1.0 / num_models)
    for _ in range(10_000):
        next_trust = trust @ matrix
        if np.linalg.norm(next_trust - trust, ord=1) < 1e-12:
            return next_trust
        trust = next_trust
    raise RuntimeError("Pairwise EigenTrust did not converge")


def fit_pairwise_btd(
    rows: np.ndarray,
    *,
    initial: np.ndarray,
    num_models: int = 8,
    dimension: int = 2,
    max_iterations: int = 1_000,
) -> PairwiseFit:
    """Fit the published rank-two Davidson likelihood deterministically."""
    result = minimize(
        partial(
            _btd_loss_gradient,
            num_models=num_models,
            dimension=dimension,
        ),
        initial,
        args=(rows,),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 30,
        },
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Pairwise BTD optimization produced a non-finite result")
    if not result.success:
        raise RuntimeError(
            "Pairwise BTD optimization failed after "
            f"{result.nit} iterations: {result.message}"
        )
    return PairwiseFit(
        parameters=result.x,
        trust=_pairwise_trust(
            result.x, num_models=num_models, dimension=dimension
        ),
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        parameter_l2=float(np.linalg.norm(result.x)),
    )


def fit_pairwise_btd_multistart(
    rows: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_models: int = 8,
    dimension: int = 2,
    max_iterations: int = 1_000,
    near_loss_tolerance: float = 1e-7,
    trust_l1_tolerance: float = 1e-3,
    reject_unstable: bool = True,
) -> PairwiseFit:
    """Choose the best converged fit and detect incompatible near-optima."""
    if len(initials) < 2:
        raise ValueError("Multi-start BTD requires at least two initializations")
    fits: list[tuple[int, PairwiseFit]] = []
    errors: list[str] = []
    for start_index, initial in enumerate(initials):
        try:
            fit = fit_pairwise_btd(
                rows,
                initial=initial,
                num_models=num_models,
                dimension=dimension,
                max_iterations=max_iterations,
            )
        except RuntimeError as error:
            errors.append(f"start {start_index}: {error}")
        else:
            fits.append((start_index, fit))
    if len(fits) < 2:
        detail = "; ".join(errors) if errors else "fewer than two starts converged"
        raise RuntimeError(f"Pairwise BTD multi-start validation failed: {detail}")

    best_start, best = min(fits, key=lambda item: item[1].loss)
    loss_cutoff = best.loss + max(near_loss_tolerance, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= loss_cutoff]
    max_l1 = max(float(np.abs(fit.trust - best.trust).sum()) for fit in near)
    stable_near_optima = max_l1 <= trust_l1_tolerance
    if reject_unstable and not stable_near_optima:
        raise RuntimeError(
            "Pairwise BTD has near-optimal fits with incompatible trust vectors: "
            f"maximum L1 distance {max_l1:.6g}"
        )
    return PairwiseFit(
        parameters=best.parameters,
        trust=best.trust,
        loss=best.loss,
        iterations=best.iterations,
        attempted_starts=len(initials),
        successful_starts=len(fits),
        best_start=best_start,
        near_optimal_starts=len(near),
        max_near_optimal_trust_l1=max_l1,
        stable_near_optima=stable_near_optima,
        gradient_l2=best.gradient_l2,
        parameter_l2=best.parameter_l2,
        start_diagnostics=tuple(
            {
                "start": start_index,
                "loss": fit.loss,
                "iterations": fit.iterations,
                "gradient_l2": fit.gradient_l2,
                "parameter_l2": fit.parameter_l2,
                "trust": fit.trust.tolist(),
            }
            for start_index, fit in fits
        ),
        failed_starts=tuple(errors),
    )


def initial_btd_parameters(seed: int, num_models: int = 8, dimension: int = 2):
    """Create a deterministic small-normal embedding initialization."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0.0, 0.1, 2 * num_models * dimension)
    return np.concatenate((embeddings, np.zeros(num_models)))


def concatenate_blocks(
    blocks: dict[int, np.ndarray], scenario_ids: np.ndarray
) -> np.ndarray:
    """Expand sampled scenario IDs into their complete comparison blocks."""
    return np.concatenate([blocks[int(scenario)] for scenario in scenario_ids])


def sha256_file(path: Path) -> str:
    """Return a source artifact's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jackknife_interval(
    point: np.ndarray,
    leave_group_out: np.ndarray,
    *,
    omitted_sizes: np.ndarray,
    total_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unequal delete-group pseudovalue estimates and normal intervals."""
    omitted_sizes = np.asarray(omitted_sizes, dtype=float)
    if leave_group_out.shape[0] != len(omitted_sizes):
        raise ValueError("Each jackknife replicate requires one omitted-group size")
    if np.any(omitted_sizes <= 0) or not np.isclose(omitted_sizes.sum(), total_size):
        raise ValueError("Omitted groups must form a partition of the full sample")

    expansion = total_size / omitted_sizes
    pseudovalues = (
        expansion[:, None] * point
        - (expansion[:, None] - 1.0) * leave_group_out
    )
    estimate = np.sum(pseudovalues / expansion[:, None], axis=0)
    variance = np.mean(
        np.square(pseudovalues - estimate)
        / (expansion[:, None] - 1.0),
        axis=0,
    )
    standard_error = np.sqrt(variance)
    return (
        estimate,
        standard_error,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )


def multistart_initials(
    warm_start: np.ndarray, *, seed: int, cold_starts: int = 2
) -> list[np.ndarray]:
    """Build a deterministic warm-plus-cold initialization bank."""
    if cold_starts < 1:
        raise ValueError("At least one cold BTD start is required")
    return [warm_start] + [
        initial_btd_parameters(seed + offset) for offset in range(cold_starts)
    ]


def ranking_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compare two trust vectors without treating their scales as identical."""
    return {
        "l1": float(np.abs(left - right).sum()),
        "spearman": float(spearmanr(left, right).statistic),
        "kendall": float(kendalltau(left, right).statistic),
    }


def _fit_diagnostics(fit: PairwiseFit) -> dict[str, object]:
    """Return serializable optimizer and multi-start diagnostics."""
    return {
        "loss": fit.loss,
        "iterations": fit.iterations,
        "attempted_starts": fit.attempted_starts,
        "successful_starts": fit.successful_starts,
        "best_start": fit.best_start,
        "near_optimal_starts": fit.near_optimal_starts,
        "max_near_optimal_trust_l1": fit.max_near_optimal_trust_l1,
        "stable_near_optima": fit.stable_near_optima,
        "gradient_l2": fit.gradient_l2,
        "parameter_l2": fit.parameter_l2,
        "starts": list(fit.start_diagnostics),
        "failed_starts": list(fit.failed_starts),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _save_bootstrap_checkpoint(
    output_dir: Path,
    *,
    numerical_draws: np.ndarray,
    pairwise_draws: np.ndarray,
    completed: np.ndarray,
    diagnostics: list[dict | None],
    seed: int,
    input_fingerprint: str,
) -> None:
    """Atomically persist completed bootstrap draws for exact resumption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "bootstrap_checkpoint.npz"
    with tempfile.NamedTemporaryFile(
        "wb", dir=output_dir, suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(
            handle,
            numerical_draws=numerical_draws,
            pairwise_draws=pairwise_draws,
            completed=completed,
            seed=np.asarray(seed),
            input_fingerprint=np.asarray(input_fingerprint),
        )
        temporary = Path(handle.name)
    os.replace(temporary, checkpoint)
    _atomic_json(output_dir / "bootstrap_checkpoint_diagnostics.json", diagnostics)


def _load_bootstrap_checkpoint(
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    input_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict | None]] | None:
    """Load a compatible bootstrap checkpoint when one exists."""
    checkpoint = output_dir / "bootstrap_checkpoint.npz"
    diagnostic_path = output_dir / "bootstrap_checkpoint_diagnostics.json"
    if not checkpoint.exists() or not diagnostic_path.exists():
        return None
    with np.load(checkpoint) as data:
        numerical_draws = data["numerical_draws"]
        pairwise_draws = data["pairwise_draws"]
        completed = data["completed"]
        checkpoint_seed = int(data["seed"])
        checkpoint_fingerprint = str(data["input_fingerprint"])
    expected_shape = (bootstrap_samples, 8)
    if (
        numerical_draws.shape != expected_shape
        or pairwise_draws.shape != expected_shape
        or completed.shape != (bootstrap_samples,)
        or checkpoint_seed != seed
        or checkpoint_fingerprint != input_fingerprint
    ):
        raise ValueError("Bootstrap checkpoint does not match this run")
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if len(diagnostics) != bootstrap_samples:
        raise ValueError("Bootstrap checkpoint diagnostics have the wrong length")
    diagnostic_mask = np.asarray([item is not None for item in diagnostics])
    if not np.array_equal(diagnostic_mask, completed):
        raise ValueError("Bootstrap checkpoint completion metadata is inconsistent")
    if (
        not np.isfinite(numerical_draws[completed]).all()
        or not np.isfinite(pairwise_draws[completed]).all()
    ):
        raise ValueError("Bootstrap checkpoint contains non-finite completed draws")
    return numerical_draws, pairwise_draws, completed, diagnostics


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
    # Keep collection-pass provenance through tie reconciliation. Flattening
    # first makes distinct repeated passes indistinguishable and drops groups
    # containing more than one forward/reverse pair.
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
    legacy_pairwise_rows = concatenate_blocks(
        legacy_blocks, legacy_scenarios
    )
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

    # Every bootstrap draw selects the same scenarios for both methods.
    bootstrap_rng = np.random.default_rng(seed)
    sampled_positions = bootstrap_rng.integers(
        0,
        len(matched_scenarios),
        size=(bootstrap_samples, len(matched_scenarios)),
    )
    checkpoint = _load_bootstrap_checkpoint(
        output_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        input_fingerprint=input_fingerprint,
    )
    if checkpoint is None:
        numerical_draws = np.full((bootstrap_samples, 8), np.nan, dtype=float)
        pairwise_draws = np.full((bootstrap_samples, 8), np.nan, dtype=float)
        completed_draws = np.zeros(bootstrap_samples, dtype=bool)
        pairwise_diagnostics: list[dict | None] = [None] * bootstrap_samples
    else:
        (
            numerical_draws,
            pairwise_draws,
            completed_draws,
            pairwise_diagnostics,
        ) = checkpoint
        print(
            f"Resuming {int(completed_draws.sum())}/{bootstrap_samples} "
            "bootstrap draws",
            flush=True,
        )

    def run_bootstrap_draw(
        draw_index: int, positions: np.ndarray
    ) -> tuple[int, np.ndarray, PairwiseFit]:
        numerical = compute_trust(
            matched_ratings[:, positions, :]
        ).trust
        sampled_scenarios = matched_scenarios[positions]
        fit = fit_pairwise_btd_multistart(
            concatenate_blocks(blocks, sampled_scenarios),
            initials=multistart_initials(
                pairwise_point.parameters,
                seed=seed + 10_000 + 2 * draw_index,
            ),
            reject_unstable=False,
        )
        return draw_index, numerical, fit

    completed = int(completed_draws.sum())
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [
        executor.submit(
            run_bootstrap_draw,
            int(draw_index),
            sampled_positions[draw_index],
        )
        for draw_index in np.flatnonzero(~completed_draws)
    ]
    try:
        for future in as_completed(futures):
            draw_index, numerical, fit = future.result()
            numerical_draws[draw_index] = numerical
            pairwise_draws[draw_index] = fit.trust
            completed_draws[draw_index] = True
            pairwise_diagnostics[draw_index] = {
                "draw": draw_index,
                **_fit_diagnostics(fit),
            }
            completed += 1
            if completed % 50 == 0 or completed == bootstrap_samples:
                _save_bootstrap_checkpoint(
                    output_dir,
                    numerical_draws=numerical_draws,
                    pairwise_draws=pairwise_draws,
                    completed=completed_draws,
                    diagnostics=pairwise_diagnostics,
                    seed=seed,
                    input_fingerprint=input_fingerprint,
                )
                print(f"Bootstrap {completed}/{bootstrap_samples}", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        _save_bootstrap_checkpoint(
            output_dir,
            numerical_draws=numerical_draws,
            pairwise_draws=pairwise_draws,
            completed=completed_draws,
            diagnostics=pairwise_diagnostics,
            seed=seed,
            input_fingerprint=input_fingerprint,
        )
        raise
    else:
        executor.shutdown(wait=True)

    if not completed_draws.all():
        raise RuntimeError("Bootstrap ended with incomplete draws")

    if any(diagnostic is None for diagnostic in pairwise_diagnostics):
        raise RuntimeError("Missing pairwise bootstrap fit diagnostics")
    unstable_draws = [
        int(diagnostic["draw"])
        for diagnostic in pairwise_diagnostics
        if diagnostic is not None and not diagnostic["stable_near_optima"]
    ]

    numerical_lower, numerical_upper = np.quantile(
        numerical_draws, [0.025, 0.975], axis=0
    )
    pairwise_lower, pairwise_upper = np.quantile(
        pairwise_draws, [0.025, 0.975], axis=0
    )
    numerical_ranks = (-numerical_draws).argsort(axis=1).argsort(axis=1) + 1
    pairwise_ranks = (-pairwise_draws).argsort(axis=1).argsort(axis=1) + 1
    numerical_rank_lower, numerical_rank_upper = np.quantile(
        numerical_ranks, [0.025, 0.975], axis=0
    )
    pairwise_rank_lower, pairwise_rank_upper = np.quantile(
        pairwise_ranks, [0.025, 0.975], axis=0
    )
    paired_difference = numerical_draws - pairwise_draws
    paired_difference_lower, paired_difference_upper = np.quantile(
        paired_difference, [0.025, 0.975], axis=0
    )
    bootstrap_metrics = [
        {
            "draw": index,
            **ranking_metrics(numerical_draws[index], pairwise_draws[index]),
        }
        for index in range(bootstrap_samples)
    ]

    # Fixed groups are omitted, not fitted alone: the complements remain identified.
    partition_rng = np.random.default_rng(seed + 1)
    shuffled = partition_rng.permutation(len(matched_scenarios))
    group_positions = np.array_split(shuffled, jackknife_groups)
    leave_out_rows: list[dict] = []
    group_rows: list[dict] = []
    jackknife_diagnostics: list[dict] = []
    numerical_leave_out = np.empty((jackknife_groups, 8), dtype=float)
    pairwise_leave_out = np.empty((jackknife_groups, 8), dtype=float)
    for group_index, positions in enumerate(group_positions):
        omitted_scenarios = matched_scenarios[positions]
        keep = np.ones(len(matched_scenarios), dtype=bool)
        keep[positions] = False
        numerical_leave_out[group_index] = compute_trust(
            matched_ratings[:, keep, :]
        ).trust
        leave_out_fit = fit_pairwise_btd_multistart(
            concatenate_blocks(blocks, matched_scenarios[keep]),
            initials=multistart_initials(
                pairwise_point.parameters,
                seed=seed + 20_000 + 2 * group_index,
            ),
        )
        pairwise_leave_out[group_index] = leave_out_fit.trust
        jackknife_diagnostics.append(
            {
                "group": group_index,
                "omitted_scenarios": len(positions),
                "retained_scenarios": int(keep.sum()),
                **_fit_diagnostics(leave_out_fit),
            }
        )
        for scenario in omitted_scenarios:
            group_rows.append({"scenario_index": int(scenario), "group": group_index})
        for method, values in (
            ("numerical", numerical_leave_out[group_index]),
            ("pairwise", pairwise_leave_out[group_index]),
        ):
            ranks = (-values).argsort().argsort() + 1
            for model_id, model_name in enumerate(model_names):
                leave_out_rows.append(
                    {
                        "omitted_group": group_index,
                        "omitted_scenarios": len(positions),
                        "retained_scenarios": int(keep.sum()),
                        "method": method,
                        "model_id": model_id,
                        "model_name": model_name,
                        "trust": float(values[model_id]),
                        "rank": int(ranks[model_id]),
                    }
                )

    omitted_sizes = np.asarray([len(group) for group in group_positions])
    numerical_jackknife = jackknife_interval(
        numerical_point,
        numerical_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    pairwise_jackknife = jackknife_interval(
        pairwise_point.trust,
        pairwise_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    difference_point = numerical_point - pairwise_point.trust
    difference_jackknife = jackknife_interval(
        difference_point,
        numerical_leave_out - pairwise_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    summary_rows = []
    numerical_point_ranks = (-numerical_point).argsort().argsort() + 1
    pairwise_point_ranks = (-pairwise_point.trust).argsort().argsort() + 1
    for model_id, model_name in enumerate(model_names):
        summary_rows.append(
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
        )

    metric_names = ("l1", "spearman", "kendall")
    metric_intervals = {
        name: np.quantile(
            [row[name] for row in bootstrap_metrics], [0.025, 0.5, 0.975]
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
            "pairwise_optimizer": "deterministic multi-start L-BFGS-B with analytic gradient",
            "pairwise_cleaning": PAIRWISE_CLEANING,
            "interval": "95% percentile bootstrap",
            "partition_interval": "unequal delete-group jackknife normal interval",
            "jackknife_group_size_note": (
                "Delete-m_j pseudovalues account for one group of 88 scenarios "
                "and nine groups of 87"
            ),
            "uncertainty_scope": "scenario selection with stored target and judge responses held fixed",
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
                **_fit_diagnostics(pairwise_point),
                "agreement_with_published": ranking_metrics(
                    pairwise_point.trust, published_pairwise
                ),
            },
            "legacy_cleaning_sensitivity": {
                "trust": legacy_pairwise_point.trust.tolist(),
                "elo": eigentrust_elo(legacy_pairwise_point.trust).tolist(),
                "fit": _fit_diagnostics(legacy_pairwise_point),
                "agreement_with_primary_cleaning": ranking_metrics(
                    legacy_pairwise_point.trust, pairwise_point.trust
                ),
            },
            "numerical_vs_pairwise_refit": ranking_metrics(
                numerical_point, pairwise_point.trust
            ),
        },
        "bootstrap": {
            "draw_shape": [bootstrap_samples, len(matched_scenarios)],
            "optimizer_unstable_draw_count": len(unstable_draws),
            "optimizer_unstable_draws": unstable_draws,
            "scenario_draws_reproducible_from": {
                "generator": "numpy.default_rng",
                "seed": seed,
                "low": 0,
                "high": len(matched_scenarios),
            },
            "paired_metric_95_interval": metric_intervals,
            "pairwise_fit_diagnostics": pairwise_diagnostics,
        },
        "jackknife": {
            "groups": [
                matched_scenarios[positions].tolist()
                for positions in group_positions
            ],
            "pairwise_fit_diagnostics": jackknife_diagnostics,
            "numerical_bias_corrected_estimate": numerical_jackknife[0].tolist(),
            "numerical_standard_error": numerical_jackknife[1].tolist(),
            "numerical_lower": numerical_jackknife[2].tolist(),
            "numerical_upper": numerical_jackknife[3].tolist(),
            "pairwise_bias_corrected_estimate": pairwise_jackknife[0].tolist(),
            "pairwise_standard_error": pairwise_jackknife[1].tolist(),
            "pairwise_lower": pairwise_jackknife[2].tolist(),
            "pairwise_upper": pairwise_jackknife[3].tolist(),
            "paired_difference_bias_corrected_estimate": difference_jackknife[0].tolist(),
            "paired_difference_standard_error": difference_jackknife[1].tolist(),
            "paired_difference_lower": difference_jackknife[2].tolist(),
            "paired_difference_upper": difference_jackknife[3].tolist(),
        },
    }

    _atomic_json(output_dir / "results.json", result)
    _atomic_csv(
        output_dir / "bootstrap_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    _atomic_csv(
        output_dir / "jackknife_leave_group_out.csv",
        list(leave_out_rows[0]),
        leave_out_rows,
    )
    _atomic_csv(
        output_dir / "scenario_groups.csv",
        ["scenario_index", "group"],
        group_rows,
    )
    _atomic_csv(
        output_dir / "paired_bootstrap_metrics.csv",
        list(bootstrap_metrics[0]),
        bootstrap_metrics,
    )
    bootstrap_trust_rows = []
    for draw_index in range(bootstrap_samples):
        row: dict[str, int | float] = {"draw": draw_index}
        for model_id in range(8):
            row[f"numerical_{model_id}"] = numerical_draws[draw_index, model_id]
            row[f"pairwise_{model_id}"] = pairwise_draws[draw_index, model_id]
        bootstrap_trust_rows.append(row)
    _atomic_csv(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--jackknife-groups", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        ratings_path=args.ratings,
        evaluations_path=args.evaluations,
        meta_path=args.meta,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        jackknife_groups=args.jackknife_groups,
        workers=args.workers,
        seed=args.seed,
    )
    comparison = result["point"]["numerical_vs_pairwise_refit"]
    print(
        f"Matched 871 scenarios: Spearman={comparison['spearman']:.3f}, "
        f"Kendall={comparison['kendall']:.3f}"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
