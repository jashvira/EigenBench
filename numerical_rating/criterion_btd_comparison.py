"""Compare direct rating margins with criterion-conditioned pairwise BTD.

Both models use the score ``u[criterion, judge] @ v[model]`` and train on the
same scenario, criterion, judge, and model-pair rows. Direct ratings use MSE on
the rating gap; pairwise trits use Davidson cross-entropy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

from numerical_rating.data import sha256_text
from numerical_rating.direct_embedding_comparison import (
    _fixed_score_logits,
    _softmax_logits,
    aligned_embedding_metrics,
    classification_metrics,
    fit_direct_calibration,
    fit_rank_svd,
    geometry_metrics,
    load_criterion_tensor,
    scenario_folds,
    standardize_ratings,
)
from numerical_rating.scenario_uncertainty import (
    _evaluation_key,
    _extract_corrected_comparisons,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_criterion_round_robin/ratings.csv"
)
DEFAULT_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_round_robin/"
    "direct_embedding_comparison/criterion_btd_comparison"
)


@dataclass(frozen=True)
class MatchedCriterionData:
    ratings: np.ndarray
    criterion_ids: list[str]
    rating_scenario_ids: np.ndarray
    scenario_ids: np.ndarray
    judge_names: list[str]
    model_names: list[str]
    blocks: dict[int, np.ndarray]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class MarginFit:
    parameters: np.ndarray
    row_vectors: np.ndarray
    model_vectors: np.ndarray
    scores: np.ndarray
    loss: float
    iterations: int
    gradient_l2: float
    loss_trace: tuple[dict[str, float | int], ...] = ()
    attempted_starts: int = 1
    successful_starts: int = 1
    best_start: int = 0
    min_near_score_correlation: float = 1.0
    start_diagnostics: tuple[dict[str, object], ...] = ()
    failed_starts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriterionBTDFit:
    parameters: np.ndarray
    row_vectors: np.ndarray
    model_vectors: np.ndarray
    log_ties: np.ndarray
    scores: np.ndarray
    loss: float
    iterations: int
    gradient_l2: float
    loss_trace: tuple[dict[str, float | int], ...] = ()
    attempted_starts: int = 1
    successful_starts: int = 1
    best_start: int = 0
    min_near_score_correlation: float = 1.0
    start_diagnostics: tuple[dict[str, object], ...] = ()
    failed_starts: tuple[str, ...] = ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_hashes(path: Path) -> dict[tuple[int, int], str]:
    hashes: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scenario_index"]), int(row["model_id"]))
            response_hash = row["response_hash"]
            if key in hashes and hashes[key] != response_hash:
                raise ValueError(f"Response hash changed for cell {key}")
            hashes[key] = response_hash
    return hashes


def _responses_match(
    evaluation: dict, response_hashes: dict[tuple[int, int], str]
) -> bool:
    scenario = int(evaluation["scenario_index"])
    return all(
        response_hashes[(scenario, int(evaluation[model_key]))]
        == sha256_text(evaluation[response_key])
        for model_key, response_key in (
            ("eval1", "eval1 response"),
            ("eval2", "eval2 response"),
        )
    )


def load_matched_criterion_data(
    ratings_path: Path, evaluations_path: Path
) -> MatchedCriterionData:
    """Load exact-response criterion ratings and pairwise rows."""
    (
        ratings,
        criterion_ids,
        criterion_hashes,
        rating_scenarios,
        judge_names,
        model_names,
    ) = load_criterion_tensor(ratings_path)
    response_hashes = _response_hashes(ratings_path)

    evaluations = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique: list[dict] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        key = _evaluation_key(evaluation)
        if key not in seen:
            seen.add(key)
            unique.append(evaluation)

    constitutions = {evaluation["constitution"] for evaluation in unique}
    if len(constitutions) != 1:
        raise ValueError("Expected one constitution in pairwise evaluations")
    criterion_texts = next(iter(constitutions)).splitlines()
    if [sha256_text(text) for text in criterion_texts] != criterion_hashes:
        raise ValueError("Numerical and pairwise criterion text differs")

    exact = [
        evaluation
        for evaluation in unique
        if _responses_match(evaluation, response_hashes)
    ]
    comparisons, extraction = _extract_corrected_comparisons(
        exact, num_criteria=len(criterion_ids)
    )
    if extraction["incomplete_passes"]:
        raise ValueError("Response filtering split a bidirectional collection pass")

    rating_scenario_set = set(rating_scenarios)
    blocks: dict[int, list[list[int]]] = {}
    for criterion, scenario, judge, left, right, choice in comparisons:
        if scenario not in rating_scenario_set:
            raise ValueError(f"Pairwise scenario {scenario} lacks numerical ratings")
        row_index = int(criterion) * ratings.shape[1] + int(judge)
        blocks.setdefault(int(scenario), []).append(
            [row_index, int(left), int(right), int(choice)]
        )
    block_arrays = {
        scenario: np.asarray(rows, dtype=np.int64)
        for scenario, rows in blocks.items()
    }
    for evaluation in exact:
        for model_key, name_key in (
            ("eval1", "eval1_name"),
            ("eval2", "eval2_name"),
        ):
            model = int(evaluation[model_key])
            if str(evaluation[name_key]) != model_names[model]:
                raise ValueError(f"Pairwise target ID {model} has the wrong name")
        judge = int(evaluation["judge"])
        pairwise_judge = str(evaluation["judge_name"])
        expected_judge = judge_names[judge]
        if pairwise_judge != expected_judge and not (
            judge == 3
            and pairwise_judge == "Grok 4"
            and expected_judge == "Grok 4.3"
        ):
            raise ValueError(f"Pairwise judge ID {judge} has the wrong name")

    canonical_tuples = {
        (scenario, row[0], min(row[1], row[2]), max(row[1], row[2]))
        for scenario, rows in block_arrays.items()
        for row in rows
    }
    return MatchedCriterionData(
        ratings=ratings,
        criterion_ids=criterion_ids,
        rating_scenario_ids=np.asarray(rating_scenarios, dtype=np.int64),
        scenario_ids=np.asarray(sorted(block_arrays), dtype=np.int64),
        judge_names=judge_names,
        model_names=model_names,
        blocks=block_arrays,
        diagnostics={
            "raw_pairwise_records": len(evaluations),
            "exact_duplicates_removed": len(evaluations) - len(unique),
            "response_mismatched_records_removed": len(unique) - len(exact),
            "exact_response_pairwise_records": len(exact),
            "matched_criterion_rows": len(comparisons),
            "unique_matched_tuples": len(canonical_tuples),
            "matched_scenarios": len(block_arrays),
            **extraction,
        },
    )


def _unpack_factors(
    parameters: np.ndarray, *, num_rows: int, num_models: int, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    row_size = num_rows * dimension
    row_vectors = parameters[:row_size].reshape(num_rows, dimension)
    model_vectors = parameters[row_size:].reshape(num_models, dimension)
    model_vectors = model_vectors - model_vectors.mean(axis=0, keepdims=True)
    return row_vectors, model_vectors


def margin_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return MSE and gradient for criterion-conditioned score gaps."""
    row_index, left, right, _ = rows.T
    row_vectors, model_vectors = _unpack_factors(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    prediction = np.sum(
        row_vectors[row_index] * (model_vectors[left] - model_vectors[right]),
        axis=1,
    )
    error = prediction - targets
    loss = float(np.mean(np.square(error)))
    coefficient = 2.0 * error / len(rows)

    row_gradient = np.zeros_like(row_vectors)
    model_gradient = np.zeros_like(model_vectors)
    np.add.at(
        row_gradient,
        row_index,
        coefficient[:, None] * (model_vectors[left] - model_vectors[right]),
    )
    contribution = coefficient[:, None] * row_vectors[row_index]
    np.add.at(model_gradient, left, contribution)
    np.add.at(model_gradient, right, -contribution)
    model_gradient -= model_gradient.mean(axis=0, keepdims=True)
    return loss, np.concatenate((row_gradient.ravel(), model_gradient.ravel()))


def _fit_margin_once(
    rows: np.ndarray,
    targets: np.ndarray,
    initial: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> MarginFit:
    objective = partial(
        margin_loss_gradient,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    trace: list[dict[str, float | int]] = []

    def record(parameters: np.ndarray) -> None:
        loss, gradient = objective(parameters, rows, targets)
        trace.append(
            {
                "iteration": len(trace),
                "loss": float(loss),
                "gradient_l2": float(np.linalg.norm(gradient)),
            }
        )

    record(initial)
    result = minimize(
        objective,
        initial,
        args=(rows, targets),
        method="L-BFGS-B",
        jac=True,
        callback=record,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 30,
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Margin MSE optimization failed: {result.message}")
    if not np.isclose(trace[-1]["loss"], result.fun, rtol=1e-10, atol=1e-12):
        record(result.x)
    row_vectors, model_vectors = _unpack_factors(
        result.x,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    return MarginFit(
        parameters=result.x,
        row_vectors=row_vectors,
        model_vectors=model_vectors,
        scores=row_vectors @ model_vectors.T,
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        loss_trace=tuple(trace),
    )


def _score_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    return float(pearsonr(left.ravel(), right.ravel()).statistic)


def fit_margin_multistart(
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> MarginFit:
    fits: list[tuple[int, MarginFit]] = []
    errors: list[str] = []
    for start, initial in enumerate(initials):
        try:
            fit = _fit_margin_once(
                rows,
                targets,
                initial,
                num_rows=num_rows,
                num_models=num_models,
                dimension=dimension,
                max_iterations=max_iterations,
            )
        except RuntimeError as error:
            errors.append(f"start {start}: {error}")
        else:
            fits.append((start, fit))
    if not fits:
        raise RuntimeError("All margin MSE starts failed: " + "; ".join(errors))
    best_start, best = min(fits, key=lambda item: item[1].loss)
    cutoff = best.loss + max(1e-8, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= cutoff]
    min_correlation = min(_score_correlation(best.scores, fit.scores) for fit in near)
    return MarginFit(
        parameters=best.parameters,
        row_vectors=best.row_vectors,
        model_vectors=best.model_vectors,
        scores=best.scores,
        loss=best.loss,
        iterations=best.iterations,
        gradient_l2=best.gradient_l2,
        loss_trace=best.loss_trace,
        attempted_starts=len(initials),
        successful_starts=len(fits),
        best_start=best_start,
        min_near_score_correlation=min_correlation,
        start_diagnostics=tuple(
            {
                "start": start,
                "loss": fit.loss,
                "iterations": fit.iterations,
                "gradient_l2": fit.gradient_l2,
                "trace": list(fit.loss_trace),
                "score_correlation_to_best": _score_correlation(
                    best.scores, fit.scores
                ),
            }
            for start, fit in fits
        ),
        failed_starts=tuple(errors),
    )


def _unpack_btd(
    parameters: np.ndarray, *, num_rows: int, num_models: int, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor_size = (num_rows + num_models) * dimension
    row_vectors, model_vectors = _unpack_factors(
        parameters[:factor_size],
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    return row_vectors, model_vectors, parameters[factor_size:]


def criterion_btd_logits(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> np.ndarray:
    row_index, left, right, _ = rows.T
    row_vectors, model_vectors, log_ties = _unpack_btd(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    left_score = np.sum(row_vectors[row_index] * model_vectors[left], axis=1)
    right_score = np.sum(row_vectors[row_index] * model_vectors[right], axis=1)
    return np.column_stack(
        (
            log_ties[row_index] + 0.5 * (left_score + right_score),
            left_score,
            right_score,
        )
    )


def criterion_btd_loss_gradient(
    parameters: np.ndarray,
    rows: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
) -> tuple[float, np.ndarray]:
    """Return Davidson cross-entropy and gradient for criterion rows."""
    row_index, left, right, choice = rows.T
    row_vectors, model_vectors, _ = _unpack_btd(
        parameters,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    probabilities = _softmax_logits(
        criterion_btd_logits(
            parameters,
            rows,
            num_rows=num_rows,
            num_models=num_models,
            dimension=dimension,
        )
    )
    index = np.arange(len(rows))
    loss = float(-np.log(np.maximum(probabilities[index, choice], 1e-300)).mean())
    residual = probabilities
    residual[index, choice] -= 1.0
    residual /= len(rows)
    left_coefficient = 0.5 * residual[:, 0] + residual[:, 1]
    right_coefficient = 0.5 * residual[:, 0] + residual[:, 2]

    row_gradient = np.zeros_like(row_vectors)
    model_gradient = np.zeros_like(model_vectors)
    tie_gradient = np.zeros(num_rows)
    np.add.at(
        row_gradient,
        row_index,
        left_coefficient[:, None] * model_vectors[left]
        + right_coefficient[:, None] * model_vectors[right],
    )
    np.add.at(
        model_gradient,
        left,
        left_coefficient[:, None] * row_vectors[row_index],
    )
    np.add.at(
        model_gradient,
        right,
        right_coefficient[:, None] * row_vectors[row_index],
    )
    model_gradient -= model_gradient.mean(axis=0, keepdims=True)
    np.add.at(tie_gradient, row_index, residual[:, 0])
    return loss, np.concatenate(
        (row_gradient.ravel(), model_gradient.ravel(), tie_gradient)
    )


def _fit_btd_once(
    rows: np.ndarray,
    initial: np.ndarray,
    *,
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> CriterionBTDFit:
    objective = partial(
        criterion_btd_loss_gradient,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    trace: list[dict[str, float | int]] = []

    def record(parameters: np.ndarray) -> None:
        loss, gradient = objective(parameters, rows)
        trace.append(
            {
                "iteration": len(trace),
                "loss": float(loss),
                "gradient_l2": float(np.linalg.norm(gradient)),
            }
        )

    record(initial)
    result = minimize(
        objective,
        initial,
        args=(rows,),
        method="L-BFGS-B",
        jac=True,
        callback=record,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 30,
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Criterion BTD optimization failed: {result.message}")
    if not np.isclose(trace[-1]["loss"], result.fun, rtol=1e-10, atol=1e-12):
        record(result.x)
    row_vectors, model_vectors, log_ties = _unpack_btd(
        result.x,
        num_rows=num_rows,
        num_models=num_models,
        dimension=dimension,
    )
    return CriterionBTDFit(
        parameters=result.x,
        row_vectors=row_vectors,
        model_vectors=model_vectors,
        log_ties=log_ties,
        scores=row_vectors @ model_vectors.T,
        loss=float(result.fun),
        iterations=int(result.nit),
        gradient_l2=float(np.linalg.norm(result.jac)),
        loss_trace=tuple(trace),
    )


def fit_btd_multistart(
    rows: np.ndarray,
    *,
    initials: list[np.ndarray],
    num_rows: int,
    num_models: int,
    dimension: int,
    max_iterations: int,
) -> CriterionBTDFit:
    fits: list[tuple[int, CriterionBTDFit]] = []
    errors: list[str] = []
    for start, initial in enumerate(initials):
        try:
            fit = _fit_btd_once(
                rows,
                initial,
                num_rows=num_rows,
                num_models=num_models,
                dimension=dimension,
                max_iterations=max_iterations,
            )
        except RuntimeError as error:
            errors.append(f"start {start}: {error}")
        else:
            fits.append((start, fit))
    if not fits:
        raise RuntimeError("All criterion BTD starts failed: " + "; ".join(errors))
    best_start, best = min(fits, key=lambda item: item[1].loss)
    cutoff = best.loss + max(1e-8, abs(best.loss) * 1e-8)
    near = [fit for _, fit in fits if fit.loss <= cutoff]
    min_correlation = min(_score_correlation(best.scores, fit.scores) for fit in near)
    return CriterionBTDFit(
        parameters=best.parameters,
        row_vectors=best.row_vectors,
        model_vectors=best.model_vectors,
        log_ties=best.log_ties,
        scores=best.scores,
        loss=best.loss,
        iterations=best.iterations,
        gradient_l2=best.gradient_l2,
        loss_trace=best.loss_trace,
        attempted_starts=len(initials),
        successful_starts=len(fits),
        best_start=best_start,
        min_near_score_correlation=min_correlation,
        start_diagnostics=tuple(
            {
                "start": start,
                "loss": fit.loss,
                "iterations": fit.iterations,
                "gradient_l2": fit.gradient_l2,
                "trace": list(fit.loss_trace),
                "score_correlation_to_best": _score_correlation(
                    best.scores, fit.scores
                ),
            }
            for start, fit in fits
        ),
        failed_starts=tuple(errors),
    )


def _factor_initials(
    standardized_ratings: np.ndarray,
    *,
    dimension: int,
    starts: int,
    seed: int,
) -> list[np.ndarray]:
    warm = fit_rank_svd(standardized_ratings.mean(axis=1), dimension)
    initials = [
        np.concatenate(
            (
                warm.judges.ravel(),
                (warm.models - warm.models.mean(axis=0, keepdims=True)).ravel(),
            )
        )
    ]
    rng = np.random.default_rng(seed)
    parameter_count = (
        standardized_ratings.shape[0] + standardized_ratings.shape[2]
    ) * dimension
    initials.extend(
        rng.normal(0.0, 0.1, parameter_count) for _ in range(starts - 1)
    )
    return initials


def fit_svd_direct(
    standardized_ratings: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    dimension: int,
) -> MarginFit:
    """Fit the proposed direct model by truncated SVD of mean ratings."""
    svd = fit_rank_svd(standardized_ratings.mean(axis=1), dimension)
    model_vectors = svd.models - svd.models.mean(axis=0, keepdims=True)
    scores = svd.judges @ model_vectors.T
    prediction = scores[rows[:, 0], rows[:, 1]] - scores[rows[:, 0], rows[:, 2]]
    loss = float(np.mean(np.square(prediction - targets)))
    parameters = np.concatenate((svd.judges.ravel(), model_vectors.ravel()))
    trace = ({"iteration": 0, "loss": loss, "gradient_l2": 0.0},)
    return MarginFit(
        parameters=parameters,
        row_vectors=svd.judges,
        model_vectors=model_vectors,
        scores=scores,
        loss=loss,
        iterations=0,
        gradient_l2=0.0,
        loss_trace=trace,
        start_diagnostics=(
            {
                "start": 0,
                "loss": loss,
                "iterations": 0,
                "gradient_l2": 0.0,
                "trace": list(trace),
                "solver": "truncated_svd",
                "score_correlation_to_best": 1.0,
            },
        ),
    )


def _btd_initials(
    direct: MarginFit,
    calibration: np.ndarray,
    *,
    starts: int,
    seed: int,
) -> list[np.ndarray]:
    root_scale = float(np.sqrt(np.exp(calibration[0])))
    warm = np.concatenate(
        (
            (root_scale * direct.row_vectors).ravel(),
            (root_scale * direct.model_vectors).ravel(),
            calibration[1:],
        )
    )
    initials = [warm]
    rng = np.random.default_rng(seed)
    factor_count = direct.row_vectors.size + direct.model_vectors.size
    initials.extend(
        np.concatenate(
            (rng.normal(0.0, 0.1, factor_count), np.zeros(len(calibration) - 1))
        )
        for _ in range(starts - 1)
    )
    return initials


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
    probabilities = _softmax_logits(logits)

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


def _mean_se(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows])
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def _fit_models(
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
            initials=_factor_initials(
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
        initials=_btd_initials(
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
            all_positions, test_positions, assume_unique=True
        )
        train_ratings, test_ratings, _ = standardize_ratings(
            matched[:, train_positions, :], matched[:, test_positions, :]
        )
        train_rows, train_targets = rows_and_targets(
            data.blocks, data.scenario_ids[train_positions], train_ratings
        )
        test_rows, test_targets = rows_and_targets(
            data.blocks, data.scenario_ids[test_positions], test_ratings
        )
        direct, calibration, btd = _fit_models(
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
        direct_logits = _fixed_score_logits(calibration, direct.scores, test_rows)
        btd_logits = criterion_btd_logits(
            btd.parameters,
            test_rows,
            num_rows=train_ratings.shape[0],
            num_models=train_ratings.shape[2],
            dimension=dimension,
        )
        direct_classification = calibration_metrics(
            direct_logits, test_rows[:, 3]
        )
        btd_classification = calibration_metrics(btd_logits, test_rows[:, 3])
        choice_parts.append(test_rows[:, 3])
        direct_probability_parts.append(_softmax_logits(direct_logits))
        btd_probability_parts.append(_softmax_logits(btd_logits))
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
    """Bin held-out tie and strict-win probabilities for visual checks."""
    output: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model_name, probabilities in (
        ("direct", direct_probabilities),
        ("btd", btd_probabilities),
    ):
        series = [
            ("tie", probabilities[:, 0], choices == 0),
        ]
        strict = choices != 0
        strict_win = probabilities[strict, 1] / probabilities[strict, 1:].sum(
            axis=1
        )
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
    data: MatchedCriterionData, *, folds: int, seed: int
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
            all_positions, test_positions, assume_unique=True
        )
        train_ratings, test_ratings, _ = standardize_ratings(
            matched[:, train_positions, :], matched[:, test_positions, :]
        )
        test_rows, test_targets = rows_and_targets(
            data.blocks, data.scenario_ids[test_positions], test_ratings
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


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    data = load_matched_criterion_data(args.ratings, args.evaluations)
    if len(data.scenario_ids) != 871:
        raise ValueError(f"Expected 871 matched scenarios, found {len(data.scenario_ids)}")
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
    direct, calibration, btd = _fit_models(
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
        heldout_choices, direct_probabilities, btd_probabilities
    )
    rank_rows, rank_summary = svd_rank_curve(
        data, folds=args.folds, seed=args.seed + 1
    )
    geometry = geometry_metrics(direct.scores, btd.scores)
    aligned, _, _, _, _ = aligned_embedding_metrics(
        direct.scores, btd.scores, args.dimension
    )
    results: dict[str, object] = {
        "data": {
            "ratings_path": str(args.ratings),
            "ratings_sha256": _sha256_file(args.ratings),
            "evaluations_path": str(args.evaluations),
            "evaluations_sha256": _sha256_file(args.evaluations),
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
                "Judge slot 3 is Grok 4.3 for criterion ratings and Grok 4 for pairwise trits."
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
            "direct_min_near_score_correlation": (
                direct.min_near_score_correlation
            ),
            "btd_min_near_score_correlation": btd.min_near_score_correlation,
            "direct_starts": list(direct.start_diagnostics),
            "btd_starts": list(btd.start_diagnostics),
            "direct_failed_starts": list(direct.failed_starts),
            "btd_failed_starts": list(btd.failed_starts),
            "criterion_judge_scales": scales.tolist(),
        },
        "geometry": {**geometry, **aligned},
        "heldout": {
            "direct_mse": _mean_se(folds, "direct_test_mse"),
            "direct_mse_reduction_vs_zero": _mean_se(
                folds, "direct_test_mse_reduction_vs_zero"
            ),
            "direct_nll": _mean_se(folds, "direct_nll"),
            "btd_nll": _mean_se(folds, "btd_nll"),
            "nll_gap_direct_minus_btd": _mean_se(
                folds, "nll_gap_direct_minus_btd"
            ),
            "direct_accuracy": _mean_se(folds, "direct_accuracy"),
            "btd_accuracy": _mean_se(folds, "btd_accuracy"),
            "direct_tie_ece": _mean_se(folds, "direct_tie_ece"),
            "btd_tie_ece": _mean_se(folds, "btd_tie_ece"),
            "direct_win_loss_ece": _mean_se(folds, "direct_win_loss_ece"),
            "btd_win_loss_ece": _mean_se(folds, "btd_win_loss_ece"),
            "score_pearson": _mean_se(folds, "score_pearson"),
            "margin_pearson": _mean_se(folds, "margin_pearson"),
            "margin_sign_agreement": _mean_se(
                folds, "margin_sign_agreement"
            ),
        },
        "svd_rank_curve": rank_summary,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "folds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
        writer.writeheader()
        writer.writerows(folds)
    with (args.output / "reliability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reliability[0]))
        writer.writeheader()
        writer.writerows(reliability)
    with (args.output / "svd_rank_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)
    optimization_rows = []
    for model_name, diagnostics in (
        ("direct", direct.start_diagnostics),
        ("btd", btd.start_diagnostics),
    ):
        for start in diagnostics:
            for point in start["trace"]:
                optimization_rows.append(
                    {
                        "model": model_name,
                        "start": start["start"],
                        **point,
                    }
                )
    with (args.output / "optimization_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(optimization_rows[0]))
        writer.writeheader()
        writer.writerows(optimization_rows)
    with (args.output / "score_matrices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "criterion_id",
            "judge_id",
            "judge_name",
            "model_id",
            "model_name",
            "direct_score",
            "btd_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(direct.scores.shape[0]):
            criterion = row_index // len(data.judge_names)
            judge = row_index % len(data.judge_names)
            for model in range(direct.scores.shape[1]):
                writer.writerow(
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
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument(
        "--direct-method",
        choices=("svd", "matched_margin"),
        default="svd",
    )
    return parser.parse_args()


def main() -> None:
    results = run_analysis(parse_args())
    heldout = results["heldout"]
    geometry = results["geometry"]
    print(
        "Matched criterion comparison: "
        f"direct NLL={heldout['direct_nll']['mean']:.6f}, "
        f"BTD NLL={heldout['btd_nll']['mean']:.6f}, "
        f"score r={geometry['score_pearson']:.6f}"
    )


if __name__ == "__main__":
    main()
