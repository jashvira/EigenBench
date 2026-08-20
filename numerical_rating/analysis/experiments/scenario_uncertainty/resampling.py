"""Scenario-level bootstrap and delete-group jackknife routines."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from numerical_rating.analysis.aggregation.eigentrust import compute_trust
from numerical_rating.analysis.data_loading.pairwise_judgments import concatenate_blocks
from numerical_rating.analysis.model_fitting.pairwise_btd import (
    PairwiseFit,
    fit_diagnostics,
    fit_pairwise_btd_multistart,
    multistart_initials,
)
from numerical_rating.analysis.utilities.artifact_io import atomic_json, atomic_npz


@dataclass(frozen=True)
class BootstrapResult:
    sampled_positions: np.ndarray
    numerical_draws: np.ndarray
    pairwise_draws: np.ndarray
    diagnostics: list[dict]
    unstable_draws: list[int]


@dataclass(frozen=True)
class JackknifeResult:
    group_positions: list[np.ndarray]
    numerical_leave_out: np.ndarray
    pairwise_leave_out: np.ndarray
    numerical_interval: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    pairwise_interval: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    difference_interval: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    diagnostics: list[dict]
    leave_out_rows: list[dict]
    group_rows: list[dict]


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
        expansion[:, None] * point - (expansion[:, None] - 1.0) * leave_group_out
    )
    estimate = np.sum(pseudovalues / expansion[:, None], axis=0)
    variance = np.mean(
        np.square(pseudovalues - estimate) / (expansion[:, None] - 1.0),
        axis=0,
    )
    standard_error = np.sqrt(variance)
    return (
        estimate,
        standard_error,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )


def ranking_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compare two trust vectors without assuming equal scales."""
    return {
        "l1": float(np.abs(left - right).sum()),
        "spearman": float(spearmanr(left, right).statistic),
        "kendall": float(kendalltau(left, right).statistic),
    }


def save_bootstrap_checkpoint(
    output_dir: Path,
    *,
    numerical_draws: np.ndarray,
    pairwise_draws: np.ndarray,
    completed: np.ndarray,
    diagnostics: list[dict | None],
    seed: int,
    input_fingerprint: str,
) -> None:
    """Persist completed bootstrap draws for exact resumption."""
    atomic_npz(
        output_dir / "bootstrap_checkpoint.npz",
        numerical_draws=numerical_draws,
        pairwise_draws=pairwise_draws,
        completed=completed,
        seed=np.asarray(seed),
        input_fingerprint=np.asarray(input_fingerprint),
    )
    atomic_json(output_dir / "bootstrap_checkpoint_diagnostics.json", diagnostics)


def load_bootstrap_checkpoint(
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    input_fingerprint: str,
    num_models: int = 8,
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
    expected_shape = (bootstrap_samples, num_models)
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


def paired_scenario_bootstrap(
    *,
    matched_ratings: np.ndarray,
    matched_scenarios: np.ndarray,
    blocks: dict[int, np.ndarray],
    pairwise_point: PairwiseFit,
    output_dir: Path,
    bootstrap_samples: int,
    workers: int,
    seed: int,
    input_fingerprint: str,
) -> BootstrapResult:
    """Resample identical scenario draws for direct and pairwise methods."""
    rng = np.random.default_rng(seed)
    sampled_positions = rng.integers(
        0,
        len(matched_scenarios),
        size=(bootstrap_samples, len(matched_scenarios)),
    )
    checkpoint = load_bootstrap_checkpoint(
        output_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        input_fingerprint=input_fingerprint,
        num_models=matched_ratings.shape[2],
    )
    if checkpoint is None:
        shape = (bootstrap_samples, matched_ratings.shape[2])
        numerical_draws = np.full(shape, np.nan, dtype=float)
        pairwise_draws = np.full(shape, np.nan, dtype=float)
        completed_draws = np.zeros(bootstrap_samples, dtype=bool)
        diagnostics: list[dict | None] = [None] * bootstrap_samples
    else:
        numerical_draws, pairwise_draws, completed_draws, diagnostics = checkpoint
        print(
            f"Resuming {int(completed_draws.sum())}/{bootstrap_samples} "
            "bootstrap draws",
            flush=True,
        )

    def run_draw(
        draw_index: int,
        positions: np.ndarray,
    ) -> tuple[int, np.ndarray, PairwiseFit]:
        numerical = compute_trust(matched_ratings[:, positions, :]).trust
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
            run_draw,
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
            diagnostics[draw_index] = {
                "draw": draw_index,
                **fit_diagnostics(fit),
            }
            completed += 1
            if completed % 50 == 0 or completed == bootstrap_samples:
                save_bootstrap_checkpoint(
                    output_dir,
                    numerical_draws=numerical_draws,
                    pairwise_draws=pairwise_draws,
                    completed=completed_draws,
                    diagnostics=diagnostics,
                    seed=seed,
                    input_fingerprint=input_fingerprint,
                )
                print(f"Bootstrap {completed}/{bootstrap_samples}", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        save_bootstrap_checkpoint(
            output_dir,
            numerical_draws=numerical_draws,
            pairwise_draws=pairwise_draws,
            completed=completed_draws,
            diagnostics=diagnostics,
            seed=seed,
            input_fingerprint=input_fingerprint,
        )
        raise
    else:
        executor.shutdown(wait=True)

    if not completed_draws.all():
        raise RuntimeError("Bootstrap ended with incomplete draws")
    if any(diagnostic is None for diagnostic in diagnostics):
        raise RuntimeError("Missing pairwise bootstrap fit diagnostics")
    complete_diagnostics = [
        diagnostic for diagnostic in diagnostics if diagnostic is not None
    ]
    unstable_draws = [
        int(diagnostic["draw"])
        for diagnostic in complete_diagnostics
        if not diagnostic["stable_near_optima"]
    ]
    return BootstrapResult(
        sampled_positions=sampled_positions,
        numerical_draws=numerical_draws,
        pairwise_draws=pairwise_draws,
        diagnostics=complete_diagnostics,
        unstable_draws=unstable_draws,
    )


def delete_group_jackknife(
    *,
    matched_ratings: np.ndarray,
    matched_scenarios: np.ndarray,
    blocks: dict[int, np.ndarray],
    model_names: list[str],
    numerical_point: np.ndarray,
    pairwise_point: PairwiseFit,
    groups: int,
    seed: int,
) -> JackknifeResult:
    """Fit complements of a fixed random partition of scenarios."""
    rng = np.random.default_rng(seed + 1)
    shuffled = rng.permutation(len(matched_scenarios))
    group_positions = list(np.array_split(shuffled, groups))
    leave_out_rows: list[dict] = []
    group_rows: list[dict] = []
    diagnostics: list[dict] = []
    numerical_leave_out = np.empty((groups, len(model_names)), dtype=float)
    pairwise_leave_out = np.empty((groups, len(model_names)), dtype=float)
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
        diagnostics.append(
            {
                "group": group_index,
                "omitted_scenarios": len(positions),
                "retained_scenarios": int(keep.sum()),
                **fit_diagnostics(leave_out_fit),
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
    numerical_interval = jackknife_interval(
        numerical_point,
        numerical_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    pairwise_interval = jackknife_interval(
        pairwise_point.trust,
        pairwise_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    difference_interval = jackknife_interval(
        numerical_point - pairwise_point.trust,
        numerical_leave_out - pairwise_leave_out,
        omitted_sizes=omitted_sizes,
        total_size=len(matched_scenarios),
    )
    return JackknifeResult(
        group_positions=group_positions,
        numerical_leave_out=numerical_leave_out,
        pairwise_leave_out=pairwise_leave_out,
        numerical_interval=numerical_interval,
        pairwise_interval=pairwise_interval,
        difference_interval=difference_interval,
        diagnostics=diagnostics,
        leave_out_rows=leave_out_rows,
        group_rows=group_rows,
    )
