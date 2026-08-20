"""Load and reconcile EigenBench pairwise judgment records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pipeline.utils.comparisons import (
    extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria,
)

PAIRWISE_CLEANING = "corrected"


@dataclass(frozen=True)
class PairwiseData:
    """Scenario comparison blocks and cleaning diagnostics."""

    blocks: dict[int, np.ndarray]
    diagnostics: dict[str, int | str]


def evaluation_key(evaluation: dict) -> str:
    """Serialize an evaluation deterministically for exact deduplication."""
    serialized = json.dumps(
        evaluation,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _content_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _collection_pass_key(evaluation: dict) -> tuple:
    """Identify reversed judgments over the same responses and reflections."""
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


def reconcile_transpose_pair(left: list[int], right: list[int]) -> list[list[int]]:
    """Apply the published tie rule to one forward/reverse judgment pair."""
    left_choice = left[-1]
    right_choice = right[-1]
    if left_choice == 0 or right_choice == 0 or left_choice != right_choice:
        return [left, right]
    return [left[:-1] + [0], right[:-1] + [0]]


def reconcile_pass_comparisons(comparisons: list[list[int]]) -> list[list[int]]:
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
        cleaned.extend(reconcile_transpose_pair(rows[0], rows[1]))
    return cleaned


def extract_corrected_comparisons(
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
        comparisons.extend(reconcile_pass_comparisons(pass_comparisons))
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
            key = evaluation_key(evaluation)
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
            evaluations,
            num_criteria=num_criteria,
        )
    else:
        comparisons, pass_diagnostics = extract_corrected_comparisons(
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
        scenario: np.asarray(rows, dtype=np.int64) for scenario, rows in blocks.items()
    }
    return PairwiseData(
        blocks=block_arrays,
        diagnostics={
            "cleaning": cleaning,
            "raw_evaluations": raw_evaluations,
            "exact_duplicates_removed": duplicates_removed,
            "evaluations_after_deduplication": len(evaluations),
            "extracted_criterion_rows": extracted_rows,
            "retained_criterion_rows": sum(len(rows) for rows in block_arrays.values()),
            "repeated_unordered_groups": repeated_groups,
            **pass_diagnostics,
        },
    )


def load_pairwise_blocks(
    evaluations_path: Path,
    *,
    num_criteria: int,
    cleaning: Literal["published_legacy", "corrected"] = PAIRWISE_CLEANING,
) -> dict[int, np.ndarray]:
    """Compatibility wrapper returning only scenario blocks."""
    return load_pairwise_data(
        evaluations_path,
        num_criteria=num_criteria,
        cleaning=cleaning,
    ).blocks


def concatenate_blocks(
    blocks: dict[int, np.ndarray], scenario_ids: np.ndarray
) -> np.ndarray:
    """Expand sampled scenario IDs into their complete comparison blocks."""
    return np.concatenate([blocks[int(scenario)] for scenario in scenario_ids])
