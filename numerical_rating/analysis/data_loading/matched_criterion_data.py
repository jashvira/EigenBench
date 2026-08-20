"""Match criterion ratings to pairwise judgments over the same responses."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from numerical_rating.analysis.data_loading.criterion_ratings import (
    load_criterion_tensor,
)
from numerical_rating.analysis.data_loading.pairwise_judgments import (
    evaluation_key,
    extract_corrected_comparisons,
)
from numerical_rating.collection.data import sha256_text


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


def response_hashes(path: Path) -> dict[tuple[int, int], str]:
    hashes: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scenario_index"]), int(row["model_id"]))
            response_hash = row["response_hash"]
            if key in hashes and hashes[key] != response_hash:
                raise ValueError(f"Response hash changed for cell {key}")
            hashes[key] = response_hash
    return hashes


def responses_match(
    evaluation: dict,
    expected_hashes: dict[tuple[int, int], str],
) -> bool:
    scenario = int(evaluation["scenario_index"])
    return all(
        expected_hashes[(scenario, int(evaluation[model_key]))]
        == sha256_text(evaluation[response_key])
        for model_key, response_key in (
            ("eval1", "eval1 response"),
            ("eval2", "eval2 response"),
        )
    )


def load_matched_criterion_data(
    ratings_path: Path,
    evaluations_path: Path,
) -> MatchedCriterionData:
    """Load criterion ratings and pairwise rows over identical responses."""
    (
        ratings,
        criterion_ids,
        criterion_hashes,
        rating_scenarios,
        judge_names,
        model_names,
    ) = load_criterion_tensor(ratings_path)
    expected_hashes = response_hashes(ratings_path)

    evaluations = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique: list[dict] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        key = evaluation_key(evaluation)
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
        if responses_match(evaluation, expected_hashes)
    ]
    comparisons, extraction = extract_corrected_comparisons(
        exact,
        num_criteria=len(criterion_ids),
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
        scenario: np.asarray(rows, dtype=np.int64) for scenario, rows in blocks.items()
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
            judge == 3 and pairwise_judge == "Grok 4" and expected_judge == "Grok 4.3"
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
