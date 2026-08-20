"""Criterion-wise numerical-rating data loading."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_criterion_tensor(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[str], list[str]]:
    """Load a complete criterion-judge-scenario-model rating tensor."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("Criterion ratings file is empty")

    criterion_ids = sorted({row["dimension_id"] for row in rows})
    judge_ids = sorted({int(row["judge_id"]) for row in rows})
    scenario_ids = sorted({int(row["scenario_index"]) for row in rows})
    model_ids = sorted({int(row["model_id"]) for row in rows})
    if judge_ids != list(range(len(judge_ids))):
        raise ValueError("Criterion ratings require contiguous judge IDs")
    if model_ids != list(range(len(model_ids))):
        raise ValueError("Criterion ratings require contiguous model IDs")

    expected_rows = (
        len(criterion_ids) * len(judge_ids) * len(scenario_ids) * len(model_ids)
    )
    if len(rows) != expected_rows:
        raise ValueError("Criterion ratings do not form a complete rectangle")

    criterion_position = {
        criterion_id: position for position, criterion_id in enumerate(criterion_ids)
    }
    scenario_position = {
        scenario_id: position for position, scenario_id in enumerate(scenario_ids)
    }
    tensor = np.full(
        (len(criterion_ids), len(judge_ids), len(scenario_ids), len(model_ids)),
        np.nan,
    )
    criterion_hashes: list[str | None] = [None] * len(criterion_ids)
    judge_names: list[str | None] = [None] * len(judge_ids)
    model_names: list[str | None] = [None] * len(model_ids)
    for row in rows:
        criterion = criterion_position[row["dimension_id"]]
        judge = int(row["judge_id"])
        scenario = scenario_position[int(row["scenario_index"])]
        model = int(row["model_id"])
        if np.isfinite(tensor[criterion, judge, scenario, model]):
            raise ValueError("Criterion ratings contain a duplicate cell")
        tensor[criterion, judge, scenario, model] = float(row["score"])

        criterion_hash = row["dimension_hash"]
        if criterion_hashes[criterion] not in (None, criterion_hash):
            raise ValueError("Criterion text hash changed within the ratings file")
        criterion_hashes[criterion] = criterion_hash
        if judge_names[judge] not in (None, row["judge_name"]):
            raise ValueError("Judge name changed within the ratings file")
        judge_names[judge] = row["judge_name"]
        if model_names[model] not in (None, row["model_name"]):
            raise ValueError("Model name changed within the ratings file")
        model_names[model] = row["model_name"]

    if not np.isfinite(tensor).all():
        raise ValueError("Criterion ratings contain a missing or non-finite cell")
    return (
        tensor,
        criterion_ids,
        [str(value) for value in criterion_hashes],
        scenario_ids,
        [str(value) for value in judge_names],
        [str(value) for value in model_names],
    )
