"""Load validated numerical-rating tensors."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_numerical_tensor(path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load a complete judge-by-scenario-by-model rating tensor."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    judge_ids = sorted({int(row["judge_id"]) for row in rows})
    scenario_ids = sorted({int(row["scenario_index"]) for row in rows})
    model_ids = sorted({int(row["model_id"]) for row in rows})
    if judge_ids != list(range(len(judge_ids))):
        raise ValueError("Judge IDs must be contiguous from zero")
    if model_ids != list(range(len(model_ids))):
        raise ValueError("Model IDs must be contiguous from zero")
    if len(judge_ids) != len(model_ids):
        raise ValueError("EigenTrust requires matching judge and model counts")

    scenario_position = {
        scenario: position for position, scenario in enumerate(scenario_ids)
    }
    tensor = np.full(
        (len(judge_ids), len(scenario_ids), len(model_ids)), np.nan, dtype=float
    )
    model_names: list[str | None] = [None] * len(model_ids)
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
