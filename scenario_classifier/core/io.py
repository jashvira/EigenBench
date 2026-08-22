"""Shared paths and loaders for scenario classifier audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "outputs"
CORPORA_ROOT = DATA_ROOT / "corpora"

DEFAULT_CARD_CORPUS = CORPORA_ROOT / "retrieval_units" / "criteria_cards_local_anchors_v0_4.jsonl"
DEFAULT_RAW_CORPUS = CORPORA_ROOT / "retrieval_units" / "raw_criteria_embedding_units_local_anchors_v0_2.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(
    path: Path,
    *,
    scenario_field: str = "scenario",
    index_field: str = "scenario_index",
    label_field: str = "primary_constitution",
    confidence_field: str = "confidence",
) -> list[dict[str, Any]]:
    """Load either a JSON list of scenario strings or JSON/JSONL records."""
    text = path.read_text(encoding="utf-8").strip()
    payload: Any
    if text.startswith("["):
        payload = json.loads(text)
    else:
        payload = load_jsonl(path)

    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL rows")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if isinstance(item, str):
            rows.append({"scenario_index": idx, "scenario": item})
            continue
        if not isinstance(item, dict) or not isinstance(item.get(scenario_field), str):
            raise ValueError(f"{path} row {idx} must be a scenario string or object with {scenario_field!r}")

        row = {
            "scenario_index": item.get(index_field, idx),
            "scenario": item[scenario_field],
        }
        label = item.get(label_field, item.get("annotation_primary"))
        confidence = item.get(confidence_field, item.get("annotation_confidence"))
        if label is not None:
            row["annotation_primary"] = label
        if confidence is not None:
            row["annotation_confidence"] = confidence
        rows.append(row)

    return rows
