"""Shared paths and loaders for scenario classifier audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "outputs"
SCENARIOS_ROOT = DATA_ROOT / "scenarios"
CORPORA_ROOT = DATA_ROOT / "corpora"
LABELS_ROOT = DATA_ROOT / "labels"
VALUEARENA_ROOT = DATA_ROOT / "valuearena"

DEFAULT_SCENARIOS = SCENARIOS_ROOT / "oasst_questions.json"
DEFAULT_AIRISK_SCENARIOS = SCENARIOS_ROOT / "airiskdilemmas.json"
DEFAULT_CARD_CORPUS = CORPORA_ROOT / "criteria_cards" / "criteria_cards_local_anchors_v0_4.jsonl"
DEFAULT_RAW_CORPUS = CORPORA_ROOT / "criteria_cards" / "raw_criteria_embedding_units_local_anchors_v0_1.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scenarios(path: Path) -> list[str]:
    """Load a scenario JSON list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return payload
