"""Merge subagent AIRisk constitution-match annotations.

The subagents write one JSONL file per batch. This script validates that every
batch has one annotation per input scenario, checks the small label schema, and
writes combined JSONL/CSV/summary outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import LABELS_ROOT, REPO_ROOT


DEFAULT_OUTPUT_DIR = LABELS_ROOT / "constitution_matches"
DEFAULT_BATCH_DIR = DEFAULT_OUTPUT_DIR / "subagent_batches"
DEFAULT_OVERRIDES = DEFAULT_OUTPUT_DIR / "human_review_overrides.jsonl"

ALLOWED_STATUSES = {"live", "weak"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
CONSTITUTION_LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def valid_constitution_label(value: Any) -> bool:
    """Return whether a local annotation label has the expected slug shape."""
    return isinstance(value, str) and bool(CONSTITUTION_LABEL_RE.fullmatch(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(value)
    return rows


def validate_row(row: dict[str, Any], *, path: Path) -> None:
    """Validate one annotation row."""
    if not isinstance(row.get("scenario_index"), int):
        raise ValueError(f"{path}: missing integer scenario_index in {row!r}")
    matches = row.get("matches")
    if not isinstance(matches, list):
        raise ValueError(f"{path}: matches must be a list for scenario {row.get('scenario_index')}")
    seen: set[str] = set()
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError(f"{path}: match must be an object for scenario {row['scenario_index']}")
        constitution = match.get("constitution")
        status = match.get("status")
        if not valid_constitution_label(constitution):
            raise ValueError(f"{path}: bad constitution {constitution!r} for scenario {row['scenario_index']}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{path}: bad status {status!r} for scenario {row['scenario_index']}")
        if constitution in seen:
            raise ValueError(f"{path}: duplicate constitution {constitution!r} for scenario {row['scenario_index']}")
        seen.add(constitution)
    primary = row.get("primary_constitution")
    if primary != "none" and not valid_constitution_label(primary):
        raise ValueError(f"{path}: bad primary_constitution {primary!r} for scenario {row['scenario_index']}")
    if primary == "none" and any(match["status"] == "live" for match in matches):
        raise ValueError(f"{path}: primary_constitution is none despite live match for scenario {row['scenario_index']}")
    if primary != "none" and primary not in {match["constitution"] for match in matches}:
        raise ValueError(f"{path}: primary_constitution is not in matches for scenario {row['scenario_index']}")
    if primary != "none" and any(match["constitution"] == primary and match["status"] != "live" for match in matches):
        if any(match["status"] == "live" for match in matches):
            raise ValueError(f"{path}: primary_constitution is weak despite live match for scenario {row['scenario_index']}")
    if row.get("confidence") not in ALLOWED_CONFIDENCE:
        raise ValueError(f"{path}: bad confidence {row.get('confidence')!r} for scenario {row['scenario_index']}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless annotation inconsistencies."""
    if row["matches"] and all(match["status"] == "weak" for match in row["matches"]):
        return {**row, "primary_constitution": "none"}
    return row


def flatten_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten match lists into compact CSV fields."""
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        matches = row["matches"]
        kindness_valuearena = row.get("kindness_valuearena", {})
        flat_rows.append(
            {
                "scenario_index": row["scenario_index"],
                "scenario": row.get("scenario", ""),
                "primary_constitution": row["primary_constitution"],
                "confidence": row["confidence"],
                "match_constitutions": ";".join(match["constitution"] for match in matches),
                "match_statuses": ";".join(match["status"] for match in matches),
                "match_reasons": " | ".join(f"{match['constitution']}: {match.get('reason', '')}" for match in matches),
                "review_override": row.get("review_override", False),
                "review_note": row.get("review_note", ""),
                "valuearena_total_choices": kindness_valuearena.get("total_choices", ""),
                "valuearena_mean_tie_rate": kindness_valuearena.get("mean_tie_rate", ""),
                "valuearena_strong_signal_count": kindness_valuearena.get("strong_signal_count", ""),
                "valuearena_strong_signal_criteria": kindness_valuearena.get("strong_signal_criteria", ""),
            }
        )
    return flat_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as JSONL."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize constitution-match annotations."""
    primary_counts = Counter(row["primary_constitution"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    match_counts: Counter[str] = Counter()
    live_counts: Counter[str] = Counter()
    weak_counts: Counter[str] = Counter()
    match_count_per_scenario = Counter(len(row["matches"]) for row in rows)

    for row in rows:
        for match in row["matches"]:
            constitution = match["constitution"]
            match_counts[constitution] += 1
            if match["status"] == "live":
                live_counts[constitution] += 1
            else:
                weak_counts[constitution] += 1

    return {
        "scenario_count": len(rows),
        "review_override_count": sum(bool(row.get("review_override")) for row in rows),
        "weak_only_no_primary_count": sum(row["primary_constitution"] == "none" and bool(row["matches"]) for row in rows),
        "primary_constitution_counts": dict(sorted(primary_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "any_match_counts": dict(sorted(match_counts.items())),
        "live_match_counts": dict(sorted(live_counts.items())),
        "weak_match_counts": dict(sorted(weak_counts.items())),
        "match_count_per_scenario": dict(sorted(match_count_per_scenario.items())),
    }


def apply_overrides(rows: list[dict[str, Any]], overrides_path: Path | None) -> list[dict[str, Any]]:
    """Apply reviewer overrides by scenario index."""
    if overrides_path is None or not overrides_path.exists():
        return rows
    overrides = {row["scenario_index"]: row for row in read_jsonl(overrides_path)}
    reviewed: list[dict[str, Any]] = []
    for row in rows:
        override = overrides.get(row["scenario_index"])
        if override is None:
            reviewed.append(row)
            continue
        validate_row(override, path=overrides_path)
        reviewed.append(normalize_row({**row, **override, "review_override": True}))
    missing = set(overrides) - {row["scenario_index"] for row in rows}
    if missing:
        raise ValueError(f"{overrides_path}: overrides for unknown scenario_index values: {sorted(missing)}")
    return reviewed


def merge(batch_dir: Path, output_dir: Path, overrides_path: Path | None) -> dict[str, Any]:
    """Validate and merge all batch annotations."""
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []

    for batch in manifest["batches"]:
        input_path = REPO_ROOT / batch["input"]
        output_path = REPO_ROOT / batch["output"]
        input_rows = read_jsonl(input_path)
        input_by_index = {row["scenario_index"]: row for row in input_rows}
        expected_indices = set(input_by_index)
        actual_rows = read_jsonl(output_path)
        actual_indices = [row.get("scenario_index") for row in actual_rows]
        if len(actual_rows) != batch["n"]:
            raise ValueError(f"{output_path}: expected {batch['n']} rows, found {len(actual_rows)}")
        if set(actual_indices) != expected_indices:
            raise ValueError(f"{output_path}: scenario_index set does not match input")
        if len(actual_indices) != len(set(actual_indices)):
            raise ValueError(f"{output_path}: duplicate scenario_index in batch output")
        for row in actual_rows:
            validate_row(row, path=output_path)
            all_rows.append(normalize_row({**input_by_index[row["scenario_index"]], **row}))

    all_rows = sorted(apply_overrides(all_rows, overrides_path), key=lambda row: row["scenario_index"])
    if len(all_rows) != manifest["n_scenarios"]:
        raise ValueError(f"expected {manifest['n_scenarios']} merged rows, found {len(all_rows)}")
    if len({row["scenario_index"] for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate scenario_index in merged rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "airisk_623_constitution_matches.jsonl", all_rows)
    write_csv(output_dir / "airisk_623_constitution_matches.csv", flatten_for_csv(all_rows))
    summary = build_summary(all_rows)
    (output_dir / "airisk_623_constitution_matches_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    return parser.parse_args()


def main() -> None:
    """Run the merge."""
    args = parse_args()
    summary = merge(args.batch_dir, args.output_dir, args.overrides)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
