"""Merge high-confidence constitution-match audit batches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import OUTPUT_ROOT, REPO_ROOT


DEFAULT_AUDIT_DIR = OUTPUT_ROOT / "constitution_matches" / "high_confidence_audit"
ALLOWED_VERDICTS = {"agree", "downgrade", "relabel", "remove"}
ALLOWED_CONSTITUTIONS = {
    "kindness",
    "conservatism",
    "deep_ecology",
    "none",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL objects from a file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def validate_audit_row(row: dict[str, Any], path: Path) -> None:
    """Validate one audit row."""
    if not isinstance(row.get("scenario_index"), int):
        raise ValueError(f"{path}: missing integer scenario_index")
    if row.get("verdict") not in ALLOWED_VERDICTS:
        raise ValueError(f"{path}: bad verdict {row.get('verdict')!r} for {row.get('scenario_index')}")
    if row.get("suggested_primary") not in ALLOWED_CONSTITUTIONS:
        raise ValueError(f"{path}: bad suggested_primary {row.get('suggested_primary')!r} for {row.get('scenario_index')}")
    if row.get("suggested_confidence") not in ALLOWED_CONFIDENCE:
        raise ValueError(f"{path}: bad suggested_confidence {row.get('suggested_confidence')!r} for {row.get('scenario_index')}")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError(f"{path}: missing reason for {row.get('scenario_index')}")


def normalize_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless schema slips in audit rows."""
    if row.get("verdict") == "remove":
        normalized = dict(row)
        if normalized.get("suggested_primary") is None:
            normalized["suggested_primary"] = "none"
        if normalized.get("suggested_confidence") is None:
            normalized["suggested_confidence"] = "low"
        return normalized
    return row


def flatten_row(source: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Combine source annotation and audit verdict for CSV output."""
    return {
        "scenario_index": source["scenario_index"],
        "current_primary": source["primary_constitution"],
        "current_confidence": source["confidence"],
        "verdict": audit["verdict"],
        "suggested_primary": audit["suggested_primary"],
        "suggested_confidence": audit["suggested_confidence"],
        "audit_reason": audit["reason"],
        "current_matches": json.dumps(source["matches"], ensure_ascii=False),
        "scenario": source["scenario"],
    }


def merge(audit_dir: Path) -> dict[str, Any]:
    """Merge all audit batches and write JSONL/CSV/summary artifacts."""
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    merged: list[dict[str, Any]] = []
    for batch in manifest["batches"]:
        input_path = REPO_ROOT / batch["input"]
        output_path = REPO_ROOT / batch["output"]
        input_rows = read_jsonl(input_path)
        audit_rows = read_jsonl(output_path)
        input_by_index = {row["scenario_index"]: row for row in input_rows}
        audit_by_index = {row["scenario_index"]: row for row in audit_rows}
        if len(input_rows) != batch["n"]:
            raise ValueError(f"{input_path}: expected {batch['n']} input rows, found {len(input_rows)}")
        if len(audit_rows) != batch["n"]:
            raise ValueError(f"{output_path}: expected {batch['n']} audit rows, found {len(audit_rows)}")
        if set(input_by_index) != set(audit_by_index):
            raise ValueError(f"{output_path}: scenario_index set does not match input")
        for raw_row in audit_rows:
            row = normalize_audit_row(raw_row)
            validate_audit_row(row, output_path)
            merged.append({"source": input_by_index[row["scenario_index"]], "audit": row})

    merged = sorted(merged, key=lambda row: row["source"]["scenario_index"])
    flat_rows = [flatten_row(row["source"], row["audit"]) for row in merged]
    verdict_counts = Counter(row["audit"]["verdict"] for row in merged)
    suggested_primary_counts = Counter(row["audit"]["suggested_primary"] for row in merged)
    current_primary_counts = Counter(row["source"]["primary_constitution"] for row in merged)
    by_current_verdict = Counter((row["source"]["primary_constitution"], row["audit"]["verdict"]) for row in merged)

    summary = {
        "audited_high_confidence_rows": len(merged),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "current_primary_counts": dict(sorted(current_primary_counts.items())),
        "suggested_primary_counts": dict(sorted(suggested_primary_counts.items())),
        "by_current_primary_and_verdict": {
            f"{primary}:{verdict}": count for (primary, verdict), count in sorted(by_current_verdict.items())
        },
        "flagged_rows": [row for row in flat_rows if row["verdict"] != "agree"],
    }

    (audit_dir / "merged_high_confidence_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    with (audit_dir / "merged_high_confidence_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the merge."""
    summary = merge(parse_args().audit_dir)
    printable = {key: value for key, value in summary.items() if key != "flagged_rows"}
    printable["flagged_row_count"] = len(summary["flagged_rows"])
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
