"""Aggregate Inspect logs from pointwise constitution-rating runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCORER_NAME = "whole_constitution_score"


def score_records(log_path: Path) -> list[dict[str, object]]:
    """Extract parsed whole-constitution scores from one Inspect JSON log."""
    data = json.loads(log_path.read_text(encoding="utf-8"))
    records = []

    for sample in data.get("samples", []):
        score = sample.get("scores", {}).get(SCORER_NAME, {})
        metadata = {
            **(sample.get("metadata") or {}),
            **(score.get("metadata") or {}),
        }
        if not isinstance(metadata, dict):
            continue
        parsed = metadata.get("parsed")
        # Only valid judge JSON enters the CSV; parse errors remain in the log.
        if not isinstance(parsed, dict) or "score" not in parsed:
            continue
        records.append(
            {
                "scenario_index": metadata["scenario_index"],
                "scenario": metadata.get("scenario", ""),
                "model_id": metadata["model_id"],
                "model_name": metadata.get("model_name", ""),
                "model_api_id": metadata.get("model_api_id", ""),
                "response_hash": metadata.get("response_hash", ""),
                "score": float(parsed["score"]),
                "rationale": parsed.get("rationale", ""),
                "constitution": metadata.get("constitution", ""),
                "constitution_hash": metadata.get("constitution_hash", ""),
                "prompt_hash": metadata.get("prompt_hash", ""),
                "source": metadata.get("source", ""),
            }
        )
    return sorted(records, key=lambda row: (int(row["scenario_index"]), int(row["model_id"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario_index",
        "model_id",
        "model_name",
        "model_api_id",
        "score",
        "rationale",
        "response_hash",
        "constitution",
        "constitution_hash",
        "prompt_hash",
        "source",
        "scenario",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def model_means(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return mean score by model."""
    scores: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        scores[(int(row["model_id"]), str(row["model_name"]))].append(float(row["score"]))
    out = []
    for (model_id, model_name), values in sorted(scores.items()):
        out.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "n": len(values),
                "mean_score": sum(values) / len(values),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate pointwise whole-constitution rating logs."
    )
    parser.add_argument("log", type=Path, help="Inspect JSON log file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/output/numerical_rating/pointwise_scores.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/output/numerical_rating/model_means.json"),
    )
    args = parser.parse_args()

    rows = score_records(args.log)
    write_csv(args.out, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "log": str(args.log),
                "rows": len(rows),
                "model_means": model_means(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote model means to {args.summary}")


if __name__ == "__main__":
    main()
