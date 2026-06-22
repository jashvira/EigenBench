"""Build the AIRisk retrieval comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scenario_classifier.audits.airisk_embedding_retrieval import DEFAULT_ANNOTATIONS
from scenario_classifier.audits.embedding_audit import slugify_model
from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, OUTPUT_ROOT, load_jsonl
from scenario_classifier.core.ranking_metrics import recall_at_k
from scenario_classifier.retrieval.retrievers import BM25Retriever


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "embedding_audit"
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "airisk_623_retrieval_comparison.json"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "airisk_623_retrieval_comparison.csv"
DEFAULT_MODELS = [
    ("google/gemini-embedding-2", "Gemini"),
    ("qwen/qwen3-embedding-8b", "Qwen 8B"),
]
KS = [1, 2, 3, 5]


def labelled_high_conf(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep high-confidence rows with an assigned constitution."""
    return [
        row
        for row in rows
        if row["annotation_confidence"] == "high" and row["annotation_primary"] != "none"
    ]


def percent(value: float) -> float:
    """Convert a 0-1 recall value to a one-decimal percentage."""
    return round(value * 100, 1)


def table_row(system: str, rows: list[dict[str, Any]], ranking_key: str) -> dict[str, Any]:
    """Compute one display row."""
    recall = recall_at_k(rows, ranking_key, KS, hit_mode="unit")
    return {
        "system": system,
        **{f"top_{k}": percent(recall[f"recall@{k}"]) for k in KS},
    }


def load_embedding_rows(model: str, output_dir: Path) -> list[dict[str, Any]]:
    """Load cached AIRisk retrieval rows for one embedding model."""
    path = output_dir / slugify_model(model) / "airisk_623_embedding_retrieval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing retrieval rows: {path}")
    return load_jsonl(path)


def bm25_rows(annotations: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    """Run BM25 and return rows shaped like embedding retrieval output."""
    card_retriever = BM25Retriever(load_jsonl(DEFAULT_CARD_CORPUS))
    raw_retriever = BM25Retriever(load_jsonl(DEFAULT_RAW_CORPUS))
    rows: list[dict[str, Any]] = []
    for row in annotations:
        rows.append(
            {
                "scenario_index": row["scenario_index"],
                "scenario": row["scenario"],
                "annotation_primary": row["primary_constitution"],
                "annotation_confidence": row["confidence"],
                "card_top": card_retriever.retrieve(row["scenario"], top_k),
                "raw_top": raw_retriever.retrieve(row["scenario"], top_k),
            }
        )
    return rows


def build_table(args: argparse.Namespace) -> dict[str, Any]:
    """Build the retrieval comparison payload."""
    annotations = load_jsonl(args.annotations)
    table_rows: list[dict[str, Any]] = []

    embedding_rows = {
        label: labelled_high_conf(load_embedding_rows(model, args.output_dir))
        for model, label in DEFAULT_MODELS
    }
    for label in ["Gemini", "Qwen 8B"]:
        table_rows.append(table_row(f"{label} cards", embedding_rows[label], "card_top"))
    for label in ["Gemini", "Qwen 8B"]:
        table_rows.append(table_row(f"{label} raw criteria", embedding_rows[label], "raw_top"))

    bm25 = labelled_high_conf(bm25_rows(annotations, top_k=max(KS)))
    table_rows.append(table_row("BM25 cards", bm25, "card_top"))
    table_rows.append(table_row("BM25 raw criteria", bm25, "raw_top"))

    return {
        "metric": "unit_recall",
        "metric_definition": "Rank cards/criteria; recall@k is a hit if any of the first k retrieved units maps to the assigned constitution.",
        "ks": KS,
        "scenario_count": len(annotations),
        "high_conf_labelled_count": len(embedding_rows["Gemini"]),
        "rows": table_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the comparison table as CSV."""
    fields = ["system", "top_1", "top_2", "top_3", "top_5"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    payload = build_table(args)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.csv, payload["rows"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
