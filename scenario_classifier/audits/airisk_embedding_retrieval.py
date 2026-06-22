"""Run card and raw-criteria embedding retrieval on the 623 AIRisk scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scenario_classifier.audits.embedding_audit import DEFAULT_MODEL, load_dotenv, load_or_embed, slugify_model
from scenario_classifier.audits.embedding_audit import card_text, raw_text
from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, OUTPUT_ROOT, load_jsonl
from scenario_classifier.core.ranking_metrics import recall_at_k
from scenario_classifier.retrieval.retrievers import EmbeddingRetriever


DEFAULT_ANNOTATIONS = OUTPUT_ROOT / "constitution_matches" / "airisk_623_constitution_matches.jsonl"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "embedding_audit"


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load reviewed AIRisk scenario annotations."""
    rows = load_jsonl(path)
    for row in rows:
        if "scenario" not in row or "scenario_index" not in row:
            raise ValueError(f"{path} must contain scenario and scenario_index fields")
    return rows


def count_top_constitutions(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count top-1 constitutions for a retrieval system."""
    return dict(sorted(Counter(row[key][0]["constitution"] for row in rows).items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL records."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact top-1 retrieval rows."""
    fields = [
        "scenario_index",
        "annotation_primary",
        "annotation_confidence",
        "scenario",
        "card_top_constitution",
        "card_top_score",
        "card_top_criterion",
        "raw_top_constitution",
        "raw_top_score",
        "raw_top_criterion",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            card_top = row["card_top"][0]
            raw_top = row["raw_top"][0]
            writer.writerow(
                {
                    "scenario_index": row["scenario_index"],
                    "annotation_primary": row["annotation_primary"],
                    "annotation_confidence": row["annotation_confidence"],
                    "scenario": row["scenario"],
                    "card_top_constitution": card_top["constitution"],
                    "card_top_score": card_top["score"],
                    "card_top_criterion": card_top["criterion_id"],
                    "raw_top_constitution": raw_top["constitution"],
                    "raw_top_score": raw_top["score"],
                    "raw_top_criterion": raw_top["criterion_id"],
                }
            )


def run_model(model: str, args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    """Run embedding retrieval for one model."""
    annotations = load_annotations(args.annotations)
    card_docs = load_jsonl(DEFAULT_CARD_CORPUS)
    raw_docs = load_jsonl(DEFAULT_RAW_CORPUS)
    scenarios = [row["scenario"] for row in annotations]

    scenario_vectors = load_or_embed(
        model=model,
        corpus_name="airisk_623",
        texts=scenarios,
        api_key=api_key,
        batch_size=args.batch_size,
        force=args.force_scenarios,
    )
    card_vectors = load_or_embed(
        model=model,
        corpus_name="cards",
        texts=[card_text(doc) for doc in card_docs],
        api_key=api_key,
        batch_size=args.batch_size,
        force=False,
    )
    raw_vectors = load_or_embed(
        model=model,
        corpus_name="raw_criteria",
        texts=[raw_text(doc) for doc in raw_docs],
        api_key=api_key,
        batch_size=args.batch_size,
        force=False,
    )

    card_rankings = EmbeddingRetriever(card_docs, card_vectors).retrieve_many(scenario_vectors, args.top_k)
    raw_rankings = EmbeddingRetriever(raw_docs, raw_vectors).retrieve_many(scenario_vectors, args.top_k)

    rows = []
    for idx, annotation in enumerate(annotations):
        rows.append(
            {
                "scenario_index": annotation["scenario_index"],
                "scenario": annotation["scenario"],
                "annotation_primary": annotation["primary_constitution"],
                "annotation_confidence": annotation["confidence"],
                "card_top": card_rankings[idx],
                "raw_top": raw_rankings[idx],
            }
        )

    unit_ks = [1, 2, 3, 5, 10, 20]
    collapsed_ks = [1, 2, 3, 4, 5]
    labelled_count = sum(row["annotation_primary"] != "none" for row in rows)
    high_conf_rows = [row for row in rows if row["annotation_confidence"] == "high"]
    high_conf_labelled = [row for row in high_conf_rows if row["annotation_primary"] != "none"]
    summary = {
        "model": model,
        "scenarios": len(rows),
        "cards": len(card_docs),
        "raw_criteria": len(raw_docs),
        "top_k": args.top_k,
        "labelled_scenarios": labelled_count,
        "annotation_primary_counts": dict(sorted(Counter(row["annotation_primary"] for row in rows).items())),
        "annotation_confidence_counts": dict(sorted(Counter(row["annotation_confidence"] for row in rows).items())),
        "card_top_constitution_counts": count_top_constitutions(rows, "card_top"),
        "raw_top_constitution_counts": count_top_constitutions(rows, "raw_top"),
        "card_raw_top1_agree": sum(
            row["card_top"][0]["constitution"] == row["raw_top"][0]["constitution"] for row in rows
        ),
        "metric_definitions": {
            "unit_recall": "Rank cards/criteria; recall@k is a hit if any of the first k retrieved units maps to the assigned constitution.",
            "collapsed_constitution_recall": "Rank cards/criteria, collapse to distinct constitutions preserving first occurrence, then compute recall@k over that constitution shortlist.",
        },
        "card_unit_recall": recall_at_k(rows, "card_top", unit_ks, hit_mode="unit"),
        "raw_unit_recall": recall_at_k(rows, "raw_top", unit_ks, hit_mode="unit"),
        "card_collapsed_constitution_recall": recall_at_k(rows, "card_top", collapsed_ks, hit_mode="collapsed"),
        "raw_collapsed_constitution_recall": recall_at_k(rows, "raw_top", collapsed_ks, hit_mode="collapsed"),
        "high_conf_labelled_scenarios": len(high_conf_labelled),
        "high_conf_card_unit_recall": recall_at_k(high_conf_rows, "card_top", unit_ks, hit_mode="unit"),
        "high_conf_raw_unit_recall": recall_at_k(high_conf_rows, "raw_top", unit_ks, hit_mode="unit"),
        "high_conf_card_collapsed_constitution_recall": recall_at_k(
            high_conf_rows,
            "card_top",
            collapsed_ks,
            hit_mode="collapsed",
        ),
        "high_conf_raw_collapsed_constitution_recall": recall_at_k(
            high_conf_rows,
            "raw_top",
            collapsed_ks,
            hit_mode="collapsed",
        ),
    }

    model_dir = DEFAULT_OUTPUT_ROOT / slugify_model(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(model_dir / "airisk_623_embedding_retrieval.jsonl", rows)
    write_csv(model_dir / "airisk_623_embedding_retrieval.csv", rows)
    (model_dir / "airisk_623_embedding_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-scenarios", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY")
    for model in args.model or [DEFAULT_MODEL]:
        print(json.dumps(run_model(model, args, api_key), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
