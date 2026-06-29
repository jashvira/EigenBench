"""Run BM25 retrieval over an explicit scenario file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, OUTPUT_ROOT
from scenario_classifier.core.io import load_jsonl, load_records
from scenario_classifier.retrieval.retrievers import BM25Retriever


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "bm25_retrieval"
CORPUS_CHOICES = ("raw", "cards", "both")
DEFAULT_TOP_K = 8


def wants_cards(corpus: str) -> bool:
    return corpus in {"cards", "both"}


def wants_raw(corpus: str) -> bool:
    return corpus in {"raw", "both"}


def margin(top: list[dict[str, Any]]) -> float:
    """Return the score gap between the top two candidates."""
    if not top:
        return 0.0
    if len(top) == 1:
        return float(top[0]["score"])
    return round(float(top[0]["score"]) - float(top[1]["score"]), 6)


def flag(card_top: list[dict[str, Any]], raw_top: list[dict[str, Any]]) -> str:
    """Compare the top card hit and top raw-criterion hit for audit triage."""
    card_score = float(card_top[0]["score"]) if card_top else 0.0
    raw_score = float(raw_top[0]["score"]) if raw_top else 0.0
    if card_score == 0 and raw_score == 0:
        return "no_match"
    if card_score > 0 and raw_score == 0:
        return "card_only"
    if card_score == 0 and raw_score > 0:
        return "raw_only"
    if card_top[0]["constitution"] == raw_top[0]["constitution"]:
        return "both_agree"
    return "disagree"


def positive_candidates(*ranked_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge positive BM25 hits from selected corpora without duplicates."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ranked in ranked_lists:
        for item in ranked:
            if float(item["score"]) <= 0:
                continue
            criterion_id = str(item["criterion_id"])
            if criterion_id in seen:
                continue
            seen.add(criterion_id)
            candidates.append(item)
    return candidates


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval outcomes for the audit summary."""
    flag_counts = Counter(row["flag"] for row in rows)
    gate_counts = Counter(row["gate_decision"] for row in rows)
    card_counts = Counter(row["card_top_constitution"] for row in rows if row["card_top_score"] > 0)
    raw_counts = Counter(row["raw_top_constitution"] for row in rows if row["raw_top_score"] > 0)
    candidate_counts = [row["candidate_count"] for row in rows]
    return {
        "rows": len(rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "gate_decision_counts": dict(sorted(gate_counts.items())),
        "avg_candidate_count": round(sum(candidate_counts) / max(len(candidate_counts), 1), 3),
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
        "card_top_constitution_counts": dict(sorted(card_counts.items())),
        "raw_top_constitution_counts": dict(sorted(raw_counts.items())),
    }


def compact_top(top: list[dict[str, Any]]) -> str:
    """Compact ranked hits for the CSV export."""
    return " | ".join(f"{item['rank']}. {item['criterion_id']} ({item['score']})" for item in top)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write audit rows as compact JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the retrieval table in spreadsheet-friendly form."""
    fields = ["scenario_index"]
    if rows and "annotation_primary" in rows[0]:
        fields.extend(["annotation_primary", "annotation_confidence"])
    fields.extend(
        [
            "scenario",
            "gate_decision",
            "candidate_count",
            "candidate_constitutions",
            "flag",
            "card_top_constitution",
            "card_top_score",
            "card_margin",
            "card_top",
            "raw_top_constitution",
            "raw_top_score",
            "raw_margin",
            "raw_top",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "scenario_index": row["scenario_index"],
                "scenario": row["scenario"],
                "gate_decision": row["gate_decision"],
                "candidate_count": row["candidate_count"],
                "candidate_constitutions": ";".join(row["candidate_constitutions"]),
                "flag": row["flag"],
                "card_top_constitution": row["card_top_constitution"],
                "card_top_score": row["card_top_score"],
                "card_margin": row["card_margin"],
                "card_top": compact_top(row["card_top"]),
                "raw_top_constitution": row["raw_top_constitution"],
                "raw_top_score": row["raw_top_score"],
                "raw_margin": row["raw_margin"],
                "raw_top": compact_top(row["raw_top"]),
            }
            if "annotation_primary" in row:
                out["annotation_primary"] = row["annotation_primary"]
                out["annotation_confidence"] = row.get("annotation_confidence", "")
            writer.writerow(out)


def build_rows(records: list[dict[str, Any]], *, top_k: int, corpus: str) -> list[dict[str, Any]]:
    """Run BM25 over the selected corpora and return ranked candidates."""
    card_index = BM25Retriever(load_jsonl(DEFAULT_CARD_CORPUS)) if wants_cards(corpus) else None
    raw_index = BM25Retriever(load_jsonl(DEFAULT_RAW_CORPUS)) if wants_raw(corpus) else None

    rows: list[dict[str, Any]] = []
    for record in records:
        scenario = record["scenario"]
        card_top = card_index.top_k(scenario, top_k) if card_index else []
        raw_top = raw_index.top_k(scenario, top_k) if raw_index else []
        candidates = positive_candidates(card_top, raw_top)
        row = dict(record)
        row.update(
            {
                "flag": flag(card_top, raw_top),
                "gate_decision": "pass_bm25" if candidates else "no_lexical_match",
                "candidate_count": len(candidates),
                "candidate_constitutions": sorted({str(item["constitution"]) for item in candidates}),
                "candidate_criteria": [item["criterion_id"] for item in candidates],
                "card_top_constitution": card_top[0]["constitution"] if card_top else "",
                "card_top_score": card_top[0]["score"] if card_top else 0.0,
                "card_margin": margin(card_top),
                "card_top": card_top,
                "raw_top_constitution": raw_top[0]["constitution"] if raw_top else "",
                "raw_top_score": raw_top[0]["score"] if raw_top else 0.0,
                "raw_margin": margin(raw_top),
                "raw_top": raw_top,
            }
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="JSON list or JSONL records containing scenarios")
    parser.add_argument("--output-name", required=True, help="output prefix")
    parser.add_argument("--scenario-field", default="scenario")
    parser.add_argument("--index-field", default="scenario_index")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--corpus", choices=CORPUS_CHOICES, default="raw")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run BM25 retrieval and write JSONL, CSV, and summary JSON."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.records, scenario_field=args.scenario_field, index_field=args.index_field)
    rows = build_rows(records, top_k=args.top_k, corpus=args.corpus)
    summary = summarize(rows) | {
        "dataset": args.output_name,
        "records": str(args.records),
        "corpus": args.corpus,
        "top_k": args.top_k,
    }

    output_prefix = f"{args.output_name}_bm25_retrieval"
    jsonl_path = output_dir / f"{output_prefix}.jsonl"
    csv_path = output_dir / f"{output_prefix}.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"records: {len(records)}")
    print(f"corpus: {args.corpus}")
    print(f"flag_counts: {summary['flag_counts']}")
    print(f"gate_decision_counts: {summary['gate_decision_counts']}")
    print(f"avg_candidate_count: {summary['avg_candidate_count']}")
    print(f"wrote: {jsonl_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
