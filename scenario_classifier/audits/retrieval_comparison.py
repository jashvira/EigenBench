"""Build the labelled retrieval comparison table.

Raw criteria are the default report. Cards remain available via `--corpus cards`
or `--corpus both`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, OUTPUT_ROOT, load_jsonl
from scenario_classifier.core.ranking_metrics import recall_at_k
from scenario_classifier.retrieval.retrievers import BM25Retriever


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "embedding_retrieval"
CORPUS_CHOICES = ("raw", "cards", "both")
DEFAULT_MODELS = [
    ("google/gemini-embedding-2", "Gemini"),
    ("qwen/qwen3-embedding-8b", "Qwen 8B"),
]
KS = [1, 2, 3, 5]


def slugify_model(model: str) -> str:
    """Map an OpenRouter model slug to a filesystem-safe folder name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model).strip("_")


def wants_cards(corpus: str) -> bool:
    return corpus in {"cards", "both"}


def wants_raw(corpus: str) -> bool:
    return corpus in {"raw", "both"}


def card_text(doc: dict[str, Any]) -> str:
    return str(doc.get("embedding_text") or doc.get("claim") or "")


def raw_text(doc: dict[str, Any]) -> str:
    return str(doc.get("embedding_text") or doc.get("criterion_text") or "")


def corpus_hash(texts: list[str]) -> str:
    """Hash corpus text in the same order used for cached embeddings."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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


def expected_corpus_hashes(corpus: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if wants_cards(corpus):
        hashes["cards"] = corpus_hash([card_text(doc) for doc in load_jsonl(DEFAULT_CARD_CORPUS)])
    if wants_raw(corpus):
        hashes["raw_criteria"] = corpus_hash([raw_text(doc) for doc in load_jsonl(DEFAULT_RAW_CORPUS)])
    return hashes


def check_embedding_cache(model: str, model_dir: Path, expected: dict[str, str]) -> None:
    for corpus_name, expected_hash in expected.items():
        manifest_path = model_dir / f"{corpus_name}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing embedding manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != expected_hash:
            raise RuntimeError(
                f"stale {corpus_name} embeddings for {model}; "
                "rerun scenario_classifier.audits.embedding_retrieval for this dataset and corpus"
            )


def load_embedding_rows(model: str, output_dir: Path, output_name: str) -> list[dict[str, Any]]:
    """Load cached retrieval rows for one embedding model."""
    model_dir = output_dir / slugify_model(model)
    path = model_dir / f"{output_name}_embedding_retrieval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing retrieval rows: {path}")
    return load_jsonl(path)


def require_ranking(rows: list[dict[str, Any]], ranking_key: str, system: str) -> None:
    """Fail clearly if cached retrieval rows were generated without a corpus."""
    if rows and ranking_key not in rows[0]:
        raise RuntimeError(
            f"cached {system} rows do not contain {ranking_key}; "
            "rerun scenario_classifier.audits.embedding_retrieval with the same --corpus"
        )


def bm25_rows(annotations: list[dict[str, Any]], *, top_k: int, corpus: str) -> list[dict[str, Any]]:
    """Run BM25 and return rows shaped like embedding retrieval output."""
    card_retriever = BM25Retriever(load_jsonl(DEFAULT_CARD_CORPUS)) if wants_cards(corpus) else None
    raw_retriever = BM25Retriever(load_jsonl(DEFAULT_RAW_CORPUS)) if wants_raw(corpus) else None
    rows: list[dict[str, Any]] = []
    for row in annotations:
        rows.append(
            {
                "scenario_index": row["scenario_index"],
                "scenario": row["scenario"],
                "annotation_primary": row["primary_constitution"],
                "annotation_confidence": row["confidence"],
            }
            | ({"card_top": card_retriever.retrieve(row["scenario"], top_k)} if card_retriever else {})
            | ({"raw_top": raw_retriever.retrieve(row["scenario"], top_k)} if raw_retriever else {})
        )
    return rows


def build_table(args: argparse.Namespace) -> dict[str, Any]:
    """Build the retrieval comparison payload."""
    annotations = load_jsonl(args.annotations)
    table_rows: list[dict[str, Any]] = []
    expected_hashes = expected_corpus_hashes(args.corpus)
    for model, _ in DEFAULT_MODELS:
        check_embedding_cache(model, args.output_dir / slugify_model(model), expected_hashes)

    embedding_rows = {
        label: labelled_high_conf(load_embedding_rows(model, args.output_dir, args.output_name))
        for model, label in DEFAULT_MODELS
    }
    if wants_cards(args.corpus):
        for label in ["Gemini", "Qwen 8B"]:
            require_ranking(embedding_rows[label], "card_top", label)
            table_rows.append(table_row(f"{label} cards", embedding_rows[label], "card_top"))
    if wants_raw(args.corpus):
        for label in ["Gemini", "Qwen 8B"]:
            require_ranking(embedding_rows[label], "raw_top", label)
            table_rows.append(table_row(f"{label} raw criteria", embedding_rows[label], "raw_top"))

    bm25 = labelled_high_conf(bm25_rows(annotations, top_k=max(KS), corpus=args.corpus))
    if wants_cards(args.corpus):
        table_rows.append(table_row("BM25 cards", bm25, "card_top"))
    if wants_raw(args.corpus):
        table_rows.append(table_row("BM25 raw criteria", bm25, "raw_top"))

    return {
        "metric": "unit_recall",
        "dataset": args.output_name,
        "corpus": args.corpus,
        "metric_definition": "Rank selected corpus units; recall@k is a hit if any of the first k retrieved units maps to the assigned constitution.",
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
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-name", required=True, help="embedding retrieval output prefix")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--corpus", choices=CORPUS_CHOICES, default="raw")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    try:
        payload = build_table(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    json_path = args.json or args.output_dir / f"{args.output_name}_retrieval_comparison.json"
    csv_path = args.csv or args.output_dir / f"{args.output_name}_retrieval_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, payload["rows"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
