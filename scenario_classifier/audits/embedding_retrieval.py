"""Generic embedding retrieval runner for scenario-classifier corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests

from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, OUTPUT_ROOT, REPO_ROOT
from scenario_classifier.core.io import load_jsonl, load_records
from scenario_classifier.core.ranking_metrics import recall_at_k
from scenario_classifier.retrieval.retrievers import EmbeddingRetriever


DEFAULT_MODEL = "google/gemini-embedding-2"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "embedding_retrieval"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_APP_REFERER = "https://github.com/Constitutional-Evals/"
OPENROUTER_APP_TITLE = "Project 1 Scenario Classifier"
CORPUS_CHOICES = ("raw", "cards", "both")


def load_dotenv(path: Path) -> None:
    """Load the repo-local OpenRouter key if the shell has not set one."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            os.environ.setdefault("OPENROUTER_API_KEY", line.split("=", 1)[1].strip())


def slugify_model(model: str) -> str:
    """Map an OpenRouter model slug to a filesystem-safe folder name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model).strip("_")


def wants_cards(corpus: str) -> bool:
    """Return whether this run should include criteria cards."""
    return corpus in {"cards", "both"}


def wants_raw(corpus: str) -> bool:
    """Return whether this run should include raw criteria."""
    return corpus in {"raw", "both"}


def card_text(doc: dict[str, Any]) -> str:
    """Return the text indexed for a compressed criteria card."""
    return str(doc.get("embedding_text") or doc.get("claim") or "")


def raw_text(doc: dict[str, Any]) -> str:
    """Return the cleaned criterion words indexed for raw-criteria retrieval."""
    return str(doc.get("embedding_text") or doc.get("criterion_text") or "")


def corpus_hash(texts: list[str]) -> str:
    """Hash corpus text so cached vectors are invalidated when inputs change."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def embed_batch(model: str, texts: list[str], api_key: str) -> list[list[float]]:
    """Call OpenRouter's OpenAI-compatible embeddings endpoint for one batch."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_APP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }
    payload = {"model": model, "input": texts}

    for attempt in range(5):
        response = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2**attempt)
            continue
        if not response.ok:
            raise RuntimeError(f"OpenRouter embeddings failed: {response.status_code} {response.text[:500]}")
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"OpenRouter embeddings returned no data: {str(payload)[:500]}")
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    raise RuntimeError("OpenRouter embeddings failed after retries")


def load_or_embed(
    *,
    model: str,
    corpus_name: str,
    texts: list[str],
    api_key: str,
    batch_size: int,
    force: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> np.ndarray:
    """Load cached vectors for a corpus, or embed and cache them."""
    model_dir = output_dir / slugify_model(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = model_dir / f"{corpus_name}_embeddings.npy"
    manifest_path = model_dir / f"{corpus_name}_manifest.json"
    manifest = {"model": model, "corpus": corpus_name, "count": len(texts), "sha256": corpus_hash(texts)}

    if not force and vectors_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
            print(f"{model}: cached {corpus_name}")
            return np.load(vectors_path)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed_batch(model, batch, api_key))
        print(f"{model}: embedded {min(start + len(batch), len(texts))}/{len(texts)} {corpus_name}")

    array = np.array(vectors, dtype=np.float32)
    np.save(vectors_path, array)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return array


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write full top-k retrieval rows."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact top-1 retrieval rows."""
    fields = ["scenario_index"]
    if rows and "annotation_primary" in rows[0]:
        fields.extend(["annotation_primary", "annotation_confidence"])
    fields.append("scenario")
    if rows and "card_top" in rows[0]:
        fields.extend(["card_top_constitution", "card_top_score", "card_top_criterion"])
    if rows and "raw_top" in rows[0]:
        fields.extend(["raw_top_constitution", "raw_top_score", "raw_top_criterion"])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "scenario_index": row["scenario_index"],
                "scenario": row["scenario"],
            }
            if "annotation_primary" in row:
                out["annotation_primary"] = row["annotation_primary"]
                out["annotation_confidence"] = row.get("annotation_confidence", "")
            if "card_top" in row:
                card_top = row["card_top"][0]
                out.update(
                    {
                        "card_top_constitution": card_top["constitution"],
                        "card_top_score": card_top["score"],
                        "card_top_criterion": card_top["criterion_id"],
                    }
                )
            if "raw_top" in row:
                raw_top = row["raw_top"][0]
                out.update(
                    {
                        "raw_top_constitution": raw_top["constitution"],
                        "raw_top_score": raw_top["score"],
                        "raw_top_criterion": raw_top["criterion_id"],
                    }
                )
            writer.writerow(out)


def count_top_constitutions(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count top-1 constitutions for a retrieval system."""
    return dict(sorted(Counter(row[key][0]["constitution"] for row in rows).items()))


def add_label_metrics(summary: dict[str, Any], rows: list[dict[str, Any]], corpus: str) -> None:
    """Add recall metrics when records contain reviewed constitution labels."""
    if not rows or "annotation_primary" not in rows[0]:
        return

    unit_ks = [1, 2, 3, 5, 10, 20]
    collapsed_ks = [1, 2, 3, 4, 5]
    high_conf_rows = [row for row in rows if row.get("annotation_confidence") == "high"]
    high_conf_labelled = [row for row in high_conf_rows if row["annotation_primary"] != "none"]

    summary.update(
        {
            "labelled_scenarios": sum(row["annotation_primary"] != "none" for row in rows),
            "annotation_primary_counts": dict(sorted(Counter(row["annotation_primary"] for row in rows).items())),
            "annotation_confidence_counts": dict(
                sorted(Counter(row.get("annotation_confidence", "") for row in rows).items())
            ),
            "metric_definitions": {
                "unit_recall": "Rank selected corpus units; recall@k is a hit if any of the first k retrieved units maps to the assigned constitution.",
                "collapsed_constitution_recall": "Rank selected corpus units, collapse to distinct constitutions preserving first occurrence, then compute recall@k over that constitution shortlist.",
            },
            "high_conf_labelled_scenarios": len(high_conf_labelled),
        }
    )

    if wants_cards(corpus):
        summary["card_top_constitution_counts"] = count_top_constitutions(rows, "card_top")
        summary["card_unit_recall"] = recall_at_k(rows, "card_top", unit_ks, hit_mode="unit")
        summary["card_collapsed_constitution_recall"] = recall_at_k(
            rows,
            "card_top",
            collapsed_ks,
            hit_mode="collapsed",
        )
        summary["high_conf_card_unit_recall"] = recall_at_k(high_conf_rows, "card_top", unit_ks, hit_mode="unit")
        summary["high_conf_card_collapsed_constitution_recall"] = recall_at_k(
            high_conf_rows,
            "card_top",
            collapsed_ks,
            hit_mode="collapsed",
        )

    if wants_raw(corpus):
        summary["raw_top_constitution_counts"] = count_top_constitutions(rows, "raw_top")
        summary["raw_unit_recall"] = recall_at_k(rows, "raw_top", unit_ks, hit_mode="unit")
        summary["raw_collapsed_constitution_recall"] = recall_at_k(rows, "raw_top", collapsed_ks, hit_mode="collapsed")
        summary["high_conf_raw_unit_recall"] = recall_at_k(high_conf_rows, "raw_top", unit_ks, hit_mode="unit")
        summary["high_conf_raw_collapsed_constitution_recall"] = recall_at_k(
            high_conf_rows,
            "raw_top",
            collapsed_ks,
            hit_mode="collapsed",
        )

    if wants_cards(corpus) and wants_raw(corpus):
        summary["card_raw_top1_agree"] = sum(
            row["card_top"][0]["constitution"] == row["raw_top"][0]["constitution"] for row in rows
        )


def run_model(
    *,
    model: str,
    records: list[dict[str, Any]],
    output_name: str,
    corpus: str,
    api_key: str,
    top_k: int,
    batch_size: int,
    force_scenarios: bool,
    force_corpora: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Embed scenarios, rank requested corpora, and write retrieval outputs."""
    include_cards = wants_cards(corpus)
    include_raw = wants_raw(corpus)
    card_docs = load_jsonl(DEFAULT_CARD_CORPUS) if include_cards else []
    raw_docs = load_jsonl(DEFAULT_RAW_CORPUS) if include_raw else []
    scenarios = [row["scenario"] for row in records]

    scenario_vectors = load_or_embed(
        model=model,
        corpus_name=output_name,
        texts=scenarios,
        api_key=api_key,
        batch_size=batch_size,
        force=force_scenarios,
        output_dir=output_dir,
    )

    card_scores = None
    card_rankings = None
    if include_cards:
        card_vectors = load_or_embed(
            model=model,
            corpus_name="cards",
            texts=[card_text(doc) for doc in card_docs],
            api_key=api_key,
            batch_size=batch_size,
            force=force_corpora,
            output_dir=output_dir,
        )
        card_retriever = EmbeddingRetriever(card_docs, card_vectors)
        card_scores = card_retriever.scores(scenario_vectors)
        card_rankings = card_retriever.retrieve_many(scenario_vectors, top_k)

    raw_scores = None
    raw_rankings = None
    if include_raw:
        raw_vectors = load_or_embed(
            model=model,
            corpus_name="raw_criteria",
            texts=[raw_text(doc) for doc in raw_docs],
            api_key=api_key,
            batch_size=batch_size,
            force=force_corpora,
            output_dir=output_dir,
        )
        raw_retriever = EmbeddingRetriever(raw_docs, raw_vectors)
        raw_scores = raw_retriever.scores(scenario_vectors)
        raw_rankings = raw_retriever.retrieve_many(scenario_vectors, top_k)

    rows = []
    for idx, record in enumerate(records):
        row = dict(record)
        if card_rankings is not None:
            row["card_top"] = card_rankings[idx]
        if raw_rankings is not None:
            row["raw_top"] = raw_rankings[idx]
        rows.append(row)

    model_dir = output_dir / slugify_model(model)
    retrieval_jsonl = model_dir / f"{output_name}_embedding_retrieval.jsonl"
    retrieval_csv = model_dir / f"{output_name}_embedding_retrieval.csv"
    summary_path = model_dir / f"{output_name}_embedding_retrieval_summary.json"
    summary: dict[str, Any] = {
        "model": model,
        "corpus": corpus,
        "dataset": output_name,
        "scenarios": len(records),
        "top_k": top_k,
    }
    if include_cards and card_scores is not None:
        summary["cards"] = len(card_docs)
        summary["card_top_score_avg"] = round(float(card_scores.max(axis=1).mean()), 6)
    if include_raw and raw_scores is not None:
        summary["raw_criteria"] = len(raw_docs)
        summary["raw_top_score_avg"] = round(float(raw_scores.max(axis=1).mean()), 6)
    add_label_metrics(summary, rows, corpus)

    write_jsonl(retrieval_jsonl, rows)
    write_csv(retrieval_csv, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {retrieval_jsonl}")
    print(f"wrote: {retrieval_csv}")
    print(f"wrote: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the generic runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="JSON list or JSONL records containing scenarios")
    parser.add_argument("--output-name", required=True, help="cache/output prefix, e.g. airisk_623")
    parser.add_argument("--model", action="append", default=[], help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--scenario-field", default="scenario")
    parser.add_argument("--index-field", default="scenario_index")
    parser.add_argument("--label-field", default="primary_constitution")
    parser.add_argument("--confidence-field", default="confidence")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--corpus", choices=CORPUS_CHOICES, default="raw")
    parser.add_argument("--force-scenarios", action="store_true")
    parser.add_argument("--force-corpora", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY")

    records = load_records(
        args.records,
        scenario_field=args.scenario_field,
        index_field=args.index_field,
        label_field=args.label_field,
        confidence_field=args.confidence_field,
    )
    for model in args.model or [DEFAULT_MODEL]:
        print(
            json.dumps(
                run_model(
                    model=model,
                    records=records,
                    output_name=args.output_name,
                    corpus=args.corpus,
                    api_key=api_key,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    force_scenarios=args.force_scenarios,
                    force_corpora=args.force_corpora,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
