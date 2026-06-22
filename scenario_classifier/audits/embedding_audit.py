"""Run OpenRouter embedding retrieval over OASST scenarios.

The script embeds the same scenario/card/raw-criteria corpora used by the BM25
audit, caches vectors per model, and ranks criteria by cosine similarity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from scenario_classifier.core.io import DEFAULT_CARD_CORPUS, DEFAULT_RAW_CORPUS, DEFAULT_SCENARIOS
from scenario_classifier.core.io import OUTPUT_ROOT, REPO_ROOT, load_jsonl, load_scenarios
from scenario_classifier.retrieval.retrievers import EmbeddingRetriever


DEFAULT_MODEL = "google/gemini-embedding-2"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "embedding_audit"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_APP_REFERER = "https://github.com/Constitutional-Evals/"
OPENROUTER_APP_TITLE = "Project 1 Scenario Classifier"


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


def card_text(doc: dict[str, Any]) -> str:
    """Return the text indexed for a compressed criteria card."""
    return str(doc.get("embedding_text") or doc.get("claim") or "")


def raw_text(doc: dict[str, Any]) -> str:
    """Return the text indexed for a raw criterion baseline record."""
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
        data = response.json()["data"]
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
) -> np.ndarray:
    """Load cached vectors for a corpus, or embed and cache them."""
    model_dir = DEFAULT_OUTPUT_DIR / slugify_model(model)
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a spreadsheet-friendly top-1 summary."""
    fields = [
        "scenario_index",
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
                    "scenario": row["scenario"],
                    "card_top_constitution": card_top["constitution"],
                    "card_top_score": card_top["score"],
                    "card_top_criterion": card_top["criterion_id"],
                    "raw_top_constitution": raw_top["constitution"],
                    "raw_top_score": raw_top["score"],
                    "raw_top_criterion": raw_top["criterion_id"],
                }
            )


def run_model(model: str, *, args: argparse.Namespace, api_key: str) -> None:
    """Embed corpora for one model and write retrieval outputs."""
    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    card_docs = load_jsonl(DEFAULT_CARD_CORPUS)
    raw_docs = load_jsonl(DEFAULT_RAW_CORPUS)

    scenario_vectors = load_or_embed(
        model=model,
        corpus_name="oasst",
        texts=scenarios,
        api_key=api_key,
        batch_size=args.batch_size,
        force=args.force,
    )
    card_vectors = load_or_embed(
        model=model,
        corpus_name="cards",
        texts=[card_text(doc) for doc in card_docs],
        api_key=api_key,
        batch_size=args.batch_size,
        force=args.force,
    )
    raw_vectors = load_or_embed(
        model=model,
        corpus_name="raw_criteria",
        texts=[raw_text(doc) for doc in raw_docs],
        api_key=api_key,
        batch_size=args.batch_size,
        force=args.force,
    )

    card_retriever = EmbeddingRetriever(card_docs, card_vectors)
    raw_retriever = EmbeddingRetriever(raw_docs, raw_vectors)
    card_scores = card_retriever.scores(scenario_vectors)
    raw_scores = raw_retriever.scores(scenario_vectors)
    card_rankings = card_retriever.retrieve_many(scenario_vectors, args.top_k)
    raw_rankings = raw_retriever.retrieve_many(scenario_vectors, args.top_k)
    rows = [
        {
            "scenario_index": idx,
            "scenario": scenario,
            "card_top": card_rankings[idx],
            "raw_top": raw_rankings[idx],
        }
        for idx, scenario in enumerate(scenarios)
    ]

    model_dir = DEFAULT_OUTPUT_DIR / slugify_model(model)
    retrieval_jsonl = model_dir / "oasst_embedding_retrieval.jsonl"
    retrieval_csv = model_dir / "oasst_embedding_retrieval.csv"
    summary_path = model_dir / "summary.json"
    summary = {
        "model": model,
        "scenarios": len(scenarios),
        "cards": len(card_docs),
        "raw_criteria": len(raw_docs),
        "top_k": args.top_k,
        "card_top_score_avg": round(float(card_scores.max(axis=1).mean()), 6),
        "raw_top_score_avg": round(float(raw_scores.max(axis=1).mean()), 6),
    }

    write_jsonl(retrieval_jsonl, rows)
    write_csv(retrieval_csv, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {retrieval_jsonl}")
    print(f"wrote: {retrieval_csv}")
    print(f"wrote: {summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help=f"default: {DEFAULT_MODEL}")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY")

    for model in args.model or [DEFAULT_MODEL]:
        run_model(model, args=args, api_key=api_key)


if __name__ == "__main__":
    main()
