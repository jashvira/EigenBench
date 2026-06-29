"""Build the raw-criteria retrieval corpus for the scenario classifier."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import DATA_ROOT, DEFAULT_RAW_CORPUS, REPO_ROOT

CONSTITUTIONS_DIR = REPO_ROOT / "data" / "constitutions"
OUTPUT_PATH = DEFAULT_RAW_CORPUS
PUBLIC_CONSTITUTIONS = (
    "kindness",
    "conservatism",
    "deep_ecology",
)
PRIVATE_ANCHORS_DIR = DATA_ROOT / "private_anchors"
VERSION = "raw-v0.2"
CRITERION_PREFIX_RE = re.compile(
    r"^\s*Criterion\s+\d+\s*(?:for\s+[^:]+?)?(?:\s+is)?\s*:\s*",
    flags=re.IGNORECASE,
)
COMPARATIVE_PREFIX_RE = re.compile(
    r"^\s*prefer(?:s)?\s+(?:the\s+)?(?:assistant\s+)?(?:response|answer|reply)s?\s+(?:that|which|whose)\b[,\s]*",
    flags=re.IGNORECASE,
)
PLACEHOLDER_SUFFIX_RE = re.compile(r"\s*\(PLACEHOLDER\s+(?:\u2014|-)\s+misaligned anchor\)\s*$")


def criterion_number(text: str, source_order: int) -> int:
    match = re.match(r"\s*Criterion\s+(\d+)\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else source_order


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_criteria(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        criteria = payload
    elif isinstance(payload, dict):
        criteria = payload.get("criteria")
    else:
        criteria = None

    if not isinstance(criteria, list) or not all(isinstance(item, str) for item in criteria):
        if not isinstance(criteria, list) or not all(
            isinstance(item, dict) and isinstance(item.get("comparative"), str)
            for item in criteria
        ):
            raise ValueError(f"{path} must contain a criteria list")
        return [item["comparative"] for item in criteria]
    return criteria


def embedding_text_from_criterion(text: str) -> str:
    """Return only the criterion wording used for retrieval embeddings.

    Source files may label criteria as `Criterion 4 for Kindness:` and phrase
    them as `prefer the response that ...`. Those wrappers are evaluation
    syntax, not the criterion content we want in the vector.
    """
    text = CRITERION_PREFIX_RE.sub("", text, count=1).strip()
    text = COMPARATIVE_PREFIX_RE.sub("", text, count=1).strip()
    return PLACEHOLDER_SUFFIX_RE.sub("", text).strip()


def constitution_paths() -> list[tuple[str, Path]]:
    """Return tracked public constitutions plus optional ignored local anchors."""
    paths = [(constitution, CONSTITUTIONS_DIR / f"{constitution}.json") for constitution in PUBLIC_CONSTITUTIONS]
    if PRIVATE_ANCHORS_DIR.exists():
        paths.extend((path.stem, path) for path in sorted(PRIVATE_ANCHORS_DIR.glob("*.json")))
    return paths


def build_units() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for constitution, path in constitution_paths():
        source_sha = sha256_bytes(path)
        criteria = load_criteria(path)

        for source_order, criterion_text in enumerate(criteria, start=1):
            criterion_index = criterion_number(criterion_text, source_order)
            rows.append(
                {
                    "criterion_id": f"raw_criteria.{constitution}.criterion_{criterion_index:02d}",
                    "embedding_system": "raw_criteria",
                    "constitution": constitution,
                    "criterion_index": criterion_index,
                    "source_order": source_order,
                    "criterion_text": criterion_text,
                    "embedding_text": embedding_text_from_criterion(criterion_text),
                    "source_path": str(path.relative_to(REPO_ROOT)),
                    "source_sha256": source_sha,
                    "criterion_sha256": sha256_text(criterion_text),
                    "version": VERSION,
                }
            )

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    sources: dict[str, str] = {}
    for row in rows:
        counts[row["constitution"]] = counts.get(row["constitution"], 0) + 1
        sources[row["source_path"]] = row["source_sha256"]

    manifest = {
        "embedding_system": "raw_criteria",
        "version": VERSION,
        "row_count": len(rows),
        "constitution_count": len(counts),
        "counts_by_constitution": dict(sorted(counts.items())),
        "embedding_text_policy": "criterion wording only; Criterion N / constitution prefixes, prefer-the-response wrappers, and placeholder anchor suffixes stripped",
        "sources": dict(sorted(sources.items())),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_units()
    write_jsonl(OUTPUT_PATH, rows)
    write_manifest(OUTPUT_PATH.with_suffix(".manifest.json"), rows)

    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
