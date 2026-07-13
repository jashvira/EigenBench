"""Overlay repaired samples into one complete Inspect evaluation archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import ijson
import zipfile_zstd  # noqa: F401  Enables zstd support in Python's zipfile module.


ARCHIVE_COMPRESSION = zipfile.ZIP_DEFLATED


def sha256_file(path: Path) -> str:
    """Return the content hash recorded with the materialized log."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_samples(path: Path) -> dict[str, bytes]:
    """Read the replacement sample records keyed by Inspect sample ID."""
    with zipfile.ZipFile(path) as archive:
        samples = {
            sample["id"]: raw
            for info in archive.infolist()
            if info.filename.startswith("samples/")
            for raw in [archive.read(info.filename)]
            for sample in [json.loads(raw)]
        }
    if not samples:
        raise ValueError("Repair log contains no samples")
    return samples


def write_repaired_summaries(
    source: zipfile.ZipFile,
    output: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    replacements: dict[str, bytes],
) -> None:
    """Stream the sample summary list while replacing repaired cells."""
    target_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    target_info.compress_type = ARCHIVE_COMPRESSION
    with source.open(info) as source_file, output.open(target_info, "w") as target:
        target.write(b"[")
        first = True
        for sample in ijson.items(source_file, "item", use_float=True):
            if not first:
                target.write(b",")
            replacement = replacements.get(sample["id"])
            payload = replacement if replacement is not None else json.dumps(
                sample, separators=(",", ":")
            ).encode("utf-8")
            target.write(payload)
            first = False
        target.write(b"]")


def materialize(
    base_path: Path,
    repair_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> None:
    """Write an 8,000-sample log with repaired samples overlaid in place."""
    replacements = repair_samples(repair_path)
    manifest_hash = sha256_file(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        zipfile.ZipFile(base_path) as base_archive,
        zipfile.ZipFile(
            output_path, "w", compression=ARCHIVE_COMPRESSION, compresslevel=3
        ) as output,
    ):
        seen_replacements: set[str] = set()
        for info in base_archive.infolist():
            if info.filename == "summaries.json":
                write_repaired_summaries(base_archive, output, info, replacements)
                continue

            raw = base_archive.read(info.filename)
            if info.filename.startswith("samples/"):
                sample = json.loads(raw)
                replacement = replacements.get(sample["id"])
                if replacement is not None:
                    raw = replacement
                    seen_replacements.add(sample["id"])
            elif info.filename == "reductions.json":
                reductions = json.loads(raw)
                for reduction in reductions:
                    for score in reduction.get("samples", []):
                        replacement = replacements.get(score["sample_id"])
                        if replacement is not None:
                            sample = json.loads(replacement)
                            score.update(sample["scores"][reduction["scorer"]])
                raw = json.dumps(reductions, separators=(",", ":")).encode("utf-8")
            elif info.filename == "header.json":
                header = json.loads(raw)
                metadata = dict(header.get("metadata") or {})
                metadata.update(
                    {
                        "repair_overlay_cells": len(replacements),
                        "repair_overlay_manifest_hash": manifest_hash,
                    }
                )
                header["metadata"] = metadata
                header["eval"]["metadata"] = dict(header["eval"].get("metadata") or {})
                header["eval"]["metadata"].update(metadata)
                raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            output.writestr(info.filename, raw, compress_type=ARCHIVE_COMPRESSION)

    if seen_replacements != replacements.keys():
        raise ValueError("Repair log contains samples outside the base log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.base, args.repair, args.manifest, args.output)


if __name__ == "__main__":
    main()
