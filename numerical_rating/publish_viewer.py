"""Publish complete numerical round-robin logs with Inspect's native viewer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import zipfile_zstd  # noqa: F401  Enables zstd support in Python's zipfile module.

from numerical_rating.data import (
    RatingConfig,
    load_config,
    load_constitution_text,
    provenance_path,
    repo_path,
    sha256_text,
)
from numerical_rating.materialize_eval_log import materialize
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_REPAIR_CELLS = (
    ROOT
    / "data/output/valuearena/processed/full8_kindness/"
    "askreddit_1000_repaired_cells.json"
)
DEFAULT_SOURCES = (
    ROOT / "runs/numerical_rating/kindness_1000_round_robin",
    ROOT / "runs/numerical_rating/kindness_1000_round_robin_gemini4096",
    ROOT / "runs/numerical_rating/kindness_1000_round_robin_repairs",
)
DEFAULT_OUTPUT = ROOT / "docs/numerical_rating"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class SelectedLog:
    """A complete judge log selected for publication."""

    judge_model: str
    path: Path
    mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append", type=repo_path, default=[])
    parser.add_argument(
        "--repair-cells", type=repo_path, default=DEFAULT_REPAIR_CELLS
    )
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def expected_hashes(config: RatingConfig) -> tuple[str, str]:
    """Return the constitution and prompt hashes recorded by the task."""
    return (
        sha256_text(load_constitution_text(config.constitution)),
        sha256_text(config.prompt.read_text(encoding="utf-8")),
    )


def repair_cell_count(path: Path) -> int:
    """Return the number of response cells replaced by the repair run."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Repair manifest contains no cells")
    return len(cells)


def complete_logs(
    config: RatingConfig,
    sources: list[Path],
    config_name: str,
    *,
    expected_samples: int,
    cell_manifest: Path | None = None,
) -> list[SelectedLog]:
    """Select the newest valid complete log for each configured judge."""
    constitution_hash, prompt_hash = expected_hashes(config)
    expected_manifest = (
        str(cell_manifest.relative_to(ROOT)) if cell_manifest is not None else None
    )
    expected_manifest_hash = (
        sha256_text(cell_manifest.read_text(encoding="utf-8"))
        if cell_manifest is not None
        else None
    )
    expected_models = {judge.model for judge in config.judges}
    expected_config = provenance_path(config_name)
    generation_limits = {
        judge.model: (
            judge.max_tokens
            if judge.max_tokens is not None
            else config.generation_max_tokens
        )
        for judge in config.judges
    }
    selected: dict[str, SelectedLog] = {}

    for source in sources:
        for path in source.glob("*.eval"):
            with zipfile.ZipFile(path) as archive:
                header = json.loads(archive.read("header.json"))
            evaluation = header["eval"]
            metadata = evaluation.get("metadata") or {}
            results = header.get("results") or {}
            judge_model = str(metadata.get("judge_model") or "")
            if (
                judge_model not in expected_models
                or header.get("status") != "success"
                or header.get("invalidated")
                or evaluation.get("task") != "pointwise_constitution_rating"
                or metadata.get("run_id") != config.run_id
                or metadata.get("config") != expected_config
                or metadata.get("constitution_hash") != constitution_hash
                or metadata.get("prompt_hash") != prompt_hash
                or metadata.get("generation_max_tokens")
                != generation_limits.get(judge_model)
                or metadata.get("generation_temperature")
                != config.generation_temperature
                or metadata.get("cell_manifest") != expected_manifest
                or metadata.get("cell_manifest_hash") != expected_manifest_hash
                or results.get("total_samples") != expected_samples
                or results.get("completed_samples") != expected_samples
            ):
                continue

            candidate = SelectedLog(
                judge_model=judge_model,
                path=path,
                mtime=path.stat().st_mtime,
            )
            current = selected.get(judge_model)
            if current is None or candidate.mtime > current.mtime:
                selected[judge_model] = candidate

    missing = expected_models - selected.keys()
    if missing:
        raise ValueError("Missing complete judge logs: " + ", ".join(sorted(missing)))
    return [selected[judge.model] for judge in config.judges]


def validate_log_sizes(logs: list[SelectedLog]) -> None:
    """Reject logs that cannot be committed to GitHub Pages."""
    oversized = [
        log.path
        for log in logs
        if log.path.stat().st_size >= MAX_GITHUB_FILE_BYTES
    ]
    if oversized:
        raise ValueError(
            "Logs exceed GitHub's per-file limit: "
            + ", ".join(path.name for path in oversized)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_final_logs(
    base_logs: list[SelectedLog],
    repair_logs: list[SelectedLog],
    directory: Path,
    repair_cells: Path,
) -> list[Path]:
    """Materialize one final complete log per judge."""
    repair_by_judge = {log.judge_model: log for log in repair_logs}
    directory.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []

    for selected in base_logs:
        repair_path = repair_by_judge[selected.judge_model].path
        path = directory / selected.path.name
        materialize(selected.path, repair_path, repair_cells, path)
        final_paths.append(path)
        gc.collect()

    return final_paths


def publish_final_logs(paths: list[Path], output_dir: Path) -> None:
    """Replace the log payload while retaining the existing Inspect viewer."""
    if not (output_dir / "index.html").is_file():
        raise FileNotFoundError("Inspect viewer assets are missing")
    if not (ROOT / "docs/.nojekyll").is_file():
        raise FileNotFoundError("docs/.nojekyll is missing")
    logs_dir = output_dir / "logs"
    with tempfile.TemporaryDirectory(prefix="eigenbench-final-log-stage-") as work:
        stage = Path(work)
        for path in paths:
            target = stage / path.name
            subprocess.run(["cp", path, target], check=True)
        subprocess.run(
            ["rsync", "-a", "--delete", f"{stage}/", f"{logs_dir}/"], check=True
        )
    from inspect_ai.log import write_log_dir_manifest

    write_log_dir_manifest(str(logs_dir), filename="listing.json")
    manifest_path = logs_dir / "listing.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for header in manifest.values():
        header.pop("reductions", None)
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    published_hashes = {
        path.name: sha256_file(logs_dir / path.name)
        for path in paths
    }
    source_hashes = {path.name: sha256_file(path) for path in paths}
    if published_hashes != source_hashes:
        raise ValueError("Published logs differ from materialized logs")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sources = args.source or list(DEFAULT_SOURCES)
    base_logs = complete_logs(
        config,
        sources,
        args.config,
        expected_samples=config.expected_response_cells,
    )
    if not args.repair_cells.exists():
        raise FileNotFoundError(f"Repair manifest is missing: {args.repair_cells}")
    repair_logs = complete_logs(
        config,
        sources,
        args.config,
        expected_samples=repair_cell_count(args.repair_cells),
        cell_manifest=args.repair_cells,
    )
    validate_log_sizes(base_logs + repair_logs)
    with tempfile.TemporaryDirectory(prefix="eigenbench-numerical-view-") as work:
        work_dir = Path(work)
        final_dir = work_dir / "final"
        final_logs = materialize_final_logs(
            base_logs, repair_logs, final_dir, args.repair_cells
        )
        publish_final_logs(final_logs, args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "logs": [path.name for path in final_logs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
