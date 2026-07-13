"""Publish complete numerical round-robin logs with Inspect's native viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from inspect_ai.log import EvalLogInfo, list_eval_logs, read_eval_log

from numerical_rating.data import (
    RatingConfig,
    load_config,
    load_constitution_text,
    provenance_path,
    repo_path,
    sha256_text,
)
from numerical_rating.round_robin_analysis import load_ratings, rating_tensor
from numerical_rating.round_robin_analysis import (
    DEFAULT_REPAIR_CELLS,
    reconcile_ratings,
    repaired_cell_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_SOURCES = (
    ROOT / "runs/numerical_rating/kindness_1000_round_robin",
    ROOT / "runs/numerical_rating/kindness_1000_round_robin_gemini4096",
    ROOT / "runs/numerical_rating/kindness_1000_round_robin_repairs",
)
DEFAULT_OUTPUT = ROOT / "docs/numerical_rating"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
MAX_PAGES_SITE_BYTES = 1_000_000_000


@dataclass(frozen=True)
class SelectedLog:
    """A complete judge log selected for publication."""

    judge_model: str
    path: Path
    info: EvalLogInfo
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
        for info in list_eval_logs(str(source), formats=["eval"]):
            location = urlparse(info.name)
            path = Path(unquote(location.path))
            log = read_eval_log(info, header_only=True)
            metadata = log.eval.metadata or {}
            results = log.results
            judge_model = str(metadata.get("judge_model") or "")
            if (
                judge_model not in expected_models
                or log.status != "success"
                or log.invalidated
                or log.eval.task != "pointwise_constitution_rating"
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
                or not results
                or results.total_samples != expected_samples
                or results.completed_samples != expected_samples
            ):
                continue

            candidate = SelectedLog(
                judge_model=judge_model,
                path=path,
                info=info,
                mtime=float(info.mtime or 0),
            )
            current = selected.get(judge_model)
            if current is None or candidate.mtime > current.mtime:
                selected[judge_model] = candidate

    missing = expected_models - selected.keys()
    if missing:
        raise ValueError("Missing complete judge logs: " + ", ".join(sorted(missing)))
    return [selected[judge.model] for judge in config.judges]


def validate_logs(
    config: RatingConfig,
    base_logs: list[SelectedLog],
    repair_logs: list[SelectedLog],
) -> None:
    """Read every score and validate the full judge-scenario-target tensor."""
    logs = base_logs + repair_logs
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
    base_ratings = load_ratings([log.info for log in base_logs], config)
    repairs = load_ratings([log.info for log in repair_logs], config)
    ratings, _ = reconcile_ratings(base_ratings, repairs, config)
    tensor, _ = rating_tensor(ratings, config)
    if tensor.size != len(config.judges) * config.expected_response_cells:
        raise AssertionError("Validated tensor has an unexpected size")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_logs(logs: list[SelectedLog], directory: Path) -> None:
    """Hard-link selected logs into an isolated bundle input directory."""
    directory.mkdir(parents=True)
    for log in logs:
        target = directory / log.path.name
        try:
            os.link(log.path, target)
        except OSError:
            shutil.copy2(log.path, target)


def patch_log_directory(index_path: Path) -> None:
    """Remove the temporary absolute path embedded by Inspect's bundler."""
    text = index_path.read_text(encoding="utf-8")
    replacement = (
        '<script id="log_dir_context" type="application/json">'
        '{"log_dir": "logs", "abs_log_dir": "logs"}</script>'
    )
    text, count = re.subn(
        r'<script id="log_dir_context" type="application/json">.*?</script>',
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Inspect bundle has no unique log_dir_context element")
    index_path.write_text(text, encoding="utf-8")


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_bundle(
    log_dir: Path,
    site_dir: Path,
    output_dir: Path,
    expected_count: int,
) -> None:
    """Verify listing, bytes, paths, and hosting limits before publication."""
    listing = json.loads((site_dir / "logs/listing.json").read_text(encoding="utf-8"))
    if not isinstance(listing, dict) or len(listing) != expected_count:
        raise ValueError("Inspect bundle listing does not contain the expected logs")
    if not all(str(name).endswith(".eval") for name in listing):
        raise ValueError("Inspect bundle listing contains a non-eval log")

    source_hashes = {
        path.name: sha256_file(path)
        for path in log_dir.glob("*.eval")
    }
    bundle_hashes = {
        path.name: sha256_file(path)
        for path in (site_dir / "logs").glob("*.eval")
    }
    if source_hashes != bundle_hashes:
        raise ValueError("Bundled logs differ from the selected source logs")
    docs_dir = (ROOT / "docs").resolve()
    output_path = output_dir.resolve()
    replaced_size = (
        directory_size(output_dir)
        if output_dir.exists() and output_path.is_relative_to(docs_dir)
        else 0
    )
    projected_docs_size = (
        directory_size(docs_dir) - replaced_size + directory_size(site_dir)
    )
    if projected_docs_size >= MAX_PAGES_SITE_BYTES:
        raise ValueError("Projected docs directory exceeds the GitHub Pages site limit")
    if not (ROOT / "docs/.nojekyll").is_file():
        raise ValueError("docs/.nojekyll is missing")


def build_bundle(log_dir: Path, site_dir: Path) -> None:
    """Build and normalize an Inspect static viewer bundle."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "view",
            "bundle",
            "--log-dir",
            str(log_dir),
            "--output-dir",
            str(site_dir),
            "--display",
            "plain",
        ],
        check=True,
    )
    patch_log_directory(site_dir / "index.html")


def publish(site_dir: Path, output_dir: Path) -> None:
    """Replace the hosted numerical viewer with deletion semantics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{site_dir}/", f"{output_dir}/"],
        check=True,
    )


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
        expected_samples=repaired_cell_count(args.repair_cells, config),
        cell_manifest=args.repair_cells,
    )
    validate_logs(config, base_logs, repair_logs)
    logs = base_logs + repair_logs

    with tempfile.TemporaryDirectory(prefix="eigenbench-numerical-view-") as work:
        work_dir = Path(work)
        selected_dir = work_dir / "selected"
        site_dir = work_dir / "site"
        stage_logs(logs, selected_dir)
        build_bundle(selected_dir, site_dir)
        validate_bundle(selected_dir, site_dir, args.output_dir, len(logs))
        publish(site_dir, args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "logs": [log.path.name for log in logs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
