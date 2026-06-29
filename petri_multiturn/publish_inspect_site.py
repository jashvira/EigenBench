"""Publish selected Petri logs through Inspect's static viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petri_multiturn.tags import merge_tags, run_tags


DEFAULT_SOURCE = ROOT / "runs" / "petri_multiturn"
OUTPUT_DIR = ROOT / "docs"
SITE_LOG_DIR = "petri_logs"


@dataclass(frozen=True)
class EvalLog:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def task(self) -> str | None:
        eval_info = self.data.get("eval") or {}
        return eval_info.get("task")

    @property
    def status(self) -> str | None:
        return self.data.get("status")

    @property
    def started_at(self) -> str:
        stats = self.data.get("stats") or {}
        return str(stats.get("started_at") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GitHub Pages Inspect viewer from local Petri logs.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help=(
            "Log file or directory to scan. Can be repeated. Defaults to "
            "runs/petri_multiturn."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Only publish logs whose Inspect task name matches this value.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Publish every successful task. Without this, at least one --task is required.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def candidate_files(sources: list[Path]) -> list[Path]:
    paths = sources or [DEFAULT_SOURCE]
    files: list[Path] = []
    for source in paths:
        source = source.resolve()
        if source.is_file() and source.suffix == ".json":
            if source.name != "listing.json":
                files.append(source)
        elif source.is_dir():
            files.extend(
                path
                for path in source.rglob("*.json")
                if path.name != "listing.json"
            )
        else:
            raise SystemExit(f"Source does not exist: {source}")
    return sorted(set(files))


def select_logs(
    sources: list[Path],
    *,
    tasks: list[str],
) -> list[EvalLog]:
    selected: list[EvalLog] = []
    task_filter = set(tasks)
    for path in candidate_files(sources):
        try:
            log = EvalLog(path=path, data=load_json(path))
        except Exception as exc:
            raise SystemExit(f"Could not parse {path}: {exc}") from exc

        if log.status != "success":
            continue
        if task_filter and log.task not in task_filter:
            continue
        selected.append(log)

    if not selected:
        raise SystemExit("No logs matched the requested filters.")

    names = [log.name for log in selected]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(
            "Duplicate log basenames would collide in the static bundle: "
            + ", ".join(duplicates)
        )

    return sorted(selected, key=lambda log: (log.started_at, log.name), reverse=True)


def run_inspect_bundle(log_dir: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "inspect_ai",
        "view",
        "bundle",
        "--log-dir",
        str(log_dir),
        "--output-dir",
        str(output_dir),
        "--overwrite",
        "--display",
        "plain",
    ]
    subprocess.run(cmd, check=True)


def primary_metric(results: dict[str, Any]) -> dict[str, Any] | None:
    scores = results.get("scores") or []
    if not scores:
        return None
    metrics = scores[0].get("metrics") or {}
    if not metrics:
        return None
    return next(iter(metrics.values()))


def model_roles(eval_info: dict[str, Any]) -> dict[str, str] | None:
    roles = eval_info.get("model_roles") or {}
    mapped = {
        role: cfg.get("model") if isinstance(cfg, dict) else str(cfg)
        for role, cfg in roles.items()
    }
    return mapped or None


def log_tags(log: dict[str, Any]) -> list[str]:
    eval_info = log.get("eval") or {}
    task_args = eval_info.get("task_args_passed") or eval_info.get("task_args") or {}
    roles = model_roles(eval_info) or {}
    generated = run_tags(
        model_roles=roles,
        scenario_dataset=str(task_args.get("scenario_dataset") or "manual"),
        scenario_index=str(task_args.get("scenario_index") or ""),
        constitution=str(task_args.get("constitution") or ""),
        criterion_id=str(task_args.get("criterion_id") or "criterion_01"),
    )
    return merge_tags(log.get("tags") or [], generated)


def write_log_with_tags(log: EvalLog, target_dir: Path) -> None:
    data = dict(log.data)
    eval_info = dict(data.get("eval") or {})
    tags = log_tags(data)
    eval_info["tags"] = tags
    data["eval"] = eval_info
    data["tags"] = tags
    (target_dir / log.name).write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_selected_logs(logs: list[EvalLog], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for log in logs:
        write_log_with_tags(log, target_dir)


def enriched_listing(log_dir: Path) -> dict[str, dict[str, Any]]:
    listing: dict[str, dict[str, Any]] = {}
    logs = sorted(
        (path for path in log_dir.glob("*.json") if path.name != "listing.json"),
        reverse=True,
    )
    for path in logs:
        log = load_json(path)
        eval_info = log.get("eval") or {}
        results = log.get("results") or {}
        stats = log.get("stats") or {}
        listing[path.name] = {
            "eval_id": eval_info.get("eval_id"),
            "run_id": eval_info.get("run_id"),
            "task": eval_info.get("task"),
            "task_id": eval_info.get("task_id"),
            "task_version": eval_info.get("task_version"),
            "version": log.get("version"),
            "status": log.get("status"),
            "invalidated": log.get("invalidated"),
            "error": log.get("error"),
            "model": eval_info.get("model"),
            "model_roles": model_roles(eval_info),
            "started_at": stats.get("started_at"),
            "completed_at": stats.get("completed_at"),
            "primary_metric": primary_metric(results),
            "eval": eval_info,
            "results": results,
            "stats": stats,
            "tags": log.get("tags"),
        }
    return listing


def write_enriched_listing(log_dir: Path) -> None:
    manifest = enriched_listing(log_dir)
    (log_dir / "listing.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one {label} anchor; found {count}.")
    return text.replace(old, new, 1)


def patch_viewer_metadata(asset_path: Path) -> None:
    text = asset_path.read_text(encoding="utf-8")
    row_start = "var buildLogListRow = (item, details) => {"
    row_end = "\n\treturn row;\n};"
    start = text.find(row_start)
    if start == -1:
        raise SystemExit("Could not find Inspect buildLogListRow function.")
    end = text.find(row_end, start)
    if end == -1:
        raise SystemExit("Could not find Inspect buildLogListRow return.")
    end += len(row_end)

    before = text[:start]
    row_text = text[start:end]
    after = text[end:]

    row_text = replace_once(
        row_text,
        'const preview = item.type === "file" ? item.logPreview : void 0;\n'
        "\tlet totalTokens;",
        'const preview = item.type === "file" ? item.logPreview : void 0;\n'
        "\tconst summary = preview?.stats || preview?.results || preview?.eval ? preview : details;\n"
        "\tlet totalTokens;",
        "summary insertion",
    )
    replacements = {
        "details?.stats?.model_usage": "summary?.stats?.model_usage",
        "Object.values(details.stats.model_usage)": "Object.values(summary.stats.model_usage)",
        "details?.stats?.started_at": "summary?.stats?.started_at",
        "details?.stats?.completed_at": "summary?.stats?.completed_at",
        "new Date(details.stats.started_at)": "new Date(summary.stats.started_at)",
        "new Date(details.stats.completed_at)": "new Date(summary.stats.completed_at)",
        "details?.eval?.task_args_passed": "summary?.eval?.task_args_passed",
        "details?.eval?.task_args": "summary?.eval?.task_args",
        "details?.results?.total_samples": "summary?.results?.total_samples",
        "details?.results?.completed_samples": "summary?.results?.completed_samples",
        "details?.eval?.sandbox?.type": "summary?.eval?.sandbox?.type",
        "details?.eval?.task_file": "summary?.eval?.task_file",
        "tags: details?.tags": "tags: summary?.tags",
        "details?.error?.message": "summary?.error?.message",
        "if (details?.results?.scores)": "if (summary?.results?.scores)",
        "for (const evalScore of details.results.scores)": "for (const evalScore of summary.results.scores)",
    }
    for old, new in replacements.items():
        if old not in row_text:
            raise SystemExit(f"Missing generated viewer anchor: {old}")
        row_text = row_text.replace(old, new)
    asset_path.write_text(before + row_text + after, encoding="utf-8")


def patch_viewer_columns(asset_path: Path) -> None:
    text = asset_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "\t\tconst hidden = /* @__PURE__ */ new Set();\n"
        '\t\tif (mode === "tasks") {',
        "\t\tconst hidden = /* @__PURE__ */ new Set();\n"
        '\t\thidden.add("task");\n'
        '\t\thidden.add("taskArgs");\n'
        '\t\tif (mode === "tasks") {',
        "default hidden columns",
    )
    asset_path.write_text(text, encoding="utf-8")


def patch_index_html(index_path: Path, asset_hash: str) -> None:
    text = index_path.read_text(encoding="utf-8")
    context = json.dumps({"log_dir": SITE_LOG_DIR, "abs_log_dir": SITE_LOG_DIR})
    text = re.sub(
        r'<script id="log_dir_context" type="application/json">.*?</script>',
        f'<script id="log_dir_context" type="application/json">{context}</script>',
        text,
        count=1,
    )
    if f"id=\"log_dir_context\"" not in text:
        raise SystemExit("Could not find Inspect log_dir_context script.")

    text = re.sub(
        r'src="\.\/assets\/index\.js(?:\?v=[^"]*)?"',
        f'src="./assets/index.js?v={asset_hash}"',
        text,
        count=1,
    )
    index_path.write_text(text, encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def publish(logs: list[EvalLog]) -> None:
    with tempfile.TemporaryDirectory(prefix="petri-site-") as tmp:
        tmp_path = Path(tmp)
        staging_logs = tmp_path / "selected_logs"
        bundle_out = tmp_path / "bundle"

        copy_selected_logs(logs, staging_logs)
        run_inspect_bundle(staging_logs, bundle_out)

        bundled_logs = bundle_out / "logs"
        site_logs = bundle_out / SITE_LOG_DIR
        bundled_logs.rename(site_logs)

        write_enriched_listing(site_logs)
        (site_logs / "eval-set.json").write_text("{}\n", encoding="utf-8")
        (site_logs / "flow.yaml").write_text("", encoding="utf-8")
        patch_viewer_metadata(bundle_out / "assets" / "index.js")
        patch_viewer_columns(bundle_out / "assets" / "index.js")

        (bundle_out / ".nojekyll").touch()
        asset_hash = file_hash(bundle_out / "assets" / "index.js")
        patch_index_html(bundle_out / "index.html", asset_hash)

        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        shutil.copytree(bundle_out, OUTPUT_DIR)


def token_total(log: dict[str, Any]) -> int:
    usage = (log.get("stats") or {}).get("model_usage") or {}
    return sum((item or {}).get("total_tokens", 0) or 0 for item in usage.values())


def verify_output() -> dict[str, Any]:
    log_dir = OUTPUT_DIR / SITE_LOG_DIR
    listing = load_json(log_dir / "listing.json")
    failures: list[str] = []
    required_tag_prefixes = (
        "a:",
        "t:",
        "j:",
        "row:",
        "crit:",
    )
    tasks: set[str] = set()
    trajectories = 0
    tokens = 0

    for name, header in listing.items():
        log_path = log_dir / name
        if not log_path.exists():
            failures.append(f"Missing log file listed in manifest: {name}")
            continue
        raw = load_json(log_path)
        raw_samples = len(raw.get("samples") or [])
        listed_samples = (header.get("results") or {}).get("total_samples")
        raw_tokens = token_total(raw)
        listed_tokens = token_total(header)
        if raw_samples != listed_samples:
            failures.append(f"{name}: samples {listed_samples} != {raw_samples}")
        if raw_tokens != listed_tokens:
            failures.append(f"{name}: tokens {listed_tokens} != {raw_tokens}")
        tags = header.get("tags") or []
        for prefix in required_tag_prefixes:
            if not any(tag.startswith(prefix) for tag in tags):
                failures.append(f"{name}: missing {prefix} tag")
        task = (header.get("eval") or {}).get("task") or header.get("task")
        if task:
            tasks.add(str(task))
        trajectories += listed_samples or 0
        tokens += listed_tokens

    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(failures))
    return {
        "runs": len(listing),
        "trajectories": trajectories,
        "tokens": tokens,
        "tasks": sorted(tasks),
    }


def main() -> None:
    args = parse_args()
    if not args.task and not args.all_tasks:
        raise SystemExit("Pass --task TASK to publish selected runs, or --all-tasks.")
    logs = select_logs(args.source, tasks=args.task)
    publish(logs)
    summary = verify_output()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
