"""Publish complete direct-rating logs as an Inspect view bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import zipfile_zstd  # noqa: F401  Register zstd compression with zipfile.
from inspect_ai.log import EvalLogInfo

from numerical_rating.analysis.reports.criterion_report import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CRITERION_ANALYSIS,
)
from numerical_rating.analysis.reports.round_robin_report import (
    completed_logs,
    eval_log_path,
)
from numerical_rating.collection.data import (
    ConstitutionCriterion,
    RatingConfig,
    load_config,
    load_constitution_criteria,
    provenance_path,
    repo_path,
    sha256_file,
)
from numerical_rating.collection.run_criteria import (
    CRITERION_PROMPT,
    criterion_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_WHOLE_SOURCE = ROOT / "runs/numerical_rating/kindness_1000_round_robin"
DEFAULT_CRITERION_SOURCE = (
    ROOT / "runs/numerical_rating/kindness_1000_criterion_round_robin"
)
DEFAULT_WHOLE_OUTPUT = ROOT / "docs/numerical_rating"
DEFAULT_CRITERION_OUTPUT = ROOT / "docs/numerical_rating_criteria"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--rating-mode",
        choices=("whole_constitution", "criterion_wise"),
        default="whole_constitution",
    )
    parser.add_argument("--source", type=repo_path)
    parser.add_argument("--output-dir", type=repo_path)
    parser.add_argument("--generation-max-tokens", type=int)
    return parser.parse_args()


def copy_public_log(source: Path, target: Path) -> None:
    """Copy an Inspect log while redacting the local home path."""
    private_home = str(Path.home()).encode()
    with (
        zipfile.ZipFile(source) as source_archive,
        zipfile.ZipFile(target, "w") as target_archive,
    ):
        for entry in source_archive.infolist():
            payload = source_archive.read(entry).replace(private_home, b"<home>")
            target_archive.writestr(entry, payload)


def validate_criterion_analysis(
    logs: list[EvalLogInfo],
    *,
    config: RatingConfig,
    criteria: tuple[ConstitutionCriterion, ...],
) -> None:
    result_path = DEFAULT_CRITERION_ANALYSIS / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Criterion analysis results are missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_criteria = [
        (criterion.criterion_id, criterion.text_hash) for criterion in criteria
    ]
    observed_criteria = [
        (entry.get("criterion_id"), entry.get("criterion_text_hash"))
        for entry in result.get("criteria", [])
    ]
    if (
        result.get("scenario_count") != config.expected_scenarios
        or result.get("judge_count") != len(config.judges)
        or result.get("target_count") != config.expected_models
        or result.get("criterion_count") != len(criteria)
        or observed_criteria != expected_criteria
    ):
        raise ValueError("Criterion analysis shape or provenance is invalid")

    expected_sources = [
        {
            "judge_model": judge.model,
            "path": provenance_path(eval_log_path(log)),
            "sha256": sha256_file(eval_log_path(log)),
        }
        for judge, log in zip(config.judges, logs)
    ]
    if result.get("source_logs") != expected_sources:
        raise ValueError("Criterion analysis does not match the selected logs")


def select_logs(
    args: argparse.Namespace,
    config: RatingConfig,
) -> tuple[list[EvalLogInfo], Path]:
    if args.rating_mode == "whole_constitution":
        logs = completed_logs(
            args.source or DEFAULT_WHOLE_SOURCE,
            config,
            config_name=args.config,
            expected_samples=config.expected_response_cells,
        )
        return logs, args.output_dir or DEFAULT_WHOLE_OUTPUT

    criteria = load_constitution_criteria(config.constitution)
    max_tokens = (
        args.generation_max_tokens
        if args.generation_max_tokens is not None
        else config.criterion_generation_max_tokens
    )
    logs = completed_logs(
        args.source or DEFAULT_CRITERION_SOURCE,
        config,
        config_name=args.config,
        expected_samples=config.expected_response_cells,
        task_name="pointwise_criterion_rating",
        run_id=f"{config.run_id}_criteria",
        prompt_path=CRITERION_PROMPT,
        generation_limits={judge.model: max_tokens for judge in config.judges},
        required_metadata={
            "rating_mode": "criterion_wise",
            "criteria": criterion_provenance(criteria),
            "criterion_count": len(criteria),
        },
    )
    validate_criterion_analysis(logs, config=config, criteria=criteria)
    return logs, args.output_dir or DEFAULT_CRITERION_OUTPUT


def publish_logs(logs: list[EvalLogInfo], output_dir: Path) -> None:
    inspect = Path(sys.executable).with_name("inspect")
    if not inspect.is_file():
        raise FileNotFoundError(f"Inspect CLI is missing: {inspect}")

    with tempfile.TemporaryDirectory(prefix="eigenbench-view-") as work:
        stage = Path(work) / "logs"
        stage.mkdir()
        for log in logs:
            source = eval_log_path(log)
            target = stage / source.name
            copy_public_log(source, target)
            if target.stat().st_size >= MAX_GITHUB_FILE_BYTES:
                raise ValueError(f"Published log exceeds GitHub's limit: {target.name}")
        subprocess.run(
            [
                str(inspect),
                "view",
                "bundle",
                "--log-dir",
                str(stage),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ],
            check=True,
        )

    docs = ROOT / "docs"
    if output_dir == docs or docs in output_dir.parents:
        docs.mkdir(exist_ok=True)
        (docs / ".nojekyll").touch()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logs, output_dir = select_logs(args, config)
    publish_logs(logs, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "logs": [eval_log_path(log).name for log in logs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
