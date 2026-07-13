"""Run the configured numerical judges as one resumable Inspect eval set."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set
from inspect_ai.util import AdaptiveConcurrency

from numerical_rating.data import load_config, repo_path
from numerical_rating.run_pointwise import pointwise_constitution_rating


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"
DEFAULT_LOG_DIR = ROOT / "runs/numerical_rating/kindness_1000_round_robin"
PARTIAL_LOG_ADAPTIVE_CONNECTIONS = AdaptiveConcurrency.model_validate("5-20-50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=repo_path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--judge", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--connections-per-judge", type=int, default=4)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-on-error", type=int, default=1)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--generation-max-tokens", type=int)
    parser.add_argument("--cell-manifest", type=repo_path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    selected = [
        judge
        for judge in config.judges
        if not args.judge or judge.name in args.judge or judge.model in args.judge
    ]
    if not selected:
        raise ValueError("No configured judges matched --judge")

    load_dotenv(ROOT / ".env")
    key = os.environ.get("PETRI_OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("PETRI_OPENROUTER_API_KEY is not set")
    os.environ["OPENROUTER_API_KEY"] = key

    tasks = []
    for judge in selected:
        max_tokens = (
            args.generation_max_tokens
            if args.generation_max_tokens is not None
            else judge.max_tokens
        )
        task_kwargs = {"config": args.config, "judge_model": judge.model}
        if max_tokens is not None:
            task_kwargs["generation_max_tokens"] = max_tokens
        if args.cell_manifest is not None:
            task_kwargs["cell_manifest"] = str(args.cell_manifest)
        tasks.append(pointwise_constitution_rating(**task_kwargs))
    success, _ = eval_set(
        tasks=tasks,
        log_dir=str(args.log_dir),
        log_format="eval",
        display="plain",
        max_tasks=min(args.max_tasks, len(tasks)),
        # Each sample makes one judge call, so one cap controls both layers.
        max_samples=args.connections_per_judge,
        max_connections=args.connections_per_judge,
        # Preserve partial-log task identity; max_connections takes precedence.
        adaptive_connections=PARTIAL_LOG_ADAPTIVE_CONNECTIONS,
        max_retries=args.http_retries,
        retry_attempts=args.retry_attempts,
        retry_immediate=True,
        retry_cleanup=False,
        retry_on_error=args.retry_on_error,
        fail_on_error=True,
        continue_on_fail=True,
        timeout=args.timeout,
        log_buffer=10,
        log_dir_allow_dirty=bool(args.judge or args.cell_manifest),
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
