"""Run all configured judges for criterion-wise numerical ratings."""

from __future__ import annotations

from numerical_rating.data import load_config, repo_path
from numerical_rating.run_criteria import pointwise_criterion_rating
from numerical_rating.run_round_robin import parse_args, run


DEFAULT_LOG_DIR = repo_path(
    "runs/numerical_rating/kindness_1000_criterion_round_robin"
)


def main() -> int:
    """Run or resume the criterion-wise round robin."""
    args = parse_args(default_log_dir=DEFAULT_LOG_DIR)
    if args.generation_max_tokens is None:
        args.generation_max_tokens = load_config(
            args.config
        ).criterion_generation_max_tokens
    return run(args, task_factory=pointwise_criterion_rating)


if __name__ == "__main__":
    raise SystemExit(main())
