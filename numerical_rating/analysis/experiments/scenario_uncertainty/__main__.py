"""Scenario-level uncertainty for numerical and pairwise rankings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .uncertainty_analysis import run_analysis

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RATINGS = (
    ROOT / "data/output/numerical_rating/kindness_1000_round_robin/ratings.csv"
)
DEFAULT_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
DEFAULT_META = ROOT / "data/output/valuearena/raw/runs/8_models/kindness/meta.json"
DEFAULT_OUTPUT = (
    ROOT / "data/output/numerical_rating/kindness_1000_round_robin/scenario_uncertainty"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--jackknife-groups", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        ratings_path=args.ratings,
        evaluations_path=args.evaluations,
        meta_path=args.meta,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        jackknife_groups=args.jackknife_groups,
        workers=args.workers,
        seed=args.seed,
    )
    comparison = result["point"]["numerical_vs_pairwise_refit"]
    print(
        f"Matched 871 scenarios: Spearman={comparison['spearman']:.3f}, "
        f"Kendall={comparison['kendall']:.3f}"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
