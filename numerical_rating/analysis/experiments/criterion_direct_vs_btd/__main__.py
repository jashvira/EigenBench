"""Compare direct rating margins with criterion-conditioned pairwise BTD."""

from __future__ import annotations

import argparse
from pathlib import Path

from .comparison import run_analysis

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RATINGS = (
    ROOT
    / "data/output/numerical_rating/kindness_1000_criterion_round_robin/ratings.csv"
)
DEFAULT_EVALUATIONS = (
    ROOT / "data/output/valuearena/raw/runs/8_models/kindness/evaluations.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "data/output/numerical_rating/kindness_1000_round_robin/"
    "direct_embedding_comparison/criterion_btd_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument(
        "--direct-method",
        choices=("svd", "matched_margin"),
        default="svd",
    )
    return parser.parse_args()


def main() -> None:
    results = run_analysis(parse_args())
    heldout = results["heldout"]
    geometry = results["geometry"]
    print(
        "Matched criterion comparison: "
        f"direct NLL={heldout['direct_nll']['mean']:.6f}, "
        f"BTD NLL={heldout['btd_nll']['mean']:.6f}, "
        f"score r={geometry['score_pearson']:.6f}"
    )


if __name__ == "__main__":
    main()
