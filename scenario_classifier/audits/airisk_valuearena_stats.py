"""Summarize the ValueArena rows that truly match local AIRiskDilemmas.

The processed ValueArena table has some rows labelled as AIRiskDilemmas whose
scenario text is actually from Reddit. This script keeps only rows whose
scenario text appears in the local AIRiskDilemmas scenario file.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scenario_classifier.core.io import DEFAULT_AIRISK_SCENARIOS, OUTPUT_ROOT, REPO_ROOT, VALUEARENA_ROOT


DEFAULT_INPUT = VALUEARENA_ROOT / "processed" / "scenario_criterion_tie_rates.csv"
DEFAULT_INPUT_SCENARIOS = DEFAULT_AIRISK_SCENARIOS
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "airiskdilemmas_623_stats"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_scenarios(path: Path) -> list[str]:
    """Load the local AIRiskDilemmas scenario list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return payload


def int_field(row: dict[str, str], key: str) -> int:
    """Parse integer-ish CSV fields."""
    return int(float(row[key]))


def float_field(row: dict[str, str], key: str) -> float:
    """Parse float CSV fields."""
    return float(row[key])


def quantiles(values: list[float]) -> dict[str, float]:
    """Return compact quantiles for a numeric list."""
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return round(ordered[round((len(ordered) - 1) * q)], 6)

    return {
        "min": round(ordered[0], 6),
        "p10": pick(0.10),
        "median": pick(0.50),
        "p90": pick(0.90),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.mean(ordered), 6),
    }


def clean_row(row: dict[str, str]) -> dict[str, Any]:
    """Normalize numeric fields and add derived behavioral stats."""
    eval1_wins = int_field(row, "eval1_win_count")
    eval2_wins = int_field(row, "eval2_win_count")
    non_tie_count = eval1_wins + eval2_wins
    strict_choice_rate = float_field(row, "strict_choice_rate")
    winner_margin = abs(eval1_wins - eval2_wins) / max(non_tie_count, 1)
    return {
        "run_id": row["run_id"],
        "constitution": row["constitution"],
        "dataset_range": row["dataset_range"],
        "scenario_index": int_field(row, "scenario_index"),
        "criterion_index": int_field(row, "criterion_index"),
        "scenario": row["scenario"],
        "n_choices": int_field(row, "n_choices"),
        "tie_count": int_field(row, "tie_count"),
        "non_tie_count": non_tie_count,
        "eval1_win_count": eval1_wins,
        "eval2_win_count": eval2_wins,
        "tie_rate": round(float_field(row, "tie_rate"), 6),
        "strict_choice_rate": round(strict_choice_rate, 6),
        "winner_margin": round(winner_margin, 6),
        "behavioral_signal": round(strict_choice_rate * winner_margin, 6),
    }


def filter_airisk_rows(rows: list[dict[str, str]], scenario_texts: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep rows labelled AIRisk whose scenario text appears in local AIRisk."""
    dataset_rows = [row for row in rows if "airiskdilemmas" in row["dataset_range"].lower()]
    matched = [clean_row(row) for row in dataset_rows if row["scenario"] in scenario_texts]
    return matched, {
        "source_rows": len(rows),
        "dataset_string_airisk_rows": len(dataset_rows),
        "text_matched_airisk_rows": len(matched),
        "dataset_string_airisk_unique_scenarios": len({row["scenario"] for row in dataset_rows}),
        "text_matched_airisk_unique_scenarios": len({row["scenario"] for row in matched}),
    }


def criterion_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize choice/tie behavior by criterion."""
    by_criterion: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_criterion[int(row["criterion_index"])].append(row)

    stats = []
    for criterion_index, items in sorted(by_criterion.items()):
        tie_rates = [float(row["tie_rate"]) for row in items]
        n_choices = [int(row["n_choices"]) for row in items]
        margins = [float(row["winner_margin"]) for row in items]
        stats.append(
            {
                "criterion_index": criterion_index,
                "scenario_count": len(items),
                "total_choices": sum(n_choices),
                "n_choices_mean": round(statistics.mean(n_choices), 3),
                "n_choices_median": round(statistics.median(n_choices), 3),
                "tie_rate_mean": round(statistics.mean(tie_rates), 6),
                "tie_rate_median": round(statistics.median(tie_rates), 6),
                "winner_margin_mean": round(statistics.mean(margins), 6),
                "low_tie_rows_le_0_25": sum(rate <= 0.25 for rate in tie_rates),
                "high_tie_rows_ge_0_80": sum(rate >= 0.80 for rate in tie_rates),
                "rows_n_choices_ge_30": sum(count >= 30 for count in n_choices),
                "strong_signal_rows": sum(
                    int(row["n_choices"]) >= 30
                    and float(row["tie_rate"]) <= 0.25
                    and float(row["winner_margin"]) >= 0.20
                    for row in items
                ),
            }
        )
    return stats


def scenario_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the eight criterion rows into one row per scenario."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario"])].append(row)

    stats = []
    for scenario, items in grouped.items():
        criteria = sorted(int(row["criterion_index"]) for row in items)
        tie_rates = [float(row["tie_rate"]) for row in items]
        n_choices = [int(row["n_choices"]) for row in items]
        signals = [float(row["behavioral_signal"]) for row in items]
        strongest = max(items, key=lambda row: float(row["behavioral_signal"]))
        strong_rows = [
            row
            for row in items
            if int(row["n_choices"]) >= 30
            and float(row["tie_rate"]) <= 0.25
            and float(row["winner_margin"]) >= 0.20
        ]
        stats.append(
            {
                "scenario_index": min(int(row["scenario_index"]) for row in items),
                "scenario": scenario,
                "criterion_rows": len(items),
                "criterion_indices": ";".join(str(index) for index in criteria),
                "total_choices": sum(n_choices),
                "mean_n_choices": round(statistics.mean(n_choices), 3),
                "mean_tie_rate": round(statistics.mean(tie_rates), 6),
                "min_tie_rate": round(min(tie_rates), 6),
                "max_behavioral_signal": round(max(signals), 6),
                "strong_signal_count": len(strong_rows),
                "strong_signal_criteria": ";".join(str(row["criterion_index"]) for row in strong_rows),
                "strongest_criterion": strongest["criterion_index"],
                "strongest_tie_rate": strongest["tie_rate"],
                "strongest_winner_margin": strongest["winner_margin"],
                "strongest_signal": strongest["behavioral_signal"],
            }
        )
    return sorted(
        stats,
        key=lambda row: (
            -int(row["strong_signal_count"]),
            -float(row["max_behavioral_signal"]),
            -int(row["total_choices"]),
            int(row["scenario_index"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_summary(
    *,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    filter_counts: dict[str, int],
    criterion_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a compact top-level stats summary."""
    tie_rates = [float(row["tie_rate"]) for row in rows]
    n_choices = [int(row["n_choices"]) for row in rows]
    scenario_criterion_counts = Counter(row["criterion_rows"] for row in scenario_rows)
    return {
        "input": str(args.input.relative_to(REPO_ROOT)),
        "scenario_file": str(args.scenarios.relative_to(REPO_ROOT)),
        "local_airisk_scenario_count": len(scenarios),
        **filter_counts,
        "coverage_of_local_airisk_scenarios": round(filter_counts["text_matched_airisk_unique_scenarios"] / len(scenarios), 6),
        "constitution_counts": dict(sorted(Counter(row["constitution"] for row in rows).items())),
        "criterion_indices": sorted({int(row["criterion_index"]) for row in rows}),
        "criterion_rows_per_scenario": dict(sorted(scenario_criterion_counts.items())),
        "n_choices": quantiles([float(value) for value in n_choices]),
        "tie_rate": quantiles(tie_rates),
        "total_choices": sum(n_choices),
        "low_tie_rows_le_0_25": sum(row["tie_rate"] <= 0.25 for row in rows),
        "high_tie_rows_ge_0_80": sum(row["tie_rate"] >= 0.80 for row in rows),
        "rows_n_choices_ge_30": sum(row["n_choices"] >= 30 for row in rows),
        "strong_signal_rows": sum(row["strong_signal_rows"] for row in criterion_rows),
        "strong_signal_scenarios": sum(row["strong_signal_count"] > 0 for row in scenario_rows),
        "top_scenarios_by_strong_signal": scenario_rows[:15],
    }


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    """Write a short human-readable stats note."""
    body = f"""# AIRiskDilemmas ValueArena 623 Stats

This folder contains stats for processed ValueArena rows that both:

1. have `dataset_range` containing `airiskdilemmas`
2. have scenario text present in `scenario_classifier/data/scenarios/airiskdilemmas.json`

This avoids Reddit-style rows that were present under the AIRisk dataset label.

## Headline

- Local AIRiskDilemmas scenarios: `{summary["local_airisk_scenario_count"]}`
- ValueArena text-matched AIRisk scenarios: `{summary["text_matched_airisk_unique_scenarios"]}`
- Criterion rows: `{summary["text_matched_airisk_rows"]}`
- Coverage: `{summary["coverage_of_local_airisk_scenarios"]:.1%}`
- Constitution: `{", ".join(summary["constitution_counts"].keys())}`
- Criterion indices: `{summary["criterion_indices"]}`

## Signal

- Low-tie rows, `tie_rate <= 0.25`: `{summary["low_tie_rows_le_0_25"]}`
- High-tie rows, `tie_rate >= 0.80`: `{summary["high_tie_rows_ge_0_80"]}`
- Rows with `n_choices >= 30`: `{summary["rows_n_choices_ge_30"]}`
- Strong signal rows, `n_choices >= 30`, `tie_rate <= 0.25`, `winner_margin >= 0.20`: `{summary["strong_signal_rows"]}`
- Strong signal scenarios: `{summary["strong_signal_scenarios"]}`

See `summary.json`, `criterion_stats.csv`, and `scenario_stats.csv`.
"""
    path.write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_INPUT_SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Write AIRisk-matched ValueArena stats."""
    args = parse_args()
    args.input = args.input.resolve()
    args.scenarios = args.scenarios.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = read_scenarios(args.scenarios)
    rows, filter_counts = filter_airisk_rows(read_csv(args.input), set(scenarios))
    if not rows:
        raise SystemExit("No AIRiskDilemmas rows matched the local scenario file.")

    c_stats = criterion_stats(rows)
    s_stats = scenario_stats(rows)
    summary = build_summary(
        rows=rows,
        scenarios=scenarios,
        filter_counts=filter_counts,
        criterion_rows=c_stats,
        scenario_rows=s_stats,
        args=args,
    )

    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "criterion_stats.csv", c_stats)
    write_csv(args.output_dir / "scenario_stats.csv", s_stats)
    write_readme(args.output_dir / "README.md", summary)

    print(f"local_airisk_scenarios: {summary['local_airisk_scenario_count']}")
    print(f"matched_scenarios: {summary['text_matched_airisk_unique_scenarios']}")
    print(f"criterion_rows: {summary['text_matched_airisk_rows']}")
    print(f"coverage: {summary['coverage_of_local_airisk_scenarios']:.1%}")
    print(f"strong_signal_rows: {summary['strong_signal_rows']}")
    print(f"strong_signal_scenarios: {summary['strong_signal_scenarios']}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
