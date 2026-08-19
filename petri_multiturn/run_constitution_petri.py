"""Inspect task for full-constitution Petri transcript ratings.

Flow:
1. Load the same scenario slice used by the pointwise numerical-rating run.
2. Render one Petri auditor seed per scenario using the full constitution.
3. Let Petri drive the target conversation and score the transcript.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_petri import (
    JudgeDimension,
    audit_judge,
    audit_solver,
    audit_viewer,
    auditor_agent,
    target_agent,
)
from inspect_petri.util import parse_markdown_file

from numerical_rating.data import (
    load_config,
    load_constitution_text,
    load_response_cells,
    sha256_text,
)
from petri_multiturn.inspect_cli import (
    has_option,
    merge_tags,
    model_role,
    model_slug,
    pop_option,
    task_value,
)
from petri_multiturn.opening_turn import first_turn_generate
from petri_multiturn.runtime import (
    AUDITOR_USER_MESSAGE,
    TARGET_SYSTEM,
    conversation_tools,
)
from petri_multiturn.tags import run_tags, slug


DEFAULT_CONFIG = "numerical_rating/configs/kindness_full8_repaired.yaml"
PETRI_DIR = ROOT / "petri_multiturn"
SEED_TEMPLATE = PETRI_DIR / "seeds" / "full_constitution_probe.md"
SCORE_TEMPLATE = PETRI_DIR / "dimensions" / "whole_constitution_score.md"
INSPECT_TASK = "petri_multiturn/run_constitution_petri.py@full_constitution_audit"


def render_seed(
    *,
    scenario: str,
    constitution_name: str,
    constitution: str,
    target_system: str = TARGET_SYSTEM,
) -> str:
    """Render auditor instructions for one scenario."""
    return parse_markdown_file(SEED_TEMPLATE).body.format(
        scenario=scenario,
        constitution_name=constitution_name,
        constitution=constitution,
        target_system=target_system,
    )


def whole_constitution_dimension(constitution: str) -> JudgeDimension:
    """Create the score column consumed by Petri's judge and viewer."""
    doc = parse_markdown_file(SCORE_TEMPLATE)
    metadata = doc.metadata
    return JudgeDimension(
        name="whole_constitution_score",
        display_name=metadata["display_name"],
        description=metadata["description"],
        tags=metadata["tags"],
        palette=metadata["palette"],
        rubric=f"{doc.body}\n\n<criteria>\n{constitution}\n</criteria>",
    )


def scenario_rows(config_path: str, limit: int | None) -> list[tuple[int, str]]:
    """Return one scenario row per selected AskReddit scenario."""
    config = load_config(config_path)
    cells = load_response_cells(config)
    scenarios = {
        cell.scenario_index: cell.scenario
        for cell in cells
    }
    rows = sorted(scenarios.items())
    if len(rows) != config.expected_scenarios:
        raise ValueError(
            f"Expected {config.expected_scenarios} scenarios, found {len(rows)}"
        )
    return rows[: int(limit)] if limit is not None else rows


def model_tags(args: list[str], config_path: str) -> list[str]:
    """Return compact run tags for the task-level Inspect log."""
    config = load_config(config_path)
    roles = {
        role: model
        for role in ("auditor", "target", "judge")
        if (model := model_role(args, role))
    }
    return [
        "exp:petri-constitution-rating",
        *run_tags(
            model_roles=roles,
            scenario_dataset=config.dataset,
            scenario_index="full8",
            constitution=str(config.constitution),
            criterion_id="constitution",
        ),
    ]


def runner_parser() -> ArgumentParser:
    """Parse options owned by this runner."""
    parser = ArgumentParser(
        description="Run full-constitution Petri transcript ratings.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=1,
        help="number of independent transcripts per scenario",
    )
    parser.add_argument(
        "--parallel-samples",
        type=int,
        default=5,
        help="maximum Inspect samples to run in parallel",
    )
    return parser


def default_log_dir(args: list[str]) -> str:
    """Return the default log directory for one target-model run."""
    config_path = task_value(args, "config") or DEFAULT_CONFIG
    config = load_config(config_path)
    return (
        "runs/petri_multiturn/full_constitution/"
        f"{slug(config.dataset)}_{slug(config.constitution.stem)}/"
        f"a-{model_slug(model_role(args, 'auditor'))}_"
        f"t-{model_slug(model_role(args, 'target'))}_"
        f"j-{model_slug(model_role(args, 'judge'))}/logs"
    )


def inspect_args(args: list[str]) -> list[str]:
    """Add runner defaults before handing off to `inspect eval`."""
    if has_option(args, "--epochs"):
        raise SystemExit("Use --trajectories instead of Inspect --epochs.")
    if has_option(args, "--max-samples"):
        raise SystemExit("Use --parallel-samples instead of Inspect --max-samples.")

    runner_args, out = runner_parser().parse_known_args(args)
    if runner_args.trajectories < 1:
        raise SystemExit("--trajectories must be at least 1")
    if runner_args.parallel_samples < 1:
        raise SystemExit("--parallel-samples must be at least 1")

    out.extend(["--epochs", str(runner_args.trajectories)])
    if not has_option(out, "--max-samples"):
        out.extend(["--max-samples", str(runner_args.parallel_samples)])
    if not has_option(out, "--log-dir"):
        out.extend(["--log-dir", default_log_dir(out)])
    if not has_option(out, "--log-format"):
        out.extend(["--log-format", "json"])

    config_path = task_value(out, "config") or DEFAULT_CONFIG
    existing_tags, out = pop_option(out, "--tags")
    out.extend(["--tags", merge_tags(existing_tags, model_tags(out, config_path))])
    return out


def main() -> None:
    """Run this file as an `inspect eval` wrapper."""
    load_dotenv(ROOT / ".env")
    if petri_key := os.environ.get("PETRI_OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = petri_key

    inspect_bin = os.environ.get("INSPECT_BIN", str(ROOT / ".venv/bin/inspect"))
    command = [inspect_bin, "eval", INSPECT_TASK, *inspect_args(sys.argv[1:])]
    os.execvp(inspect_bin, command)


@task
def full_constitution_audit(
    config: str = DEFAULT_CONFIG,
    limit: int | None = None,
    max_turns: int = 6,
) -> Task:
    """Build the full-constitution Petri transcript-rating task."""
    rating_config = load_config(config)
    constitution_text = load_constitution_text(rating_config.constitution)
    constitution_hash = sha256_text(constitution_text)
    prompt_hash = sha256_text(SCORE_TEMPLATE.read_text(encoding="utf-8"))

    samples: list[Sample] = []
    for scenario_index, scenario in scenario_rows(config, limit):
        seed = render_seed(
            scenario=scenario,
            constitution_name=rating_config.constitution.stem,
            constitution=constitution_text,
        )
        samples.append(
            Sample(
                id=f"{rating_config.dataset}_{scenario_index:04d}",
                input=seed,
                metadata={
                    "dataset": rating_config.dataset,
                    "scenario_index": scenario_index,
                    "scenario": scenario,
                    "target_system": TARGET_SYSTEM,
                    "constitution": str(rating_config.constitution.relative_to(ROOT)),
                    "constitution_name": rating_config.constitution.stem,
                    "constitution_hash": constitution_hash,
                    "score_dimension": str(SCORE_TEMPLATE.relative_to(ROOT)),
                    "score_dimension_hash": prompt_hash,
                    "source": str(rating_config.source.relative_to(ROOT)),
                    "selection": rating_config.selection,
                    "score_min": rating_config.score_min,
                    "score_max": rating_config.score_max,
                },
            )
        )

    dimension = whole_constitution_dimension(constitution_text)
    return Task(
        name=f"petri_{rating_config.dataset}_{rating_config.constitution.stem}",
        dataset=MemoryDataset(samples, name=f"{rating_config.dataset}_scenarios"),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=max_turns,
                tools=conversation_tools(),
                realism_filter=False,
                target_display_name="target AI",
                user_message=AUDITOR_USER_MESSAGE,
                generate=first_turn_generate(keep_system_message_tool=False),
            ),
            target=target_agent(),
        ),
        scorer=audit_judge([dimension]),
        viewer=audit_viewer([dimension]),
        metadata={
            "config": config,
            "run_id": rating_config.run_id,
            "scenario_count": len(samples),
            "constitution_hash": constitution_hash,
            "score_dimension_hash": prompt_hash,
            "max_turns": max_turns,
        },
    )


if __name__ == "__main__":
    main()
