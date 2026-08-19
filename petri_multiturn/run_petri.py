"""Inspect task for Petri criterion probes.

Flow:
1. Load one constitution criterion.
2. Render auditor seed instructions from the scenario and criterion.
3. Wire Petri's runner, target, judge, and viewer.
"""

import json
import os
import re
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
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

from petri_multiturn.inspect_cli import (
    has_option,
    merge_tags,
    model_role,
    model_slug,
    pop_option,
    task_config,
    task_value,
)
from petri_multiturn.opening_turn import first_turn_generate, task_text
from petri_multiturn.runtime import (
    AUDITOR_USER_MESSAGE,
    TARGET_SYSTEM,
    conversation_tools,
)
from petri_multiturn.tags import (
    criterion_tag,
    run_tags,
    slug,
)


PETRI_DIR = ROOT / "petri_multiturn"
SEED_TEMPLATE = PETRI_DIR / "seeds" / "criterion_probe.md"
PROBE_QUALITY_TEMPLATE = PETRI_DIR / "dimensions" / "probe_quality.md"
INSPECT_TASK = "petri_multiturn/run_petri.py@criterion_audit"


def load_criterion(path: Path, criterion_id: str) -> str:
    """Return the selected criterion from a plain-list constitution file."""
    match = re.fullmatch(r"criterion_(\d+)", criterion_id)
    if not match:
        raise ValueError("criterion_id must look like criterion_01")

    criteria = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(criteria, list):
        raise ValueError(f"{path} must contain a JSON list of criteria")

    index = int(match.group(1)) - 1
    if index < 0 or index >= len(criteria):
        raise ValueError(f"criterion_id must select one of {len(criteria)} criteria")

    return str(criteria[index])


def render_seed(
    *,
    scenario: str,
    constitution_name: str,
    criterion_id: str,
    criterion: str,
    target_system: str = TARGET_SYSTEM,
) -> str:
    """Render the auditor instructions for this probe."""
    return parse_markdown_file(SEED_TEMPLATE).body.format(
        scenario=scenario,
        constitution_name=constitution_name,
        criterion_id=criterion_id,
        criterion=criterion,
        target_system=target_system,
    )


def probe_quality_dimension(criterion_id: str, criterion: str) -> JudgeDimension:
    """Return a criterion-conditioned Petri judge dimension."""
    doc = parse_markdown_file(PROBE_QUALITY_TEMPLATE)
    metadata = doc.metadata
    rubric_extra = (
        f"Selected criterion ID: {criterion_id}\n"
        f"Selected criterion text:\n{criterion}\n\n"
        "Score transcript quality for this criterion-conditioned probe."
    )

    return JudgeDimension(
        name="probe_quality",
        display_name=metadata["display_name"],
        description=metadata["description"],
        tags=metadata["tags"],
        palette=metadata["palette"],
        rubric=f"{doc.body}\n\n{rubric_extra}",
    )


def reject_sampling_options(args: list[str]) -> None:
    """Keep Inspect sampling internals out of the runner interface."""
    if has_option(args, "--epochs"):
        raise SystemExit("Use --trajectories instead of Inspect --epochs.")
    if has_option(args, "--max-samples"):
        raise SystemExit(
            "Use --parallel-trajectories instead of Inspect --max-samples."
        )


def generated_tags(args: list[str]) -> list[str]:
    config = task_config(args)
    constitution = (
        task_value(args, "constitution", config) or "data/constitutions/kindness.json"
    )
    roles = {
        role: model
        for role in ("auditor", "target", "judge")
        if (model := model_role(args, role))
    }
    return run_tags(
        model_roles=roles,
        scenario_dataset=task_value(args, "scenario_dataset", config) or "manual",
        scenario_index=task_value(args, "scenario_index", config) or "",
        constitution=constitution,
        criterion_id=task_value(args, "criterion_id", config) or "criterion_01",
    )


def criterion_slug(criterion_id: str) -> str:
    """Return the compact criterion name used in Petri run directories."""
    return criterion_tag(criterion_id)


def name_part(value: str) -> str:
    """Return a task-name-safe component."""
    return slug(value).replace("-", "_")


def task_name(
    *,
    scenario_dataset: str,
    scenario_index: str,
    constitution_name: str,
    criterion_id: str,
) -> str:
    """Return the Inspect task name for this scenario and criterion."""
    parts = [name_part(scenario_dataset or "manual")]
    if scenario_index:
        parts.append(name_part(scenario_index))
    parts.extend([name_part(constitution_name), criterion_slug(criterion_id)])
    return "_".join(parts)


def runner_parser() -> ArgumentParser:
    """Return the CLI parser for options owned by this wrapper."""
    parser = ArgumentParser(
        description="Run a Petri criterion probe.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=5,
        help="number of independent transcripts to generate",
    )
    parser.add_argument(
        "--parallel-trajectories",
        type=int,
        default=None,
        help="number of transcripts to run concurrently",
    )
    return parser


def default_log_dir(args: list[str]) -> str:
    """Return the default log directory implied by CLI task/model args."""
    config = task_config(args)
    constitution = (
        task_value(args, "constitution", config) or "data/constitutions/kindness.json"
    )
    criterion_id = task_value(args, "criterion_id", config) or "criterion_01"
    return (
        "runs/petri_multiturn/"
        f"{slug(Path(constitution).stem)}_{criterion_slug(criterion_id)}/"
        f"a-{model_slug(model_role(args, 'auditor'))}_"
        f"t-{model_slug(model_role(args, 'target'))}_"
        f"j-{model_slug(model_role(args, 'judge'))}/logs"
    )


def inspect_args(args: list[str]) -> list[str]:
    """Add Petri defaults to Inspect CLI args that did not specify them."""
    reject_sampling_options(args)
    runner_args, out = runner_parser().parse_known_args(args)
    if runner_args.trajectories < 1:
        raise SystemExit("--trajectories must be at least 1")
    if runner_args.parallel_trajectories is None:
        runner_args.parallel_trajectories = min(5, runner_args.trajectories)
    if runner_args.parallel_trajectories < 1:
        raise SystemExit("--parallel-trajectories must be at least 1")
    if runner_args.parallel_trajectories > runner_args.trajectories:
        raise SystemExit("--parallel-trajectories cannot exceed --trajectories")

    out.extend(
        [
            "--epochs",
            str(runner_args.trajectories),
            "--max-samples",
            str(runner_args.parallel_trajectories),
        ]
    )
    if not has_option(out, "--log-dir"):
        out.extend(["--log-dir", default_log_dir(out)])
    if not has_option(out, "--log-format"):
        out.extend(["--log-format", "json"])
    existing_tags, out = pop_option(out, "--tags")
    out.extend(["--tags", merge_tags(existing_tags, generated_tags(out))])
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
def criterion_audit(
    scenario: str | Sequence[str] = "",
    scenario_dataset: str = "manual",
    scenario_index: str = "",
    constitution: str = "data/constitutions/kindness.json",
    criterion_id: str = "criterion_01",
    max_turns: int = 6,
) -> Task:
    """Build the Petri criterion-probe task."""
    scenario_text = task_text(scenario).strip()
    if not scenario_text:
        raise ValueError("scenario is required")

    constitution_path = ROOT / constitution
    criterion = load_criterion(constitution_path, criterion_id)
    seed = render_seed(
        scenario=scenario_text,
        constitution_name=constitution_path.stem,
        criterion_id=criterion_id,
        criterion=criterion,
    )

    dimension = probe_quality_dimension(criterion_id, criterion)
    metadata = {
        "scenario": scenario_text,
        "scenario_dataset": scenario_dataset,
        "scenario_index": scenario_index,
        "target_system": TARGET_SYSTEM,
        "constitution": constitution,
        "constitution_name": constitution_path.stem,
        "criterion_id": criterion_id,
        "criterion_text": criterion,
    }

    return Task(
        name=task_name(
            scenario_dataset=scenario_dataset,
            scenario_index=scenario_index,
            constitution_name=constitution_path.stem,
            criterion_id=criterion_id,
        ),
        dataset=MemoryDataset([Sample(input=seed, metadata=metadata)], name="seeds"),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=max_turns,
                tools=conversation_tools(),
                realism_filter=False,
                target_display_name="target AI",
                user_message=AUDITOR_USER_MESSAGE,
                generate=first_turn_generate(keep_system_message_tool=True),
            ),
            target=target_agent(),
        ),
        scorer=audit_judge([dimension]),
        viewer=audit_viewer([dimension]),
    )


if __name__ == "__main__":
    main()
