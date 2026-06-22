"""Inspect task for Petri probe discovery from one constitution criterion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_petri import (
    JudgeDimension,
    audit_judge,
    audit_solver,
    audit_viewer,
    auditor_agent,
    auditor_tools,
    target_agent,
)

from petri_multiturn.prompts import AUDITOR_PROMPT, PROBE_QUALITY_RUBRIC


ROOT = Path(__file__).resolve().parents[1]


def sha256_text(text: str) -> str:
    """Return the SHA-256 hash for exact UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash for the constitution file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def criterion_ordinal(criterion_id: str) -> int:
    """Convert IDs like criterion_01 into their 1-based ordinal."""
    match = re.fullmatch(r"criterion_(\d+)", criterion_id)
    if not match:
        raise ValueError("criterion_id must look like criterion_01")
    return int(match.group(1))


def criterion_record(path: Path, criterion_id: str) -> dict[str, str]:
    """Return criterion text and hashes for a constitution criterion."""
    criteria = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(criteria, list):
        raise ValueError(f"{path} must contain a JSON list of criteria")

    ordinal = criterion_ordinal(criterion_id)
    if ordinal < 1 or ordinal > len(criteria):
        raise ValueError(f"criterion_id must select one of {len(criteria)} criteria")

    criterion = str(criteria[ordinal - 1])
    return {
        "criterion_text": criterion,
        "criterion_hash": sha256_text(criterion),
        "constitution_version": sha256_file(path),
    }


def criterion_seed(
    constitution: str,
    criterion_id: str,
    record: dict[str, str],
    scenario: str,
    initial_response: str,
    scenario_id: str,
) -> Sample:
    """Build the single Inspect sample consumed by Petri's auditor."""
    constitution_path = ROOT / constitution
    prompt = AUDITOR_PROMPT.format(
        constitution_name=constitution_path.stem,
        criterion_id=criterion_id,
        criterion=record["criterion_text"],
        scenario=scenario,
        initial_response=initial_response,
    )

    return Sample(
        id=f"{constitution_path.stem}.{criterion_id}.{scenario_id}",
        input=prompt,
        metadata={
            "constitution_path": constitution,
            "constitution_name": constitution_path.stem,
            "constitution_version": record["constitution_version"],
            "criterion_id": criterion_id,
            "criterion_text": record["criterion_text"],
            "criterion_hash": record["criterion_hash"],
            "scenario_id": scenario_id,
            "scenario": scenario,
            "scenario_hash": sha256_text(scenario),
            "initial_response": initial_response,
            "initial_response_hash": sha256_text(initial_response),
        },
    )


def probe_quality_dimension(criterion: str) -> JudgeDimension:
    """Return Petri's judge dimension for deciding whether a probe is useful."""
    return JudgeDimension(
        name="probe_quality",
        description="Quality of the criterion-conditioned Petri probe transcript.",
        tags=["probe_quality"],
        palette="good-high",
        rubric=PROBE_QUALITY_RUBRIC.format(criterion=criterion),
    )


@task
def criterion_audit(
    scenario: str = "",
    initial_response: str = "",
    constitution: str = "data/constitutions/kindness.json",
    criterion_id: str = "criterion_01",
    scenario_id: str = "manual",
    max_turns: int = 6,
) -> Task:
    """Build an Inspect-native Petri task for follow-up probing."""
    if not scenario:
        raise ValueError("scenario is required")
    if not initial_response:
        raise ValueError("initial_response is required")

    constitution_path = ROOT / constitution
    record = criterion_record(constitution_path, criterion_id)
    dimensions = [probe_quality_dimension(record["criterion_text"])]
    return Task(
        dataset=MemoryDataset(
            [
                criterion_seed(
                    constitution,
                    criterion_id,
                    record,
                    scenario,
                    initial_response,
                    scenario_id,
                )
            ],
            name="criterion_probe",
        ),
        solver=audit_solver(
            auditor=auditor_agent(
                max_turns=max_turns,
                tools=auditor_tools(rollback=False, target_tools="none"),
                realism_filter=True,
            ),
            target=target_agent(),
        ),
        scorer=audit_judge(dimensions),
        viewer=audit_viewer(dimensions),
    )
