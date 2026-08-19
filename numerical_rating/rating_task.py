"""Shared Inspect task construction for cached-response ratings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ResponseSchema
from inspect_ai.scorer import Scorer
from inspect_ai.solver import generate

from numerical_rating.data import (
    load_config,
    load_constitution_text,
    load_repaired_cell_manifest,
    load_response_cells,
    provenance_path,
    sha256_file,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "You are a careful evaluator. Follow the user's requested JSON schema exactly."
)


def model_slug(name: str) -> str:
    """Return a short stable label for sample IDs."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def model_tag(model: str) -> str:
    """Return the model name without routing/provider prefixes."""
    return model.rsplit("/", 1)[-1]


def render_prompt(*, prompt_path: Path, values: Mapping[str, str]) -> str:
    """Render named fields in a rating prompt."""
    rendered = prompt_path.read_text(encoding="utf-8")
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered


def reject_json_constant(value: str) -> None:
    """Reject non-standard JSON values such as NaN and Infinity."""
    raise ValueError(f"invalid JSON constant: {value}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a judge completion."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped, parse_constant=reject_json_constant)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(
            match.group(0), parse_constant=reject_json_constant
        )

    if not isinstance(parsed, dict):
        raise ValueError("Judge output must be a JSON object")
    return parsed


def build_pointwise_rating_task(
    *,
    config: str,
    prompt_path: Path,
    scorer: Scorer,
    task_name: str,
    run_id: str,
    rating_mode: str,
    prompt_constitution: str | None = None,
    sample_metadata: Mapping[str, Any] | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    version_from_artifacts: bool = False,
    response_schema: ResponseSchema | None = None,
    limit: int | None = None,
    cell_manifest: str | None = None,
    judge_model: str | None = None,
    generation_max_tokens: int | None = None,
) -> Task:
    """Build one cached-response rating task with shared provenance."""
    if limit is not None and cell_manifest is not None:
        raise ValueError("limit and cell_manifest cannot be combined")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    rating_config = load_config(config)
    active_judge_model = judge_model or rating_config.judge_model
    active_max_tokens = (
        generation_max_tokens
        if generation_max_tokens is not None
        else rating_config.generation_max_tokens
    )
    judge_id = next(
        (
            index
            for index, judge in enumerate(rating_config.judges)
            if judge.model == active_judge_model
        ),
        None,
    )
    judge_name = (
        rating_config.judges[judge_id].name
        if judge_id is not None
        else model_tag(active_judge_model)
    )
    constitution_text = load_constitution_text(rating_config.constitution)
    constitution_hash = sha256_text(constitution_text)
    prompt_hash = sha256_text(prompt_path.read_text(encoding="utf-8"))
    cells = load_response_cells(rating_config)
    repair_manifest = None
    if cell_manifest is not None:
        repair_manifest = load_repaired_cell_manifest(
            cell_manifest,
            config=rating_config,
            response_cells=cells,
        )
        cells = list(repair_manifest.cells)
    elif limit is not None:
        if limit > len(cells):
            raise ValueError(
                f"limit {limit} exceeds the {len(cells)} available response cells"
            )
        cells = cells[: int(limit)]

    repair_metadata = (
        {
            "cell_manifest": provenance_path(repair_manifest.path),
            "cell_manifest_hash": repair_manifest.manifest_hash,
        }
        if repair_manifest is not None
        else {}
    )
    extra_metadata = dict(sample_metadata or {})
    extra_task_metadata = dict(task_metadata or {})
    prompt_values = {
        "constitution": prompt_constitution or constitution_text,
    }

    samples = []
    for cell in cells:
        sample_id = (
            f"{rating_config.dataset}_{cell.scenario_index:04d}_"
            f"{cell.model_id}_{model_slug(cell.model_name)}"
        )
        target_model = f"{cell.model_name} ({cell.model_api_id})"
        samples.append(
            Sample(
                id=sample_id,
                input=[
                    ChatMessageSystem(content=SYSTEM_PROMPT),
                    ChatMessageUser(
                        content=render_prompt(
                            prompt_path=prompt_path,
                            values={
                                **prompt_values,
                                "scenario": cell.scenario,
                                "response": cell.response,
                            },
                        )
                    ),
                ],
                target=target_model,
                metadata={
                    "dataset": rating_config.dataset,
                    "scenario_index": cell.scenario_index,
                    "scenario": cell.scenario,
                    "model_id": cell.model_id,
                    "model_name": cell.model_name,
                    "model_api_id": cell.model_api_id,
                    "target_model": cell.model_name,
                    "target_model_api_id": cell.model_api_id,
                    "response_hash": cell.response_hash,
                    "constitution": str(
                        rating_config.constitution.relative_to(ROOT)
                    ),
                    "constitution_name": rating_config.constitution.stem,
                    "constitution_hash": constitution_hash,
                    "constitution_version": constitution_hash,
                    "prompt": str(prompt_path.relative_to(ROOT)),
                    "prompt_hash": prompt_hash,
                    "rating_mode": rating_mode,
                    "judge_id": judge_id,
                    "judge_name": judge_name,
                    "judge_model": active_judge_model,
                    "source": str(rating_config.source.relative_to(ROOT)),
                    "selection": rating_config.selection,
                    "score_min": rating_config.score_min,
                    "score_max": rating_config.score_max,
                    **extra_metadata,
                    **repair_metadata,
                },
            )
        )

    version: str | int = 0
    if version_from_artifacts:
        version = sha256_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "prompt_hash": prompt_hash,
                    "constitution_hash": constitution_hash,
                    "source_hash": sha256_file(rating_config.source),
                    "score_min": rating_config.score_min,
                    "score_max": rating_config.score_max,
                    "generation_max_tokens": active_max_tokens,
                    "generation_temperature": (
                        rating_config.generation_temperature
                    ),
                    "response_schema": (
                        response_schema.model_dump(mode="json")
                        if response_schema is not None
                        else None
                    ),
                },
                sort_keys=True,
            )
        )
    if repair_manifest is not None:
        version = (
            sha256_text(f"{version}|{repair_manifest.manifest_hash}")
            if version_from_artifacts
            else repair_manifest.manifest_hash
        )

    return Task(
        name=task_name,
        version=version,
        dataset=MemoryDataset(samples, name=run_id),
        model=active_judge_model,
        solver=generate(
            max_tokens=active_max_tokens,
            temperature=rating_config.generation_temperature,
            response_schema=response_schema,
        ),
        scorer=scorer,
        metadata={
            "config": provenance_path(config),
            "run_id": run_id,
            "rating_mode": rating_mode,
            "judge_model": active_judge_model,
            "judge_id": judge_id,
            "judge_name": judge_name,
            "response_cells": len(samples),
            "constitution_hash": constitution_hash,
            "constitution_version": constitution_hash,
            "prompt_hash": prompt_hash,
            "generation_max_tokens": active_max_tokens,
            "generation_temperature": rating_config.generation_temperature,
            "response_schema": (
                response_schema.model_dump(mode="json")
                if response_schema is not None
                else None
            ),
            **extra_task_metadata,
            **repair_metadata,
        },
        tags=[f"judge:{model_tag(active_judge_model)}"],
    )
