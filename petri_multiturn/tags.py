"""Compact tags for Petri run logs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path


GENERATED_PREFIXES = (
    "a:",
    "t:",
    "j:",
    "row:",
    "crit:",
    "auditor:",
    "target:",
    "judge:",
    "dataset-row:",
    "criterion:",
)


def slug(value: object) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-")


def tag_component(value: str) -> str:
    return slug(value).replace("-", "_")


def criterion_tag(criterion_id: str) -> str:
    match = re.fullmatch(r"criterion_0*([0-9]+)", criterion_id)
    if match:
        return f"c{int(match.group(1)):02d}"
    return slug(criterion_id)


def canonical_model_id(model: str) -> str:
    parts = [slug(part) for part in model.split("/") if part]
    while parts and parts[0] in {"openrouter"}:
        parts = parts[1:]
    return "/".join(parts) or "unknown"


def model_tag(model: str) -> str:
    return canonical_model_id(model).rsplit("/", 1)[-1]


def run_tags(
    *,
    model_roles: Mapping[str, str],
    scenario_dataset: str,
    scenario_index: str,
    constitution: str,
    criterion_id: str,
) -> list[str]:
    tags = []
    for role, prefix in (("auditor", "a"), ("target", "t"), ("judge", "j")):
        if model := model_roles.get(role):
            tags.append(f"{prefix}:{model_tag(model)}")

    dataset = tag_component(scenario_dataset or "manual")
    index = tag_component(scenario_index or "")
    tags.append(f"row:{dataset}:{index}" if index else f"row:{dataset}")

    constitution_name = tag_component(Path(constitution).stem)
    tags.append(f"crit:{constitution_name}:{criterion_tag(criterion_id)}")
    return tags


def merge_tags(existing: Iterable[str], generated: Iterable[str]) -> list[str]:
    tags = [
        tag
        for tag in existing
        if tag and not tag.startswith(GENERATED_PREFIXES)
    ]
    tags.extend(generated)
    return list(dict.fromkeys(tags))
