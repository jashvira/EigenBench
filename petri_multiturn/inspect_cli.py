"""Shared Inspect CLI helpers for Petri runner wrappers."""

from __future__ import annotations

from pathlib import Path

import yaml

from petri_multiturn.tags import merge_tags as merge_tag_list, slug


def has_option(args: list[str], name: str) -> bool:
    """Return whether an Inspect CLI option is already present."""
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def task_arg(args: list[str], name: str) -> str | None:
    """Return a `-T name=value` task argument from the CLI args."""
    for i, arg in enumerate(args):
        if arg == "-T" and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith("-T") and "=" in arg:
            value = arg[2:]
        else:
            continue

        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return None


def task_config_path(args: list[str]) -> Path | None:
    """Return the Inspect task-config path from CLI args."""
    for i, arg in enumerate(args):
        if arg == "--task-config" and i + 1 < len(args):
            return Path(args[i + 1])
        if arg.startswith("--task-config="):
            return Path(arg.split("=", 1)[1])
    return None


def task_config(args: list[str]) -> dict[str, object]:
    """Return task arguments loaded from `--task-config`."""
    path = task_config_path(args)
    if path is None:
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a task-argument mapping")
    return data


def task_value(
    args: list[str],
    name: str,
    config: dict[str, object] | None = None,
) -> str | None:
    """Return a task argument from `-T` or `--task-config`."""
    if value := task_arg(args, name):
        return value
    value = (config if config is not None else task_config(args)).get(name)
    return str(value) if value is not None else None


def pop_option(args: list[str], name: str) -> tuple[str | None, list[str]]:
    """Remove a single-value CLI option and return its value."""
    value: str | None = None
    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
            skip_next = True
            continue
        if arg.startswith(f"{name}="):
            value = arg.split("=", 1)[1]
            continue
        out.append(arg)
    return value, out


def model_role(args: list[str], role: str) -> str | None:
    """Return the model configured for an Inspect `--model-role`."""
    for i, arg in enumerate(args):
        if arg == "--model-role" and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith("--model-role="):
            value = arg.split("=", 1)[1]
        else:
            continue

        if value.startswith(f"{role}="):
            return value.split("=", 1)[1]
    return None


def merge_tags(existing: str | None, generated: list[str]) -> str:
    """Merge user tags with generated role/data tags."""
    tags = [tag.strip() for tag in (existing or "").split(",") if tag.strip()]
    return ",".join(merge_tag_list(tags, generated))


def model_slug(model: str | None) -> str:
    """Return the compact model name used in Petri run directories."""
    if not model:
        return "unknown"

    name = model.rsplit("/", 1)[-1]
    if name.startswith("claude-sonnet-"):
        name = f"sonnet{name.removeprefix('claude-sonnet-')}"
    elif name.startswith("claude-opus-"):
        name = f"opus{name.removeprefix('claude-opus-')}"
    return slug(name)
