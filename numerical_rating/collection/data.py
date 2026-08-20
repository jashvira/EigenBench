"""Load reused ValueArena responses for pointwise numerical ratings."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResponseModel:
    name: str
    api_id: str


@dataclass(frozen=True)
class JudgeModel:
    name: str
    model: str
    target_model_id: int
    proxy_for: str | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class RatingConfig:
    run_id: str
    dataset: str
    constitution: Path
    source: Path
    prompt: Path
    selection: str
    generation_max_tokens: int
    criterion_generation_max_tokens: int
    generation_temperature: float
    expected_scenarios: int
    expected_models: int
    expected_response_cells: int
    score_min: int
    score_max: int
    response_models: dict[int, ResponseModel]
    judges: tuple[JudgeModel, ...]


@dataclass(frozen=True)
class ResponseCell:
    scenario_index: int
    scenario: str
    model_id: int
    model_name: str
    model_api_id: str
    response: str
    response_hash: str


@dataclass(frozen=True)
class ConstitutionCriterion:
    criterion_id: str
    text: str
    text_hash: str


def repo_path(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else ROOT / raw


def provenance_path(path: str | Path) -> str:
    """Return one stable path spelling for logged repository artifacts."""
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def sha256_text(text: str) -> str:
    """Return a stable short hash for logged text provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    """Return a stable short hash for a repository data artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def is_valid_response(value: object) -> bool:
    """Reject empty text and known provider-failure artifacts."""
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    if stripped.startswith("Error in ") and " API call:" in stripped[:160]:
        return False
    return "<|reserved_token_" not in stripped


def load_config(path: str | Path) -> RatingConfig:
    """Load the numerical-rating run manifest."""
    data = yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    response_models_raw = data.get("response_models")
    if not isinstance(response_models_raw, dict):
        raise ValueError(f"{path} must define response_models")
    response_models = {
        int(model_index): ResponseModel(
            name=str(model_spec["name"]),
            api_id=str(model_spec["id"]),
        )
        for model_index, model_spec in response_models_raw.items()
    }
    judges = tuple(
        JudgeModel(
            name=str(spec["name"]),
            model=str(spec["model"]),
            target_model_id=int(spec["target_model_id"]),
            proxy_for=(str(spec["proxy_for"]) if spec.get("proxy_for") else None),
            max_tokens=(
                int(spec["max_tokens"]) if spec.get("max_tokens") is not None else None
            ),
        )
        for spec in data.get("judges") or []
    )
    target_model_ids = [judge.target_model_id for judge in judges]
    if sorted(target_model_ids) != list(range(len(response_models))):
        raise ValueError(f"{path} judges must map one-to-one onto response model IDs")

    generation = data.get("generation") or {}
    criterion_generation = data.get("criterion_generation") or {}
    scale = data.get("score_scale") or {}
    return RatingConfig(
        run_id=str(data["run_id"]),
        dataset=str(data["dataset"]),
        constitution=repo_path(data["constitution"]),
        source=repo_path(data["source"]),
        prompt=repo_path(data["prompt"]),
        selection=str(data["selection"]),
        generation_max_tokens=int(generation.get("max_tokens", 512)),
        criterion_generation_max_tokens=int(
            criterion_generation.get("max_tokens", generation.get("max_tokens", 512))
        ),
        generation_temperature=float(generation.get("temperature", 0)),
        expected_scenarios=int(data["expected_scenarios"]),
        expected_models=int(data["expected_models"]),
        expected_response_cells=int(data["expected_response_cells"]),
        score_min=int(scale.get("min", 1)),
        score_max=int(scale.get("max", 10)),
        response_models=response_models,
        judges=judges,
    )


def load_constitution_criteria(path: Path) -> tuple[ConstitutionCriterion, ...]:
    """Load stable IDs and text provenance for each constitution criterion."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{path} must contain a list of constitution criteria")

    criteria = []
    for position, text in enumerate(payload, start=1):
        match = re.match(r"Criterion\s+(\d+)\b", text)
        if match is None:
            raise ValueError(
                f"{path} criterion {position} has no explicit criterion number"
            )
        number = int(match.group(1))
        criteria.append(
            ConstitutionCriterion(
                criterion_id=f"criterion_{number:02d}",
                text=text,
                text_hash=sha256_text(text),
            )
        )

    criterion_ids = [criterion.criterion_id for criterion in criteria]
    if len(criterion_ids) != len(set(criterion_ids)):
        raise ValueError(f"{path} contains duplicate criterion IDs")
    return tuple(criteria)


def load_constitution_text(path: Path) -> str:
    """Return the exact constitution text shown to a holistic judge."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{path} must contain a list of constitution criteria")
    return "\n".join(payload)


def load_response_cells(config: RatingConfig) -> list[ResponseCell]:
    """Load the scenario-by-model response cache used by round-robin runs."""
    if len(config.response_models) != config.expected_models:
        raise ValueError(
            f"Expected {config.expected_models} configured models, "
            f"found {len(config.response_models)}"
        )
    records = json.loads(config.source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{config.source} must contain a JSON list")

    model_ids = {
        model.name: model_id for model_id, model in config.response_models.items()
    }
    cells: dict[tuple[int, int], ResponseCell] = {}
    for record in records:
        scenario_index = int(record["scenario_index"])
        scenario = str(record["scenario"])
        responses = record.get("responses")
        if not isinstance(responses, dict):
            raise ValueError(f"Scenario {scenario_index} has no response mapping")
        if set(responses) != set(model_ids):
            raise ValueError(
                f"Scenario {scenario_index} response models do not match the manifest"
            )
        for model_name, response in responses.items():
            if not isinstance(response, str):
                raise ValueError(
                    f"Scenario {scenario_index} has a non-text response "
                    f"for {model_name}"
                )
            model_id = model_ids[model_name]
            expected_model = config.response_models[model_id]
            _add_response_cell(
                cells,
                scenario_index=scenario_index,
                scenario=scenario,
                model_id=model_id,
                model_name=model_name,
                model_api_id=expected_model.api_id,
                response=response,
            )

    return validate_response_cells(config, cells)


def _add_response_cell(
    cells: dict[tuple[int, int], ResponseCell],
    *,
    scenario_index: int,
    scenario: str,
    model_id: int,
    model_name: str,
    model_api_id: str,
    response: str,
) -> None:
    if not is_valid_response(response):
        raise ValueError(
            f"Invalid response artifact for scenario={scenario_index} model={model_id}"
        )
    key = (scenario_index, model_id)
    cell = ResponseCell(
        scenario_index=scenario_index,
        scenario=scenario,
        model_id=model_id,
        model_name=model_name,
        model_api_id=model_api_id,
        response=response,
        response_hash=sha256_text(response),
    )
    existing = cells.get(key)
    if existing is None:
        cells[key] = cell
        return
    if existing != cell:
        raise ValueError(
            f"Conflicting response cell for scenario={scenario_index} model={model_id}"
        )


def validate_response_cells(
    config: RatingConfig, cells: dict[tuple[int, int], ResponseCell]
) -> list[ResponseCell]:
    """Check that the configured slice is a full scenario-by-model rectangle."""
    out = sorted(cells.values(), key=lambda cell: (cell.scenario_index, cell.model_id))
    scenario_count = len({cell.scenario_index for cell in out})
    if scenario_count != config.expected_scenarios:
        raise ValueError(
            f"Expected {config.expected_scenarios} scenarios, found {scenario_count}"
        )
    if len(out) != config.expected_response_cells:
        raise ValueError(
            f"Expected {config.expected_response_cells} response cells, "
            f"found {len(out)}"
        )

    counts: dict[int, int] = {}
    for cell in out:
        counts[cell.scenario_index] = counts.get(cell.scenario_index, 0) + 1
    bad = {
        scenario_index: count
        for scenario_index, count in counts.items()
        if count != config.expected_models
    }
    if bad:
        raise ValueError(f"Scenarios without {config.expected_models} models: {bad}")
    return out
