"""Load reused ValueArena responses for pointwise numerical ratings."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


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
    slice_summary: Path | None
    prompt: Path
    selection: str
    judge_model: str
    generation_max_tokens: int
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
class RepairedCellManifest:
    path: Path
    manifest_hash: str
    cells: tuple[ResponseCell, ...]


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
                int(spec["max_tokens"])
                if spec.get("max_tokens") is not None
                else None
            ),
        )
        for spec in data.get("judges") or []
    )
    target_model_ids = [judge.target_model_id for judge in judges]
    if judges and sorted(target_model_ids) != list(range(len(response_models))):
        raise ValueError(f"{path} judges must map one-to-one onto response model IDs")

    generation = data.get("generation") or {}
    scale = data.get("score_scale") or {}
    return RatingConfig(
        run_id=str(data["run_id"]),
        dataset=str(data["dataset"]),
        constitution=repo_path(data["constitution"]),
        source=repo_path(data["source"]),
        slice_summary=(
            repo_path(data["slice_summary"])
            if data.get("slice_summary")
            else None
        ),
        prompt=repo_path(data["prompt"]),
        selection=str(data["selection"]),
        judge_model=str(data["judge_model"]),
        generation_max_tokens=int(generation.get("max_tokens", 512)),
        generation_temperature=float(generation.get("temperature", 0)),
        expected_scenarios=int(data["expected_scenarios"]),
        expected_models=int(data["expected_models"]),
        expected_response_cells=int(data["expected_response_cells"]),
        score_min=int(scale.get("min", 1)),
        score_max=int(scale.get("max", 10)),
        response_models=response_models,
        judges=judges,
    )


def load_constitution_text(path: Path) -> str:
    """Return the exact constitution text shown to the judge."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{path} must contain a list of constitution criteria")
    return "\n".join(payload)


def load_slice_indices(path: Path) -> list[int]:
    """Load the scenario indices in the precomputed full-8 slice."""
    data = json.loads(path.read_text(encoding="utf-8"))
    indices = data.get("scenario_indices")
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError(f"{path} must contain integer scenario_indices")
    return indices


def iter_jsonl(path: Path):
    """Yield records from a ValueArena JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_response_cells_csv(config: RatingConfig) -> list[ResponseCell]:
    """Load one already-deduped response row per scenario/model cell."""
    scenario_indices = (
        set(load_slice_indices(config.slice_summary))
        if config.slice_summary
        else None
    )
    cells: dict[tuple[int, int], ResponseCell] = {}

    with config.source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario_index = int(row["scenario_index"])
            if scenario_indices is not None and scenario_index not in scenario_indices:
                continue
            model_id = int(row["model_id"])
            expected_model = config.response_models.get(model_id)
            if expected_model is None:
                raise ValueError(f"Unexpected model index in source: {model_id}")
            model_name = str(row["model_name"])
            if model_name != expected_model.name:
                raise ValueError(
                    f"Model name mismatch for index {model_id}: "
                    f"{model_name!r} != {expected_model.name!r}"
                )
            _add_response_cell(
                cells,
                scenario_index=scenario_index,
                scenario=str(row["scenario"]),
                model_id=model_id,
                model_name=model_name,
                model_api_id=expected_model.api_id,
                response=str(row["response"]),
            )

    return validate_response_cells(config, cells)


def load_response_cache(config: RatingConfig) -> list[ResponseCell]:
    """Load the scenario-by-model response cache used by round-robin runs."""
    records = json.loads(config.source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{config.source} must contain a JSON list")

    model_ids = {
        model.name: model_id
        for model_id, model in config.response_models.items()
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
            model_id = model_ids[model_name]
            expected_model = config.response_models[model_id]
            _add_response_cell(
                cells,
                scenario_index=scenario_index,
                scenario=scenario,
                model_id=model_id,
                model_name=model_name,
                model_api_id=expected_model.api_id,
                response=str(response),
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
            "Invalid response artifact for "
            f"scenario={scenario_index} model={model_id}"
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
    if existing.response != response or existing.model_name != model_name:
        raise ValueError(
            "Conflicting response cell for "
            f"scenario={scenario_index} model={model_id}"
        )


def load_response_cells(config: RatingConfig) -> list[ResponseCell]:
    """Return the reused response cells for the configured full-8 slice."""
    if len(config.response_models) != config.expected_models:
        raise ValueError(
            f"Expected {config.expected_models} configured models, "
            f"found {len(config.response_models)}"
        )
    if config.source.suffix == ".csv":
        return load_response_cells_csv(config)
    if config.source.suffix == ".json":
        return load_response_cache(config)

    scenario_indices = (
        set(load_slice_indices(config.slice_summary))
        if config.slice_summary
        else None
    )
    cells: dict[tuple[int, int], ResponseCell] = {}

    for record in iter_jsonl(config.source):
        scenario_index = int(record["scenario_index"])
        if scenario_indices is not None and scenario_index not in scenario_indices:
            continue
        scenario = str(record["scenario"])
        # ValueArena stores two evaluee responses per pairwise row; dedupe them
        # into one response cell per scenario/model before numerical scoring.
        for prefix in ("eval1", "eval2"):
            response = record.get(f"{prefix} response")
            model_id = record.get(prefix)
            model_name = record.get(f"{prefix}_name")
            if not isinstance(response, str) or not response.strip():
                continue
            if not isinstance(model_id, int) or not isinstance(model_name, str):
                continue
            expected_model = config.response_models.get(model_id)
            if expected_model is None:
                raise ValueError(f"Unexpected model index in source: {model_id}")
            if model_name != expected_model.name:
                raise ValueError(
                    f"Model name mismatch for index {model_id}: "
                    f"{model_name!r} != {expected_model.name!r}"
                )
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


def load_repaired_cell_manifest(
    path: str | Path,
    *,
    config: RatingConfig,
    response_cells: list[ResponseCell],
) -> RepairedCellManifest:
    """Resolve a repaired-cell manifest against the current response cache."""
    manifest_path = repo_path(path)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(manifest_text)
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")

    manifest_cells = payload.get("cells")
    if not isinstance(manifest_cells, list):
        raise ValueError(f"{manifest_path} must contain a cells list")

    cache_by_key = {
        (cell.scenario_index, cell.model_id): cell for cell in response_cells
    }
    required_fields = {
        "scenario_index",
        "model_id",
        "model_name",
        "response_hash",
    }
    seen: set[tuple[int, int]] = set()
    selected: list[ResponseCell] = []

    for position, item in enumerate(manifest_cells):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest cell {position} must be a JSON object")
        missing = sorted(required_fields - item.keys())
        if missing:
            raise ValueError(f"Manifest cell {position} is missing fields: {missing}")

        scenario_index = item["scenario_index"]
        model_id = item["model_id"]
        model_name = item["model_name"]
        response_hash = item["response_hash"]
        if type(scenario_index) is not int or type(model_id) is not int:
            raise ValueError(
                f"Manifest cell {position} must use integer scenario_index and model_id"
            )
        if not isinstance(model_name, str) or not isinstance(response_hash, str):
            raise ValueError(
                f"Manifest cell {position} must use string model_name and response_hash"
            )

        key = (scenario_index, model_id)
        if key in seen:
            raise ValueError(
                "Duplicate repaired cell for "
                f"scenario={scenario_index} model={model_id}"
            )
        seen.add(key)

        expected_model = config.response_models.get(model_id)
        if expected_model is None:
            raise ValueError(f"Unexpected model ID in manifest: {model_id}")
        if model_name != expected_model.name:
            raise ValueError(
                f"Model name mismatch for ID {model_id}: "
                f"{model_name!r} != {expected_model.name!r}"
            )

        cache_cell = cache_by_key.get(key)
        if cache_cell is None:
            raise ValueError(
                "Manifest cell is absent from the current response cache: "
                f"scenario={scenario_index} model={model_id}"
            )
        if model_name != cache_cell.model_name:
            raise ValueError(
                "Manifest model does not match the current response cache for "
                f"scenario={scenario_index} model={model_id}"
            )
        if response_hash != cache_cell.response_hash:
            raise ValueError(
                "Response hash mismatch for "
                f"scenario={scenario_index} model={model_id}: "
                f"{response_hash!r} != {cache_cell.response_hash!r}"
            )
        selected.append(cache_cell)

    return RepairedCellManifest(
        path=manifest_path,
        manifest_hash=sha256_text(manifest_text),
        cells=tuple(selected),
    )


def validate_response_cells(
    config: RatingConfig, cells: dict[tuple[int, int], ResponseCell]
) -> list[ResponseCell]:
    """Check that the configured slice is a full scenario-by-model rectangle."""
    out = sorted(cells.values(), key=lambda cell: (cell.scenario_index, cell.model_id))
    # The comparison slice is only valid if every selected scenario has all models.
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
