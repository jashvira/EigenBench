"""Inspect task for pointwise whole-constitution numerical ratings."""

from __future__ import annotations

import math

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from numerical_rating.collection.data import load_config
from numerical_rating.collection.rating_task import (
    build_pointwise_rating_task,
    extract_json_object,
)

DEFAULT_CONFIG = "numerical_rating/configs/kindness_1000_round_robin.yaml"


@scorer(metrics=[], name="whole_constitution_score")
def whole_constitution_score(score_min: int = 1, score_max: int = 10):
    """Parse the judge's whole-constitution score and rationale."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""
        try:
            parsed = extract_json_object(completion)
            value = float(parsed["score"])
            if not math.isfinite(value):
                raise ValueError("score must be finite")
            if value < score_min or value > score_max:
                raise ValueError(f"score {value} outside [{score_min}, {score_max}]")
            rationale = str(parsed.get("rationale", "")).strip()
            return Score(
                value=value,
                explanation=rationale,
                metadata={"parsed": parsed},
            )
        except Exception as exc:
            raise ValueError(f"Invalid numerical-rating response: {exc}") from exc

    return score


@task
def pointwise_constitution_rating(
    config: str = DEFAULT_CONFIG,
    limit: int | None = None,
    judge_model: str | None = None,
    generation_max_tokens: int | None = None,
) -> Task:
    """Build the fixed-judge pointwise numerical-rating task."""
    rating_config = load_config(config)
    return build_pointwise_rating_task(
        config=config,
        prompt_path=rating_config.prompt,
        scorer=whole_constitution_score(
            score_min=rating_config.score_min,
            score_max=rating_config.score_max,
        ),
        task_name="pointwise_constitution_rating",
        run_id=rating_config.run_id,
        rating_mode="whole_constitution",
        limit=limit,
        judge_model=judge_model,
        generation_max_tokens=generation_max_tokens,
    )
