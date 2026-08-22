"""Ranking metrics for scenario-to-constitution retrieval.

Retrievers rank cards or raw criteria. Each retrieved unit has a constitution,
so top-k can be measured before or after collapsing repeated constitutions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def collapse_to_constitutions(ranking: Iterable[dict[str, Any]]) -> list[str]:
    """Return distinct constitutions in first-seen ranking order."""
    constitutions: list[str] = []
    seen: set[str] = set()
    for item in ranking:
        constitution = str(item.get("constitution") or "")
        if not constitution or constitution in seen:
            continue
        seen.add(constitution)
        constitutions.append(constitution)
    return constitutions


def unit_top_k_hit(ranking: list[dict[str, Any]], target_constitution: str, k: int) -> bool:
    """Return whether any of the first k retrieved units maps to the target."""
    if target_constitution == "none":
        return False
    return any(item.get("constitution") == target_constitution for item in ranking[:k])


def collapsed_constitution_top_k_hit(
    ranking: list[dict[str, Any]],
    target_constitution: str,
    k: int,
) -> bool:
    """Return whether the target appears after deduplicating constitutions."""
    if target_constitution == "none":
        return False
    return target_constitution in collapse_to_constitutions(ranking)[:k]


def recall_at_k(
    rows: list[dict[str, Any]],
    ranking_key: str,
    ks: list[int],
    *,
    target_key: str = "annotation_primary",
    hit_mode: str = "unit",
) -> dict[str, float]:
    """Compute unit-level or collapsed-constitution recall@k."""
    annotated = [row for row in rows if row[target_key] != "none"]
    if not annotated:
        return {f"recall@{k}": 0.0 for k in ks}

    if hit_mode == "unit":
        hit_fn = unit_top_k_hit
    elif hit_mode == "collapsed":
        hit_fn = collapsed_constitution_top_k_hit
    else:
        raise ValueError(f"unknown hit_mode {hit_mode!r}; expected 'unit' or 'collapsed'")

    return {
        f"recall@{k}": round(
            sum(hit_fn(row[ranking_key], row[target_key], k) for row in annotated) / len(annotated),
            6,
        )
        for k in ks
    }
