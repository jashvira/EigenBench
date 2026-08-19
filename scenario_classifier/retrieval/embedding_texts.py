"""Model-specific text formatting for asymmetric embedding retrieval."""

from __future__ import annotations

from typing import Any


ASYMMETRIC_FORMAT_VERSION = "asymmetric-retrieval-v2-shared-query-instruction"
QUERY_INSTRUCTION = (
    "Retrieve constitution criteria relevant for judging assistant answers to this scenario. "
    "A criterion is relevant if two plausible assistant answers could differ on it and that difference "
    "could affect the preference judgement. Do not retrieve criteria that are merely thematically related."
)
def is_gemini(model: str) -> bool:
    """Return whether the OpenRouter model slug is a Gemini embedding model."""
    return "gemini" in model.lower()


def is_qwen(model: str) -> bool:
    """Return whether the OpenRouter model slug is a Qwen embedding model."""
    return "qwen" in model.lower()


def raw_text(doc: dict[str, Any]) -> str:
    """Return the raw criterion words used as the document body."""
    return str(doc.get("embedding_text") or doc.get("criterion_text") or "")


def card_text(doc: dict[str, Any]) -> str:
    """Return the compressed card text used as the document body."""
    return str(doc.get("embedding_text") or doc.get("claim") or "")


def format_scenario_query(model: str, scenario: str) -> str:
    """Format a scenario as the query side of asymmetric retrieval."""
    if is_gemini(model):
        return f"task: search result | query: {QUERY_INSTRUCTION} Scenario: {scenario}"
    if is_qwen(model):
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {scenario}"
    return scenario


def format_raw_document(model: str, doc: dict[str, Any]) -> str:
    """Format a raw criterion as the document side of asymmetric retrieval."""
    text = raw_text(doc)
    if is_gemini(model):
        return f"title: none | text: {text}"
    return text


def format_card_document(model: str, doc: dict[str, Any]) -> str:
    """Format a criteria card as the document side of asymmetric retrieval."""
    text = card_text(doc)
    if is_gemini(model):
        return f"title: none | text: {text}"
    return text
