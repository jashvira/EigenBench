"""Fixed first-turn controls for Petri conversations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from inspect_ai.model import (
    CachePolicy,
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    Model,
    ModelOutput,
)
from inspect_ai.tool import Tool, ToolCall
from inspect_petri import AuditorGenerate
from inspect_petri.target import controller


def task_text(value: object) -> str:
    """Return text from Inspect task args, preserving comma-split values."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ",".join(str(part) for part in value)
    return str(value)


def initial_turn_output(
    *,
    scenario: str,
    target_system: str,
    model_name: str,
) -> ModelOutput:
    """Return the auditor tool calls that start the Petri transcript."""
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="",
                    model=model_name,
                    tool_calls=[
                        ToolCall(
                            id=str(uuid.uuid4()),
                            function="set_system_message",
                            arguments={"system_message": target_system},
                        ),
                        ToolCall(
                            id=str(uuid.uuid4()),
                            function="send_message",
                            arguments={"message": scenario},
                        ),
                        ToolCall(
                            id=str(uuid.uuid4()),
                            function="resume",
                            arguments={},
                        ),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def target_has_been_resumed(messages: Sequence[ChatMessage]) -> bool:
    """Return True once Petri has run the target at least once."""
    return any(
        message.role == "tool" and getattr(message, "function", None) == "resume"
        for message in messages
    )


def tool_registry_name(tool: Tool) -> str:
    """Inspect stores the Petri tool name on registry metadata."""
    info = getattr(tool, "__registry_info__", None)
    return str(getattr(info, "name", "")).rsplit("/", 1)[-1]


def without_system_message_tool(tools: list[Tool]) -> list[Tool]:
    """Drop system-prompt editing after the fixed opening turn."""
    return [
        tool
        for tool in tools
        if tool_registry_name(tool) != "set_system_message"
    ]


def first_turn_generate(*, keep_system_message_tool: bool) -> AuditorGenerate:
    """Set up the target once, then run model-driven auditor turns."""

    async def generate(
        model: Model,
        messages: list[ChatMessage],
        tools: list[Tool],
        cache: bool | CachePolicy,
    ) -> ModelOutput:
        """Send the fixed opening turn before delegating to the auditor."""
        if not target_has_been_resumed(messages):
            metadata = controller().state.metadata
            return initial_turn_output(
                scenario=task_text(metadata["scenario"]),
                target_system=task_text(metadata["target_system"]),
                model_name=model.name,
            )

        if not keep_system_message_tool:
            tools = without_system_message_tool(tools)
        return await model.generate(input=messages, tools=tools, cache=cache)

    return generate
