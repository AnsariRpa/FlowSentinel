"""Reusable pre-tool-call grounding guard for Strands Agents.

Keep application-specific validators outside this module and pass them to
GroundedToolGuard. The guard records every decision in audit_log.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from strands.interventions import Guide, InterventionHandler, Proceed

from _decider_client import Decider


DEFAULT_THRESHOLD = 0.45

DEFAULT_QUESTIONS = {
    "grounded": Decider.noul(
        "Are all proposed tool argument values grounded in the user's request?",
        {
            "true": "Every argument value is explicitly supported by the user.",
            "false": "At least one argument value was guessed or invented.",
        },
    ),
    "needs_clarification": Decider.noul(
        "Does this proposed action require clarification before execution?",
        {
            "true": "Important information is missing or ambiguous.",
            "false": "The user's request provides enough information.",
        },
    ),
}

Validator = Callable[[str, Mapping[str, Any], str], str | None]


def conversation_text(messages: list[dict[str, Any]]) -> str:
    """Render text blocks from Strands messages for the decision model."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{role}: {block['text']}")
    return "\n".join(lines)


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Return user text blocks, preserving their order."""
    return "\n".join(
        block.get("text", "")
        for message in messages
        if message.get("role") == "user"
        for block in message.get("content", [])
        if isinstance(block, dict) and "text" in block
    )


def require_explicit_task_title(
    tool_name: str, arguments: Mapping[str, Any], user_text: str
) -> str | None:
    """Example deterministic validator for the demo's create_task tool.

    Returns an explanation when the title is missing; otherwise returns None.
    This checks the user's request, not the model-proposed title.
    """
    if tool_name != "create_task":
        return None

    has_explicit_title = re.search(
        r"\b(?:named|called|titled)\s+['\"]?[^.!?\n'\"]+",
        user_text,
        re.IGNORECASE,
    )
    if not has_explicit_title:
        return "The user did not specify a task title."
    return None


class GroundedToolGuard(InterventionHandler):
    """Evaluate proposed tool calls before execution.

    Args:
        decider: Client for the running Strands Decider endpoint.
        threshold: Experimental decision threshold; calibrate for your workflow.
        validators: Optional deterministic validators keyed by tool name or "*".
            Each receives (tool_name, proposed_arguments, user_text) and returns
            a blocking explanation or None.
        questions: Optional typed questions to ask Decider.
    """

    name = "flowsentinel-grounded-tool-guard"

    def __init__(
        self,
        decider: Decider,
        threshold: float = DEFAULT_THRESHOLD,
        validators: Mapping[str, Validator] | None = None,
        questions: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self.decider = decider
        self.threshold = threshold
        self.validators = dict(validators or {})
        self.questions = dict(questions or DEFAULT_QUESTIONS)
        self.audit_log: list[dict[str, Any]] = []

    def before_tool_call(self, event, **kwargs):
        tool_name = event.tool_use["name"]
        arguments = event.tool_use.get("input") or {}
        messages = event.agent.messages
        user_text = latest_user_text(messages)

        validator = self.validators.get(tool_name) or self.validators.get("*")
        if validator:
            reason = validator(tool_name, arguments, user_text)
            if reason:
                record = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "verdict": "CLARIFY_VALIDATION",
                    "reason": reason,
                    "decision_latency_ms": 0.0,
                }
                self.audit_log.append(record)
                print("\n[FLOWSENTINEL VALIDATION]")
                print(json.dumps(record, indent=2, default=str))
                return Guide(
                    feedback=(
                        f"{reason} Do not invent missing information. "
                        "Ask the user a concise clarification question before "
                        "attempting the tool again."
                    )
                )

        state = (
            "Evaluate the proposed tool call using the conversation below.\n\n"
            f"CONVERSATION:\n{conversation_text(messages)}\n\n"
            f"PROPOSED TOOL: {tool_name}\n"
            f"ARGUMENTS: {json.dumps(arguments, default=str)}"
        )

        answers = self.decider.ask(state, self.questions)
        grounded = float(answers["grounded"]["noul"])
        clarification = float(answers["needs_clarification"]["noul"])

        if grounded < self.threshold:
            verdict = "BLOCK_UNGROUNDED"
            action = Guide(
                feedback=(
                    "The proposed arguments are not grounded in the user's request. "
                    "Do not invent details. Ask the user for the missing information "
                    "or correct the proposal using only supported details."
                )
            )
        elif clarification >= self.threshold:
            verdict = "CLARIFY"
            action = Guide(
                feedback=(
                    "The request needs clarification before executing the tool. "
                    "Ask the user a question instead of guessing."
                )
            )
        else:
            verdict = "PROCEED"
            action = Proceed()

        record = {
            "tool": tool_name,
            "arguments": arguments,
            "grounded_probability": round(grounded, 3),
            "clarification_probability": round(clarification, 3),
            "verdict": verdict,
            "decision_latency_ms": self.decider.last_latency_ms,
        }
        self.audit_log.append(record)
        print("\n[DECIDER 2B]")
        print(json.dumps(record, indent=2, default=str))
        return action
