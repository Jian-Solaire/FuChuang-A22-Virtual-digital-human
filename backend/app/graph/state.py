"""Graph state models."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    meta: dict[str, Any]
    turn_input: dict[str, Any]
    previous_context: dict[str, Any]
    perception_result: dict[str, Any]
    current_psych_state: dict[str, Any]
    reassessment_result: dict[str, Any]
    retrieval_context: dict[str, Any]
    output_a: dict[str, Any]
    output_b: dict[str, Any]
    runtime_flags: dict[str, Any]
