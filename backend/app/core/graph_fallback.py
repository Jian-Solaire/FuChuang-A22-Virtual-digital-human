"""Fallback graph runner used when LangGraph is unavailable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SequentialGraphRunner:
    """A tiny runner with an `invoke` method mirroring LangGraph usage."""

    def __init__(
        self,
        steps: list[Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        self._steps = steps

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        current = dict(state)
        for step in self._steps:
            current = step(current)
        return current
