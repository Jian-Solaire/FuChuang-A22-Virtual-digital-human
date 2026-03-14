"""Identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_session_id() -> str:
    return f"sess_{uuid4().hex[:12]}"


def new_turn_id() -> str:
    return f"turn_{uuid4().hex[:12]}"
