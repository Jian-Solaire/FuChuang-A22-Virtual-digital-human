"""Shared API envelope helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestMeta(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    source: str = "api"


class ResponseMeta(RequestMeta):
    status: str = "ok"


class EnvelopeBase(BaseModel):
    schema_version: str = "1.0"
    extensions: dict[str, Any] = Field(default_factory=dict)


class EnvelopeRequest(EnvelopeBase):
    meta: RequestMeta = Field(default_factory=RequestMeta)


class EnvelopeResponse(EnvelopeBase):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: dict[str, Any] = Field(default_factory=dict)
