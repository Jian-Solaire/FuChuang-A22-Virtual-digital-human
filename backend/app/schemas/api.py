"""Common API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.app.core.envelope import EnvelopeRequest, EnvelopeResponse, RequestMeta, ResponseMeta


class HealthPayload(BaseModel):
    name: str
    env: str
    services: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: HealthPayload


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=lambda: ResponseMeta(status="error"))
    payload: ErrorPayload


class DebugQueryOptions(BaseModel):
    trace: bool = False


class SearchFiltersModel(BaseModel):
    document_id: str | None = None
    source_path: str | None = None
    base_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = None
    user_id: str | None = None
    memory_type: str | None = None


class SearchOptionsModel(BaseModel):
    mode: str = "auto"
    knowledge_type: str | None = None
    document_scope: str | None = None
    top_k: int = 5
    filters: SearchFiltersModel = Field(default_factory=SearchFiltersModel)
    pageindex_top_k: int = 3


class QueryEnvelope(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: dict[str, Any]
