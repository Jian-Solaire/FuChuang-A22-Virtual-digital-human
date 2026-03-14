"""Knowledge management schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.core.envelope import EnvelopeRequest, EnvelopeResponse, RequestMeta, ResponseMeta
from backend.app.schemas.api import DebugQueryOptions, SearchOptionsModel


class KnowledgeBaseCreatePayload(BaseModel):
    name: str
    description: str | None = None
    namespace: str = "default"
    route: Literal["auto", "milvus", "pageindex", "hybrid"] = "auto"
    knowledge_type: str = "knowledge"
    tags: list[str] = Field(default_factory=list)


class KnowledgeDocument(BaseModel):
    title: str
    content: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentIngestPayload(BaseModel):
    documents: list[KnowledgeDocument]


class KnowledgeQueryPayload(BaseModel):
    query: str
    options: SearchOptionsModel = Field(default_factory=SearchOptionsModel)
    debug: DebugQueryOptions = Field(default_factory=DebugQueryOptions)


class KnowledgeBaseRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: KnowledgeBaseCreatePayload


class KnowledgeDocumentIngestRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: KnowledgeDocumentIngestPayload


class KnowledgeQueryRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: KnowledgeQueryPayload


class KnowledgeBaseResponsePayload(BaseModel):
    base_id: str
    name: str
    route: str
    knowledge_type: str
    namespace: str
    tags: list[str] = Field(default_factory=list)


class KnowledgeBaseStatusPayload(BaseModel):
    base_id: str
    status: str
    route: str
    document_count: int = 0
    pageindex_cache_count: int = 0


class KnowledgeIngestResponsePayload(BaseModel):
    base_id: str
    route: str
    ingested_count: int
    pageindex_cache_count: int = 0


class KnowledgeQueryResponsePayload(BaseModel):
    query: str
    route_used: str
    retrieval_context: dict[str, Any]


class KnowledgeBaseResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: KnowledgeBaseResponsePayload


class KnowledgeBaseStatusResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: KnowledgeBaseStatusPayload


class KnowledgeIngestResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: KnowledgeIngestResponsePayload


class KnowledgeQueryResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: KnowledgeQueryResponsePayload
