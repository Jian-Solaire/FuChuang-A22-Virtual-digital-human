"""Knowledge routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas.knowledge import (
    KnowledgeBaseRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseResponsePayload,
    KnowledgeBaseStatusPayload,
    KnowledgeBaseStatusResponse,
    KnowledgeDocumentIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeIngestResponsePayload,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeQueryResponsePayload,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/bases", response_model=KnowledgeBaseResponse)
def create_knowledge_base(request_body: KnowledgeBaseRequest, request: Request) -> KnowledgeBaseResponse:
    container = request.app.state.container
    base = container.retrieval_orchestrator.create_base(
        name=request_body.payload.name,
        description=request_body.payload.description,
        namespace=request_body.payload.namespace,
        route=request_body.payload.route,
        knowledge_type=request_body.payload.knowledge_type,
        tags=request_body.payload.tags,
    )
    return KnowledgeBaseResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": request_body.meta.user_id,
            "session_id": request_body.meta.session_id,
            "turn_id": request_body.meta.turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "status": "ok",
        },
        payload=KnowledgeBaseResponsePayload(**asdict(base)),
    )


@router.post("/bases/{base_id}/documents", response_model=KnowledgeIngestResponse)
def ingest_documents(
    base_id: str,
    request_body: KnowledgeDocumentIngestRequest,
    request: Request,
) -> KnowledgeIngestResponse:
    container = request.app.state.container
    try:
        result = container.retrieval_orchestrator.ingest_documents(
            base_id,
            [document.model_dump() for document in request_body.payload.documents],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return KnowledgeIngestResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": request_body.meta.user_id,
            "session_id": request_body.meta.session_id,
            "turn_id": request_body.meta.turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "status": "ok",
        },
        payload=KnowledgeIngestResponsePayload(**result),
    )


@router.get("/bases/{base_id}", response_model=KnowledgeBaseResponse)
def get_base(base_id: str, request: Request) -> KnowledgeBaseResponse:
    container = request.app.state.container
    base = container.retrieval_orchestrator.get_base(base_id)
    if not base:
        raise HTTPException(status_code=404, detail=f"Unknown knowledge base: {base_id}")
    return KnowledgeBaseResponse(
        schema_version=container.settings.schema_version,
        payload=KnowledgeBaseResponsePayload(**asdict(base)),
    )


@router.get("/bases/{base_id}/status", response_model=KnowledgeBaseStatusResponse)
def get_base_status(base_id: str, request: Request) -> KnowledgeBaseStatusResponse:
    container = request.app.state.container
    status = container.retrieval_orchestrator.get_base_status(base_id)
    return KnowledgeBaseStatusResponse(
        schema_version=container.settings.schema_version,
        payload=KnowledgeBaseStatusPayload(**status),
    )


@router.post("/query", response_model=KnowledgeQueryResponse)
def query_knowledge(request_body: KnowledgeQueryRequest, request: Request) -> KnowledgeQueryResponse:
    container = request.app.state.container
    options = request_body.payload.options
    retrieval_context = container.retrieval_orchestrator.query(
        query=request_body.payload.query,
        mode=options.mode,
        knowledge_type=options.knowledge_type,
        document_scope=options.document_scope,
        filters=options.filters.model_dump(),
        top_k=options.top_k,
        pageindex_top_k=options.pageindex_top_k,
        debug_trace=request_body.payload.debug.trace,
    )
    return KnowledgeQueryResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": request_body.meta.user_id,
            "session_id": request_body.meta.session_id,
            "turn_id": request_body.meta.turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "status": "ok",
        },
        payload=KnowledgeQueryResponsePayload(
            query=request_body.payload.query,
            route_used=retrieval_context["route_used"],
            retrieval_context=retrieval_context,
        ),
    )


@router.post("/query/milvus", response_model=KnowledgeQueryResponse)
def query_milvus_debug(request_body: KnowledgeQueryRequest, request: Request) -> KnowledgeQueryResponse:
    container = request.app.state.container
    options = request_body.payload.options
    retrieval_context = container.retrieval_orchestrator.debug_query_milvus(
        query=request_body.payload.query,
        filters=options.filters.model_dump(),
        top_k=options.top_k,
    )
    return KnowledgeQueryResponse(
        schema_version=container.settings.schema_version,
        payload=KnowledgeQueryResponsePayload(
            query=request_body.payload.query,
            route_used="milvus",
            retrieval_context=retrieval_context,
        ),
    )


@router.post("/query/pageindex", response_model=KnowledgeQueryResponse)
def query_pageindex_debug(request_body: KnowledgeQueryRequest, request: Request) -> KnowledgeQueryResponse:
    container = request.app.state.container
    options = request_body.payload.options
    retrieval_context = container.retrieval_orchestrator.debug_query_pageindex(
        query=request_body.payload.query,
        filters=options.filters.model_dump(),
        top_k=options.pageindex_top_k,
    )
    return KnowledgeQueryResponse(
        schema_version=container.settings.schema_version,
        payload=KnowledgeQueryResponsePayload(
            query=request_body.payload.query,
            route_used="pageindex",
            retrieval_context=retrieval_context,
        ),
    )
