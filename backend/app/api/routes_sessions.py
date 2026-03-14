"""Session routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.core.ids import new_session_id, new_turn_id
from backend.app.schemas.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionCreateResponsePayload,
    SessionEndRequest,
    SessionEndResponse,
    SessionEndResponsePayload,
    SessionTurnRequest,
    SessionTurnResponse,
    SessionTurnResponsePayload,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse)
def create_session(request_body: SessionCreateRequest, request: Request) -> SessionCreateResponse:
    container = request.app.state.container
    session_id = new_session_id()
    container.memory_store.create_session(
        session_id,
        {
            "profile_seed": request_body.payload.profile_seed,
            "avatar_id": request_body.payload.avatar_id,
        },
    )
    return SessionCreateResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": request_body.payload.user_id,
            "session_id": session_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "turn_id": None,
            "status": "ok",
        },
        payload=SessionCreateResponsePayload(session_id=session_id, thread_id=session_id),
    )


@router.post("/{session_id}/turns", response_model=SessionTurnResponse)
def create_turn(session_id: str, request_body: SessionTurnRequest, request: Request) -> SessionTurnResponse:
    container = request.app.state.container
    session = container.memory_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")

    turn_id = request_body.meta.turn_id or new_turn_id()
    graph_input = {
        "meta": {
            "request_id": request_body.meta.request_id,
            "user_id": request_body.meta.user_id or session.get("profile_seed", {}).get("user_id"),
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
        },
        "turn_input": request_body.payload.model_dump(),
        "runtime_flags": {"mock_mode": True},
    }
    result = container.graph.invoke(graph_input)
    return SessionTurnResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": graph_input["meta"]["user_id"],
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "status": "ok",
        },
        payload=SessionTurnResponsePayload(
            dialog_state=result["output_b"]["dialog_state_update"],
            current_psych_state=result["current_psych_state"],
            retrieval_context=result["retrieval_context"],
            output_a=result["output_a"],
            output_b=result["output_b"],
        ),
        extensions={
            "provider_raw": request_body.extensions.get("provider_raw", {}),
            "feature_candidates": request_body.extensions.get("feature_candidates", {}),
        },
    )


@router.post("/{session_id}/end", response_model=SessionEndResponse)
def end_session(session_id: str, request_body: SessionEndRequest, request: Request) -> SessionEndResponse:
    container = request.app.state.container
    ended = container.memory_store.end_session(session_id, save_history=request_body.payload.save_history)
    if not ended:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return SessionEndResponse(
        schema_version=container.settings.schema_version,
        meta={
            "request_id": request_body.meta.request_id,
            "user_id": request_body.meta.user_id,
            "session_id": session_id,
            "turn_id": request_body.meta.turn_id,
            "timestamp": request_body.meta.timestamp,
            "source": request_body.meta.source,
            "status": "ok",
        },
        payload=SessionEndResponsePayload(session_id=session_id, ended=True),
    )
