"""Health route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.schemas.api import HealthPayload, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    container = request.app.state.container
    payload = HealthPayload(
        name=container.settings.app_name,
        env=container.settings.env,
        services={
            "redis_configured": container.settings.redis_enabled,
            "milvus_configured": container.settings.milvus_enabled,
            "pageindex_configured": container.settings.pageindex_enabled,
            "deepseek_mock_only": container.settings.deepseek_mock_only,
        },
    )
    return HealthResponse(
        schema_version=container.settings.schema_version,
        payload=payload,
    )
