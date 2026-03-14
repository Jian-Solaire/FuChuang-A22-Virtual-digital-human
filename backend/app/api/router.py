"""Top-level API router."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_knowledge import router as knowledge_router
from backend.app.api.routes_sessions import router as sessions_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(sessions_router)
api_router.include_router(knowledge_router)
