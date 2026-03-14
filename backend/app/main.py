"""FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.dependencies import build_container


def create_app() -> FastAPI:
    container = build_container()
    app = FastAPI(title=container.settings.app_name, version=container.settings.schema_version)
    app.state.container = container
    app.include_router(api_router, prefix=container.settings.api_prefix)
    return app


app = create_app()
