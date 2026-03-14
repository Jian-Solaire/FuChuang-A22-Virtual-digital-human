"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["APP_ENV"] = "test"
os.environ["DEEPSEEK_MOCK_ONLY"] = "1"
os.environ["MILVUS_ENABLED"] = "0"
os.environ["PAGEINDEX_ENABLED"] = "1"
os.environ["REDIS_ENABLED"] = "0"

from backend.app.main import create_app


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)
