"""Application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _as_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _normalize_host(value: str) -> str:
    return "localhost" if value.strip().lower() == "local" else value


_load_env_file(ENV_FILE)


@dataclass(slots=True)
class Settings:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    app_name: str = "Digital Human Backend"
    api_prefix: str = "/api/v1"
    schema_version: str = "1.0"
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "dev"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    redis_enabled: bool = field(default_factory=lambda: _as_bool("REDIS_ENABLED", False))

    deepseek_api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_timeout_s: float = field(default_factory=lambda: _as_float("DEEPSEEK_TIMEOUT_S", 60.0))
    deepseek_mock_only: bool = field(default_factory=lambda: _as_bool("DEEPSEEK_MOCK_ONLY", True))

    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "psych-avatar-dev"))
    risk_threshold: float = field(default_factory=lambda: _as_float("RISK_THRESHOLD", 0.7))

    milvus_enabled: bool = field(default_factory=lambda: _as_bool("MILVUS_ENABLED", False))
    milvus_host: str = field(default_factory=lambda: _normalize_host(os.getenv("MILVUS_HOST", "localhost")))
    milvus_port: int = field(default_factory=lambda: _as_int("MILVUS_PORT", 19530))
    milvus_user: str | None = field(default_factory=lambda: os.getenv("MILVUS_USER") or None)
    milvus_password: str | None = field(default_factory=lambda: os.getenv("MILVUS_PASSWORD") or None)
    milvus_token: str | None = field(default_factory=lambda: os.getenv("MILVUS_TOKEN") or None)
    milvus_secure: bool = field(default_factory=lambda: _as_bool("MILVUS_SECURE", False))
    milvus_database: str = field(default_factory=lambda: os.getenv("MILVUS_DATABASE", "default"))
    milvus_section_collection: str = field(default_factory=lambda: os.getenv("MILVUS_SECTION_COLLECTION", "rag_sections"))
    milvus_chunk_collection: str = field(default_factory=lambda: os.getenv("MILVUS_CHUNK_COLLECTION", "rag_chunks"))
    milvus_memory_collection: str = field(default_factory=lambda: os.getenv("MILVUS_MEMORY_COLLECTION", "psych_memory"))

    pageindex_enabled: bool = field(default_factory=lambda: _as_bool("PAGEINDEX_ENABLED", False))
    pageindex_cache_dir: Path = field(
        default_factory=lambda: _as_path(
            "PAGEINDEX_CACHE_DIR",
            PROJECT_ROOT / "backend" / ".runtime" / "pageindex_cache",
        )
    )
    data_dir: Path = field(
        default_factory=lambda: _as_path(
            "APP_DATA_DIR",
            PROJECT_ROOT / "backend" / ".runtime",
        )
    )
    knowledge_dir: Path = field(
        default_factory=lambda: _as_path(
            "KNOWLEDGE_DIR",
            PROJECT_ROOT / "data" / "knowledge",
        )
    )
    default_namespace: str = field(default_factory=lambda: os.getenv("DEFAULT_NAMESPACE", "default"))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_dimension: int = field(default_factory=lambda: _as_int("EMBEDDING_DIMENSION", 1024))
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "auto"))
    embedding_enable_dense: bool = field(default_factory=lambda: _as_bool("EMBEDDING_ENABLE_DENSE", True))
    embedding_enable_sparse: bool = field(default_factory=lambda: _as_bool("EMBEDDING_ENABLE_SPARSE", True))
    embedding_batch_size: int = field(default_factory=lambda: _as_int("EMBEDDING_BATCH_SIZE", 8))
    model_cache_dir: Path = field(
        default_factory=lambda: _as_path(
            "MODEL_CACHE_DIR",
            PROJECT_ROOT / "models" / "bge-m3",
        )
    )
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_enabled: bool = field(default_factory=lambda: _as_bool("RERANKER_ENABLED", True))
    reranker_top_n: int = field(default_factory=lambda: _as_int("RERANKER_TOP_N", 20))
    reranker_cache_dir: Path = field(
        default_factory=lambda: _as_path(
            "RERANKER_CACHE_DIR",
            PROJECT_ROOT / "models",
        )
    )


def get_settings() -> Settings:
    settings = Settings()
    settings.pageindex_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.reranker_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
