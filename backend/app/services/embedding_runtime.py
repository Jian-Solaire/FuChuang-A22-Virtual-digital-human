"""Local-first embedding and reranking helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.config import Settings
from backend.app.services.text_match import token_score


def _is_valid_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_tokenizer = any(
        (path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "spiece.model",
            "vocab.txt",
        )
    )
    has_weights = any(
        (path / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
    )
    return has_config and has_tokenizer and has_weights


def _iter_search_roots(model_name: str, cache_dir: Path, project_root: Path) -> list[Path]:
    model_key = model_name.replace("/", "--")
    candidates = [
        cache_dir,
        cache_dir / f"models--{model_key}",
        project_root / "models",
        project_root / "models" / model_key,
        project_root / "models" / f"models--{model_key}",
    ]

    for env_name in ("MODEL_CACHE_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        env_value = os.getenv(env_name)
        if not env_value:
            continue
        root = Path(env_value)
        candidates.extend([root, root / f"models--{model_key}"])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def resolve_local_model_path(model_name: str, cache_dir: Path, project_root: Path) -> Path | None:
    direct = Path(model_name)
    if _is_valid_model_dir(direct):
        return direct

    for root in _iter_search_roots(model_name, cache_dir, project_root):
        if _is_valid_model_dir(root):
            return root

        snapshots_dir = root / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        for snapshot_path in sorted(snapshots_dir.iterdir(), reverse=True):
            if _is_valid_model_dir(snapshot_path):
                return snapshot_path

    return None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _coerce_sparse_vector(raw_sparse: Any) -> dict[int, float]:
    if not isinstance(raw_sparse, dict):
        return {}
    sparse: dict[int, float] = {}
    for key, value in raw_sparse.items():
        try:
            sparse[int(key)] = float(value)
        except Exception:
            continue
    return sparse


@dataclass(slots=True)
class EncodedBatch:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class LocalEmbeddingRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = _resolve_device(settings.embedding_device)
        self.model_path = resolve_local_model_path(
            settings.embedding_model,
            settings.model_cache_dir,
            settings.project_root,
        )
        self._encoder = None
        self._encoder_backend = "lexical"
        self._load_encoder()

    def _load_encoder(self) -> None:
        if self.model_path is None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel

            kwargs = {
                "use_fp16": self.device.startswith("cuda"),
            }
            try:
                kwargs["device"] = self.device
                self._encoder = BGEM3FlagModel(str(self.model_path), **kwargs)
            except TypeError:
                kwargs.pop("device", None)
                self._encoder = BGEM3FlagModel(str(self.model_path), **kwargs)
            self._encoder_backend = "flagembedding"
            return
        except Exception:
            self._encoder = None

        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(str(self.model_path), device=self.device)
            self._encoder_backend = "sentence_transformers"
        except Exception:
            self._encoder = None
            self._encoder_backend = "lexical"

    def encode(self, texts: list[str]) -> EncodedBatch:
        if not texts:
            return EncodedBatch(dense=[], sparse=[])

        normalized = [text or "" for text in texts]
        dense_vectors = [[0.0] * self.settings.embedding_dimension for _ in normalized]
        sparse_vectors = [{} for _ in normalized]

        if self._encoder_backend == "flagembedding" and self._encoder is not None:
            encoded = self._encoder.encode(
                normalized,
                batch_size=self.settings.embedding_batch_size,
                return_dense=self.settings.embedding_enable_dense,
                return_sparse=self.settings.embedding_enable_sparse,
                return_colbert_vecs=False,
                max_length=8192,
            )
            raw_dense = encoded.get("dense_vecs") or encoded.get("dense") or []
            raw_sparse = encoded.get("lexical_weights") or []
            if hasattr(raw_dense, "tolist"):
                raw_dense = raw_dense.tolist()
            dense_vectors = [list(map(float, item)) for item in raw_dense] if raw_dense else dense_vectors
            sparse_vectors = [_coerce_sparse_vector(item) for item in raw_sparse] if raw_sparse else sparse_vectors
            return EncodedBatch(dense=dense_vectors, sparse=sparse_vectors)

        if self._encoder_backend == "sentence_transformers" and self._encoder is not None:
            raw_dense = self._encoder.encode(normalized, batch_size=self.settings.embedding_batch_size)
            if hasattr(raw_dense, "tolist"):
                raw_dense = raw_dense.tolist()
            dense_vectors = [list(map(float, item)) for item in raw_dense]
            return EncodedBatch(dense=dense_vectors, sparse=sparse_vectors)

        return EncodedBatch(dense=dense_vectors, sparse=sparse_vectors)

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        batch = self.encode([text])
        dense = batch.dense[0] if batch.dense else [0.0] * self.settings.embedding_dimension
        sparse = batch.sparse[0] if batch.sparse else {}
        return dense, sparse


class LocalRerankerRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = _resolve_device(settings.embedding_device)
        self.model_path = resolve_local_model_path(
            settings.reranker_model,
            settings.reranker_cache_dir,
            settings.project_root,
        )
        self._reranker = None
        self._backend = "lexical"
        self._load_reranker()

    def _load_reranker(self) -> None:
        if not self.settings.reranker_enabled or self.model_path is None:
            return

        try:
            from FlagEmbedding import FlagReranker

            kwargs = {
                "use_fp16": self.device.startswith("cuda"),
            }
            try:
                kwargs["device"] = self.device
                self._reranker = FlagReranker(str(self.model_path), **kwargs)
            except TypeError:
                kwargs.pop("device", None)
                self._reranker = FlagReranker(str(self.model_path), **kwargs)
            self._backend = "flagembedding"
            return
        except Exception:
            self._reranker = None

        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(str(self.model_path), device=self.device)
            self._backend = "cross_encoder"
        except Exception:
            self._reranker = None
            self._backend = "lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        if self._backend == "flagembedding" and self._reranker is not None:
            scores = self._reranker.compute_score([[query, passage] for passage in passages])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            return [float(score) for score in scores]

        if self._backend == "cross_encoder" and self._reranker is not None:
            scores = self._reranker.predict([[query, passage] for passage in passages])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            return [float(score) for score in scores]

        return [token_score(query, passage) for passage in passages]

def get_embedding_runtime(settings: Settings) -> LocalEmbeddingRuntime:
    return LocalEmbeddingRuntime(settings)


def get_reranker_runtime(settings: Settings) -> LocalRerankerRuntime:
    return LocalRerankerRuntime(settings)
