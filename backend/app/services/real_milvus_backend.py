"""Optional real Milvus backend for knowledge and memory retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.app.config import Settings
from backend.app.services.embedding_runtime import get_embedding_runtime, get_reranker_runtime


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _rrf_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(dense_results, start=1):
        row = fused.setdefault(str(result["id"]), dict(result))
        row["dense_rank"] = rank

    for rank, result in enumerate(sparse_results, start=1):
        row = fused.setdefault(str(result["id"]), dict(result))
        row["sparse_rank"] = rank

    ranked: list[dict[str, Any]] = []
    for row in fused.values():
        score = 0.0
        if row.get("dense_rank") is not None:
            score += 1.0 / (rrf_k + row["dense_rank"])
        if row.get("sparse_rank") is not None:
            score += 1.0 / (rrf_k + row["sparse_rank"])
        row["score"] = score
        ranked.append(row)

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked


@dataclass(slots=True)
class PreparedDocument:
    sections: list[dict[str, Any]]
    chunks: list[dict[str, Any]]


class RealMilvusBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alias = f"digital_human_{uuid4().hex[:8]}"
        self.embedding_runtime = get_embedding_runtime(settings)
        self.reranker_runtime = get_reranker_runtime(settings)
        self._connect()
        self.section_collection = self._ensure_section_collection()
        self.chunk_collection = self._ensure_chunk_collection()
        self.memory_collection = self._ensure_memory_collection()

    def _connect(self) -> None:
        connect_kwargs: dict[str, Any] = {
            "alias": self.alias,
            "host": self.settings.milvus_host,
            "port": self.settings.milvus_port,
            "secure": self.settings.milvus_secure,
        }
        if self.settings.milvus_token:
            connect_kwargs["token"] = self.settings.milvus_token
        else:
            connect_kwargs["user"] = self.settings.milvus_user or ""
            connect_kwargs["password"] = self.settings.milvus_password or ""
        connections.connect(**connect_kwargs)
        utility.list_collections(using=self.alias)

    def _string_field(
        self,
        name: str,
        max_length: int,
        *,
        is_primary: bool = False,
        partition_key: bool = False,
    ) -> FieldSchema:
        try:
            return FieldSchema(
                name=name,
                dtype=DataType.VARCHAR,
                max_length=max_length,
                is_primary=is_primary,
                is_partition_key=partition_key,
            )
        except TypeError:
            return FieldSchema(
                name=name,
                dtype=DataType.VARCHAR,
                max_length=max_length,
                is_primary=is_primary,
            )

    def _ensure_section_collection(self) -> Collection:
        fields = [
            self._string_field("id", 256, is_primary=True),
            self._string_field("namespace", 128, partition_key=True),
            self._string_field("document_id", 256),
            self._string_field("node_id", 256),
            self._string_field("parent_node_id", 256),
            FieldSchema(name="level", dtype=DataType.INT64),
            self._string_field("tree_path", 2048),
            self._string_field("title", 1024),
            self._string_field("summary", 8192),
            self._string_field("content", 65535),
            self._string_field("heading_path", 2048),
            self._string_field("source_path", 2048),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dimension),
            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
            self._string_field("metadata_json", 16384),
        ]
        return self._ensure_collection(
            self.settings.milvus_section_collection,
            fields,
            description="Digital human section nodes",
        )

    def _ensure_chunk_collection(self) -> Collection:
        fields = [
            self._string_field("id", 256, is_primary=True),
            self._string_field("namespace", 128, partition_key=True),
            self._string_field("document_id", 256),
            self._string_field("node_id", 256),
            self._string_field("parent_node_id", 256),
            self._string_field("section_id", 256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            self._string_field("heading_path", 2048),
            self._string_field("source_path", 2048),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="char_start", dtype=DataType.INT64),
            FieldSchema(name="char_end", dtype=DataType.INT64),
            self._string_field("content", 65535),
            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dimension),
            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
            self._string_field("metadata_json", 16384),
        ]
        return self._ensure_collection(
            self.settings.milvus_chunk_collection,
            fields,
            description="Digital human chunk nodes",
        )

    def _ensure_memory_collection(self) -> Collection:
        fields = [
            self._string_field("id", 256, is_primary=True),
            self._string_field("user_id", 128, partition_key=True),
            self._string_field("memory_type", 128),
            self._string_field("text", 8192),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dimension),
            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
            self._string_field("metadata_json", 16384),
        ]
        return self._ensure_collection(
            self.settings.milvus_memory_collection,
            fields,
            description="Digital human long-term memory",
        )

    def _ensure_collection(
        self,
        name: str,
        fields: list[FieldSchema],
        *,
        description: str,
    ) -> Collection:
        if utility.has_collection(name, using=self.alias):
            collection = Collection(name, using=self.alias)
        else:
            schema = CollectionSchema(fields=fields, description=description)
            collection = Collection(name=name, schema=schema, using=self.alias)

        self._ensure_indexes(collection)
        collection.load()
        return collection

    def _ensure_indexes(self, collection: Collection) -> None:
        existing_indexes = {index.field_name for index in collection.indexes}

        if "dense_vec" not in existing_indexes:
            collection.create_index(
                field_name="dense_vec",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "IP",
                    "params": {"M": 32, "efConstruction": 200},
                },
            )

        if "sparse_vec" not in existing_indexes:
            try:
                collection.create_index(
                    field_name="sparse_vec",
                    index_params={
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "IP",
                        "params": {},
                    },
                )
            except Exception:
                pass

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        metadata_json = normalized.get("metadata_json", {})
        normalized["metadata_json"] = json.dumps(metadata_json, ensure_ascii=False)[:16384]
        for key, limit in {
            "id": 256,
            "namespace": 128,
            "document_id": 256,
            "node_id": 256,
            "parent_node_id": 256,
            "section_id": 256,
            "tree_path": 2048,
            "title": 1024,
            "summary": 8192,
            "content": 65535,
            "heading_path": 2048,
            "source_path": 2048,
            "user_id": 128,
            "memory_type": 128,
            "text": 8192,
        }.items():
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = value[:limit]
        normalized.setdefault("sparse_vec", {})
        return normalized

    def _build_sections_and_chunks(
        self,
        *,
        namespace: str,
        document_id: str,
        title: str,
        content: str,
        source_path: str | None,
        metadata: dict[str, Any],
    ) -> PreparedDocument:
        sections: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        lines = [line.rstrip() for line in (content or "").splitlines()]

        heading_stack: list[str] = []
        current_title = title
        current_level = 1
        current_lines: list[str] = []
        section_index = 0

        def flush_section() -> None:
            nonlocal section_index
            body = "\n".join(line for line in current_lines if line.strip()).strip()
            if not body and not sections and not title:
                return

            section_index += 1
            section_id = f"{document_id}::section::{section_index}"
            heading_path = " / ".join(heading_stack or [current_title])
            summary = (body[:240] or current_title).strip()
            sections.append(
                {
                    "id": section_id,
                    "namespace": namespace,
                    "document_id": document_id,
                    "node_id": f"section_{section_index}",
                    "parent_node_id": "",
                    "level": current_level,
                    "tree_path": f"/{section_index}",
                    "title": current_title,
                    "summary": summary,
                    "content": body or current_title,
                    "heading_path": heading_path,
                    "source_path": source_path or "",
                    "page_start": -1,
                    "page_end": -1,
                    "metadata_json": metadata,
                }
            )

            chunk_text = body or current_title
            chunk_size = 500
            overlap = 80
            start = 0
            chunk_index = 0
            while start < len(chunk_text):
                end = min(len(chunk_text), start + chunk_size)
                piece = chunk_text[start:end].strip()
                if piece:
                    chunk_index += 1
                    chunks.append(
                        {
                            "id": f"{section_id}::chunk::{chunk_index}",
                            "namespace": namespace,
                            "document_id": document_id,
                            "node_id": f"chunk_{section_index}_{chunk_index}",
                            "parent_node_id": f"section_{section_index}",
                            "section_id": section_id,
                            "chunk_index": chunk_index,
                            "heading_path": heading_path,
                            "source_path": source_path or "",
                            "page_start": -1,
                            "page_end": -1,
                            "char_start": start,
                            "char_end": end,
                            "content": piece,
                            "metadata_json": metadata,
                        }
                    )
                if end >= len(chunk_text):
                    break
                start = max(end - overlap, start + 1)

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if current_lines:
                    flush_section()
                    current_lines = []

                current_level = len(line) - len(line.lstrip("#"))
                current_title = line.lstrip("#").strip() or title
                heading_stack = heading_stack[: max(current_level - 1, 0)] + [current_title]
                current_lines.append(current_title)
                continue

            current_lines.append(line)

        if current_lines or not sections:
            if not heading_stack:
                heading_stack = [title]
            flush_section()

        return PreparedDocument(sections=sections, chunks=chunks)

    def upsert_document(
        self,
        *,
        namespace: str,
        document_id: str,
        title: str,
        content: str,
        source_path: str | None,
        metadata: dict[str, Any],
    ) -> None:
        prepared = self._build_sections_and_chunks(
            namespace=namespace,
            document_id=document_id,
            title=title,
            content=content,
            source_path=source_path,
            metadata=metadata,
        )

        section_texts = [f"{item['title']}\n{item['summary']}\n{item['content']}" for item in prepared.sections]
        chunk_texts = [item["content"] for item in prepared.chunks]

        section_embeddings = self.embedding_runtime.encode(section_texts)
        chunk_embeddings = self.embedding_runtime.encode(chunk_texts)

        for item, dense, sparse in zip(prepared.sections, section_embeddings.dense, section_embeddings.sparse):
            item["dense_vec"] = dense
            item["sparse_vec"] = sparse

        for item, dense, sparse in zip(prepared.chunks, chunk_embeddings.dense, chunk_embeddings.sparse):
            item["dense_vec"] = dense
            item["sparse_vec"] = sparse

        self.delete_document(document_id=document_id, namespace=namespace)
        if prepared.sections:
            self.section_collection.insert([self._normalize_row(row) for row in prepared.sections])
        if prepared.chunks:
            self.chunk_collection.insert([self._normalize_row(row) for row in prepared.chunks])
        self.section_collection.flush()
        self.chunk_collection.flush()
        self.section_collection.load()
        self.chunk_collection.load()

    def delete_document(self, *, document_id: str, namespace: str) -> None:
        expr = f'namespace == "{_escape(namespace)}" and document_id == "{_escape(document_id)}"'
        try:
            self.section_collection.delete(expr)
            self.chunk_collection.delete(expr)
        except Exception:
            return

    def _chunk_output_fields(self) -> list[str]:
        return [
            "namespace",
            "document_id",
            "node_id",
            "parent_node_id",
            "section_id",
            "chunk_index",
            "heading_path",
            "source_path",
            "page_start",
            "page_end",
            "char_start",
            "char_end",
            "content",
            "metadata_json",
        ]

    def _memory_output_fields(self) -> list[str]:
        return ["user_id", "memory_type", "text", "created_at", "metadata_json"]

    def _knowledge_expr(
        self,
        *,
        namespace: str,
        document_ids: list[str] | None,
        filters: dict[str, Any],
    ) -> str:
        clauses = [f'namespace == "{_escape(namespace)}"']
        if document_ids:
            escaped = '", "'.join(_escape(document_id) for document_id in document_ids)
            clauses.append(f'document_id in ["{escaped}"]')
        if filters.get("document_id"):
            clauses.append(f'document_id == "{_escape(filters["document_id"])}"')
        if filters.get("source_path"):
            clauses.append(f'source_path == "{_escape(filters["source_path"])}"')
        return " and ".join(clauses)

    def search_knowledge(
        self,
        *,
        query: str,
        namespace: str,
        top_k: int,
        document_ids: list[str] | None,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        dense_query, sparse_query = self.embedding_runtime.encode_query(query)
        expr = self._knowledge_expr(namespace=namespace, document_ids=document_ids, filters=filters)
        search_limit = max(top_k * 4, 12)

        dense_results = self._search_collection(
            self.chunk_collection,
            anns_field="dense_vec",
            data=[dense_query],
            limit=search_limit,
            expr=expr,
            output_fields=self._chunk_output_fields(),
            param={"metric_type": "IP", "params": {"ef": 64}},
        )
        sparse_results: list[dict[str, Any]] = []
        if sparse_query:
            sparse_results = self._search_collection(
                self.chunk_collection,
                anns_field="sparse_vec",
                data=[sparse_query],
                limit=search_limit,
                expr=expr,
                output_fields=self._chunk_output_fields(),
                param={"metric_type": "IP", "params": {}},
            )

        fused = _rrf_fusion(dense_results, sparse_results) if sparse_results else dense_results
        fused = self._rerank(query, fused)
        return [
            {
                "id": item["id"],
                "text": item.get("content", ""),
                "score": float(item.get("score", 0.0)),
                "source": "milvus_knowledge",
                "source_path": item.get("source_path"),
                "metadata": {
                    "document_id": item.get("document_id"),
                    "node_id": item.get("node_id"),
                    **item.get("metadata", {}),
                },
            }
            for item in fused[:top_k]
        ]

    def upsert_memory(self, *, user_id: str, memory_type: str, text: str, metadata: dict[str, Any]) -> None:
        dense_vec, sparse_vec = self.embedding_runtime.encode_query(text)
        row = {
            "id": f"{user_id}::{memory_type}::{uuid4().hex[:10]}",
            "user_id": user_id,
            "memory_type": memory_type,
            "text": text,
            "created_at": int(time()),
            "dense_vec": dense_vec,
            "sparse_vec": sparse_vec,
            "metadata_json": metadata,
        }
        self.memory_collection.insert([self._normalize_row(row)])
        self.memory_collection.flush()
        self.memory_collection.load()

    def search_memory(
        self,
        *,
        query: str,
        user_id: str,
        memory_type: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        dense_query, sparse_query = self.embedding_runtime.encode_query(query)
        clauses = [f'user_id == "{_escape(user_id)}"']
        if memory_type:
            clauses.append(f'memory_type == "{_escape(memory_type)}"')
        expr = " and ".join(clauses)
        search_limit = max(top_k * 4, 12)
        dense_results = self._search_collection(
            self.memory_collection,
            anns_field="dense_vec",
            data=[dense_query],
            limit=search_limit,
            expr=expr,
            output_fields=self._memory_output_fields(),
            param={"metric_type": "IP", "params": {"ef": 64}},
        )
        sparse_results: list[dict[str, Any]] = []
        if sparse_query:
            sparse_results = self._search_collection(
                self.memory_collection,
                anns_field="sparse_vec",
                data=[sparse_query],
                limit=search_limit,
                expr=expr,
                output_fields=self._memory_output_fields(),
                param={"metric_type": "IP", "params": {}},
            )

        fused = _rrf_fusion(dense_results, sparse_results) if sparse_results else dense_results
        fused = self._rerank(query, fused, text_key="text")
        return [
            {
                "id": item["id"],
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
                "source": "milvus_memory",
                "metadata": {
                    "memory_type": item.get("memory_type"),
                    **item.get("metadata", {}),
                },
            }
            for item in fused[:top_k]
        ]

    def _rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        text_key: str = "content",
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits
        limit = min(len(hits), self.settings.reranker_top_n)
        passages = [hit.get(text_key, "") for hit in hits[:limit]]
        scores = self.reranker_runtime.score(query, passages)
        reranked = list(hits)
        for hit, rerank_score in zip(reranked[:limit], scores):
            hit["score"] = float(rerank_score)
        reranked[:limit] = sorted(reranked[:limit], key=lambda item: item.get("score", 0.0), reverse=True)
        return reranked

    def _search_collection(
        self,
        collection: Collection,
        *,
        anns_field: str,
        data: list[Any],
        limit: int,
        expr: str,
        output_fields: list[str],
        param: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            results = collection.search(
                data=data,
                anns_field=anns_field,
                param=param,
                limit=limit,
                expr=expr,
                output_fields=output_fields,
            )
        except Exception:
            return []

        parsed: list[dict[str, Any]] = []
        for group in results:
            for hit in group:
                entity = getattr(hit, "entity", {})
                entity_get = entity.get if hasattr(entity, "get") else lambda key, default=None: getattr(entity, key, default)
                row = {
                    "id": str(hit.id),
                    "score": float(getattr(hit, "distance", 0.0)),
                }
                for field_name in output_fields:
                    row[field_name] = entity_get(field_name)
                metadata_json = entity_get("metadata_json")
                if metadata_json:
                    try:
                        row["metadata"] = json.loads(metadata_json)
                    except Exception:
                        row["metadata"] = {}
                else:
                    row["metadata"] = {}
                parsed.append(row)
        return parsed
