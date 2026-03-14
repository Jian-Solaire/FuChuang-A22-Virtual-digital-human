"""Unified retrieval routing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from backend.app.services.milvus_adapter import KnowledgeBaseRecord, MilvusAdapter
from backend.app.services.pageindex_adapter import PageIndexAdapter


LONG_DOC_TYPES = {"manual", "guide", "book", "policy", "long_doc"}
MEMORY_TYPES = {"memory", "profile", "recent_context"}


class RetrievalOrchestrator:
    def __init__(self, milvus: MilvusAdapter, pageindex: PageIndexAdapter) -> None:
        self.milvus = milvus
        self.pageindex = pageindex
        self.knowledge_bases: dict[str, KnowledgeBaseRecord] = {}

    def create_base(
        self,
        *,
        name: str,
        description: str | None,
        namespace: str,
        route: str,
        knowledge_type: str,
        tags: list[str],
    ) -> KnowledgeBaseRecord:
        if route == "auto":
            route = "pageindex" if knowledge_type in LONG_DOC_TYPES else "milvus"
        base = KnowledgeBaseRecord(
            base_id=f"kb_{uuid4().hex[:10]}",
            name=name,
            description=description,
            namespace=namespace,
            route=route,
            knowledge_type=knowledge_type,
            tags=list(tags),
        )
        self.knowledge_bases[base.base_id] = base
        self.milvus.register_base(base)
        return base

    def get_base(self, base_id: str) -> KnowledgeBaseRecord | None:
        return self.knowledge_bases.get(base_id)

    def get_base_status(self, base_id: str) -> dict[str, Any]:
        base = self.get_base(base_id)
        if not base:
            return {
                "base_id": base_id,
                "status": "missing",
                "route": "unknown",
                "document_count": 0,
                "pageindex_cache_count": 0,
            }
        milvus_status = self.milvus.base_status(base_id)
        pageindex_status = self.pageindex.base_status(base_id)
        return {
            "base_id": base.base_id,
            "status": "ready",
            "route": base.route,
            "document_count": milvus_status.get("document_count", 0),
            "pageindex_cache_count": pageindex_status.get("pageindex_cache_count", 0),
        }

    def ingest_documents(self, base_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        base = self.get_base(base_id)
        if not base:
            raise KeyError(f"Unknown knowledge base: {base_id}")

        normalized = []
        for index, document in enumerate(documents, start=1):
            normalized.append(
                {
                    "id": f"{base_id}_doc_{index + len(self.milvus.documents.get(base_id, []))}",
                    **document,
                }
            )

        milvus_count = 0
        pageindex_count = 0
        if base.route in {"milvus", "hybrid"}:
            milvus_count = self.milvus.ingest_documents(base_id, normalized)
        if base.route in {"pageindex", "hybrid"}:
            pageindex_count = self.pageindex.ingest_documents(base_id, normalized)

        return {
            "base_id": base.base_id,
            "route": base.route,
            "ingested_count": max(milvus_count, pageindex_count),
            "pageindex_cache_count": pageindex_count,
        }

    def _should_add_pageindex(
        self,
        *,
        mode: str,
        knowledge_type: str | None,
        document_scope: str | None,
        milvus_hits: list[dict[str, Any]],
    ) -> bool:
        if mode == "pageindex":
            return True
        if mode == "milvus":
            return False
        if knowledge_type in MEMORY_TYPES:
            return False
        if knowledge_type in LONG_DOC_TYPES or document_scope == "long_doc":
            return True
        return len(milvus_hits) < 2

    def query(
        self,
        *,
        query: str,
        mode: str = "auto",
        knowledge_type: str | None = None,
        document_scope: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        pageindex_top_k: int = 3,
        debug_trace: bool = False,
    ) -> dict[str, Any]:
        filters = filters or {}
        memory_hits: list[dict[str, Any]] = []
        knowledge_hits: list[dict[str, Any]] = []
        pageindex_hits: list[dict[str, Any]] = []
        trace: dict[str, Any] = {"route_decisions": []}

        route_used = mode
        if mode in {"auto", "milvus"} or knowledge_type in MEMORY_TYPES:
            memory_only = knowledge_type in MEMORY_TYPES
            milvus_hits = self.milvus.query(
                query,
                top_k=top_k,
                filters=filters,
                memory_only=memory_only,
            )
            if memory_only:
                memory_hits = milvus_hits
            else:
                knowledge_hits = milvus_hits
            trace["route_decisions"].append("milvus")
        else:
            milvus_hits = []

        if self._should_add_pageindex(
            mode=mode,
            knowledge_type=knowledge_type,
            document_scope=document_scope,
            milvus_hits=milvus_hits,
        ):
            pageindex_response = self.pageindex.query(
                query,
                top_k=pageindex_top_k,
                filters=filters,
            )
            pageindex_hits = pageindex_response["results"]
            trace["route_decisions"].append("pageindex")
            trace.update(pageindex_response["trace"])
            if mode == "auto":
                route_used = "hybrid"
        elif mode == "auto":
            route_used = "milvus"

        source_breakdown = {
            "memory_hits": len(memory_hits),
            "knowledge_hits": len(knowledge_hits),
            "pageindex_hits": len(pageindex_hits),
        }
        return {
            "memory_hits": memory_hits,
            "knowledge_hits": knowledge_hits,
            "pageindex_hits": pageindex_hits,
            "trace": trace if debug_trace else {"route_decisions": trace["route_decisions"]},
            "route_used": route_used,
            "source_breakdown": source_breakdown,
        }

    def debug_query_milvus(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        hits = self.milvus.query(query, top_k=top_k, filters=filters or {})
        return {
            "route_used": "milvus",
            "memory_hits": [],
            "knowledge_hits": hits,
            "pageindex_hits": [],
            "trace": {"route_decisions": ["milvus"]},
            "source_breakdown": {"memory_hits": 0, "knowledge_hits": len(hits), "pageindex_hits": 0},
        }

    def debug_query_pageindex(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        result = self.pageindex.query(query, top_k=top_k, filters=filters or {})
        return {
            "route_used": "pageindex",
            "memory_hits": [],
            "knowledge_hits": [],
            "pageindex_hits": result["results"],
            "trace": result["trace"],
            "source_breakdown": {"memory_hits": 0, "knowledge_hits": 0, "pageindex_hits": len(result["results"])},
        }
