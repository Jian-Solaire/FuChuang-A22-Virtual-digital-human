"""Milvus adapter with mock-first fallback and optional real backend."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import Settings
from backend.app.services.text_match import token_score


@dataclass(slots=True)
class KnowledgeBaseRecord:
    base_id: str
    name: str
    description: str | None
    namespace: str
    route: str
    knowledge_type: str
    tags: list[str] = field(default_factory=list)


class MilvusAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self.documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.long_term_memory: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"profile_memory": [], "episodic_memory": [], "risk_history": []}
        )
        self.real_backend: Any | None = None
        if settings and settings.milvus_enabled:
            try:
                from backend.app.services.real_milvus_backend import RealMilvusBackend

                self.real_backend = RealMilvusBackend(settings)
            except Exception:
                self.real_backend = None

    def register_base(self, record: KnowledgeBaseRecord) -> None:
        self.knowledge_bases[record.base_id] = record

    def get_base(self, base_id: str) -> KnowledgeBaseRecord | None:
        return self.knowledge_bases.get(base_id)

    def ingest_documents(self, base_id: str, documents: list[dict[str, Any]]) -> int:
        self.documents[base_id].extend(deepcopy(documents))
        base = self.get_base(base_id)
        if self.real_backend and base:
            for document in documents:
                metadata = dict(document.get("metadata", {}))
                metadata.update({"base_id": base_id, "base_name": base.name, "knowledge_type": base.knowledge_type})
                self.real_backend.upsert_document(
                    namespace=base.namespace,
                    document_id=document["id"],
                    title=document.get("title", document["id"]),
                    content=document.get("content") or document.get("title", ""),
                    source_path=document.get("source_path"),
                    metadata=metadata,
                )
        return len(documents)

    def base_status(self, base_id: str) -> dict[str, Any]:
        base = self.get_base(base_id)
        return {
            "base_id": base_id,
            "status": "ready" if base else "missing",
            "route": base.route if base else "unknown",
            "document_count": len(self.documents.get(base_id, [])),
        }

    def _mock_query_knowledge(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        base_id = filters.get("base_id")
        candidate_base_ids = [base_id] if base_id else list(self.documents.keys())
        for candidate_base_id in candidate_base_ids:
            for document in self.documents.get(candidate_base_id, []):
                score = token_score(query, document.get("content", "") or document.get("title", ""))
                if score <= 0:
                    continue
                hits.append(
                    {
                        "id": document["id"],
                        "text": document.get("content") or document.get("title", ""),
                        "score": score,
                        "source": "milvus_knowledge",
                        "source_path": document.get("source_path"),
                        "metadata": {
                            "base_id": candidate_base_id,
                            **document.get("metadata", {}),
                        },
                    }
                )
        return sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]

    def _mock_query_memory(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        user_id = filters.get("user_id")
        if not user_id:
            return []
        memories = self.long_term_memory.get(user_id, {})
        memory_type = filters.get("memory_type")
        buckets = [memory_type] if memory_type else list(memories.keys())
        hits: list[dict[str, Any]] = []
        for bucket in buckets:
            for item in memories.get(bucket, []):
                score = token_score(query, item.get("text", ""))
                if score <= 0:
                    continue
                hits.append(
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "score": score,
                        "source": "milvus_memory",
                        "metadata": {"memory_type": bucket, **item.get("metadata", {})},
                    }
                )
        return sorted(hits, key=lambda item: item["score"], reverse=True)[:top_k]

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        memory_only: bool = False,
    ) -> list[dict[str, Any]]:
        filters = filters or {}

        if memory_only:
            if self.real_backend and filters.get("user_id"):
                hits = self.real_backend.search_memory(
                    query=query,
                    user_id=filters["user_id"],
                    memory_type=filters.get("memory_type"),
                    top_k=top_k,
                )
                if hits:
                    return hits
            return self._mock_query_memory(query, top_k=top_k, filters=filters)

        if self.real_backend:
            base_id = filters.get("base_id")
            base = self.get_base(base_id) if base_id else None
            namespace = base.namespace if base else (self.settings.default_namespace if self.settings else "default")
            document_ids = None
            if base_id:
                document_ids = [document["id"] for document in self.documents.get(base_id, [])]
            hits = self.real_backend.search_knowledge(
                query=query,
                namespace=namespace,
                top_k=top_k,
                document_ids=document_ids,
                filters=filters,
            )
            if hits:
                return hits

        return self._mock_query_knowledge(query, top_k=top_k, filters=filters)

    def load_context(self, user_id: str | None) -> dict[str, Any]:
        if not user_id:
            return {
                "user_profile": {},
                "risk_history": [],
                "episodic_memory": [],
            }
        memory = self.long_term_memory.get(user_id, {})
        return {
            "user_profile": deepcopy(memory.get("profile_memory", [])[-3:]),
            "risk_history": deepcopy(memory.get("risk_history", [])[-5:]),
            "episodic_memory": deepcopy(memory.get("episodic_memory", [])[-5:]),
        }

    def apply_output_b(self, user_id: str | None, output_b: dict[str, Any]) -> None:
        if not user_id:
            return
        memory = self.long_term_memory[user_id]

        for fact in output_b.get("memory_update", {}).get("facts_to_store", []):
            row = {
                "id": f"epi_{len(memory['episodic_memory']) + 1}",
                "text": f"{fact.get('key')}: {fact.get('value')}",
                "metadata": fact,
            }
            memory["episodic_memory"].append(row)
            if self.real_backend:
                self.real_backend.upsert_memory(
                    user_id=user_id,
                    memory_type="episodic_memory",
                    text=row["text"],
                    metadata=fact,
                )

        if output_b.get("profile_update"):
            row = {
                "id": f"profile_{len(memory['profile_memory']) + 1}",
                "text": str(output_b["profile_update"]),
                "metadata": output_b["profile_update"],
            }
            memory["profile_memory"].append(row)
            if self.real_backend:
                self.real_backend.upsert_memory(
                    user_id=user_id,
                    memory_type="profile_memory",
                    text=row["text"],
                    metadata=output_b["profile_update"],
                )

        risk_event = output_b.get("risk_event")
        if risk_event:
            row = {
                "id": f"risk_{len(memory['risk_history']) + 1}",
                "text": risk_event.get("summary", "risk_event"),
                "metadata": risk_event,
            }
            memory["risk_history"].append(row)
            if self.real_backend:
                self.real_backend.upsert_memory(
                    user_id=user_id,
                    memory_type="risk_history",
                    text=row["text"],
                    metadata=risk_event,
                )
