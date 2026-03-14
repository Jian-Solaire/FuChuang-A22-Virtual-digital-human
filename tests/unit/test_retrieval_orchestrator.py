from pathlib import Path

from backend.app.services.milvus_adapter import MilvusAdapter
from backend.app.services.pageindex_adapter import PageIndexAdapter
from backend.app.services.retrieval_orchestrator import RetrievalOrchestrator


def build_orchestrator(tmp_path: Path) -> RetrievalOrchestrator:
    return RetrievalOrchestrator(MilvusAdapter(), PageIndexAdapter(tmp_path))


def test_auto_route_prefers_pageindex_for_long_docs(tmp_path: Path):
    orchestrator = build_orchestrator(tmp_path)
    base = orchestrator.create_base(
        name="Manual",
        description=None,
        namespace="default",
        route="auto",
        knowledge_type="manual",
        tags=[],
    )
    orchestrator.ingest_documents(
        base.base_id,
        [{"title": "睡眠手册", "content": "# 睡眠卫生\n保持规律作息，减少压力。", "metadata": {}}],
    )
    result = orchestrator.query(
        query="睡眠 手册 压力",
        mode="auto",
        knowledge_type="manual",
        document_scope="long_doc",
        filters={"base_id": base.base_id},
        debug_trace=True,
    )
    assert result["route_used"] == "hybrid"
    assert result["pageindex_hits"]


def test_memory_query_returns_memory_hits(tmp_path: Path):
    orchestrator = build_orchestrator(tmp_path)
    orchestrator.milvus.apply_output_b(
        "u001",
        {
            "memory_update": {"facts_to_store": [{"key": "sleep_problem", "value": True, "confidence": 0.9}]},
            "profile_update": {},
        },
    )
    result = orchestrator.query(
        query="sleep_problem",
        mode="auto",
        knowledge_type="memory",
        filters={"user_id": "u001", "memory_type": "episodic_memory"},
    )
    assert result["memory_hits"]
    assert result["source_breakdown"]["memory_hits"] == 1
