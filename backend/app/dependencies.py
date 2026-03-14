"""Dependency container."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.config import Settings, get_settings
from backend.app.graph.builder import build_graph
from backend.app.graph.nodes import GraphDependencies
from backend.app.services.deepseek_client import DeepSeekClient
from backend.app.services.memory_store import MemoryStore
from backend.app.services.milvus_adapter import MilvusAdapter
from backend.app.services.pageindex_adapter import PageIndexAdapter
from backend.app.services.retrieval_orchestrator import RetrievalOrchestrator


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    memory_store: MemoryStore
    milvus: MilvusAdapter
    pageindex: PageIndexAdapter
    retrieval_orchestrator: RetrievalOrchestrator
    deepseek_client: DeepSeekClient
    graph: object


def build_container() -> AppContainer:
    settings = get_settings()
    memory_store = MemoryStore()
    milvus = MilvusAdapter(settings=settings)
    pageindex = PageIndexAdapter(settings.pageindex_cache_dir)
    retrieval_orchestrator = RetrievalOrchestrator(milvus, pageindex)
    deepseek_client = DeepSeekClient(settings)
    graph = build_graph(
        GraphDependencies(
            settings=settings,
            memory_store=memory_store,
            retrieval_orchestrator=retrieval_orchestrator,
            deepseek_client=deepseek_client,
        )
    )
    return AppContainer(
        settings=settings,
        memory_store=memory_store,
        milvus=milvus,
        pageindex=pageindex,
        retrieval_orchestrator=retrieval_orchestrator,
        deepseek_client=deepseek_client,
        graph=graph,
    )
