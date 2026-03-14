from backend.app.config import get_settings
from backend.app.graph.nodes import GraphDependencies, writeback_node
from backend.app.services.deepseek_client import DeepSeekClient
from backend.app.services.memory_store import MemoryStore
from backend.app.services.milvus_adapter import MilvusAdapter
from backend.app.services.pageindex_adapter import PageIndexAdapter
from backend.app.services.retrieval_orchestrator import RetrievalOrchestrator


def test_writeback_updates_memory_layers(tmp_path):
    memory_store = MemoryStore()
    memory_store.create_session("sess_1", {"profile_seed": {}, "avatar_id": None})
    milvus = MilvusAdapter()
    settings = get_settings()
    deps = GraphDependencies(
        settings=settings,
        memory_store=memory_store,
        retrieval_orchestrator=RetrievalOrchestrator(milvus, PageIndexAdapter(tmp_path)),
        deepseek_client=DeepSeekClient(settings),
    )
    state = {
        "meta": {"session_id": "sess_1", "user_id": "u001"},
        "output_a": {"response": {"reply_text": "hello"}},
        "output_b": {
            "dialog_state_update": {"stage": "exploration"},
            "state_record": {"current_psych_state": {"emotion": {"dominant": "anxiety"}}},
            "memory_update": {
                "short_term_summary": "summary",
                "facts_to_store": [{"key": "stress", "value": "high"}],
            },
            "profile_update": {"preferred_style": "supportive_gentle"},
        },
    }
    writeback_node(state, deps)
    session = memory_store.get_session("sess_1")
    assert session["recent_summaries"][-1] == "summary"
    assert milvus.long_term_memory["u001"]["episodic_memory"]
