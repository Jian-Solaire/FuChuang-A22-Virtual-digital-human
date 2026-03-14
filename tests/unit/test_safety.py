from backend.app.config import get_settings
from backend.app.graph.nodes import GraphDependencies, safety_node
from backend.app.services.deepseek_client import DeepSeekClient
from backend.app.services.memory_store import MemoryStore
from backend.app.services.milvus_adapter import MilvusAdapter
from backend.app.services.pageindex_adapter import PageIndexAdapter
from backend.app.services.retrieval_orchestrator import RetrievalOrchestrator


def test_safety_node_rewrites_high_risk_output(tmp_path):
    settings = get_settings()
    deps = GraphDependencies(
        settings=settings,
        memory_store=MemoryStore(),
        retrieval_orchestrator=RetrievalOrchestrator(MilvusAdapter(), PageIndexAdapter(tmp_path)),
        deepseek_client=DeepSeekClient(settings),
    )
    state = {
        "perception_result": {"text_observation": {"merged_text": "我不想活了"}},
        "current_psych_state": {"risk": {"crisis_risk": 0.95}},
        "output_a": {"response": {"reply_text": "原始回复"}},
        "output_b": {},
    }
    result = safety_node(state, deps)
    assert result["output_a"]["response"]["reply_type"] == "safety_support"
    assert result["output_b"]["risk_event"]["trigger"] in {"crisis_keyword", f"crisis_risk>={settings.risk_threshold}"}
