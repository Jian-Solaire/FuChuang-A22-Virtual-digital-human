"""Graph node implementations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from backend.app.config import Settings
from backend.app.services.deepseek_client import DeepSeekClient
from backend.app.services.memory_store import MemoryStore
from backend.app.services.retrieval_orchestrator import RetrievalOrchestrator


ANXIETY_KEYWORDS = {"睡不着", "压力", "焦虑", "紧张", "stress", "anxiety"}
CRISIS_KEYWORDS = {"不想活", "自杀", "伤害自己", "结束生命", "伤害他人"}


@dataclass(slots=True)
class GraphDependencies:
    settings: Settings
    memory_store: MemoryStore
    retrieval_orchestrator: RetrievalOrchestrator
    deepseek_client: DeepSeekClient


def _extract_keywords(text: str) -> list[str]:
    keywords = []
    for candidate in ANXIETY_KEYWORDS | CRISIS_KEYWORDS:
        if candidate in text:
            keywords.append(candidate)
    return keywords


def load_context(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    session_id = state["meta"]["session_id"]
    user_id = state["meta"].get("user_id")
    session = deps.memory_store.get_session(session_id) or {}
    long_term = deps.retrieval_orchestrator.milvus.load_context(user_id)
    previous_context = {
        "baseline_state": session.get("baseline_state", {}),
        "dialog_state": session.get("dialog_state", {}),
        "memory_context": {
            "recent_summary": "\n".join(session.get("recent_summaries", [])[-1:]),
            "recent_outputs": session.get("recent_outputs", []),
        },
        "user_profile": session.get("profile_seed", {}),
        "long_term_context": long_term,
    }
    state["previous_context"] = previous_context
    return state


def perception_stub(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    turn_input = state["turn_input"]
    if turn_input.get("normalized_observation"):
        perception = turn_input["normalized_observation"]
    else:
        text = turn_input.get("text_input") or ""
        keywords = _extract_keywords(text)
        anxiety_hint = 0.78 if any(word in text for word in ANXIETY_KEYWORDS) else 0.2
        crisis_hint = 0.92 if any(word in text for word in CRISIS_KEYWORDS) else 0.02
        perception = {
            "text_observation": {
                "text_input": turn_input.get("text_input"),
                "asr_text": turn_input.get("text_input"),
                "merged_text": text,
                "asr_confidence": 0.95 if text else 0.0,
                "text_features": {
                    "keywords": keywords,
                    "emotion_cues": {
                        "anxiety_hint": anxiety_hint,
                        "depression_hint": 0.2,
                        "hopelessness_hint": crisis_hint,
                    },
                    "topic_tags": ["stress"] if keywords else [],
                },
            },
            "audio_observation": {
                "speech_rate": 4.1,
                "pause_ratio": 0.18,
                "avg_pitch": 206.0,
                "pitch_var": 31.0,
                "energy_mean": 0.62,
                "energy_var": 0.12,
                "voice_stability": 0.45,
                "prosody": {"arousal": 0.72, "valence": 0.33, "tension": anxiety_hint},
                "anomaly_flags": {"long_silence": False, "sudden_speed_change": False, "abnormal_volume": False},
            },
            "video_observation": {
                "face_detected": True,
                "face_confidence": 0.96,
                "blink_rate": 0.31,
                "gaze_stability": 0.42,
                "head_motion": 0.58,
                "facial_tension": anxiety_hint,
                "expression": {"valence": 0.3, "arousal": 0.61, "dominant": "tense"},
                "quality": {"occlusion": 0.1, "lighting_ok": True, "usable": True},
            },
        }
    state["perception_result"] = deepcopy(perception)
    return state


def fusion_node(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    perception = state["perception_result"]
    text_obs = perception["text_observation"]
    audio_obs = perception["audio_observation"]
    video_obs = perception["video_observation"]
    anxiety = round(
        min(
            1.0,
            text_obs["text_features"]["emotion_cues"].get("anxiety_hint", 0.0) * 0.5
            + audio_obs["prosody"].get("tension", 0.0) * 0.3
            + float(video_obs.get("facial_tension", 0.0)) * 0.2,
        ),
        4,
    )
    crisis = 0.06
    if any(term in text_obs["merged_text"] for term in CRISIS_KEYWORDS):
        crisis = 0.9
    current = {
        "emotion": {
            "valence": 0.3 if anxiety > 0.5 else 0.48,
            "arousal": max(0.3, anxiety),
            "dominant": "anxiety" if anxiety >= 0.45 else "neutral",
        },
        "risk": {
            "anxiety_tendency": anxiety,
            "depression_tendency": 0.28 if anxiety > 0.5 else 0.12,
            "bipolar_risk": 0.05,
            "crisis_risk": crisis,
        },
        "interaction": {
            "engagement": 0.72 if text_obs["merged_text"] else 0.45,
            "openness": 0.62 if text_obs["merged_text"] else 0.35,
            "defensiveness": 0.28 if text_obs["merged_text"] else 0.55,
        },
        "symptoms": {
            "sleep_problem": {"value": "睡不着" in text_obs["merged_text"], "confidence": 0.86 if "睡不着" in text_obs["merged_text"] else 0.2},
            "stress_increase": {"value": "压力" in text_obs["merged_text"], "confidence": 0.84 if "压力" in text_obs["merged_text"] else 0.2},
        },
        "confidence": 0.79,
        "evidence": {
            "text": text_obs["text_features"]["keywords"],
            "audio": ["语气偏紧张"] if anxiety > 0.4 else [],
            "video": ["面部紧张"] if video_obs.get("facial_tension", 0.0) > 0.4 else [],
        },
        "fusion_comment": "文本与音视频线索共同提示当前存在紧张和压力倾向。",
    }
    state["current_psych_state"] = current
    return state


def reassessment_node(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    baseline = state["previous_context"].get("baseline_state", {})
    baseline_risk = baseline.get("risk", {})
    baseline_interaction = baseline.get("interaction", {})
    current = state["current_psych_state"]
    result = {
        "based_on_last_turn_intervention": state["previous_context"]["dialog_state"].get("last_intervention_type"),
        "goal_achievement": "partial",
        "trend": "slightly_positive" if current["interaction"]["defensiveness"] < baseline_interaction.get("defensiveness", 0.4) else "steady",
        "state_change": {
            "anxiety_tendency_delta": round(current["risk"]["anxiety_tendency"] - baseline_risk.get("anxiety_tendency", 0.0), 4),
            "depression_tendency_delta": round(current["risk"]["depression_tendency"] - baseline_risk.get("depression_tendency", 0.0), 4),
            "engagement_delta": round(current["interaction"]["engagement"] - baseline_interaction.get("engagement", 0.0), 4),
            "defensiveness_delta": round(current["interaction"]["defensiveness"] - baseline_interaction.get("defensiveness", 0.0), 4),
        },
        "judgement": "用户愿意继续表达，建议保持温和探查并持续关注风险变化。",
    }
    state["reassessment_result"] = result
    return state


def retrieval_node(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    text = state["perception_result"]["text_observation"]["merged_text"]
    meta = state["meta"]
    knowledge_type = "recent_context"
    if "手册" in text or "指南" in text:
        knowledge_type = "manual"
    retrieval_context = deps.retrieval_orchestrator.query(
        query=text or "supportive context",
        mode="auto",
        knowledge_type=knowledge_type,
        document_scope="long_doc" if knowledge_type == "manual" else None,
        filters={
            "user_id": meta.get("user_id"),
            "session_id": meta.get("session_id"),
        },
        top_k=5,
        pageindex_top_k=3,
        debug_trace=True,
    )
    state["retrieval_context"] = retrieval_context
    return state


def decision_output_a(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    graph_input = {
        "previous_context": state["previous_context"],
        "perception_result": state["perception_result"],
        "current_psych_state": state["current_psych_state"],
        "retrieval_context": state["retrieval_context"],
    }
    state["output_a"] = deps.deepseek_client.generate_output_a(graph_input)
    return state


def decision_output_b(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    graph_input = {
        "previous_context": state["previous_context"],
        "perception_result": state["perception_result"],
        "current_psych_state": state["current_psych_state"],
        "retrieval_context": state["retrieval_context"],
        "reassessment_result": state["reassessment_result"],
    }
    state["output_b"] = deps.deepseek_client.generate_output_b(graph_input)
    return state


def safety_node(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    merged_text = state["perception_result"]["text_observation"]["merged_text"]
    crisis_risk = state["current_psych_state"]["risk"]["crisis_risk"]
    trigger = None
    if crisis_risk >= deps.settings.risk_threshold:
        trigger = f"crisis_risk>={deps.settings.risk_threshold}"
    if any(term in merged_text for term in CRISIS_KEYWORDS):
        trigger = "crisis_keyword"

    if not trigger:
        return state

    state["output_a"] = deps.deepseek_client.rewrite_safety_output_a(state["output_a"], reason=trigger)
    state["output_b"]["risk_event"] = {
        "summary": "Detected elevated crisis risk during session turn",
        "trigger": trigger,
        "crisis_risk": crisis_risk,
    }
    state["output_b"]["escalation_advice"] = {
        "recommended": True,
        "action": "provide_hotline_and_human_support",
    }
    state["output_b"]["hotline_hint"] = {
        "country": "local",
        "message": "Offer local emergency and crisis hotline guidance.",
    }
    return state


def writeback_node(state: dict[str, Any], deps: GraphDependencies) -> dict[str, Any]:
    session_id = state["meta"]["session_id"]
    user_id = state["meta"].get("user_id")
    output_b = state["output_b"]
    deps.memory_store.save_turn(
        session_id,
        dialog_state=output_b["dialog_state_update"],
        baseline_state=output_b["state_record"]["current_psych_state"],
        short_term_summary=output_b["memory_update"]["short_term_summary"],
        output_a=state["output_a"],
    )
    deps.retrieval_orchestrator.milvus.apply_output_b(user_id, output_b)
    return state
