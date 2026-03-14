"""DeepSeek client with deterministic mock fallback."""

from __future__ import annotations

import json
from typing import Any

import httpx

from backend.app.config import Settings
from backend.app.graph.prompts import OUTPUT_A_PROMPT, OUTPUT_B_PROMPT, SYSTEM_PROMPT_CORE


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _mock_output_a(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        psych_state = graph_input["current_psych_state"]
        dominant = psych_state["emotion"]["dominant"]
        merged_text = graph_input["perception_result"]["text_observation"]["merged_text"]
        reply = (
            "听起来你现在承受了不少压力，我会陪你一步一步说清楚。"
            if dominant == "anxiety"
            else "谢谢你愿意继续说下去，我们可以慢慢梳理你的感受。"
        )
        question = "最近哪件事最让你放不下？" if merged_text else "你现在最想先从哪件事说起？"
        return {
            "dialog_state": graph_input["previous_context"]["dialog_state"],
            "response": {
                "reply_text": f"{reply}{question}",
                "reply_type": "empathy_plus_probe",
                "follow_up_question": question,
                "tts_style": {"emotion": "warm", "speed": 0.95, "pause_level": "medium"},
                "avatar_action": {"preset": "gentle_listen", "intensity": 0.4},
            },
        }

    def _mock_output_b(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        current = graph_input["current_psych_state"]
        reassessment = graph_input["reassessment_result"]
        summary = graph_input["retrieval_context"]
        return {
            "reassessment_result": reassessment,
            "state_record": {
                "current_psych_state": current,
                "store_as_next_turn_baseline": True,
            },
            "memory_update": {
                "short_term_summary": "用户本轮表达了压力与情绪线索，系统建议继续温和探查。",
                "facts_to_store": [
                    {"type": "topic", "key": "dominant_emotion", "value": current["emotion"]["dominant"], "confidence": current["confidence"]},
                    {"type": "retrieval", "key": "route_used", "value": summary["route_used"], "confidence": 1.0},
                ],
            },
            "profile_update": {"preferred_style": "supportive_gentle", "communication_speed": "slow_medium"},
            "dialog_state_update": {
                "stage": "exploration",
                "substage": "stress_probe",
                "rapport_level": min(0.95, graph_input["previous_context"]["dialog_state"].get("rapport_level", 0.4) + 0.06),
                "user_engagement": current["interaction"]["engagement"],
                "need_follow_up": True,
                "last_intervention_type": "supportive_exploration",
            },
            "next_turn_plan": {
                "baseline_state": current,
                "expected_goal": "明确当前压力来源，并继续观察风险变化",
                "intervention_type": "supportive_exploration",
                "next_observation_focus": ["是否愿意展开诱因", "紧张程度是否缓和", "是否出现高风险线索"],
            },
        }

    def _call_json_completion(
        self,
        *,
        instruction: str,
        graph_input: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if self.settings.deepseek_mock_only or not self.settings.deepseek_api_key:
            return fallback

        payload = {
            "model": self.settings.deepseek_model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CORE},
                {
                    "role": "user",
                    "content": (
                        f"{instruction}\n"
                        "Return only valid JSON.\n"
                        "Fill every required field and keep structure stable.\n"
                        f"Input:\n{json.dumps(graph_input, ensure_ascii=False)}"
                    ),
                },
            ],
        }
        try:
            with httpx.Client(
                base_url=self.settings.deepseek_base_url.rstrip("/"),
                timeout=self.settings.deepseek_timeout_s,
            ) as client:
                response = client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _deep_merge(fallback, _extract_json(content))
        except Exception:
            return fallback

    def generate_output_a(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        fallback = self._mock_output_a(graph_input)
        return self._call_json_completion(
            instruction=(
                f"{OUTPUT_A_PROMPT}\n"
                "JSON schema:\n"
                "{"
                '"dialog_state": object, '
                '"response": {'
                '"reply_text": string, '
                '"reply_type": string, '
                '"follow_up_question": string, '
                '"tts_style": object, '
                '"avatar_action": object'
                "}"
                "}"
            ),
            graph_input=graph_input,
            fallback=fallback,
        )

    def generate_output_b(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        fallback = self._mock_output_b(graph_input)
        return self._call_json_completion(
            instruction=(
                f"{OUTPUT_B_PROMPT}\n"
                "JSON schema:\n"
                "{"
                '"reassessment_result": object, '
                '"state_record": object, '
                '"memory_update": object, '
                '"profile_update": object, '
                '"dialog_state_update": object, '
                '"next_turn_plan": object'
                "}"
            ),
            graph_input=graph_input,
            fallback=fallback,
        )

    def rewrite_safety_output_a(self, output_a: dict[str, Any], *, reason: str) -> dict[str, Any]:
        safe_text = (
            "我注意到你刚才提到的内容可能涉及较高风险。"
            "如果你现在有伤害自己或他人的冲动，请立刻联系当地紧急热线、身边可信任的人，或尽快寻求专业帮助。"
            "如果你愿意，我也可以先陪你把最紧急的部分说清楚。"
        )
        rewritten = dict(output_a)
        rewritten["response"] = dict(output_a.get("response", {}))
        rewritten["response"]["reply_text"] = safe_text
        rewritten["response"]["reply_type"] = "safety_support"
        rewritten["response"]["follow_up_question"] = "你现在身边有没有可以立刻联系的人？"
        rewritten.setdefault("safety", {})["reason"] = reason
        return rewritten
