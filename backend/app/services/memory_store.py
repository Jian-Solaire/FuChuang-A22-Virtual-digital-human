"""Short-term session memory with optional Redis hook points."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MemoryStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = {
            "session_id": session_id,
            "dialog_state": {
                "stage": "initial_contact",
                "rapport_level": 0.3,
                "user_engagement": 0.5,
                "need_follow_up": True,
                "last_intervention_type": None,
            },
            "baseline_state": {
                "emotion": {"valence": 0.5, "arousal": 0.5, "dominant": "neutral"},
                "risk": {
                    "anxiety_tendency": 0.2,
                    "depression_tendency": 0.1,
                    "bipolar_risk": 0.0,
                    "crisis_risk": 0.0,
                },
                "interaction": {
                    "engagement": 0.5,
                    "openness": 0.4,
                    "defensiveness": 0.4,
                },
                "confidence": 0.5,
            },
            "recent_summaries": [],
            "recent_outputs": [],
            "ended": False,
            "profile_seed": deepcopy(payload.get("profile_seed", {})),
            "avatar_id": payload.get("avatar_id"),
        }
        self._sessions[session_id] = session
        return deepcopy(session)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.get(session_id)
        return deepcopy(session) if session else None

    def save_turn(
        self,
        session_id: str,
        *,
        dialog_state: dict[str, Any],
        baseline_state: dict[str, Any],
        short_term_summary: str,
        output_a: dict[str, Any],
    ) -> None:
        session = self._sessions.setdefault(session_id, {})
        session["dialog_state"] = deepcopy(dialog_state)
        session["baseline_state"] = deepcopy(baseline_state)
        session.setdefault("recent_summaries", []).append(short_term_summary)
        session["recent_summaries"] = session["recent_summaries"][-3:]
        session.setdefault("recent_outputs", []).append(deepcopy(output_a))
        session["recent_outputs"] = session["recent_outputs"][-3:]

    def end_session(self, session_id: str, save_history: bool = True) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session["ended"] = True
        session["save_history"] = save_history
        return True
