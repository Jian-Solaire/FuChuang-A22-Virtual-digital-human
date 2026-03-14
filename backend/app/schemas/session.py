"""Session and graph-facing schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.core.envelope import EnvelopeRequest, EnvelopeResponse, RequestMeta, ResponseMeta


class TextFeatures(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    emotion_cues: dict[str, float] = Field(default_factory=dict)
    topic_tags: list[str] = Field(default_factory=list)


class TextObservation(BaseModel):
    text_input: str | None = None
    asr_text: str | None = None
    merged_text: str = ""
    asr_confidence: float = 0.0
    text_features: TextFeatures = Field(default_factory=TextFeatures)


class AudioObservation(BaseModel):
    speech_rate: float = 0.0
    pause_ratio: float = 0.0
    avg_pitch: float = 0.0
    pitch_var: float = 0.0
    energy_mean: float = 0.0
    energy_var: float = 0.0
    voice_stability: float = 0.0
    prosody: dict[str, float] = Field(default_factory=dict)
    anomaly_flags: dict[str, bool] = Field(default_factory=dict)


class VideoObservation(BaseModel):
    face_detected: bool = False
    face_confidence: float = 0.0
    blink_rate: float = 0.0
    gaze_stability: float = 0.0
    head_motion: float = 0.0
    facial_tension: float = 0.0
    expression: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)


class NormalizedObservation(BaseModel):
    text_observation: TextObservation = Field(default_factory=TextObservation)
    audio_observation: AudioObservation = Field(default_factory=AudioObservation)
    video_observation: VideoObservation = Field(default_factory=VideoObservation)


class TurnInputPayload(BaseModel):
    input_type: Literal["audio_video_text", "audio_text", "text"] = "text"
    text_input: str | None = None
    audio_url: str | None = None
    video_url: str | None = None
    audio_chunk_seq: int | None = None
    is_final_chunk: bool = True
    client_state: dict[str, Any] = Field(default_factory=dict)
    normalized_observation: NormalizedObservation | None = None


class SessionCreatePayload(BaseModel):
    user_id: str
    avatar_id: str | None = None
    profile_seed: dict[str, Any] = Field(default_factory=dict)


class SessionEndPayload(BaseModel):
    save_history: bool = True


class SessionCreateRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: SessionCreatePayload


class SessionTurnRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: TurnInputPayload


class SessionEndRequest(EnvelopeRequest):
    meta: RequestMeta = Field(default_factory=RequestMeta)
    payload: SessionEndPayload


class SessionCreateResponsePayload(BaseModel):
    session_id: str
    thread_id: str
    created: bool = True


class SessionTurnResponsePayload(BaseModel):
    dialog_state: dict[str, Any]
    current_psych_state: dict[str, Any]
    retrieval_context: dict[str, Any]
    output_a: dict[str, Any]
    output_b: dict[str, Any]


class SessionEndResponsePayload(BaseModel):
    session_id: str
    ended: bool = True


class SessionCreateResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: SessionCreateResponsePayload


class SessionTurnResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: SessionTurnResponsePayload


class SessionEndResponse(EnvelopeResponse):
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    payload: SessionEndResponsePayload
