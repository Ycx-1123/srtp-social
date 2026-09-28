from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain import EvidenceStatus


LiveStatus = Literal["ready", "running", "stopped", "error"]
ListeningState = Literal["idle", "listening", "transcribing", "analyzing", "unavailable"]
TreeMode = Literal["observing", "friendly", "signal", "risk", "recovering"]


class FaceBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class VisionObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    status: Literal["idle", "detected", "no_face", "unavailable", "error"] = "idle"
    face_detected: bool = False
    box: FaceBox | None = None
    expression: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    features: dict[str, float | str | bool] = Field(default_factory=dict)
    provenance: Literal["measured"] = "measured"

    @classmethod
    def neutral(cls, at_ms: int = 0) -> "VisionObservation":
        return cls(at_ms=at_ms)

    @classmethod
    def no_face(cls, at_ms: int) -> "VisionObservation":
        return cls(at_ms=max(0, at_ms), status="no_face")


class AudioObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    status: Literal["idle", "listening", "speech", "silent", "unavailable", "error"] = "idle"
    rms: float = Field(default=0.0, ge=0.0, le=1.0)
    peak: float = Field(default=0.0, ge=0.0, le=1.0)
    speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    pace: float = Field(default=0.0, ge=0.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    features: dict[str, float | str | bool] = Field(default_factory=dict)
    provenance: Literal["measured"] = "measured"

    @classmethod
    def neutral(cls, at_ms: int = 0) -> "AudioObservation":
        return cls(at_ms=at_ms)


class TranscriptObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    status: Literal[
        "idle",
        "transcribing",
        "completed",
        "silent",
        "too_short",
        "stale",
        "unavailable",
        "error",
    ] = "idle"
    text: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    peak_volume: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    provenance: Literal["measured"] = "measured"

    @classmethod
    def neutral(cls, at_ms: int = 0) -> "TranscriptObservation":
        return cls(at_ms=at_ms)


class TreeState(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: TreeMode
    health: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    bloom: float = Field(ge=0.0, le=1.0)
    wind: float = Field(ge=0.0, le=1.0)

    @classmethod
    def observing(cls) -> "TreeState":
        return cls(mode="observing", health=0.55, risk=0.0, bloom=0.08, wind=0.12)


class LiveState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: LiveStatus
    listening_state: ListeningState
    elapsed_ms: int = Field(default=0, ge=0)
    last_frame_at_ms: int | None = Field(default=None, ge=0)
    signal_sequence: int = Field(default=0, ge=0)
    signal_activity: float = Field(default=0.0, ge=0.0, le=100.0)
    sbi: float = Field(ge=0.0, le=100.0)
    friendliness: float = Field(ge=0.0, le=100.0)
    evidence_status: EvidenceStatus
    tree: TreeState
    vision: VisionObservation
    audio: AudioObservation
    transcript: TranscriptObservation
    metrics: dict[str, float] = Field(default_factory=dict)
    suggestion: str = "正在等待真实视觉与语音证据。"
    explanation: str = "证据不足，系统保持中性观察。"
    provenance: Literal["measured"] = "measured"

    @classmethod
    def neutral(cls, session_id: str) -> "LiveState":
        return cls(
            session_id=session_id,
            status="ready",
            listening_state="idle",
            signal_sequence=0,
            signal_activity=0.0,
            sbi=0.0,
            friendliness=100.0,
            evidence_status="insufficient",
            tree=TreeState.observing(),
            vision=VisionObservation.neutral(),
            audio=AudioObservation.neutral(),
            transcript=TranscriptObservation.neutral(),
            metrics={
                "microaggression_risk": 0.0,
                "tone_pressure": 0.0,
                "visual_tension": 0.0,
                "semantic_bias": 0.0,
            },
        )
