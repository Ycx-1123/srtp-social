from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Modality = Literal["semantic", "acoustic", "vision", "pose", "context"]
MetricProvenance = Literal["measured", "target", "simulated"]
EvidenceStatus = Literal["insufficient", "partial", "sufficient"]
RiskLevel = Literal["safe", "observe", "warning", "high", "critical"]
InterventionAction = Literal["observe", "ambient", "nudge", "pause"]
InterventionTone = Literal["neutral", "gentle", "firm", "recovery"]


class ScenarioPhase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    description: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "ScenarioPhase":
        if self.end_ms <= self.start_ms:
            raise ValueError("phase end_ms must be greater than start_ms")
        return self


class PerceptionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    modality: Modality
    features: dict[str, float | str | bool]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "scenario"
    evidence: str = ""


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    description: str = ""
    duration_ms: int = Field(gt=0)
    seed: int = 0
    participants: list[str] = Field(default_factory=list)
    phases: list[ScenarioPhase] = Field(default_factory=list)
    events: list[PerceptionEvent]

    @model_validator(mode="after")
    def validate_bounds(self) -> "ScenarioDefinition":
        if self.events and self.events[-1].at_ms > self.duration_ms:
            raise ValueError("last event exceeds scenario duration")
        if any(phase.end_ms > self.duration_ms for phase in self.phases):
            raise ValueError("phase exceeds scenario duration")
        return self


class FusionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    sbi: float = Field(ge=0.0, le=100.0)
    energy: float = Field(ge=0.0)
    instantaneous_risk: float = Field(ge=0.0)
    risk_level: RiskLevel
    evidence_status: EvidenceStatus
    contributions: dict[str, float]
    reliabilities: dict[str, float]
    dominant_modalities: list[str]
    explanation: str
    provenance: MetricProvenance = "simulated"


class InterventionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    action: InterventionAction
    message: str
    reason: str
    intensity: float = Field(ge=0.0, le=1.0)
    tone: InterventionTone = "neutral"
    provenance: MetricProvenance = "simulated"


class SessionFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    phase: str = ""
    speaker: str = ""
    dialogue: str = ""
    snapshot: FusionSnapshot
    intervention: InterventionEvent
    modality_features: dict[str, dict[str, float | str | bool]] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    scenario_id: str
    peak_sbi: float
    mean_sbi: float
    risk_distribution: dict[str, int]
    intervention_counts: dict[str, int]
    dominant_modalities: list[str]
    key_events: list[dict[str, float | int | str]]
    recovery_intervals_ms: list[int]
    anonymized_transcript: str
    recommendations: list[str]
    provenance: MetricProvenance = "simulated"


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    scenario_id: str
    scenario_title: str
    session_id: str
    status: Literal["ready", "running", "paused", "completed"]
    elapsed_ms: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    speed: float = Field(gt=0)
    phase: str = ""
    speaker: str = ""
    dialogue: str = ""
    fusion: FusionSnapshot
    intervention: InterventionEvent
    modality_features: dict[str, dict[str, float | str | bool]]
    persistence: dict[str, str]
