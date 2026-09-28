from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .domain import FusionSnapshot, MetricProvenance, PerceptionEvent


RISK_MODALITIES = ("semantic", "acoustic", "vision", "pose")
FEATURE_KEYS = {
    "semantic": ("offense", "microbias"),
    "acoustic": ("arousal",),
    "vision": ("negative", "micro_expression"),
    "pose": ("closed_posture", "gaze_avoidance", "interruption"),
}
MODALITY_LABELS = {
    "semantic": "语义预设或冒犯表达",
    "acoustic": "语速与声学唤醒",
    "vision": "持续负面表情",
    "pose": "封闭姿态或互动失衡",
}


@dataclass(frozen=True)
class FusionConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.38,
            "acoustic": 0.24,
            "vision": 0.20,
            "pose": 0.18,
        }
    )
    synergy: float = 0.42
    energy_decay: float = 0.45
    freshness_decay: float = 0.32
    threshold: float = 0.55
    slope: float = 7.0
    minimum_reliability: float = 0.22
    maximum_dt_seconds: float = 10.0


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.energy = 0.0
        self.last_update_ms: int | None = None
        self.latest_events: dict[str, PerceptionEvent] = {}

    def reset(self) -> None:
        self.energy = 0.0
        self.last_update_ms = None
        self.latest_events.clear()

    def update(
        self,
        events: Sequence[PerceptionEvent],
        now_ms: int,
        *,
        provenance: MetricProvenance = "simulated",
    ) -> FusionSnapshot:
        now_ms = max(0, int(now_ms))
        if self.last_update_ms is None:
            dt_seconds = 1.0
        else:
            dt_seconds = max(0.0, (now_ms - self.last_update_ms) / 1000.0)
            dt_seconds = min(dt_seconds, self.config.maximum_dt_seconds)
        self.last_update_ms = now_ms

        for item in events:
            if item.modality in RISK_MODALITIES:
                self.latest_events[item.modality] = item

        has_new_risk_evidence = any(item.modality in RISK_MODALITIES for item in events)

        contributions = {name: 0.0 for name in RISK_MODALITIES}
        reliabilities = {name: 0.0 for name in RISK_MODALITIES}
        gated_values: dict[str, float] = {}

        for item in self.latest_events.values():
            value = self._extract_value(item)
            age_seconds = max(0.0, (now_ms - item.at_ms) / 1000.0)
            reliability = item.confidence * math.exp(-self.config.freshness_decay * age_seconds)
            gated = value * reliability if reliability >= self.config.minimum_reliability else 0.0
            reliabilities[item.modality] = max(reliabilities[item.modality], reliability)
            gated_values[item.modality] = max(gated_values.get(item.modality, 0.0), gated)
            contributions[item.modality] = max(
                contributions[item.modality],
                self.config.weights[item.modality] * gated,
            )

        active = [value for value in gated_values.values() if value > 0.05]
        synergy = 0.0
        if len(active) >= 2:
            synergy = self.config.synergy * math.prod(active) ** (1.0 / len(active))
        instantaneous = (
            min(1.5, sum(contributions.values()) + synergy)
            if has_new_risk_evidence
            else 0.0
        )

        decay = math.exp(-self.config.energy_decay * dt_seconds)
        self.energy = max(0.0, self.energy * decay + instantaneous * dt_seconds)
        sbi = 100.0 / (1.0 + math.exp(-self.config.slope * (self.energy - self.config.threshold)))
        sbi = max(0.0, min(100.0, sbi))

        evidence_status = self._evidence_status(gated_values, reliabilities)
        risk_level = self._risk_level(sbi, evidence_status)
        dominant = [
            name
            for name, value in sorted(contributions.items(), key=lambda pair: pair[1], reverse=True)
            if value > 0.01
        ][:3]

        return FusionSnapshot(
            at_ms=now_ms,
            sbi=round(sbi, 2),
            energy=round(self.energy, 6),
            instantaneous_risk=round(instantaneous, 6),
            risk_level=risk_level,
            evidence_status=evidence_status,
            contributions={name: round(value, 6) for name, value in contributions.items()},
            reliabilities={name: round(value, 6) for name, value in reliabilities.items()},
            dominant_modalities=dominant,
            explanation=self._explain(dominant, evidence_status, synergy),
            provenance=provenance,
        )

    @staticmethod
    def _extract_value(event: PerceptionEvent) -> float:
        values: list[float] = []
        for key in FEATURE_KEYS[event.modality]:
            raw = event.features.get(key, 0.0)
            if isinstance(raw, bool):
                values.append(1.0 if raw else 0.0)
            elif isinstance(raw, (int, float)):
                values.append(max(0.0, min(1.0, float(raw))))
        return max(values, default=0.0)

    def _evidence_status(
        self,
        gated_values: dict[str, float],
        reliabilities: dict[str, float],
    ) -> str:
        active = [
            name
            for name, value in gated_values.items()
            if value > 0.08 and reliabilities.get(name, 0.0) >= self.config.minimum_reliability
        ]
        if len(active) >= 2:
            return "sufficient"
        if len(active) == 1:
            return "partial"
        return "insufficient"

    @staticmethod
    def _risk_level(sbi: float, evidence_status: str) -> str:
        if evidence_status == "insufficient":
            return "safe" if sbi < 25 else "observe"
        if sbi < 25:
            return "safe"
        if sbi < 45:
            return "observe"
        if sbi < 70:
            return "warning"
        if sbi < 85:
            return "high"
        return "critical"

    @staticmethod
    def _explain(dominant: list[str], evidence_status: str, synergy: float) -> str:
        if evidence_status == "insufficient":
            return "当前信号置信度或模态覆盖不足，系统保持观察，不进行强干预。"
        if not dominant:
            return "未发现持续的细微社交偏差信号。"
        labels = "、".join(MODALITY_LABELS[name] for name in dominant)
        if synergy > 0:
            return f"{labels}在同一时间窗内相互印证，跨模态协同项提高了风险估计。"
        return f"当前主要线索来自{labels}，仍需结合后续上下文确认。"
