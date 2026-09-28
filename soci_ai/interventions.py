from __future__ import annotations

from .domain import FusionSnapshot, InterventionEvent, MetricProvenance


class InterventionPolicy:
    def __init__(self, cooldown_ms: int = 8000):
        self.cooldown_ms = cooldown_ms
        self.last_intervention_ms: int | None = None
        self.was_high = False

    def reset(self) -> None:
        self.last_intervention_ms = None
        self.was_high = False

    def decide(
        self,
        snapshot: FusionSnapshot,
        now_ms: int,
        context: str,
        *,
        provenance: MetricProvenance = "simulated",
    ) -> InterventionEvent:
        now_ms = max(0, int(now_ms))
        if snapshot.evidence_status != "sufficient":
            return self._observe(now_ms, "insufficient_evidence", provenance)

        risk_is_high = snapshot.sbi >= 70
        if (
            snapshot.sbi >= 45
            and self.last_intervention_ms is not None
            and now_ms - self.last_intervention_ms < self.cooldown_ms
        ):
            self.was_high = self.was_high or risk_is_high
            return self._observe(now_ms, "cooldown_active", provenance)

        if snapshot.sbi >= 85:
            self.was_high = True
            return self._emit(
                now_ms,
                "pause",
                "建议短暂停顿，把讨论重新聚焦到约束、证据和下一步行动。",
                "sustained_critical_risk",
                0.90,
                "firm",
                provenance,
            )

        if snapshot.sbi >= 45:
            self.was_high = self.was_high or risk_is_high
            return self._emit(
                now_ms,
                "nudge",
                self._template(snapshot.dominant_modalities, context),
                "cross_modal_risk",
                0.62 if snapshot.sbi < 70 else 0.76,
                "gentle",
                provenance,
            )

        if self.was_high:
            self.was_high = False
            return self._emit(
                now_ms,
                "ambient",
                "交流状态正在恢复，继续保持轮流发言和事实导向的表达。",
                "recovery_detected",
                0.25,
                "recovery",
                provenance,
                start_cooldown=False,
            )

        if snapshot.sbi >= 25:
            return InterventionEvent(
                at_ms=now_ms,
                action="ambient",
                message="系统正在观察轻微波动，暂不打断当前交流。",
                reason="low_intensity_signal",
                intensity=0.18,
                tone="neutral",
                provenance=provenance,
            )
        return self._observe(now_ms, "stable", provenance)

    def _emit(
        self,
        now_ms: int,
        action: str,
        message: str,
        reason: str,
        intensity: float,
        tone: str,
        provenance: MetricProvenance,
        *,
        start_cooldown: bool = True,
    ) -> InterventionEvent:
        if start_cooldown:
            self.last_intervention_ms = now_ms
        return InterventionEvent(
            at_ms=now_ms,
            action=action,
            message=message,
            reason=reason,
            intensity=intensity,
            tone=tone,
            provenance=provenance,
        )

    @staticmethod
    def _observe(
        now_ms: int,
        reason: str,
        provenance: MetricProvenance,
    ) -> InterventionEvent:
        messages = {
            "insufficient_evidence": "证据不足，系统保持观察。",
            "cooldown_active": "已给出提示，冷却期内避免重复打扰。",
            "stable": "交流状态稳定，系统静默运行。",
        }
        return InterventionEvent(
            at_ms=now_ms,
            action="observe",
            message=messages[reason],
            reason=reason,
            intensity=0.0,
            tone="neutral",
            provenance=provenance,
        )

    @staticmethod
    def _template(dominant_modalities: list[str], context: str) -> str:
        primary = dominant_modalities[0] if dominant_modalities else "semantic"
        templates = {
            "semantic": "把对人的概括换成可核对的事实，再说明你希望解决的具体问题。",
            "acoustic": "语速和音量正在升高，可以留出一个停顿后再陈述关键依据。",
            "vision": "面部紧绷信号正在持续，可以放松眉间和下颌，再继续表达观点。",
            "pose": "抢话和封闭姿态正在增加，建议给对方一个完整的发言回合。",
        }
        message = templates.get(primary, templates["semantic"])
        if context and primary == "semantic":
            return message
        return message
