from __future__ import annotations

import re
from collections import Counter

from .domain import SessionFrame, SessionSummary
from .store import SessionStore


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")


class ReportBuilder:
    def __init__(self, store: SessionStore):
        self.store = store

    def build(self, session_id: str) -> SessionSummary:
        record = self.store.get_session(session_id)
        frames = [SessionFrame.model_validate(item) for item in record["states"]]
        transcript_rows = record["transcript"]
        session = record["session"]

        values = [frame.snapshot.sbi for frame in frames]
        risk_counts = Counter(frame.snapshot.risk_level for frame in frames)
        actions = Counter(frame.intervention.action for frame in frames)
        modalities = Counter(
            modality
            for frame in frames
            for modality in frame.snapshot.dominant_modalities
        )
        key_events = [
            {
                "at_ms": frame.at_ms,
                "sbi": frame.snapshot.sbi,
                "risk_level": frame.snapshot.risk_level,
                "phase": frame.phase,
                "dialogue": frame.dialogue,
            }
            for frame in sorted(frames, key=lambda item: item.snapshot.sbi, reverse=True)[:5]
        ]

        transcript = "\n".join(
            f"[{self._format_time(row['at_ms'])}] {row['speaker']}: {row['text']}"
            for row in transcript_rows
        )
        anonymized = EMAIL_RE.sub("[EMAIL]", PHONE_RE.sub("[PHONE]", transcript))

        return SessionSummary(
            session_id=session_id,
            scenario_id=str(session["scenario_id"]),
            peak_sbi=round(max(values, default=0.0), 2),
            mean_sbi=round(sum(values) / len(values), 2) if values else 0.0,
            risk_distribution=dict(risk_counts),
            intervention_counts=dict(actions),
            dominant_modalities=[name for name, _ in modalities.most_common(4)],
            key_events=key_events,
            recovery_intervals_ms=self._recovery_intervals(frames),
            anonymized_transcript=anonymized,
            recommendations=self._recommendations(modalities, values),
            provenance="simulated",
        )

    @staticmethod
    def _format_time(at_ms: int) -> str:
        seconds, millis = divmod(int(at_ms), 1000)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    @staticmethod
    def _recovery_intervals(frames: list[SessionFrame]) -> list[int]:
        high_started: int | None = None
        intervals: list[int] = []
        for frame in sorted(frames, key=lambda item: item.at_ms):
            if frame.snapshot.sbi >= 70 and high_started is None:
                high_started = frame.at_ms
            elif frame.snapshot.sbi < 35 and high_started is not None:
                intervals.append(frame.at_ms - high_started)
                high_started = None
        return intervals

    @staticmethod
    def _recommendations(modalities: Counter[str], values: list[float]) -> list[str]:
        recommendations: list[str] = []
        if modalities["semantic"]:
            recommendations.append("减少对群体或角色的概括，优先陈述可核对的事实与具体约束。")
        if modalities["acoustic"]:
            recommendations.append("在高压讨论中主动保留停顿，将语速维持在个人稳定区间。")
        if modalities["pose"]:
            recommendations.append("减少抢话和封闭姿态，为对方保留完整发言回合。")
        if not recommendations and values:
            recommendations.append("本次交流整体平稳，继续保持轮流发言和事实导向表达。")
        if not values:
            recommendations.append("本次会话没有足够的结构化事件，暂不生成行为结论。")
        return recommendations
