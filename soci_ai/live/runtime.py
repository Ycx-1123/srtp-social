from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from ..domain import PerceptionEvent
from ..fusion import FusionConfig, FusionEngine
from ..interventions import InterventionPolicy
from .audio import AudioFeaturePayload, SenseVoiceTranscriber, validate_audio_features
from .domain import (
    AudioObservation,
    FaceBox,
    LiveState,
    TranscriptObservation,
    TreeState,
    VisionObservation,
)
from .semantics import MicroaggressionAnalyzer


class TreeStateMapper:
    def __init__(self):
        self.had_risk = False

    def reset(self) -> None:
        self.had_risk = False

    def map(
        self,
        sbi: float,
        evidence_status: str,
        previous: TreeState,
        *,
        measured_context: bool = False,
    ) -> TreeState:
        risk = max(0.0, min(1.0, sbi / 100.0))
        if sbi >= 25:
            self.had_risk = True
        if self.had_risk and sbi < 25 and previous.mode in {"signal", "risk"}:
            return TreeState(
                mode="recovering",
                health=max(previous.health, 0.66),
                risk=risk,
                bloom=0.42,
                wind=0.18,
            )
        if evidence_status == "insufficient" and not measured_context:
            return TreeState.observing()
        if sbi >= 70:
            return TreeState(mode="risk", health=0.25, risk=risk, bloom=0.02, wind=0.92)
        if sbi >= 25:
            return TreeState(mode="signal", health=0.58, risk=risk, bloom=0.20, wind=0.55)
        return TreeState(
            mode="friendly",
            health=max(0.72, 0.98 - risk * 0.35),
            risk=risk,
            bloom=max(0.55, 0.94 - risk * 0.82),
            wind=min(0.30, 0.12 + risk * 0.45),
        )


class LiveSessionRuntime:
    def __init__(
        self,
        *,
        vision: Any,
        transcriber: SenseVoiceTranscriber | Any,
        analyzer: MicroaggressionAnalyzer,
        fusion: FusionEngine | None = None,
        policy: InterventionPolicy | None = None,
        session_id: str | None = None,
    ):
        self.vision = vision
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.fusion = fusion or FusionEngine(
            FusionConfig(threshold=0.35, slope=7.0, energy_decay=0.40)
        )
        self.policy = policy or InterventionPolicy()
        self.session_id = session_id or uuid.uuid4().hex
        self._state = LiveState.neutral(self.session_id)
        self._tree_mapper = TreeStateMapper()
        self._vision_lock = asyncio.Lock()
        self._audio_lock = asyncio.Lock()
        self._pending_frame: tuple[bytes, int] | None = None
        self._started_monotonic: float | None = None

    @property
    def vision_calls(self) -> list[bytes]:
        return list(getattr(self.vision, "calls", []))

    async def start(self) -> LiveState:
        if self._state.status == "running":
            return self._state
        if self._state.status == "stopped":
            return self._state
        self._started_monotonic = time.monotonic()
        warmup = getattr(self.vision, "warmup", None)
        if callable(warmup):
            await asyncio.to_thread(warmup)
        self._state = self._state.model_copy(
            update={"status": "running", "listening_state": "listening"}
        )
        return self._state

    async def stop(self) -> LiveState:
        if self._state.status != "stopped":
            self._state = self._state.model_copy(
                update={"status": "stopped", "listening_state": "idle"}
            )
            close = getattr(self.vision, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        return self._state

    def state(self) -> LiveState:
        return self._state

    def snapshot(self) -> dict[str, object]:
        return self._state.model_dump(mode="json")

    async def accept_vision(
        self,
        *,
        negative: float,
        confidence: float,
        at_ms: int,
    ) -> LiveState:
        observation = VisionObservation(
            at_ms=at_ms,
            status="detected",
            face_detected=True,
            box=FaceBox(x=0.2, y=0.15, width=0.6, height=0.7),
            expression="observed",
            confidence=confidence,
            features={"negative": max(0.0, min(1.0, negative))},
        )
        return self._accept_vision_observation(observation)

    async def accept_audio(
        self,
        *,
        arousal: float,
        confidence: float,
        at_ms: int,
    ) -> LiveState:
        observation = AudioObservation(
            at_ms=at_ms,
            status="speech",
            arousal=max(0.0, min(1.0, arousal)),
            confidence=max(0.0, min(1.0, confidence)),
            features={"arousal": max(0.0, min(1.0, arousal))},
        )
        event = PerceptionEvent(
            at_ms=at_ms,
            modality="acoustic",
            features=observation.features,
            confidence=observation.confidence,
            source="live_audio",
        )
        return self._apply([event], at_ms, audio=observation)

    async def accept_transcript(self, text: str, *, at_ms: int) -> LiveState:
        transcript = TranscriptObservation(
            at_ms=at_ms,
            status="completed",
            text=text,
            confidence=0.92 if text else 0.0,
        )
        event = self.analyzer.analyze(text, at_ms)
        return self._apply([event], at_ms, transcript=transcript, listening_state="listening")

    async def submit_frame(self, image: bytes, captured_at_ms: int) -> LiveState:
        self._pending_frame = (image, max(0, int(captured_at_ms)))
        if self._vision_lock.locked():
            return self._state
        async with self._vision_lock:
            while self._pending_frame is not None:
                newest, newest_at = self._pending_frame
                self._pending_frame = None
                observation = await asyncio.to_thread(self.vision.analyze, newest, newest_at)
                self._accept_vision_observation(observation)
            return self._state

    async def submit_audio_features(
        self,
        payload: AudioFeaturePayload | dict[str, float | int],
    ) -> LiveState:
        parsed = payload if isinstance(payload, AudioFeaturePayload) else AudioFeaturePayload(**payload)
        observation = validate_audio_features(parsed)
        if observation.status == "silent":
            return self._apply([], observation.at_ms, audio=observation)
        event = PerceptionEvent(
            at_ms=observation.at_ms,
            modality="acoustic",
            features=observation.features,
            confidence=observation.confidence,
            source="browser_audio_features",
        )
        return self._apply([event], observation.at_ms, audio=observation)

    async def submit_audio_chunk(
        self,
        pcm: bytes,
        sample_rate: int,
        at_ms: int,
    ) -> LiveState:
        if self._audio_lock.locked():
            return self._state
        async with self._audio_lock:
            self._state = self._state.model_copy(update={"listening_state": "transcribing"})
            transcript = await asyncio.to_thread(
                self.transcriber.transcribe_pcm16,
                pcm,
                sample_rate,
                at_ms,
            )
            if transcript.status == "completed":
                self._state = self._state.model_copy(update={"listening_state": "analyzing"})
                event = self.analyzer.analyze(transcript.text, at_ms)
                return self._apply(
                    [event],
                    at_ms,
                    transcript=transcript,
                    listening_state="listening",
                )
            listening = "unavailable" if transcript.status in {"unavailable", "error"} else "listening"
            return self._apply([], at_ms, transcript=transcript, listening_state=listening)

    async def tick(self, at_ms: int) -> LiveState:
        return self._apply([], at_ms)

    def _accept_vision_observation(self, observation: VisionObservation) -> LiveState:
        events: list[PerceptionEvent] = []
        if observation.face_detected:
            events.append(
                PerceptionEvent(
                    at_ms=observation.at_ms,
                    modality="vision",
                    features=observation.features,
                    confidence=observation.confidence,
                    source="live_camera",
                    evidence=observation.expression,
                )
            )
        return self._apply(
            events,
            observation.at_ms,
            vision=observation,
            last_frame_at_ms=observation.at_ms,
        )

    def _apply(
        self,
        events: list[PerceptionEvent],
        at_ms: int,
        *,
        vision: VisionObservation | None = None,
        audio: AudioObservation | None = None,
        transcript: TranscriptObservation | None = None,
        listening_state: str | None = None,
        last_frame_at_ms: int | None = None,
    ) -> LiveState:
        elapsed_ms = max(self._state.elapsed_ms, max(0, int(at_ms)))
        snapshot = self.fusion.update(events, elapsed_ms, provenance="measured")
        intervention = self.policy.decide(
            snapshot,
            elapsed_ms,
            transcript.text if transcript else self._state.transcript.text,
            provenance="measured",
        )
        current_vision = vision or self._state.vision
        current_audio = audio or self._state.audio
        current_transcript = transcript or self._state.transcript
        signal_activity = self._signal_activity(
            current_vision,
            current_audio,
            current_transcript,
        )
        measured_context = (
            current_vision.face_detected
            or current_audio.status == "speech"
            or current_transcript.status == "completed"
        )
        tree = self._tree_mapper.map(
            snapshot.sbi,
            snapshot.evidence_status,
            self._state.tree,
            measured_context=measured_context,
        )
        weights = self.fusion.config.weights

        def normalized(modality: str) -> float:
            contribution = snapshot.contributions.get(modality, 0.0)
            return min(100.0, 100.0 * contribution / max(weights.get(modality, 1.0), 1e-6))

        update: dict[str, object] = {
            "elapsed_ms": elapsed_ms,
            "signal_sequence": self._state.signal_sequence + 1,
            "signal_activity": signal_activity,
            "sbi": snapshot.sbi,
            "friendliness": round(100.0 - snapshot.sbi, 2),
            "evidence_status": snapshot.evidence_status,
            "tree": tree,
            "metrics": {
                "microaggression_risk": snapshot.sbi,
                "tone_pressure": round(normalized("acoustic"), 2),
                "visual_tension": round(normalized("vision"), 2),
                "semantic_bias": round(normalized("semantic"), 2),
            },
            "suggestion": intervention.message,
            "explanation": snapshot.explanation,
        }
        if vision is not None:
            update["vision"] = vision
        if audio is not None:
            update["audio"] = audio
        if transcript is not None:
            update["transcript"] = transcript
        if listening_state is not None:
            update["listening_state"] = listening_state
        if last_frame_at_ms is not None:
            update["last_frame_at_ms"] = last_frame_at_ms
        self._state = self._state.model_copy(update=update)
        return self._state

    @staticmethod
    def _signal_activity(
        vision: VisionObservation,
        audio: AudioObservation,
        transcript: TranscriptObservation,
    ) -> float:
        """Measured sensing intensity; deliberately separate from social-risk scoring."""
        face = 0.0
        if vision.face_detected:
            face = 18.0 + vision.confidence * 30.0
        voice = audio.rms * 65.0 + audio.peak * 15.0 + audio.speech_ratio * 25.0
        language = 12.0 if transcript.status == "completed" and transcript.text else 0.0
        return round(max(0.0, min(100.0, face + voice + language)), 2)
