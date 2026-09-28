from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .domain import AudioObservation, TranscriptObservation


class AudioFeaturePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_ms: int = Field(ge=0)
    rms: float = Field(ge=0.0, le=1.0)
    peak: float = Field(ge=0.0, le=1.0)
    speech_ratio: float = Field(ge=0.0, le=1.0)
    pace: float = Field(default=0.0, ge=0.0, le=12.0)


class AudioModelUnavailable(RuntimeError):
    """Raised internally when the optional local ASR engine cannot load."""


def validate_audio_features(payload: AudioFeaturePayload) -> AudioObservation:
    pace_pressure = max(0.0, min(1.0, (payload.pace - 3.0) / 5.0))
    arousal = min(
        1.0,
        payload.rms * 0.40
        + payload.peak * 0.35
        + payload.speech_ratio * 0.15
        + pace_pressure * 0.10,
    )
    speaking = payload.speech_ratio >= 0.08 or payload.rms >= 0.03
    confidence = min(1.0, 0.42 + payload.speech_ratio * 0.45) if speaking else 0.25
    return AudioObservation(
        at_ms=payload.at_ms,
        status="speech" if speaking else "silent",
        rms=payload.rms,
        peak=payload.peak,
        speech_ratio=payload.speech_ratio,
        pace=payload.pace,
        arousal=round(arousal, 6),
        confidence=round(confidence, 6),
        features={
            "arousal": round(arousal, 6),
            "rms": payload.rms,
            "peak": payload.peak,
            "speech_ratio": payload.speech_ratio,
            "pace": payload.pace,
        },
    )


class SenseVoiceTranscriber:
    MIN_SAMPLES = 3200
    SILENCE_PEAK = 500

    def __init__(
        self,
        model_dir: Path,
        *,
        model_factory: Callable[..., Any] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self._model_factory = model_factory
        self._loaded_model: Any | None = None
        self._last_processed_at_ms: int | None = None

    def transcribe_pcm16(
        self,
        pcm: bytes,
        sample_rate: int,
        at_ms: int,
    ) -> TranscriptObservation:
        at_ms = max(0, int(at_ms))
        if self._last_processed_at_ms is not None and at_ms < self._last_processed_at_ms:
            return TranscriptObservation(at_ms=at_ms, status="stale")
        if sample_rate != 16000 or len(pcm) % 2 or len(pcm) // 2 < self.MIN_SAMPLES:
            return TranscriptObservation(at_ms=at_ms, status="too_short")
        samples = np.frombuffer(pcm, dtype="<i2")
        peak = int(np.max(np.abs(samples.astype(np.int32, copy=False)))) if samples.size else 0
        if peak < self.SILENCE_PEAK:
            return TranscriptObservation(at_ms=at_ms, status="silent", peak_volume=peak)

        self._last_processed_at_ms = at_ms
        started = time.perf_counter()
        try:
            model = self._model()
            audio = np.ascontiguousarray(samples.astype(np.float32) / 32768.0)
            result = model.generate(input=audio, is_final=True)
        except AudioModelUnavailable:
            return TranscriptObservation(at_ms=at_ms, status="unavailable", peak_volume=peak)
        except Exception:
            return TranscriptObservation(at_ms=at_ms, status="error", peak_volume=peak)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = ""
        if result and isinstance(result[0], dict):
            raw_text = str(result[0].get("text", "")).strip()
        tags = re.findall(r"<\|([^|]+)\|>", raw_text)
        text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()
        return TranscriptObservation(
            at_ms=at_ms,
            status="completed",
            text=text,
            tags=tags,
            confidence=0.92 if text else 0.35,
            peak_volume=peak,
            duration_seconds=round(samples.size / sample_rate, 4),
            latency_ms=round(latency_ms, 2),
        )

    def _model(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model
        if self._model_factory is None:
            if not self.model_dir.is_dir():
                raise AudioModelUnavailable(f"SenseVoice model not found: {self.model_dir}")
            try:
                from funasr import AutoModel  # type: ignore
            except ImportError as exc:
                raise AudioModelUnavailable("FunASR is not installed") from exc
            self._model_factory = AutoModel
        self._loaded_model = self._model_factory(
            model=str(self.model_dir),
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
        )
        return self._loaded_model
