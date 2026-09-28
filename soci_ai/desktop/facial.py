"""Calibrated facial-action cues for the native desktop capture path.

The input is MediaPipe's named blendshape coefficients, not emotion scores.
Category names follow its official face_blendshapes_graph.cc label table:
https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc

Eight neutral observations establish an individual median and MAD noise gate
for each coefficient. Keep a relaxed face during calibration; the baseline
then stays fixed until reset so a sustained expression cannot be learned away.
Landmarks validate the observed frame only. No image-axis distances are used
as expression evidence, so face size and in-plane rotation cannot manufacture
additional geometric cues. The legacy web tracker is deliberately separate.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median, pstdev
from typing import Any, Iterable, Mapping


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _blendshape_map(categories: Iterable[Any] | Mapping[str, float]) -> dict[str, float]:
    pairs = categories.items() if isinstance(categories, Mapping) else (
        (getattr(item, "category_name", None), getattr(item, "score", None))
        for item in categories
    )
    scores = {}
    for name, value in pairs:
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if name is not None and math.isfinite(score):
            scores[str(name)] = _clamp(score)
    return scores


def _valid_landmarks(landmarks: Iterable[Any] | None) -> bool:
    if landmarks is None:
        return True  # Blendshape-only callers remain supported.
    count = 0
    for point in landmarks:
        try:
            coordinates = (float(point.x), float(point.y), float(getattr(point, "z", 0.0)))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in coordinates):
            return False
        count += 1
    return count > 0


class FacialCueTracker:
    """Measure visible facial-action changes, never intent or emotion.

    For each coefficient: gate = max(0.006, 3 * 1.4826 * MAD), and
    strength = clip((max(0, score - median - gate) / span) ** 0.75),
    where span = max(channel_span, 6 * gate). Channel spans are 0.18 for
    brows/downturn, 0.16 for lip pressure, and 0.30 for smile. These are
    display sensitivity settings, not clinically validated thresholds.

    Each channel takes the strongest individually calibrated component.
    Only brow/downturn enter the sustained cue; smile is independent.
    Lip pressure/pucker remains a legacy diagnostic value, not risk evidence:
    these coefficients also change during ordinary articulation.
    """

    _GROUPS = {
        "brow_tension": ("browDownLeft", "browDownRight"),
        "lip_tension": ("mouthPressLeft", "mouthPressRight", "mouthPucker"),
        "mouth_downturn": ("mouthFrownLeft", "mouthFrownRight"),
        "smile": ("mouthSmileLeft", "mouthSmileRight"),
    }
    _SPANS = {"brow_tension": 0.18, "lip_tension": 0.16, "mouth_downturn": 0.18, "smile": 0.30}
    _NEGATIVE_GROUPS = ("brow_tension", "mouth_downturn")
    _ACTIVATION = 0.18

    def __init__(self, *, calibration_frames: int = 8, sustain_ms: int = 250):
        self.calibration_frames = max(6, int(calibration_frames))
        self.sustain_ms = max(250, int(sustain_ms))
        self.reset()

    def reset(self) -> None:
        self._samples: dict[str, list[float]] = {
            name: [] for names in self._GROUPS.values() for name in names
        }
        self._sample_count = 0
        self._baseline: dict[str, float] = {}
        self._noise: dict[str, float] = {}
        self._strength_history: deque[float] = deque(maxlen=6)
        self._onset_ms: int | None = None
        self._last_at_ms: int | None = None

    def _break_continuity(self) -> None:
        self._onset_ms = None
        self._strength_history.clear()
        if not self._baseline:
            self._sample_count = 0
            for values in self._samples.values():
                values.clear()

    def update(
        self,
        categories: Iterable[Any] | Mapping[str, float],
        at_ms: int,
        *,
        face_detected: bool = True,
        landmarks: Iterable[Any] | None = None,
    ) -> dict[str, float | bool | str]:
        at_ms = max(0, int(at_ms))
        if self._last_at_ms is not None and not 0 < at_ms - self._last_at_ms <= 750:
            self._break_continuity()
        self._last_at_ms = at_ms

        scores = _blendshape_map(categories)
        scores = {name: value for name, value in scores.items() if name in self._samples}
        if not face_detected or not scores or not _valid_landmarks(landmarks):
            self._break_continuity()
            return self._empty()

        output = self._empty()
        output.update(facial_cue_available=True, cue_source="mediapipe_blendshapes")
        if not self._baseline:
            self._sample_count += 1
            for name, value in scores.items():
                self._samples[name].append(value)
            if self._sample_count >= self.calibration_frames:
                for name, values in self._samples.items():
                    # A missing coefficient must not be calibrated as zero.
                    if len(values) != self._sample_count:
                        continue
                    center = median(values)
                    mad = median(abs(value - center) for value in values)
                    self._baseline[name] = center
                    self._noise[name] = max(0.006, 3.0 * 1.4826 * mad)
                if not self._baseline:
                    self._break_continuity()
            output["baseline_ready"] = bool(self._baseline)
            output["calibration_progress"] = _clamp(self._sample_count / self.calibration_frames)
            return output

        strength = {}
        for group, names in self._GROUPS.items():
            components = []
            for name in names:
                if name not in scores or name not in self._baseline:
                    continue
                gate = self._noise[name]
                excess = max(0.0, scores[name] - self._baseline[name] - gate)
                span = max(self._SPANS[group], 6.0 * gate)
                components.append(_clamp((excess / span) ** 0.75))
            strength[group] = max(components, default=0.0)

        combined = max(strength[group] for group in self._NEGATIVE_GROUPS)
        self._strength_history.append(combined)
        if combined >= self._ACTIVATION:
            if self._onset_ms is None:
                self._onset_ms = at_ms
        else:
            self._onset_ms = None
        duration = at_ms - self._onset_ms if self._onset_ms is not None else 0
        variability = pstdev(self._strength_history) if len(self._strength_history) > 1 else 0.0
        stability = _clamp(1.0 - variability / 0.45)
        confidence = 0.55 + 0.45 * stability if combined >= self._ACTIVATION else 0.0
        output.update(strength)
        output.update(
            baseline_ready=True,
            calibration_progress=1.0,
            cue_duration_ms=float(duration),
            facial_cue_confidence=confidence,
            micro_expression=combined * confidence if duration >= self.sustain_ms else 0.0,
        )
        return output

    @staticmethod
    def _empty() -> dict[str, float | bool | str]:
        return {
            "brow_tension": 0.0, "lip_tension": 0.0, "mouth_downturn": 0.0,
            "smile": 0.0, "micro_expression": 0.0, "cue_duration_ms": 0.0,
            "facial_cue_confidence": 0.0, "baseline_ready": False,
            "facial_cue_available": False, "calibration_progress": 0.0,
            "cue_source": "unavailable",
        }
