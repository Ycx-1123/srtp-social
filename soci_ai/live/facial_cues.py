from __future__ import annotations

from collections import deque
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Iterable, Mapping


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _blendshape_map(categories: Iterable[Any] | Mapping[str, float]) -> dict[str, float]:
    if isinstance(categories, Mapping):
        return {str(name): _clamp(value) for name, value in categories.items()}
    result: dict[str, float] = {}
    for item in categories:
        name = getattr(item, "category_name", None)
        score = getattr(item, "score", None)
        if name is not None and score is not None:
            result[str(name)] = _clamp(score)
    return result


class FacialCueTracker:
    """Tracks observable facial-action changes; it does not infer intent or emotion."""

    _GROUPS = {
        "brow_tension": ("browDownLeft", "browDownRight"),
        "lip_tension": ("mouthPressLeft", "mouthPressRight", "mouthPucker"),
        "mouth_downturn": ("mouthFrownLeft", "mouthFrownRight"),
    }

    def __init__(self, *, calibration_frames: int = 4, sustain_ms: int = 800):
        self.calibration_frames = max(3, int(calibration_frames))
        self.sustain_ms = max(250, int(sustain_ms))
        self._baseline_samples: dict[str, list[float]] = {key: [] for key in self._GROUPS}
        self._baseline: dict[str, float] = {}
        self._strength_history: deque[float] = deque(maxlen=6)
        self._onset_ms: int | None = None
        self._last_at_ms: int | None = None

    def reset(self) -> None:
        self._baseline_samples = {key: [] for key in self._GROUPS}
        self._baseline.clear()
        self._strength_history.clear()
        self._onset_ms = None
        self._last_at_ms = None

    def update(
        self,
        blendshapes: Iterable[Any] | Mapping[str, float],
        at_ms: int,
        *,
        face_detected: bool = True,
    ) -> dict[str, float | bool]:
        at_ms = max(0, int(at_ms))
        if not face_detected:
            self._onset_ms = None
            self._strength_history.clear()
            self._last_at_ms = at_ms
            return self._empty()

        raw_scores = _blendshape_map(blendshapes)
        raw = {
            group: max((raw_scores.get(name, 0.0) for name in names), default=0.0)
            for group, names in self._GROUPS.items()
        }
        if not self._baseline:
            for key, value in raw.items():
                self._baseline_samples[key].append(value)
            if min(map(len, self._baseline_samples.values())) >= self.calibration_frames:
                self._baseline = {
                    key: median(values[: self.calibration_frames])
                    for key, values in self._baseline_samples.items()
                }
            self._last_at_ms = at_ms
            output = self._empty()
            output["baseline_ready"] = bool(self._baseline)
            return output

        strength = {
            key: _clamp((value - self._baseline[key] - 0.12) / 0.42)
            for key, value in raw.items()
        }
        brow = strength["brow_tension"]
        lip = strength["lip_tension"]
        downturn = strength["mouth_downturn"]
        combined = max(brow, lip, downturn)

        # A large timestamp gap means the stream was interrupted, not continuous evidence.
        if self._last_at_ms is not None and at_ms - self._last_at_ms > 750:
            self._onset_ms = None
            self._strength_history.clear()
        self._last_at_ms = at_ms
        self._strength_history.append(combined)

        if combined >= 0.22:
            if self._onset_ms is None:
                self._onset_ms = at_ms
        else:
            self._onset_ms = None

        duration = max(0, at_ms - self._onset_ms) if self._onset_ms is not None else 0
        variability = pstdev(self._strength_history) if len(self._strength_history) > 1 else 0.0
        stability = _clamp(1.0 - variability / 0.45)
        cue_confidence = (0.55 + stability * 0.45) if combined >= 0.22 else 0.0
        sustained = duration >= self.sustain_ms and cue_confidence >= 0.55
        micro_expression = combined * cue_confidence if sustained else 0.0
        return {
            "brow_tension": brow,
            "lip_tension": lip,
            "mouth_downturn": downturn,
            "micro_expression": micro_expression,
            "cue_duration_ms": float(duration),
            "facial_cue_confidence": cue_confidence,
            "baseline_ready": True,
        }

    @staticmethod
    def _empty() -> dict[str, float | bool]:
        return {
            "brow_tension": 0.0,
            "lip_tension": 0.0,
            "mouth_downturn": 0.0,
            "micro_expression": 0.0,
            "cue_duration_ms": 0.0,
            "facial_cue_confidence": 0.0,
            "baseline_ready": False,
        }


class MediaPipeFacialCueDetector:
    """Lightweight local Face Landmarker blendshape adapter."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed") from exc

        self._mp = mp
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._tracker = FacialCueTracker()

    def analyze_bgr(self, frame: Any, at_ms: int, cv2_module: Any) -> dict[str, float | bool]:
        rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        blendshape_frames = getattr(result, "face_blendshapes", None) or []
        if not blendshape_frames:
            return self._tracker.update({}, at_ms, face_detected=False)
        return self._tracker.update(blendshape_frames[0], at_ms)

    def mark_no_face(self, at_ms: int) -> None:
        self._tracker.update({}, at_ms, face_detected=False)

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()
