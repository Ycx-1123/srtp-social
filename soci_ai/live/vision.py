from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import Any, Callable, MutableMapping

import numpy as np

from .domain import FaceBox, VisionObservation
from .facial_cues import MediaPipeFacialCueDetector


class FrameValidationError(ValueError):
    """Raised when an uploaded camera frame cannot be safely decoded."""


class VisionModelUnavailable(RuntimeError):
    """Raised when local vision dependencies or weights are unavailable."""


NEGATIVE_SEVERITY = {
    "angry": 1.0,
    "anger": 1.0,
    "愤怒": 1.0,
    "disgust": 0.95,
    "厌恶": 0.95,
    "fear": 0.55,
    "恐惧": 0.55,
    "sad": 0.45,
    "sadness": 0.45,
    "悲伤": 0.45,
    "surprise": 0.10,
    "惊讶": 0.10,
    "neutral": 0.05,
    "中性": 0.05,
    "happy": 0.0,
    "happiness": 0.0,
    "高兴": 0.0,
}


def configure_ultralytics_runtime(
    directory: Path,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Keep Ultralytics' mutable settings inside the local project runtime."""
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    (os.environ if environ is None else environ)["YOLO_CONFIG_DIR"] = str(target)
    return target


class YoloVisionAdapter:
    MAX_FRAME_BYTES = 2_000_000

    def __init__(
        self,
        weights_path: Path,
        *,
        cv2_module: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
        ultralytics_config_dir: Path | None = None,
        classification_interval_ms: int = 500,
        background_classification: bool = False,
        facial_model_path: Path | None = None,
        facial_cue_detector: Any | None = None,
    ):
        self.weights_path = Path(weights_path)
        self._cv2_module = cv2_module
        self._model_factory = model_factory
        self._ultralytics_config_dir = ultralytics_config_dir
        self._classification_interval_ms = max(0, int(classification_interval_ms))
        self._background_classification = background_classification
        self._facial_model_path = Path(facial_model_path) if facial_model_path is not None else None
        self._facial_cue_detector = facial_cue_detector
        self._facial_cue_disabled = False
        self._last_prediction: tuple[str, float, float] | None = None
        self._last_prediction_at_ms: int | None = None
        self._prediction_inflight = False
        self._prediction_lock = threading.Lock()
        self._model: Any | None = None
        self._face_detector: Any | None = None

    def analyze(self, image_bytes: bytes, captured_at_ms: int) -> VisionObservation:
        frame = self._decode_and_resize(image_bytes, longest_edge=480)
        face = self._largest_face(frame)
        if face is None:
            self._facial_features(frame, max(0, int(captured_at_ms)), face_detected=False)
            return VisionObservation.no_face(captured_at_ms)
        x, y, width, height = face
        prediction = self._prediction_for(
            frame[y : y + height, x : x + width],
            max(0, int(captured_at_ms)),
        )
        frame_height, frame_width = frame.shape[:2]
        expression, confidence, negative = prediction
        facial_features = self._facial_features(frame, max(0, int(captured_at_ms)))
        return VisionObservation(
            at_ms=max(0, int(captured_at_ms)),
            status="detected",
            face_detected=True,
            box=FaceBox(
                x=x / frame_width,
                y=y / frame_height,
                width=width / frame_width,
                height=height / frame_height,
            ),
            expression=expression,
            confidence=confidence,
            features={
                "negative": negative,
                "expression": expression,
                **facial_features,
            },
        )

    def _facial_features(
        self,
        frame: np.ndarray,
        captured_at_ms: int,
        *,
        face_detected: bool = True,
    ) -> dict[str, float | bool]:
        if self._facial_cue_disabled:
            return self._empty_facial_features()
        if self._facial_cue_detector is None:
            if self._facial_model_path is None or not self._facial_model_path.is_file():
                return self._empty_facial_features()
            try:
                self._facial_cue_detector = MediaPipeFacialCueDetector(self._facial_model_path)
            except Exception as exc:
                logging.getLogger(__name__).warning("Facial action detector unavailable: %s", exc)
                self._facial_cue_disabled = True
                return self._empty_facial_features()
        if not face_detected:
            mark_no_face = getattr(self._facial_cue_detector, "mark_no_face", None)
            if callable(mark_no_face):
                mark_no_face(captured_at_ms)
            return self._empty_facial_features(available=True)
        try:
            features = self._facial_cue_detector.analyze_bgr(frame, captured_at_ms, self._cv2())
            return {**self._empty_facial_features(available=True), **features}
        except Exception as exc:
            logging.getLogger(__name__).warning("Facial action frame skipped: %s", exc)
            return self._empty_facial_features(available=True)

    def warmup(self) -> None:
        """Load the optional landmarker before the first time-limited frame request."""
        if self._facial_cue_detector is not None or self._facial_cue_disabled:
            return
        if self._facial_model_path is None or not self._facial_model_path.is_file():
            return
        try:
            self._facial_cue_detector = MediaPipeFacialCueDetector(self._facial_model_path)
        except Exception as exc:
            logging.getLogger(__name__).warning("Facial action detector unavailable: %s", exc)
            self._facial_cue_disabled = True

    @staticmethod
    def _empty_facial_features(*, available: bool = False) -> dict[str, float | bool]:
        return {
            "brow_tension": 0.0,
            "lip_tension": 0.0,
            "mouth_downturn": 0.0,
            "micro_expression": 0.0,
            "cue_duration_ms": 0.0,
            "facial_cue_confidence": 0.0,
            "baseline_ready": False,
            "facial_cue_available": available,
        }

    def close(self) -> None:
        detector = self._facial_cue_detector
        self._facial_cue_detector = None
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    def _cv2(self) -> Any:
        if self._cv2_module is None:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise VisionModelUnavailable("OpenCV is not installed") from exc
            self._cv2_module = cv2
        return self._cv2_module

    def _decode_and_resize(self, image_bytes: bytes, *, longest_edge: int) -> np.ndarray:
        if not image_bytes:
            raise FrameValidationError("empty_frame")
        if len(image_bytes) > self.MAX_FRAME_BYTES:
            raise FrameValidationError("frame_too_large")
        cv2 = self._cv2()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise FrameValidationError("invalid_image")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise FrameValidationError("invalid_dimensions")
        scale = min(1.0, longest_edge / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _detector(self) -> Any:
        if self._face_detector is None:
            cv2 = self._cv2()
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            self._face_detector = cv2.CascadeClassifier(cascade_path)
        return self._face_detector

    def _largest_face(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        cv2 = self._cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if faces is None or len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(int(x), frame_width - 1))
        y = max(0, min(int(y), frame_height - 1))
        width = max(1, min(int(width), frame_width - x))
        height = max(1, min(int(height), frame_height - y))
        return x, y, width, height

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._model_factory is None:
            if not self.weights_path.is_file():
                raise VisionModelUnavailable(f"YOLO weights not found: {self.weights_path}")
            if self._ultralytics_config_dir is not None:
                configure_ultralytics_runtime(self._ultralytics_config_dir)
            try:
                from ultralytics import YOLO  # type: ignore
            except ImportError as exc:
                raise VisionModelUnavailable("Ultralytics is not installed") from exc
            self._model_factory = YOLO
        self._model = self._model_factory(str(self.weights_path))
        if hasattr(self._model, "to"):
            self._model.to("cpu")
        return self._model

    def _predict(self, face: np.ndarray) -> tuple[str, float, float]:
        if face.size == 0:
            raise FrameValidationError("empty_face_crop")
        results = self._load_model()(face, device="cpu", verbose=False)
        if not results or getattr(results[0], "probs", None) is None:
            raise VisionModelUnavailable("YOLO returned no classification probabilities")
        result = results[0]
        index = int(result.probs.top1)
        expression = str(result.names[index]).strip().casefold()
        confidence = float(result.probs.top1conf.item())
        return expression, max(0.0, min(1.0, confidence)), self._expected_negative(result, index)

    @staticmethod
    def _expected_negative(result: Any, top_index: int) -> float:
        """Use the complete FER distribution so a near-tie is not reduced to neutral."""
        probs = result.probs
        distribution = getattr(probs, "data", None)
        if distribution is None:
            expression = str(result.names[top_index]).strip().casefold()
            return NEGATIVE_SEVERITY.get(expression, 0.25)
        if hasattr(distribution, "detach"):
            distribution = distribution.detach().cpu()
        if hasattr(distribution, "tolist"):
            distribution = distribution.tolist()
        if not isinstance(distribution, (list, tuple)) or not distribution:
            expression = str(result.names[top_index]).strip().casefold()
            return NEGATIVE_SEVERITY.get(expression, 0.25)
        total = sum(max(0.0, float(value)) for value in distribution)
        if total <= 0.0:
            expression = str(result.names[top_index]).strip().casefold()
            return NEGATIVE_SEVERITY.get(expression, 0.25)
        weighted = 0.0
        names = result.names
        for index, value in enumerate(distribution):
            label = names.get(index, "") if hasattr(names, "get") else names[index] if index < len(names) else ""
            name = str(label).strip().casefold()
            weighted += max(0.0, float(value)) * NEGATIVE_SEVERITY.get(name, 0.25)
        return max(0.0, min(1.0, weighted / total))

    def _prediction_for(self, face: np.ndarray, captured_at_ms: int) -> tuple[str, float, float]:
        with self._prediction_lock:
            prediction_due = (
                self._last_prediction_at_ms is None
                or captured_at_ms - self._last_prediction_at_ms >= self._classification_interval_ms
            )
            if prediction_due and self._background_classification and not self._prediction_inflight:
                self._prediction_inflight = True
                # Throttle attempts too, including failures before any result is cached.
                self._last_prediction_at_ms = captured_at_ms
                threading.Thread(
                    target=self._refresh_prediction,
                    args=(face.copy(), captured_at_ms),
                    daemon=True,
                ).start()
            elif prediction_due and not self._background_classification:
                self._last_prediction = self._predict(face)
                self._last_prediction_at_ms = captured_at_ms
            return self._last_prediction or ("neutral", 0.5, NEGATIVE_SEVERITY["neutral"])

    def _refresh_prediction(self, face: np.ndarray, captured_at_ms: int) -> None:
        try:
            prediction = self._predict(face)
            with self._prediction_lock:
                self._last_prediction = prediction
                self._last_prediction_at_ms = captured_at_ms
        except Exception as exc:
            # A non-critical classifier failure must not interrupt camera localisation.
            logging.getLogger(__name__).warning("Background emotion classification failed: %s", exc)
        finally:
            with self._prediction_lock:
                self._prediction_inflight = False
