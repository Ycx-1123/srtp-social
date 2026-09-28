"""Latest-frame camera capture and local facial landmarks for the desktop UI.

OpenCV and MediaPipe load only on worker threads. MediaPipe's optional
TensorFlow import supports documentation generation, not Face Landmarker;
the native runtime suppresses that optional dependency before importing it.
No emotion classifier, Torch model, or frame queue is used here.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from soci_ai.desktop.facial import FacialCueTracker


def _create_landmarker(model_path: Path) -> tuple[Any, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
    # MediaPipe gracefully handles ImportError for its optional doc generator.
    # Preserve an already imported TensorFlow module if the host uses it.
    sys.modules.setdefault("tensorflow", None)
    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def _landmark_box(landmarks: Any) -> dict[str, float] | None:
    points = [(float(point.x), float(point.y)) for point in landmarks]
    points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if not points:
        return None
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    top, bottom = min(y for _, y in points), max(y for _, y in points)
    pad_x, pad_y = (right - left) * 0.08, (bottom - top) * 0.08
    left, right = max(0.0, left - pad_x), min(1.0, right + pad_x)
    top, bottom = max(0.0, top - pad_y), min(1.0, bottom + pad_y)
    if right <= left or bottom <= top:
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


class CameraVisionWorker:
    """Two daemon workers sharing one immutable latest-frame slot.

    start() and stop() never load models or wait for native calls. get_frame()
    returns the published BGR array for read-only preview; consumers must copy
    it before modifying it. A restart waits until the previous workers exit.
    captured_at uses time.monotonic(), as does the controller's local timeline.
    """

    inference_interval_seconds = 0.08

    def __init__(self, model_path: Path, camera_index: int = 0):
        self.model_path = Path(model_path)
        self.camera_index = int(camera_index)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stop.set()
        self._calibrate = threading.Event()
        self._threads: list[threading.Thread] = []
        self._frame: tuple[int, float, Any] | None = None
        self._sequence = 0
        self._camera_ready = False
        self._vision_ready = False
        self._error: str | None = None
        self._vision = self._empty_vision("stopped")

    @staticmethod
    def _empty_vision(status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "face_detected": False, "box": None,
            "features": FacialCueTracker._empty(),
            "latency_ms": 0.0, "captured_at": None,
            "status": status, "error": error,
        }

    def start(self) -> None:
        with self._lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._stop.clear()
            self._calibrate.clear()
            self._frame = None
            self._sequence = 0
            self._error = None
            self._camera_ready = False
            self._vision_ready = False
            self._vision = self._empty_vision("starting")
            self._threads = [
                threading.Thread(target=self._capture_loop, name="desktop-camera", daemon=True),
                threading.Thread(target=self._vision_loop, name="desktop-landmarks", daemon=True),
            ]
            for thread in self._threads:
                thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame = None
            self._vision = self._empty_vision("error" if self._error else "stopped", self._error)

    def calibrate(self) -> None:
        """Reset the personal facial baseline on the inference worker."""
        self._calibrate.set()

    def get_frame(self) -> tuple[int, float, Any] | None:
        with self._lock:
            return self._frame

    def get_vision(self) -> dict[str, Any]:
        with self._lock:
            vision = {**self._vision, "features": dict(self._vision["features"])}
            if vision["box"] is not None:
                vision["box"] = dict(vision["box"])
            return vision

    def status(self) -> dict[str, Any]:
        with self._lock:
            alive = any(thread.is_alive() for thread in self._threads)
            state = (
                "error" if self._error else "stopping" if self._stop.is_set() and alive
                else "stopped" if self._stop.is_set() else "running" if self._camera_ready
                and self._vision_ready else "starting"
            )
            return {
                "status": state, "running": alive and not self._stop.is_set(),
                "workers_alive": alive,
                "camera_ready": self._camera_ready, "vision_ready": self._vision_ready,
                "camera_index": self.camera_index, "frame_seq": self._sequence,
                "error": self._error,
            }

    def _fail(self, source: str, error: Exception) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._error = f"{source}: {error}"
            self._frame = None
            self._vision = self._empty_vision("error", self._error)
            self._stop.set()

    def _capture_loop(self) -> None:
        camera = None
        try:
            import cv2

            if self._stop.is_set():
                return
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            for property_id, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_BUFFERSIZE, 1),
            ):
                camera.set(property_id, value)
            with self._lock:
                self._camera_ready = True
            last_good = time.monotonic()
            while not self._stop.is_set():
                ok, frame = camera.read()
                captured_at = time.monotonic()
                if self._stop.is_set():
                    break
                if not ok or frame is None:
                    if captured_at - last_good > 3.0:
                        raise RuntimeError("Camera has not returned a frame for three seconds")
                    self._stop.wait(0.01)
                    continue
                last_good = captured_at
                snapshot = frame.copy()
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._sequence += 1
                    self._frame = (self._sequence, captured_at, snapshot)
        except Exception as exc:
            self._fail("Camera", exc)
        finally:
            if camera is not None:
                camera.release()
            with self._lock:
                self._camera_ready = False

    def _vision_loop(self) -> None:
        landmarker = None
        try:
            import cv2

            if self._stop.is_set():
                return
            mp, landmarker = _create_landmarker(self.model_path)
            tracker = FacialCueTracker(calibration_frames=8, sustain_ms=250)
            with self._lock:
                self._vision_ready = True
            last_sequence, last_timestamp = 0, -1
            while not self._stop.is_set():
                started = time.monotonic()
                if self._calibrate.is_set():
                    tracker.reset()
                    self._calibrate.clear()
                latest = self.get_frame()
                if latest is None or latest[0] == last_sequence:
                    self._stop.wait(self.inference_interval_seconds)
                    continue
                sequence, captured_at, frame = latest
                last_sequence = sequence
                timestamp_ms = max(last_timestamp + 1, int(captured_at * 1000))
                last_timestamp = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)
                landmarks = getattr(result, "face_landmarks", None) or []
                box = _landmark_box(landmarks[0]) if landmarks else None
                blendshapes = getattr(result, "face_blendshapes", None) or []
                features = tracker.update(
                    blendshapes[0] if blendshapes else {}, timestamp_ms,
                    face_detected=box is not None,
                    landmarks=landmarks[0] if landmarks else [],
                )
                observation = {
                    "face_detected": box is not None, "box": box,
                    "features": features,
                    "latency_ms": (time.monotonic() - started) * 1000,
                    "captured_at": captured_at,
                    "status": "detected" if box else "no_face", "error": None,
                }
                with self._lock:
                    if not self._stop.is_set():
                        self._vision = observation
                self._stop.wait(max(0.0, self.inference_interval_seconds - (time.monotonic() - started)))
        except Exception as exc:
            self._fail("Face landmarks", exc)
        finally:
            if landmarker is not None:
                landmarker.close()
            with self._lock:
                self._vision_ready = False
