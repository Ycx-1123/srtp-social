from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakeCamera:
    def __init__(self):
        self.frames = queue.Queue()
        self.last = None
        self.released = threading.Event()
        self.read_value = 0
        self.settings = []

    def isOpened(self):
        return True

    def set(self, key, value):
        self.settings.append((key, value))

    def read(self):
        try:
            self.last = self.frames.get(timeout=0.01)
        except queue.Empty:
            pass
        if isinstance(self.last, Exception):
            raise self.last
        if self.last is None:
            return False, None
        self.read_value = int(self.last[0, 0, 0])
        return True, self.last

    def release(self):
        self.released.set()

    def submit(self, value):
        self.frames.put(np.full((48, 64, 3), value, dtype=np.uint8))


class FakeLandmarker:
    def __init__(self, *, blocked=False, face=True):
        self.values = []
        self.timestamps = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.blocked = blocked
        self.face = face
        self.scores = {"browDownLeft": 0.0, "mouthSmileLeft": 0.02}
        self.landmarks = [SimpleNamespace(x=0.2, y=0.3), SimpleNamespace(x=0.6, y=0.7)]

    def detect_for_video(self, image, timestamp_ms):
        self.values.append(int(image.data[0, 0, 0]))
        self.timestamps.append(timestamp_ms)
        self.started.set()
        if self.blocked and len(self.values) == 1:
            self.release.wait(timeout=2)
        return SimpleNamespace(
            face_landmarks=[self.landmarks] if self.face else [],
            face_blendshapes=[[SimpleNamespace(category_name=name, score=score)
                              for name, score in self.scores.items()]],
        )

    def close(self):
        self.closed.set()


class DesktopCaptureTest(unittest.TestCase):
    def wait_for(self, predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("worker did not produce expected state")

    def worker(self, detector):
        from soci_ai.desktop.capture import CameraVisionWorker

        camera = FakeCamera()
        cv2 = SimpleNamespace(
            CAP_DSHOW=700, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38, COLOR_BGR2RGB=4,
            VideoCapture=lambda *_args: camera,
            cvtColor=lambda frame, _mode: frame[:, :, ::-1].copy(),
        )
        mp = SimpleNamespace(
            Image=lambda **kwargs: SimpleNamespace(**kwargs),
            ImageFormat=SimpleNamespace(SRGB=1),
        )
        self.enterContext(patch.dict(sys.modules, {"cv2": cv2}))
        self.enterContext(patch("soci_ai.desktop.capture._create_landmarker", return_value=(mp, detector)))
        worker = CameraVisionWorker(Path("fake.task"))
        self.addCleanup(worker.stop)
        self.addCleanup(detector.release.set)
        return worker, camera

    def test_landmark_box_is_padded_and_clamped_to_normalized_frame(self):
        from soci_ai.desktop.capture import _landmark_box

        box = _landmark_box([SimpleNamespace(x=0.2, y=0.3), SimpleNamespace(x=0.6, y=0.7)])
        for key, value in {"x": 0.168, "y": 0.268, "width": 0.464, "height": 0.464}.items():
            self.assertAlmostEqual(box[key], value)
        edge = _landmark_box([SimpleNamespace(x=-0.1, y=-0.1), SimpleNamespace(x=1.1, y=1.1)])
        self.assertEqual(edge, {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0})
        self.assertIsNone(_landmark_box([]))

    def test_blocked_inference_drops_intermediate_frames_and_stop_is_nonblocking(self):
        detector = FakeLandmarker(blocked=True)
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.assertTrue(detector.started.wait(timeout=1))
        first = worker.get_frame()
        camera.submit(2)
        camera.submit(3)
        self.wait_for(lambda: camera.read_value == 3)
        self.assertEqual(int(worker.get_frame()[2][0, 0, 0]), 3)
        self.assertEqual(int(first[2][0, 0, 0]), 1)
        self.assertEqual(detector.values, [1])
        detector.release.set()
        self.wait_for(lambda: len(detector.values) >= 2)
        self.assertEqual(detector.values[:2], [1, 3])
        self.assertGreater(detector.timestamps[1], detector.timestamps[0])
        vision = worker.get_vision()
        self.assertTrue(vision["face_detected"])
        self.assertAlmostEqual(vision["box"]["x"], 0.168)
        self.assertIsNone(vision["error"])
        started = time.monotonic()
        worker.stop()
        self.assertLess(time.monotonic() - started, 0.05)
        self.assertTrue(camera.released.wait(timeout=1))
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_stop_does_not_wait_for_blocked_detector(self):
        detector = FakeLandmarker(blocked=True)
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.assertTrue(detector.started.wait(timeout=1))
        self.assertTrue(worker.status()["workers_alive"])
        started = time.monotonic()
        worker.stop()
        self.assertLess(time.monotonic() - started, 0.05)
        self.assertTrue(camera.released.wait(timeout=1))
        self.assertFalse(detector.closed.is_set())
        self.assertTrue(worker.status()["workers_alive"])
        detector.release.set()
        self.assertTrue(detector.closed.wait(timeout=1))
        self.wait_for(lambda: not worker.status()["workers_alive"])
        self.assertFalse(worker.get_vision()["face_detected"])

    def test_no_face_and_calibration_are_real_tracker_outputs(self):
        detector = FakeLandmarker()
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.wait_for(lambda: worker.get_vision()["features"].get("baseline_ready", False))
        worker.calibrate()
        self.wait_for(lambda: not worker.get_vision()["features"].get("baseline_ready", False))
        detector.face = False
        self.wait_for(lambda: worker.get_vision()["status"] == "no_face")
        self.assertIsNone(worker.get_vision()["box"])
        self.assertEqual(worker.get_vision()["features"]["micro_expression"], 0.0)
        worker.stop()
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_model_initialization_error_is_visible_without_opening_devices(self):
        from soci_ai.desktop.capture import CameraVisionWorker

        with patch("soci_ai.desktop.capture._create_landmarker", side_effect=RuntimeError("missing model")):
            # Prevent the independent capture boundary from opening a real camera.
            camera = FakeCamera()
            cv2 = SimpleNamespace(
                CAP_DSHOW=700, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38, VideoCapture=lambda *_args: camera,
            )
            with patch.dict(sys.modules, {"cv2": cv2}):
                worker = CameraVisionWorker(Path("missing.task"))
                worker.start()
                self.wait_for(lambda: bool(worker.status()["error"]))
                self.assertIn("missing model", worker.status()["error"])
                self.assertEqual(worker.get_vision()["status"], "error")
                worker.stop()

    def test_capture_publishes_real_smile_channel_without_negative_cue(self):
        detector = FakeLandmarker()
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.wait_for(lambda: worker.get_vision()["features"]["baseline_ready"])
        detector.scores = {"browDownLeft": 0.0, "mouthSmileLeft": 0.25}
        self.wait_for(lambda: worker.get_vision()["features"].get("smile", 0) > 0.5)
        features = worker.get_vision()["features"]
        self.assertEqual(features["brow_tension"], 0)
        self.assertEqual(features["micro_expression"], 0)
        self.assertEqual(features["cue_source"], "mediapipe_blendshapes")
        worker.stop()
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_face_box_alone_cannot_publish_missing_expression_as_available(self):
        detector = FakeLandmarker()
        detector.scores = {}
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.wait_for(lambda: worker.get_vision()["face_detected"])
        features = worker.get_vision()["features"]
        self.assertFalse(features["facial_cue_available"])
        self.assertFalse(features["baseline_ready"])
        self.assertEqual(features["micro_expression"], 0)
        worker.stop()
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_real_landmark_validity_reaches_tracker_even_when_box_is_valid(self):
        detector = FakeLandmarker()
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.wait_for(lambda: worker.get_vision()["features"]["baseline_ready"])
        detector.scores = {"browDownLeft": 0.8, "mouthSmileLeft": 0.02}
        detector.landmarks = [SimpleNamespace(x=0.2, y=0.3, z=float("nan")),
                              SimpleNamespace(x=0.6, y=0.7, z=0.0)]
        self.wait_for(lambda: not worker.get_vision()["features"]["facial_cue_available"])
        self.assertTrue(worker.get_vision()["face_detected"])
        self.assertEqual(worker.get_vision()["features"]["brow_tension"], 0)
        worker.stop()
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_camera_error_discards_the_last_preview_frame(self):
        detector = FakeLandmarker(blocked=True)
        worker, camera = self.worker(detector)
        camera.submit(1)
        worker.start()
        self.assertTrue(detector.started.wait(timeout=1))
        self.assertIsNotNone(worker.get_frame())

        camera.frames.put(RuntimeError("camera disconnected"))

        self.wait_for(lambda: worker.status()["status"] == "error")
        self.assertIn("camera disconnected", worker.status()["error"])
        self.assertIsNone(worker.get_frame())
        self.assertFalse(worker.get_vision()["face_detected"])
        self.assertTrue(camera.released.wait(timeout=1))
        detector.release.set()
        self.assertTrue(detector.closed.wait(timeout=1))

    def test_import_is_lightweight_and_does_not_load_native_models(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import soci_ai.desktop.capture; assert not any(name in sys.modules for name in ('cv2', 'mediapipe', 'tensorflow', 'torch', 'ultralytics'))"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
