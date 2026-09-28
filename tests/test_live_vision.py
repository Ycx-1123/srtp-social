from __future__ import annotations

import unittest
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from soci_ai.live.vision import FrameValidationError, YoloVisionAdapter, configure_ultralytics_runtime


JPEG_BYTES = b"valid-jpeg"


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces

    def detectMultiScale(self, *_args, **_kwargs):
        return np.asarray(self.faces)


class FakeCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2GRAY = 2
    INTER_AREA = 3
    data = SimpleNamespace(haarcascades="")

    def __init__(self, faces=((10, 20, 50, 50), (70, 10, 10, 10))):
        self.faces = faces

    @staticmethod
    def imdecode(payload, _mode):
        if bytes(payload) != JPEG_BYTES:
            return None
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @staticmethod
    def cvtColor(frame, _mode):
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    @staticmethod
    def resize(frame, size, interpolation=None):
        del interpolation
        return np.zeros((size[1], size[0], frame.shape[2]), dtype=frame.dtype)

    def CascadeClassifier(self, _path):
        return FakeCascade(self.faces)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeModel:
    def __init__(self, factory):
        self.factory = factory

    def to(self, device):
        self.factory.device = device
        return self

    def __call__(self, _face, **kwargs):
        self.factory.inference_calls += 1
        self.factory.kwargs = kwargs
        probs = SimpleNamespace(top1=0, top1conf=Scalar(0.9))
        return [SimpleNamespace(probs=probs, names={0: "angry"})]


class FakeModelFactory:
    def __init__(self):
        self.calls = 0
        self.inference_calls = 0
        self.device = None
        self.kwargs = {}

    def __call__(self, _path):
        self.calls += 1
        return FakeModel(self)


class BlockingModel(FakeModel):
    def __call__(self, face, **kwargs):
        self.factory.started_calls += 1
        self.factory.started.set()
        self.factory.release.wait(timeout=2)
        return super().__call__(face, **kwargs)


class BlockingModelFactory(FakeModelFactory):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.started_calls = 0

    def __call__(self, _path):
        self.calls += 1
        return BlockingModel(self)


class LiveVisionTest(unittest.TestCase):
    def test_ultralytics_runtime_uses_a_writable_project_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "ultralytics"
            environment = {"YOLO_CONFIG_DIR": "C:/blocked/global/path"}

            configured = configure_ultralytics_runtime(target, environment)

            self.assertEqual(configured, target.resolve())
            self.assertEqual(environment["YOLO_CONFIG_DIR"], str(target.resolve()))
            self.assertTrue(target.is_dir())

    def test_adapter_returns_largest_face_box_and_negative_risk(self):
        factory = FakeModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=factory,
        )

        result = adapter.analyze(JPEG_BYTES, 1200)

        self.assertTrue(result.face_detected)
        self.assertEqual(
            result.box.model_dump(),
            {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5},
        )
        self.assertEqual(result.expression, "angry")
        self.assertGreater(result.features["negative"], 0.7)
        self.assertEqual(result.provenance, "measured")
        self.assertEqual(factory.calls, 1)
        self.assertEqual(factory.device, "cpu")
        self.assertEqual(factory.kwargs["device"], "cpu")

    def test_face_action_cues_are_attached_to_the_live_observation(self):
        class FakeCueDetector:
            def __init__(self):
                self.calls = []

            def analyze_bgr(self, frame, at_ms, _cv2):
                self.calls.append((frame.shape, at_ms))
                return {
                    "brow_tension": 0.86,
                    "lip_tension": 0.72,
                    "mouth_downturn": 0.0,
                    "micro_expression": 0.0,
                    "cue_duration_ms": 250.0,
                    "facial_cue_confidence": 0.91,
                    "baseline_ready": True,
                }

        cue_detector = FakeCueDetector()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=FakeModelFactory(),
            facial_cue_detector=cue_detector,
        )

        result = adapter.analyze(JPEG_BYTES, 1250)

        self.assertEqual(result.features["brow_tension"], 0.86)
        self.assertEqual(result.features["lip_tension"], 0.72)
        self.assertEqual(result.features["facial_cue_confidence"], 0.91)
        self.assertTrue(result.features["facial_cue_available"])
        self.assertEqual(cue_detector.calls[0][1], 1250)

    def test_rejects_invalid_and_oversized_frames_before_loading_model(self):
        factory = FakeModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=factory,
        )

        with self.assertRaises(FrameValidationError):
            adapter.analyze(b"not-an-image", 1)
        with self.assertRaises(FrameValidationError):
            adapter.analyze(b"x" * 2_000_001, 2)

        self.assertEqual(factory.calls, 0)

    def test_no_face_returns_neutral_observation_without_loading_model(self):
        factory = FakeModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(faces=()),
            model_factory=factory,
        )

        result = adapter.analyze(JPEG_BYTES, 500)

        self.assertFalse(result.face_detected)
        self.assertEqual(result.status, "no_face")
        self.assertEqual(factory.calls, 0)

    def test_face_box_updates_between_throttled_emotion_classifications(self):
        factory = FakeModelFactory()
        cv2 = FakeCv2(faces=((10, 20, 50, 50),))
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=cv2,
            model_factory=factory,
            classification_interval_ms=1500,
        )
        first = adapter.analyze(JPEG_BYTES, 1000)
        adapter._face_detector.faces = ((40, 20, 50, 50),)

        moved = adapter.analyze(JPEG_BYTES, 1200)

        self.assertEqual(first.box.x, 0.1)
        self.assertEqual(moved.box.x, 0.4)
        self.assertEqual(factory.inference_calls, 1)

    def test_default_emotion_classification_refreshes_within_half_a_second(self):
        factory = FakeModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=factory,
        )

        adapter.analyze(JPEG_BYTES, 1000)
        adapter.analyze(JPEG_BYTES, 1250)
        adapter.analyze(JPEG_BYTES, 1500)

        self.assertEqual(factory.inference_calls, 2)

    def test_background_classification_does_not_block_face_box_delivery(self):
        factory = BlockingModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=factory,
            background_classification=True,
        )

        started = time.perf_counter()
        immediate = adapter.analyze(JPEG_BYTES, 1000)
        elapsed = time.perf_counter() - started

        self.assertTrue(immediate.face_detected)
        self.assertLess(elapsed, 0.15)
        self.assertTrue(factory.started.wait(timeout=1))
        factory.release.set()
        for _ in range(20):
            refreshed = adapter.analyze(JPEG_BYTES, 3000)
            if refreshed.expression == "angry":
                break
            time.sleep(0.01)
        self.assertEqual(refreshed.expression, "angry")

    def test_due_frames_keep_updating_boxes_while_classifier_is_inflight(self):
        factory = BlockingModelFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(faces=((10, 20, 50, 50),)),
            model_factory=factory,
            background_classification=True,
        )
        workers = []
        try:
            first = adapter.analyze(JPEG_BYTES, 1000)
            self.assertEqual(first.expression, "neutral")
            self.assertTrue(factory.started.wait(timeout=1))
            for at_ms, x in ((1600, 30), (2200, 40)):
                adapter._face_detector.faces = ((x, 20, 50, 50),)
                observations = []
                delivered = threading.Event()

                def analyze_frame():
                    observations.append(adapter.analyze(JPEG_BYTES, at_ms))
                    delivered.set()

                worker = threading.Thread(target=analyze_frame, daemon=True)
                workers.append(worker)
                worker.start()
                self.assertTrue(delivered.wait(timeout=0.15), "face box waited for classifier")
                self.assertEqual(observations[0].box.x, x / 100)
                self.assertEqual(observations[0].expression, "neutral")
                self.assertEqual(factory.started_calls, 1)
        finally:
            factory.release.set()
            for worker in workers:
                worker.join(timeout=1)

    def test_failed_background_classification_is_logged_and_retried_on_interval(self):
        class FailingModel(FakeModel):
            def __call__(self, _face, **_kwargs):
                self.factory.inference_calls += 1
                raise RuntimeError("classification failed")

        class FailingFactory(FakeModelFactory):
            def __call__(self, _path):
                self.calls += 1
                return FailingModel(self)

        factory = FailingFactory()
        adapter = YoloVisionAdapter(
            Path("best.pt"),
            cv2_module=FakeCv2(),
            model_factory=factory,
            background_classification=True,
        )

        def wait_for_classifier():
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                with adapter._prediction_lock:
                    if not adapter._prediction_inflight:
                        return
                time.sleep(0.005)
            self.fail("classifier did not finish")

        with self.assertLogs("soci_ai.live.vision", level="WARNING") as logs:
            adapter.analyze(JPEG_BYTES, 1000)
            wait_for_classifier()
            for at_ms in (1001, 1100, 1499):
                observed = adapter.analyze(JPEG_BYTES, at_ms)
                wait_for_classifier()
                self.assertTrue(observed.face_detected)
                self.assertEqual(factory.inference_calls, 1)
            adapter.analyze(JPEG_BYTES, 1500)
            wait_for_classifier()
            self.assertEqual(factory.inference_calls, 2)
        self.assertTrue(any("classification failed" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
