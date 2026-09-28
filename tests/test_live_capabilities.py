from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from soci_ai.config import Settings
from soci_ai.live.capabilities import detect_live_capabilities
from soci_ai.live.domain import LiveState


class LiveCapabilitiesTest(unittest.TestCase):
    def test_capabilities_resolve_existing_models_without_importing_them(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            yolo = root / "best.pt"
            sense = root / "SenseVoiceSmall"
            yolo.write_bytes(b"weights")
            sense.mkdir()
            before = set(sys.modules)

            settings = Settings(
                root=root,
                yolo_weights=yolo,
                sensevoice_model_dir=sense,
            )
            result = detect_live_capabilities(settings)
            loaded = set(sys.modules) - before

            self.assertEqual(result.camera_capture, "browser")
            self.assertEqual(result.yolo.status, "ready")
            self.assertEqual(result.facial_actions.status, "model_missing")
            self.assertIn(result.sensevoice.status, {"ready", "package_missing"})
            self.assertTrue({"ultralytics", "torch", "funasr"}.isdisjoint(loaded))

    def test_live_state_starts_neutral_and_measured(self):
        state = LiveState.neutral("session-1")
        self.assertEqual(state.provenance, "measured")
        self.assertEqual(state.evidence_status, "insufficient")
        self.assertEqual(state.tree.mode, "observing")
        self.assertEqual(state.friendliness, 100.0)
        self.assertIsNone(state.last_frame_at_ms)

    def test_settings_read_live_model_overrides_and_clamp_fps(self):
        root = Path.cwd()
        settings = Settings.from_env(
            root,
            environ={
                "SOCI_YOLO_WEIGHTS": str(root / "custom.pt"),
                "SOCI_SENSEVOICE_MODEL_DIR": str(root / "sense"),
                "SOCI_LIVE_VIDEO_FPS": "12",
            },
        )
        self.assertEqual(settings.yolo_weights, (root / "custom.pt").resolve())
        self.assertEqual(settings.sensevoice_model_dir, (root / "sense").resolve())
        self.assertEqual(settings.live_video_fps, 6.0)

    def test_face_landmarker_model_override_is_exposed_as_a_local_capability(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "landmarker.task"
            model.write_bytes(b"model")

            settings = Settings.from_env(
                root,
                environ={"SOCI_FACE_LANDMARKER_MODEL": str(model)},
            )

            self.assertEqual(settings.face_landmarker_model, model.resolve())
            self.assertIn(detect_live_capabilities(settings).facial_actions.status, {"ready", "package_missing"})


if __name__ == "__main__":
    unittest.main()
