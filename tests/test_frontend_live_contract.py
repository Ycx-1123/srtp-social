from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings


class FrontendLiveContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Settings.from_env(Path.cwd(), environ={})))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_live_media_module_uses_real_camera_and_microphone(self):
        response = self.client.get("/assets/js/media.js")
        self.assertEqual(response.status_code, 200)
        source = response.text
        self.assertIn("navigator.mediaDevices.getUserMedia", source)
        self.assertIn("video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: \"user\" }", source)
        self.assertIn("audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }", source)
        self.assertIn('facingMode: "user"', source)
        self.assertNotIn("MediaRecorder", source)
        self.assertIn("encodePcm16", source)
        self.assertIn("AudioWorkletNode", source)
        self.assertIn("createScriptProcessor", source)

    def test_default_view_is_real_self_check_and_replay_is_labeled(self):
        html = self.client.get("/").text
        for element_id in [
            "living-tree",
            "camera-video",
            "face-overlay",
            "facial-cue-readout",
            "audio-wave",
            "speech-state",
            "live-transcript",
            "start-live",
            "stop-live",
            "live-sbi",
            "live-friendliness",
        ]:
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("备用回放", html)
        live_markup = html.split('id="view-live"', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("播放场景", live_markup)
        self.assertIn("MEASURED", live_markup)

    def test_live_interface_loads_tree_and_controller_modules(self):
        source = self.client.get("/assets/js/app.js").text
        self.assertIn('from "./tree.js"', source)
        self.assertIn('from "./live.js"', source)
        self.assertIn("new LivingTree", source)
        self.assertIn("new LiveController", source)
        styles = self.client.get("/assets/styles.css").text
        self.assertIn(".tree-stage", styles)
        self.assertIn(".self-camera", styles)
        self.assertIn(".voice-console", styles)

    def test_face_overlay_uses_source_dimensions_and_mirror_flag(self):
        response = self.client.get("/assets/js/live.js")
        self.assertEqual(response.status_code, 200)
        source = response.text
        self.assertIn("state.vision.box", source)
        self.assertIn("video.videoWidth", source)
        self.assertIn("data-mirrored", source)
        self.assertIn("strokeRect", source)

    def test_live_client_calls_only_local_realtime_routes(self):
        source = self.client.get("/assets/js/live-api.js").text
        for route in [
            "/api/live/capabilities",
            "/api/live/sessions",
            "/frame",
            "/audio-features",
            "/audio-chunk",
            "/stop",
        ]:
            self.assertIn(route, source)
        self.assertNotIn("https://", source)
        self.assertNotIn("http://", source)


if __name__ == "__main__":
    unittest.main()
