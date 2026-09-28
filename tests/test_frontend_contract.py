import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings


class FrontendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Settings.from_env(Path.cwd(), environ={})))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_live_view_contains_defense_critical_regions(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        for element_id in [
            "sbi-value",
            "evidence-status",
            "dialogue-stream",
            "modality-grid",
            "intervention-message",
            "scenario-controls",
            "connection-status",
            "sbi-chart",
        ]:
            self.assertIn(f'id="{element_id}"', html)

    def test_frontend_assets_are_local_and_served(self):
        html = self.client.get("/").text
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertEqual(self.client.get("/assets/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/assets/js/app.js").status_code, 200)
        self.assertIn('type="module"', html)

    def test_accessible_control_labels_are_present(self):
        html = self.client.get("/").text
        for label in ["播放场景", "暂停播放", "重置场景", "播放速度", "选择演示场景"]:
            self.assertIn(label, html)

    def test_dialogue_updates_do_not_scroll_the_whole_page(self):
        javascript = self.client.get("/assets/js/render.js").text
        self.assertNotIn("scrollIntoView", javascript)
        self.assertIn("scrollTop", javascript)

    def test_realtime_client_has_polling_fallback_and_transport_dependency(self):
        javascript = self.client.get("/assets/js/api.js").text
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        launcher = Path("start_showcase.ps1").read_text(encoding="utf-8")
        self.assertIn("startPolling", javascript)
        self.assertIn("/api/runtime/state", javascript)
        self.assertIn("websockets", requirements)
        self.assertIn("websockets", launcher)

    def test_reset_clears_stale_dialogue_and_profile_history(self):
        javascript = self.client.get("/assets/js/app.js").text
        self.assertIn("clearSessionView", javascript)
        self.assertIn("dialogue-stream", javascript)
        self.assertIn("appState.history=[]", javascript)

    def test_live_friendliness_formatter_preserves_visible_tenths(self):
        live_javascript = self.client.get("/assets/js/live.js").text
        app_javascript = self.client.get("/assets/js/app.js").text
        self.assertIn("export function formatLiveScore", live_javascript)
        self.assertIn("formatLiveScore(friendliness)", app_javascript)

    def test_live_view_exposes_measured_activity_and_short_transcript_segments(self):
        html = self.client.get("/").text
        app_javascript = self.client.get("/assets/js/app.js").text
        media_javascript = self.client.get("/assets/js/media.js").text
        self.assertIn('id="live-sequence"', html)
        self.assertIn('id="live-activity"', html)
        self.assertIn("signal_sequence", app_javascript)
        self.assertIn("MAX_UTTERANCE_SECONDS = 3", media_javascript)
        self.assertIn("SILENCE_HOLD_MS = 450", media_javascript)

    def test_audio_capture_uses_adaptive_noise_floor_for_low_volume_speech(self):
        media_javascript = self.client.get("/assets/js/media.js").text
        self.assertIn("this.noiseFloor", media_javascript)
        self.assertIn("adaptiveActive", media_javascript)
        self.assertIn("CALIBRATION_SAMPLES", media_javascript)
        self.assertIn("normalizeForTranscription", media_javascript)


if __name__ == "__main__":
    unittest.main()
