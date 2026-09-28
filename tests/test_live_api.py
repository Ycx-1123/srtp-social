from __future__ import annotations

import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings
from soci_ai.live.domain import FaceBox, TranscriptObservation, VisionObservation
from soci_ai.live.runtime import LiveSessionRuntime
from soci_ai.live.semantics import MicroaggressionAnalyzer


JPEG_BYTES = b"JPEG_BYTES"
LOUD_PCM = (12000).to_bytes(2, "little", signed=True) * 4000


class StubVision:
    calls: list[bytes]

    def __init__(self):
        self.calls = []

    def analyze(self, image: bytes, at_ms: int):
        self.calls.append(image)
        return VisionObservation(
            at_ms=at_ms,
            status="detected",
            face_detected=True,
            box=FaceBox(x=0.1, y=0.1, width=0.5, height=0.5),
            expression="neutral",
            confidence=0.9,
            features={"negative": 0.05, "expression": "neutral"},
        )


class StubTranscriber:
    def transcribe_pcm16(self, _pcm: bytes, _sample_rate: int, at_ms: int):
        return TranscriptObservation(
            at_ms=at_ms,
            status="completed",
            text="女生不适合学工科",
            tags=["ANGRY"],
            confidence=0.92,
        )


def runtime_factory(session_id: str) -> LiveSessionRuntime:
    return LiveSessionRuntime(
        vision=StubVision(),
        transcriber=StubTranscriber(),
        analyzer=MicroaggressionAnalyzer.default(),
        session_id=session_id,
    )


class LiveApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(
            create_app(
                Settings.from_env(Path.cwd(), environ={}),
                live_runtime_factory=runtime_factory,
            )
        )
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_live_session_accepts_real_inputs_and_never_persists_raw_media(self):
        session = self.client.post("/api/live/sessions").json()
        session_id = session["session_id"]
        frame = self.client.post(
            f"/api/live/sessions/{session_id}/frame?at_ms=1000",
            content=JPEG_BYTES,
            headers={"content-type": "image/jpeg"},
        )
        audio = self.client.post(
            f"/api/live/sessions/{session_id}/audio-chunk?sample_rate=16000&at_ms=2000",
            content=LOUD_PCM,
            headers={"content-type": "application/octet-stream"},
        )

        self.assertEqual(frame.status_code, 200)
        self.assertEqual(audio.status_code, 200)
        stored = self.client.app.state.store.get_session(session_id)
        serialized = json.dumps(stored, ensure_ascii=False)
        self.assertNotIn("JPEG_BYTES", serialized)
        self.assertNotIn("raw_audio", serialized)
        self.assertIn("女生不适合学工科", serialized)

    def test_start_and_stop_are_idempotent(self):
        first = self.client.post("/api/live/sessions").json()
        session_id = first["session_id"]
        second_stop = self.client.post(f"/api/live/sessions/{session_id}/stop")
        third_stop = self.client.post(f"/api/live/sessions/{session_id}/stop")
        self.assertEqual(second_stop.status_code, 200)
        self.assertEqual(third_stop.status_code, 200)
        self.assertEqual(third_stop.json()["status"], "stopped")

    def test_invalid_frame_type_and_unknown_session_are_structured(self):
        session_id = self.client.post("/api/live/sessions").json()["session_id"]
        invalid = self.client.post(
            f"/api/live/sessions/{session_id}/frame?at_ms=1000",
            content=b"bad",
            headers={"content-type": "text/plain"},
        )
        missing = self.client.get("/api/live/sessions/not-real/state")
        self.assertEqual(invalid.status_code, 415)
        self.assertEqual(invalid.json()["detail"]["code"], "unsupported_frame_type")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"]["code"], "unknown_live_session")

    def test_capabilities_and_audio_features_routes(self):
        capabilities = self.client.get("/api/live/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(capabilities.json()["camera_capture"], "browser")
        session_id = self.client.post("/api/live/sessions").json()["session_id"]
        response = self.client.post(
            f"/api/live/sessions/{session_id}/audio-features",
            json={"at_ms": 600, "rms": 0.4, "peak": 0.8, "speech_ratio": 0.7, "pace": 5.0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["metrics"]["tone_pressure"], 0)


if __name__ == "__main__":
    unittest.main()
