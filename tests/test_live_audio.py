from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from soci_ai.live.audio import (
    AudioFeaturePayload,
    SenseVoiceTranscriber,
    validate_audio_features,
)


LOUD_PCM = (np.ones(4000, dtype="<i2") * 12000).tobytes()
SILENT_PCM = np.zeros(4000, dtype="<i2").tobytes()


class FakeSenseVoiceModel:
    def __init__(self):
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return [{"text": "<|zh|><|ANGRY|>女生不适合学工科"}]


class LiveAudioTest(unittest.TestCase):
    def test_sensevoice_result_strips_tags_and_keeps_emotion(self):
        model = FakeSenseVoiceModel()
        transcriber = SenseVoiceTranscriber(
            Path("SenseVoiceSmall"),
            model_factory=lambda **_kwargs: model,
        )

        result = transcriber.transcribe_pcm16(LOUD_PCM, 16000, 3000)

        self.assertEqual(result.text, "女生不适合学工科")
        self.assertIn("ANGRY", result.tags)
        self.assertEqual(result.status, "completed")
        self.assertEqual(model.calls, 1)

    def test_silent_short_and_stale_audio_are_not_transcribed(self):
        model = FakeSenseVoiceModel()
        transcriber = SenseVoiceTranscriber(
            Path("SenseVoiceSmall"),
            model_factory=lambda **_kwargs: model,
        )

        self.assertEqual(transcriber.transcribe_pcm16(SILENT_PCM, 16000, 1000).status, "silent")
        self.assertEqual(transcriber.transcribe_pcm16(b"\0\0", 16000, 2000).status, "too_short")
        transcriber.transcribe_pcm16(LOUD_PCM, 16000, 3000)
        self.assertEqual(transcriber.transcribe_pcm16(LOUD_PCM, 16000, 2500).status, "stale")
        self.assertEqual(model.calls, 1)

    def test_browser_audio_features_produce_measured_arousal(self):
        observation = validate_audio_features(
            AudioFeaturePayload(
                at_ms=700,
                rms=0.55,
                peak=0.90,
                speech_ratio=0.75,
                pace=5.8,
            )
        )

        self.assertEqual(observation.status, "speech")
        self.assertGreater(observation.arousal, 0.6)
        self.assertEqual(observation.features["arousal"], observation.arousal)
        self.assertEqual(observation.provenance, "measured")


if __name__ == "__main__":
    unittest.main()
