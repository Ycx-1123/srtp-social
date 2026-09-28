from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"


class FakeStream:
    def __init__(self):
        self.pending = 0
        self.text = ""
        self.endpoint = False

    def accept_waveform(self, sample_rate, samples):
        self.pending += 1


class FakeRecognizer:
    """Only the external native model boundary is replaced in these tests."""

    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return stream.pending > 0

    def decode_stream(self, stream):
        stream.pending -= 1
        stream.text, stream.endpoint = next(self.predictions, ("", False))

    def get_result(self, stream):
        return stream.text

    def is_endpoint(self, stream):
        return stream.endpoint

    def reset(self, stream):
        stream.text = ""
        stream.endpoint = False


class DesktopASRTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_dir = Path(self.directory.name)
        for filename in ("model.int8.onnx", "tokens.txt"):
            (self.model_dir / filename).touch()

    def asr(self, predictions):
        from soci_ai.desktop.asr import StreamingChineseASR

        recognizer = FakeRecognizer(predictions)
        module = SimpleNamespace(OnlineRecognizer=SimpleNamespace(
            from_zipformer2_ctc=lambda **kwargs: recognizer,
        ))
        context = patch.dict(sys.modules, {"sherpa_onnx": module})
        context.start()
        self.addCleanup(context.stop)
        return StreamingChineseASR(self.model_dir)

    def test_returns_partial_before_endpoint_and_suppresses_unchanged_text(self):
        asr = self.asr([("你", False), ("你", False), ("你好", False)])
        audio = np.zeros(1600, dtype=np.float32)
        first = asr.feed(audio)
        self.assertEqual([(u.text, u.is_final) for u in first], [("你", False)])
        self.assertGreaterEqual(first[0].latency_ms, 0)
        self.assertEqual(asr.feed(audio), [])
        self.assertEqual([(u.text, u.is_final) for u in asr.feed(audio)], [("你好", False)])

    def test_endpoint_finalizes_unchanged_text_and_starts_a_new_utterance(self):
        asr = self.asr([("你好", False), ("你好", True), ("再见", False)])
        audio = np.zeros(1600, dtype=np.float32)
        asr.feed(audio)
        self.assertEqual([(u.text, u.is_final) for u in asr.feed(audio)], [("你好", True)])
        self.assertEqual([(u.text, u.is_final) for u in asr.feed(audio)], [("再见", False)])

    def test_empty_endpoint_does_not_generate_a_transcript(self):
        asr = self.asr([("", True)])
        self.assertEqual(asr.feed(np.zeros(1600, dtype=np.float32)), [])

    def test_reset_discards_previous_partial_and_close_rejects_more_audio(self):
        asr = self.asr([("你好", False), ("你好", False)])
        audio = np.zeros(1600, dtype=np.float32)
        asr.feed(audio)
        asr.reset()
        self.assertEqual([u.text for u in asr.feed(audio)], ["你好"])
        asr.close()
        asr.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            asr.feed(audio)

    def test_rejects_long_audio_blocks_instead_of_building_a_backlog(self):
        asr = self.asr([("新音频", False)])
        with self.assertRaises(BufferError):
            asr.feed(np.zeros(32000, dtype=np.float32))
        self.assertEqual([u.text for u in asr.feed(np.zeros(1600, dtype=np.float32))], ["新音频"])

    def test_rejects_invalid_audio_and_empty_audio_does_not_load_model(self):
        from soci_ai.desktop.asr import StreamingChineseASR

        asr = StreamingChineseASR(Path("missing"))
        self.assertEqual(asr.feed(np.zeros(0, dtype=np.float32)), [])
        for samples, rate in ((np.zeros((100, 2)), 16000),
                              (np.array([np.nan], dtype=np.float32), 16000),
                              (np.zeros(100, dtype=np.int16), 16000),
                              (np.zeros(100, dtype=np.float32), 0)):
            with self.subTest(shape=samples.shape, dtype=samples.dtype, rate=rate):
                with self.assertRaises(ValueError):
                    asr.feed(samples, rate)

    def test_missing_model_error_identifies_required_files(self):
        from soci_ai.desktop.asr import StreamingChineseASR

        asr = StreamingChineseASR(self.model_dir / "missing")
        with self.assertRaisesRegex(FileNotFoundError, "model.int8.onnx.*tokens.txt"):
            asr.load()

    def test_missing_optional_native_dependency_is_actionable(self):
        from soci_ai.desktop.asr import StreamingChineseASR

        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            with self.assertRaisesRegex(RuntimeError, "sherpa-onnx"):
                StreamingChineseASR(self.model_dir).load()

    def test_one_block_preserves_partial_changes_before_its_endpoint(self):
        asr = self.asr([("你", False), ("你好", True)])
        self.assertEqual(
            [(u.text, u.is_final) for u in asr.feed(np.zeros(3200, dtype=np.float32))],
            [("你", False), ("你好", True)],
        )


@unittest.skipUnless((REAL_MODEL_DIR / "model.int8.onnx").is_file(), "Optional official ASR model is not installed")
class RealDesktopASRTest(unittest.TestCase):
    def test_official_recording_produces_live_partials_and_silence_endpoint(self):
        from soci_ai.desktop.asr import StreamingChineseASR

        wav = REAL_MODEL_DIR / "test_wavs" / "0.wav"
        if not wav.is_file():
            self.skipTest("Official model's test recording is not installed")
        with wave.open(str(wav), "rb") as recording:
            self.assertEqual(recording.getnchannels(), 1)
            self.assertEqual(recording.getsampwidth(), 2)
            sample_rate = recording.getframerate()
            audio = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").astype(np.float32) / 32768
        asr = StreamingChineseASR(REAL_MODEL_DIR).load()
        self.addCleanup(asr.close)
        partial_positions = []
        updates = []
        step = sample_rate // 10
        for offset in range(0, len(audio), step):
            current = asr.feed(audio[offset:offset + step], sample_rate)
            partial_positions.extend(offset + step for u in current if not u.is_final)
            updates.extend(current)
        for _ in range(20):
            updates.extend(asr.feed(np.zeros(step, dtype=np.float32), sample_rate))
        self.assertTrue(partial_positions, "Model must provide text while the recording is being streamed")
        self.assertLess(partial_positions[0], len(audio), "First text should arrive before the recording ends")
        final_texts = [u.text for u in updates if u.is_final]
        self.assertTrue(final_texts, "Trailing silence should finalize speech")
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for text in final_texts for ch in text))


if __name__ == "__main__":
    unittest.main()
