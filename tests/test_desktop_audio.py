import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from soci_ai.desktop.audio import BoundedAudioBuffer, AudioRecognitionWorker
from soci_ai.desktop.asr import RecognitionUpdate


class BoundedAudioTests(unittest.TestCase):
    def test_overflow_drops_oldest_not_newest_and_marks_discontinuity(self):
        queue = BoundedAudioBuffer(max_seconds=.3, sample_rate=10)
        for index in range(5):
            queue.push(np.array([index], dtype=np.float32), index / 10)
        samples, at, reset = queue.pop(timeout=0)
        self.assertEqual(samples[0], 2)
        self.assertTrue(reset)
        self.assertEqual(queue.dropped_samples, 2)
        self.assertLessEqual(queue.backlog_ms, 300)

    def test_idle_buffer_is_not_a_blocking_stop(self):
        queue = BoundedAudioBuffer()
        self.assertIsNone(queue.pop(timeout=0))
        queue.close()
        self.assertIsNone(queue.pop(timeout=.01))

    def test_capture_storage_owns_its_callback_data(self):
        queue = BoundedAudioBuffer()
        samples = np.ones(1600, dtype=np.float32)
        queue.push(samples, 1)
        samples[:] = 0
        self.assertEqual(queue.pop(timeout=0)[0].mean(), 1)

    def test_oversized_block_is_bounded_too(self):
        queue = BoundedAudioBuffer(max_seconds=.2, sample_rate=10)
        queue.push(np.arange(10, dtype=np.float32), 0)
        samples, _, reset = queue.pop(timeout=0)
        self.assertEqual(samples.tolist(), [8, 9])
        self.assertTrue(reset)

    def test_hardware_gap_requires_asr_reset_even_without_queue_overflow(self):
        queue = BoundedAudioBuffer()
        queue.mark_discontinuity()
        queue.push(np.zeros(1600, dtype=np.float32), 1)
        self.assertTrue(queue.pop(timeout=0)[2])

    def test_hardware_gap_discards_pre_gap_blocks_before_resetting_new_audio(self):
        queue = BoundedAudioBuffer()
        queue.push(np.ones(1600, dtype=np.float32), 1)
        queue.push(np.ones(1600, dtype=np.float32) * 2, 1.1)
        queue.mark_discontinuity()
        queue.push(np.ones(1600, dtype=np.float32) * 3, 1.2)
        samples, _, reset = queue.pop(timeout=0)
        self.assertTrue(reset)
        self.assertEqual(samples.mean(), 3)
        self.assertEqual(queue.dropped_samples, 3200)

    def test_slow_decoder_does_not_publish_an_old_result_as_fresh(self):
        worker = AudioRecognitionWorker(Path("unused-test-model"))
        clock = [1.0]
        class FakeRecognizer:
            def __init__(self, *args): pass
            def load(self): return self
            def feed(self, samples):
                clock[0] = 3.0
                return [RecognitionUpdate("女生不适合学工科", True, 2000)]
            def reset(self): worker.cancel.set()
            def close(self): pass
        class FakeStream:
            def __init__(self, **kwargs): self.callback = kwargs["callback"]
            def __enter__(self):
                self.callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
                return self
            def __exit__(self, *args): pass
        with patch("soci_ai.desktop.audio.StreamingChineseASR", FakeRecognizer), patch("soci_ai.desktop.audio.monotonic", lambda: clock[0]), patch.dict("sys.modules", {"sounddevice": SimpleNamespace(InputStream=FakeStream)}):
            worker._run()
        state = worker.snapshot()
        self.assertEqual(state["updates"], [])
        self.assertEqual(state["transcript"]["text"], "")
        self.assertEqual(state["asr_resets"], 1)

    def test_reset_generation_and_partial_clear_share_one_publication(self):
        worker = AudioRecognitionWorker(Path("unused-test-model"))
        worker.transcript = {"text": "abandoned partial"}
        worker.updates.append({"text": "abandoned partial"})
        observed = []
        class ObservingLock:
            def __enter__(self):
                observed.append((worker.reset_count, len(worker.updates)))
            def __exit__(self, *args): pass
        worker.lock = ObservingLock()
        worker._reset_transcript("reset")
        self.assertEqual(observed, [(0, 1)])
        self.assertEqual(worker.reset_count, 1)
        self.assertEqual(worker.transcript["text"], "")
        self.assertEqual(len(worker.updates), 0)


if __name__ == "__main__":
    unittest.main()
