from __future__ import annotations

import asyncio
import threading
import unittest

from soci_ai.live.domain import FaceBox, TranscriptObservation, TreeState, VisionObservation
from soci_ai.live.audio import AudioFeaturePayload
from soci_ai.live.runtime import LiveSessionRuntime, TreeStateMapper
from soci_ai.live.semantics import MicroaggressionAnalyzer


class BlockingVision:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[bytes] = []

    def analyze(self, image: bytes, at_ms: int) -> VisionObservation:
        self.calls.append(image)
        self.started.set()
        self.release.wait(timeout=2)
        return VisionObservation(
            at_ms=at_ms,
            status="detected",
            face_detected=True,
            box=FaceBox(x=0.1, y=0.1, width=0.5, height=0.5),
            expression="angry",
            confidence=0.9,
            features={"negative": 0.8, "expression": "angry"},
        )


class StubTranscriber:
    def transcribe_pcm16(self, _pcm: bytes, _sample_rate: int, at_ms: int):
        return TranscriptObservation(
            at_ms=at_ms,
            status="completed",
            text="我们先看事实",
            confidence=0.9,
        )


class BlockingTranscriber:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe_pcm16(self, _pcm: bytes, _sample_rate: int, at_ms: int):
        self.started.set()
        self.release.wait(timeout=2)
        return TranscriptObservation(
            at_ms=at_ms,
            status="completed",
            text="我们先看事实",
            confidence=0.9,
        )


class LiveRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, vision=None, transcriber=None):
        return LiveSessionRuntime(
            vision=vision or BlockingVision(),
            transcriber=transcriber or StubTranscriber(),
            analyzer=MicroaggressionAnalyzer.default(),
        )

    async def test_live_multimodal_events_raise_self_risk_and_then_recover(self):
        runtime = self.make_runtime()
        await runtime.start()
        await runtime.accept_vision(negative=0.72, confidence=0.9, at_ms=1000)
        await runtime.accept_audio(arousal=0.74, confidence=0.88, at_ms=1200)
        await runtime.accept_transcript("女生不适合学工科", at_ms=1500)

        high = runtime.state()
        self.assertEqual(high.provenance, "measured")
        self.assertEqual(high.evidence_status, "sufficient")
        self.assertGreaterEqual(high.sbi, 45)
        self.assertIn(high.tree.mode, {"signal", "risk"})

        await runtime.tick(12000)
        self.assertEqual(runtime.state().tree.mode, "recovering")

    async def test_busy_runtime_drops_stale_frames(self):
        vision = BlockingVision()
        runtime = self.make_runtime(vision)
        await runtime.start()
        first = asyncio.create_task(runtime.submit_frame(b"first", 1000))
        await asyncio.to_thread(vision.started.wait, 1)
        await runtime.submit_frame(b"stale-middle", 1100)
        await runtime.submit_frame(b"latest", 1300)
        vision.release.set()
        await first

        self.assertEqual(runtime.vision_calls, [b"first", b"latest"])
        self.assertEqual(runtime.state().last_frame_at_ms, 1300)

    async def test_start_and_stop_are_idempotent(self):
        runtime = self.make_runtime()
        first = await runtime.start()
        second = await runtime.start()
        self.assertEqual(second.session_id, first.session_id)
        await runtime.stop()
        stopped = await runtime.stop()
        self.assertEqual(stopped.status, "stopped")

    async def test_low_risk_measured_face_enters_friendly_tree_state(self):
        runtime = self.make_runtime()
        await runtime.start()

        await runtime.accept_vision(negative=0.02, confidence=0.9, at_ms=1000)

        self.assertEqual(runtime.state().tree.mode, "friendly")

    async def test_sustained_facial_action_cue_moves_live_sbi_without_waiting_for_fer_refresh(self):
        runtime = self.make_runtime()
        await runtime.start()
        neutral = VisionObservation(
            at_ms=1000,
            status="detected",
            face_detected=True,
            box=FaceBox(x=0.1, y=0.1, width=0.5, height=0.5),
            confidence=0.9,
            features={"negative": 0.05, "micro_expression": 0.0},
        )
        baseline = runtime._accept_vision_observation(neutral).sbi

        snapshots = []
        for at_ms in range(1500, 2750, 250):
            observation = neutral.model_copy(
                update={
                    "at_ms": at_ms,
                    "features": {"negative": 0.05, "micro_expression": 0.9},
                }
            )
            snapshots.append(runtime._accept_vision_observation(observation).sbi)

        self.assertGreater(snapshots[0], baseline + 5)
        self.assertGreater(snapshots[-1], snapshots[0] + 8)

    def test_friendly_tree_target_changes_with_live_risk_score(self):
        mapper = TreeStateMapper()
        previous = TreeState.observing()

        lower = mapper.map(8.0, "insufficient", previous, measured_context=True)
        higher = mapper.map(18.0, "insufficient", lower, measured_context=True)

        self.assertEqual(lower.mode, "friendly")
        self.assertEqual(higher.mode, "friendly")
        self.assertNotEqual(lower.health, higher.health)
        self.assertNotEqual(lower.bloom, higher.bloom)

    async def test_audio_feature_updates_continue_during_transcription(self):
        transcriber = BlockingTranscriber()
        runtime = self.make_runtime(transcriber=transcriber)
        await runtime.start()
        pending_transcript = asyncio.create_task(
            runtime.submit_audio_chunk(b"\x60\x1f" * 4000, 16000, 1000)
        )
        await asyncio.to_thread(transcriber.started.wait, 1)

        live_state = await runtime.submit_audio_features(
            AudioFeaturePayload(at_ms=1200, rms=0.38, peak=0.66, speech_ratio=0.8, pace=4.5)
        )

        self.assertEqual(live_state.listening_state, "transcribing")
        self.assertEqual(live_state.signal_sequence, 1)
        self.assertGreater(live_state.signal_activity, 0)
        transcriber.release.set()
        await pending_transcript


if __name__ == "__main__":
    unittest.main()
