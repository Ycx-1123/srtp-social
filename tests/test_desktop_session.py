import unittest
from soci_ai.desktop.session import DesktopSession


class FakeCamera:
    def __init__(self, *args, **kwargs):
        self.alive = False
        self.cancelled = False
    def start(self): self.alive = True
    def stop(self): self.cancelled = True
    def status(self): return {"workers_alive": self.alive, "status": "stopping" if self.cancelled else "running"}
    def get_frame(self): return None
    def get_vision(self): return {"captured_at": None, "face_detected": False, "features": {}}
    def calibrate(self): pass


class FakeAudio:
    def __init__(self, *args, **kwargs): self.alive = False
    def start(self): self.alive = True
    def stop(self): self.alive = False
    def is_alive(self): return self.alive
    def snapshot(self): return {"audio": {}, "transcript": {}, "updates": []}


class SessionTests(unittest.TestCase):
    def test_missing_face_baseline_does_not_hide_real_language_warning(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        session.audio.snapshot = lambda: {"audio": {}, "transcript": {}, "updates": [
            {"text": "女生不适合学工科", "is_final": True, "captured_at": 100, "observed_at": 100}
        ]}
        state = session.poll()
        self.assertEqual(state.get('language_alert', {}).get('text'), '女生不适合学工科')
        self.assertIn('语言', state['suggestion_title'])

    def test_missing_baseline_does_not_replace_readable_language_card_after_score_decays(self):
        now = [100]
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: now[0])
        session.start()
        session.assessment.observe_text('学历低一点的人更适合跑腿', 0, True)
        session.poll()
        now[0] = 109
        state = session.poll()
        self.assertTrue(state.get('language_alert'))
        self.assertIn('学历', state['suggestion_title'])

    def test_ended_session_keeps_aggregate_coaching(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        session.assessment.observe_text("女生不适合学工科", 0, True)
        session.poll()
        session.stop()
        self.assertTrue(any(card["kind"] == "language" for card in session.latest_report["advice"]))

    def test_repeated_start_does_not_open_duplicate_devices(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        self.assertTrue(session.start())
        camera = session.camera
        self.assertFalse(session.start())
        self.assertIs(camera, session.camera)

    def test_stop_clears_live_scores_and_restart_waits_for_device_release(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        session.poll()
        state = session.stop()
        self.assertIsNone(state["sbi"])
        self.assertEqual(state["status"], "stopped")
        self.assertFalse(session.start())
        session.camera.alive = False
        self.assertTrue(session.start())

    def test_missing_timestamp_does_not_break_startup_poll(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        self.assertIsNone(session.poll()["sbi"])

    def test_stop_still_releases_devices_when_a_provider_is_broken(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        def broken(): raise RuntimeError("native provider error")
        session.camera.get_vision = broken
        session.stop()
        self.assertTrue(session.camera.cancelled)
        self.assertFalse(session.audio.alive)

    def test_old_transcript_is_not_made_fresh_by_delayed_decode(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        session.audio.snapshot = lambda: {"audio": {}, "transcript": {}, "updates": [
            {"text": "女生不适合学工科", "is_final": True, "captured_at": 90, "observed_at": 100}
        ]}
        self.assertIsNone(session.poll()["sbi"])
        self.assertEqual(len(session.assessment.transcripts), 0)

    def test_start_does_not_reuse_the_previous_session_report(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.latest_report = {"transcripts": [{"text": "previous session"}]}
        session.start()
        self.assertEqual(session.latest_report, {})

    def test_shutdown_completion_waits_for_native_worker_release(self):
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: 100)
        session.start()
        session.stop()
        self.assertFalse(session.shutdown_finished())
        session.camera.alive = False
        self.assertTrue(session.shutdown_finished())

    def test_reset_discards_unconfirmed_semantic_evidence(self):
        now = [100.0]
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda: now[0])
        session.start()
        session.audio.snapshot = lambda: {"audio": {}, "transcript": {}, "asr_resets": 0, "updates": [
            {"text": "女生不适合学工科", "is_final": False, "captured_at": now[0], "observed_at": now[0]}
        ]}
        self.assertGreater(session.poll()["sbi"], 0)
        now[0] += .1
        session.audio.snapshot = lambda: {"audio": {}, "transcript": {}, "asr_resets": 1, "updates": []}
        self.assertIsNone(session.poll()["sbi"])


if __name__ == "__main__": unittest.main()
