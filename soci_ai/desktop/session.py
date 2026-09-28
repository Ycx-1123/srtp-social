"""Device lifecycle and newest-evidence polling without any Qt dependency."""
from time import monotonic
from .audio import AudioRecognitionWorker
from .capture import CameraVisionWorker
from .core import RealtimeAssessment
from .resources import model_root
from .asr import MODEL_NAME


class DesktopSession:
    def __init__(self, *, camera_factory=CameraVisionWorker, audio_factory=AudioRecognitionWorker, clock=monotonic):
        self.camera_factory, self.audio_factory, self.clock = camera_factory, audio_factory, clock
        self.camera = self.audio = None
        self.running = False
        self.started_at = clock()
        self.elapsed = 0
        self.assessment = RealtimeAssessment()
        self.last_seq = 0
        self.last_fps_time = clock()
        self.camera_fps = 0
        self.telemetry = {}
        self.message = ""
        self.last_asr_resets = 0
        self.latest_state = {"status": "ready", "sbi": None, "friendliness": None}
        self.latest_report = {}

    def start(self, camera_index=0, microphone=None):
        if self.running:
            return False
        if ((self.camera and self.camera.status().get("workers_alive")) or
                (self.audio and self.audio.is_alive())):
            self.message = "设备仍在释放，请稍后再开始；不会重复打开。"
            return False
        self.started_at = self.clock()
        self.elapsed = 0
        self.assessment = RealtimeAssessment()
        self.last_seq = 0
        self.last_fps_time = self.started_at
        self.latest_report = {}
        self.telemetry = {}
        self.camera_fps = 0
        self.last_asr_resets = 0
        self.message = "开始后保持自然表情约 1 秒，程序会自动记住你放松时的表情，再比较后续动作。"
        self.camera = self.camera_factory(model_root() / "face_landmarker.task", camera_index=int(camera_index or 0))
        self.audio = self.audio_factory(model_root() / MODEL_NAME, device=microphone)
        self.running = True
        self.camera.start()
        self.audio.start()
        return True

    def stop(self):
        if self.running:
            self.elapsed = max(0, self.clock() - self.started_at)
        self.running = False
        if self.camera:
            self.camera.stop()
        if self.audio:
            self.audio.stop()
        focus = self.assessment.focus(self.elapsed, force=True)
        if self.latest_report:
            self.latest_report["elapsed_seconds"] = round(self.elapsed, 1)
            self.latest_report['session_focus'] = focus
        self.latest_state = {"status": "stopped", "sbi": None, "friendliness": None,
                             "elapsed_ms": int(self.elapsed * 1000),
                             'session_focus': focus,
                             "tree": {"mode": "observing", "health": .55, "risk": 0, "bloom": 0, "wind": 0},
                             "suggestion": "会话已结束，摄像头与麦克风正在释放；回顾保留真实记录。"}
        return self.latest_state

    def calibrate(self):
        if self.running:
            self.camera.calibrate()
            self.message = "正在重新记录自然表情，请放松眉嘴并保持约 1 秒。"

    def shutdown_finished(self):
        return not ((self.camera and self.camera.status().get("workers_alive")) or
                    (self.audio and self.audio.is_alive()))

    def frame(self):
        return self.camera.get_frame() if self.running and self.camera else None

    def poll(self):
        if not self.running:
            return self.latest_state
        now = self.clock()
        self.elapsed = max(0, now - self.started_at)
        vision = self.camera.get_vision()
        captured = vision.get("captured_at")
        vision["captured_at"] = captured - self.started_at if captured is not None else -100
        audio_state = self.audio.snapshot()
        resets = audio_state.get("asr_resets", 0)
        if resets != self.last_asr_resets:
            self.assessment.invalidate_partial()
            self.last_asr_resets = resets
        audio = audio_state.get("audio", {})
        if "captured_at" in audio:
            audio["captured_at"] -= self.started_at
        for update in audio_state.get("updates", []):
            captured_text_at = update.get("captured_at", update["observed_at"])
            if 0 <= now - captured_text_at <= 1:
                self.assessment.observe_text(update["text"], max(0, captured_text_at - self.started_at), update["is_final"])
        camera_status = self.camera.status()
        frame_seq = camera_status.get("frame_seq", 0)
        if now - self.last_fps_time >= 1:
            self.camera_fps = (frame_seq - self.last_seq) / (now - self.last_fps_time)
            self.last_seq, self.last_fps_time = frame_seq, now
        self.telemetry = {key: audio_state.get(key) for key in ("audio_backlog_ms", "asr_latency_ms", "asr_decode_ms", "audio_dropped_samples", "asr_resets")}
        self.telemetry.update(camera_fps=self.camera_fps, vision_latency_ms=vision.get("latency_ms"),
                              vision_frame_age_ms=None if captured is None else round((now - captured) * 1000, 1),
                              asr_status=audio_state.get("transcript", {}).get("status", ""))
        state = self.assessment.update(self.elapsed, vision=vision, audio=audio)
        issues = [item for item in (camera_status.get("error"), audio_state.get("error")) if item]
        state.update(status="running", transcript=audio_state.get("transcript", {}), telemetry=self.telemetry)
        if issues:
            state["suggestion_title"] = "检查设备与输入"
            state["suggestion"] = "；".join(issues)
        elif not vision.get("features", {}).get("baseline_ready") and not state.get('language_alert') and state["metrics"]["semantic_bias"] < .35:
            state["suggestion_title"] = "正在记住自然表情"
            state["suggestion"] = self.message
        self.latest_state = state
        self.latest_report = self.assessment.report(self.elapsed, self.telemetry, {
            "face_landmarker": camera_status.get("status"), "asr": self.telemetry["asr_status"],
            "assessment": "原型规则；尚未进行微冒犯准确率验证", "device_errors": "；".join(issues)})
        return state
