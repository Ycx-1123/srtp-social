"""Bounded audio capture and one streaming ASR worker, independent of GUI/video."""
from collections import deque
from threading import Condition, Event, Lock, Thread
from time import monotonic
import numpy as np

from .asr import StreamingChineseASR


class BoundedAudioBuffer:
    def __init__(self, max_seconds=.8, sample_rate=16000):
        self.capacity = max(1, int(max_seconds * sample_rate))
        self.sample_rate = sample_rate
        self.queue = deque()
        self.count = 0
        self.dropped_samples = 0
        self.discontinuous = False
        self.closed = False
        self.condition = Condition()

    @property
    def backlog_ms(self):
        with self.condition:
            return self.count * 1000 / self.sample_rate

    def push(self, samples, at):
        samples = np.array(samples, dtype=np.float32, copy=True).reshape(-1)
        with self.condition:
            if self.closed:
                return
            if samples.size > self.capacity:
                removed = samples.size - self.capacity
                samples = samples[-self.capacity:]
                at += removed / self.sample_rate
                self.dropped_samples += removed
                self.discontinuous = True
            while self.queue and self.count + samples.size > self.capacity:
                dropped, _ = self.queue.popleft()
                self.count -= dropped.size
                self.dropped_samples += dropped.size
                self.discontinuous = True
            self.queue.append((samples, at))
            self.count += samples.size
            self.condition.notify()

    def pop(self, timeout=.1):
        with self.condition:
            if not self.queue and not self.closed:
                self.condition.wait(timeout)
            if self.closed or not self.queue:
                return None
            samples, at = self.queue.popleft()
            self.count -= samples.size
            reset = self.discontinuous
            self.discontinuous = False
            return samples, at, reset

    def close(self):
        with self.condition:
            self.closed = True
            self.queue.clear()
            self.count = 0
            self.condition.notify_all()

    def mark_discontinuity(self):
        with self.condition:
            # Hardware lost samples: discard all queued pre-gap blocks so the
            # reset flag belongs to the first post-gap block, not an older one.
            self.dropped_samples += self.count
            self.queue.clear()
            self.count = 0
            self.discontinuous = True


class AudioRecognitionWorker:
    def __init__(self, model_dir, device=None):
        self.model_dir, self.device = model_dir, device
        self.buffer = BoundedAudioBuffer()
        self.cancel = Event()
        self.lock = Lock()
        self.audio = {"status": "未启动", "rms": 0, "peak": 0}
        self.transcript = {"status": "未启动", "text": "", "is_final": False}
        self.updates = deque(maxlen=32)
        self.error = ""
        self.decode_ms = 0.0
        self.capture_age_ms = 0.0
        self.reset_count = 0

    def start(self):
        self.thread = Thread(target=self._run, daemon=True, name="streaming-chinese-asr")
        self.thread.start()

    def stop(self):
        self.cancel.set()
        self.buffer.close()

    def is_alive(self):
        return hasattr(self, "thread") and self.thread.is_alive()

    def _reset_transcript(self, status):
        # Publish one reset generation atomically with removal of old partials.
        with self.lock:
            self.reset_count += 1
            self.transcript = {"text": "", "is_final": False, "status": status}
            self.updates.clear()

    def _callback(self, indata, frames, time_info, status):
        if self.cancel.is_set():
            return
        samples = indata[:, 0]
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
        at = monotonic() - frames / 16000
        # Relative level changes, with conservative absolute gate. Loudness is
        # a supporting cue only; it does not itself establish unfriendliness.
        if len(self.levels) < 15:
            self.levels.append(rms)
        baseline = max(.015, float(np.median(self.levels))) if self.levels else .015
        pressure = max(0, min(1, (rms - max(.09, baseline * 2.5)) / .16))
        with self.lock:
            self.audio = {"rms": rms, "peak": peak, "pressure": pressure, "captured_at": at,
                          "status": "麦克风采集中" if not status else str(status)}
        if getattr(status, "input_overflow", False):
            self.buffer.mark_discontinuity()
        self.buffer.push(samples, at)

    def _run(self):
        recognizer = StreamingChineseASR(self.model_dir)
        self.levels = deque(maxlen=15)
        try:
            with self.lock:
                self.transcript["status"] = "正在加载本地流式模型"
            recognizer.load()
            if self.cancel.is_set():
                return
            import sounddevice as sd
            with sd.InputStream(device=self.device, samplerate=16000, channels=1, dtype="float32", blocksize=1600, latency="low", callback=self._callback):
                with self.lock:
                    self.transcript["status"] = "正在监听 · 本地流式"
                while not self.cancel.is_set():
                    item = self.buffer.pop()
                    if item is None:
                        continue
                    samples, captured_at, reset = item
                    age = monotonic() - captured_at
                    if reset or age > 1:
                        recognizer.reset()
                        self._reset_transcript("已丢弃过期音频，正在重新监听")
                        if age > 1:
                            continue
                    started = monotonic()
                    results = recognizer.feed(samples)
                    self.decode_ms = (monotonic() - started) * 1000
                    self.capture_age_ms = (monotonic() - captured_at) * 1000
                    if self.cancel.is_set():
                        break
                    if self.capture_age_ms > 1000:
                        recognizer.reset()
                        self.buffer.mark_discontinuity()
                        self._reset_transcript("识别超时，已丢弃旧结果并重新监听")
                        continue
                    with self.lock:
                        for result in results:
                            observed_at = monotonic()
                            update = {"text": result.text, "is_final": result.is_final,
                                      "latency_ms": self.capture_age_ms, "decode_ms": result.latency_ms,
                                      "observed_at": observed_at,
                                      "captured_at": captured_at,
                                      "status": "已确认" if result.is_final else "正在识别 · 临时文字"}
                            self.transcript = update
                            self.updates.append(update)
                        if not results and monotonic() - self.transcript.get("observed_at", 0) > 1.5:
                            self.transcript["status"] = "正在监听 · 本地流式"
        except Exception as exc:
            self.error = str(exc)
            with self.lock:
                self.audio["status"] = "麦克风/模型错误"
                self.transcript["status"] = "识别错误：" + str(exc)
        finally:
            recognizer.close()
            self.buffer.close()

    def snapshot(self):
        with self.lock:
            updates = list(self.updates)
            self.updates.clear()
            return {"audio": dict(self.audio), "transcript": dict(self.transcript), "updates": updates,
                    "audio_backlog_ms": self.buffer.backlog_ms, "asr_latency_ms": self.capture_age_ms,
                    "asr_decode_ms": self.decode_ms, "audio_dropped_samples": self.buffer.dropped_samples,
                    "asr_resets": self.reset_count, "error": self.error}
