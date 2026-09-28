"""Small, genuinely streaming Mandarin ASR with an optional native runtime.

Call from one audio worker, never from the microphone callback or GUI thread.
The caller owns its bounded capture queue and should reset after dropped audio.
No microphone, network, Torch, or batch transcription work happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL_NAME = "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)


@dataclass(frozen=True)
class RecognitionUpdate:
    text: str
    is_final: bool
    latency_ms: float


class StreamingChineseASR:
    """An online Zipformer2 CTC recognizer with no audio backlog.

    Construction is cheap; call load() on a worker before microphone capture.
    feed() accepts at most one second of normalized mono floating point audio,
    normally 100 ms. latency_ms measures processing within the current feed,
    excluding capture/queue age and initial model loading. Partial hypotheses
    can change; consumers should replace their current partial rather than append.
    This object is confined to a single worker thread.
    """

    max_audio_seconds = 1.0

    def __init__(self, model_dir: Path, num_threads: int = 2):
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self._recognizer = None
        self._stream = None
        self._last_text = ""
        self._closed = False

    def load(self) -> StreamingChineseASR:
        """Load the local model once; report absent files/dependencies clearly."""
        if self._closed:
            raise RuntimeError("StreamingChineseASR is closed")
        if self._recognizer is not None:
            return self
        model = self.model_dir / "model.int8.onnx"
        tokens = self.model_dir / "tokens.txt"
        missing = [str(path) for path in (model, tokens) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing streaming ASR files: {', '.join(missing)}. "
                f"Extract {MODEL_URL} and select its {MODEL_NAME} directory."
            )
        try:
            sherpa = import_module("sherpa_onnx")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Streaming recognition needs the native sherpa-onnx package. "
                "Install requirements-desktop.txt with the application's Python interpreter."
            ) from exc
        self._recognizer = sherpa.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(tokens),
            model=str(model),
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.2,
            rule2_min_trailing_silence=0.6,
            rule3_min_utterance_length=15.0,
        )
        self._stream = self._recognizer.create_stream()
        return self

    def feed(self, samples: np.ndarray, sample_rate: int = 16000) -> list[RecognitionUpdate]:
        if self._closed:
            raise RuntimeError("StreamingChineseASR is closed")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        audio = np.asarray(samples)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("samples must be mono floating point audio normalized to [-1, 1]")
        if not np.isfinite(audio).all() or (audio.size and np.max(np.abs(audio)) > 1.0):
            raise ValueError("samples must contain finite normalized audio in [-1, 1]")
        if audio.size > sample_rate * self.max_audio_seconds:
            raise BufferError("ASR accepts at most one second per feed; bound the capture queue and reset after drops")
        if not audio.size:
            return []
        self.load()
        started = perf_counter()
        recognizer = self._recognizer
        stream = self._stream
        updates = []
        # Process small pieces so a long accepted block cannot hide intermediate
        # hypotheses or carry audio from two utterances into one endpoint reset.
        step = max(1, int(sample_rate // 10))
        for offset in range(0, audio.size, step):
            stream.accept_waveform(int(sample_rate), np.ascontiguousarray(audio[offset:offset + step], dtype=np.float32))
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                text = recognizer.get_result(stream).strip()
                endpoint = recognizer.is_endpoint(stream)
                if text and (text != self._last_text or endpoint):
                    updates.append(RecognitionUpdate(text, endpoint, (perf_counter() - started) * 1000))
                self._last_text = text
                if endpoint:
                    recognizer.reset(stream)
                    self._last_text = ""
        return updates

    def reset(self) -> None:
        """Discard the current utterance and pending audio after a discontinuity."""
        if self._closed:
            raise RuntimeError("StreamingChineseASR is closed")
        if self._recognizer is not None:
            self._stream = self._recognizer.create_stream()
        self._last_text = ""

    def close(self) -> None:
        """Release native recognition state; safe to call more than once."""
        self._stream = None
        self._recognizer = None
        self._last_text = ""
        self._closed = True
