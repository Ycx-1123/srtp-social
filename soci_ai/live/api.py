from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..config import Settings
from ..store import SessionStore
from .audio import AudioFeaturePayload, SenseVoiceTranscriber
from .capabilities import detect_live_capabilities
from .runtime import LiveSessionRuntime
from .semantics import MicroaggressionAnalyzer
from .vision import FrameValidationError, VisionModelUnavailable, YoloVisionAdapter


LiveRuntimeFactory = Callable[[str], LiveSessionRuntime]


class LiveRuntimeRegistry:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        runtime_factory: LiveRuntimeFactory | None = None,
    ):
        self.settings = settings
        self.store = store
        self.runtime_factory = runtime_factory or self._default_runtime
        self.runtimes: dict[str, LiveSessionRuntime] = {}

    def _default_runtime(self, session_id: str) -> LiveSessionRuntime:
        weights = self.settings.yolo_weights or Path("missing-yolo-weights.pt")
        model_dir = self.settings.sensevoice_model_dir or Path("missing-sensevoice-model")
        return LiveSessionRuntime(
            vision=YoloVisionAdapter(
                weights,
                ultralytics_config_dir=self.settings.root / "data" / "ultralytics",
                background_classification=True,
                facial_model_path=self.settings.face_landmarker_model,
            ),
            transcriber=SenseVoiceTranscriber(model_dir),
            analyzer=MicroaggressionAnalyzer.default(),
            session_id=session_id,
        )

    async def create(self) -> LiveSessionRuntime:
        session_id = self.store.start_session("live_self_check")
        runtime = self.runtime_factory(session_id)
        self.runtimes[session_id] = runtime
        await runtime.start()
        self.store.append_live_state(session_id, runtime.state())
        return runtime

    def get(self, session_id: str) -> LiveSessionRuntime:
        try:
            return self.runtimes[session_id]
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "unknown_live_session",
                    "message": "Realtime session was not found.",
                    "field": "session_id",
                },
            ) from exc

    def persist(self, runtime: LiveSessionRuntime) -> None:
        self.store.append_live_state(runtime.session_id, runtime.state())

    async def stop(self, session_id: str) -> LiveSessionRuntime:
        runtime = self.get(session_id)
        was_stopped = runtime.state().status == "stopped"
        await runtime.stop()
        if not was_stopped:
            self.persist(runtime)
            self.store.finish_session(session_id)
        return runtime

    async def shutdown(self) -> None:
        for session_id, runtime in tuple(self.runtimes.items()):
            if runtime.state().status != "stopped":
                await runtime.stop()
                self.persist(runtime)
                self.store.finish_session(session_id)


def create_live_router(
    registry: LiveRuntimeRegistry,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix="/api/live", tags=["live-self-check"])

    @router.get("/capabilities")
    async def capabilities() -> dict[str, object]:
        return detect_live_capabilities(settings).to_dict()

    @router.post("/sessions")
    async def start_session() -> dict[str, object]:
        return (await registry.create()).snapshot()

    @router.get("/sessions/{session_id}/state")
    async def state(session_id: str) -> dict[str, object]:
        return registry.get(session_id).snapshot()

    @router.post("/sessions/{session_id}/frame")
    async def submit_frame(
        session_id: str,
        request: Request,
        at_ms: int,
    ) -> dict[str, object]:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type not in {"image/jpeg", "image/webp"}:
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "unsupported_frame_type",
                    "message": "Camera frames must be JPEG or WebP.",
                    "field": "content-type",
                },
            )
        runtime = registry.get(session_id)
        try:
            result = await runtime.submit_frame(await request.body(), at_ms)
        except FrameValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": str(exc), "message": "Camera frame is invalid.", "field": "frame"},
            ) from exc
        except VisionModelUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "vision_unavailable", "message": str(exc), "field": "vision"},
            ) from exc
        await asyncio.to_thread(registry.persist, runtime)
        return result.model_dump(mode="json")

    @router.post("/sessions/{session_id}/audio-features")
    async def submit_audio_features(
        session_id: str,
        payload: AudioFeaturePayload,
    ) -> dict[str, object]:
        runtime = registry.get(session_id)
        result = await runtime.submit_audio_features(payload)
        await asyncio.to_thread(registry.persist, runtime)
        return result.model_dump(mode="json")

    @router.post("/sessions/{session_id}/audio-chunk")
    async def submit_audio_chunk(
        session_id: str,
        request: Request,
        sample_rate: int,
        at_ms: int,
    ) -> dict[str, object]:
        if sample_rate != 16000:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_sample_rate", "message": "PCM rate must be 16000 Hz.", "field": "sample_rate"},
            )
        runtime = registry.get(session_id)
        result = await runtime.submit_audio_chunk(await request.body(), sample_rate, at_ms)
        await asyncio.to_thread(registry.persist, runtime)
        return result.model_dump(mode="json")

    @router.post("/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> dict[str, object]:
        return (await registry.stop(session_id)).snapshot()

    return router
