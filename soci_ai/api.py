from __future__ import annotations

import uuid
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .adapters import ScenarioCatalog, discover_optional_adapters
from .config import Settings
from .fusion import FusionEngine
from .interventions import InterventionPolicy
from .reporting import ReportBuilder
from .runtime import ShowcaseRuntime
from .scenarios import ScenarioCursor
from .store import SessionStore
from .live.api import LiveRuntimeFactory, LiveRuntimeRegistry, create_live_router


class SpeedRequest(BaseModel):
    value: float


class SeekRequest(BaseModel):
    at_ms: int


def create_app(
    settings: Settings,
    *,
    live_runtime_factory: LiveRuntimeFactory | None = None,
) -> FastAPI:
    scenario_dir = settings.root / "scenarios"
    if not scenario_dir.exists():
        scenario_dir = Path(__file__).resolve().parents[1] / "scenarios"
    catalog = ScenarioCatalog(scenario_dir)
    scenarios = catalog.list()
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {scenario_dir}")
    store = SessionStore(settings.root / "data" / "sessions.db")
    runtime = ShowcaseRuntime(
        ScenarioCursor(scenarios[0]),
        FusionEngine(),
        InterventionPolicy(),
        store,
        mode=settings.mode,
    )
    live_registry = LiveRuntimeRegistry(settings, store, live_runtime_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await live_registry.shutdown()
        await runtime.shutdown()
        store.close()

    app = FastAPI(title="SOCI-AI Studio", version=settings.version, lifespan=lifespan)
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.runtime = runtime
    app.state.store = store
    app.state.live_registry = live_registry
    app.include_router(create_live_router(live_registry, settings))

    web_dir = settings.root / "web"
    if not web_dir.exists():
        web_dir = Path(__file__).resolve().parents[1] / "web"
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    evidence_path = settings.root / "evidence" / "model_cards.json"
    if not evidence_path.exists():
        evidence_path = Path(__file__).resolve().parents[1] / "evidence" / "model_cards.json"
    dataset_audit_path = settings.root / "evidence" / "dataset_audit.json"
    if not dataset_audit_path.exists():
        dataset_audit_path = Path(__file__).resolve().parents[1] / "evidence" / "dataset_audit.json"

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "mode": settings.mode,
            "network": "enabled" if settings.allow_network else "disabled",
            "heavy_models": "enabled" if settings.load_heavy_models else "disabled",
            "version": settings.version,
        }

    @app.get("/api/evidence")
    async def evidence() -> dict[str, object]:
        model_cards = json.loads(evidence_path.read_text(encoding="utf-8"))
        return {
            "model_cards": model_cards,
            "runtime_adapters": [
                item.to_dict()
                for item in discover_optional_adapters(active=settings.load_heavy_models)
            ],
            "privacy": {
                "local_only_default": not settings.allow_network,
                "raw_media_persisted": False,
                "stored_data": "去标识化文本、结构化特征、融合分数与干预日志",
                "boundary": "原始音视频仅在设备内存中处理，展示模式完全使用合成场景事件。",
            },
            "provenance_legend": {
                "measured": "由当前数据或脚本直接计算",
                "target": "工程目标，尚未形成独立验证结论",
                "simulated": "确定性场景引擎生成，用于机制演示",
            },
        }

    @app.get("/api/evidence/dataset")
    async def dataset_evidence() -> dict[str, object]:
        if not dataset_audit_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "dataset_audit_missing",
                    "message": "Run scripts/audit_dataset.py before requesting dataset evidence.",
                    "field": "dataset_audit",
                },
            )
        return json.loads(dataset_audit_path.read_text(encoding="utf-8"))

    @app.get("/api/scenarios")
    async def list_scenarios() -> dict[str, list[dict[str, object]]]:
        return {
            "scenarios": [
                {
                    "id": scenario.id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "duration_ms": scenario.duration_ms,
                    "phases": [phase.model_dump(mode="json") for phase in scenario.phases],
                }
                for scenario in catalog.list()
            ]
        }

    @app.get("/api/adapters")
    async def adapters() -> dict[str, list[dict[str, str | bool]]]:
        return {
            "adapters": [
                item.to_dict()
                for item in discover_optional_adapters(active=settings.load_heavy_models)
            ]
        }

    @app.get("/api/runtime/state")
    async def runtime_state() -> dict[str, object]:
        return runtime.snapshot()

    @app.post("/api/runtime/load/{scenario_id}")
    async def load_runtime(scenario_id: str) -> dict[str, object]:
        try:
            return await runtime.load(ScenarioCursor(catalog.get(scenario_id)))
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_scenario", "message": str(exc), "field": "scenario_id"},
            ) from exc

    @app.post("/api/runtime/start")
    async def start_runtime() -> dict[str, object]:
        return await runtime.start()

    @app.post("/api/runtime/pause")
    async def pause_runtime() -> dict[str, object]:
        return await runtime.pause()

    @app.post("/api/runtime/reset")
    async def reset_runtime() -> dict[str, object]:
        return await runtime.reset()

    @app.post("/api/runtime/seek")
    async def seek_runtime(request: SeekRequest) -> dict[str, object]:
        return await runtime.seek(request.at_ms)

    @app.post("/api/runtime/speed")
    async def speed_runtime(request: SpeedRequest) -> dict[str, object]:
        try:
            return await runtime.set_speed(request.value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_speed", "message": str(exc), "field": "value"},
            ) from exc

    @app.get("/api/report/{session_id}")
    async def report(session_id: str) -> dict[str, object]:
        try:
            return ReportBuilder(store).build(session_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "unknown_session", "message": str(exc), "field": "session_id"},
            ) from exc

    @app.get("/api/runtime/report")
    async def runtime_report() -> dict[str, object]:
        return ReportBuilder(store).build(runtime.session_id).model_dump(mode="json")

    @app.websocket("/ws/state")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        client_id = uuid.uuid4().hex
        queue = runtime.attach_client(client_id)
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            runtime.detach_client(client_id)

    return app
