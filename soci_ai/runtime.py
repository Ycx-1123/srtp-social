from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from collections.abc import Iterable

from .domain import (
    FusionSnapshot,
    InterventionEvent,
    PerceptionEvent,
    RuntimeSnapshot,
    SessionFrame,
)
from .fusion import FusionEngine
from .interventions import InterventionPolicy
from .scenarios import ScenarioCursor
from .store import SessionStore


class ShowcaseRuntime:
    def __init__(
        self,
        cursor: ScenarioCursor,
        fusion: FusionEngine,
        policy: InterventionPolicy,
        store: SessionStore,
        *,
        mode: str = "showcase",
    ):
        self.cursor = cursor
        self.fusion = fusion
        self.policy = policy
        self.store = store
        self.mode = mode
        self.speed = 1.0
        self.status = "ready"
        self.phase = cursor.scenario.phases[0].id if cursor.scenario.phases else ""
        self.speaker = ""
        self.dialogue = ""
        self.modality_features: dict[str, dict[str, float | str | bool]] = {
            name: {} for name in ("semantic", "acoustic", "vision", "pose", "context")
        }
        self.session_id = self.store.start_session(cursor.scenario.id)
        self._fusion_snapshot = self._empty_fusion()
        self._intervention = self._empty_intervention()
        self._running = asyncio.Event()
        self._lock = asyncio.Lock()
        self._shutdown = False
        self.playback_task: asyncio.Task[None] | None = None
        self._subscribers: dict[str, asyncio.Queue[dict[str, object]]] = {}

    @property
    def active_playback_tasks(self) -> int:
        return int(self.playback_task is not None and not self.playback_task.done())

    async def start(self) -> dict[str, object]:
        async with self._lock:
            if self.playback_task is None or self.playback_task.done():
                self.playback_task = asyncio.create_task(self._playback_loop(), name="soci-showcase-playback")
            self.status = "running"
            self._running.set()
            await self._publish()
        return self.snapshot()

    async def pause(self) -> dict[str, object]:
        async with self._lock:
            self.status = "paused"
            self._running.clear()
            await self._publish()
        return self.snapshot()

    async def reset(self) -> dict[str, object]:
        async with self._lock:
            self._running.clear()
            self.status = "ready"
            self.cursor.reset()
            self.fusion.reset()
            self.policy.reset()
            self.phase = self.cursor.scenario.phases[0].id if self.cursor.scenario.phases else ""
            self.speaker = ""
            self.dialogue = ""
            self.modality_features = {name: {} for name in self.modality_features}
            self._fusion_snapshot = self._empty_fusion()
            self._intervention = self._empty_intervention()
            self.session_id = self.store.start_session(self.cursor.scenario.id)
            await self._publish()
        return self.snapshot()

    async def load(self, cursor: ScenarioCursor) -> dict[str, object]:
        async with self._lock:
            self._running.clear()
            self.cursor = cursor
            self.status = "ready"
            self.fusion.reset()
            self.policy.reset()
            self.phase = cursor.scenario.phases[0].id if cursor.scenario.phases else ""
            self.speaker = ""
            self.dialogue = ""
            self.modality_features = {name: {} for name in self.modality_features}
            self._fusion_snapshot = self._empty_fusion()
            self._intervention = self._empty_intervention()
            self.session_id = self.store.start_session(cursor.scenario.id)
            await self._publish()
        return self.snapshot()

    async def set_speed(self, value: float) -> dict[str, object]:
        if value not in {0.5, 1.0, 1.5, 2.0, 4.0}:
            raise ValueError("speed must be one of 0.5, 1, 1.5, 2, or 4")
        self.speed = float(value)
        await self._publish()
        return self.snapshot()

    async def seek(self, elapsed_ms: int) -> dict[str, object]:
        async with self._lock:
            target = max(0, min(int(elapsed_ms), self.cursor.scenario.duration_ms))
            self.cursor.reset()
            self.fusion.reset()
            self.policy.reset()
            self.modality_features = {name: {} for name in self.modality_features}
            self.phase = self.cursor.scenario.phases[0].id if self.cursor.scenario.phases else ""
            self.speaker = ""
            self.dialogue = ""
            events = self.cursor.advance(target)
            await self._process_events(events, persist=True)
            if self._fusion_snapshot.at_ms < target:
                self._fusion_snapshot = self.fusion.update([], target)
                self._intervention = self.policy.decide(self._fusion_snapshot, target, self.dialogue)
            self.status = "completed" if target >= self.cursor.scenario.duration_ms else "paused"
            await self._publish()
        return self.snapshot()

    def attach_client(self, client_id: str) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=1)
        queue.put_nowait(self.snapshot())
        self._subscribers[client_id] = queue
        return queue

    def detach_client(self, client_id: str) -> None:
        self._subscribers.pop(client_id, None)

    def snapshot(self) -> dict[str, object]:
        return RuntimeSnapshot(
            mode=self.mode,
            scenario_id=self.cursor.scenario.id,
            scenario_title=self.cursor.scenario.title,
            session_id=self.session_id,
            status=self.status,
            elapsed_ms=self.cursor.elapsed_ms,
            duration_ms=self.cursor.scenario.duration_ms,
            speed=self.speed,
            phase=self.phase,
            speaker=self.speaker,
            dialogue=self.dialogue,
            fusion=self._fusion_snapshot,
            intervention=self._intervention,
            modality_features=self.modality_features,
            persistence={
                "mode": self.store.persistence_mode,
                "warning": self.store.warning,
            },
        ).model_dump(mode="json")

    async def shutdown(self) -> None:
        self._shutdown = True
        self._running.set()
        if self.playback_task is not None:
            self.playback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.playback_task
        self.playback_task = None

    async def _playback_loop(self) -> None:
        last_tick = time.monotonic()
        while not self._shutdown:
            await self._running.wait()
            await asyncio.sleep(0.1)
            now = time.monotonic()
            delta_ms = int((now - last_tick) * 1000 * self.speed)
            last_tick = now
            if not self._running.is_set():
                continue
            target = min(self.cursor.scenario.duration_ms, self.cursor.elapsed_ms + max(delta_ms, 1))
            events = self.cursor.advance(target)
            await self._process_events(events, persist=True)
            if not events:
                self._fusion_snapshot = self.fusion.update([], target)
                self._intervention = self.policy.decide(self._fusion_snapshot, target, self.dialogue)
            if target >= self.cursor.scenario.duration_ms:
                self.status = "completed"
                self._running.clear()
                self.store.finish_session(self.session_id)
            await self._publish()

    async def _process_events(self, events: Iterable[PerceptionEvent], *, persist: bool) -> None:
        groups: dict[int, list[PerceptionEvent]] = defaultdict(list)
        for event in events:
            groups[event.at_ms].append(event)
        for at_ms in sorted(groups):
            group = groups[at_ms]
            for event in group:
                self.modality_features[event.modality] = dict(event.features)
                if event.modality == "context":
                    self.phase = str(event.features.get("phase", self.phase))
                    self.speaker = str(event.features.get("speaker", self.speaker))
                    self.dialogue = str(event.features.get("dialogue", self.dialogue))
            self._fusion_snapshot = self.fusion.update(group, at_ms)
            self._intervention = self.policy.decide(self._fusion_snapshot, at_ms, self.dialogue)
            if persist:
                self.store.append_state(
                    self.session_id,
                    SessionFrame(
                        at_ms=at_ms,
                        phase=self.phase,
                        speaker=self.speaker,
                        dialogue=self.dialogue if any(item.modality == "context" for item in group) else "",
                        snapshot=self._fusion_snapshot,
                        intervention=self._intervention,
                        modality_features=self.modality_features,
                    ),
                )

    async def _publish(self) -> None:
        snapshot = self.snapshot()
        for queue in list(self._subscribers.values()):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(snapshot)

    @staticmethod
    def _empty_fusion() -> FusionSnapshot:
        return FusionSnapshot(
            at_ms=0,
            sbi=2.08,
            energy=0.0,
            instantaneous_risk=0.0,
            risk_level="safe",
            evidence_status="insufficient",
            contributions={name: 0.0 for name in ("semantic", "acoustic", "vision", "pose")},
            reliabilities={name: 0.0 for name in ("semantic", "acoustic", "vision", "pose")},
            dominant_modalities=[],
            explanation="等待多模态证据进入时间窗。",
            provenance="simulated",
        )

    @staticmethod
    def _empty_intervention() -> InterventionEvent:
        return InterventionEvent(
            at_ms=0,
            action="observe",
            message="系统已就绪，保持静默观察。",
            reason="ready",
            intensity=0.0,
            tone="neutral",
            provenance="simulated",
        )

