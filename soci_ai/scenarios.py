from __future__ import annotations

import bisect
import json
from pathlib import Path

from pydantic import ValidationError

from .domain import PerceptionEvent, ScenarioDefinition


def load_scenario(path: Path) -> ScenarioDefinition:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"scenario file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid scenario JSON at line {exc.lineno}: {exc.msg}") from exc

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("scenario events must be a list")
    previous = -1
    for index, raw_event in enumerate(events):
        try:
            event = PerceptionEvent.model_validate(raw_event)
        except ValidationError as exc:
            raise ValueError(f"event {index} failed validation: {exc}") from exc
        if event.at_ms < previous:
            raise ValueError(f"event {index} timestamp decreases from {previous} to {event.at_ms}")
        previous = event.at_ms
    try:
        return ScenarioDefinition.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"scenario validation failed: {exc}") from exc


class ScenarioCursor:
    def __init__(self, scenario: ScenarioDefinition):
        self.scenario = scenario
        self.index = 0
        self.elapsed_ms = 0
        self._timestamps = [event.at_ms for event in scenario.events]

    def advance(self, elapsed_ms: int) -> list[PerceptionEvent]:
        bounded = max(self.elapsed_ms, min(int(elapsed_ms), self.scenario.duration_ms))
        self.elapsed_ms = bounded
        emitted: list[PerceptionEvent] = []
        while self.index < len(self.scenario.events):
            event = self.scenario.events[self.index]
            if event.at_ms > bounded:
                break
            emitted.append(event)
            self.index += 1
        return emitted

    def seek(self, elapsed_ms: int) -> None:
        self.elapsed_ms = max(0, min(int(elapsed_ms), self.scenario.duration_ms))
        self.index = bisect.bisect_right(self._timestamps, self.elapsed_ms)

    def reset(self) -> None:
        self.index = 0
        self.elapsed_ms = 0

