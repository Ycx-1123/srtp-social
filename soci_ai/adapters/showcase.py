from __future__ import annotations

from pathlib import Path

from ..domain import ScenarioDefinition
from ..scenarios import load_scenario


class ScenarioCatalog:
    DEFAULT_SCENARIO_ID = "workplace_conflict"

    def __init__(self, directory: Path):
        self.directory = directory

    def list(self) -> list[ScenarioDefinition]:
        scenarios = [load_scenario(path) for path in sorted(self.directory.glob("*.json"))]
        return sorted(
            scenarios,
            key=lambda item: (item.id != self.DEFAULT_SCENARIO_ID, item.title),
        )

    def get(self, scenario_id: str) -> ScenarioDefinition:
        matches = [scenario for scenario in self.list() if scenario.id == scenario_id]
        if not matches:
            raise KeyError(f"unknown scenario: {scenario_id}")
        return matches[0]
