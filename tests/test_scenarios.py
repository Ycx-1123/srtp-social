import tempfile
import unittest
from pathlib import Path

from soci_ai.domain import ScenarioDefinition
from soci_ai.scenarios import ScenarioCursor, load_scenario


class ScenarioTest(unittest.TestCase):
    def test_cursor_emits_each_event_once(self):
        scenario = ScenarioDefinition.model_validate(
            {
                "id": "demo",
                "title": "Demo",
                "duration_ms": 2000,
                "seed": 7,
                "phases": [],
                "events": [
                    {
                        "at_ms": 200,
                        "modality": "semantic",
                        "features": {"offense": 0.2},
                        "confidence": 0.8,
                    },
                    {
                        "at_ms": 1000,
                        "modality": "vision",
                        "features": {"negative": 0.6},
                        "confidence": 0.9,
                    },
                ],
            }
        )
        cursor = ScenarioCursor(scenario)
        self.assertEqual([event.at_ms for event in cursor.advance(500)], [200])
        self.assertEqual([event.at_ms for event in cursor.advance(1200)], [1000])
        self.assertEqual(cursor.advance(1200), [])

    def test_decreasing_timestamp_reports_event_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(
                '{"id":"bad","title":"Bad","duration_ms":1000,"seed":1,"phases":[],"events":['
                '{"at_ms":800,"modality":"vision","features":{},"confidence":1},'
                '{"at_ms":200,"modality":"semantic","features":{},"confidence":1}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "event 1.*timestamp"):
                load_scenario(path)

    def test_cursor_seek_and_reset_are_deterministic(self):
        scenario = ScenarioDefinition.model_validate(
            {
                "id": "demo",
                "title": "Demo",
                "duration_ms": 2000,
                "seed": 7,
                "phases": [],
                "events": [
                    {"at_ms": 100, "modality": "context", "features": {}, "confidence": 1},
                    {"at_ms": 900, "modality": "context", "features": {}, "confidence": 1},
                ],
            }
        )
        cursor = ScenarioCursor(scenario)
        cursor.advance(1000)
        cursor.seek(500)
        self.assertEqual(cursor.elapsed_ms, 500)
        self.assertEqual([event.at_ms for event in cursor.advance(1000)], [900])
        cursor.reset()
        self.assertEqual(cursor.elapsed_ms, 0)
        self.assertEqual([event.at_ms for event in cursor.advance(100)], [100])

    def test_workplace_scenario_covers_all_modalities_and_phases(self):
        scenario = load_scenario(Path("scenarios/workplace_conflict.json"))
        self.assertEqual(scenario.duration_ms, 45000)
        self.assertEqual({event.modality for event in scenario.events}, {"semantic", "acoustic", "vision", "pose", "context"})
        self.assertEqual([phase.id for phase in scenario.phases], ["baseline", "escalation", "intervention", "recovery"])


if __name__ == "__main__":
    unittest.main()
