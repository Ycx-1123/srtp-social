import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings
from soci_ai.fusion import FusionEngine
from soci_ai.interventions import InterventionPolicy
from soci_ai.runtime import ShowcaseRuntime
from soci_ai.scenarios import ScenarioCursor, load_scenario
from soci_ai.store import SessionStore


class RuntimeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        scenario = load_scenario(Path("scenarios/workplace_conflict.json"))
        self.runtime = ShowcaseRuntime(
            ScenarioCursor(scenario),
            FusionEngine(),
            InterventionPolicy(),
            SessionStore.in_memory(),
        )

    async def asyncTearDown(self):
        await self.runtime.shutdown()
        self.runtime.store.close()

    async def test_reset_after_disconnect_keeps_single_task(self):
        await self.runtime.start()
        first_task = self.runtime.playback_task
        self.runtime.detach_client("client-1")
        await self.runtime.reset()
        self.assertEqual(self.runtime.cursor.elapsed_ms, 0)
        self.assertIs(self.runtime.playback_task, first_task)
        self.assertEqual(self.runtime.active_playback_tasks, 1)

    async def test_subscriber_queue_keeps_latest_snapshot(self):
        queue = self.runtime.attach_client("client-1")
        await self.runtime.seek(15000)
        await self.runtime.seek(25000)
        self.assertEqual(queue.qsize(), 1)
        latest = queue.get_nowait()
        self.assertEqual(latest["elapsed_ms"], 25000)


class RuntimeApiTest(unittest.TestCase):
    def setUp(self):
        settings = Settings.from_env(Path.cwd(), environ={})
        self.client = TestClient(create_app(settings))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_runtime_controls_and_snapshot(self):
        self.assertEqual(self.client.post("/api/runtime/load/workplace_conflict").status_code, 200)
        self.assertEqual(
            self.client.post("/api/runtime/speed", json={"value": 2.0}).json()["speed"],
            2.0,
        )
        body = self.client.get("/api/runtime/state").json()
        self.assertEqual(body["scenario_id"], "workplace_conflict")
        self.assertEqual(body["mode"], "showcase")

    def test_scenarios_and_websocket_expose_initial_state(self):
        scenarios = self.client.get("/api/scenarios").json()["scenarios"]
        self.assertEqual(scenarios[0]["id"], "workplace_conflict")
        with self.client.websocket_connect("/ws/state") as socket:
            initial = socket.receive_json()
        self.assertEqual(initial["scenario_id"], "workplace_conflict")
        self.assertIn("fusion", initial)

    def test_invalid_speed_returns_structured_error(self):
        response = self.client.post("/api/runtime/speed", json={"value": 9.0})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_speed")

    def test_showcase_app_does_not_import_heavy_packages(self):
        before = set(sys.modules)
        app = create_app(Settings.from_env(Path.cwd(), environ={}))
        loaded = set(sys.modules) - before
        app.state.store.close()
        self.assertTrue({"torch", "transformers", "funasr", "ultralytics", "cv2"}.isdisjoint(loaded))


if __name__ == "__main__":
    unittest.main()
