import sys
import subprocess
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings
from tests.test_live_api import JPEG_BYTES, runtime_factory


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Settings.from_env(Path.cwd(), environ={})))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_all_scenarios_finish_and_generate_reports(self):
        scenario_ids = ["workplace_conflict", "interview_microbias", "classroom_silence"]
        listed = {item["id"] for item in self.client.get("/api/scenarios").json()["scenarios"]}
        self.assertEqual(set(scenario_ids), listed)
        for scenario_id in scenario_ids:
            self.assertEqual(self.client.post(f"/api/runtime/load/{scenario_id}").status_code, 200)
            final = self.client.post("/api/runtime/seek", json={"at_ms": 999999})
            self.assertEqual(final.status_code, 200)
            self.assertEqual(final.json()["status"], "completed")
            report = self.client.get("/api/runtime/report").json()
            self.assertEqual(report["scenario_id"], scenario_id)
            self.assertEqual(report["provenance"], "simulated")
            self.assertGreater(report["peak_sbi"], 0)

    def test_showcase_imports_no_heavy_ml_modules(self):
        imported = set(sys.modules)
        app = create_app(Settings.from_env(Path.cwd(), environ={}))
        loaded = set(sys.modules) - imported
        app.state.store.close()
        forbidden = {"torch", "transformers", "funasr", "ultralytics", "cv2"}
        self.assertTrue(forbidden.isdisjoint(loaded))

    def test_defense_delivery_files_exist(self):
        for relative in [
            "start_showcase.ps1",
            "start_showcase.bat",
            "README.md",
            "docs/architecture.md",
            "docs/defense-storyboard.md",
            "docs/metrics-policy.md",
            "docs/live-demo-checklist.md",
            "requirements-live.txt",
        ]:
            path = Path(relative)
            self.assertTrue(path.is_file(), relative)
            minimum_size = 20 if relative.endswith("requirements-live.txt") else 200
            self.assertGreater(path.stat().st_size, minimum_size, relative)

    def test_launchers_start_edge_mode_and_document_real_capabilities(self):
        powershell = Path("start_showcase.ps1").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn('$env:SOCI_MODE = "edge"', powershell)
        self.assertIn('$env:SOCI_PORT = "8001"', powershell)
        self.assertIn("print_capabilities", powershell)
        self.assertIn("单人自检", readme)
        self.assertIn("声纹注册与多人分离：扩展接口已设计，当前未实现", readme)
        self.assertIn("python -m pip install -r requirements-live.txt", readme)

    def test_windows_powershell_can_parse_the_one_click_launcher(self):
        command = (
            "$errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "(Resolve-Path '.\\start_showcase.ps1'), [ref]$null, [ref]$errors) > $null; "
            "if ($errors.Count) { $errors | ForEach-Object Message; exit 1 }"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_launcher_check_only_mode_completes_without_starting_the_server(self):
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path("start_showcase.ps1").resolve()),
                "-CheckOnly",
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('"sensevoice"', completed.stdout)
        self.assertNotIn("Starting local realtime self-check", completed.stdout)

    def test_full_live_api_smoke_with_injected_local_adapters(self):
        with TestClient(
            create_app(
                Settings.from_env(Path.cwd(), environ={}),
                live_runtime_factory=runtime_factory,
            )
        ) as client:
            session_id = client.post("/api/live/sessions").json()["session_id"]
            frame = client.post(
                f"/api/live/sessions/{session_id}/frame?at_ms=1000",
                content=JPEG_BYTES,
                headers={"content-type": "image/jpeg"},
            )
            self.assertEqual(frame.status_code, 200)
            self.assertEqual(client.get(f"/api/live/sessions/{session_id}/state").json()["provenance"], "measured")
            self.assertEqual(client.post(f"/api/live/sessions/{session_id}/stop").json()["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
