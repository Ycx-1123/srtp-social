import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings


class ConfigApiTest(unittest.TestCase):
    def test_default_settings_are_offline_showcase(self):
        settings = Settings.from_env(Path("."), environ={})
        self.assertEqual(settings.mode, "showcase")
        self.assertFalse(settings.allow_network)
        self.assertFalse(settings.load_heavy_models)

    def test_health_exposes_safe_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings.from_env(Path(folder), environ={})
            with TestClient(create_app(settings)) as client:
                body = client.get("/api/health").json()
        self.assertEqual(
            body,
            {
                "status": "ok",
                "mode": "showcase",
                "network": "disabled",
                "heavy_models": "disabled",
                "version": "0.1.0",
            },
        )


if __name__ == "__main__":
    unittest.main()
