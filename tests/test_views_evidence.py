import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings


class ViewsEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Settings.from_env(Path.cwd(), environ={})))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_every_evidence_metric_declares_provenance(self):
        response = self.client.get("/api/evidence")
        self.assertEqual(response.status_code, 200)
        cards = response.json()["model_cards"]
        self.assertGreaterEqual(len(cards), 5)
        for card in cards:
            self.assertIn("limitations", card)
            for metric in card["metrics"]:
                self.assertIn(metric["provenance"], {"measured", "target", "simulated"})

    def test_all_views_and_dynamic_regions_exist(self):
        html = self.client.get("/").text
        for view in ["view-live", "view-explain", "view-intervention", "view-profile", "view-evidence"]:
            self.assertIn(f'id="{view}"', html)
        for region in ["contribution-ranking", "policy-comparison", "profile-trend", "model-card-grid"]:
            self.assertIn(f'id="{region}"', html)

    def test_frontend_exports_secondary_renderers(self):
        javascript = self.client.get("/assets/js/render.js").text
        for renderer in ["renderExplanation", "renderInterventionComparison", "renderProfile", "renderEvidence"]:
            self.assertIn(f"function {renderer}", javascript)

    def test_evidence_exposes_runtime_health_and_privacy_boundary(self):
        payload = self.client.get("/api/evidence").json()
        self.assertIn("runtime_adapters", payload)
        self.assertTrue(payload["privacy"]["local_only_default"])
        self.assertFalse(payload["privacy"]["raw_media_persisted"])


if __name__ == "__main__":
    unittest.main()
