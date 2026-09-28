from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings


class TreeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(Settings.from_env(Path.cwd(), environ={})))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_tree_is_procedural_and_state_driven(self):
        response = self.client.get("/assets/js/tree.js")
        self.assertEqual(response.status_code, 200)
        source = response.text
        for token in [
            "requestAnimationFrame",
            "setTarget",
            "drawBranch",
            "drawLeaf",
            "drawBloom",
            "fallingLeaves",
            "prefers-reduced-motion",
            "devicePixelRatio",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("new Image(", source)

    def test_tree_has_performance_budgets_and_all_visual_modes(self):
        source = self.client.get("/assets/js/tree.js").text
        for budget in ["1200", "700", "360"]:
            self.assertIn(budget, source)
        for mode in ["observing", "friendly", "signal", "risk", "recovering"]:
            self.assertIn(mode, source)
        for feature in ["drawRoots", "updateFallingLeaves", "drawParticles", "frameTimes"]:
            self.assertIn(feature, source)


if __name__ == "__main__":
    unittest.main()
