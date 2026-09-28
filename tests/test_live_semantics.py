from __future__ import annotations

import unittest

from soci_ai.live.semantics import MicroaggressionAnalyzer


class LiveSemanticsTest(unittest.TestCase):
    def test_semantic_analyzer_explains_gender_stereotype(self):
        event = MicroaggressionAnalyzer.default().analyze("女生不适合学工科", 3000)

        self.assertGreaterEqual(event.features["microbias"], 0.9)
        self.assertEqual(event.features["category"], "gender_stereotype")
        self.assertIn("女生不适合", event.evidence)
        self.assertEqual(event.source, "live_semantics")

    def test_unmatched_text_is_not_treated_as_negative(self):
        event = MicroaggressionAnalyzer.default().analyze("我们一起核对一下实验结果", 4000)

        self.assertEqual(event.features["microbias"], 0.0)
        self.assertEqual(event.features["category"], "none")
        self.assertEqual(event.evidence, "")

    def test_patterns_cover_five_explainable_categories(self):
        analyzer = MicroaggressionAnalyzer.default()
        self.assertTrue(
            {
                "gender_stereotype",
                "regional_stereotype",
                "ability_dismissal",
                "exclusion",
                "belittling",
            }.issubset(analyzer.categories)
        )


if __name__ == "__main__":
    unittest.main()
