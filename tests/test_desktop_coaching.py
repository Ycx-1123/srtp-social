import unittest

from soci_ai.desktop.core import RealtimeAssessment


def face(at, **features):
    return {"captured_at": at, "face_detected": True,
            "features": {"baseline_ready": True, **features}}


class CoachingTests(unittest.TestCase):
    def test_smile_gets_positive_guidance_not_negative_sbi(self):
        model = RealtimeAssessment()
        state = model.update(0, vision=face(0, smile=.8))
        self.assertEqual(state["sbi"], 0)
        self.assertIn("微笑", state.get("suggestion", ""))
        self.assertEqual(state["tree"].get("smile"), .8)

    def test_brow_guidance_names_observed_action(self):
        state = RealtimeAssessment().update(0, vision=face(0, brow_tension=.7, micro_expression=.7))
        self.assertIn("眉", state.get("suggestion_title", ""))

    def test_semantic_warning_takes_priority_over_smile(self):
        model = RealtimeAssessment()
        model.observe_text("女生不适合学工科", 0, True)
        state = model.update(0, vision=face(0, smile=.9))
        self.assertIn("语言", state.get("suggestion_title", ""))
        self.assertGreater(state["sbi"], 30)

    def test_review_has_evidence_based_language_rewrite(self):
        model = RealtimeAssessment()
        model.observe_text("女生不适合学工科", 1, True)
        model.update(1)
        report = model.report(2, {}, {})
        cards = report.get("advice", [])
        self.assertTrue(any(x["kind"] == "language" and "女生不适合学工科" in x["body"] for x in cards))
        self.assertTrue(any(x["kind"] == "semantic" for x in report.get("moments", [])))

    def test_missing_input_does_not_produce_positive_review(self):
        model = RealtimeAssessment()
        model.update(0)
        report = model.report(1, {}, {})
        self.assertIsNone(report.get("peak_sbi"))
        self.assertEqual(report.get("moments"), [])
        self.assertTrue(any(x["kind"] == "availability" for x in report.get("advice", [])))

    def test_coverage_ignores_missing_and_uncalibrated_face(self):
        model = RealtimeAssessment()
        model.update(0)
        model.update(.1, vision=face(.1, brow_tension=.6, micro_expression=.6))
        model.update(.2, vision=face(.2, smile=.8))
        model.update(3)
        report = model.report(3, {}, {})
        summary = report.get("observation_summary", {})
        self.assertAlmostEqual(summary.get("face_seconds", -1), .2)
        self.assertAlmostEqual(summary.get("tension_seconds", -1), .1)
        self.assertAlmostEqual(summary.get("smile_seconds", -1), .1)
        self.assertEqual(summary.get("peaks", {}).get("brow_tension"), .6)

    def test_context_moments_are_bounded_not_a_per_frame_table(self):
        model = RealtimeAssessment()
        for i in range(300):
            at = i / 10
            model.update(at, vision=face(at, brow_tension=.8, micro_expression=.8))
        report = model.report(30, {}, {})
        self.assertGreater(len(report.get("moments", [])), 0)
        self.assertLessEqual(len(report["moments"]), 6)


if __name__ == "__main__":
    unittest.main()
