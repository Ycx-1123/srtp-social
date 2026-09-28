import unittest

from soci_ai.desktop.core import RealtimeAssessment, classify_text


class DesktopAssessmentTests(unittest.TestCase):
    def test_no_evidence_does_not_invent_a_friendly_score(self):
        model = RealtimeAssessment()
        self.assertIsNone(model.update(0)["sbi"])

    def test_facial_change_reacts_in_under_half_a_second(self):
        model = RealtimeAssessment()
        neutral = {"face_detected": True, "captured_at": 0, "features": {"baseline_ready": True, "micro_expression": 0}}
        self.assertEqual(model.update(0, vision=neutral)["sbi"], 0)
        tense = {"face_detected": True, "captured_at": .1, "features": {"baseline_ready": True, "micro_expression": 1, "brow_tension": 1}}
        first = model.update(.1, vision=tense)
        tense["captured_at"] = .4
        later = model.update(.4, vision=tense)
        self.assertGreater(first["sbi"], 20)
        self.assertGreater(later["sbi"], 45)
        self.assertLess(later["friendliness"], 55)

    def test_semantic_partial_and_final_are_real_evidence(self):
        model = RealtimeAssessment()
        model.observe_text("女生不适合学工科", .1, False)
        first = model.update(.1)
        self.assertGreater(first["metrics"]["semantic_bias"], .5)
        model.observe_text("女生不适合学工科", .2, True)
        later = model.update(.3)
        self.assertGreater(later["sbi"], first["sbi"])
        self.assertEqual(len(model.transcripts), 1)

    def test_stale_signals_remove_scores_and_do_not_keep_a_green_tree(self):
        model = RealtimeAssessment()
        model.update(0, vision={"face_detected": True, "captured_at": 0, "features": {"baseline_ready": True}})
        state = model.update(3)
        self.assertIsNone(state["sbi"])
        self.assertEqual(state["tree"]["mode"], "observing")

    def test_no_face_means_no_visual_tension(self):
        model = RealtimeAssessment()
        state = model.update(1, vision={"face_detected": False, "captured_at": 1, "features": {"micro_expression": 1}})
        self.assertEqual(state["metrics"]["visual_tension"], 0)
        self.assertIsNone(state["sbi"])

    def test_constant_input_is_not_randomly_jittered(self):
        model = RealtimeAssessment()
        outputs = []
        for i in range(10):
            outputs.append(model.update(i / 10, vision={"face_detected": True, "captured_at": i / 10, "features": {"baseline_ready": True}})["sbi"])
        self.assertEqual(set(outputs), {0.0})

    def test_risk_is_not_the_same_as_detecting_speech(self):
        model = RealtimeAssessment()
        state = model.update(1, audio={"captured_at": 1, "rms": .05, "peak": .2, "pressure": 0})
        self.assertEqual(state["metrics"]["tone_pressure"], 0)

    def test_rules_do_not_flag_obvious_rejection_of_stereotype(self):
        self.assertGreater(classify_text("我觉得女生不适合学工科")[0], .9)
        self.assertEqual(classify_text("不要说女生不适合学工科")[0], 0)

    def test_report_peak_includes_short_unsampled_changes(self):
        model = RealtimeAssessment()
        model.update(0)
        model.observe_text("女生不适合学工科", .1, True)
        live = model.update(.1)
        report = model.report(.2, {}, {})
        self.assertEqual(report["peak_sbi"], live["sbi"])


if __name__ == "__main__":
    unittest.main()
