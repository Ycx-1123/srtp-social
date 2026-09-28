from __future__ import annotations

import unittest
from types import SimpleNamespace

from soci_ai.desktop.capture import FacialCueTracker


def blendshapes(**overrides: float) -> dict[str, float]:
    scores = {
        "browDownLeft": 0.04, "browDownRight": 0.04,
        "mouthPressLeft": 0.03, "mouthPressRight": 0.03,
        "mouthPucker": 0.02,
        "mouthFrownLeft": 0.025, "mouthFrownRight": 0.025,
        "mouthSmileLeft": 0.02, "mouthSmileRight": 0.02,
    }
    return {**scores, **overrides}


class DesktopFacialTest(unittest.TestCase):
    def calibrated(self, scores=None):
        tracker = FacialCueTracker(calibration_frames=8, sustain_ms=250)
        for at_ms in range(0, 640, 80):
            result = tracker.update(scores or blendshapes(), at_ms)
        self.assertTrue(result["baseline_ready"])
        return tracker

    def test_small_brow_change_is_visible_immediately_and_requires_persistence(self):
        tracker = self.calibrated()
        subtle = blendshapes(browDownLeft=0.075, browDownRight=0.07)
        immediate = tracker.update(subtle, 640)
        self.assertGreater(immediate["brow_tension"], 0.20)
        self.assertLess(immediate["brow_tension"], 0.65)
        self.assertEqual(immediate["micro_expression"], 0)
        for at_ms in (720, 800, 880):
            before = tracker.update(subtle, at_ms)
        self.assertEqual(before["micro_expression"], 0)
        sustained = tracker.update(subtle, 960)
        self.assertGreater(sustained["micro_expression"], 0.18)
        self.assertEqual(sustained["cue_duration_ms"], 320)

    def test_lip_press_and_downturn_have_sensitive_separate_channels(self):
        tracker = self.calibrated()
        result = tracker.update(blendshapes(mouthPressLeft=0.09, mouthFrownRight=0.08), 640)
        self.assertGreater(result["lip_tension"], 0.3)
        self.assertGreater(result["mouth_downturn"], 0.3)
        self.assertEqual(result["brow_tension"], 0)

    def test_personal_noise_gate_suppresses_neutral_jitter(self):
        tracker = FacialCueTracker(calibration_frames=8)
        for index, value in enumerate((0.038, 0.046, 0.040, 0.049, 0.041, 0.048, 0.039, 0.047)):
            tracker.update(blendshapes(browDownLeft=value), index * 80)
        for index, value in enumerate((0.052, 0.047, 0.050, 0.043, 0.039, 0.053)):
            result = tracker.update(blendshapes(browDownLeft=value), 640 + index * 80)
            self.assertEqual(result["brow_tension"], 0)
            self.assertEqual(result["micro_expression"], 0)

    def test_individual_baselines_preserve_subtle_change_on_less_active_side(self):
        tracker = self.calibrated(blendshapes(browDownLeft=0.42, browDownRight=0.04))
        result = tracker.update(blendshapes(browDownLeft=0.42, browDownRight=0.085), 640)
        self.assertGreater(result["brow_tension"], 0.25)

    def test_one_calibration_outlier_does_not_hide_subtle_change(self):
        tracker = FacialCueTracker(calibration_frames=8)
        for index, value in enumerate((0.04, 0.04, 0.5, 0.04, 0.04, 0.04, 0.04, 0.04)):
            tracker.update(blendshapes(browDownLeft=value), index * 80)
        result = tracker.update(blendshapes(browDownLeft=0.075), 640)
        self.assertGreater(result["brow_tension"], 0.20)

    def test_smile_is_visible_and_never_counts_as_negative_evidence(self):
        tracker = self.calibrated()
        for at_ms in range(640, 1200, 80):
            result = tracker.update(blendshapes(mouthSmileLeft=0.22, mouthSmileRight=0.20), at_ms)
            self.assertGreater(result["smile"], 0.5)
            self.assertEqual(result["brow_tension"], 0)
            self.assertEqual(result["lip_tension"], 0)
            self.assertEqual(result["mouth_downturn"], 0)
            self.assertEqual(result["micro_expression"], 0)
            self.assertEqual(result["cue_duration_ms"], 0)

    def test_smile_does_not_bridge_separate_negative_cues(self):
        tracker = self.calibrated()
        frown = blendshapes(browDownLeft=0.15)
        tracker.update(frown, 640)
        tracker.update(blendshapes(mouthSmileLeft=0.8), 720)
        result = tracker.update(frown, 960)
        self.assertEqual(result["micro_expression"], 0)
        self.assertEqual(result["cue_duration_ms"], 0)

    def test_default_calibration_needs_eight_valid_neutral_observations(self):
        tracker = FacialCueTracker()
        for index in range(7):
            result = tracker.update(blendshapes(), index * 80)
            self.assertFalse(result["baseline_ready"])
            self.assertEqual(result["brow_tension"], 0)
        self.assertTrue(tracker.update(blendshapes(), 560)["baseline_ready"])

    def test_missing_scores_never_calibrate_or_become_zero_baseline(self):
        tracker = FacialCueTracker(calibration_frames=8)
        for at_ms in range(0, 800, 80):
            result = tracker.update({}, at_ms)
            self.assertFalse(result["baseline_ready"])
            self.assertFalse(result["facial_cue_available"])
        for at_ms in range(800, 1360, 80):
            self.assertFalse(tracker.update(blendshapes(), at_ms)["baseline_ready"])
        self.assertTrue(tracker.update(blendshapes(), 1360)["baseline_ready"])

    def test_face_loss_clears_sustain_and_partial_calibration(self):
        tracker = FacialCueTracker(calibration_frames=8)
        for at_ms in range(0, 400, 80):
            tracker.update(blendshapes(), at_ms)
        tracker.update({}, 400, face_detected=False)
        for at_ms in range(480, 1040, 80):
            self.assertFalse(tracker.update(blendshapes(), at_ms)["baseline_ready"])
        self.assertTrue(tracker.update(blendshapes(), 1040)["baseline_ready"])
        frown = blendshapes(browDownLeft=0.3)
        for at_ms in (1120, 1200, 1280, 1360, 1440):
            result = tracker.update(frown, at_ms)
        self.assertGreater(result["micro_expression"], 0)
        missing = tracker.update({}, 1520, face_detected=False)
        self.assertEqual(missing["micro_expression"], 0)
        self.assertEqual(missing["smile"], 0)
        resumed = tracker.update(frown, 1600)
        self.assertEqual(resumed["micro_expression"], 0)
        self.assertEqual(resumed["cue_duration_ms"], 0)

    def test_stream_gap_or_backwards_timestamp_cannot_count_as_persistence(self):
        for next_time in (2000, 600):
            with self.subTest(next_time=next_time):
                tracker = self.calibrated()
                frown = blendshapes(browDownLeft=0.3)
                tracker.update(frown, 640)
                tracker.update(frown, 720)
                result = tracker.update(frown, next_time)
                self.assertEqual(result["micro_expression"], 0)
                self.assertEqual(result["cue_duration_ms"], 0)

    def test_reset_starts_new_baseline_and_clears_all_channels(self):
        tracker = self.calibrated()
        tracker.update(blendshapes(browDownLeft=0.7, mouthSmileLeft=0.7), 640)
        tracker.reset()
        result = tracker.update(blendshapes(), 720)
        self.assertFalse(result["baseline_ready"])
        self.assertEqual(result["smile"], 0)
        self.assertEqual(result["micro_expression"], 0)

    def test_sensitivity_is_monotonic_and_bounded_without_adapting_expression_away(self):
        tracker = self.calibrated()
        values = []
        for index, score in enumerate((0.04, 0.055, 0.075, 0.11, 0.22, 0.6, 1.0)):
            result = tracker.update(blendshapes(browDownLeft=score), 640 + index * 80)
            values.append(result["brow_tension"])
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)
        self.assertEqual(sorted(values), values)
        self.assertLessEqual(values[-1], 1)
        for at_ms in range(1200, 5200, 80):
            result = tracker.update(blendshapes(browDownLeft=0.11), at_ms)
        self.assertGreater(result["brow_tension"], 0.4)
        self.assertGreater(result["micro_expression"], 0.3)

    def test_mediapipe_category_objects_and_landmarks_are_accepted(self):
        tracker = self.calibrated()
        categories = [SimpleNamespace(category_name=name, score=value)
                      for name, value in blendshapes(mouthSmileLeft=0.22).items()]
        landmarks = [SimpleNamespace(x=0.2, y=0.3, z=0.0), SimpleNamespace(x=0.7, y=0.8, z=0.0)]
        result = tracker.update(categories, 640, face_detected=True, landmarks=landmarks)
        self.assertGreater(result["smile"], 0.5)
        self.assertEqual(result["cue_source"], "mediapipe_blendshapes")
        self.assertTrue(result["facial_cue_available"])

    def test_invalid_model_values_and_landmarks_never_create_expression(self):
        tracker = self.calibrated()
        result = tracker.update({"browDownLeft": float("nan"), "mouthSmileLeft": float("inf")}, 640)
        self.assertFalse(result["facial_cue_available"])
        self.assertEqual(result["micro_expression"], 0)
        self.assertEqual(result["smile"], 0)
        result = tracker.update(blendshapes(browDownLeft=1.0), 720,
                                landmarks=[SimpleNamespace(x=float("nan"), y=0.5)])
        self.assertFalse(result["facial_cue_available"])
        self.assertEqual(result["brow_tension"], 0)


if __name__ == "__main__":
    unittest.main()
