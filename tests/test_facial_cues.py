from __future__ import annotations

import unittest
from types import SimpleNamespace

from soci_ai.live.facial_cues import FacialCueTracker


def blendshapes(**overrides: float) -> list[SimpleNamespace]:
    scores = {
        "browDownLeft": 0.04,
        "browDownRight": 0.05,
        "mouthPressLeft": 0.03,
        "mouthPressRight": 0.04,
        "mouthPucker": 0.02,
        "mouthFrownLeft": 0.03,
        "mouthFrownRight": 0.02,
    }
    scores.update(overrides)
    return [
        SimpleNamespace(category_name=name, score=value)
        for name, value in scores.items()
    ]


class FacialCueTrackerTest(unittest.TestCase):
    def test_one_frame_frown_is_visible_but_does_not_become_sustained_risk(self):
        tracker = FacialCueTracker(calibration_frames=3, sustain_ms=800)
        for at_ms in (0, 250, 500):
            baseline = tracker.update(blendshapes(), at_ms)
        self.assertTrue(baseline["baseline_ready"])

        transient = tracker.update(
            blendshapes(browDownLeft=0.92, browDownRight=0.88),
            750,
        )
        relaxed = tracker.update(blendshapes(), 1000)

        self.assertGreater(transient["brow_tension"], 0.8)
        self.assertEqual(transient["micro_expression"], 0.0)
        self.assertEqual(relaxed["micro_expression"], 0.0)

    def test_sustained_brow_and_lip_cues_reach_risk_after_persistence_window(self):
        tracker = FacialCueTracker(calibration_frames=3, sustain_ms=800)
        for at_ms in (0, 250, 500):
            tracker.update(blendshapes(), at_ms)

        result = None
        for at_ms in (750, 1000, 1250, 1500, 1750):
            result = tracker.update(
                blendshapes(
                    browDownLeft=0.92,
                    browDownRight=0.88,
                    mouthPressLeft=0.84,
                    mouthPressRight=0.80,
                ),
                at_ms,
            )

        self.assertGreater(result["brow_tension"], 0.8)
        self.assertGreater(result["lip_tension"], 0.7)
        self.assertGreaterEqual(result["cue_duration_ms"], 800)
        self.assertGreater(result["facial_cue_confidence"], 0.5)
        self.assertGreater(result["micro_expression"], 0.8)

    def test_personal_baseline_suppresses_stable_asymmetry(self):
        tracker = FacialCueTracker(calibration_frames=3, sustain_ms=800)
        naturally_furrowed = blendshapes(browDownLeft=0.42, browDownRight=0.40)
        for at_ms in (0, 250, 500):
            tracker.update(naturally_furrowed, at_ms)

        result = None
        for at_ms in (750, 1000, 1250, 1500, 1750):
            result = tracker.update(naturally_furrowed, at_ms)

        self.assertLess(result["brow_tension"], 0.1)
        self.assertEqual(result["micro_expression"], 0.0)

    def test_sustained_lip_press_is_detected_without_brow_lowering(self):
        tracker = FacialCueTracker(calibration_frames=3, sustain_ms=800)
        for at_ms in (0, 250, 500):
            tracker.update(blendshapes(), at_ms)

        result = None
        for at_ms in (750, 1000, 1250, 1500, 1750):
            result = tracker.update(
                blendshapes(mouthPressLeft=0.88, mouthPressRight=0.84, mouthPucker=0.76),
                at_ms,
            )

        self.assertLess(result["brow_tension"], 0.1)
        self.assertGreater(result["lip_tension"], 0.8)
        self.assertGreater(result["micro_expression"], 0.75)

    def test_stream_gap_breaks_continuity_before_risk_activation(self):
        tracker = FacialCueTracker(calibration_frames=3, sustain_ms=800)
        for at_ms in (0, 250, 500):
            tracker.update(blendshapes(), at_ms)

        high = blendshapes(browDownLeft=0.94, browDownRight=0.92)
        tracker.update(high, 750)
        tracker.update(high, 1000)
        after_gap = tracker.update(high, 2500)

        self.assertEqual(after_gap["micro_expression"], 0.0)
        self.assertEqual(after_gap["cue_duration_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
