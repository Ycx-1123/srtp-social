import math
import unittest

from soci_ai.domain import PerceptionEvent
from soci_ai.fusion import FusionEngine


def event(modality: str, value: float, confidence: float, at_ms: int = 1000) -> PerceptionEvent:
    key = {
        "semantic": "offense",
        "acoustic": "arousal",
        "vision": "negative",
        "pose": "closed_posture",
    }[modality]
    return PerceptionEvent(
        at_ms=at_ms,
        modality=modality,
        features={key: value},
        confidence=confidence,
    )


class FusionTest(unittest.TestCase):
    def test_provenance_defaults_to_simulated_and_accepts_measured(self):
        simulated = FusionEngine().update([event("vision", 0.6, 0.9)], 1000)
        measured = FusionEngine().update(
            [event("vision", 0.6, 0.9)],
            1000,
            provenance="measured",
        )
        self.assertEqual(simulated.provenance, "simulated")
        self.assertEqual(measured.provenance, "measured")

    def test_asynchronous_modalities_fuse_within_freshness_window(self):
        engine = FusionEngine()
        engine.update(
            [event("semantic", 0.8, 0.95, 1000)],
            1000,
        )
        snapshot = engine.update(
            [event("acoustic", 0.72, 0.92, 2000)],
            2000,
        )
        self.assertEqual(snapshot.evidence_status, "sufficient")
        self.assertGreater(snapshot.contributions["semantic"], 0)
        self.assertGreater(snapshot.contributions["acoustic"], 0)
        self.assertIn("跨模态", snapshot.explanation)

    def test_energy_decays_without_new_stimulus(self):
        engine = FusionEngine()
        first = engine.update([event("semantic", 0.9, 1.0)], now_ms=1000)
        second = engine.update([], now_ms=3000)
        self.assertLess(second.energy, first.energy)
        self.assertLess(second.sbi, first.sbi)

    def test_cross_modal_synergy_exceeds_single_modality(self):
        one = FusionEngine().update([event("semantic", 0.7, 1.0)], 1000)
        many = FusionEngine().update(
            [
                event("semantic", 0.7, 1.0),
                event("acoustic", 0.7, 1.0),
                event("vision", 0.7, 1.0),
            ],
            1000,
        )
        self.assertGreater(many.instantaneous_risk, one.instantaneous_risk)

    def test_missing_low_confidence_modalities_are_finite_and_insufficient(self):
        snapshot = FusionEngine().update([event("vision", 1.0, 0.05)], 1000)
        self.assertTrue(math.isfinite(snapshot.sbi))
        self.assertEqual(snapshot.evidence_status, "insufficient")
        self.assertIn(snapshot.risk_level, {"safe", "observe"})

    def test_zero_delta_and_long_gap_stay_bounded(self):
        engine = FusionEngine()
        first = engine.update([event("semantic", 1.0, 1.0)], 1000)
        same_time = engine.update([event("acoustic", 1.0, 1.0)], 1000)
        after_gap = engine.update([], 10_000_000)
        for snapshot in (first, same_time, after_gap):
            self.assertGreaterEqual(snapshot.sbi, 0)
            self.assertLessEqual(snapshot.sbi, 100)
            self.assertTrue(math.isfinite(snapshot.energy))

    def test_absent_features_contribute_zero_without_crashing(self):
        empty = PerceptionEvent(at_ms=1000, modality="pose", features={}, confidence=0.9)
        snapshot = FusionEngine().update([empty], 1000)
        self.assertEqual(snapshot.contributions["pose"], 0.0)
        self.assertEqual(snapshot.evidence_status, "insufficient")

    def test_sustained_facial_action_cue_contributes_to_visual_risk(self):
        facial_cue = PerceptionEvent(
            at_ms=1000,
            modality="vision",
            features={"negative": 0.05, "micro_expression": 0.8},
            confidence=0.9,
        )

        snapshot = FusionEngine().update([facial_cue], 1000)

        self.assertAlmostEqual(snapshot.contributions["vision"], 0.144)
        self.assertGreater(snapshot.instantaneous_risk, 0.1)

    def test_reset_clears_accumulated_energy_and_context(self):
        engine = FusionEngine()
        engine.update([event("semantic", 1.0, 1.0)], 1000)
        engine.reset()
        snapshot = engine.update([], 0)
        self.assertAlmostEqual(snapshot.energy, 0.0)
        self.assertEqual(snapshot.dominant_modalities, [])


if __name__ == "__main__":
    unittest.main()
