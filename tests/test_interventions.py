import unittest

from soci_ai.domain import FusionSnapshot
from soci_ai.interventions import InterventionPolicy


def make_snapshot(
    sbi: float,
    evidence: str,
    dominant: list[str] | None = None,
) -> FusionSnapshot:
    return FusionSnapshot(
        at_ms=1000,
        sbi=sbi,
        energy=1.0,
        instantaneous_risk=0.6,
        risk_level="critical" if sbi >= 85 else "high" if sbi >= 70 else "warning",
        evidence_status=evidence,
        contributions={"semantic": 0.6, "acoustic": 0.2, "vision": 0.0, "pose": 0.0},
        reliabilities={"semantic": 0.9, "acoustic": 0.8, "vision": 0.0, "pose": 0.0},
        dominant_modalities=dominant or ["semantic", "acoustic"],
        explanation="语义与声学证据持续出现",
        provenance="simulated",
    )


class InterventionTest(unittest.TestCase):
    def test_high_sbi_with_insufficient_evidence_only_observes(self):
        result = InterventionPolicy().decide(make_snapshot(91, "insufficient"), 1000, "")
        self.assertEqual(result.action, "observe")
        self.assertEqual(result.reason, "insufficient_evidence")

    def test_cooldown_suppresses_duplicate_nudge(self):
        policy = InterventionPolicy(cooldown_ms=8000)
        first = policy.decide(make_snapshot(72, "sufficient"), 1000, "语速持续升高")
        second = policy.decide(make_snapshot(75, "sufficient"), 3000, "语速仍然偏快")
        self.assertEqual(first.action, "nudge")
        self.assertEqual(second.action, "observe")
        self.assertEqual(second.reason, "cooldown_active")

    def test_recovery_uses_positive_ambient_feedback(self):
        policy = InterventionPolicy()
        policy.decide(make_snapshot(82, "sufficient"), 1000, "冲突")
        result = policy.decide(make_snapshot(22, "sufficient"), 12000, "恢复")
        self.assertEqual(result.action, "ambient")
        self.assertEqual(result.tone, "recovery")

    def test_semantic_nudge_is_contextual_and_actionable(self):
        result = InterventionPolicy().decide(
            make_snapshot(66, "sufficient", ["semantic"]),
            1000,
            "你们技术同学总把事情想复杂",
        )
        self.assertEqual(result.action, "nudge")
        self.assertIn("事实", result.message)
        self.assertNotIn("冷静", result.message)

    def test_reset_clears_cooldown(self):
        policy = InterventionPolicy(cooldown_ms=8000)
        policy.decide(make_snapshot(66, "sufficient"), 1000, "冲突")
        policy.reset()
        result = policy.decide(make_snapshot(66, "sufficient"), 2000, "新的会话")
        self.assertEqual(result.action, "nudge")


if __name__ == "__main__":
    unittest.main()
