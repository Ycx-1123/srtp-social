import json
import tempfile
import unittest
from pathlib import Path

from soci_ai.domain import FusionSnapshot, InterventionEvent, SessionFrame
from soci_ai.reporting import ReportBuilder
from soci_ai.store import SessionStore
from soci_ai.live.domain import LiveState


def sample_state(at_ms: int, sbi: float, action: str = "nudge") -> SessionFrame:
    return SessionFrame(
        at_ms=at_ms,
        phase="escalation",
        speaker="产品负责人",
        dialogue="你们技术同学总把事情想复杂",
        snapshot=FusionSnapshot(
            at_ms=at_ms,
            sbi=sbi,
            energy=0.8,
            instantaneous_risk=0.7,
            risk_level="warning" if sbi < 70 else "high",
            evidence_status="sufficient",
            contributions={"semantic": 0.4, "acoustic": 0.2, "vision": 0.1, "pose": 0.0},
            reliabilities={"semantic": 0.9, "acoustic": 0.8, "vision": 0.7, "pose": 0.0},
            dominant_modalities=["semantic", "acoustic"],
            explanation="语义与声学线索相互印证",
            provenance="simulated",
        ),
        intervention=InterventionEvent(
            at_ms=at_ms,
            action=action,
            message="将概括换成可核对的事实",
            reason="cross_modal_risk",
            intensity=0.6,
            tone="gentle",
            provenance="simulated",
        ),
        modality_features={"semantic": {"offense": 0.7}},
    )


class StoreReportingTest(unittest.TestCase):
    def test_live_state_persistence_contains_no_raw_media_fields(self):
        store = SessionStore.in_memory()
        session_id = store.start_session("live_self_check")
        state = LiveState.neutral(session_id)
        store.append_live_state(session_id, state)
        payload = store.get_session(session_id)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIn('"provenance": "measured"', serialized)
        self.assertNotIn("raw_frame", serialized)
        self.assertNotIn("raw_audio", serialized)
        store.close()

    def test_session_round_trip_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SessionStore(Path(folder) / "sessions.db")
            session_id = store.start_session("workplace_conflict")
            store.append_state(session_id, sample_state(at_ms=1000, sbi=62))
            store.append_state(session_id, sample_state(at_ms=2000, sbi=82, action="pause"))
            summary = ReportBuilder(store).build(session_id)
            store.close()
        self.assertEqual(summary.peak_sbi, 82)
        self.assertEqual(summary.mean_sbi, 72)
        self.assertEqual(summary.intervention_counts["nudge"], 1)
        self.assertEqual(summary.intervention_counts["pause"], 1)
        self.assertEqual(summary.provenance, "simulated")

    def test_report_anonymizes_phone_and_email(self):
        store = SessionStore.in_memory()
        session_id = store.start_session("demo")
        store.append_text(session_id, 500, "用户", "联系我 13812345678 或 test@example.com")
        text = ReportBuilder(store).build(session_id).anonymized_transcript
        store.close()
        self.assertNotIn("13812345678", text)
        self.assertNotIn("test@example.com", text)
        self.assertIn("[PHONE]", text)
        self.assertIn("[EMAIL]", text)

    def test_unwritable_database_falls_back_to_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            blocked = Path(folder) / "file"
            blocked.write_text("not a directory", encoding="utf-8")
            store = SessionStore(blocked / "sessions.db")
        self.assertEqual(store.persistence_mode, "memory")
        self.assertEqual(store.warning, "database_unwritable")
        store.close()

    def test_unknown_session_is_rejected(self):
        store = SessionStore.in_memory()
        with self.assertRaisesRegex(KeyError, "unknown session"):
            store.append_state("missing", sample_state(at_ms=1000, sbi=50))
        store.close()


if __name__ == "__main__":
    unittest.main()
