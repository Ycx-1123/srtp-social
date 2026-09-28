import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from soci_ai.api import create_app
from soci_ai.config import Settings
from soci_ai.dataset_audit import DatasetAuditError, audit_dataset


class DatasetAuditTest(unittest.TestCase):
    def test_detects_cross_split_content_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = root / "train" / "happy"
            val = root / "val" / "happy"
            train.mkdir(parents=True)
            val.mkdir(parents=True)
            (train / "a.jpg").write_bytes(b"same-image")
            (val / "b.jpg").write_bytes(b"same-image")
            report = audit_dataset(root)
        self.assertEqual(report.cross_split_exact_duplicates, 1)
        self.assertFalse(report.independent_evaluation_valid)
        self.assertEqual(report.split_counts, {"train": 1, "val": 1})

    def test_counts_empty_files_and_relative_path_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                target = root / split / "neutral"
                target.mkdir(parents=True)
                (target / "same.jpg").write_bytes(b"" if split == "train" else b"different")
            report = audit_dataset(root)
        self.assertEqual(report.empty_file_count, 1)
        self.assertEqual(report.same_relative_path_collisions, 1)
        self.assertEqual(report.class_counts["train"]["neutral"], 1)

    def test_duplicate_within_one_split_does_not_invalidate_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "train" / "happy"
            target.mkdir(parents=True)
            (target / "a.jpg").write_bytes(b"same")
            (target / "b.jpg").write_bytes(b"same")
            report = audit_dataset(root)
        self.assertEqual(report.cross_split_exact_duplicates, 0)
        self.assertTrue(report.independent_evaluation_valid)

    def test_distinguishes_leaked_file_count_from_hash_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                target = root / split / "happy"
                target.mkdir(parents=True)
                (target / "a.jpg").write_bytes(b"same")
                (target / "b.jpg").write_bytes(b"same")
            report = audit_dataset(root)
        self.assertEqual(report.duplicate_hash_groups, 1)
        self.assertEqual(report.cross_split_exact_duplicates, 2)

    def test_missing_dataset_root_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing"
            with self.assertRaisesRegex(DatasetAuditError, "dataset root does not exist"):
                audit_dataset(missing)

    def test_generated_audit_is_exposed_to_evidence_view(self):
        with TestClient(create_app(Settings.from_env(Path.cwd(), environ={}))) as client:
            response = client.get("/api/evidence/dataset")
            html = client.get("/").text
            javascript = client.get("/assets/js/render.js").text
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_files"], 1774)
        self.assertIn('id="dataset-audit-card"', html)
        self.assertIn("function renderDatasetAudit", javascript)


if __name__ == "__main__":
    unittest.main()
