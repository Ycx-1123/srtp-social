import importlib
import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "resources/language/social_bias_examples.json"
DATABASE = CORPUS.with_suffix(".sqlite")


class DesktopLanguageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("soci_ai.desktop.language"),
                             "The desktop language database detector is not implemented")
        self.language = importlib.import_module("soci_ai.desktop.language")

    def test_user_example_returns_evidence_and_concrete_rewrite(self):
        result = self.language.analyze_text("女生就是学不好工科吧")
        self.assertGreaterEqual(result["score"], .8)
        self.assertEqual(result["category"], "gender_ability")
        self.assertIn("性别", result["explanation"])
        self.assertTrue(result["rewrite"])
        self.assertTrue(result["example_id"])
        self.assertEqual(result["matched_text"], "女生就是学不好工科吧")
        self.assertEqual(self.language.classify_text("女生就是学不好工科吧"),
                         (result["score"], result["explanation"]))

    def test_bounded_wording_variants_and_legacy_example(self):
        for text in ("女生啊就是学不好工科吧", "我觉得女生不适合学工科",
                     "女孩子嘛天生就不擅长编程", " 女生 就是 学不好 工科 吧！ "):
            with self.subTest(text=text):
                self.assertGreaterEqual(self.language.classify_text(text)[0], .8)
        self.assertGreater(self.language.classify_text("女生不适合学工科")[0], .9)

    def test_rejection_quotation_and_specific_task_reasons_do_not_flag(self):
        controls = (
            "不要说女生学不好工科", "我反对“女生就是学不好工科吧”这种说法",
            "有人说“女生学不好工科”，我不同意", "女生不是学不好工科",
            "女生不适合今天发言，因为嗓子疼", "你不用参加今天会议，可以休息",
            "我们在讨论‘女生就是学不好工科吧’这句话为什么有偏见",
            "文章引用了‘女生就是学不好工科吧’作为反面例子",
            "有人说女生不适合学工科，这种说法不对", "我不认为女生不适合学工科",
            "不是所有女生都学不好工科", "并不是女生不适合学工科",
            "女生并非天生不擅长编程", "大家都可以报名，有问题一起解决。",
        )
        for text in controls:
            with self.subTest(text=text):
                self.assertEqual(self.language.classify_text(text)[0], 0)

    def test_rejected_phrase_does_not_hide_a_separate_biased_assertion(self):
        result = self.language.analyze_text("不要说女生学不好工科。年纪大了就别学编程了。")
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["category"], "age_bias")

    def test_rejection_variants_and_unquoted_educational_reference_do_not_flag(self):
        for text in (
            "不是女生不适合学工科，而是这次教学资源不足",
            "我并不觉得女生不适合学工科", "谁说女生不适合学工科",
            "不要觉得女生不适合学工科", "女生不适合学工科是偏见",
            "我们讨论女生就是学不好工科吧这句话为什么有偏见",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.language.classify_text(text)[0], 0)

    def test_explicit_endorsement_of_reported_bias_still_flags(self):
        for text in ("有人说“女生不适合学工科”，我觉得很对",
                     "有人说“女生就是学不好工科吧”，我也这么认为",
                     "不要说女生学不好工科，但女生当技术负责人容易镇不住场"):
            with self.subTest(text=text):
                self.assertGreater(self.language.classify_text(text)[0], 0)

    def test_newline_separates_rejection_from_independent_assertion(self):
        result = self.language.analyze_text("不要说女生学不好工科\n年纪大了就别学编程了")
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["category"], "age_bias")

    def test_negated_rejection_is_not_treated_as_rejection(self):
        result = self.language.analyze_text("我们不反对女生学不好工科这个观点。")
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["category"], "gender_ability")

    def test_stronger_wording_variant_wins_over_weaker_exact_example(self):
        result = self.language.analyze_text("别这么敏感，女孩子嘛天生就不擅长编程")
        self.assertGreaterEqual(result["score"], .9)
        self.assertEqual(result["category"], "gender_ability")
        self.assertIn("编程", result["matched_text"])

    def test_corpus_has_unique_authored_samples_and_explicit_provenance(self):
        data = json.loads(CORPUS.read_text(encoding="utf-8"))
        rows = data["examples"]
        bias = [row for row in rows if row["label"] == "bias"]
        controls = [row for row in rows if row["label"] == "control"]
        self.assertEqual(len(bias), 300)
        self.assertEqual(len(controls), 60)
        self.assertEqual(len({row["text"] for row in rows}), 360)
        self.assertEqual(len({row["id"] for row in rows}), 360)
        self.assertEqual(len({row["category"] for row in bias}), 12)
        for row in rows:
            self.assertEqual(row["provenance"], "synthetic_authored")
            self.assertEqual(row["source_role"], "original_sample")
            self.assertTrue(row["context"])
            self.assertTrue(row["explanation"])
            self.assertTrue(row["rewrite"])
        self.assertFalse(data["metadata"]["empirically_validated"])
        self.assertFalse(data["metadata"]["public_posts_imported"])

    def test_every_authored_positive_and_control_has_expected_runtime_behavior(self):
        rows = json.loads(CORPUS.read_text(encoding="utf-8"))["examples"]
        for row in rows:
            with self.subTest(example=row["id"], text=row["text"]):
                result = self.language.analyze_text(row["text"])
                if row["label"] == "bias":
                    self.assertGreater(result["score"], 0)
                    self.assertEqual(result["category"], row["category"])
                    self.assertEqual(result["example_id"], row["id"])
                else:
                    self.assertEqual(result["score"], 0)

    def test_runtime_reads_bundled_database_and_reports_counts(self):
        status = self.language.corpus_status()
        self.assertEqual(status["source"], "sqlite")
        self.assertEqual(status["bias_examples"], 300)
        self.assertEqual(status["control_examples"], 60)
        self.assertEqual(status["categories"], 12)
        self.assertEqual(status["provenance"], "synthetic_authored")
        with closing(sqlite3.connect(DATABASE)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0], 360)
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_database_builder_reproduces_queryable_json_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "examples.sqlite"
            command = [sys.executable, str(ROOT / "tools/build_language_database.py"),
                       "--source", str(CORPUS), "--output", str(destination)]
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with closing(sqlite3.connect(destination)) as connection:
                record = connection.execute(
                    "SELECT text, label, provenance FROM examples WHERE id = 'GA001'").fetchone()
                self.assertEqual(record, ("女生就是学不好工科吧", "bias", "synthetic_authored"))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM examples WHERE label = 'control'").fetchone()[0], 60)

    def test_warmed_detector_is_suitable_for_live_text_updates(self):
        self.language.analyze_text("女生就是学不好工科吧")
        start = time.perf_counter()
        for _ in range(500):
            self.language.analyze_text("我们先把项目需求说清楚，再请大家提出各自的意见。")
        self.assertLess(time.perf_counter() - start, 1.5)

    def test_empty_and_long_input_remain_bounded_and_explained(self):
        empty = self.language.analyze_text("")
        self.assertEqual(empty["score"], 0)
        self.assertTrue(empty["context_note"])
        start = time.perf_counter()
        self.language.analyze_text("今天讨论项目。" * 20000)
        self.assertLess(time.perf_counter() - start, 1.5)


if __name__ == "__main__":
    unittest.main()
