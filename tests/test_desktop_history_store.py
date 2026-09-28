import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from soci_ai.desktop.history_store import HistoryStore


def report(value=70):
    return {'elapsed_seconds': 30, 'average_sbi': value, 'peak_sbi': value,
            'history': [{'at_ms': 1000, 'sbi': value, 'cues': ['疑似学历偏见']}],
            'transcripts': [{'at_ms': 1000, 'text': '确认字幕'}],
            'advice': [{'title': '语言', 'body': '用任务要求代替身份判断'}],
            'provenance': 'test_inputs', 'limitations': '测试报告'}


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'history.sqlite3'
        self.store = HistoryStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def create(self, key='a', value=70):
        self.store.create_session(key, '2026-09-27T09:00:00+08:00', report(value), 'test')

    def test_reopen_delete_only_selected_no_late_resurrection(self):
        for key, value in [('a', 70), ('b', 30)]:
            self.create(key, value)
            self.assertTrue(self.store.finish(key, report(value), '2026-09-27T09:01:00+08:00'))
        self.store.close()
        self.store = HistoryStore(self.path)
        self.assertTrue(self.store.delete_session('a'))
        self.assertFalse(self.store.checkpoint('a', report()))
        self.assertFalse(self.store.finish('a', report(), '2026-09-27T09:02:00+08:00'))
        self.assertIsNone(self.store.load_session('a'))
        self.assertEqual(self.store.load_session('b')['report']['average_sbi'], 30)

    def test_final_is_immutable_and_running_cannot_delete(self):
        self.create()
        self.assertFalse(self.store.delete_session('a'))
        self.store.finish('a', report(40), '2026-09-27T09:01:00+08:00')
        self.store.finish('a', report(99), '2026-09-27T09:02:00+08:00')
        self.assertFalse(self.store.checkpoint('a', report(99)))
        self.create('a', 88)
        self.assertEqual(self.store.load_session('a')['report']['average_sbi'], 40)

    def test_recovery_empty_signal_and_pagination(self):
        self.store.create_session('empty', '2026-09-27T09:00:00+08:00', {'average_sbi': None, 'peak_sbi': None}, 'test')
        self.assertEqual(self.store.recover_interrupted(), 1)
        self.assertEqual(self.store.recover_interrupted(), 0)
        self.assertIsNone(self.store.load_session('empty')['average_sbi'])
        for i in range(204):
            self.create(str(i))
        rows = sum((self.store.list_sessions(offset, 100) for offset in (0, 100, 200)), [])
        self.assertEqual(len(rows), 205)
        self.assertEqual(len({r['id'] for r in rows}), 205)
        self.assertNotIn('report', rows[0])

    def test_media_not_saved_and_partial_transcript_filtered(self):
        data = report()
        data['frame'] = b'video'
        data['audio'] = [1, 2, 3]
        data['transcripts'].append({'text': '未确认', 'is_final': False})
        self.store.create_session('a', '2026-09-27T09:00:00+08:00', data, 'test')
        stored = self.store.load_session('a')['report']
        self.assertNotIn('frame', stored)
        self.assertNotIn('audio', stored)
        self.assertEqual(len(stored['transcripts']), 1)
        data['history'][0]['raw_audio'] = [0, 1]
        with self.assertRaises(ValueError):
            self.store.checkpoint('a', data)

    def test_retry_recovery_excludes_sessions_created_this_launch(self):
        self.create('old')
        self.create('live')
        self.assertEqual(self.store.recover_interrupted(exclude_session_ids=['live']), 1)
        self.assertEqual(self.store.load_session('old')['status'], 'interrupted')
        self.assertEqual(self.store.load_session('live')['status'], 'running')
        self.assertTrue(self.store.checkpoint('live', report(30)))

    def test_single_bad_json_leaves_other_record_readable(self):
        self.create('a')
        self.create('b', 20)
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("UPDATE sessions SET report_json='broken' WHERE id='a'")
        self.assertEqual(len(self.store.list_sessions()), 2)
        with self.assertRaises(ValueError):
            self.store.load_session('a')
        self.assertEqual(self.store.load_session('b')['report']['average_sbi'], 20)

    def test_valid_json_with_invalid_report_shapes_is_isolated_and_preserved(self):
        self.create('bad')
        self.create('good', 20)
        cases = ({'advice': ['damaged advice']}, {'moments': ['bad']},
                 {'transcripts': 'bad'}, {'history': {}}, {'events': [3]},
                 {'moments': [{'start_ms': 'bad', 'end_ms': 100}]},
                 {'moments': [{'end_ms': 100}]})
        for malformed in cases:
            with self.subTest(malformed=malformed):
                raw = json.dumps({'elapsed_seconds': 30, **malformed})
                with self.store.db:
                    self.store.db.execute('UPDATE sessions SET report_json=? WHERE id=?', (raw, 'bad'))
                with self.assertRaisesRegex(ValueError, '无法读取'):
                    self.store.load_session('bad')
                self.assertEqual(self.store.db.execute("SELECT report_json FROM sessions WHERE id='bad'").fetchone()[0], raw)
                self.assertEqual(self.store.load_session('good')['average_sbi'], 20)
                self.assertEqual(len(self.store.list_sessions()), 2)

    def test_future_schema_not_changed(self):
        self.store.close()
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute('PRAGMA user_version=2')
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            HistoryStore(self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_non_database_not_overwritten(self):
        bad = Path(self.tmp.name) / 'bad.db'
        bad.write_bytes(b'not a database' * 10)
        with self.assertRaises(sqlite3.DatabaseError):
            HistoryStore(bad)
        self.assertEqual(bad.read_bytes(), b'not a database' * 10)

    def test_locked_write_rolls_back_and_can_retry(self):
        self.create()
        other = sqlite3.connect(self.path)
        try:
            other.execute('BEGIN EXCLUSIVE')
            with self.assertRaises(sqlite3.OperationalError):
                self.store.checkpoint('a', report(90))
        finally:
            other.rollback()
            other.close()
        self.assertEqual(self.store.load_session('a')['average_sbi'], 70)
        self.assertTrue(self.store.checkpoint('a', report(90)))

    def test_unwritable_directory_is_error_not_false_success(self):
        with patch.object(Path, 'mkdir', side_effect=PermissionError('no space')):
            with self.assertRaises(PermissionError):
                HistoryStore(Path(self.tmp.name) / 'new' / 'history.db')

    def test_schema_creation_failure_rolls_back_all_ddl(self):
        path = Path(self.tmp.name) / 'interrupted-schema.db'
        connect = sqlite3.connect
        class FailIndex(sqlite3.Connection):
            def execute(self, sql, *args):
                if sql.startswith('CREATE INDEX'):
                    raise sqlite3.OperationalError('simulated disk full')
                return super().execute(sql, *args)
        with patch('soci_ai.desktop.history_store.sqlite3.connect',
                   side_effect=lambda path, **kwargs: connect(path, factory=FailIndex, **kwargs)):
            with self.assertRaises(sqlite3.OperationalError):
                HistoryStore(path)
        with closing(connect(path)) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])
        # A retry must be able to initialize the untouched empty schema.
        retry = HistoryStore(path)
        retry.close()


if __name__ == '__main__':
    unittest.main()
