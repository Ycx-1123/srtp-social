from pathlib import Path
import tempfile
import threading
import time
import unittest

from soci_ai.desktop.history_worker import HistoryWorker
from soci_ai.desktop.history_store import HistoryStore
from tests.test_desktop_history_store import report


def collect(worker, count, timeout=5):
    results = []
    end = time.monotonic() + timeout
    while len(results) < count and time.monotonic() < end:
        results.extend(worker.drain_results())
        threading.Event().wait(.005)
    if len(results) < count:
        raise AssertionError((count, results))
    return results


class WorkerTests(unittest.TestCase):
    def test_blocking_disk_does_not_block_submit_coalesces_immutable_snapshots(self):
        entered, release = threading.Event(), threading.Event()
        class SlowStore(HistoryStore):
            def create_session(self, **kw):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test timed out')
                super().create_session(**kw)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.db'
            worker = HistoryWorker(path, store_factory=SlowStore)
            worker.start()
            worker.submit('create', 1, session_id='a', started_at='2026-09-27T09:00:00+08:00', report=report(1), app_version='test')
            self.assertTrue(entered.wait(2))
            try:
                for i in range(100):
                    data = report(i)
                    worker.submit('checkpoint', i+2, session_id='a', report=data)
                    data['history'][0]['sbi'] = -1
                release.set()
                results = collect(worker, 101)
                self.assertTrue(all(row['ok'] for row in results))
                worker.request_shutdown()
                worker.join(3)
                self.assertFalse(worker.is_alive())
                store = HistoryStore(path)
                try:
                    self.assertEqual(store.load_session('a')['report']['history'][0]['sbi'], 99)
                finally:
                    store.close()
            finally:
                release.set()
                worker.request_shutdown()
                worker.join(3)

    def test_control_order_finish_and_delete_never_resurrect(self):
        with tempfile.TemporaryDirectory() as root:
            worker = HistoryWorker(Path(root) / 'history.db')
            worker.start()
            try:
                worker.submit('create', 1, session_id='a', started_at='2026-09-27T09:00:00+08:00', report=report(), app_version='test')
                worker.submit('checkpoint', 2, session_id='a', report=report(90))
                worker.submit('finish', 3, session_id='a', report=report(55), ended_at='2026-09-27T09:01:00+08:00')
                worker.submit('delete', 4, session_id='a')
                worker.submit('checkpoint', 5, session_id='a', report=report(88))
                worker.submit('load', 6, session_id='a')
                rows = collect(worker, 6)
                load = next(row for row in rows if row['request_id'] == 6)
                self.assertIsNone(load['value'])
                self.assertTrue(next(row for row in rows if row['request_id'] == 4)['value'])
            finally:
                worker.request_shutdown()
                worker.join(3)

    def test_initial_open_failure_retry_and_paging(self):
        attempts = []
        def factory(path):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError('read only')
            return HistoryStore(path)
        with tempfile.TemporaryDirectory() as root:
            worker = HistoryWorker(Path(root) / 'history.db', store_factory=factory)
            worker.start()
            try:
                worker.submit('recover', 1)
                self.assertFalse(collect(worker, 1)[0]['ok'])
                worker.submit('retry', 2)
                self.assertTrue(collect(worker, 1)[0]['ok'])
                worker.submit('list', 3, offset=0, limit=100)
                self.assertEqual(collect(worker, 1)[0]['value'], {'items': [], 'offset': 0, 'has_more': False})
            finally:
                worker.request_shutdown()
                worker.join(3)
