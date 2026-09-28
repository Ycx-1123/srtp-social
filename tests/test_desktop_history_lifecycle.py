from pathlib import Path
import subprocess
import sys
import os
import tempfile
import unittest

from soci_ai.desktop.lifecycle import SessionHistoryLifecycle
from soci_ai.desktop.history_controller import HistoryController
from tests.test_desktop_history_controller import Mailbox
from tests.test_desktop_history_store import report
from tests import test_desktop_hover_ui as qt_helpers


class FakeSession:
    running = False
    latest_report = {}
    elapsed = 0
    released = False
    def start(self, *args):
        self.running = True
        return True
    def stop(self):
        self.running = False
        self.latest_report = report(44)
        return {'status': 'stopped'}
    def poll(self):
        self.latest_report = report(55)
        return {'status': 'running'}
    def shutdown_finished(self):
        return self.released


class LifecycleTests(unittest.TestCase):
    run_qt = qt_helpers.HoverPresentationTests.run_qt
    def setUp(self):
        self.worker = Mailbox()
        self.history = HistoryController(self.worker)
        self.session = FakeSession()
        self.runtime = SessionHistoryLifecycle(self.session, self.history)

    def test_start_stop_close_saves_one_and_waits_for_writes_and_devices(self):
        self.assertTrue(self.runtime.start(0, None))
        self.runtime.poll()
        self.runtime.stop()
        self.runtime.begin_close()
        self.assertEqual(sum(row[0] == 'create' for row in self.worker.sent), 1)
        self.assertEqual(sum(row[0] == 'finish' for row in self.worker.sent), 1)
        self.session.released = True
        self.assertFalse(self.runtime.can_shutdown())
        for row in list(self.worker.sent):
            self.worker.reply(row, True if row[0] == 'finish' else None)
        self.history.handle_results()
        self.assertTrue(self.runtime.can_shutdown())

    def test_no_session_close_creates_no_empty_history(self):
        self.runtime.begin_close()
        self.session.released = True
        self.assertTrue(self.runtime.can_shutdown())
        self.assertEqual(self.worker.sent, [])

    def test_failed_final_keeps_window_or_requires_explicit_discard(self):
        self.runtime.start(0, None)
        self.runtime.begin_close()
        for row in list(self.worker.sent):
            self.worker.reply(row, ok=False)
        self.history.handle_results()
        self.session.released = True
        self.assertFalse(self.runtime.can_shutdown())
        self.runtime.cancel_close()
        self.assertFalse(self.runtime.closing)
        self.assertFalse(self.session.running)
        self.runtime.begin_close()
        self.runtime.discard_unsaved()
        self.assertTrue(self.runtime.can_shutdown())

    def test_poll_stop_do_not_replace_selected_historical_report(self):
        self.runtime.start(0, None)
        self.history.select('b')
        self.worker.reply(self.worker.sent[-1], {'report': report(77)})
        self.history.handle_results()
        self.runtime.poll()
        self.runtime.stop()
        self.assertEqual(self.history.export_report()['average_sbi'], 77)

    def test_window_close_is_deferred_until_explicit_allow(self):
        self.run_qt('''
w = StudioWindow()
w.set_deferred_close(True)
w.show()
seen=[]
w.closing.connect(lambda: seen.append(1))
w.close()
app.processEvents()
assert w.isVisible() and seen == [1]
w.close()
assert seen == [1]
w.allow_close()
app.processEvents()
assert not w.isVisible()
''')

    def test_cli_history_phases_across_processes_and_no_device_access(self):
        with tempfile.TemporaryDirectory() as root:
            env = {**os.environ, 'SOCI_AI_TEST_DATA_DIR': root}
            for phase in ('write', 'read', 'delete', 'read-after-delete'):
                result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'soci_ai.desktop', '--self-test-history', '--history-test-phase', phase], env=env, capture_output=True, text=True, encoding='utf-8', timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                import json
                payload = json.loads((Path(root) / f'history-self-test-{phase}.json').read_text(encoding='utf-8'))
                self.assertTrue(payload['ok'])
                self.assertFalse(payload['camera_opened'])
                self.assertFalse(payload['microphone_opened'])
            self.assertFalse((Path(root) / 'video').exists())
