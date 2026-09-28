from copy import deepcopy
import unittest

from soci_ai.desktop.history_controller import HistoryController
from tests.test_desktop_history_store import report


class Mailbox:
    def __init__(self):
        self.sent, self.results = [], []
    def submit(self, operation, request_id, **payload):
        self.sent.append((operation, request_id, deepcopy(payload)))
    def drain_results(self):
        result, self.results = self.results, []
        return result
    def reply(self, message, value=None, ok=True):
        op, key, payload = message
        self.results.append(dict(operation=op, request_id=key, session_id=payload.get('session_id'), value=value, ok=ok, error='' if ok else 'disk full'))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.worker = Mailbox()
        self.now = 0
        self.controller = HistoryController(self.worker, clock=lambda: self.now)

    def begin(self):
        return self.controller.begin(report(10), '2026-09-27T09:00:00+08:00', 'test')

    def test_checkpoint_exactly_five_seconds_and_final_idempotent(self):
        key = self.begin()
        self.worker.reply(self.worker.sent[0], None)
        self.controller.handle_results()
        self.now = 4.9
        self.controller.update_current(report(20))
        self.assertFalse(any(row[0] == 'checkpoint' for row in self.worker.sent))
        self.now = 5
        self.controller.update_current(report(30))
        self.assertEqual(self.worker.sent[-1][0], 'checkpoint')
        self.assertEqual(self.worker.sent[-1][2]['session_id'], key)
        self.controller.finish_current(report(50), '2026-09-27T09:01:00+08:00')
        self.controller.finish_current(report(90), '2026-09-27T09:02:00+08:00')
        finals = [row for row in self.worker.sent if row[0] == 'finish']
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0][2]['report']['average_sbi'], 50)

    def test_two_failed_finals_survive_new_sessions_and_retry(self):
        keys = []
        for value in (40, 70):
            key = self.begin()
            keys.append(key)
            create = self.worker.sent[-1]
            self.worker.reply(create, ok=False)
            self.controller.handle_results()
            self.controller.finish_current(report(value), '2026-09-27T09:01:00+08:00')
            self.worker.reply(self.worker.sent[-1], ok=False)
            self.controller.handle_results()
        self.assertNotEqual(*keys)
        self.assertEqual(self.controller.save_status, '保存失败')
        self.controller.retry_failed()
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        sent = [row for row in self.worker.sent if row[0] == 'create' and row[2]['session_id'] in keys]
        self.assertEqual(len(sent), 4)
        for message in sent[-2:]:
            self.worker.reply(message, None)
        self.controller.handle_results()
        finals = [row for row in self.worker.sent if row[0] == 'finish'][-2:]
        self.assertEqual({row[2]['report']['average_sbi'] for row in finals}, {40, 70})
        for message in finals:
            self.worker.reply(message, True)
        self.controller.handle_results()
        self.assertFalse(self.controller.pending_writes())
        self.assertEqual(self.controller.save_status, '已保存')

    def test_selection_token_current_updates_and_export(self):
        self.begin()
        self.controller.select('a')
        load_a = self.worker.sent[-1]
        self.controller.select('b')
        load_b = self.worker.sent[-1]
        self.worker.reply(load_b, {'id': 'b', 'report': report(77)})
        self.worker.reply(load_a, {'id': 'a', 'report': report(22)})
        self.controller.handle_results()
        self.controller.update_current(report(1))
        self.assertEqual(self.controller.export_report()['average_sbi'], 77)
        self.controller.select('bad')
        self.assertEqual(self.controller.export_report(), {})
        self.worker.reply(self.worker.sent[-1], ok=False)
        self.controller.handle_results()
        self.assertEqual(self.controller.export_report(), {})
        self.assertTrue(self.controller.load_error)

    def test_pagination_default_latest_and_preserved_selection(self):
        self.controller.refresh_history()
        self.worker.reply(self.worker.sent[-1], {'items': [{'id': 'a', 'status': 'completed'}], 'offset': 0, 'has_more': True})
        self.controller.handle_results()
        self.assertEqual(self.controller.selected_id, 'a')
        self.controller.refresh_history(1)
        self.worker.reply(self.worker.sent[-1], {'items': [{'id': 'b', 'status': 'completed'}], 'offset': 1, 'has_more': False})
        self.controller.handle_results()
        self.assertEqual([r['id'] for r in self.controller.items], ['a', 'b'])
        self.assertEqual(self.controller.selected_id, 'a')
        self.assertFalse(self.controller.has_more)

    def test_delete_running_not_allowed_then_neighbour_selected(self):
        self.controller.items = [{'id': 'a', 'status': 'running'}, {'id': 'b', 'status': 'completed'}]
        self.controller.select('a')
        before = len(self.worker.sent)
        self.controller.delete_selected()
        self.assertEqual(len(self.worker.sent), before)
        self.controller.items[0]['status'] = 'completed'
        self.controller.delete_selected()
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        self.assertEqual(self.controller.selected_id, 'b')

    def test_refresh_retains_deep_selection_metadata_without_skipping_pages(self):
        self.controller.items = [{'id': str(i), 'status': 'completed'} for i in range(205)]
        self.controller.select('204')
        self.worker.reply(self.worker.sent[-1], {'id': '204', 'status': 'completed', 'report': report(77)})
        self.controller.handle_results()
        self.controller.refresh_history()
        rows = [{'id': 'new-' + str(i), 'status': 'completed'} for i in range(100)]
        self.worker.reply(self.worker.sent[-1], {'items': rows, 'offset': 0, 'has_more': True})
        self.controller.handle_results()
        self.assertEqual(self.controller.selected_id, '204')
        self.assertEqual(self.controller.export_report()['average_sbi'], 77)
        self.assertEqual(len(self.controller.items), 100)  # No sparse item counted in pagination offset.
        self.controller.delete_selected()
        self.assertEqual(self.worker.sent[-1][0], 'delete')
        self.assertEqual(self.worker.sent[-1][2]['session_id'], '204')
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        self.assertEqual(self.controller.selected_id, 'new-99')

    def test_retry_failed_recovery_precedes_replay_and_excludes_live_session(self):
        recovery = self.controller._send('recover')
        self.worker.reply(self.worker.sent[-1], ok=False)
        self.controller.handle_results()
        key = self.begin()
        self.worker.reply(self.worker.sent[-1], None)
        self.controller.handle_results()
        self.controller.retry_failed()
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        recoveries = [row for row in self.worker.sent if row[0] == 'recover']
        self.assertEqual(len(recoveries), 2)
        self.assertIn(key, recoveries[-1][2]['exclude_session_ids'])
        self.assertEqual(self.controller.save_status, '保存失败')
        self.assertTrue(self.controller.pending_writes())  # Close waits for recovery before replay.
        self.worker.reply(recoveries[-1], 1)
        self.controller.handle_results()
        self.assertEqual(self.controller.save_error, '')

    def test_each_failed_session_remains_selectable_for_manual_export(self):
        keys = []
        for score in (22, 77):
            key = self.begin()
            keys.append(key)
            self.worker.reply(self.worker.sent[-1], ok=False)
            self.controller.handle_results()
            self.controller.finish_current(report(score), '2026-09-27T09:01:00+08:00')
            self.worker.reply(self.worker.sent[-1], ok=False)
            self.controller.handle_results()
        self.assertEqual(self.controller.items, [])
        visible = self.controller.display_items()
        self.assertEqual({row['id'] for row in visible}, set(keys))
        self.assertTrue(all(row['status'] == 'unsaved' for row in visible))
        before = len(self.worker.sent)
        self.controller.select(keys[0])
        self.assertEqual(self.controller.export_report()['average_sbi'], 22)
        self.controller.delete_selected()
        self.assertEqual(len(self.worker.sent), before)
        self.controller.select(keys[1])
        self.assertEqual(self.controller.export_report()['average_sbi'], 77)

    def test_last_current_record_delete_clears_current_export(self):
        key = self.begin()
        self.worker.reply(self.worker.sent[-1], None)
        self.controller.handle_results()
        self.controller.finish_current(report(22), '2026-09-27T09:01:00+08:00')
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        self.controller.items = [{'id': key, 'status': 'completed'}]
        self.controller.select(key)
        self.controller.delete_selected()
        self.worker.reply(self.worker.sent[-1], True)
        self.controller.handle_results()
        self.controller.select(None)
        self.assertEqual(self.controller.export_report(), {})

    def test_successful_list_retry_clears_only_list_origin_error(self):
        self.controller.refresh_history()
        self.worker.reply(self.worker.sent[-1], ok=False)
        self.controller.handle_results()
        self.assertTrue(self.controller.load_error)
        self.controller.refresh_history()
        self.worker.reply(self.worker.sent[-1], {'items': [], 'offset': 0, 'has_more': False})
        self.controller.handle_results()
        self.assertEqual(self.controller.load_error, '')
