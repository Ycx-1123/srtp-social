"""User-visible regression checks for opt-in history and speech-safe cues."""
from pathlib import Path
import tempfile
import time
import unittest
import os
import subprocess
import sys

from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.advice import meeting_advice
from soci_ai.desktop.facial import FacialCueTracker
from soci_ai.desktop.history_controller import HistoryController
from soci_ai.desktop.history_worker import HistoryWorker
from soci_ai.desktop.history_store import HistoryStore
from soci_ai.desktop.lifecycle import SessionHistoryLifecycle
from tests.test_desktop_facial import blendshapes
from tests.test_desktop_history_store import report
from tests.test_desktop_history_lifecycle import FakeSession
from tests.test_desktop_hover_ui import HoverPresentationTests


class SpeechSafeCuesTests(unittest.TestCase):
    def test_pressing_or_puckering_lips_alone_cannot_raise_sbi_or_warning(self):
        tracker, assessment = FacialCueTracker(), RealtimeAssessment()
        for at in range(0, 640, 80):
            tracker.update(blendshapes(), at)
        for at in range(640, 2640, 80):
            features = tracker.update(blendshapes(mouthPressLeft=.9, mouthPucker=.9), at)
            state = assessment.update(at / 1000, vision={
                'captured_at': at / 1000, 'face_detected': True, 'features': features})
            self.assertEqual(state['sbi'], 0)
            self.assertNotIn('唇部', state['suggestion_title'])
        self.assertEqual(assessment.report(3, {}, {})['moments'], [])

    def test_old_lip_feature_is_not_a_fallback_negative_channel(self):
        assessment = RealtimeAssessment()
        for at in (0, .1, .2, .3):
            state = assessment.update(at, vision={'captured_at': at, 'face_detected': True,
                'features': {'baseline_ready': True, 'lip_tension': 1}})
        self.assertEqual(state['sbi'], 0)

    def test_brow_and_downturned_corners_still_have_sustained_response(self):
        for action in ('browDownLeft', 'mouthFrownRight'):
            with self.subTest(action=action):
                tracker = FacialCueTracker()
                for at in range(0, 640, 80):
                    tracker.update(blendshapes(), at)
                for at in range(640, 1200, 80):
                    cues = tracker.update(blendshapes(**{action: .3}), at)
                self.assertGreater(cues['micro_expression'], .5)


class OptInHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'history.sqlite3'
        self.worker = HistoryWorker(self.path)
        worker = self.worker
        self.addCleanup(lambda: (worker.request_shutdown(), worker.join(3)))
        self.worker.start()
        self.controller = HistoryController(self.worker, auto_save=False)

    def tearDown(self):
        self.worker.request_shutdown()
        self.worker.join(3)
        self.tmp.cleanup()

    def settle(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.controller.handle_results()
            if not self.controller._requests:
                return
            time.sleep(.01)
        self.fail('history did not settle')

    def ended(self):
        key = self.controller.begin(report(30), '2026-09-27T09:00:00+08:00', 'test')
        self.controller.update_current(report(40))
        self.controller.finish_current(report(70), '2026-09-27T09:01:00+08:00')
        return key

    def test_no_disk_history_until_user_selects_save(self):
        self.ended()
        self.settle()
        self.assertTrue(self.controller.needs_save_choice())
        with_store = HistoryStore(self.path)
        try:
            self.assertEqual(with_store.list_sessions(), [])
        finally:
            with_store.close()

    def test_chosen_save_survives_reopen_with_curve_advice_and_single_delete(self):
        key = self.ended()
        self.assertTrue(self.controller.save_current())
        self.assertFalse(self.controller.save_current(), 'double click must not duplicate')
        self.settle()
        self.assertFalse(self.controller.needs_save_choice())
        self.worker.request_shutdown()
        self.worker.join(3)
        self.worker = HistoryWorker(self.path)
        self.worker.start()
        self.controller = HistoryController(self.worker, auto_save=False)
        self.controller.refresh_history()
        self.settle()
        self.assertEqual(self.controller.selected_id, key)
        saved = self.controller.export_report()
        self.assertEqual(saved['history'][0]['cues'], ['疑似学历偏见'])
        self.assertEqual(saved['advice'][0]['body'], '用任务要求代替身份判断')
        self.assertEqual(saved['peak_sbi'], 70)
        self.controller.delete_selected()
        self.settle()
        self.assertEqual(self.controller.items, [])
        self.assertEqual(self.controller.export_report(), {})

    def test_discarded_choice_leaves_no_reopen_record(self):
        self.ended()
        self.assertTrue(self.controller.discard_current())
        self.assertFalse(self.controller.needs_save_choice())
        self.controller.refresh_history()
        self.settle()
        self.assertEqual(self.controller.items, [])

    def test_deleting_last_old_record_preserves_new_unsaved_review(self):
        old_key = self.ended()
        self.controller.save_current()
        self.settle()
        self.ended()
        self.controller.select(old_key)
        self.settle()
        self.controller.delete_selected()
        self.settle()
        self.assertEqual(self.controller.items, [])
        self.assertIsNone(self.controller.selected_id)
        self.assertTrue(self.controller.needs_save_choice())
        self.assertEqual(self.controller.export_report()['peak_sbi'], 70)

    def test_cannot_start_next_session_without_resolving_unsaved_choice(self):
        self.ended()
        with self.assertRaises(RuntimeError):
            self.controller.begin(report(), '2026-09-27T10:00:00+08:00', 'test')

    def test_close_waits_for_choice_and_still_allows_explicit_discard(self):
        runtime = SessionHistoryLifecycle(FakeSession(), self.controller)
        runtime.start(0, None)
        runtime.begin_close()
        runtime.session.released = True
        self.assertFalse(runtime.can_shutdown())
        self.controller.discard_current()
        self.assertTrue(runtime.can_shutdown())

    def test_runtime_does_not_open_devices_before_resolving_previous_report(self):
        self.ended()
        session = FakeSession()
        runtime = SessionHistoryLifecycle(session, self.controller)
        self.assertFalse(runtime.start(0, None))
        self.assertFalse(session.running)


class SummaryReportTests(unittest.TestCase):
    def test_brief_stronger_peak_does_not_replace_sustained_brow_summary(self):
        cards = meeting_advice({'face_seconds':10, 'tension_seconds':5,
            'action_seconds':{'brow_tension':5, 'mouth_downturn':.1},
            'peaks':{'brow_tension':.4, 'mouth_downturn':.9}}, [], [])
        expression = next(card for card in cards if card['kind']=='expression')
        self.assertIn('眉部紧张', expression['body'])
        self.assertNotIn('嘴角下压', expression['body'])

    def test_report_summarizes_recurrent_brows_and_specific_bias_category(self):
        model = RealtimeAssessment()
        for i in range(100):
            at = i / 10
            model.update(at, vision={'captured_at': at, 'face_detected': True,
                'features': {'baseline_ready': True, 'brow_tension': .8, 'micro_expression': .8}},
                audio={'captured_at': at, 'pressure': .8})
        model.observe_text('学历低一点的人更适合跑腿', 9.8, True)
        advice = model.report(10, {}, {})['advice']
        overview = next((card for card in advice if card['kind'] == 'overview'), {})
        self.assertIn('眉', overview.get('body', ''))
        self.assertIn('学历', overview.get('body', ''))
        self.assertIn('音量', overview.get('body', ''))
        self.assertNotIn('情绪不自然', overview.get('body', ''))

    def test_no_signal_report_does_not_invent_fast_speech_or_high_pitch(self):
        model = RealtimeAssessment()
        model.update(0)
        cards = model.report(1, {}, {})['advice']
        overview = next((card for card in cards if card['kind'] == 'overview'), {})
        self.assertIn('不足', overview.get('body', ''))
        self.assertNotIn('语速过快', str(cards))
        self.assertNotIn('音调过高', str(cards))


class CompleteSessionRetentionTests(unittest.TestCase):
    def test_long_session_curve_retains_start_and_end(self):
        model = RealtimeAssessment()
        for i in range(7201):
            at = i * .6
            model.update(at, audio={'captured_at':at, 'pressure':.2})
        rows = model.report(at, {}, {})['history']
        self.assertEqual(len(rows), 7201)
        self.assertEqual(rows[0]['at_ms'], 0)
        self.assertEqual(rows[-1]['at_ms'], int(at * 1000))

    def test_full_confirmed_text_and_bias_events_survive_long_session(self):
        model = RealtimeAssessment()
        model.observe_text('学历低一点的人更适合跑腿', 0, True)
        for i in range(1, 1001):
            model.observe_text('女生就是学不好工科吧', i, True)
        result = model.report(1001, {}, {})
        self.assertEqual(len(result['transcripts']), 1001)
        self.assertEqual(len(result['events']), 1001)
        self.assertEqual(result['observation_summary']['semantic_hits'], 1001)
        self.assertIn('学历', result['advice'][0]['body'])
        self.assertEqual(result['transcripts'][0]['at_ms'], 0)


class SavedReviewUITests(unittest.TestCase):
    run_qt = HoverPresentationTests.run_qt

    def test_save_and_export_are_separate_and_lip_metric_is_absent(self):
        self.run_qt('''
w = StudioWindow()
w.set_history_save_available(True)
assert w.history_save.isEnabled()
assert '保存' in w.history_save.text()
w.set_report({'elapsed_seconds':30,'history':[{'at_ms':0,'sbi':70,'cues':['疑似学历偏见']}],
    'advice':[{'title':'本场总结','body':'需要留意学历概括。'}]})
assert w.export_button.isEnabled()
assert 'lip_tension' not in w.face_metrics
seen=[]
w.history_save_requested.connect(lambda:seen.append('save'))
w.history_save.click()
assert seen==['save']
w.set_history_items([{'id':'saved','status':'completed','started_at':'2026-09-27T09:00:00+08:00'}], 'saved')
w.set_history_save_available(True)
assert not w.history_save.isEnabled(), 'historical report must not save the unrelated current session'
w.close()
''')

    def test_desktop_entry_saves_only_on_button_and_reloads_full_review(self):
        script = '''
import ctypes, tempfile
ctypes.windll.kernel32.SetErrorMode(3)
from pathlib import Path
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer
from soci_ai.desktop import resources, session
from soci_ai.desktop.history_store import HistoryStore
from tests.test_desktop_history_lifecycle import FakeSession
from tests.test_desktop_history_store import report
root=tempfile.TemporaryDirectory()
resources.output_root=lambda:Path(root.name)
resources.export_root=lambda:Path(root.name)/'exports'
class Devices(FakeSession):
    latest_state={'status':'ready'}
    message=''
    released=True
    def calibrate(self):pass
    def frame(self):return None
    def poll(self):
        self.latest_report=report(55)
        self.latest_state={'status':'running' if self.running else 'stopped'}
        return self.latest_state
    def stop(self):
        self.running=False
        self.latest_report=report(70)
        self.latest_state={'status':'stopped'}
        return self.latest_state
session.DesktopSession=Devices
import sounddevice
sounddevice.query_devices=lambda:[]
real_app=QtWidgets.QApplication
errors=[]
def window():
    return next(w for w in real_app.instance().topLevelWidgets() if hasattr(w,'history_save'))
def begin():
    try:
        w=window()
        w.start_requested.emit()
        w.stop_requested.emit()
        with_store=HistoryStore(Path(root.name)/'history.sqlite3')
        assert with_store.list_sessions()==[], 'entry must not auto-save before choice'
        with_store.close()
        assert w.pages.currentIndex()==1
        assert w.history_save.isEnabled()
        w.history_save.click()
        QTimer.singleShot(700, verify)
    except Exception as e:
        errors.append(str(e));real_app.instance().quit()
def verify():
    try:
        w=window()
        with_store=HistoryStore(Path(root.name)/'history.sqlite3')
        rows=with_store.list_sessions()
        assert len(rows)==1 and rows[0]['status']=='completed', rows
        saved=with_store.load_session(rows[0]['id'])['report']
        assert saved['history'][0]['cues']==['疑似学历偏见']
        assert saved['advice'][0]['body']=='用任务要求代替身份判断'
        with_store.close()
        assert not w.history_save.isEnabled()
        w.close()
    except Exception as e:
        errors.append(str(e));real_app.instance().quit()
class App(real_app):
    def __init__(self,*args):
        super().__init__(*args)
        QTimer.singleShot(150,begin)
        QTimer.singleShot(6000,self.quit)
QtWidgets.QApplication=App
from soci_ai.desktop.__main__ import main
assert main([])==0
assert not errors, errors
root.cleanup()
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
            env={**os.environ, 'QT_QPA_PLATFORM':'offscreen'},
            capture_output=True, text=True, encoding='utf-8', timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)
