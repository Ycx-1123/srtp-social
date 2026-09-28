"""Specific instant cues and stable, evidence-based whole-session priorities."""
import unittest

from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.session import DesktopSession
from tests.test_desktop_session import FakeCamera, FakeAudio
from tests import test_desktop_compact_ui as qt_helpers


def face(at, **features):
    return {'captured_at':at, 'face_detected':True,
            'features':{'baseline_ready':True, **features}}


class SpecificLiveGuidanceTests(unittest.TestCase):
    def test_brow_advice_does_not_give_generic_mouth_instruction(self):
        state = RealtimeAssessment().update(0, vision=face(0, brow_tension=.8))
        self.assertIn('舒展眉头', state['suggestion'])
        self.assertNotIn('嘴', state['suggestion'])

    def test_downturned_corners_get_their_own_instruction(self):
        state = RealtimeAssessment().update(0, vision=face(0, mouth_downturn=.8))
        self.assertIn('放松嘴角', state['suggestion'])
        self.assertNotIn('眉', state['suggestion'])

    def test_both_observed_actions_have_both_instructions(self):
        state = RealtimeAssessment().update(0, vision=face(0, brow_tension=.8, mouth_downturn=.7))
        self.assertIn('舒展眉头', state['suggestion'])
        self.assertIn('放松嘴角', state['suggestion'])

    def test_new_specific_action_preempts_previous_reading_hold(self):
        model = RealtimeAssessment()
        model.update(0, vision=face(0, brow_tension=.8))
        changed = model.update(.3, vision=face(.3, mouth_downturn=.8))
        self.assertIn('嘴角', changed['suggestion'])
        self.assertNotIn('眉', changed['suggestion'])


class CumulativeFocusTests(unittest.TestCase):
    def sequence(self, model, start, end, **features):
        state = None
        for i in range(start, end):
            at = i / 10
            state = model.update(at, vision=face(at, **features))
        return state

    def items(self, state):
        self.assertIn('session_focus', state, 'whole-session priorities must reach the real state consumer')
        return state['session_focus']['items']

    def test_focus_uses_cumulative_observation_not_just_the_last_frame(self):
        model = RealtimeAssessment()
        self.sequence(model, 0, 81, brow_tension=.8, micro_expression=.8)
        # At the ten-second refresh the last face is relaxed; eight observed
        # seconds of brows still deserve a whole-session reminder.
        state = self.sequence(model, 81, 101)
        items = self.items(state)
        self.assertTrue(any(item['text']=='舒展眉头' for item in items), items)
        self.assertTrue(any('本场' in item['detail'] for item in items))

    def test_single_short_peak_does_not_become_a_whole_session_priority(self):
        model = RealtimeAssessment()
        self.sequence(model, 0, 100)
        state = model.update(10, vision=face(10, mouth_downturn=1, micro_expression=1))
        self.assertFalse(any(item['key']=='mouth_downturn' for item in self.items(state)))

    def test_focus_does_not_refresh_on_every_new_utterance(self):
        model = RealtimeAssessment()
        model.observe_text('学历低一点的人更适合跑腿', 0, True)
        first = self.items(model.update(0))
        model.observe_text('女生就是学不好工科吧', 1, True)
        state = model.update(1)
        self.assertEqual(self.items(state), first)
        self.assertIn('性别', state['suggestion_title'], 'instant language feedback stays independent')
        later = self.items(model.update(10))
        self.assertTrue(any('性别' in item['detail'] for item in later))

    def test_partial_or_retracted_language_does_not_enter_cumulative_focus(self):
        model = RealtimeAssessment()
        model.observe_text('女生不适合学工科', 0, False)
        state = model.update(0)
        self.assertFalse(any(item['key']=='language' for item in self.items(state)))
        model.observe_text('女生不适合学工科这种说法不对', 1, True)
        state = model.update(10)
        self.assertFalse(any(item['key']=='language' for item in self.items(state)))

    def test_cumulative_pressure_advice_never_claims_fast_speech_or_high_pitch(self):
        model = RealtimeAssessment()
        for i in range(101):
            at = i / 10
            state = model.update(at, audio={'captured_at':at, 'pressure':.8, 'rms':.25})
        items = self.items(state)
        self.assertTrue(any(item['text']=='降低音量' for item in items))
        self.assertNotIn('语速过快', str(items))
        self.assertNotIn('音调过高', str(items))

    def test_no_signal_does_not_invent_quiet_voice_or_good_performance(self):
        model = RealtimeAssessment()
        state = model.update(10)
        items = self.items(state)
        self.assertEqual(items, [])
        self.assertIn('信号', state['session_focus']['status'])

    def test_end_preserves_final_focus_and_restart_clears_it(self):
        now = [100]
        session = DesktopSession(camera_factory=FakeCamera, audio_factory=FakeAudio, clock=lambda:now[0])
        session.start()
        session.assessment.observe_text('学历低一点的人更适合跑腿', 0, True)
        session.poll()
        now[0] = 103
        stopped = session.stop()
        items = self.items(stopped)
        self.assertTrue(any(item['text']=='避免学历概括' for item in items))
        self.assertEqual(session.latest_report['session_focus']['items'], items)
        session.camera.alive = False
        session.start()
        self.assertEqual(self.items(session.poll()), [])


class LayeredFocusUITests(unittest.TestCase):
    def run_qt(self, body):
        qt_helpers.CompactPresentationTests().run_qt(body)

    def test_tiny_ring_keeps_score_and_caption_separate_and_legible(self):
        self.run_qt('''
from unittest.mock import patch, MagicMock
from PySide6.QtGui import QFontMetricsF
window.resize(760,480)
window.ring.set_value(55)
for height in (48,64,100,175):
    window.ring.setFixedHeight(height)
    draws=[]
    painter=MagicMock()
    font=[None]
    painter.setFont.side_effect=lambda value: font.__setitem__(0,value)
    painter.drawText.side_effect=lambda rect,alignment,text: draws.append((rect,font[0],text))
    with patch('soci_ai.desktop.ui.QPainter',return_value=painter):
        window.ring.paintEvent(None)
    score,caption=draws
    for rect,face,text in draws:
        bounds=QFontMetricsF(face).boundingRect(text)
        assert bounds.width()<=rect.width(), (height,text,'clipped width')
        assert bounds.height()<=rect.height(), (height,text,'clipped height')
    assert score[0].bottom()<=caption[0].top(), (height,'overlapping score and caption')
''')

    def test_three_nonred_priorities_fit_in_the_existing_single_screen(self):
        self.run_qt('''
focus = {'status':'截至目前 · 本场重点', 'items':[
    {'key':'brow_tension','text':'舒展眉头','detail':'本场眉部紧张较常出现。','tone':'amber'},
    {'key':'mouth_downturn','text':'放松嘴角','detail':'本场嘴角下压较常出现。','tone':'blue'},
    {'key':'pressure','text':'降低音量','detail':'本场音量压力多次升高。','tone':'amber'}]}
assert hasattr(window,'focus_card'), 'missing visible whole-session focus'
for width,height in [(1440,840),(1280,720),(960,600),(760,480)]:
    window.resize(width,height)
    window.set_state({**state,'session_focus':focus})
    for _ in range(3): app.processEvents()
    viewport=window.live_scroll.viewport()
    assert window.live_scroll.verticalScrollBar().maximum()==0, (width,height)
    for hint in window.focus_labels:
        if hint.isVisible():
            pos=hint.mapTo(viewport,QPoint(0,0))
            assert viewport.rect().contains(pos+QPoint(hint.width()-1,hint.height()-1)), (width,height,hint.text())
            assert hint.font().weight()>=600
            color=hint.palette().color(hint.foregroundRole())
            assert not (color.red()>color.green()*1.4 and color.red()>color.blue()*1.4), color.name()
    assert '舒展眉头' in window.focus_labels[0].text()
    assert '本场' in window.focus_labels[0].toolTip()
window.set_state({'status':'stopped','session_focus':focus})
assert '舒展眉头' in window.focus_labels[0].text()
window.set_state({'status':'starting'})
assert not any(hint.text() for hint in window.focus_labels), 'new session cannot keep previous advice'
''')
