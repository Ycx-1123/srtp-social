"""Consumer-visible presentation checks, using Qt without opening devices."""
import os
import subprocess
import sys
import unittest


class CompactPresentationTests(unittest.TestCase):
    def run_qt(self, body):
        bootstrap = '''
import sys
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QLabel
from soci_ai.desktop.ui import StudioWindow
from soci_ai.desktop.core import RealtimeAssessment
app = QApplication([])
window = StudioWindow()
window.show()
engine = RealtimeAssessment()
engine.observe_text('学历低一点的人更适合跑腿', 0, True)
state = {**engine.update(0), 'status': 'running',
         'vision': {'face_detected': True, 'status': 'detected',
                    'features': {'baseline_ready': True, 'brow_tension': .6}},
         'transcript': {'text': '学历低一点的人更适合跑腿', 'status': '已确认', 'is_final': True}}
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', bootstrap + body + '\nwindow.close()'],
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_entire_live_page_fits_without_scrolling_or_clipped_panels(self):
        self.run_qt('''
for width, height in [(1440,840), (1280,720), (960,600), (760,480)]:
    window.resize(width, height)
    window.set_state(state)
    for _ in range(3): app.processEvents()
    viewport = window.live_scroll.viewport()
    assert window.live_scroll.verticalScrollBar().maximum() == 0, (width, height, 'vertical overflow')
    assert window.live_scroll.horizontalScrollBar().maximum() == 0, (width, height, 'horizontal overflow')
    for widget in [window.feedback_card, window.camera_card, window.camera,
                   window.tree, window.score_card, window.transcript, window.mic_level]:
        origin = widget.mapTo(viewport, QPoint(0,0))
        assert viewport.rect().contains(origin), (width, height, widget, origin)
        assert viewport.rect().contains(origin + QPoint(widget.width()-1, widget.height()-1)), (width, height, widget, widget.size())
''')

    def test_initial_geometry_on_small_work_area_still_fits_live_page(self):
        self.run_qt('''
from soci_ai.desktop.__main__ import initial_geometry
from PySide6.QtGui import QFontDatabase
for font in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/segoeui.ttf']:
    QFontDatabase.addApplicationFont(font)
for available in [(0,0,760,480),(0,0,960,540),(0,0,1280,680)]:
    x,y,width,height = initial_geometry(available)
    window.setMinimumSize(min(760,width),min(480,height))
    window.resize(width,height)
    window.set_state(state)
    for _ in range(3): app.processEvents()
    viewport = window.live_scroll.viewport()
    assert window.live_scroll.verticalScrollBar().maximum() == 0, (available,width,height)
    for widget in [window.camera,window.tree,window.ring,window.transcript]:
        pos = widget.mapTo(viewport,QPoint(0,0))
        assert viewport.rect().contains(pos+QPoint(widget.width()-1,widget.height()-1)), (available,widget)
''')

    def test_language_alert_never_moves_or_resizes_camera_tree_or_voice(self):
        self.run_qt('''
window.resize(1280,720)
window.set_state({'status':'running'})
app.processEvents()
before = [w.geometry() for w in [window.feedback_card, window.camera_card, window.tree, window.score_card, window.transcript]]
window.set_state(state)
app.processEvents()
after = [w.geometry() for w in [window.feedback_card, window.camera_card, window.tree, window.score_card, window.transcript]]
assert before == after, (before, after)
assert window.feedback_quote.isVisible()
assert '学历' in window.feedback_title.text()
assert '任务分配' in window.feedback_tags.text()
long_text = '这是一段很长但真实保留的识别文本，' * 50
window.set_state({**state, 'language_alert': {**state['language_alert'], 'text':long_text,
                  'tags':['群体概括','资资格预设','任务分配']*20, 'rewrite':long_text},
                  'transcript': {'text':long_text,'is_final':False,'status':'正在识别'}})
app.processEvents()
assert before == [w.geometry() for w in [window.feedback_card, window.camera_card, window.tree, window.score_card, window.transcript]]
assert window.transcript.text() == long_text
assert window.transcript.toolTip() == long_text, 'Eliding must never destroy the full transcript'
assert window.feedback_quote.toolTip() == long_text
window.set_state({'status':'stopped'})
app.processEvents()
assert not window.feedback_quote.isVisible()
assert '等待' in window.transcript.text()
''')

    def test_main_page_has_only_brow_and_downturn_rows_and_no_english_decorations(self):
        self.run_qt('''
window.set_state(state)
app.processEvents()
assert set(window.face_metrics) == {'brow_tension','mouth_downturn'}
for widget in window.findChildren(QLabel):
    if widget.isVisible():
        text = widget.text()
        for unwanted in ['SELF CHECK', 'CAMERA', 'VOICE', 'LIVE AUDIO', 'detected', 'INTERACTION STUDIO', 'LOCAL DESKTOP', '动作线索，不等于', '若开始时已皱眉']:
            assert unwanted not in text, text
''')

    def test_device_choices_remain_selectable_inside_settings(self):
        self.run_qt('''
window.set_devices([('摄像头 0',0),('摄像头 2',2)], [('默认麦克风',None),('USB microphone',7)])
window.camera_combo.setCurrentIndex(1)
window.microphone_combo.setCurrentIndex(1)
assert window.selected_camera() == 2
assert window.selected_microphone() == 7
assert not window.camera_combo.isVisible(), 'Devices must not consume main-page height'
window.settings_button.click()
app.processEvents()
assert window.camera_combo.isVisible()
assert window.microphone_combo.isVisible()
window.device_dialog.close()
''')

    def test_system_intro_titles_have_visible_hierarchy(self):
        self.run_qt('''
window._show_page(2)
app.processEvents()
for card in window.intro_cards:
    title = card.layout().itemAt(1).widget()
    assert title.font().pixelSize() >= 18, title.text()
    assert title.font().weight() >= 650, title.text()
''')


if __name__ == '__main__':
    unittest.main()
