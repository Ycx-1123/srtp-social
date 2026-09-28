"""Exercise the actual Qt presentation with controlled inputs, never devices."""
import os
import subprocess
import sys
import unittest


class FeedbackUiTests(unittest.TestCase):
    def test_language_labels_and_action_are_visible_above_camera_and_tree(self):
        script = '''
import sys
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.ui import StudioWindow
from soci_ai.desktop.core import RealtimeAssessment
app = QApplication([])
window = StudioWindow()
window.resize(1280, 720)
window.show()
engine = RealtimeAssessment()
engine.observe_text('学历低一点的人更适合跑腿', 0, True)
window.set_state({**engine.update(0), 'status': 'running'})
app.processEvents()
assert hasattr(window, 'feedback_card'), 'Language feedback needs a prominent main-page card'
assert window.feedback_card.isVisible()
assert '学历' in window.feedback_title.text()
assert '任务分配' in window.feedback_tags.text()
assert '学历低一点的人更适合跑腿' in window.feedback_quote.text()
assert '分工' in window.feedback_rewrite.text()
assert window.feedback_card.geometry().bottom() <= window.camera_card.geometry().top()
for size in [(1280,720), (960,600), (760,480)]:
    window.resize(*size)
    app.processEvents()
    assert window.pages.currentWidget().horizontalScrollBar().maximum() == 0, size
window.set_state({'status': 'stopped'})
assert not window.feedback_quote.isVisible(), 'Stopped sessions must clear live alerts'
window.close()
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
