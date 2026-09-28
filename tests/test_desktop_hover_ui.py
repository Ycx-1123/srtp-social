"""Offscreen consumer checks: bare tracking outline and short peak labels."""
import os
import subprocess
import sys
import unittest


class HoverPresentationTests(unittest.TestCase):
    def run_qt(self, body):
        bootstrap = '''
import sys
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, QEvent
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from soci_ai.desktop.ui import CameraPreview, StudioWindow
from soci_ai.desktop.charts import SessionTimeline
app = QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', bootstrap + body],
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tracking_outline_has_no_text_and_still_moves_with_bounds(self):
        self.run_qt('''
import numpy as np
camera = CameraPreview()
camera.resize(400,300)
camera.show()
frame = np.zeros((300,400,3),dtype=np.uint8)
camera.set_frame(frame, {'x':.25,'y':.2,'width':.3,'height':.5})
app.processEvents()
image = camera.grab().toImage()
box = camera.box_rect()
inside = box.adjusted(12,12,-12,-12).toRect()
assert all(image.pixelColor(x,y).green()<100 for y in range(inside.top(),inside.bottom()) for x in range(inside.left(),inside.right())), 'No text may be drawn inside the tracking outline'
edge = image.pixelColor(round(box.center().x()),round(box.top()))
assert edge.green()>150, 'The outline itself must remain visible'
camera.set_frame(frame, {'x':.1,'y':.2,'width':.3,'height':.5})
assert camera.box_rect().left()>box.left(), 'Mirrored tracking bounds must continue to move'
camera.close()
''')

    def test_hover_keeps_labels_aligned_after_sorting_and_never_invents_missing_reason(self):
        self.run_qt('''
chart = SessionTimeline()
chart.resize(900,300)
chart.show()
chart.set_history([
    {'at_ms':2000,'sbi':80,'cues':['眉部紧张增强']},
    {'at_ms':None,'sbi':90,'cues':['错误记录']},
    {'at_ms':0,'sbi':75,'cues':['疑似学历偏见']},
    {'at_ms':3000,'sbi':63},
    {'at_ms':4000,'sbi':None,'cues':['过期线索']},
    {'at_ms':5000,'sbi':62,'cues':['唇部收紧增强'],'cue_phase':'recovering'},
], 5)
app.processEvents()
for at, wanted, unwanted in [(0,'疑似学历偏见','眉部'),(2,'眉部紧张增强','学历'),
                             (3,'未记录该点的具体线索','疑似'),(4,'暂无有效信号','过期'),
                             (5,'唇部收紧增强（回落中）','学历')]:
    point = chart._position(at,50)
    QTest.mouseMove(chart,QPoint(round(point.x()),round(point.y())))
    app.processEvents()
    tooltip = chart.toolTip()
    assert wanted in tooltip, (at,tooltip)
    assert unwanted not in tooltip, (at,tooltip)
app.sendEvent(chart,QEvent(QEvent.Type.Leave))
assert chart.toolTip()=='', 'Old peak label must disappear when pointer leaves'
chart.close()
''')

    def test_review_passes_per_sample_labels_to_the_displayed_curve(self):
        self.run_qt('''
window = StudioWindow()
window.resize(1440,840)
window.show()
window.set_report({'elapsed_seconds':2,'history':[
    {'at_ms':0,'sbi':80,'cues':['疑似学历偏见']},
    {'at_ms':2000,'sbi':70,'cues':['嘴角下压增强']},
]})
window._show_page(1)
app.processEvents()
point = window.timeline._position(2,50)
QTest.mouseMove(window.timeline,QPoint(round(point.x()),round(point.y())))
app.processEvents()
assert '嘴角下压增强' in window.timeline.toolTip()
assert '学历' not in window.timeline.toolTip()
window.close()
''')

    def test_red_peaks_paint_different_reasons_not_just_different_tooltip_metadata(self):
        self.run_qt('''
chart = SessionTimeline()
chart.resize(900,300)
chart.show()
images = []
for cause in ['疑似学历偏见','眉部紧张增强']:
    chart.set_history([{'at_ms':1000,'sbi':75,'cues':[cause]}], 2)
    app.processEvents()
    point = chart._position(1,50)
    QTest.mouseMove(chart,QPoint(round(point.x()),round(point.y())))
    app.processEvents()
    images.append(chart.grab().toImage())
assert images[0] != images[1], 'The painted hover card must change with the peak label'
chart.close()
''')


if __name__ == '__main__':
    unittest.main()
