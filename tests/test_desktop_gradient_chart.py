import os
import subprocess
import sys
import unittest


class GradientChartRenderingTests(unittest.TestCase):
    def test_only_sbi_is_painted_and_stroke_is_thick_and_value_colored(self):
        code = '''
import sys
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.charts import SessionTimeline
app = QApplication([])
chart = SessionTimeline()
chart.resize(900,300)
chart.show()
history = [{'at_ms':i*1000,'sbi':v,'friendliness':None if v is None else 100-v} for i,v in enumerate([10,10,10,None,80,80,80])]
chart.set_history(history, 6)
app.processEvents()
first = chart.grab().toImage()
chart.set_history([{**x,'friendliness':0} for x in history], 6)
app.processEvents()
assert first == chart.grab().toImage(), 'Changing the complementary metric must not draw a second line'
gap_x = round(chart._position(3,0).x())
for y in range(round(chart.plot_rect().top()),round(chart.plot_rect().bottom())):
    color = first.pixelColor(gap_x,y)
    assert not (color.red()>180 and color.red()>color.green()*1.3), 'No risk trace may bridge missing samples'
    assert not (color.green()>150 and color.green()>color.red()*1.2), 'No measured trace may bridge missing samples'
for at, value, mode in [(1,10,'green'),(5,80,'red')]:
    point = chart._position(at,value)
    colors = [first.pixelColor(round(point.x()),round(point.y())+offset) for offset in range(-4,5)]
    if mode == 'green':
        count = sum(c.green()>150 and c.green()>c.red()*1.2 for c in colors)
    else:
        count = sum(c.red()>180 and c.red()>c.green()*1.3 for c in colors)
    assert count >= 4, (mode,count,[(c.red(),c.green(),c.blue()) for c in colors])
chart.close()
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
