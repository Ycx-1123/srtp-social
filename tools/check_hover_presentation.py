"""No-device visual QA with controlled inputs, never a user-session result."""
from pathlib import Path
import json
import os
import sys

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.ui import StudioWindow


if __name__ == '__main__':
    app = QApplication([])
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/segoeui.ttf'):
        if Path(filename).is_file():
            QFontDatabase.addApplicationFont(filename)
    app.setFont(QFont('Microsoft YaHei UI', 10))
    output = ROOT / 'artifacts/native-hover-ui'
    output.mkdir(parents=True, exist_ok=True)
    engine = RealtimeAssessment()
    for i in range(161):
        at = i / 10
        if i == 20:
            engine.observe_text('学历低一点的人更适合跑腿', at, True)
        if i == 40:
            engine.observe_text('我们按具体经验来分工', at, True)
        tense = 9 <= at <= 10.5
        engine.update(at, vision={'face_detected': True, 'captured_at': at,
                                 'features': {'baseline_ready': True,
                                              'brow_tension': .9 if tense else 0,
                                              'micro_expression': .9 if tense else 0}},
                      audio={'captured_at': at, 'pressure': .8 if tense else 0})
    window = StudioWindow()
    window.resize(1440,840)
    window.show()
    window.tree._timer.stop()
    window.set_report(engine.report(16, {}, {}))
    window._show_page(1)
    app.processEvents()
    labels = {}
    for name, at, expected in [('language', 2.7, '疑似学历偏见'),
                               ('face-voice', 9.7, '眉部紧张增强')]:
        point = window.timeline._position(at, 70)
        QTest.mouseMove(window.timeline, QPoint(round(point.x()),round(point.y())))
        app.processEvents()
        text = window.timeline.toolTip()
        assert expected in text, text
        assert '学历低一点的人更适合跑腿' not in text
        assert window.grab().save(str(output / f'hover-{name}.png'))
        labels[name] = text
    camera = window.camera
    window._show_page(0)
    window.set_state({'status':'running'})
    camera.set_frame(np.full((300,400,3), 18, dtype=np.uint8),
                     {'x':.25,'y':.2,'width':.3,'height':.5})
    app.processEvents()
    assert camera.grab().save(str(output/'bare-tracking-outline.png'))
    window.close()
    print(json.dumps({'ok':True, 'controlled_fixtures_only':True,
                      'camera_opened':False, 'microphone_opened':False,
                      'hover_labels':labels}, ensure_ascii=False))
