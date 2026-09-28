"""No-device offscreen QA. Test fixtures are not project performance results."""
from pathlib import Path
import json
import math
import os
import sys
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.ui import StudioWindow


if __name__ == '__main__':
    app = QApplication([])
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/segoeui.ttf'):
        if Path(filename).is_file():
            QFontDatabase.addApplicationFont(filename)
    app.setFont(QFont('Microsoft YaHei UI', 10))
    output = ROOT / 'artifacts/native-compact-ui'
    output.mkdir(parents=True, exist_ok=True)
    engine = RealtimeAssessment()
    engine.observe_text('学历低一点的人更适合跑腿', 0, True)
    state = {**engine.update(.3), 'status': 'running', 'elapsed_ms': 72000,
             'vision': {'face_detected': True, 'status': 'detected', 'features': {
                 'baseline_ready': True, 'brow_tension': .43, 'lip_tension': .15, 'mouth_downturn': .08}},
             'audio': {'rms': .15, 'status': '麦克风采集中'},
             'transcript': {'text': '学历低一点的人更适合跑腿', 'status': '已确认', 'is_final': True}}
    window = StudioWindow()
    window.show()
    checks = []
    for width, height in ((1440,840), (1280,720), (960,600), (760,480), (696,416)):
        window.setMinimumSize(min(760,width),min(480,height))
        window.resize(width,height)
        for name, snapshot in (('neutral', {'status':'running'}), ('alert', state)):
            window.set_state(snapshot)
            for _ in range(3): app.processEvents()
            viewport = window.live_scroll.viewport()
            assert window.live_scroll.verticalScrollBar().maximum() == 0, (width,height)
            assert window.live_scroll.horizontalScrollBar().maximum() == 0, (width,height)
            for widget in (window.feedback_card, window.camera_card, window.tree, window.score_card, window.voice_card):
                pos = widget.mapTo(viewport,QPoint(0,0))
                assert viewport.rect().contains(pos)
                assert viewport.rect().contains(pos+QPoint(widget.width()-1,widget.height()-1))
            window.tree._timer.stop()
            for _ in range(70):
                window.tree._last_frame = time.monotonic()-1/30
                window.tree._animate()
            assert window.grab().save(str(output/f'{name}-{width}x{height}.png'))
            checks.append({'size':[width,height],'state':name,'overflow':False})
    # Explicit controlled chart fixture, not fabricated recording/report data.
    history = []
    for i in range(361):
        at = i*.5
        value = min(100, 5+3*math.sin(at/8)**2 +
                    35*math.exp(-((at-38)/7)**2) +
                    77*math.exp(-((at-90)/10)**2) +
                    62*math.exp(-((at-138)/5)**2))
        history.append({'at_ms':round(at*1000),'sbi':None if 62<at<66 else value})
    events = [{'at_ms':i*1000,'modality':'vision','value':.8,'title':'唇部收紧增强','evidence':'动作变化'} for i in (8,15,26,34,40)]
    events += engine.events
    from soci_ai.desktop.advice import key_moments
    window.resize(1440,840)
    window.set_report({'elapsed_seconds':180, 'average_sbi':25.1, 'peak_sbi':82,
                       'history':history,'moments':key_moments(events),
                       'transcripts':engine.transcripts})
    window._show_page(1)
    for _ in range(3): app.processEvents()
    assert window.grab().save(str(output/'review.png'))
    assert window.timeline.grab().save(str(output/'gradient-chart.png'))
    window._show_page(2)
    for _ in range(3): app.processEvents()
    assert window.grab().save(str(output/'introduction.png'))
    window.mechanism.verticalScrollBar().setValue(450)
    app.processEvents()
    assert window.grab().save(str(output/'formulas.png'))
    window.close()
    print(json.dumps({'ok':True,'controlled_fixtures_only':True,'camera_opened':False,'microphone_opened':False,'checks':checks},ensure_ascii=False))
