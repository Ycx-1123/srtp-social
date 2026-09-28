"""Offscreen synthetic-input QA; never opens devices or writes user history."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.ui import StudioWindow


def main():
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    output = Path(__file__).resolve().parents[1] / 'artifacts/layered-guidance-ui'
    output.mkdir(parents=True, exist_ok=True)
    model = RealtimeAssessment()
    for i in range(101):
        at = i / 10
        state = model.update(at,
            vision={'captured_at':at, 'face_detected':True, 'features':{
                'baseline_ready':True, 'brow_tension':.6, 'mouth_downturn':.4,
                'micro_expression':.6}},
            audio={'captured_at':at, 'pressure':.6, 'rms':.15, 'status':'离屏测试'})
    state['status'] = 'running'
    state['transcript'] = {'text':'离屏布局测试，未采集摄像头或麦克风。', 'is_final':True}
    window = StudioWindow()
    window.show()
    for width, height in ((1440,840),(1280,720),(960,600),(760,480),(696,416)):
        window.setMinimumSize(min(760,width), min(480,height))
        window.resize(width,height)
        window.set_state(state)
        for _ in range(5):
            app.processEvents()
        assert window.live_scroll.verticalScrollBar().maximum() == 0, (width,height)
        path = output / f'live-{width}x{height}.png'
        assert window.grab().save(str(path))
        print(path)
    window.close()


if __name__ == '__main__':
    main()
