"""Offscreen visual QA; synthetic examples never enter the history store."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from soci_ai.desktop.ui import StudioWindow


def main():
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    output = Path(__file__).resolve().parents[1] / 'artifacts/native-history-ui'
    output.mkdir(parents=True, exist_ok=True)
    w = StudioWindow()
    w.set_history_items([dict(id='qa-only', started_at='2026-09-27T09:00:00+08:00',
        status='completed', elapsed_seconds=120, average_sbi=23.5, peak_sbi=78)], 'qa-only')
    w.set_history_status('已保存 · 本机历史（离屏界面测试）')
    w.set_report(dict(elapsed_seconds=120, average_sbi=23.5, peak_sbi=78,
        history=[dict(at_ms=i*1000, sbi=78 if 40 <= i <= 45 else 8+(i%15),
                      cues=['疑似学历偏见'] if 40 <= i <= 45 else []) for i in range(121)],
        advice=[dict(title='语言表达', body='按经验、能力与本人意愿分工。')],
        transcripts=[dict(at_ms=41000,text='测试字幕，不是真实会话')],
        limitations='离屏视觉测试，未采集任何摄像头或麦克风信号。'))
    w.show()
    for width,height in ((1440,840),(1280,720),(960,600)):
        w.resize(width,height)
        for page, name in ((1,'review'),(0,'live')):
            w._show_page(page)
            app.processEvents()
            path = output / f'{name}-{width}x{height}.png'
            assert w.grab().save(str(path))
            print(path)
    w.close()


if __name__ == '__main__':
    main()
