"""Offscreen presentation QA using controlled text, no camera or microphone."""
from pathlib import Path
import json
import os
import sys
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.ui import StudioWindow
from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.language import analyze_text


if __name__ == '__main__':
    app = QApplication([])
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/segoeui.ttf'):
        if Path(filename).is_file():
            QFontDatabase.addApplicationFont(filename)
    app.setFont(QFont('Microsoft YaHei UI', 10))
    output = ROOT / 'artifacts/native-language-feedback'
    output.mkdir(parents=True, exist_ok=True)
    window = StudioWindow()
    rows = json.loads((ROOT / 'resources/language/social_bias_examples.json').read_text(encoding='utf-8'))['examples']
    seen, tags = set(), set()
    for row in rows:
        if row['label'] != 'bias':
            continue
        result = analyze_text(row['text'])
        tags.update(result['tags'])
        if row['category'] in seen:
            continue
        seen.add(row['category'])
        engine = RealtimeAssessment()
        engine.observe_text(row['text'], 0, True)
        state = None
        for index in range(11):
            state = engine.update(index / 10)
        window.resize(1280, 900)
        window.show()
        window.set_state({**state, 'status': 'running', 'transcript': {'text': row['text'], 'status': '离屏测试输入'}})
        app.processEvents()
        assert window.feedback_card.isVisible()
        assert result['category_label'] in window.feedback_title.text()
        assert window.grab().save(str(output / (row['category'] + '.png')))
    for phrase, prefix in (('学历低一点的人更适合跑腿', 'education'), ('学历低一点的人更适合跑退', 'review')):
        engine = RealtimeAssessment()
        engine.observe_text(phrase, 0, True)
        for index in range(11):
            state = engine.update(index / 10)
        for width, height in ((1280, 720), (960, 600), (760, 480)):
            window.resize(width, height)
            window.set_state({**state, 'status': 'running', 'transcript': {'text': phrase, 'status': '离屏测试输入'}})
            app.processEvents()
            window.tree._timer.stop()
            for _ in range(70):
                window.tree._last_frame = time.monotonic() - 1 / 30
                window.tree._animate()
            app.processEvents()
            assert window.live_scroll.horizontalScrollBar().maximum() == 0
            assert window.feedback_card.geometry().bottom() <= window.camera_card.geometry().top()
            assert window.grab().save(str(output / f'{prefix}-{width}x{height}.png'))
    window.close()
    print(json.dumps({'ok': True, 'controlled_text_only': True, 'camera_opened': False,
                      'microphone_opened': False, 'categories_checked': len(seen), 'distinct_tags': len(tags)}))
