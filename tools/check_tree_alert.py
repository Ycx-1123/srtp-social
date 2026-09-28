"""Render controlled UI inputs offscreen; never open a camera or microphone."""
from pathlib import Path
import json
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtGui import QFont, QFontDatabase, QImage
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.tree import LivingTree
from soci_ai.desktop.ui import StudioWindow


def settle(widget, steps=70):
    for _ in range(steps):
        widget._last_frame = time.monotonic() - 1 / 30
        widget._animate()
    app.processEvents()


def red_pixels(widget):
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    raw = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
    rgb = raw.reshape(image.height(), image.bytesPerLine())[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
    rgb = rgb[85:-80, 25:-25].astype(float)
    return int(((rgb[:, :, 0] > 80) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.4) &
                (rgb[:, :, 0] > rgb[:, :, 2] * 1.1)).sum())


if __name__ == "__main__":
    app = QApplication([])
    for font_file in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(font_file).is_file():
            QFontDatabase.addApplicationFont(font_file)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    output = ROOT / "artifacts/native-red-tree"
    output.mkdir(parents=True, exist_ok=True)
    tree = LivingTree()
    tree.resize(520, 620)
    tree.show()
    app.processEvents()
    tree._timer.stop()
    tree.set_state({"mode": "friendly", "risk": .08, "health": .932, "bloom": .8}, 8)
    settle(tree)
    friendly_red = red_pixels(tree)
    assert tree.grab().save(str(output / "friendly.png"))
    tree.set_state({"mode": "signal", "risk": .599, "health": .49, "bloom": 0}, 59.9)
    assert not tree._target["alert"]
    tree.set_state({"mode": "risk", "risk": .6, "health": .49, "bloom": 0}, 60)
    assert tree._target["alert"]
    # About 0.4 seconds of animation ticks should already make the tree red.
    settle(tree, 12)
    onset_red = red_pixels(tree)
    assert onset_red > friendly_red + 2500, (friendly_red, onset_red)
    settle(tree)
    assert tree.grab().save(str(output / "risk-threshold.png"))
    tree.set_state({"mode": "risk", "risk": .86, "health": .269, "bloom": 0}, 86)
    settle(tree)
    assert tree.grab().save(str(output / "risk-strong.png"))
    tree.set_state({"mode": "friendly", "risk": .08, "health": .932, "bloom": .8}, 8)
    settle(tree)
    recovered_red = red_pixels(tree)
    assert recovered_red < onset_red / 4, (onset_red, recovered_red)
    tree.close()

    window = StudioWindow()
    reset_signals = []
    window.calibrate_requested.connect(lambda: reset_signals.append(True))
    running = {"status": "running", "sbi": 60, "friendliness": 40,
               "tree": {"mode": "risk", "risk": .6, "health": .49, "bloom": 0},
               "vision": {"face_detected": True, "features": {"baseline_ready": False}},
               "transcript": {"text": "离屏界面检查 · 测试输入", "status": "测试输入"}}
    window.set_state(running)
    window.calibrate_button.click()
    assert not reset_signals, "Reset must wait until the current calibration finishes"
    running["vision"]["features"]["baseline_ready"] = True
    window.set_state(running)
    window.calibrate_button.click()
    assert len(reset_signals) == 1
    for width, height in ((1280, 720), (960, 600), (760, 480)):
        window.resize(width, height)
        window.show()
        app.processEvents()
        window.tree._timer.stop()
        settle(window.tree)
        assert window.live_scroll.horizontalScrollBar().maximum() == 0
        assert window.grab().save(str(output / f"window-{width}x{height}.png"))
    window.close()
    print(json.dumps({"ok": True, "file_only": True, "camera_opened": False,
                      "microphone_opened": False, "friendly_red_pixels": friendly_red,
                      "risk_onset_red_pixels": onset_red, "recovered_red_pixels": recovered_red,
                      "sizes_checked": ["1280x720", "960x600", "760x480"]}))
