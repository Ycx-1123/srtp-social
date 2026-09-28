"""Isolated, no-device checks of the native presentation contract."""
import os
import subprocess
import sys
import unittest


class NativeReportUiTests(unittest.TestCase):
    def test_live_report_refresh_preserves_expanded_transcripts(self):
        code = '''
import sys
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from PySide6.QtWidgets import QApplication
from soci_ai.desktop.ui import StudioWindow
app = QApplication([])
window = StudioWindow()
window.tree._timer.stop()
report = {"elapsed_seconds": 1, "transcripts": [{"at_ms": 0, "text": "test"}], "history": []}
window.set_report(report)
window.transcripts_toggle.setChecked(True)
window.set_report({**report, "elapsed_seconds": 2})
assert window.transcripts_toggle.isChecked(), "Periodic refresh must not collapse the user's open transcript"
assert not window.review.isHidden()
window.set_report({})
assert not window.transcripts_toggle.isChecked()
assert window.review.isHidden()
window.close()
'''
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
