import unittest
from tests import test_desktop_hover_ui as qt_helpers


class NoticeUITests(unittest.TestCase):
    run_qt = qt_helpers.HoverPresentationTests.run_qt
    def test_bundled_notices_readable_from_system_intro(self):
        self.run_qt('''
w=StudioWindow()
w._show_page(2)
w.notice_button.click()
app.processEvents()
assert w.document_dialog.isVisible()
assert 'PySide6' in w.document_browser.toPlainText()
assert 'GNU LESSER GENERAL PUBLIC LICENSE' in w.document_browser.toPlainText()
w.document_dialog.close()
w.close()
''')
