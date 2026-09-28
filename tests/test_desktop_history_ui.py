import unittest
from tests import test_desktop_hover_ui as qt_helpers


class HistoryUITests(unittest.TestCase):
    run_qt = qt_helpers.HoverPresentationTests.run_qt
    def test_history_select_clears_old_report_and_refresh_does_not_emit(self):
        self.run_qt('''
w = StudioWindow()
w.show()
w._show_page(1)
seen = []
w.history_selected.connect(seen.append)
items = [{'id':'a','started_at':'2026-09-27T09:00:00+08:00','status':'completed','elapsed_seconds':60,'average_sbi':20,'peak_sbi':80},
         {'id':'b','started_at':'2026-09-27T10:00:00+08:00','status':'interrupted','elapsed_seconds':30,'average_sbi':10,'peak_sbi':60}]
w.set_history_items(items,'a',has_more=True)
assert seen == []
assert '已结束' in w.history_combo.itemText(1)
assert '中断' in w.history_combo.itemText(2)
assert w.history_more.isVisibleTo(w)
w.set_report({'average_sbi':20,'elapsed_seconds':60})
w.history_combo.setCurrentIndex(2)
assert seen == ['b']
assert w.displayed_report() == {} and not w.export_button.isEnabled()
w.set_report({'average_sbi':10,'elapsed_seconds':30})
assert w.displayed_report()['average_sbi']==10
w.set_history_items(items,'b',has_more=False)
assert seen==['b']
assert not w.history_more.isVisibleTo(w)
w.set_report({})
w.set_history_items([],None)
assert not w.history_delete.isEnabled()
assert not w.export_button.isEnabled()
w.close()
''')

    def test_delete_confirmation_and_running_guard(self):
        self.run_qt('''
from unittest.mock import patch
from PySide6.QtWidgets import QMessageBox
w=StudioWindow()
w._show_page(1)
seen=[]
w.history_delete_requested.connect(seen.append)
items=[{'id':'a','started_at':'2026-09-27T09:00:00+08:00','status':'running','elapsed_seconds':60}]
w.set_history_items(items,'a')
assert not w.history_delete.isEnabled()
items[0]['status']='completed'
w.set_history_items(items,'a')
with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):
    w.history_delete.click()
assert seen==[]
with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes) as confirm:
    w.history_delete.click()
    assert '09:00' in confirm.call_args.args[2]
    assert '已导出' in confirm.call_args.args[2]
assert seen==['a']
w.set_history_status('保存失败 · 磁盘不可写',error=True)
assert w.history_retry.isVisibleTo(w)
w.close()
''')

    def test_refresh_keeps_visible_selected_row_outside_current_page(self):
        self.run_qt('''
w=StudioWindow()
w._show_page(1)
items=[{'id':str(i),'started_at':'2026-09-27T09:00:00+08:00','status':'completed','elapsed_seconds':60} for i in range(205)]
w.set_history_items(items,'204',has_more=False)
w.set_report({'average_sbi':77,'elapsed_seconds':60})
w.set_history_items(items[:100],'204',has_more=True)
assert w.history_combo.currentData() == '204'
assert '当前会话' not in w.history_combo.currentText()
assert w.displayed_report()['average_sbi']==77
assert w.history_delete.isEnabled()
w.set_history_items(items[:100],None,has_more=True)
assert w.history_combo.currentData() is None
w.close()
''')
