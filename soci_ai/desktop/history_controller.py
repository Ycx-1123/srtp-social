"""Pure presentation/lifecycle coordination, independent of Qt and devices."""
from copy import deepcopy
from time import monotonic
from uuid import uuid4


class HistoryController:
    def __init__(self, worker, clock=monotonic, *, auto_save=True):
        self.worker, self.clock = worker, clock
        self.auto_save = auto_save
        self._save_requested = set()
        self.active_id = self.selected_id = None
        self.items, self.has_more = [], False
        self.selected_report, self._current = {}, {}
        self._selected_item = None
        self.save_error = self.load_error = ''
        self._load_error_operation = ''
        self._service_error = ''
        self._recovery_needed = False
        self._replay_after_recovery = False
        self._running = False
        self._sequence = 0
        self._requests = {}
        self._records, self._created, self._failed = {}, set(), set()
        self._awaiting_final = set()
        self._load_request = self._list_request = None
        self._last_checkpoint = 0.
        self._initial_list = True

    @property
    def save_status(self):
        if self.save_error:
            return '保存失败'
        if not self.auto_save and self.active_id in self._records and self.active_id not in self._save_requested:
            return '结束后可选择保存' if self._running else '本次会话尚未保存'
        return '保存中' if self.pending_writes() else '已保存'

    def _send(self, operation, **payload):
        self._sequence += 1
        key = self._sequence
        self._requests[key] = (operation, payload)
        self.worker.submit(operation, key, **payload)
        return key

    def begin(self, report: dict, started_at: str, app_version: str) -> str:
        if self._running or self.needs_save_choice():
            raise RuntimeError('请先结束当前会话，并选择保存或不保存。')
        self.active_id = key = str(uuid4())
        self._running = True
        self._last_checkpoint = self.clock()
        self._current = report
        if self.selected_id is None:
            self.selected_report = report
        self._records[key] = dict(started_at=started_at, app_version=app_version, report=deepcopy(report), ended_at=None)
        if self.auto_save:
            self._send('create', session_id=key, started_at=started_at, app_version=app_version, report=report)
        return key

    def update_current(self, report: dict) -> None:
        if not self._running:
            return
        self._current = report
        if self.selected_id is None:
            self.selected_report = report
        if self.auto_save and self.clock() - self._last_checkpoint >= 5:
            self._last_checkpoint = self.clock()
            self._records[self.active_id]['report'] = deepcopy(report)
            self._send('checkpoint', session_id=self.active_id, report=report)

    def finish_current(self, report: dict, ended_at: str) -> None:
        if not self._running:
            return
        self._running = False
        self._current = report
        if self.selected_id is None:
            self.selected_report = report
        record = self._records[self.active_id]
        record.update(report=deepcopy(report), ended_at=ended_at)
        if self.auto_save:
            self._send('finish', session_id=self.active_id, report=report, ended_at=ended_at)

    def needs_save_choice(self) -> bool:
        return (not self.auto_save and not self._running and self.active_id in self._records
                and self.active_id not in self._save_requested)

    def save_current(self) -> bool:
        if not self.needs_save_choice():
            return False
        key = self.active_id
        record = self._records[key]
        self._save_requested.add(key)
        self._awaiting_final.add(key)
        self._send('create', session_id=key, started_at=record['started_at'],
                   app_version=record['app_version'], report=record['report'])
        return True

    def discard_current(self) -> bool:
        if not self.needs_save_choice():
            return False
        self._records.pop(self.active_id)
        # Current report remains available for review/export until the next run.
        return True

    def select(self, session_id: str | None) -> None:
        retained = self._selected_item if self.selected_id == session_id else None
        self._selected_item = next((dict(row) for row in self.display_items() if row['id'] == session_id), retained)
        self.selected_id = session_id
        self.load_error = ''
        self._load_error_operation = ''
        self._load_request = None
        self.selected_report = self._current if session_id is None else {}
        if session_id is not None:
            if session_id in self._failed and session_id in self._records:
                self.selected_report = deepcopy(self._records[session_id]['report'])
            else:
                self._load_request = self._send('load', session_id=session_id)

    def display_items(self) -> list[dict]:
        """Visible metadata, including explicitly unsaved RAM-only snapshots.

        self.items remains the actual paged DB rows, so extra UI anchors do not
        advance pagination and silently skip records.
        """
        rows = {row['id']: dict(row) for row in self.items}
        for key in self._failed:
            record = self._records.get(key)
            if record:
                report = record['report']
                rows[key] = dict(id=key, started_at=record['started_at'], ended_at=record['ended_at'],
                    status='unsaved', elapsed_seconds=report.get('elapsed_seconds', 0),
                    average_sbi=report.get('average_sbi'), peak_sbi=report.get('peak_sbi'))
        if self._selected_item and self.selected_id not in rows:
            rows[self.selected_id] = dict(self._selected_item)
        return sorted(rows.values(), key=lambda row: (str(row.get('started_at', '')), row['id']), reverse=True)

    def refresh_history(self, offset: int = 0) -> None:
        self._list_request = self._send('list', offset=offset, limit=100)

    def delete_selected(self) -> None:
        row = next((row for row in self.items if row['id'] == self.selected_id), self._selected_item)
        if row and row['status'] in ('completed', 'interrupted') and not (self._running and row['id'] == self.active_id):
            self._send('delete', session_id=row['id'])

    def retry_failed(self) -> None:
        self._send('retry')

    def _replay(self):
        for key in list(self._failed):
            record = self._records.get(key)
            if not record:
                continue
            if key not in self._created:
                if record['ended_at']:
                    self._awaiting_final.add(key)
                self._send('create', session_id=key, started_at=record['started_at'],
                           app_version=record['app_version'], report=record['report'])
            elif record['ended_at']:
                self._send('finish', session_id=key, report=record['report'], ended_at=record['ended_at'])
            else:
                self._send('checkpoint', session_id=key, report=record['report'])

    def handle_results(self) -> bool:
        changed = False
        for result in self.worker.drain_results():
            request = result['request_id']
            pending = self._requests.pop(request, None)
            if pending is None:
                continue
            op, payload = pending
            key, value = payload.get('session_id'), result.get('value')
            changed = True
            if not result['ok']:
                error = result.get('error') or '无法保存历史记录。'
                if op in ('create', 'checkpoint', 'finish'):
                    self._failed.add(key)
                    self.save_error = error
                elif op == 'load' and request == self._load_request:
                    self.selected_report = {}
                    self.load_error = error
                    self._load_error_operation = op
                elif op in ('recover', 'retry'):
                    self._service_error = self.save_error = error
                    if op == 'recover':
                        self._recovery_needed = True
                elif op == 'delete' or (op == 'list' and request == self._list_request):
                    self.load_error = error
                    self._load_error_operation = op
                continue
            if op == 'create':
                self._created.add(key)
                if key in self._awaiting_final:
                    self._awaiting_final.remove(key)
                    record = self._records[key]
                    self._send('finish', session_id=key, report=record['report'], ended_at=record['ended_at'])
                elif not self._records[key]['ended_at']:
                    self._failed.discard(key)
                self.refresh_history()
            elif op == 'checkpoint' and value is not None and value != 'superseded':
                if value is True:
                    self._failed.discard(key)
            elif op == 'finish':
                self._failed.discard(key)
                self._records.pop(key, None)
                self.refresh_history()
            elif op == 'retry':
                if self._recovery_needed:
                    self._replay_after_recovery = True
                    self._send('recover', exclude_session_ids=list(self._records))
                else:
                    self._service_error = ''
                    self._replay()
                    self.refresh_history()
            elif op == 'load' and request == self._load_request:
                self.selected_report = value['report'] if value else {}
                if value and value.get('id'):
                    self._selected_item = {name: item for name, item in value.items() if name != 'report'}
                self.load_error = '' if value else '本条历史已不存在，请刷新列表。'
                self._load_error_operation = '' if value else 'load'
            elif op == 'list' and request == self._list_request:
                if self._load_error_operation == 'list':
                    self.load_error = self._load_error_operation = ''
                rows = value['items']
                if value['offset'] == 0:
                    self.items = rows
                else:
                    seen = {row['id'] for row in self.items}
                    self.items.extend(row for row in rows if row['id'] not in seen)
                self.has_more = value['has_more']
                if self._initial_list:
                    self._initial_list = False
                    if self.active_id is None and self.selected_id is None and self.items:
                        self.select(self.items[0]['id'])
            elif op == 'delete':
                position = next((i for i, row in enumerate(self.items) if row['id'] == key), len(self.items))
                self.items = [row for row in self.items if row['id'] != key]
                if key == self.active_id and not self._running:
                    self.active_id, self._current = None, {}
                if self.selected_id == key:
                    self.select(self.items[min(position, len(self.items)-1)]['id'] if self.items else None)
                self.load_error = ''
                self._load_error_operation = ''
                self.refresh_history()
            elif op == 'recover':
                self._recovery_needed = False
                self._service_error = ''
                if self._replay_after_recovery:
                    self._replay_after_recovery = False
                    self._replay()
                self.refresh_history()
            if not self._failed:
                self.save_error = self._service_error
        return changed

    def export_report(self) -> dict:
        return deepcopy(self.selected_report)

    def pending_writes(self) -> bool:
        return any(op in ('create', 'checkpoint', 'finish', 'retry', 'recover') for op, _ in self._requests.values())

    def has_unsaved_sessions(self) -> bool:
        return bool(self._failed)
