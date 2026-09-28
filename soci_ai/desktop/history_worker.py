"""One database owner thread; coalesced checkpoints and ordered controls."""
from collections import deque
from copy import deepcopy
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Condition, Thread

from .history_store import HistoryStore


class HistoryWorker(Thread):
    def __init__(self, path: Path, store_factory=HistoryStore):
        super().__init__(name='soci-history', daemon=False)
        self.path, self.store_factory = path, store_factory
        self._condition = Condition()
        self._controls, self._checkpoints = deque(), {}
        self._results = SimpleQueue()
        self._stopping = False
        self._store = None
        self._open_error = None

    def _result(self, item, *, value=None, error=None):
        op, request, data = item
        self._results.put(dict(operation=op, request_id=request, session_id=data.get('session_id'),
                               ok=error is None, value=value, error='' if error is None else str(error)))

    def submit(self, operation: str, request_id: int, **payload) -> None:
        item = (operation, request_id, deepcopy(payload))
        with self._condition:
            if self._stopping:
                self._result(item, error=RuntimeError('历史服务正在关闭，尚未保存。'))
                return
            if operation == 'checkpoint':
                key = payload['session_id']
                old = self._checkpoints.get(key)
                if old:
                    self._result(old, value='superseded')
                self._checkpoints[key] = item
            else:
                if operation in ('finish', 'delete'):
                    old = self._checkpoints.pop(payload['session_id'], None)
                    if old:
                        self._result(old, value='superseded')
                self._controls.append(item)
            self._condition.notify()

    def drain_results(self) -> list[dict]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return results

    def request_shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self):
        try:
            while True:
                with self._condition:
                    while not self._controls and not self._checkpoints and not self._stopping:
                        self._condition.wait()
                    if self._controls:
                        item = self._controls.popleft()
                    elif self._checkpoints:
                        key = next(iter(self._checkpoints))
                        item = self._checkpoints.pop(key)
                    else:
                        break
                try:
                    op, _, data = item
                    if op == 'retry':
                        if self._store:
                            self._store.close()
                        self._store = None
                        self._open_error = None
                    if self._store is None:
                        if self._open_error:
                            raise self._open_error
                        try:
                            self._store = self.store_factory(self.path)
                        except Exception as exc:
                            self._open_error = exc
                            raise
                    if op == 'retry':
                        value = True
                    elif op == 'list':
                        offset, limit = data.get('offset', 0), data.get('limit', 100)
                        rows = self._store.list_sessions(offset, limit+1)
                        value = dict(items=rows[:limit], has_more=len(rows)>limit, offset=offset)
                    else:
                        method = {'create': 'create_session', 'checkpoint': 'checkpoint', 'finish': 'finish',
                                  'load': 'load_session', 'delete': 'delete_session', 'recover': 'recover_interrupted'}.get(op)
                        if not method:
                            raise ValueError('未知历史操作。')
                        value = getattr(self._store, method)(**data)
                        if op in ('finish', 'delete') and value is False:
                            raise ValueError('记录不存在或状态不允许此操作；原数据未改变。')
                    self._result(item, value=value)
                except Exception as exc:
                    self._result(item, error=exc)
        finally:
            if self._store:
                self._store.close()
