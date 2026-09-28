"""Versioned, account-local session snapshots. No Qt or device dependencies."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


REPORT_FIELDS = frozenset(('elapsed_seconds', 'average_sbi', 'peak_sbi', 'history',
    'transcripts', 'events', 'telemetry', 'model_status', 'observation_summary',
    'advice', 'moments', 'language_corpus', 'provenance', 'limitations'))
MEDIA_FIELDS = frozenset(('frame', 'frame_bgr', 'image', 'photo', 'video', 'audio',
    'raw_audio', 'raw_video', 'samples', 'waveform', 'stream', 'partial_text'))
SUMMARY_COLUMNS = ('id', 'started_at', 'ended_at', 'updated_at', 'status',
    'elapsed_seconds', 'average_sbi', 'peak_sbi', 'app_version')


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key in MEDIA_FIELDS for key in value):
            raise ValueError('会话报告包含媒体或临时转写，不能保存。')
        return {key: _plain(item) for key, item in value.items()}
    raise ValueError('会话报告必须只包含有限数值、文字和结构化线索。')


def clean_report(report: dict) -> dict:
    if not isinstance(report, dict):
        raise ValueError('会话报告必须是结构化对象。')
    data = {key: value for key, value in report.items() if key in REPORT_FIELDS}
    for key in ('history', 'events', 'transcripts', 'advice', 'moments'):
        if key in data and (not isinstance(data[key], list) or
                            any(not isinstance(item, dict) for item in data[key])):
            raise ValueError('会话报告的 ' + key + ' 结构损坏。')
    for key in ('telemetry', 'model_status', 'observation_summary', 'language_corpus'):
        if key in data and not isinstance(data[key], dict):
            raise ValueError('会话报告的 ' + key + ' 必须是对象。')
    for key in ('elapsed_seconds', 'average_sbi', 'peak_sbi'):
        value = data.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('会话报告的 ' + key + ' 必须是有限数值。')
    for item in data.get('moments', []):
        if ('start_ms' in item) != ('end_ms' in item):
            raise ValueError('回看时刻的时间区间不完整。')
        for key in ('start_ms', 'end_ms'):
            if key in item and (type(item[key]) not in (int, float) or not math.isfinite(item[key])):
                raise ValueError('回看时刻的时间区间损坏。')
    if 'transcripts' in data:
        data['transcripts'] = [item for item in data['transcripts']
                              if isinstance(item, dict) and item.get('is_final', True) is True]
    return _plain(data)


def _iso(value=None):
    stamp = datetime.fromisoformat(value) if value else datetime.now().astimezone()
    if stamp.tzinfo is None:
        raise ValueError('会话时间必须包含时区。')
    return stamp.isoformat()


class HistoryStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=.25)
        self.db.row_factory = sqlite3.Row
        try:
            version = self.db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('历史数据库来自较新或不支持的版本；已保留原文件，请更新软件。')
            tables = {row[0] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if version == 0 and tables:
                raise ValueError('无法识别旧历史格式；已保留原文件，未自动重建。')
            if version == 0:
                with self.db:
                    # sqlite3's legacy implicit transactions exclude DDL. Start
                    # explicitly so a later CREATE/PRAGMA failure rolls back all.
                    self.db.execute('BEGIN IMMEDIATE')
                    self.db.execute('''CREATE TABLE sessions (
                        id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
                        updated_at TEXT NOT NULL, status TEXT NOT NULL
                        CHECK(status IN ('running','completed','interrupted')),
                        elapsed_seconds REAL NOT NULL, average_sbi REAL, peak_sbi REAL,
                        app_version TEXT NOT NULL, report_version INTEGER NOT NULL,
                        report_json TEXT NOT NULL)''')
                    self.db.execute('CREATE INDEX session_date ON sessions(started_at DESC, id DESC)')
                    self.db.execute('PRAGMA user_version=1')
            # Validate existing schema read-only before accepting any writes.
            self.db.execute('SELECT ' + ','.join(SUMMARY_COLUMNS) + ',report_version,report_json FROM sessions LIMIT 0')
        except Exception:
            self.db.close()
            raise

    @staticmethod
    def _snapshot(report):
        report = clean_report(report)
        return (max(0., float(report.get('elapsed_seconds') or 0)), report.get('average_sbi'),
                report.get('peak_sbi'), json.dumps(report, ensure_ascii=False, allow_nan=False))

    def create_session(self, session_id: str, started_at: str, report: dict, app_version: str) -> None:
        elapsed, average, peak, body = self._snapshot(report)
        with self.db:
            self.db.execute('''INSERT INTO sessions VALUES (?,?,NULL,?,'running',?,?,?,?,1,?)
                               ON CONFLICT(id) DO NOTHING''',
                            (session_id, _iso(started_at), _iso(), elapsed, average, peak, app_version, body))

    def checkpoint(self, session_id: str, report: dict) -> bool:
        elapsed, average, peak, body = self._snapshot(report)
        with self.db:
            result = self.db.execute("""UPDATE sessions SET updated_at=?,elapsed_seconds=?,average_sbi=?,peak_sbi=?,report_json=?
                WHERE id=? AND status='running'""", (_iso(), elapsed, average, peak, body, session_id))
        return result.rowcount == 1

    def finish(self, session_id: str, report: dict, ended_at: str) -> bool:
        elapsed, average, peak, body = self._snapshot(report)
        with self.db:
            result = self.db.execute("""UPDATE sessions SET ended_at=?,updated_at=?,status='completed',
                elapsed_seconds=?,average_sbi=?,peak_sbi=?,report_json=? WHERE id=? AND status='running'""",
                (_iso(ended_at), _iso(), elapsed, average, peak, body, session_id))
        if result.rowcount == 1:
            return True
        row = self.db.execute('SELECT status FROM sessions WHERE id=?', (session_id,)).fetchone()
        return row is not None and row['status'] == 'completed'

    def list_sessions(self, offset: int = 0, limit: int = 100) -> list[dict]:
        if offset < 0 or not 1 <= limit <= 1001:
            raise ValueError('无效的历史分页范围。')
        rows = self.db.execute('SELECT ' + ','.join(SUMMARY_COLUMNS) +
            ' FROM sessions ORDER BY started_at DESC,id DESC LIMIT ? OFFSET ?', (limit, offset))
        return [dict(row) for row in rows]

    def load_session(self, session_id: str) -> dict | None:
        row = self.db.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is None:
            return None
        try:
            if row['report_version'] != 1:
                raise ValueError('不支持的报告版本')
            report = json.loads(row['report_json'])
            if not isinstance(report, dict):
                raise ValueError('报告不是结构化对象')
            report = clean_report(report)
        except (ValueError, TypeError) as exc:
            raise ValueError('本条历史报告无法读取；原数据已保留。') from exc
        return {**{key: row[key] for key in SUMMARY_COLUMNS}, 'report': report}

    def delete_session(self, session_id: str) -> bool:
        with self.db:
            result = self.db.execute("DELETE FROM sessions WHERE id=? AND status IN ('completed','interrupted')", (session_id,))
        return result.rowcount == 1

    def recover_interrupted(self, exclude_session_ids=()) -> int:
        # A delayed startup retry must not interrupt this launch's live records.
        excluded = tuple(exclude_session_ids)
        clause = (' AND id NOT IN (' + ','.join('?' for _ in excluded) + ')') if excluded else ''
        with self.db:
            result = self.db.execute("UPDATE sessions SET status='interrupted',ended_at=updated_at WHERE status='running'" + clause, excluded)
        return result.rowcount

    def close(self) -> None:
        self.db.close()
