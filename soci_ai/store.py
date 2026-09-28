from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .domain import SessionFrame
from .live.domain import LiveState


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    sbi REAL NOT NULL,
    risk_level TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


class SessionStore:
    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path not in {None, ":memory:"} else None
        self.persistence_mode = "sqlite" if self.path is not None else "memory"
        self.warning = ""
        self._lock = threading.RLock()
        self._connection = self._connect()
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    @classmethod
    def in_memory(cls) -> "SessionStore":
        return cls(None)

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            return sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            return connection
        except (OSError, sqlite3.Error):
            self.persistence_mode = "memory"
            self.warning = "database_unwritable"
            return sqlite3.connect(":memory:", check_same_thread=False)

    def start_session(self, scenario_id: str) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions(id, scenario_id, started_at) VALUES (?, ?, ?)",
                (session_id, scenario_id, datetime.now(timezone.utc).isoformat()),
            )
            self._connection.commit()
        return session_id

    def append_state(self, session_id: str, frame: SessionFrame) -> None:
        self._ensure_session(session_id)
        snapshot = frame.snapshot
        intervention = frame.intervention
        with self._lock:
            self._connection.execute(
                "INSERT INTO states(session_id, at_ms, sbi, risk_level, evidence_status, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    frame.at_ms,
                    snapshot.sbi,
                    snapshot.risk_level,
                    snapshot.evidence_status,
                    frame.model_dump_json(),
                ),
            )
            self._connection.execute(
                "INSERT INTO interventions(session_id, at_ms, action, reason, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    intervention.at_ms,
                    intervention.action,
                    intervention.reason,
                    intervention.model_dump_json(),
                ),
            )
            if frame.dialogue:
                self._connection.execute(
                    "INSERT INTO transcript(session_id, at_ms, speaker, text) VALUES (?, ?, ?, ?)",
                    (session_id, frame.at_ms, frame.speaker, frame.dialogue),
                )
            self._connection.commit()

    def append_text(self, session_id: str, at_ms: int, speaker: str, text: str) -> None:
        self._ensure_session(session_id)
        with self._lock:
            self._connection.execute(
                "INSERT INTO transcript(session_id, at_ms, speaker, text) VALUES (?, ?, ?, ?)",
                (session_id, max(0, int(at_ms)), speaker, text),
            )
            self._connection.commit()

    def append_live_state(self, session_id: str, state: LiveState) -> None:
        """Persist derived live features only; raw frame/audio bytes never enter this API."""
        self._ensure_session(session_id)
        if state.sbi < 25:
            risk_level = "safe"
        elif state.sbi < 45:
            risk_level = "observe"
        elif state.sbi < 70:
            risk_level = "warning"
        elif state.sbi < 85:
            risk_level = "high"
        else:
            risk_level = "critical"
        with self._lock:
            self._connection.execute(
                "INSERT INTO states(session_id, at_ms, sbi, risk_level, evidence_status, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    state.elapsed_ms,
                    state.sbi,
                    risk_level,
                    state.evidence_status,
                    state.model_dump_json(),
                ),
            )
            if state.transcript.status == "completed" and state.transcript.text:
                latest = self._connection.execute(
                    "SELECT at_ms, text FROM transcript WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if latest is None or latest["at_ms"] != state.transcript.at_ms or latest["text"] != state.transcript.text:
                    self._connection.execute(
                        "INSERT INTO transcript(session_id, at_ms, speaker, text) VALUES (?, ?, ?, ?)",
                        (session_id, state.transcript.at_ms, "使用者", state.transcript.text),
                    )
            self._connection.commit()

    def finish_session(self, session_id: str) -> None:
        self._ensure_session(session_id)
        with self._lock:
            self._connection.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id),
            )
            self._connection.commit()

    def get_session(self, session_id: str) -> dict[str, object]:
        self._ensure_session(session_id)
        with self._lock:
            session = self._connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            states = self._connection.execute(
                "SELECT payload FROM states WHERE session_id = ? ORDER BY at_ms, id", (session_id,)
            ).fetchall()
            transcript = self._connection.execute(
                "SELECT at_ms, speaker, text FROM transcript WHERE session_id = ? ORDER BY at_ms, id",
                (session_id,),
            ).fetchall()
        return {
            "session": dict(session),
            "states": [json.loads(row["payload"]) for row in states],
            "transcript": [dict(row) for row in transcript],
        }

    def _ensure_session(self, session_id: str) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()
