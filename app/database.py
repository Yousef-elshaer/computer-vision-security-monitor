from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class EventStore:
    def __init__(self, path: Path):
        self.path = path
        self._write_lock = Lock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    snapshot TEXT,
                    sms_status TEXT NOT NULL DEFAULT 'not-requested',
                    detections_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def add_event(
        self,
        *,
        created_at: str,
        level: str,
        score: int,
        summary: str,
        snapshot: str | None,
        detections: list[dict],
        sms_status: str = "not-requested",
    ) -> int:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events(created_at, level, score, summary, snapshot, sms_status, detections_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, level, score, summary, snapshot, sms_status, json.dumps(detections)),
            )
            return int(cursor.lastrowid)

    def update_sms_status(self, event_id: int, status: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("UPDATE events SET sms_status = ? WHERE id = ?", (status, event_id))

    def recent_events(self, limit: int = 30) -> list[dict[str, Any]]:
        safe_limit = min(max(1, limit), 200)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["detections"] = json.loads(item.pop("detections_json"))
            events.append(item)
        return events

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            high = connection.execute(
                "SELECT COUNT(*) FROM events WHERE level IN ('high', 'critical')"
            ).fetchone()[0]
            today = connection.execute(
                "SELECT COUNT(*) FROM events WHERE date(created_at) = date('now')"
            ).fetchone()[0]
        return {"total": total, "high_risk": high, "today": today}

    def set_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings(key, value_json) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (key, payload),
            )

    def get_setting(self, key: str, default: Any) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row[0])

