from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from memory.models import LongTermFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(category, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
"""


class LongTermMemory:
    """Structured facts: projects, preferences, devices, servers, schedules.

    Deliberately boring (SQLite, upsert-by-category+key) so it's easy to
    inspect, back up, and reason about. Semantic/fuzzy recall lives in
    memory/semantic.py — this is for "what is my current WiFi adapter",
    not "find the doc where I mentioned my WiFi adapter".
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert(self, category: str, key: str, value: str) -> LongTermFact:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO facts (category, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (category, key, value, now, now),
        )
        self._conn.commit()
        return self.get(category, key)  # type: ignore[return-value]

    def get(self, category: str, key: str) -> LongTermFact | None:
        row = self._conn.execute(
            "SELECT id, category, key, value, created_at, updated_at "
            "FROM facts WHERE category = ? AND key = ?",
            (category, key),
        ).fetchone()
        return self._row_to_fact(row) if row else None

    def list_by_category(self, category: str) -> list[LongTermFact]:
        rows = self._conn.execute(
            "SELECT id, category, key, value, created_at, updated_at "
            "FROM facts WHERE category = ? ORDER BY updated_at DESC",
            (category,),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def delete(self, category: str, key: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def all(self) -> list[LongTermFact]:
        rows = self._conn.execute(
            "SELECT id, category, key, value, created_at, updated_at FROM facts "
            "ORDER BY category, key"
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    @staticmethod
    def _row_to_fact(row: tuple) -> LongTermFact:
        return LongTermFact(
            id=row[0],
            category=row[1],
            key=row[2],
            value=row[3],
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
        )

    def close(self) -> None:
        self._conn.close()