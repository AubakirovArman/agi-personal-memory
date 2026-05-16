"""v1.0: SQLite backend — migration from JSON files to relational storage."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteMemoryStore:
    """SQLite-based persistent memory with indexes for fast search."""

    def __init__(self, path: str | Path = "agim_memory.db"):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                kind TEXT DEFAULT 'fact_teach',
                source TEXT DEFAULT 'user',
                confidence REAL DEFAULT 1.0,
                tier TEXT DEFAULT 'retrieval',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                accessed_at TEXT DEFAULT (datetime('now')),
                access_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_facts_question ON facts(question);
            CREATE INDEX IF NOT EXISTS idx_facts_tier ON facts(tier);
            CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
            CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);

            CREATE TABLE IF NOT EXISTS commits (
                id TEXT PRIMARY KEY,
                fact_id TEXT,
                tier TEXT,
                question TEXT,
                answer TEXT,
                previous_json TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                rolled_back INTEGER DEFAULT 0,
                FOREIGN KEY (fact_id) REFERENCES facts(id)
            );
            CREATE INDEX IF NOT EXISTS idx_commits_created ON commits(created_at);

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                event TEXT,
                status TEXT,
                data_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
        """)
        self.conn.commit()

    def upsert_fact(self, question: str, answer: str, kind: str = "fact_teach",
                    source: str = "user", confidence: float = 1.0,
                    tier: str = "retrieval", fact_id: str | None = None,
                    metadata: dict | None = None) -> str:
        import uuid
        fid = fact_id or uuid.uuid4().hex[:12]
        self.conn.execute("""
            INSERT OR REPLACE INTO facts (id, question, answer, kind, source,
                confidence, tier, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (fid, question, answer, kind, source, confidence, tier,
              json.dumps(metadata or {})))
        self.conn.commit()
        return fid

    def lookup(self, question: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE question = ? ORDER BY confidence DESC LIMIT 1",
            (question,)).fetchone()
        if row:
            self.conn.execute(
                "UPDATE facts SET accessed_at = datetime('now'), access_count = access_count + 1 WHERE id = ?",
                (row["id"],))
            self.conn.commit()
            return dict(row)
        return None

    def search(self, query: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE question LIKE ? OR answer LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
        by_tier = {}
        for row in self.conn.execute("SELECT tier, COUNT(*) as c FROM facts GROUP BY tier"):
            by_tier[row["tier"]] = row["c"]
        by_kind = {}
        for row in self.conn.execute("SELECT kind, COUNT(*) as c FROM facts GROUP BY kind"):
            by_kind[row["kind"]] = row["c"]
        return {"total": total, "by_tier": by_tier, "by_kind": by_kind}

    def remove(self, question: str) -> bool:
        cur = self.conn.execute("DELETE FROM facts WHERE question = ?", (question,))
        self.conn.commit()
        return cur.rowcount > 0

    def log_event(self, event: str, status: str, data: dict | None = None):
        self.conn.execute("INSERT INTO events (event, status, data_json) VALUES (?, ?, ?)",
                         (event, status, json.dumps(data or {})))
        self.conn.commit()

    def recent_events(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def close(self):
        self.conn.close()
