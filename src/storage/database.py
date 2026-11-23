"""SQLite persistence shared by API replicas, the worker and the availability probe.

SQLite (WAL mode) is adequate for one host with a few processes on a local
volume. It is not safe on SMB network shares; see deploy/azure/README.md.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    params TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    log_path TEXT
);
-- At most one queued or running job per kind: prevents concurrent training runs.
CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_per_kind ON jobs(kind) WHERE status IN ('queued', 'running');

CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    original_name TEXT NOT NULL,
    label TEXT,
    status TEXT NOT NULL CHECK (status IN ('staged', 'needs_label', 'admitted', 'rejected', 'duplicate', 'discarded')),
    reason TEXT,
    notes TEXT,
    stored_path TEXT,
    pixel_hash TEXT,
    phash TEXT,
    width INTEGER,
    height INTEGER,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    admitted_at TEXT
);
CREATE INDEX IF NOT EXISTS uploads_status ON uploads(status);
CREATE INDEX IF NOT EXISTS uploads_pixel_hash ON uploads(pixel_hash);

CREATE TABLE IF NOT EXISTS health_probes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT NOT NULL,
    ok INTEGER NOT NULL,
    status_code INTEGER,
    latency_ms REAL,
    served_by TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS health_probes_checked_at ON health_probes(checked_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """Opens short-lived connections; each ``transaction()`` commits or rolls back atomically."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")  # persistent setting stored in the file
            connection.executescript(SCHEMA)  # idempotent; executescript manages its own transaction
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
