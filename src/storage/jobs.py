"""Persistent background-job records (queued -> running -> succeeded | failed)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from src.storage.database import Database, utc_now

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
TRAIN, RETRAIN = "train", "retrain"


class JobConflictError(RuntimeError):
    """Raised when an incompatible job is already queued or running."""


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    job = dict(row)
    job["params"] = json.loads(job["params"] or "{}")
    job["result"] = json.loads(job["result"]) if job["result"] else None
    return job


class JobStore:
    def __init__(self, database: Database, logs_dir: Path) -> None:
        self.db = database
        self.logs_dir = Path(logs_dir)

    def create(self, kind: str, params: dict | None = None) -> dict:
        """Queue a job; training and retraining share a single active slot."""
        job_id = uuid.uuid4().hex[:12]
        log_path = self.logs_dir / f"{job_id}.log"
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, kind, status, created_at, params, log_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (job_id, kind, QUEUED, utc_now(), json.dumps(params or {}), str(log_path)),
                )
        except sqlite3.IntegrityError as exc:
            active = (self.active(TRAIN) or self.active(RETRAIN)) if kind in (TRAIN, RETRAIN) else self.active(kind)
            raise JobConflictError(
                f"A {active['kind'] if active else kind} job is already {active['status'] if active else 'active'} "
                f"({active['id'] if active else 'unknown'})."
            ) from exc
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self.db.transaction() as conn:
            return _row(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def list(self, limit: int = 20) -> list[dict]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]

    def active(self, kind: str) -> dict | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE kind = ? AND status IN (?, ?)", (kind, QUEUED, RUNNING)).fetchone()
        return _row(row)

    def claim_next(self) -> dict | None:
        """Atomically move the oldest queued job to running and return it."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE status = ? ORDER BY created_at, rowid LIMIT 1", (QUEUED,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status = ?, started_at = ? WHERE id = ?", (RUNNING, utc_now(), row["id"]))
        return self.get(row["id"])

    def finish(self, job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
        if status not in (SUCCEEDED, FAILED):
            raise ValueError(f"Final status must be {SUCCEEDED} or {FAILED}, not {status!r}.")
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, result = ?, error = ? WHERE id = ?",
                (status, utc_now(), json.dumps(result) if result is not None else None, error, job_id),
            )

    def fail_orphaned(self, reason: str) -> int:
        """Mark jobs left 'running' by a crashed or restarted worker as failed."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE status = ?", (FAILED, utc_now(), reason, RUNNING)
            )
            return cursor.rowcount

    def read_log(self, job_id: str, max_chars: int = 20000) -> str:
        job = self.get(job_id)
        if not job or not job["log_path"] or not Path(job["log_path"]).is_file():
            return ""
        return Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")[-max_chars:]
