"""Records of uploaded images while they move from staging to the training pool.

Status flow: ``needs_label`` -> ``staged`` -> ``admitted`` (or ``discarded``).
``rejected`` and ``duplicate`` uploads are recorded for transparency but their
files are not kept. Only ``admitted`` uploads are ever used for retraining.
"""

from __future__ import annotations

import sqlite3

from src.labels import normalize_label
from src.storage.database import Database, utc_now

STAGED, NEEDS_LABEL, ADMITTED, REJECTED, DUPLICATE, DISCARDED = (
    "staged", "needs_label", "admitted", "rejected", "duplicate", "discarded",
)
POOL_STATUSES = (STAGED, NEEDS_LABEL, ADMITTED)  # uploads whose files are kept
# Legacy/synchronous batches have no job; background batches must finish first.
_READY_BATCH = "NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.id = uploads.batch_id AND jobs.kind = 'upload' AND jobs.status != 'succeeded')"

_FIELDS = (
    "id", "batch_id", "original_name", "label", "status", "reason", "notes", "stored_path",
    "pixel_hash", "phash", "width", "height", "size_bytes", "created_at", "updated_at",
)


class UploadStateError(ValueError):
    """Raised for an invalid review action (e.g. admitting an upload without a label)."""


def _rows(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


class UploadStore:
    def __init__(self, database: Database) -> None:
        self.db = database

    def add(self, record: dict) -> None:
        self.add_many([record])

    def add_many(self, records: list[dict]) -> None:
        """Commit a small screening checkpoint without opening a connection per image."""
        if not records:
            return
        now = utc_now()
        values = [{**{f: None for f in _FIELDS}, **record, "created_at": now, "updated_at": now} for record in records]
        with self.db.transaction() as conn:
            conn.executemany(
                f"INSERT INTO uploads ({', '.join(_FIELDS)}) VALUES ({', '.join('?' for _ in _FIELDS)})",
                [[row[f] for f in _FIELDS] for row in values],
            )

    def get(self, upload_id: str) -> dict | None:
        with self.db.transaction() as conn:
            row = conn.execute(f"SELECT uploads.*, ({_READY_BATCH}) AS review_ready FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _filter(status: str | None, batch_id: str | None) -> tuple[str, list]:
        query, params = "WHERE 1 = 1", []
        if status == "review":
            query += " AND status IN ('staged', 'needs_label')"
        elif status:
            query += " AND status = ?"
            params.append(status)
        if status in ("review", STAGED, NEEDS_LABEL, ADMITTED):
            query += f" AND {_READY_BATCH}"
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        return query, params

    def list(self, status: str | None = None, batch_id: str | None = None, limit: int = 500, offset: int = 0) -> list[dict]:
        where, params = self._filter(status, batch_id)
        with self.db.transaction() as conn:
            return _rows(conn.execute(f"SELECT * FROM uploads {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                                     (*params, limit, offset)).fetchall())

    def count(self, status: str | None = None, batch_id: str | None = None) -> int:
        where, params = self._filter(status, batch_id)
        with self.db.transaction() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM uploads {where}", params).fetchone()[0]

    def invalidate_batch(self, batch_id: str, reason: str) -> None:
        """Retain the audit trail, but exclude partial failed batches from training and deduplication."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE uploads SET status = ?, reason = ?, updated_at = ? WHERE batch_id = ? AND status IN (?, ?)",
                         (REJECTED, reason, utc_now(), batch_id, STAGED, NEEDS_LABEL))

    def counts(self) -> dict[str, int]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM uploads GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}

    def pool_fingerprints(self) -> list[dict]:
        """Fingerprints of kept uploads, used to detect duplicates among uploads."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                f"SELECT id, pixel_hash, phash, stored_path, status FROM uploads WHERE status IN ({', '.join('?' * len(POOL_STATUSES))})",
                POOL_STATUSES,
            ).fetchall()
        return _rows(rows)

    def set_label(self, upload_id: str, label: str) -> dict:
        """Assign or correct a label; only uploads still under review can change."""
        canonical = normalize_label(label)
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE uploads SET label = ?, status = ?, updated_at = ? WHERE id = ? AND status IN (?, ?) AND {_READY_BATCH}",
                (canonical, STAGED, utc_now(), upload_id, STAGED, NEEDS_LABEL),
            )
        if cursor.rowcount == 0:
            raise UploadStateError(f"Upload {upload_id} is not awaiting review, so its label cannot change.")
        return self.get(upload_id)

    def discard(self, upload_id: str) -> dict:
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE uploads SET status = ?, updated_at = ? WHERE id = ? AND status IN (?, ?) AND {_READY_BATCH}",
                (DISCARDED, utc_now(), upload_id, STAGED, NEEDS_LABEL),
            )
        if cursor.rowcount == 0:
            raise UploadStateError(f"Upload {upload_id} is not awaiting review.")
        return self.get(upload_id)

    def reject(self, upload_id: str, reason: str) -> None:
        """Invalidate a previously staged image; labels cannot restore eligibility."""
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE uploads SET status = ?, reason = ?, updated_at = ? WHERE id = ? AND status IN (?, ?)",
                (REJECTED, reason, utc_now(), upload_id, STAGED, NEEDS_LABEL),
            )

    def admit(self, upload_ids: list[str]) -> list[str]:
        """Move reviewed, labelled uploads into the training pool. Returns the ids admitted."""
        now = utc_now()
        admitted = []
        with self.db.transaction() as conn:
            for upload_id in upload_ids:
                cursor = conn.execute(
                    f"UPDATE uploads SET status = ?, admitted_at = ?, updated_at = ? WHERE id = ? AND status = ? AND label IS NOT NULL AND {_READY_BATCH}",
                    (ADMITTED, now, now, upload_id, STAGED),
                )
                if cursor.rowcount:
                    admitted.append(upload_id)
        return admitted

    def admitted(self) -> list[dict]:
        with self.db.transaction() as conn:
            return _rows(conn.execute(f"SELECT * FROM uploads WHERE status = ? AND {_READY_BATCH} ORDER BY admitted_at, rowid", (ADMITTED,)).fetchall())
