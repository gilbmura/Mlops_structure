"""Measured service availability from periodic health probes.

Process uptime ("this replica has been running for X") says nothing about
whether users could reach the service. Availability is measured separately: a
probe calls the public health endpoint through the load balancer at a fixed
interval and records each success or failure here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

from src.storage.database import Database, utc_now


class ProbeStore:
    def __init__(self, database: Database) -> None:
        self.db = database

    def record(self, ok: bool, status_code: int | None, latency_ms: float | None, served_by: str | None, detail: str = "") -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO health_probes (checked_at, ok, status_code, latency_ms, served_by, detail) VALUES (?, ?, ?, ?, ?, ?)",
                (utc_now(), int(ok), status_code, latency_ms, served_by, detail[:500]),
            )

    def prune(self, keep_days: int = 14) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=keep_days)).isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM health_probes WHERE checked_at < ?", (cutoff,))

    def summary(self, hours: float = 24.0) -> dict:
        since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS probes, COALESCE(SUM(ok), 0) AS ok, MIN(checked_at) AS first, MAX(checked_at) AS last, "
                "AVG(CASE WHEN ok = 1 THEN latency_ms END) AS mean_latency_ms FROM health_probes WHERE checked_at >= ?",
                (since,),
            ).fetchone()
            recent = conn.execute(
                "SELECT checked_at, ok, status_code, latency_ms, served_by, detail FROM health_probes ORDER BY id DESC LIMIT 20"
            ).fetchall()
        probes = row["probes"]
        return {
            "window_hours": hours,
            "probes": probes,
            "successful": row["ok"],
            "availability": (row["ok"] / probes) if probes else None,
            "first_probe": row["first"],
            "last_probe": row["last"],
            "mean_latency_ms": round(row["mean_latency_ms"], 1) if row["mean_latency_ms"] is not None else None,
            "recent": [dict(r) for r in recent],
        }
