"""File helpers: atomic writes so readers never observe half-written JSON or CSV."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def exclusive_lock(path: Path, timeout: float = 30.0, stale_after: float = 300.0) -> Iterator[None]:
    """Cross-process lock using an exclusively created lock file.

    Works on Windows and on a shared Docker volume. A lock older than
    ``stale_after`` seconds is assumed to belong to a crashed process and removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_after:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Could not acquire lock {path} within {timeout:.0f}s.") from None
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file in the same folder, then rename over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    """Atomically write pretty-printed JSON."""
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8"))


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    """Atomically write rows as CSV with a fixed column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
