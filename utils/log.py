"""Logging setup used by scripts, the API and the worker."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent format."""
    logging.basicConfig(level=level.upper(), format=_FORMAT)


def add_file_handler(logger: logging.Logger, path: Path) -> logging.Handler:
    """Attach a file handler (e.g. a per-job log) and return it so callers can remove it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    return handler
