"""Availability probe: calls the public /health endpoint at a fixed interval and records the outcome.

Usage:
    python -m worker.probe --interval 30

Run it against the load balancer (API_URL), not an individual replica, so the
record reflects what users would have experienced.
"""

from __future__ import annotations

import argparse
import logging
import time

import requests

from config.settings import get_settings
from src.storage.database import Database
from src.storage.probes import ProbeStore
from utils.log import configure_logging

logger = logging.getLogger("probe")


def probe_once(url: str, store: ProbeStore, timeout: float = 5.0) -> bool:
    started = time.perf_counter()
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        latency = (time.perf_counter() - started) * 1000
        ok = response.status_code == 200 and response.json().get("status") == "ok"
        detail = "" if ok else response.text[:200]
        store.record(ok, response.status_code, round(latency, 1), response.headers.get("X-Served-By"), detail)
        return ok
    except (requests.RequestException, ValueError) as exc:
        store.record(False, None, None, None, f"{type(exc).__name__}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between probes")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    store = ProbeStore(Database(settings.database_path))
    logger.info("Probing %s/health every %.0fs", settings.api_url, args.interval)
    while True:
        if not probe_once(settings.api_url, store):
            logger.warning("Health probe failed")
        store.prune()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
