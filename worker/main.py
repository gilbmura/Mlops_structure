"""Background worker: runs queued jobs one at a time, outside any web request.

Usage:
    python -m worker.main            # loop forever
    python -m worker.main --once     # process at most one job, then exit (tests / demos)

Running a single worker process is what keeps training from exhausting local
resources; the database additionally refuses a second queued/running job.
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from collections.abc import Callable

from config.settings import Settings, get_settings, load_pipeline_config
from src.registry import ModelRegistry
from src.retraining import run_retraining
from src.storage.database import Database
from src.storage.jobs import FAILED, RETRAIN, SUCCEEDED, JobStore
from src.storage.uploads import UploadStore
from utils.log import add_file_handler, configure_logging

logger = logging.getLogger("worker")

POLL_SECONDS = 3.0


def _handlers(settings: Settings, config: dict, database: Database) -> dict[str, Callable[[dict], dict]]:
    registry = ModelRegistry(settings.models_dir)
    uploads = UploadStore(database)
    return {RETRAIN: lambda job: run_retraining(job["id"], job["params"], settings, config, registry, uploads)}


def process_one(jobs: JobStore, handlers: dict[str, Callable[[dict], dict]]) -> bool:
    """Claim and run the next queued job. Returns False when the queue was empty."""
    job = jobs.claim_next()
    if job is None:
        return False
    root = logging.getLogger()
    handler = add_file_handler(root, jobs.logs_dir / f"{job['id']}.log")
    try:
        logger.info("Job %s (%s) started with params %s", job["id"], job["kind"], job["params"])
        if job["kind"] not in handlers:
            raise ValueError(f"Unknown job kind {job['kind']!r}.")
        result = handlers[job["kind"]](job)
        jobs.finish(job["id"], SUCCEEDED, result=result)
        logger.info("Job %s succeeded: %s", job["id"], result)
    except Exception as exc:  # any failure is recorded on the job instead of killing the worker
        logger.error("Job %s failed: %s\n%s", job["id"], exc, traceback.format_exc())
        jobs.finish(job["id"], FAILED, error=f"{type(exc).__name__}: {exc}")
    finally:
        root.removeHandler(handler)
        handler.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database_path)
    jobs = JobStore(database, settings.job_logs_dir)
    orphaned = jobs.fail_orphaned("Worker restarted while this job was running; no model was promoted.")
    if orphaned:
        logger.warning("Marked %d interrupted job(s) as failed", orphaned)
    handlers = _handlers(settings, load_pipeline_config(), database)
    logger.info("Worker ready; polling every %.0fs", POLL_SECONDS)

    while True:
        worked = process_one(jobs, handlers)
        if args.once:
            break
        if not worked:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
