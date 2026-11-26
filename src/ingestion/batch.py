"""Screen a disk-backed archive incrementally and checkpoint small progress summaries."""

import time
from copy import copy
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from src.ingestion.archive import ZipReader
from src.ingestion.service import IngestionService
from src.storage.upload_buffer import BufferedUploadStore


def screen_archive(path: Path, batch_id: str, service: IngestionService, progress: Callable[[dict], None]) -> dict:
    # A separate service/store prevents buffering concurrent API requests.
    service = copy(service)
    writer = BufferedUploadStore(service.store.db)
    service.store = writer
    counts, processed, total = Counter(), 0, 0
    known = {row["pixel_hash"]: row["id"] for row in service.store.pool_fingerprints()}

    def snapshot() -> dict:
        return {"batch_id": batch_id, "phase": "screening", "processed": processed, "total": total, "counts": dict(counts)}

    with ZipReader(path, service.limits, max_files=service.limits.max_files_per_batch) as reader:
        total = len(reader.infos)
        progress(snapshot())
        last_update = time.monotonic()
        try:
            for entry in reader:
                item = service.screen_archive_entry(batch_id, entry, known)
                counts[item["status"]] += 1
                processed += 1
                if processed % 100 == 0 or time.monotonic() - last_update >= 1:
                    writer.flush()  # progress never gets ahead of committed rows
                    progress(snapshot())
                    last_update = time.monotonic()
        finally:
            writer.flush()
            progress(snapshot())
    return {**snapshot(), "phase": "complete"}
