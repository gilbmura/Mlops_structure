"""Durably receive a ZIP before making it visible to the background worker."""

import hashlib
import uuid
from pathlib import Path
from typing import BinaryIO

from src.ingestion.archive import UnsafeArchiveError, ZipReader
from src.ingestion.limits import UploadLimits
from src.ingestion.service import UploadRejectedError
from src.storage.jobs import UPLOAD, JobStore


def archive_path(runtime_dir: Path, job_id: str) -> Path:
    if len(job_id) != 12 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Invalid upload job identifier.")
    return runtime_dir / "upload_archives" / f"{job_id}.zip"


def queue_zip(source: BinaryIO, name: str, runtime_dir: Path, limits: UploadLimits, jobs: JobStore) -> dict:
    """Copy in bounded chunks; validate headers without decompressing or screening images."""
    job_id = uuid.uuid4().hex[:12]
    target = archive_path(runtime_dir, job_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    size, digest = 0, hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            while chunk := source.read(min(1024 * 1024, limits.max_batch_bytes - size + 1)):
                size += len(chunk)
                if size > limits.max_batch_bytes:
                    raise UploadRejectedError("ZIP exceeds the compressed upload size limit.")
                handle.write(chunk)
                digest.update(chunk)
        with ZipReader(temporary, limits, max_files=limits.max_files_per_batch) as reader:
            total = len(reader.infos)
            if not total:
                raise UnsafeArchiveError("The ZIP contains no files.")
        # Reconnect to the existing job if a response was lost and the same upload is retried.
        active = jobs.active(UPLOAD)
        if active and active["params"].get("sha256") == digest.hexdigest():
            return active
        temporary.replace(target)
        return jobs.create(UPLOAD, {"filename": Path(name.replace("\\", "/")).name[:200],
                                   "total_files": total, "size_bytes": size, "sha256": digest.hexdigest()}, job_id=job_id)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
