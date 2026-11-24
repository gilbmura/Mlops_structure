"""Stage uploaded images for review before they can enter the training data.

Handles the three supported upload shapes with one code path:

1. one image with a selected label;
2. several images with a shared label or one label per image;
3. a ZIP of class folders (see :mod:`src.ingestion.archive`).

Every file is content-checked, fingerprinted, screened for duplicates and
evaluation leakage, and recorded. Only files that pass are written to the
staging folder, under a generated name (user filenames never become paths).
Staged files still need a human to confirm the label and admit them.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.dataset.fingerprint import fingerprint
from src.ingestion.archive import read_zip
from src.ingestion.limits import UploadLimits
from src.ingestion.screening import DatasetIndex, judge
from src.labels import UnknownLabelError, normalize_label
from src.storage.uploads import NEEDS_LABEL, REJECTED, STAGED, UploadStore
from utils.files import atomic_write_bytes
from utils.hashing import hash_to_hex
from utils.images import ImageDecodeError, channel_spread, decode_image

logger = logging.getLogger(__name__)

_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}


class ContentScreeningUnavailable(RuntimeError):
    """Calibrated content checks cannot currently be performed."""


class UploadRejectedError(ValueError):
    """The whole request is refused (too many files, too large, unsafe archive...)."""


@dataclass(frozen=True)
class IncomingFile:
    name: str
    data: bytes
    label: str | None = None          # as supplied by the user; validated here
    label_problem: str | None = None  # set when the label could not be derived (e.g. unknown ZIP folder)


def _display_name(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name[:200] or "unnamed"


class IngestionService:
    def __init__(
        self,
        store: UploadStore,
        staging_dir: Path,
        index: DatasetIndex,
        limits: UploadLimits,
        thumbnail_size: int,
        colour_threshold: float,
        content_check: Callable[[bytes], list[str]] | None = None,
    ) -> None:
        self.store = store
        self.staging_dir = Path(staging_dir)
        self.index = index
        self.limits = limits
        self.thumbnail_size = thumbnail_size
        self.colour_threshold = colour_threshold
        self.content_check = content_check

    def ingest_files(self, files: list[IncomingFile]) -> dict:
        """Screen individually uploaded images (single or multiple)."""
        if not files:
            raise UploadRejectedError("No files were uploaded.")
        if len(files) > self.limits.max_files_per_batch:
            raise UploadRejectedError(f"{len(files)} files uploaded; the limit is {self.limits.max_files_per_batch} per batch.")
        if sum(len(f.data) for f in files) > self.limits.max_batch_bytes:
            raise UploadRejectedError(f"Upload exceeds {self.limits.max_batch_bytes // (1024 * 1024)} MiB in total.")
        return self._ingest(files, skipped=[])

    def ingest_zip(self, data: bytes) -> dict:
        """Screen every image inside a class-folder ZIP. Unsafe archives raise ``UnsafeArchiveError``."""
        if len(data) > self.limits.max_batch_bytes:
            raise UploadRejectedError(f"ZIP exceeds {self.limits.max_batch_bytes // (1024 * 1024)} MiB.")
        entries, skipped = read_zip(data, self.limits, max_files=self.limits.max_files_per_batch)
        if len(entries) > self.limits.max_files_per_batch:
            raise UploadRejectedError(f"ZIP contains {len(entries)} images; the limit is {self.limits.max_files_per_batch} per batch.")
        if not entries and not skipped:
            raise UploadRejectedError("The ZIP contains no files.")
        files = [IncomingFile(e.path, e.data, e.label, e.label_problem) for e in entries]
        return self._ingest(files, [(s.path, s.reason) for s in skipped])

    def _ingest(self, files: list[IncomingFile], skipped: list[tuple[str, str]]) -> dict:
        batch_id = uuid.uuid4().hex[:12]
        known = {row["pixel_hash"]: row["id"] for row in self.store.pool_fingerprints()}
        items = [self._screen(batch_id, incoming, known) for incoming in files]
        for name, reason in skipped:
            items.append(self._record(batch_id, name, REJECTED, reason=reason))
        counts = Counter(item["status"] for item in items)
        logger.info("Upload batch %s: %s", batch_id, dict(counts))
        return {"batch_id": batch_id, "counts": dict(counts), "items": items}

    def _screen(self, batch_id: str, incoming: IncomingFile, known: dict[str, str]) -> dict:
        size = len(incoming.data)
        if size > self.limits.max_file_bytes:
            return self._record(batch_id, incoming.name, REJECTED, size=size, reason="File is larger than the per-image limit.")
        try:
            image = decode_image(incoming.data, self.limits.image)
        except ImageDecodeError as exc:
            return self._record(batch_id, incoming.name, REJECTED, size=size, reason=str(exc))

        fp = fingerprint(image, self.thumbnail_size)
        verdict = judge(self.index.matches(fp), known.get(fp.pixel_hash))
        details = {"pixel_hash": fp.pixel_hash, "phash": hash_to_hex(fp.phash), "width": fp.width, "height": fp.height, "size_bytes": size}
        notes = list(verdict.notes)
        if channel_spread(image) > self.colour_threshold:
            notes.append("Colour image: MRI slices are greyscale; check this is a genuine scan before admitting it.")
        if verdict.status != "ok":
            return self._record(batch_id, incoming.name, verdict.status, reason=verdict.reason, notes=notes, **details)

        reasons = self.check_content(incoming.data)
        if reasons:
            return self._record(batch_id, incoming.name, REJECTED,
                                reason="Unsupported image - cannot be used for retraining: " + "; ".join(reasons), **details)

        label, label_problem = self._resolve_label(incoming)
        upload_id = uuid.uuid4().hex[:16]
        stored = self.staging_dir / batch_id / f"{upload_id}{_EXTENSIONS[image.format]}"
        atomic_write_bytes(stored, incoming.data)
        known[fp.pixel_hash] = upload_id  # catch repeats later in the same batch
        status = STAGED if label else NEEDS_LABEL
        return self._record(
            batch_id, incoming.name, status, upload_id=upload_id, label=label, reason=label_problem,
            notes=notes, stored_path=str(stored), **details,
        )

    def check_content(self, data: bytes) -> list[str]:
        """Fail closed when the calibrated content screener is unavailable."""
        if self.content_check is None:
            raise ContentScreeningUnavailable("Content screening is unavailable; no images can be admitted.")
        return self.content_check(data)

    def admit(self, upload_ids: list[str]) -> list[str]:
        """Recheck stored bytes, including legacy staged uploads, before admission."""
        eligible = []
        for upload_id in dict.fromkeys(upload_ids):
            row = self.store.get(upload_id)
            if not row or row["status"] != STAGED:
                continue
            try:
                data = Path(row["stored_path"]).read_bytes()
                image = decode_image(data, self.limits.image)
                if len(data) > self.limits.max_file_bytes:
                    raise ValueError("Stored image exceeds the per-image limit.")
                fp = fingerprint(image, self.thumbnail_size)
                if fp.pixel_hash != row["pixel_hash"]:
                    raise ValueError("Stored image changed since upload; upload it again.")
                verdict = judge(self.index.matches(fp), None)
                reasons = ([verdict.reason] if verdict.status != "ok" else []) + self.check_content(data)
                if reasons:
                    self.store.reject(upload_id, "Unsupported or ineligible image: " + "; ".join(reasons))
                else:
                    eligible.append(upload_id)
            except (OSError, ImageDecodeError, ValueError) as exc:
                self.store.reject(upload_id, str(exc))
        return self.store.admit(eligible)

    @staticmethod
    def _resolve_label(incoming: IncomingFile) -> tuple[str | None, str | None]:
        if incoming.label_problem:
            return None, incoming.label_problem
        if not incoming.label:
            return None, "No label provided."
        try:
            return normalize_label(incoming.label), None
        except UnknownLabelError as exc:
            return None, str(exc)

    def _record(self, batch_id: str, name: str, status: str, upload_id: str | None = None, notes: list[str] | None = None, size: int | None = None, **fields: object) -> dict:
        record = {
            "id": upload_id or uuid.uuid4().hex[:16],
            "batch_id": batch_id,
            "original_name": _display_name(name),
            "status": status,
            "notes": json.dumps(notes or []),
            **({"size_bytes": size} if size is not None else {}),
            **fields,
        }
        self.store.add(record)
        return {**record, "notes": notes or []}
