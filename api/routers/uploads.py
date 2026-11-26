"""Upload staging and human review (admin only)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from api.dependencies import AdminDep, ServicesDep, read_upload
from api.schemas import AdmitRequest, LabelUpdate
from api.services import Services
from src.ingestion.archive import UnsafeArchiveError
from src.ingestion.queue import queue_zip
from src.ingestion.service import ContentScreeningUnavailable, IncomingFile, IngestionService, UploadRejectedError
from src.labels import UnknownLabelError
from src.storage.uploads import UploadStateError
from src.storage.jobs import UPLOAD, JobConflictError

router = APIRouter(prefix="/uploads", tags=["uploads"], dependencies=[AdminDep])


def _ingestion(services: Services) -> IngestionService:
    if services.ingestion is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Uploads are unavailable: {services.ingestion_error}")
    return services.ingestion


def _parse_labels(count: int, label: str | None, labels: str | None) -> list[str | None]:
    """Either one shared ``label`` or a JSON list ``labels`` with one entry (or null) per file."""
    if labels:
        try:
            values = json.loads(labels)
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "'labels' must be a JSON list of strings.") from exc
        if not isinstance(values, list) or len(values) != count:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'labels' must list exactly {count} entries, one per file.")
        return [str(v) if v not in (None, "") else None for v in values]
    return [label or None] * count


def _public(record: dict) -> dict:
    record = dict(record)
    record.pop("stored_path", None)  # server paths are not exposed
    if isinstance(record.get("notes"), str):
        record["notes"] = json.loads(record["notes"] or "[]")
    return record


@router.post("/images")
def upload_images(
    services: ServicesDep,
    files: list[UploadFile] = File(...),
    label: str | None = Form(None, description="Shared label for all files"),
    labels: str | None = Form(None, description="JSON list with one label per file (overrides 'label')"),
) -> dict:
    """Stage one or more images. Unknown or missing labels are flagged as 'needs_label'."""
    ingestion = _ingestion(services)
    if len(files) > services.limits.max_files_per_batch:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"At most {services.limits.max_files_per_batch} files per batch.")
    per_file = _parse_labels(len(files), label, labels)
    incoming = [
        IncomingFile(f.filename or "unnamed", read_upload(f, services.limits.max_file_bytes), lab)
        for f, lab in zip(files, per_file)
    ]
    try:
        report = ingestion.ingest_files(incoming)
    except ContentScreeningUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except UploadRejectedError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    return {**report, "items": [_public(i) for i in report["items"]]}


@router.post("/zip", status_code=status.HTTP_202_ACCEPTED)
def upload_zip(services: ServicesDep, file: UploadFile = File(..., description="ZIP of class folders")) -> dict:
    """Save a ZIP and queue screening. Poll /uploads/jobs/{id}; review after success."""
    _ingestion(services)
    try:
        return queue_zip(file.file, file.filename or "upload.zip", services.settings.runtime_dir, services.limits, services.jobs)
    except UnsafeArchiveError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"ZIP rejected: {exc}") from exc
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except UploadRejectedError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc


@router.get("/jobs")
def upload_jobs(services: ServicesDep) -> list[dict]:
    return services.jobs.list(20, kinds=(UPLOAD,))


@router.get("/jobs/{job_id}")
def upload_job(services: ServicesDep, job_id: str) -> dict:
    job = services.jobs.get(job_id)
    if not job or job["kind"] != UPLOAD:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload job not found.")
    return job


@router.get("")
def list_uploads(services: ServicesDep, status_filter: str | None = None, batch_id: str | None = None,
                 limit: int = Query(500, ge=1, le=2000), offset: int = Query(0, ge=0)) -> dict:
    items = services.uploads.list(status_filter, batch_id, limit, offset)
    return {"counts": services.uploads.counts(), "total": services.uploads.count(status_filter, batch_id),
            "items": [_public(i) for i in items]}


@router.get("/{upload_id}/image")
def upload_image(services: ServicesDep, upload_id: str) -> FileResponse:
    """Preview a kept upload (rejected and duplicate files are not stored)."""
    record = services.uploads.get(upload_id)
    if not record or not record["stored_path"] or not Path(record["stored_path"]).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No stored image for this upload.")
    return FileResponse(record["stored_path"])


@router.patch("/{upload_id}")
def set_label(services: ServicesDep, upload_id: str, body: LabelUpdate) -> dict:
    try:
        return _public(services.uploads.set_label(upload_id, body.label))
    except UnknownLabelError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except UploadStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/{upload_id}/discard")
def discard(services: ServicesDep, upload_id: str) -> dict:
    try:
        return _public(services.uploads.discard(upload_id))
    except UploadStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/admit")
def admit(services: ServicesDep, body: AdmitRequest) -> dict:
    """Admit reviewed, labelled uploads into the training pool used by the next retraining job."""
    if not body.confirm_labels_reviewed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Confirm that the labels were reviewed before admitting uploads.")
    try:
        admitted = _ingestion(services).admit(body.upload_ids)
    except ContentScreeningUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return {"admitted": admitted, "not_admitted": [i for i in body.upload_ids if i not in admitted], "counts": services.uploads.counts()}
