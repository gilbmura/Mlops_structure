"""Background job creation (admin) and status/log inspection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.dependencies import AdminDep, ServicesDep
from api.schemas import RetrainRequest, TrainRequest
from src.storage.jobs import TRAIN, RETRAIN, UPLOAD, JobConflictError

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/train", status_code=status.HTTP_202_ACCEPTED, dependencies=[AdminDep])
def request_training(services: ServicesDep, body: TrainRequest) -> dict:
    """Train from random weights on prepared data; register a candidate for manual activation."""
    for split in ("train", "validation"):
        if not (services.settings.manifests_dir / f"{split}.csv").is_file():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Prepare the training and validation data before training.")
    try:
        return services.jobs.create(TRAIN, body.model_dump())
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED, dependencies=[AdminDep])
def request_retraining(services: ServicesDep, body: RetrainRequest) -> dict:
    """Queue a retraining job. The worker trains outside this request; poll GET /jobs/{id}."""
    admitted = [row["id"] for row in services.uploads.admitted()
                if body.batch_id is None or row["batch_id"] == body.batch_id]
    if not admitted:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No admitted images in the selected batch or pool. Review and admit images first.")
    try:
        job = services.jobs.create(RETRAIN, {"smoke": body.smoke, "batch_id": body.batch_id,
                                            "upload_ids": admitted, "admitted_uploads_at_request": len(admitted)})
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return job


@router.get("")
def list_jobs(services: ServicesDep, limit: int = 20) -> list[dict]:
    return services.jobs.list(min(limit, 100), kinds=(TRAIN, RETRAIN))


@router.get("/{job_id}")
def get_job(services: ServicesDep, job_id: str) -> dict:
    job = services.jobs.get(job_id)
    if not job or job["kind"] == UPLOAD:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found.")
    return job


@router.get("/{job_id}/log")
def get_job_log(services: ServicesDep, job_id: str) -> dict:
    job = services.jobs.get(job_id)
    if not job or job["kind"] == UPLOAD:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found.")
    return {"job_id": job_id, "log": services.jobs.read_log(job_id)}
