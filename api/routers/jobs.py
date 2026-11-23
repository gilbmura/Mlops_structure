"""Background job creation (admin) and status/log inspection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.dependencies import AdminDep, ServicesDep
from api.schemas import RetrainRequest
from src.storage.jobs import RETRAIN, JobConflictError

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED, dependencies=[AdminDep])
def request_retraining(services: ServicesDep, body: RetrainRequest) -> dict:
    """Queue a retraining job. The worker trains outside this request; poll GET /jobs/{id}."""
    admitted = services.uploads.counts().get("admitted", 0)
    if admitted == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Admit at least one reviewed upload before retraining.")
    try:
        job = services.jobs.create(RETRAIN, {"smoke": body.smoke, "admitted_uploads_at_request": admitted})
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return job


@router.get("")
def list_jobs(services: ServicesDep, limit: int = 20) -> list[dict]:
    return services.jobs.list(min(limit, 100))


@router.get("/{job_id}")
def get_job(services: ServicesDep, job_id: str) -> dict:
    job = services.jobs.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found.")
    return job


@router.get("/{job_id}/log")
def get_job_log(services: ServicesDep, job_id: str) -> dict:
    if not services.jobs.get(job_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found.")
    return {"job_id": job_id, "log": services.jobs.read_log(job_id)}
