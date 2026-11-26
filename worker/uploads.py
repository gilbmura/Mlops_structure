"""ZIP screening jobs share the single worker with training to bound resource use."""

from config.settings import Settings
from src.ingestion.batch import screen_archive
from src.ingestion.factory import build_ingestion
from src.ingestion.limits import UploadLimits
from src.ingestion.queue import archive_path
from src.prediction import ActiveModelProvider
from src.registry import ModelRegistry
from src.storage.jobs import RUNNING, UPLOAD, JobStore
from src.storage.uploads import UploadStore


def run_upload(job: dict, settings: Settings, config: dict, jobs: JobStore, uploads: UploadStore) -> dict:
    path = archive_path(settings.runtime_dir, job["id"])
    jobs.progress(job["id"], {"batch_id": job["id"], "phase": "loading screening model", "processed": 0,
                               "total": job["params"]["total_files"], "counts": {}})
    try:
        provider = ActiveModelProvider(ModelRegistry(settings.models_dir), config, UploadLimits.from_config(config).image)
        predictor = provider.get()  # Freeze one model version for the entire batch.
        service = build_ingestion(settings, config, uploads, lambda: predictor)
        result = screen_archive(path, job["id"], service, lambda state: jobs.progress(job["id"], state))
        return {**result, "model_version": predictor.version}
    except Exception:
        uploads.invalidate_batch(job["id"], "ZIP processing failed; upload this batch again after resolving the error.")
        raise
    finally:
        path.unlink(missing_ok=True)


def recover_upload(job: dict | None, settings: Settings, uploads: UploadStore) -> None:
    """An interrupted job fails closed; partial images cannot be admitted on restart."""
    if job and job["kind"] == UPLOAD and job["status"] == RUNNING:
        uploads.invalidate_batch(job["id"], "ZIP processing was interrupted. Upload the batch again.")
        archive_path(settings.runtime_dir, job["id"]).unlink(missing_ok=True)
