"""Single-image prediction."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.dependencies import ServicesDep, read_upload
from api.schemas import BatchPredictionResponse, PredictionResponse
from src.batch_prediction import classify_entries
from src.ingestion.archive import UnsafeArchiveError, read_zip
from src.prediction import DISCLAIMER, ModelUnavailableError
from utils.images import ImageDecodeError

router = APIRouter(tags=["prediction"])


@router.post("/predict/zip", response_model=BatchPredictionResponse)
def predict_zip(services: ServicesDep, file: UploadFile = File(..., description="ZIP of JPEG/PNG images; labels not required")) -> dict:
    """Classify ZIP images in memory. No admin key, staging, or training-data admission."""
    data = read_upload(file, services.limits.max_batch_bytes)
    try:
        entries, skipped = read_zip(data, services.limits, infer_labels=False,
                                    max_files=services.limits.max_files_per_batch)
    except UnsafeArchiveError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not entries and not skipped:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "ZIP contains no files.")
    try:
        predictor = services.models.get()  # pin one model version for the entire response
    except ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return classify_entries(entries, skipped, predictor)


@router.post("/predict", response_model=PredictionResponse)
def predict(services: ServicesDep, file: UploadFile = File(..., description="JPEG or PNG brain MRI slice")) -> dict:
    """Classify one image into glioma / meningioma / pituitary / no_tumor.

    Non-image files are rejected with 422. Images that do not look like the
    training MRIs return ``accepted: false`` with ``rejection_reasons`` and no
    class or score. Acceptance does not prove the image is an MRI.
    """
    data = read_upload(file, services.limits.max_file_bytes)
    try:
        predictor = services.models.get()
        prediction = predictor.predict_bytes(data)
    except ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return {**prediction.as_dict(), "disclaimer": DISCLAIMER}
