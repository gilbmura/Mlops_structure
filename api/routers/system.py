"""Health, availability, model and dataset information (read-only, public)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, status

from api.dependencies import ServicesDep
from api.schemas import HealthResponse
from src.labels import CLASS_NAMES, NO_TUMOR
from src.prediction import ModelUnavailableError
from src.registry import RegistryError
from utils.files import read_json

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep, response: Response) -> dict:
    """Liveness + readiness of *this* replica. Returns 503 while no model can be served."""
    try:
        version, detail = services.models.get().version, None
    except ModelUnavailableError as exc:
        version, detail = None, str(exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if version else "degraded",
        "model_loaded": version is not None,
        "model_version": version,
        "process_uptime_seconds": services.uptime_seconds,
        "process_started_at": services.started_at,
        "served_by": services.hostname,
        "detail": detail,
    }


@router.get("/labels")
def labels() -> dict:
    """The fixed class order of the label contract (the UI reads it instead of duplicating it)."""
    return {"classes": list(CLASS_NAMES), "no_tumor": NO_TUMOR}


@router.get("/availability")
def availability(services: ServicesDep, hours: float = Query(24.0, gt=0, le=24 * 14)) -> dict:
    """Measured availability from recorded health probes, kept separate from process uptime."""
    return {
        "measured": services.probes.summary(hours),
        "this_replica": {"served_by": services.hostname, "process_uptime_seconds": services.uptime_seconds, "process_started_at": services.started_at},
        "note": "Availability counts successful /health probes through the load balancer; uptime is only how long this process has run.",
    }


@router.get("/model")
def active_model(services: ServicesDep) -> dict:
    """Metadata of the active model and the version actually loaded on this replica."""
    version = services.registry.active_version()
    if not version:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active model is registered.")
    metadata = services.registry.metadata(version)
    return {"active_version": version, "loaded_version": services.models.loaded_version, "served_by": services.hostname, "metadata": metadata}


@router.get("/models")
def list_models(services: ServicesDep) -> dict:
    """Registry state: all versions, statuses and activation history."""
    return services.registry.state()


@router.get("/models/{version}/history")
def model_history(services: ServicesDep, version: str) -> dict:
    try:
        return read_json(services.registry.version_dir(version) / "history.json")
    except RegistryError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/insights/dataset")
def dataset_insights(services: ServicesDep) -> dict:
    """Dataset card (cleaning + split summary) and image-feature analysis, if generated."""
    card_path = services.settings.manifests_dir / "dataset_card.json"
    features_path = services.settings.reports_dir / "dataset" / "feature_summary.json"
    if not card_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset card not found; run scripts/prepare_dataset.py.")
    return {"card": read_json(card_path), "features": read_json(features_path) if features_path.is_file() else None}


@router.get("/insights/evaluation")
def evaluation_insights(services: ServicesDep, version: str | None = None) -> dict:
    """Validation metrics, training history and (if run) final-test report for a model version."""
    version = version or services.registry.active_version()
    if not version:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active model is registered.")
    try:
        folder = services.registry.version_dir(version)
    except RegistryError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    reports = services.settings.reports_dir / "models" / version
    test_path = reports / "test_metrics.json"
    return {
        "version": version,
        "validation": read_json(folder / "metadata.json")["validation_metrics"],
        "history": read_json(folder / "history.json"),
        "test": read_json(test_path) if test_path.is_file() else None,
    }
