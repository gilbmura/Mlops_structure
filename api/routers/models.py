"""Manual model activation and rollback (admin only)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.dependencies import AdminDep, ServicesDep
from api.schemas import ActivationRequest
from src.registry import RegistryError

router = APIRouter(prefix="/models", tags=["models"], dependencies=[AdminDep])


@router.post("/{version}/activate")
def activate(services: ServicesDep, version: str, body: ActivationRequest) -> dict:
    """Serve a specific registered version (manual override of the automatic promotion rule)."""
    try:
        services.registry.activate(version, f"manual activation: {body.reason}")
    except RegistryError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {"active_version": services.registry.active_version()}


@router.post("/rollback")
def rollback(services: ServicesDep, body: ActivationRequest) -> dict:
    """Re-activate the model that was active before the current one."""
    try:
        version = services.registry.rollback(body.reason)
    except RegistryError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"active_version": version}
