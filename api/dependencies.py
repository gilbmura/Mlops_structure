"""FastAPI dependencies: service access, admin authentication and bounded file reads."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, UploadFile, status

from api.services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_admin(services: ServicesDep, x_admin_key: Annotated[str | None, Header()] = None) -> None:
    """Protect data-changing operations (uploads, retraining, promotion, rollback).

    Admin endpoints are disabled entirely when no ADMIN_API_KEY is configured.
    """
    expected = services.settings.admin_api_key
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin operations are disabled: set ADMIN_API_KEY on the server.")
    if not x_admin_key or not secrets.compare_digest(x_admin_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-Admin-Key header.")


AdminDep = Depends(require_admin)


def read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file, refusing anything above ``max_bytes`` without reading it all."""
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"{upload.filename or 'File'} exceeds {max_bytes // (1024 * 1024)} MiB.")
    return data
