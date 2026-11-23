"""Thin HTTP client for the FastAPI service. The UI never imports model code directly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


class ApiError(RuntimeError):
    """An API call failed; ``str(error)`` is a message suitable for the user."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ApiClient:
    base_url: str
    admin_key: str = ""
    timeout: float = 30.0

    def _request(self, method: str, path: str, admin: bool = False, timeout: float | None = None, **kwargs: Any) -> requests.Response:
        headers = {"X-Admin-Key": self.admin_key} if admin and self.admin_key else {}
        try:
            response = requests.request(method, f"{self.base_url}{path}", headers=headers, timeout=timeout or self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{response.status_code}: {detail}", response.status_code)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, path, **kwargs).json()

    # -- public, read-only ---------------------------------------------------------
    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=5)  # a 503 body is still informative
        return {**response.json(), "http_status": response.status_code}

    def predict(self, name: str, data: bytes) -> dict:
        return self._json("POST", "/predict", files={"file": (name, data)})

    def predict_zip(self, name: str, data: bytes) -> dict:
        return self._json("POST", "/predict/zip", timeout=300, files={"file": (name, data, "application/zip")})

    def labels(self) -> list[str]:
        return self._json("GET", "/labels")["classes"]

    def availability(self, hours: float = 24) -> dict:
        return self._json("GET", "/availability", params={"hours": hours})

    def model(self) -> dict:
        return self._json("GET", "/model")

    def models(self) -> dict:
        return self._json("GET", "/models")

    def dataset_insights(self) -> dict:
        return self._json("GET", "/insights/dataset")

    def evaluation(self, version: str | None = None) -> dict:
        return self._json("GET", "/insights/evaluation", params={"version": version} if version else None)

    def jobs(self, limit: int = 10) -> list[dict]:
        return self._json("GET", "/jobs", params={"limit": limit})

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/jobs/{job_id}")

    def job_log(self, job_id: str) -> str:
        return self._json("GET", f"/jobs/{job_id}/log")["log"]

    # -- admin ---------------------------------------------------------------------
    def upload_images(self, files: list[tuple[str, bytes]], labels: list[str | None]) -> dict:
        payload = [("files", (name, data)) for name, data in files]
        return self._json("POST", "/uploads/images", admin=True, timeout=300, files=payload, data={"labels": json.dumps(labels)})

    def upload_zip(self, name: str, data: bytes) -> dict:
        return self._json("POST", "/uploads/zip", admin=True, timeout=300, files={"file": (name, data, "application/zip")})

    def uploads(self, status: str | None = None) -> dict:
        return self._json("GET", "/uploads", admin=True, params={"status_filter": status} if status else None)

    def upload_image(self, upload_id: str) -> bytes:
        return self._request("GET", f"/uploads/{upload_id}/image", admin=True).content

    def set_label(self, upload_id: str, label: str) -> dict:
        return self._json("PATCH", f"/uploads/{upload_id}", admin=True, json={"label": label})

    def discard(self, upload_id: str) -> dict:
        return self._json("POST", f"/uploads/{upload_id}/discard", admin=True)

    def admit(self, upload_ids: list[str]) -> dict:
        return self._json("POST", "/uploads/admit", admin=True, json={"upload_ids": upload_ids, "confirm_labels_reviewed": True})

    def retrain(self, smoke: bool) -> dict:
        return self._json("POST", "/jobs/retrain", admin=True, json={"smoke": smoke})

    def rollback(self, reason: str) -> dict:
        return self._json("POST", "/models/rollback", admin=True, json={"reason": reason})

    def activate(self, version: str, reason: str) -> dict:
        return self._json("POST", f"/models/{version}/activate", admin=True, json={"reason": reason})
