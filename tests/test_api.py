"""API behaviour: real-model prediction, input validation, admin protection and the retraining trigger."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import matplotlib
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from src.labels import CLASS_NAMES, tumor_status_message
from src.prediction import REFUSED_MESSAGE
from src.registry import ModelRegistry
from tests.helpers import ADMIN_KEY, make_image, require_artifacts
from utils.files import read_csv

ADMIN = {"X-Admin-Key": ADMIN_KEY}


@pytest.fixture
def real_client(real_settings, settings):
    """App serving the genuinely trained model from this checkout, with an isolated database."""
    require_artifacts(real_settings)
    isolated = dataclasses.replace(real_settings, data_dir=settings.data_dir, state_dir=settings.runtime_dir, admin_api_key=ADMIN_KEY)
    with TestClient(create_app(isolated)) as client:
        yield client


@pytest.fixture
def client(settings, config, indexed_images):
    """App without a model (fast), used for validation, auth and upload/retrain flow tests."""
    with TestClient(create_app(settings, config)) as test_client:
        test_client.app.state.services.ingestion.content_check = lambda data: []
        yield test_client


@pytest.mark.needs_artifacts
def test_predict_with_trained_model_on_held_out_images(real_client, real_settings) -> None:
    metadata = ModelRegistry(real_settings.models_dir).metadata(real_client.get("/health").json()["model_version"])
    assert metadata["training"]["smoke"] is False and metadata["validation_metrics"]["macro_f1"] > 0.5
    # Every 20th held-out slice: a spread sample across classes and source collections.
    picks = read_csv(real_settings.manifests_dir / "test.csv")[::20]
    accepted = 0
    for row in picks:
        path = real_settings.data_dir / row["path"]
        response = real_client.post("/predict", files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 200, response.text
        body = response.json()
        assert "not a diagnostic" in body["disclaimer"]
        if not body["accepted"]:  # ~1 % of genuine slices are refused by design
            assert body["predicted_class"] is None and body["rejection_reasons"]
            continue
        accepted += 1
        assert body["predicted_class"] in CLASS_NAMES and not body["rejection_reasons"]
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
        assert body["confidence"] == max(body["probabilities"].values())
        assert body["message"] == tumor_status_message(body["predicted_class"])
        assert body["tumor_detected"] == (body["predicted_class"] != "no_tumor")
    assert accepted >= 0.9 * len(picks), f"only {accepted} of {len(picks)} held-out MRI slices were accepted"


@pytest.mark.needs_artifacts
def test_health_reports_model_and_replica(real_client) -> None:
    response = real_client.get("/health")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "ok" and body["model_loaded"]
    assert body["model_artifact"] == "model.h5"
    assert response.headers["X-Served-By"] == body["served_by"]
    assert body["process_uptime_seconds"] >= 0


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.needs_artifacts
def test_non_mri_images_are_refused_without_a_diagnosis(real_client) -> None:
    portrait = Image.open(Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg").convert("RGB")
    cases = {
        "colour portrait": (_png(portrait), "colour"),
        "greyscale portrait": (_png(portrait.convert("L")), "does not resemble"),  # passes the rules, caught by feature distance
        "flat colour square": (_png(Image.new("RGB", (200, 200), (200, 30, 30))), "uniform"),
    }
    for name, (data, expected_reason) in cases.items():
        response = real_client.post("/predict", files={"file": (f"{name}.png", data)})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["accepted"] is False and body["message"] == REFUSED_MESSAGE, name
        assert (body["predicted_class"], body["tumor_detected"], body["confidence"], body["probabilities"]) == (None, None, None, None), name
        assert any(expected_reason in reason for reason in body["rejection_reasons"]), (name, body["rejection_reasons"])


def test_health_without_model_is_degraded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["status"] == "degraded"
    assert response.json()["model_artifact"] is None


def test_direct_api_admission_cannot_bypass_content_checks(client):
    response = client.post("/uploads/images", headers=ADMIN,
                           files=[("files", ("scan.png", make_image(310)))], data={"labels": '["glioma"]'})
    assert response.status_code == 200, response.text
    upload_id = response.json()["items"][0]["id"]
    client.app.state.services.ingestion.content_check = lambda data: ["Not MRI-like"]
    result = client.post("/uploads/admit", headers=ADMIN,
                         json={"upload_ids": [upload_id], "confirm_labels_reviewed": True})
    assert result.status_code == 200 and result.json()["admitted"] == []
    assert client.patch(f"/uploads/{upload_id}", headers=ADMIN, json={"label": "no_tumor"}).status_code == 409


def test_upload_returns_503_when_screening_is_unavailable(client):
    client.app.state.services.ingestion.content_check = None
    response = client.post("/uploads/images", headers=ADMIN,
                           files=[("files", ("scan.png", make_image(311)))], data={"labels": '["glioma"]'})
    assert response.status_code == 503


def test_predict_without_model_returns_503(client) -> None:
    assert client.post("/predict", files={"file": ("x.jpg", make_image(1))}).status_code == 503


@pytest.mark.needs_artifacts
def test_predict_rejects_non_image_content(real_client) -> None:
    response = real_client.post("/predict", files={"file": ("scan.jpg", b"GIF89a fake bytes")})
    assert response.status_code == 422 and "image" in response.json()["detail"]


def test_admin_endpoints_require_key(client) -> None:
    files = {"files": ("a.png", make_image(1))}
    assert client.post("/uploads/images", files=files, data={"label": "glioma"}).status_code == 401
    assert client.post("/uploads/images", files=files, data={"label": "glioma"}, headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/jobs/retrain", json={}, headers=ADMIN).status_code == 400  # authorised, but nothing admitted yet


def test_admin_disabled_without_configured_key(settings, config, indexed_images) -> None:
    with TestClient(create_app(dataclasses.replace(settings, admin_api_key=""), config)) as client:
        assert client.get("/uploads", headers=ADMIN).status_code == 503


def test_upload_review_and_retrain_trigger(client) -> None:
    upload = client.post("/uploads/images", files=[("files", ("one.png", make_image(7)))], data={"label": "pituitary"}, headers=ADMIN)
    assert upload.status_code == 200
    item = upload.json()["items"][0]
    assert item["status"] == "staged" and "stored_path" not in item
    assert client.get(f"/uploads/{item['id']}/image", headers=ADMIN).status_code == 200

    assert client.post("/uploads/admit", json={"upload_ids": [item["id"]], "confirm_labels_reviewed": False}, headers=ADMIN).status_code == 400
    admitted = client.post("/uploads/admit", json={"upload_ids": [item["id"]], "confirm_labels_reviewed": True}, headers=ADMIN)
    assert admitted.json()["admitted"] == [item["id"]]

    job = client.post("/jobs/retrain", json={"smoke": True}, headers=ADMIN)
    assert job.status_code == 202 and job.json()["status"] == "queued"
    assert client.post("/jobs/retrain", json={}, headers=ADMIN).status_code == 409  # no concurrent training
    assert client.get(f"/jobs/{job.json()['id']}").json()["status"] == "queued"  # the API never trains in-request


def test_upload_label_validation_via_api(client) -> None:
    item = client.post("/uploads/images", files=[("files", ("x.png", make_image(8)))], data={"labels": '["tumour?"]'}, headers=ADMIN).json()["items"][0]
    assert item["status"] == "needs_label"
    assert client.patch(f"/uploads/{item['id']}", json={"label": "astrocytoma"}, headers=ADMIN).status_code == 422
    assert client.patch(f"/uploads/{item['id']}", json={"label": "glioma"}, headers=ADMIN).json()["status"] == "staged"
    assert client.post("/uploads/images", files=[("files", ("x.png", make_image(9)))], data={"labels": '["a", "b"]'}, headers=ADMIN).status_code == 400


def test_unsafe_zip_is_rejected(client) -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../outside.png", make_image(10))
    response = client.post("/uploads/zip", files={"file": ("bad.zip", buffer.getvalue())}, headers=ADMIN)
    assert response.status_code == 400 and "traversal" in response.json()["detail"]


def test_rejected_batch_cannot_use_previously_admitted_pool(client):
    store = client.app.state.services.uploads
    store.add({"id": "old", "batch_id": "old-batch", "original_name": "old.png", "label": "glioma", "status": "admitted"})
    store.add({"id": "bad", "batch_id": "rejected-batch", "original_name": "drawing.png", "status": "rejected"})
    for batch in ("rejected-batch", "unknown-batch"):
        response = client.post("/jobs/retrain", headers=ADMIN, json={"batch_id": batch})
        assert response.status_code == 400
    assert client.get("/jobs").json() == []
    response = client.post("/jobs/retrain", headers=ADMIN, json={})
    assert response.status_code == 202
    assert response.json()["params"]["upload_ids"] == ["old"]


def test_batch_job_snapshots_only_its_admitted_images(client):
    store = client.app.state.services.uploads
    for uid, batch, state in [("old", "previous", "admitted"), ("good", "current", "admitted"),
                               ("pending", "current", "staged"), ("bad", "current", "rejected")]:
        store.add({"id": uid, "batch_id": batch, "original_name": f"{uid}.png", "label": "glioma", "status": state})
    response = client.post("/jobs/retrain", headers=ADMIN, json={"batch_id": "current"})
    assert response.status_code == 202
    assert response.json()["params"]["upload_ids"] == ["good"]


def test_initial_training_requires_admin_data_and_valid_architecture(client):
    assert client.post("/jobs/train", json={}).status_code == 401
    assert client.post("/jobs/train", headers=ADMIN, json={}).status_code == 400
    folder = client.app.state.services.settings.manifests_dir
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        (folder / f"{split}.csv").write_text("sample_id,label,path\n")
    assert client.post("/jobs/train", headers=ADMIN, json={"architecture": "pretrained"}).status_code == 422
    response = client.post("/jobs/train", headers=ADMIN, json={"architecture": "baseline", "smoke": True})
    assert response.status_code == 202
    assert response.json()["kind"] == "train"
    assert response.json()["params"] == {"architecture": "baseline", "smoke": True}
    assert client.post("/jobs/train", headers=ADMIN, json={}).status_code == 409


def test_smoke_candidate_cannot_be_activated(client, monkeypatch):
    monkeypatch.setattr(client.app.state.services.registry, "metadata", lambda version: {"training": {"smoke": True}})
    assert client.post("/models/v001/activate", headers=ADMIN, json={"reason": "test smoke"}).status_code == 409
