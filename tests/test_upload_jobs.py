"""Durable ZIP receipt, streaming screening, progress, and incomplete-batch isolation."""

import dataclasses
import io
import zipfile
from types import SimpleNamespace

import pytest

from src.ingestion.archive import ZipReader
from src.ingestion.queue import archive_path
from src.storage.jobs import FAILED, RUNNING, UPLOAD
from tests.helpers import ADMIN_KEY, make_image
from tests.test_api import client  # noqa: F401 -- reuse the isolated API fixture
from worker.main import process_one
from worker.uploads import recover_upload, run_upload

ADMIN = {"X-Admin-Key": ADMIN_KEY}


def zipped(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_STORED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return data.getvalue()


def submit(client, payload):
    return client.post("/uploads/zip", headers=ADMIN, files={"file": ("new.zip", payload)})


def run(client, monkeypatch):
    services = client.app.state.services
    monkeypatch.setattr("worker.uploads.ActiveModelProvider", lambda *a: SimpleNamespace(get=lambda: SimpleNamespace(version="test")))
    monkeypatch.setattr("worker.uploads.build_ingestion", lambda *a: services.ingestion)
    process_one(services.jobs, {UPLOAD: lambda job: run_upload(job, services.settings, services.config, services.jobs, services.uploads)})


def test_receipt_returns_job_before_screening_then_worker_completes(client, monkeypatch):
    payload = zipped({"glioma/a.png": make_image(501), "pituitary/b.png": make_image(502), "glioma/bad.png": b"bad"})
    response = submit(client, payload)
    assert response.status_code == 202
    job = response.json()
    services = client.app.state.services
    assert job["status"] == "queued" and services.uploads.counts() == {}
    assert archive_path(services.settings.runtime_dir, job["id"]).read_bytes() == payload
    # Upload filenames and progress are admin-only, including through generic job routes.
    assert client.get("/uploads/jobs").status_code == 401
    assert client.get(f"/uploads/jobs/{job['id']}").status_code == 401
    assert client.get("/jobs").json() == []
    assert client.get(f"/jobs/{job['id']}").status_code == 404
    assert client.get(f"/jobs/{job['id']}/log").status_code == 404
    run(client, monkeypatch)
    finished = client.get(f"/uploads/jobs/{job['id']}", headers=ADMIN).json()
    assert finished["status"] == "succeeded"
    assert finished["result"]["processed"] == finished["result"]["total"] == 3
    assert finished["result"]["counts"] == {"staged": 2, "rejected": 1}
    assert not archive_path(services.settings.runtime_dir, job["id"]).exists()
    page = client.get("/uploads", headers=ADMIN, params={"status_filter": "review", "batch_id": job["id"], "limit": 1, "offset": 1}).json()
    assert page["total"] == 2 and len(page["items"]) == 1
    uid = page["items"][0]["id"]
    admitted = client.post("/uploads/admit", headers=ADMIN, json={"upload_ids": [uid], "confirm_labels_reviewed": True})
    assert admitted.json()["admitted"] == [uid]


@pytest.mark.parametrize("state", ["queued", RUNNING, FAILED])
def test_incomplete_batches_cannot_be_reviewed_or_admitted(client, state):
    services = client.app.state.services
    job = services.jobs.create(UPLOAD)
    if state != "queued":
        services.jobs.claim_next()
    if state == FAILED:
        services.jobs.finish(job["id"], FAILED, error="interrupted")
    services.uploads.add({"id": "partial", "batch_id": job["id"], "original_name": "x.png", "status": "staged", "label": "glioma"})
    assert services.uploads.get("partial")["review_ready"] == 0
    assert services.uploads.list("review") == []
    assert services.uploads.admit(["partial"]) == []
    assert client.patch("/uploads/partial", headers=ADMIN, json={"label": "pituitary"}).status_code == 409
    assert client.post("/jobs/retrain", headers=ADMIN, json={"batch_id": job["id"]}).status_code == 400


def test_corrupt_late_entry_invalidates_partial_results_and_preserves_progress(client, monkeypatch):
    payload = bytearray(zipped({"glioma/a.png": make_image(511), "glioma/b.png": make_image(512)}))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        second = archive.infolist()[1]
        position = second.header_offset + 30 + len(second.filename.encode()) + len(second.extra)
    payload[position + 20] ^= 0xFF
    job = submit(client, bytes(payload)).json()
    run(client, monkeypatch)
    services = client.app.state.services
    failed = services.jobs.get(job["id"])
    assert failed["status"] == FAILED and "corrupt" in failed["error"]
    assert failed["result"]["processed"] == 1
    assert services.uploads.list("review") == []
    assert services.uploads.counts() == {"rejected": 1}
    assert not archive_path(services.settings.runtime_dir, job["id"]).exists()


def test_retry_reuses_active_job_and_conflicting_upload_cleans_received_file(client):
    services = client.app.state.services
    first = zipped({"glioma/a.png": make_image(521)})
    job = submit(client, first).json()
    assert submit(client, first).json()["id"] == job["id"]
    conflict = submit(client, zipped({"glioma/b.png": make_image(522)}))
    assert conflict.status_code == 409
    assert [p.name for p in (services.settings.runtime_dir / "upload_archives").iterdir()] == [f"{job['id']}.zip"]


def test_receive_size_limit_removes_partial_file(client):
    services = client.app.state.services
    services.limits = dataclasses.replace(services.limits, max_batch_bytes=16)
    assert submit(client, zipped({"a.png": b"x"})).status_code == 413
    assert list((services.settings.runtime_dir / "upload_archives").iterdir()) == []
    assert services.jobs.list() == []


def test_worker_restart_keeps_checkpoint_but_blocks_interrupted_upload(client):
    services = client.app.state.services
    job = submit(client, zipped({"glioma/a.png": make_image(531)})).json()
    claimed = services.jobs.claim_next()
    services.jobs.progress(job["id"], {"processed": 1, "total": 2})
    services.uploads.add({"id": "partial", "batch_id": job["id"], "original_name": "x.png", "status": "staged", "label": "glioma"})
    services.jobs.fail_orphaned("worker restarted")
    recover_upload(claimed, services.settings, services.uploads)
    assert services.jobs.get(job["id"])["result"]["processed"] == 1
    assert services.uploads.get("partial")["status"] == "rejected"
    assert not archive_path(services.settings.runtime_dir, job["id"]).exists()


def test_zip_reader_decompresses_one_file_at_a_time(config, monkeypatch):
    from src.ingestion.limits import UploadLimits
    payload = zipped({f"glioma/{i}.png": make_image(i) for i in range(4)})
    opened = []
    original = zipfile.ZipFile.open
    def track(archive, info, *args, **kwargs):
        opened.append(info.filename)
        return original(archive, info, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", track)
    with ZipReader(io.BytesIO(payload), UploadLimits.from_config(config)) as reader:
        assert not opened
        iterator = iter(reader)
        assert next(iterator).path == "glioma/0.png"
        assert opened == ["glioma/0.png"]


def test_16000_file_job_finishes_with_counts_and_paged_reasons(client, monkeypatch):
    payload = zipped({f"__MACOSX/._{i}.png": b"metadata" for i in range(16000)})
    response = submit(client, payload)
    assert response.status_code == 202
    run(client, monkeypatch)
    job = client.app.state.services.jobs.get(response.json()["id"])
    assert job["status"] == "succeeded"
    assert job["result"]["processed"] == 16000 and job["result"]["counts"] == {"rejected": 16000}
    page = client.get("/uploads", headers=ADMIN, params={"batch_id": job["id"], "status_filter": "rejected", "limit": 25, "offset": 15975}).json()
    assert page["total"] == 16000 and len(page["items"]) == 25
