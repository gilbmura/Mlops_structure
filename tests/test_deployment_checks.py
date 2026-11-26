"""Deployment checks must never start unrelated pool training after a bad upload."""

from scripts.check_deployment import run_upload_and_retrain


def test_rejected_upload_cannot_start_pool_retraining(tmp_path, monkeypatch):
    archive = tmp_path / "demo.zip"
    archive.write_bytes(b"zip")
    calls = []

    class Response:
        status_code = 200
        text = "all rejected"

        def json(self):
            return {"batch_id": "bad", "counts": {"rejected": 1}, "items": [{"status": "rejected", "id": "bad"}]}

    def post(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr("scripts.check_deployment.requests.post", post)
    failures = []
    run_upload_and_retrain("http://example", "test-key", archive, "smoke", failures, 1)
    assert calls == ["http://example/uploads/zip"]
    assert failures and "no images" in failures[0]


def test_huge_image_header_is_a_validation_error(monkeypatch):
    import pytest
    from PIL import Image
    from utils.images import ImageDecodeError, decode_image

    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("oversized header")

    monkeypatch.setattr(Image, "open", bomb)
    with pytest.raises(ImageDecodeError):
        decode_image(b"image")


def test_failed_background_upload_cannot_start_pool_retraining(tmp_path, monkeypatch):
    from types import SimpleNamespace

    archive = tmp_path / "demo.zip"
    archive.write_bytes(b"zip")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=202, text="queued", json=lambda: {"id": "upload-job", "status": "queued"})

    def get(url, **kwargs):
        assert kwargs["headers"]["X-Admin-Key"] == "test-key"
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"id": "upload-job", "status": "failed", "error": "corrupt archive"})

    monkeypatch.setattr("scripts.check_deployment.requests.post", post)
    monkeypatch.setattr("scripts.check_deployment.requests.get", get)
    failures = []
    run_upload_and_retrain("http://example", "test-key", archive, "smoke", failures, 1)
    assert calls == ["http://example/uploads/zip"]
    assert "failed" in failures[0]
