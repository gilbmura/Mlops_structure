"""ZIP prediction with real weights, archive boundaries, and safe CSV downloads."""

import csv
import dataclasses
import io
import stat
import zipfile
from pathlib import Path

import matplotlib
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from src.ingestion.archive import UnsafeArchiveError, read_zip
from src.ingestion.limits import UploadLimits
from src.prediction import REFUSED_MESSAGE
from tests.helpers import make_image, require_artifacts
from ui.views.batch_predict import result_rows
from utils.csv_export import csv_bytes
from utils.files import read_csv


def make_zip(entries: dict[str, bytes], compression=zipfile.ZIP_STORED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.fixture
def batch_client(settings, config):
    with TestClient(create_app(settings, config)) as client:
        yield client


@pytest.mark.needs_artifacts
def test_zip_matches_single_predictions_and_never_stages(real_settings, settings):
    require_artifacts(real_settings)
    isolated = dataclasses.replace(real_settings, state_dir=settings.runtime_dir)
    row = read_csv(real_settings.manifests_dir / "test.csv")[0]
    image = (real_settings.data_dir / row["path"]).read_bytes()
    payload = make_zip({
        "scan.jpg": image, "arbitrary/deep/folder/scan.bin": image,
        "=formula.jpg": image, "broken.jpg": b"not an image",
        "nested.zip": make_zip({"inside.jpg": image}), "hidden.jpg": make_zip({"inside.jpg": image}),
    })
    with TestClient(create_app(isolated)) as client:
        services = client.app.state.services
        before = (services.uploads.counts(), services.jobs.list())
        single = client.post("/predict", files={"file": ("single.jpg", image)}).json()
        response = client.post("/predict/zip", files={"file": ("scans.zip", payload)})
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["total_files"], body["predicted_files"], body["rejected_files"]) == (6, 3, 3)
        for item in body["items"]:
            if item["status"] == "predicted":
                assert item["prediction"] == single
                assert item["prediction"]["model_version"] == body["model_version"]
            else:
                assert item["prediction"] is None and item["reason"]
        assert (services.uploads.counts(), services.jobs.list()) == before
        assert not settings.staging_dir.exists()
        rows = result_rows(body)
        exported = list(csv.DictReader(io.StringIO(csv_bytes(rows, list(rows[0])).decode("utf-8-sig"))))
        assert len(exported) == 6
        assert any(r["filename"] == "'=formula.jpg" for r in exported)
        assert all(r["predicted_class"] == "" and r["confidence"] == "" for r in exported if r["status"] == "rejected")
        invalid = client.post("/predict/zip", files={"file": ("bad-images.zip", make_zip({"fake.png": b"bad"}))})
        assert invalid.status_code == 200 and invalid.json()["predicted_files"] == 0


@pytest.mark.needs_artifacts
def test_zip_refuses_non_mri_images_without_a_class(real_settings, settings):
    require_artifacts(real_settings)
    isolated = dataclasses.replace(real_settings, state_dir=settings.runtime_dir)
    scan = (real_settings.data_dir / read_csv(real_settings.manifests_dir / "test.csv")[0]["path"]).read_bytes()
    photo = io.BytesIO()
    Image.open(Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg").save(photo, format="PNG")
    with TestClient(create_app(isolated)) as client:
        body = client.post("/predict/zip", files={"file": ("mixed.zip", make_zip({"scan.jpg": scan, "photo.png": photo.getvalue()}))}).json()
    by_name = {item["filename"]: item for item in body["items"]}
    assert by_name["scan.jpg"]["status"] == "predicted"
    refused = by_name["photo.png"]
    assert refused["status"] == "rejected" and refused["prediction"] is None
    assert refused["reason"].startswith(REFUSED_MESSAGE) and "colour" in refused["reason"]


@pytest.mark.parametrize("payload", [b"not zip", make_zip({}), make_zip({"../escape.jpg": b"x"}),
                                      make_zip({"../unsafe-directory/": b""})])
def test_invalid_archives_return_actionable_400(batch_client, payload):
    response = batch_client.post("/predict/zip", files={"file": ("x.zip", payload)})
    assert response.status_code == 400 and response.json()["detail"]


def test_batch_missing_model_returns_503(batch_client):
    response = batch_client.post("/predict/zip", files={"file": ("x.zip", make_zip({"x.png": make_image(2)}))})
    assert response.status_code == 503


def test_batch_limits_are_applied_before_inference(batch_client):
    services = batch_client.app.state.services
    original = services.limits
    services.limits = dataclasses.replace(original, max_files_per_batch=1)
    response = batch_client.post("/predict/zip", files={"file": ("x.zip", make_zip({"a": b"a", "b": b"b"}))})
    assert response.status_code == 400 and "batch limit" in response.json()["detail"]
    services.limits = dataclasses.replace(original, max_batch_bytes=16)
    assert batch_client.post("/predict/zip", files={"file": ("x.zip", b"x" * 17)}).status_code == 413
    services.limits = dataclasses.replace(original, max_zip_uncompressed_bytes=5)
    response = batch_client.post("/predict/zip", files={"file": ("x.zip", make_zip({"a": b"x" * 6}))})
    assert response.status_code == 400 and "expand" in response.json()["detail"]


def test_corrupt_entry_returns_400(batch_client):
    payload = bytearray(make_zip({"a.png": b"image bytes"}))
    payload[30 + len("a.png")] ^= 1  # alter content without updating its CRC
    response = batch_client.post("/predict/zip", files={"file": ("bad-crc.zip", bytes(payload))})
    assert response.status_code == 400 and "corrupt" in response.json()["detail"]


def test_zip_without_labels_and_per_file_limit(config):
    limits = UploadLimits.from_config(config)
    entries, rejected = read_zip(make_zip({"arbitrary/deep/folder/x.png": make_image(1)}), limits, infer_labels=False)
    assert len(entries) == 1 and entries[0].label is None and entries[0].label_problem is None and not rejected
    entries, rejected = read_zip(make_zip({"x.png": make_image(1)}), dataclasses.replace(limits, max_file_bytes=10))
    assert not entries and "per-image limit" in rejected[0].reason


def test_symlinks_and_bombs_are_rejected(config):
    limits = UploadLimits.from_config(config)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")
    with pytest.raises(UnsafeArchiveError, match="symbolic link"):
        read_zip(stream.getvalue(), limits, infer_labels=False)
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        read_zip(make_zip({"bomb": b"0" * 100_000}, zipfile.ZIP_DEFLATED), limits, infer_labels=False)


@pytest.mark.parametrize("name", ['=HYPERLINK("bad")', "+cmd", "-cmd", "@cmd", "  =cmd", "\tcmd", "\rcmd"])
def test_csv_escapes_spreadsheet_formulas(name):
    result = list(csv.DictReader(io.StringIO(csv_bytes([{"filename": name, "confidence": 0.75}],
                                                     ["filename", "confidence"]).decode("utf-8-sig"))))
    assert result[0]["filename"] == "'" + name
    assert result[0]["confidence"] == "0.75"
