"""Upload validation, ZIP safety, leakage screening and staging."""

from __future__ import annotations

import io
import stat
import zipfile
from dataclasses import replace

import pytest
from PIL import Image

from src.ingestion.archive import UnsafeArchiveError, read_zip
from src.ingestion.limits import UploadLimits
from src.ingestion.service import IncomingFile, IngestionService, UploadRejectedError
from src.ingestion.service import ContentScreeningUnavailable
from src.labels import UnknownLabelError
from src.storage.database import Database
from src.storage.uploads import UploadStateError, UploadStore
from tests.helpers import make_image


@pytest.fixture
def store(settings) -> UploadStore:
    return UploadStore(Database(settings.database_path))


@pytest.fixture
def service(settings, config, store, dataset_index) -> IngestionService:
    return IngestionService(store, settings.staging_dir, dataset_index, UploadLimits.from_config(config), 64, 8.0, content_check=lambda data: [])


def _zip(entries: dict[str, bytes], symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../etc/passwd")
    return buffer.getvalue()


def test_single_labelled_image_is_staged_under_generated_name(service, settings) -> None:
    report = service.ingest_files([IncomingFile("../../evil name.jpg", make_image(1), "Glioma")])
    item = report["items"][0]
    assert item["status"] == "staged" and item["label"] == "glioma"
    stored = settings.staging_dir / report["batch_id"]
    files = list(stored.iterdir())
    assert len(files) == 1 and files[0].stem == item["id"]  # user filename never becomes a path
    assert item["original_name"] == "evil name.jpg"


def test_multiple_images_with_individual_and_missing_labels(service) -> None:
    files = [IncomingFile(f"img{i}.png", make_image(10 + i), label) for i, label in enumerate(["pituitary", None, "unicorn"])]
    statuses = [(i["status"], i["label"]) for i in service.ingest_files(files)["items"]]
    assert statuses == [("staged", "pituitary"), ("needs_label", None), ("needs_label", None)]


def test_invalid_content_is_rejected_regardless_of_extension(service) -> None:
    items = service.ingest_files([IncomingFile("scan.jpg", b"definitely not an image", "glioma")])["items"]
    assert items[0]["status"] == "rejected" and "not a recognised image" in items[0]["reason"]


def test_batch_and_file_limits(service, config) -> None:
    limit = config["uploads"]["max_files_per_batch"]
    with pytest.raises(UploadRejectedError):
        service.ingest_files([IncomingFile(f"{i}.png", b"x", "glioma") for i in range(limit + 1)])
    service.limits = UploadLimits.from_config({**config, "uploads": {**config["uploads"], "max_file_bytes": 100}})
    assert service.ingest_files([IncomingFile("big.png", make_image(20), "glioma")])["items"][0]["status"] == "rejected"


def test_evaluation_images_are_rejected_to_prevent_leakage(service, indexed_images) -> None:
    for split in ("validation", "test"):
        item = service.ingest_files([IncomingFile("x.png", indexed_images[split], "glioma")])["items"][0]
        assert item["status"] == "rejected" and "leakage" in item["reason"]


def test_near_duplicate_of_test_image_is_rejected(service, indexed_images) -> None:
    resized = Image.open(io.BytesIO(indexed_images["test"])).resize((140, 140))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=92)
    item = service.ingest_files([IncomingFile("resized.jpg", buffer.getvalue(), "glioma")])["items"][0]
    assert item["status"] == "rejected" and "test image" in item["reason"]


def test_duplicates_of_training_data_and_other_uploads_are_skipped(service, indexed_images) -> None:
    assert service.ingest_files([IncomingFile("t.png", indexed_images["train"], "glioma")])["items"][0]["status"] == "duplicate"
    image = make_image(30)
    report = service.ingest_files([IncomingFile("a.png", image, "glioma"), IncomingFile("b.png", image, "glioma")])
    assert [i["status"] for i in report["items"]] == ["staged", "duplicate"]
    later = service.ingest_files([IncomingFile("c.png", image, "glioma")])
    assert later["items"][0]["status"] == "duplicate"


def test_retrain_demo_images_are_accepted_with_note(service, indexed_images) -> None:
    item = service.ingest_files([IncomingFile("demo.png", indexed_images["retrain_demo"], "glioma")])["items"][0]
    assert item["status"] == "staged" and any("retrain-demo" in note for note in item["notes"])


def test_zip_with_wrapper_and_single_class_folder(service) -> None:
    report = service.ingest_zip(_zip({
        "new_data/meningioma/a.png": make_image(40),
        "new_data/meningioma/b.jpg": make_image(41, fmt="JPEG"),
        "__MACOSX/new_data/._a.png": b"junk",
    }))
    staged = [i for i in report["items"] if i["status"] == "staged"]
    assert len(staged) == 2 and {i["label"] for i in staged} == {"meningioma"}
    assert any("metadata" in (i["reason"] or "") for i in report["items"] if i["status"] == "rejected")


def test_zip_unknown_and_missing_folders_need_labels(service) -> None:
    report = service.ingest_zip(_zip({"astrocytoma/a.png": make_image(50), "loose.png": make_image(51), "no_tumor/c.png": make_image(52, label_index=3)}))
    by_name = {i["original_name"]: i for i in report["items"]}
    assert by_name["a.png"]["status"] == "needs_label" and "unknown class folder" in by_name["a.png"]["reason"]
    assert by_name["loose.png"]["status"] == "needs_label"
    assert by_name["c.png"]["status"] == "staged" and by_name["c.png"]["label"] == "no_tumor"


def test_nested_archives_are_rejected(config) -> None:
    entries, skipped = read_zip(_zip({"glioma/inner.zip": _zip({"glioma/x.png": make_image(60)}), "glioma/disguised.png": _zip({"a": b"b"})}), UploadLimits.from_config(config))
    assert not entries and all("nested" in s.reason for s in skipped) and len(skipped) == 2


@pytest.mark.parametrize("name", ["../escape.png", "/abs/path.png", "C:/windows/x.png", "glioma/../../x.png"])
def test_path_traversal_rejects_whole_archive(config, name: str) -> None:
    with pytest.raises(UnsafeArchiveError):
        read_zip(_zip({"glioma/ok.png": make_image(70), name: make_image(71)}), UploadLimits.from_config(config))


def test_symlinks_are_rejected(config) -> None:
    with pytest.raises(UnsafeArchiveError, match="symbolic link"):
        read_zip(_zip({"glioma/ok.png": make_image(72)}, symlink="glioma/link.png"), UploadLimits.from_config(config))


def test_zip_bomb_and_size_limits(config) -> None:
    limits = UploadLimits.from_config(config)
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        read_zip(_zip({"glioma/bomb.png": b"\0" * 5_000_000}), limits)
    small = UploadLimits.from_config({**config, "uploads": {**config["uploads"], "max_zip_uncompressed_bytes": 1000}})
    with pytest.raises(UnsafeArchiveError, match="expand"):
        read_zip(_zip({"glioma/a.png": make_image(80)}), small)
    few = UploadLimits.from_config({**config, "uploads": {**config["uploads"], "max_zip_entries": 1}})
    with pytest.raises(UnsafeArchiveError, match="limit"):
        read_zip(_zip({"glioma/a.png": make_image(81), "glioma/b.png": make_image(82)}), few)


def test_not_a_zip(config) -> None:
    with pytest.raises(UnsafeArchiveError, match="not a valid ZIP"):
        read_zip(make_image(90), UploadLimits.from_config(config))


def test_review_flow_label_correction_and_admission(service, store) -> None:
    report = service.ingest_files([IncomingFile("a.png", make_image(100), None), IncomingFile("b.png", make_image(101), "glioma")])
    needs, staged = report["items"]
    assert store.admit([needs["id"], staged["id"]]) == [staged["id"]]  # unlabeled uploads cannot be admitted
    with pytest.raises(UnknownLabelError):
        store.set_label(needs["id"], "meningeoma")
    store.set_label(needs["id"], "meningioma")
    assert store.admit([needs["id"]]) == [needs["id"]]
    with pytest.raises(UploadStateError):
        store.set_label(needs["id"], "glioma")  # admitted labels are frozen
    assert {u["label"] for u in store.admitted()} == {"glioma", "meningioma"}


@pytest.mark.parametrize("as_zip", [False, True])
def test_content_rejection_cannot_be_overridden_by_label(service, store, as_zip):
    service.content_check = lambda data: ["Not MRI-like"]
    report = (service.ingest_zip(_zip({"glioma/a.png": make_image(201)})) if as_zip
              else service.ingest_files([IncomingFile("a.png", make_image(201), "glioma")]))
    item = report["items"][0]
    assert item["status"] == "rejected" and "Not MRI-like" in item["reason"]
    assert not item.get("stored_path")
    with pytest.raises(UploadStateError):
        store.set_label(item["id"], "no_tumor")
    assert service.admit([item["id"]]) == []


def test_admission_rechecks_legacy_staged_content(service, store):
    item = service.ingest_files([IncomingFile("a.png", make_image(202), "glioma")])["items"][0]
    service.content_check = lambda data: ["Not MRI-like"]
    assert service.admit([item["id"]]) == []
    assert store.get(item["id"])["status"] == "rejected"


def test_admission_rejects_modified_stored_image(service, store):
    from pathlib import Path
    item = service.ingest_files([IncomingFile("a.png", make_image(203), "glioma")])["items"][0]
    Path(item["stored_path"]).write_bytes(make_image(204))
    assert service.admit([item["id"]]) == []
    assert "changed since upload" in store.get(item["id"])["reason"]


def test_screening_unavailable_blocks_staging_and_admission(service, store):
    item = service.ingest_files([IncomingFile("a.png", make_image(205), "glioma")])["items"][0]
    service.content_check = None
    with pytest.raises(ContentScreeningUnavailable):
        service.ingest_files([IncomingFile("b.png", make_image(206), "glioma")])
    with pytest.raises(ContentScreeningUnavailable):
        service.admit([item["id"]])
    assert not store.admitted()


@pytest.mark.parametrize("count", [15591, 16000])
def test_large_zip_within_limits_passes_archive_validation(config, count):
    limits = UploadLimits.from_config(config)
    data = _zip({f"glioma/{i}.jpg": b"image" for i in range(count)})
    entries, skipped = read_zip(data, limits, max_files=limits.max_files_per_batch)
    assert len(entries) == count and not skipped


def test_oversized_batch_is_rejected_before_decompression(service, monkeypatch):
    # Keep the file limit below the entry limit to exercise both guards independently.
    service.limits = replace(service.limits, max_files_per_batch=2)
    data = _zip({f"glioma/{i}.jpg": b"image" for i in range(service.limits.max_files_per_batch + 1)})

    def unexpected_read(*args, **kwargs):
        pytest.fail("Oversized archive was decompressed before checking its file count")

    monkeypatch.setattr(zipfile.ZipFile, "open", unexpected_read)
    with pytest.raises(UnsafeArchiveError, match="batch limit"):
        service.ingest_zip(data)
