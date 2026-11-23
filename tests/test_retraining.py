"""End-to-end retraining on a tiny synthetic dataset: one admitted image -> job -> candidate -> decision."""

from __future__ import annotations

import pytest
import keras
import numpy as np

from src.dataset.manifests import SPLIT_FIELDS
from src.labels import CLASS_NAMES
from src.pipeline import load_training_data, run_training
from utils.hashing import sha256_file
from src.registry import ModelRegistry
from src.retraining import RetrainingError, run_retraining
from src.storage.database import Database
from src.storage.jobs import FAILED, RETRAIN, SUCCEEDED, JobStore
from src.storage.uploads import UploadStore
from tests.helpers import make_image
from utils.files import write_csv, write_json
from worker.main import process_one


def _write_split(settings, split: str, per_class: int, seed: int) -> None:
    rows = []
    for index, label in enumerate(CLASS_NAMES):
        for n in range(per_class):
            relative = f"{split}/{label}/{split}_{label}_{n}.png"
            path = settings.data_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(make_image(seed + index * 100 + n, size=(96, 96), label_index=index))
            rows.append({"sample_id": f"{split}_{label}_{n}", "label": label, "plane": "axial", "path": relative, "group_id": "g", "flags": ""})
    write_csv(settings.manifests_dir / f"{split}.csv", rows, SPLIT_FIELDS)


@pytest.fixture
def tiny_dataset(settings):
    _write_split(settings, "train", 6, 0)
    _write_split(settings, "validation", 4, 5000)
    write_json(settings.manifests_dir / "dataset_card.json", {"dataset_version": "synthetic-test"})
    return settings


@pytest.mark.slow
def test_single_admitted_image_continues_saved_weights_and_is_judged(tiny_dataset, config, monkeypatch) -> None:
    settings = tiny_dataset
    # The small baseline (no batch norm) learns the synthetic intensity classes in a few dozen steps.
    config["training"].update(architecture="baseline", max_epochs=30, learning_rate=0.01)
    database = Database(settings.database_path)
    uploads, jobs, registry = UploadStore(database), JobStore(database, settings.job_logs_dir), ModelRegistry(settings.models_dir)
    initial = run_training(load_training_data(settings, config), config, registry)
    registry.activate(initial.version, "initial custom model trained from scratch")
    parent_weights = registry.load_model(initial.version).get_weights()
    parent_path = registry.version_dir(initial.version) / "model.keras"
    parent_checksum = sha256_file(parent_path)
    original_fit = keras.Model.fit
    starts = []

    def checked_fit(model, *args, **kwargs):
        assert int(model.optimizer.iterations.numpy()) == 0
        assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(config["retraining"]["learning_rate"])
        starts.append([weight.copy() for weight in model.get_weights()])
        return original_fit(model, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", checked_fit)
    config["training"]["max_epochs"] = 2

    image_path = settings.staging_dir / "batch" / "u1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(make_image(9999, size=(96, 96), label_index=2))
    uploads.add({"id": "u1", "batch_id": "batch", "original_name": "one.png", "label": "pituitary", "status": "staged", "stored_path": str(image_path)})
    assert uploads.admit(["u1"]) == ["u1"]

    handlers = {RETRAIN: lambda job: run_retraining(job["id"], job["params"], settings, config, registry, uploads)}
    job = jobs.create(RETRAIN, {"smoke": False})
    assert process_one(jobs, handlers)
    finished = jobs.get(job["id"])
    assert finished["status"] == SUCCEEDED, finished["error"]
    result = finished["result"]
    assert result["uploads_included"] == 1
    assert result["train_images"] == 6 * len(CLASS_NAMES) + 1  # full training split + the single upload
    assert result["parent_version"] == initial.version
    assert result["initialization"] == "saved_custom_weights"
    for expected, actual in zip(parent_weights, starts[0], strict=True):
        np.testing.assert_array_equal(expected, actual)
    assert sha256_file(parent_path) == parent_checksum
    assert registry.active_version() == (result["candidate_version"] if result["promoted"] else initial.version)
    metadata = registry.metadata(result["candidate_version"])
    assert metadata["dataset"]["upload_ids"] == ["upload-u1"] and metadata["job_id"] == job["id"]
    # Each candidate ships with its own input-gate calibration, recorded in metadata too.
    assert registry.read_attachment(result["candidate_version"], "input_gate.npz")
    assert metadata["input_gate"]["threshold"] > 0

    assert metadata["training"]["initialization"] == "saved_custom_weights"
    assert metadata["parent_artifact"]["sha256"] == parent_checksum
    assert any(not np.array_equal(a, b) for a, b in zip(parent_weights, registry.load_model(result["candidate_version"]).get_weights(), strict=True))
    active_before = registry.active_version()

    # A second retraining compares against the now-active model on the same validation split.
    second = jobs.create(RETRAIN, {"smoke": True})
    process_one(jobs, handlers)
    outcome = jobs.get(second["id"])["result"]
    assert outcome["previous_active_version"] == active_before
    assert outcome["active_validation"] is not None
    assert not outcome["promoted"]
    assert registry.active_version() == active_before
    if not outcome["promoted"]:
        assert registry.state()["versions"][outcome["candidate_version"]]["status"] == "rejected"


def test_missing_upload_file_fails_the_job_without_promotion(tiny_dataset, config) -> None:
    settings = tiny_dataset
    database = Database(settings.database_path)
    uploads, jobs, registry = UploadStore(database), JobStore(database, settings.job_logs_dir), ModelRegistry(settings.models_dir)
    uploads.add({"id": "gone", "batch_id": "b", "original_name": "x.png", "label": "glioma", "status": "staged", "stored_path": str(settings.staging_dir / "missing.png")})
    uploads.admit(["gone"])
    with pytest.raises(RetrainingError):
        run_retraining("manual", {}, settings, config, registry, uploads)

    job = jobs.create(RETRAIN)
    process_one(jobs, {RETRAIN: lambda j: run_retraining(j["id"], j["params"], settings, config, registry, uploads)})
    assert jobs.get(job["id"])["status"] == FAILED
    assert registry.active_version() is None and registry.state()["versions"] == {}


def test_retraining_requires_reviewed_uploads_and_initial_model(tiny_dataset, config) -> None:
    settings = tiny_dataset
    uploads = UploadStore(Database(settings.database_path))
    registry = ModelRegistry(settings.models_dir)
    with pytest.raises(RetrainingError, match="reviewed upload"):
        run_retraining("empty", {}, settings, config, registry, uploads)
    path = settings.data_dir / "train/glioma/train_glioma_0.png"
    uploads.add({"id": "one", "batch_id": "b", "original_name": "one.png", "label": "glioma",
                 "status": "staged", "stored_path": str(path)})
    uploads.admit(["one"])
    with pytest.raises(RetrainingError, match="initial custom model"):
        run_retraining("no-parent", {}, settings, config, registry, uploads)


@pytest.mark.parametrize("problem", ["class_names", "preprocessing", "smoke"])
def test_incompatible_parent_is_rejected_before_training(tiny_dataset, config, monkeypatch, problem) -> None:
    from src.preprocessing import PreprocessingSpec

    settings = tiny_dataset
    uploads = UploadStore(Database(settings.database_path))
    path = settings.data_dir / "train/glioma/train_glioma_0.png"
    uploads.add({"id": "one", "batch_id": "b", "original_name": "one.png", "label": "glioma",
                 "status": "staged", "stored_path": str(path)})
    uploads.admit(["one"])
    registry = ModelRegistry(settings.models_dir)
    metadata = {"class_names": list(CLASS_NAMES), "preprocessing": PreprocessingSpec.from_config(config).to_dict(),
                "training": {"smoke": False}}
    if problem == "smoke":
        metadata["training"]["smoke"] = True
    else:
        metadata[problem] = []
    monkeypatch.setattr(registry, "active_version", lambda: "invalid")
    monkeypatch.setattr(registry, "metadata", lambda version: metadata)
    with pytest.raises(RetrainingError):
        run_retraining("invalid-parent", {}, settings, config, registry, uploads)
    assert registry.state()["versions"] == {}
