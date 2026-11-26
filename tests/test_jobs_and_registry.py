"""Background-job lifecycle, worker failure handling, registry promotion and rollback."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.labels import CLASS_NAMES
from src.model import build_model
from src.prediction import ActiveModelProvider
from src.promotion import decide_promotion
from src.registry import ModelRegistry, RegistryError
from src.storage.database import Database
from src.storage.jobs import FAILED, QUEUED, RETRAIN, RUNNING, SUCCEEDED, JobConflictError, JobStore
from worker.main import process_one

PROBE = np.zeros((2, 32, 32, 1), dtype=np.float32)


@pytest.fixture
def jobs(settings) -> JobStore:
    return JobStore(Database(settings.database_path), settings.job_logs_dir)


def _metadata(macro_f1: float = 0.8) -> dict:
    return {
        "architecture": "baseline",
        "class_names": list(CLASS_NAMES),
        "preprocessing": {"image_size": 32, "resample": "bilinear", "channels": 1, "padding": "letterbox-black", "normalization": "divide-by-255"},
        "validation_metrics": {"macro_f1": macro_f1},
    }


def test_only_one_active_job_per_kind(jobs: JobStore) -> None:
    job = jobs.create(RETRAIN)
    assert job["status"] == QUEUED
    with pytest.raises(JobConflictError):
        jobs.create(RETRAIN)
    claimed = jobs.claim_next()
    assert claimed["id"] == job["id"] and claimed["status"] == RUNNING
    with pytest.raises(JobConflictError):  # still blocked while running
        jobs.create(RETRAIN)
    jobs.finish(job["id"], SUCCEEDED, result={"ok": True})
    assert jobs.create(RETRAIN)["status"] == QUEUED  # allowed again once finished


def test_worker_records_failures_with_logs(jobs: JobStore) -> None:
    job = jobs.create(RETRAIN)

    def explode(_job: dict) -> dict:
        raise RuntimeError("training data missing")

    assert process_one(jobs, {RETRAIN: explode}) is True
    failed = jobs.get(job["id"])
    assert failed["status"] == FAILED and "training data missing" in failed["error"]
    assert "Traceback" in jobs.read_log(job["id"])
    assert process_one(jobs, {RETRAIN: explode}) is False  # queue is empty


def test_worker_success_and_orphan_recovery(jobs: JobStore) -> None:
    job = jobs.create(RETRAIN)
    process_one(jobs, {RETRAIN: lambda j: {"candidate_version": "v009"}})
    assert jobs.get(job["id"])["result"] == {"candidate_version": "v009"}
    orphan = jobs.create(RETRAIN)
    jobs.claim_next()
    assert jobs.fail_orphaned("worker restarted") == 1
    assert jobs.get(orphan["id"])["status"] == FAILED


def test_registry_saves_verified_artifacts_and_rolls_back(settings) -> None:
    registry = ModelRegistry(settings.models_dir)
    first = registry.register(build_model("baseline", 32, 1e-3), _metadata(), {"loss": [1.0]}, PROBE)
    second = registry.register(build_model("baseline", 32, 1e-3), _metadata(), {"loss": [1.0]}, PROBE)
    assert (first, second) == ("v001", "v002")
    for version in (first, second):
        folder = registry.version_dir(version)
        assert (folder / "model.keras").is_file() and (folder / "model.h5").is_file()
        assert registry.metadata(version)["reload_check"]["model.h5"]["max_abs_difference"] <= 1e-4

    registry.activate(first, "initial")
    registry.activate(second, "promoted")
    assert registry.state()["versions"][first]["status"] == "previous"
    assert registry.rollback("bad candidate") == first
    assert registry.active_version() == first


def test_rollback_without_history_fails(settings) -> None:
    with pytest.raises(RegistryError):
        ModelRegistry(settings.models_dir).rollback("nothing to roll back to")


def test_failed_save_leaves_no_partial_version(settings) -> None:
    registry = ModelRegistry(settings.models_dir)
    bad_probe = np.zeros((1, 16, 16, 1), dtype=np.float32)  # wrong shape -> reload check fails
    with pytest.raises(ValueError):  # Keras rejects the mismatched probe shape
        registry.register(build_model("baseline", 32, 1e-3), _metadata(), {}, bad_probe)
    assert registry.state()["versions"] == {}
    assert not any(p.name.startswith("v") for p in registry.versions_dir.iterdir())


def test_registry_rejects_wrong_class_order(settings) -> None:
    metadata = {**_metadata(), "class_names": list(reversed(CLASS_NAMES))}
    with pytest.raises(RegistryError):
        ModelRegistry(settings.models_dir).register(build_model("baseline", 32, 1e-3), metadata, {}, PROBE)


def _report(macro_f1: float, recalls: list[float]) -> dict:
    return {"macro_f1": macro_f1, "per_class": {name: {"recall": r} for name, r in zip(CLASS_NAMES, recalls)}}


def test_promotion_rules(config: dict) -> None:
    active = _report(0.90, [0.9, 0.9, 0.9, 0.9])
    assert decide_promotion(_report(0.895, [0.9, 0.88, 0.9, 0.9]), active, config).promote  # within margins
    assert not decide_promotion(_report(0.87, [0.9, 0.9, 0.9, 0.9]), active, config).promote  # macro-F1 too low
    worse_class = decide_promotion(_report(0.93, [0.95, 0.95, 0.95, 0.80]), active, config)
    assert not worse_class.promote and any("no_tumor" in r for r in worse_class.reasons)
    assert decide_promotion(_report(0.5, [0.5] * 4), None, config).promote


def test_active_model_provider_hot_swaps_and_survives_bad_artifacts(settings, config) -> None:
    registry = ModelRegistry(settings.models_dir)
    config = copy.deepcopy(config)
    config["serving"]["model_refresh_seconds"] = 0
    first = registry.register(build_model("baseline", 32, 1e-3), _metadata(), {}, PROBE)
    # Serving must work from HDF5 even when the separate training artifact is unreadable.
    (registry.version_dir(first) / "model.keras").write_bytes(b"corrupted")
    registry.activate(first, "initial")
    provider = ActiveModelProvider(registry, config)
    assert provider.get().version == first
    assert provider.get().artifact == "model.h5"

    second = registry.register(build_model("baseline", 32, 1e-3), _metadata(), {}, PROBE)
    registry.activate(second, "promote a valid HDF5 artifact")
    assert provider.get().version == second
    assert registry.rollback("verify HDF5 rollback") == first
    assert provider.get().version == first

    (registry.version_dir(second) / "model.h5").write_bytes(b"corrupted")
    registry.activate(second, "promote a corrupted HDF5 artifact")
    assert provider.get().version == first  # keeps serving the working model
    assert "Could not load" in provider.last_error
