"""Initial training uses random weights and cannot silently replace the active model."""

import pytest

from src.initial_training import run_initial_training
from src.registry import ModelRegistry
from src.storage.database import Database
from src.storage.jobs import TRAIN, RETRAIN, SUCCEEDED, JobConflictError, JobStore
from tests.test_retraining import _write_split
from utils.files import write_json


@pytest.mark.parametrize("smoke", [False, True])
def test_initial_training_registers_without_activating(settings, config, smoke):
    _write_split(settings, "train", 4, 0)
    _write_split(settings, "validation", 3, 5000)
    write_json(settings.manifests_dir / "dataset_card.json", {"dataset_version": "synthetic"})
    registry = ModelRegistry(settings.models_dir)
    result = run_initial_training("initial-job", {"architecture": "baseline", "smoke": smoke}, settings, config, registry)
    assert result["initialization"] == "random_weights" and not result["promoted"]
    assert registry.active_version() is None
    metadata = registry.metadata(result["candidate_version"])
    assert metadata["training"]["initialization"] == "random_weights"
    assert metadata["training"]["smoke"] == smoke
    assert metadata["dataset"]["uploaded_images"] == 0
    assert metadata["job_id"] == "initial-job"


@pytest.mark.parametrize("first,second", [(TRAIN, RETRAIN), (RETRAIN, TRAIN)])
def test_training_and_retraining_share_one_queue_slot(settings, first, second):
    store = JobStore(Database(settings.database_path), settings.job_logs_dir)
    job = store.create(first)
    with pytest.raises(JobConflictError):
        store.create(second)
    store.claim_next()
    with pytest.raises(JobConflictError):
        store.create(second)
    store.finish(job["id"], SUCCEEDED)
    assert store.create(second)["kind"] == second
