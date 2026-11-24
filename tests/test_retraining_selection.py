"""Queued selection cannot silently expand to other admitted batches."""

import pytest

from src.retraining import RetrainingError, _admitted_samples
from src.storage.database import Database
from src.storage.uploads import UploadStore


def test_worker_uses_only_snapshot_ids(settings, tmp_path):
    store = UploadStore(Database(settings.database_path))
    for uid in ("selected", "later"):
        path = tmp_path / f"{uid}.png"
        path.write_bytes(b"placeholder")
        store.add({"id": uid, "batch_id": uid, "original_name": path.name, "stored_path": str(path),
                   "label": "glioma", "status": "admitted"})
    assert [s.sample_id for s in _admitted_samples(store, ["selected"])] == ["upload-selected"]
    assert _admitted_samples(store, []) == []
    with pytest.raises(RetrainingError, match="no longer admitted"):
        _admitted_samples(store, ["missing"])
