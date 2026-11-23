"""Shared fixtures: isolated settings, a fast test config, and a small similarity index."""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest
from PIL import Image

from config.settings import Settings, get_settings, load_pipeline_config
from src.dataset.dedup import DuplicateRules
from src.dataset.fingerprint import fingerprint
from src.dataset.manifests import load_similarity_index, write_similarity_index
from src.ingestion.screening import DatasetIndex
from src.labels import CLASS_NAMES
from tests.helpers import ADMIN_KEY, make_image


@pytest.fixture
def config() -> dict:
    """Pipeline config shrunk for fast tests (tiny images, one short epoch)."""
    cfg = copy.deepcopy(load_pipeline_config())
    cfg["preprocessing"]["image_size"] = 32
    cfg["training"].update(batch_size=8, max_epochs=1)
    cfg["training"]["smoke"] = {"max_epochs": 1, "samples_per_class": 4}
    return cfg


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        admin_api_key=ADMIN_KEY,
        api_url="http://testserver",
        log_level="WARNING",
        tf_threads=0,
    )


@pytest.fixture
def indexed_images(settings: Settings, config: dict) -> dict[str, bytes]:
    """Write a similarity index with one synthetic image per split; return each image's bytes."""
    images = {split: make_image(1000 + i, label_index=i) for i, split in enumerate(["train", "validation", "test", "retrain_demo"])}
    size = int(config["duplicates"]["thumbnail_size"])
    entries = [
        (f"{split}-sample", split, CLASS_NAMES[0], fingerprint(Image.open(io.BytesIO(data)), size))
        for split, data in images.items()
    ]
    write_similarity_index(settings.cache_dir / "similarity_index.npz", entries)
    return images


@pytest.fixture
def dataset_index(settings: Settings, config: dict, indexed_images: dict) -> DatasetIndex:
    return DatasetIndex(load_similarity_index(settings.cache_dir / "similarity_index.npz"), DuplicateRules.from_config(config))


@pytest.fixture(scope="session")
def real_settings() -> Settings:
    """Settings of this checkout (prepared data + trained model), used read-only."""
    return get_settings()
