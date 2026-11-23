"""Read split manifests, build cached training arrays, and store the similarity index."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.dataset.fingerprint import Fingerprint
from src.labels import CLASS_TO_INDEX
from src.preprocessing import PreprocessingSpec, load_images
from utils.files import read_csv

logger = logging.getLogger(__name__)

SPLIT_FIELDS = ["sample_id", "label", "plane", "path", "group_id", "flags"]


class ManifestError(RuntimeError):
    """Raised when prepared data is missing; the message says how to create it."""


@dataclass(frozen=True)
class Sample:
    """A labelled image file used for training or evaluation."""

    sample_id: str
    label: str
    path: Path


def read_split(data_dir: Path, split: str) -> list[Sample]:
    """Load one split manifest (paths are stored relative to ``data_dir``)."""
    manifest = Path(data_dir) / "manifests" / f"{split}.csv"
    if not manifest.is_file():
        raise ManifestError(f"{manifest} not found. Run: python scripts/prepare_dataset.py --archive <zip>")
    return [Sample(row["sample_id"], row["label"], Path(data_dir) / row["path"]) for row in read_csv(manifest)]


def samples_fingerprint(samples: Sequence[Sample]) -> str:
    """Stable hash of sample identities and labels, used for dataset versions and cache keys."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda s: s.sample_id):
        digest.update(f"{sample.sample_id}|{sample.label}\n".encode())
    return digest.hexdigest()


def load_arrays(samples: Sequence[Sample], spec: PreprocessingSpec, cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized ``uint8`` images ``(N, H, W)`` and integer labels.

    Arrays are cached under a key derived from the samples *and* the
    preprocessing spec, so a changed manifest or spec never reuses stale data.
    """
    key = hashlib.sha256((samples_fingerprint(samples) + json.dumps(spec.to_dict(), sort_keys=True)).encode()).hexdigest()[:16]
    cache_file = Path(cache_dir) / f"arrays-{key}.npz" if cache_dir else None
    if cache_file and cache_file.is_file():
        with np.load(cache_file) as cached:
            return cached["x"], cached["y"]
    logger.info("Preprocessing %d images (cache miss)", len(samples))
    x = load_images([s.path for s in samples], spec)
    y = np.array([CLASS_TO_INDEX[s.label] for s in samples], dtype=np.int64)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y)
    return x, y


def write_similarity_index(path: Path, entries: Sequence[tuple[str, str, str, Fingerprint]]) -> None:
    """Persist fingerprints of all retained dataset images for upload screening."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.array([e[0] for e in entries]),
        splits=np.array([e[1] for e in entries]),
        labels=np.array([e[2] for e in entries]),
        pixel_hashes=np.array([e[3].pixel_hash for e in entries]),
        phashes=np.array([e[3].phash for e in entries], dtype=np.uint64),
        thumbnails=np.stack([e[3].thumbnail for e in entries]),
    )


def load_similarity_index(path: Path) -> dict[str, np.ndarray]:
    """Load the arrays written by :func:`write_similarity_index`."""
    if not Path(path).is_file():
        raise ManifestError(f"Similarity index {path} not found. Run scripts/prepare_dataset.py first.")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}
