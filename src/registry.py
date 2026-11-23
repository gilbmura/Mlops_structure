"""File-based model registry with validated saving, activation and rollback.

Layout under ``models/``::

    registry.json              active version + activation history + version summaries
    versions/v001/model.keras  native Keras artifact (served)
    versions/v001/model.h5     HDF5 artifact (assignment deliverable; reload-tested)
    versions/v001/metadata.json, history.json
    versions/v001/input_gate.npz  calibration of the "does this look like a training MRI?" check

A version directory only appears after both artifacts were written *and*
reloaded successfully, so a partial or failed save can never be activated.
Only models produced by this registry are loaded; user-uploaded model files are
not accepted anywhere in the application.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, UTC
from pathlib import Path

import keras
import numpy as np

from src.labels import CLASS_NAMES
from utils.files import atomic_write_bytes, exclusive_lock, read_json, write_json
from utils.hashing import sha256_file

logger = logging.getLogger(__name__)

KERAS_FILE, H5_FILE = "model.keras", "model.h5"
_PROTECTED_FILES = frozenset({KERAS_FILE, H5_FILE, "metadata.json", "history.json"})


class RegistryError(RuntimeError):
    """Raised for invalid registry operations (unknown version, nothing to roll back to...)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.state_file = self.root / "registry.json"
        self._lock_file = self.root / ".registry.lock"

    # -- reading -----------------------------------------------------------------
    def state(self) -> dict:
        if not self.state_file.is_file():
            return {"active": None, "activations": [], "versions": {}}
        return read_json(self.state_file)

    def active_version(self) -> str | None:
        return self.state()["active"]

    def version_dir(self, version: str) -> Path:
        path = self.versions_dir / version
        if version not in self.state()["versions"] or not path.is_dir():
            raise RegistryError(f"Model version {version!r} is not registered.")
        return path

    def metadata(self, version: str) -> dict:
        return read_json(self.version_dir(version) / "metadata.json")

    def load_model(self, version: str, artifact: str = KERAS_FILE) -> keras.Model:
        """Load a registered artifact (safe mode: no arbitrary code deserialisation)."""
        return keras.models.load_model(self.version_dir(version) / artifact, compile=False, safe_mode=True)

    def read_attachment(self, version: str, name: str) -> bytes | None:
        """Bytes of an auxiliary file stored with a version, or ``None`` if it has none."""
        path = self.version_dir(version) / name
        return path.read_bytes() if path.is_file() else None

    # -- writing -----------------------------------------------------------------
    def attach(self, version: str, name: str, data: bytes) -> None:
        """Add or replace an auxiliary file (e.g. a calibration) for an existing version.

        Model artifacts, metadata and history are immutable once registered.
        """
        if name in _PROTECTED_FILES or Path(name).name != name:
            raise RegistryError(f"{name!r} cannot be attached; registered artifacts are immutable.")
        with exclusive_lock(self._lock_file):
            atomic_write_bytes(self.version_dir(version) / name, data)

    def register(
        self, model: keras.Model, metadata: dict, history: dict, probe_input: np.ndarray, attachments: dict[str, bytes] | None = None
    ) -> str:
        """Save both artifacts, verify they reload with identical predictions, then register.

        ``probe_input`` is a small float batch used to compare the reloaded models'
        outputs with the in-memory model. ``attachments`` (e.g. the input-gate
        calibration) are written into the same folder before it becomes visible.
        """
        if list(metadata.get("class_names", [])) != list(CLASS_NAMES):
            raise RegistryError("Model metadata class order does not match the label contract.")
        with exclusive_lock(self._lock_file):
            state = self.state()
            version = f"v{max((int(v[1:]) for v in state['versions']), default=0) + 1:03d}"
            staging = self.versions_dir / f".staging-{version}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            try:
                model.save(staging / KERAS_FILE)
                model.save(staging / H5_FILE)
                reload_check = self._verify_reload(model, staging, probe_input)
                metadata = {
                    **metadata,
                    "version": version,
                    "registered_at": _now(),
                    "artifacts": {
                        name: {"file": name, "sha256": sha256_file(staging / name)} for name in (KERAS_FILE, H5_FILE)
                    },
                    "reload_check": reload_check,
                }
                write_json(staging / "metadata.json", metadata)
                write_json(staging / "history.json", history)
                for name, data in (attachments or {}).items():
                    if name in _PROTECTED_FILES or Path(name).name != name:
                        raise RegistryError(f"Invalid attachment name {name!r}.")
                    (staging / name).write_bytes(data)
                staging.rename(self.versions_dir / version)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            state["versions"][version] = {
                "registered_at": metadata["registered_at"],
                "architecture": metadata.get("architecture"),
                "dataset_version": metadata.get("dataset", {}).get("dataset_version"),
                "validation_macro_f1": metadata.get("validation_metrics", {}).get("macro_f1"),
                "job_id": metadata.get("job_id"),
                "status": "candidate",
            }
            write_json(self.state_file, state)
        logger.info("Registered model %s", version)
        return version

    @staticmethod
    def _verify_reload(model: keras.Model, folder: Path, probe_input: np.ndarray) -> dict:
        expected = model.predict(probe_input, verbose=0)
        result = {}
        for name in (KERAS_FILE, H5_FILE):
            reloaded = keras.models.load_model(folder / name, compile=False, safe_mode=True)
            difference = float(np.max(np.abs(reloaded.predict(probe_input, verbose=0) - expected)))
            if difference > 1e-4:
                raise RegistryError(f"{name} reload changed predictions (max difference {difference:.2e}).")
            result[name] = {"max_abs_difference": difference, "probe_images": int(len(probe_input))}
        return result

    def activate(self, version: str, reason: str) -> None:
        """Make ``version`` the served model; the previous one stays available for rollback."""
        with exclusive_lock(self._lock_file):
            state = self.state()
            if version not in state["versions"]:
                raise RegistryError(f"Model version {version!r} is not registered.")
            previous = state["active"]
            if previous and previous in state["versions"]:
                state["versions"][previous]["status"] = "previous"
            state["versions"][version]["status"] = "active"
            state["active"] = version
            state["activations"].append({"version": version, "previous": previous, "at": _now(), "reason": reason})
            write_json(self.state_file, state)
        logger.info("Activated model %s (previous: %s): %s", version, previous, reason)

    def mark(self, version: str, status: str) -> None:
        """Record a non-active status such as ``rejected`` for a candidate."""
        with exclusive_lock(self._lock_file):
            state = self.state()
            state["versions"][version]["status"] = status
            write_json(self.state_file, state)

    def rollback(self, reason: str) -> str:
        """Re-activate the model that was active before the current one."""
        activations = self.state()["activations"]
        if not activations or not activations[-1]["previous"]:
            raise RegistryError("There is no previous model to roll back to.")
        target = activations[-1]["previous"]
        self.activate(target, f"rollback: {reason}")
        return target
