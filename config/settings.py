"""Configuration loading: pipeline YAML (behaviour) and environment variables (deployment)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # Optional convenience for local runs; Docker passes variables directly.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"


@lru_cache(maxsize=4)
def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML pipeline configuration (cached; treat the result as read-only)."""
    config_path = Path(path or os.environ.get("PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG))
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class Settings:
    """Deployment settings resolved from environment variables."""

    data_dir: Path      # prepared dataset: manifests, split images, caches
    models_dir: Path    # model registry
    reports_dir: Path   # evaluation and analysis artifacts
    admin_api_key: str
    api_url: str
    log_level: str
    tf_threads: int
    # Runtime state written by the services (SQLite, staged uploads, job logs). Kept
    # separate so Docker can place it on a named volume with reliable file locking.
    state_dir: Path | None = None

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def runtime_dir(self) -> Path:
        return self.state_dir or self.data_dir / "runtime"

    @property
    def staging_dir(self) -> Path:
        return self.runtime_dir / "staging"

    @property
    def job_logs_dir(self) -> Path:
        return self.runtime_dir / "jobs"

    @property
    def database_path(self) -> Path:
        return self.runtime_dir / "app.db"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return (REPO_ROOT / value).resolve() if value and not Path(value).is_absolute() else Path(value or default)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings once per process. ``.env`` is loaded if present, without overriding real env vars."""
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings(
        data_dir=_path_from_env("APP_DATA_DIR", REPO_ROOT / "data"),
        models_dir=_path_from_env("MODELS_DIR", REPO_ROOT / "models"),
        reports_dir=_path_from_env("REPORTS_DIR", REPO_ROOT / "reports"),
        admin_api_key=os.environ.get("ADMIN_API_KEY", ""),
        api_url=os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        tf_threads=int(os.environ.get("TF_NUM_THREADS", "0")),
        state_dir=_path_from_env("APP_STATE_DIR", REPO_ROOT / "data" / "runtime"),
    )
