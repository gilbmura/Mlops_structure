"""Long-lived objects shared by request handlers, built once per API process."""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC

from config.settings import Settings
from src.dataset.manifests import ManifestError
from src.ingestion.factory import build_ingestion
from src.ingestion.limits import UploadLimits
from src.ingestion.service import IngestionService
from src.prediction import ActiveModelProvider
from src.registry import ModelRegistry
from src.storage.database import Database
from src.storage.jobs import JobStore
from src.storage.probes import ProbeStore
from src.storage.uploads import UploadStore

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    config: dict
    registry: ModelRegistry
    models: ActiveModelProvider
    jobs: JobStore
    uploads: UploadStore
    probes: ProbeStore
    limits: UploadLimits
    ingestion: IngestionService | None
    ingestion_error: str | None = None
    hostname: str = field(default_factory=socket.gethostname)
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 1)


def build_services(settings: Settings, config: dict) -> Services:
    database = Database(settings.database_path)
    registry = ModelRegistry(settings.models_dir)
    limits = UploadLimits.from_config(config)
    uploads = UploadStore(database)
    models = ActiveModelProvider(registry, config, limits.image)
    ingestion, ingestion_error = None, None
    try:
        ingestion = build_ingestion(settings, config, uploads, models.get)
    except ManifestError as exc:  # uploads need the dataset index to screen for leakage
        ingestion_error = str(exc)
        logger.warning("Uploads disabled: %s", exc)
    return Services(
        settings=settings,
        config=config,
        registry=registry,
        models=models,
        jobs=JobStore(database, settings.job_logs_dir),
        uploads=uploads,
        probes=ProbeStore(database),
        limits=limits,
        ingestion=ingestion,
        ingestion_error=ingestion_error,
    )
