"""Build the same screening service for API uploads and background ZIP jobs."""

from config.settings import Settings
from src.dataset.dedup import DuplicateRules
from src.dataset.manifests import load_similarity_index
from src.ingestion.content import screen_content
from src.ingestion.limits import UploadLimits
from src.ingestion.screening import DatasetIndex
from src.ingestion.service import IngestionService
from src.storage.uploads import UploadStore


def build_ingestion(settings: Settings, config: dict, uploads: UploadStore, get_predictor) -> IngestionService:
    index = DatasetIndex(load_similarity_index(settings.cache_dir / "similarity_index.npz"), DuplicateRules.from_config(config))
    return IngestionService(uploads, settings.staging_dir, index, UploadLimits.from_config(config),
                            int(config["duplicates"]["thumbnail_size"]), float(config["input_gate"]["max_channel_spread"]),
                            content_check=lambda data: screen_content(get_predictor, data))
