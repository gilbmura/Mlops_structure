"""Upload size and count limits, loaded from the ``uploads`` section of pipeline.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from utils.images import ImageLimits


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int
    max_files_per_batch: int
    max_batch_bytes: int
    max_zip_entries: int
    max_zip_uncompressed_bytes: int
    max_compression_ratio: float
    image_min_side: int
    image_max_side: int

    @classmethod
    def from_config(cls, config: dict) -> UploadLimits:
        return cls(**config["uploads"])

    @property
    def image(self) -> ImageLimits:
        return ImageLimits(self.image_min_side, self.image_max_side, self.image_max_side**2)
