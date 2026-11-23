"""Image fingerprints used for exact and near-duplicate detection.

Shared by dataset preparation and upload screening so both apply identical rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.preprocessing import letterbox
from utils.hashing import perceptual_hash, pixel_hash
from utils.images import to_grayscale


@dataclass(frozen=True)
class Fingerprint:
    """Identity of an image's content, independent of its file encoding."""

    pixel_hash: str        # SHA-256 of decoded greyscale pixels: equal => exact duplicate
    phash: int             # 64-bit DCT perceptual hash
    thumbnail: np.ndarray  # letterboxed greyscale thumbnail used to confirm hash matches
    width: int
    height: int
    mode: str


def fingerprint(image: Image.Image, thumbnail_size: int) -> Fingerprint:
    """Fingerprint a decoded image."""
    gray = to_grayscale(image)
    return Fingerprint(
        pixel_hash=pixel_hash(np.asarray(gray)),
        phash=perceptual_hash(gray),
        thumbnail=np.asarray(letterbox(gray, thumbnail_size), dtype=np.uint8),
        width=image.width,
        height=image.height,
        mode=image.mode,
    )
