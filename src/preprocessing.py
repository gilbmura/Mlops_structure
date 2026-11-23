"""Deterministic image preprocessing shared by training, the notebook and API inference.

Every image, whatever its size or colour mode, goes through the same steps:

1. decode by content (JPEG/PNG only) and apply EXIF orientation;
2. convert to 8-bit greyscale (BRISC RGB files have identical channels, so this is lossless for them);
3. letterbox: pad to a square with black borders, preserving aspect ratio;
4. resize to ``image_size`` x ``image_size``;
5. scale intensities to [0, 1] and add a channel axis.

Augmentation is deliberately *not* here; it lives in :mod:`src.augmentation` and is
applied to training batches only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.images import ImageLimits, decode_image, to_grayscale

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class PreprocessingSpec:
    """Everything needed to reproduce model inputs; saved in model metadata."""

    image_size: int = 128
    resample: str = "bilinear"
    channels: int = 1
    padding: str = "letterbox-black"
    normalization: str = "divide-by-255"

    @classmethod
    def from_config(cls, config: dict) -> PreprocessingSpec:
        section = config["preprocessing"]
        return cls(image_size=int(section["image_size"]), resample=section.get("resample", "bilinear"))

    @classmethod
    def from_dict(cls, values: dict) -> PreprocessingSpec:
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox(image: Image.Image, size: int, resample: str = "bilinear") -> Image.Image:
    """Greyscale, pad to a centred square with black borders, then resize to ``size``."""
    gray = to_grayscale(image)
    width, height = gray.size
    side = max(width, height)
    if width != height:
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
        gray = canvas
    return gray.resize((size, size), _RESAMPLING[resample])


def standardize(image: Image.Image, spec: PreprocessingSpec) -> np.ndarray:
    """Return the standardized ``uint8`` array of shape ``(size, size)``."""
    return np.asarray(letterbox(image, spec.image_size, spec.resample), dtype=np.uint8)


def to_model_input(images: np.ndarray) -> np.ndarray:
    """Convert ``uint8`` arrays ``(N, H, W)`` to float32 model input ``(N, H, W, 1)`` in [0, 1]."""
    return (np.asarray(images, dtype=np.float32) / 255.0)[..., np.newaxis]


def preprocess_bytes(data: bytes, spec: PreprocessingSpec, limits: ImageLimits = ImageLimits()) -> tuple[np.ndarray, Image.Image]:
    """Decode and standardize raw upload bytes.

    Returns the model-ready batch of one ``(1, H, W, 1)`` and the decoded image
    (used by callers for input sanity checks).
    """
    image = decode_image(data, limits)
    return to_model_input(standardize(image, spec)[np.newaxis]), image


def load_image_file(path: Path, spec: PreprocessingSpec, limits: ImageLimits = ImageLimits()) -> np.ndarray:
    """Read and standardize an image file from disk."""
    return standardize(decode_image(Path(path).read_bytes(), limits), spec)


def load_images(paths: Iterable[Path], spec: PreprocessingSpec, limits: ImageLimits = ImageLimits()) -> np.ndarray:
    """Standardize many files into one ``uint8`` array ``(N, H, W)``."""
    arrays = [load_image_file(path, spec, limits) for path in paths]
    if not arrays:
        return np.empty((0, spec.image_size, spec.image_size), dtype=np.uint8)
    return np.stack(arrays)
