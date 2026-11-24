"""Safe, content-based image decoding shared by data preparation, uploads and inference."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = frozenset({"JPEG", "PNG"})


class ImageDecodeError(ValueError):
    """Raised when bytes are not a supported, decodable, reasonably sized image."""


@dataclass(frozen=True)
class ImageLimits:
    """Bounds applied before and after decoding, to avoid decompression bombs."""

    min_side: int = 64
    max_side: int = 4096
    max_pixels: int = 4096 * 4096


def decode_image(data: bytes, limits: ImageLimits = ImageLimits()) -> Image.Image:
    """Decode JPEG/PNG bytes by inspecting their content, never the file extension.

    The header is checked against ``limits`` *before* pixel data is loaded, so a
    tiny file that claims huge dimensions is rejected cheaply.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError("File content is not a recognised image.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise ImageDecodeError(f"Unsupported image format {image.format!r}; use JPEG or PNG.")
    width, height = image.size
    if min(width, height) < limits.min_side:
        raise ImageDecodeError(f"Image is {width}x{height}; each side must be at least {limits.min_side}px.")
    if max(width, height) > limits.max_side or width * height > limits.max_pixels:
        raise ImageDecodeError(f"Image is {width}x{height}; the maximum side is {limits.max_side}px.")
    try:
        image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"Image data is truncated or corrupt: {exc}") from exc
    oriented = ImageOps.exif_transpose(image)
    oriented.format = image.format  # exif_transpose returns a copy without the detected format
    return oriented


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert any supported mode (L, RGB, RGBA, P, I;16 ...) to 8-bit greyscale."""
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        # Rescale 16/32-bit intensities into 0..255 instead of clipping them.
        pixels = np.asarray(image, dtype=np.float64)
        span = float(pixels.max() - pixels.min()) or 1.0
        return Image.fromarray(((pixels - pixels.min()) / span * 255).astype(np.uint8), mode="L")
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("L")


def channel_spread(image: Image.Image) -> float:
    """Mean absolute difference between RGB channels (0 for greyscale images).

    MRI slices are greyscale; a large spread signals a colour photo or an
    annotated screenshot that the model was never trained on.
    """
    if image.mode not in ("RGB", "RGBA"):
        return 0.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    return float(np.mean(np.abs(rgb - rgb.mean(axis=2, keepdims=True))))
