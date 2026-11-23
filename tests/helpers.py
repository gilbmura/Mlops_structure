"""Test helpers shared across modules."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from config.settings import Settings

ADMIN_KEY = "test-admin-key"


def make_image(seed: int, size: tuple[int, int] = (160, 160), mode: str = "L", fmt: str = "PNG", label_index: int = 0) -> bytes:
    """A unique synthetic 'scan': dark background, bright ellipse, class-dependent blob, seeded noise."""
    rng = np.random.default_rng(seed)
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse((w * 0.1, h * 0.1, w * 0.9, h * 0.9), fill=50 + 45 * label_index)  # class-dependent intensity
    cx, cy = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.55), (0.5, 0.5)][label_index]
    radius = 0.12 if label_index < 3 else 0.0
    if radius:
        draw.ellipse((w * (cx - radius), h * (cy - radius), w * (cx + radius), h * (cy + radius)), fill=220)
    pixels = np.clip(np.asarray(image, dtype=np.int16) + rng.integers(-25, 25, (h, w)), 0, 255).astype(np.uint8)
    out = Image.fromarray(pixels, "L").convert(mode)
    buffer = io.BytesIO()
    out.save(buffer, format=fmt)
    return buffer.getvalue()


def require_artifacts(settings: Settings) -> None:
    from src.registry import ModelRegistry

    if not (settings.manifests_dir / "test.csv").is_file() or not ModelRegistry(settings.models_dir).active_version():
        pytest.skip("Prepared dataset or trained model missing; run scripts/prepare_dataset.py and scripts/train_model.py --activate.")
