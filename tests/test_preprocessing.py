"""Shared preprocessing, safe decoding and the stable label contract."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.augmentation import make_dataset
from src.labels import CLASS_NAMES, CLASS_TO_INDEX, UnknownLabelError, normalize_label, tumor_status_message
from src.preprocessing import PreprocessingSpec, letterbox, preprocess_bytes, standardize
from tests.helpers import make_image
from utils.images import ImageDecodeError, ImageLimits, decode_image

SPEC = PreprocessingSpec(image_size=128)


def test_class_order_is_fixed() -> None:
    # The model's output indices depend on this exact order; changing it breaks every saved model.
    assert CLASS_NAMES == ("glioma", "meningioma", "pituitary", "no_tumor")
    assert [CLASS_TO_INDEX[name] for name in CLASS_NAMES] == [0, 1, 2, 3]


@pytest.mark.parametrize("raw, expected", [("Glioma", "glioma"), (" no-tumor ", "no_tumor"), ("No Tumor", "no_tumor")])
def test_label_normalisation(raw: str, expected: str) -> None:
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", ["notumor", "astrocytoma", "", "tumor"])
def test_unknown_labels_are_rejected_not_added(raw: str) -> None:
    with pytest.raises(UnknownLabelError):
        normalize_label(raw)


def test_tumor_messages() -> None:
    assert tumor_status_message("no_tumor") == "No tumor detected in this image."
    assert tumor_status_message("pituitary") == "Tumor detected — predicted type: pituitary."


def test_grayscale_and_rgb_versions_standardize_identically() -> None:
    gray = decode_image(make_image(1, mode="L"))
    rgb = decode_image(make_image(1, mode="RGB"))
    assert np.array_equal(standardize(gray, SPEC), standardize(rgb, SPEC))


def test_letterbox_preserves_aspect_ratio_with_black_padding() -> None:
    wide = Image.new("L", (200, 100), 255)
    boxed = np.asarray(letterbox(wide, 128))
    assert boxed.shape == (128, 128)
    assert boxed[:20].max() == 0 and boxed[-20:].max() == 0  # padded rows are black
    assert boxed[64].min() > 200                              # image content fills the middle


def test_preprocess_bytes_is_deterministic_and_normalised() -> None:
    data = make_image(2, size=(300, 240), fmt="JPEG")
    first, _ = preprocess_bytes(data, SPEC)
    second, _ = preprocess_bytes(data, SPEC)
    assert first.shape == (1, 128, 128, 1) and first.dtype == np.float32
    assert 0.0 <= first.min() and first.max() <= 1.0
    assert np.array_equal(first, second)


def test_evaluation_batches_are_not_augmented() -> None:
    images = np.stack([standardize(decode_image(make_image(i)), SPEC) for i in range(4)])
    batch, labels = next(iter(make_dataset(images, np.arange(4), batch_size=4)))
    np.testing.assert_allclose(batch.numpy()[..., 0], images / 255.0, atol=1e-6)
    assert labels.numpy().argmax(axis=1).tolist() == [0, 1, 2, 3]


def test_decode_rejects_non_images() -> None:
    with pytest.raises(ImageDecodeError, match="not a recognised image"):
        decode_image(b"%PDF-1.4 not an image")


def test_decode_rejects_unsupported_formats() -> None:
    buffer = io.BytesIO()
    Image.new("L", (100, 100)).save(buffer, format="GIF")
    with pytest.raises(ImageDecodeError, match="Unsupported image format"):
        decode_image(buffer.getvalue())


def test_decode_rejects_tiny_and_oversized_dimensions() -> None:
    with pytest.raises(ImageDecodeError, match="at least"):
        decode_image(make_image(3, size=(40, 40)))
    with pytest.raises(ImageDecodeError, match="maximum side"):
        decode_image(make_image(3, size=(300, 300)), ImageLimits(max_side=256, max_pixels=256 * 256))


def test_decode_rejects_truncated_files() -> None:
    data = make_image(4, fmt="PNG")
    with pytest.raises(ImageDecodeError):
        decode_image(data[: len(data) // 2])
