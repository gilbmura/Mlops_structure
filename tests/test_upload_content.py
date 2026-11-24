"""Upload screening must require calibration and never supply class labels."""

from types import SimpleNamespace

import pytest

from src.ingestion.content import screen_content
from src.ingestion.service import ContentScreeningUnavailable
from src.prediction import ModelUnavailableError


def test_uncalibrated_model_cannot_screen_uploads():
    with pytest.raises(ContentScreeningUnavailable, match="calibration"):
        screen_content(lambda: SimpleNamespace(gate=None), b"image")


def test_missing_model_cannot_screen_uploads():
    def unavailable():
        raise ModelUnavailableError("missing")
    with pytest.raises(ContentScreeningUnavailable, match="unavailable"):
        screen_content(unavailable, b"image")


@pytest.mark.parametrize("accepted", [True, False])
def test_screening_returns_reasons_only(accepted):
    result = SimpleNamespace(accepted=accepted, rejection_reasons=["Not MRI-like"])
    predictor = SimpleNamespace(gate=object(), predict_bytes=lambda data: result)
    assert screen_content(lambda: predictor, b"image") == ([] if accepted else ["Not MRI-like"])
