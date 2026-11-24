"""Reuse the custom model's calibrated input gate without assigning training labels."""

from collections.abc import Callable

from src.ingestion.service import ContentScreeningUnavailable
from src.prediction import ModelUnavailableError, Predictor


def screen_content(get_predictor: Callable[[], Predictor], data: bytes) -> list[str]:
    try:
        predictor = get_predictor()
        if predictor.gate is None:
            raise ContentScreeningUnavailable("The active model needs input-gate calibration before uploads can be screened.")
        result = predictor.predict_bytes(data)
    except ModelUnavailableError as exc:
        raise ContentScreeningUnavailable("The active model is unavailable for upload screening.") from exc
    return [] if result.accepted else result.rejection_reasons
