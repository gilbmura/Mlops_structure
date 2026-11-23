"""Input gate: rule checks, feature-distance calibration, refusal without a diagnosis, registry storage."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.input_gate import GATE_FILE, FeatureDistance, GateRules, border_mean, rule_violations
from src.labels import CLASS_NAMES
from src.model import build_model
from src.prediction import REFUSED_MESSAGE, Predictor
from src.preprocessing import PreprocessingSpec, standardize
from src.registry import ModelRegistry, RegistryError
from tests.helpers import make_image
from utils.images import decode_image

SPEC = PreprocessingSpec(image_size=32)


def _rules(config: dict) -> GateRules:
    return GateRules.from_config(config)


def _check(data: bytes, config: dict) -> list[str]:
    image = decode_image(data)
    return rule_violations(image, standardize(image, PreprocessingSpec(image_size=128)), _rules(config))


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_mri_like_greyscale_image_passes_rules(config) -> None:
    assert _check(make_image(1, size=(200, 180)), config) == []


def test_colour_bright_border_and_blank_images_are_refused(config) -> None:
    colour = Image.new("RGB", (200, 200), (30, 120, 200))
    assert any("colour" in r for r in _check(_png(colour), config))
    bright_border = Image.new("L", (200, 200), 230)
    bright_border.paste(0, (60, 60, 140, 140))
    assert any("edges are bright" in r for r in _check(_png(bright_border), config))
    assert any("uniform" in r for r in _check(_png(Image.new("L", (200, 200), 128)), config))
    assert any("shape" in r for r in _check(make_image(2, size=(400, 120)), config))


def test_border_mean_measures_only_the_edge_ring() -> None:
    image = np.zeros((128, 128), dtype=np.uint8)
    image[6:-6, 6:-6] = 255
    assert border_mean(image, 6) == 0.0


def test_feature_distance_threshold_and_round_trip() -> None:
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, (4, 16))
    labels = np.repeat(np.arange(4), 300)
    train = centres[labels] + rng.normal(0, 1, (1200, 16))
    validation = centres[np.repeat(np.arange(4), 100)] + rng.normal(0, 1, (400, 16))
    gate = FeatureDistance.fit(train, labels, validation, percentile=99.0)
    assert gate.summary["validation_refused_fraction"] == pytest.approx(0.01, abs=0.006)
    far = centres[:1] + 15.0
    assert gate.distances(far)[0] > gate.threshold  # far from every class mean
    assert gate.distances(validation[:5]).max() < gate.distances(far)[0]
    restored = FeatureDistance.from_bytes(gate.to_bytes())
    np.testing.assert_allclose(restored.distances(validation), gate.distances(validation))
    assert restored.threshold == gate.threshold and restored.summary == gate.summary


def _predictor(config: dict, threshold: float) -> Predictor:
    model = build_model("baseline", 32, 1e-3)
    features = 32  # the baseline's penultimate width
    gate = FeatureDistance(np.zeros((len(CLASS_NAMES), features)), np.eye(features), threshold, {})
    metadata = {"version": "vtest", "class_names": list(CLASS_NAMES), "preprocessing": SPEC.to_dict()}
    return Predictor(model, metadata, config, gate)


def test_refused_prediction_carries_no_diagnosis(config) -> None:
    refused = _predictor(config, threshold=-1.0).predict_bytes(make_image(3))  # every image is "too far"
    assert not refused.accepted and refused.message == REFUSED_MESSAGE
    assert (refused.predicted_class, refused.tumor_detected, refused.confidence, refused.probabilities) == (None, None, None, None)
    assert refused.rejection_reasons and refused.feature_distance is not None

    by_rule = _predictor(config, threshold=np.inf).predict_bytes(_png(Image.new("RGB", (200, 200), (200, 30, 30))))
    assert not by_rule.accepted and by_rule.feature_distance is None  # rules refuse before the model runs


def test_accepted_prediction_has_class_and_distance(config) -> None:
    accepted = _predictor(config, threshold=np.inf).predict_bytes(make_image(4))
    assert accepted.accepted and accepted.predicted_class in CLASS_NAMES
    assert abs(sum(accepted.probabilities.values()) - 1) < 1e-3 and accepted.feature_distance >= 0


def test_registry_stores_calibration_but_protects_artifacts(settings) -> None:
    registry = ModelRegistry(settings.models_dir)
    metadata = {"class_names": list(CLASS_NAMES), "preprocessing": SPEC.to_dict()}
    version = registry.register(build_model("baseline", 32, 1e-3), metadata, {}, np.zeros((2, 32, 32, 1), np.float32), {GATE_FILE: b"calibration"})
    assert registry.read_attachment(version, GATE_FILE) == b"calibration"
    registry.attach(version, GATE_FILE, b"recalibrated")
    assert registry.read_attachment(version, GATE_FILE) == b"recalibrated"
    for protected in ("model.keras", "model.h5", "metadata.json", "../escape.npz"):
        with pytest.raises(RegistryError):
            registry.attach(version, protected, b"x")
