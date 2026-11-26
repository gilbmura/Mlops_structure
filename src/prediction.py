"""Single-image prediction shared by the API and the notebook, plus hot model reloading.

Every image first passes the input gate (:mod:`src.input_gate`). If it fails any
check, the result is *refused*: no class, score or tumour message is returned,
only the reasons. This withholds a diagnosis for images that do not look like the
training MRIs; it cannot prove an accepted image is a brain MRI. Refused inputs
are never mapped to ``no_tumor``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field

import keras
import numpy as np
import tensorflow as tf

from src.input_gate import GATE_FILE, FeatureDistance, GateRules, feature_model, rule_violations
from src.labels import CLASS_NAMES, NO_TUMOR, tumor_status_message
from src.preprocessing import PreprocessingSpec, preprocess_bytes
from src.registry import H5_FILE, ModelRegistry
from utils.images import ImageLimits

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational prototype, not a diagnostic service. The confidence is the model's softmax score for its "
    "top class; it is not a clinical probability or a guarantee. Input checks refuse many images that are not "
    "brain MRI slices, but they cannot guarantee that an accepted image is one."
)
REFUSED_MESSAGE = "This doesn't look like a brain MRI slice, so no prediction is shown."


class ModelUnavailableError(RuntimeError):
    """Raised when no active model is registered or it cannot be loaded."""


@dataclass(frozen=True)
class Prediction:
    accepted: bool
    message: str
    model_version: str
    predicted_class: str | None = None
    tumor_detected: bool | None = None
    confidence: float | None = None
    low_confidence: bool | None = None
    probabilities: dict[str, float] | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    feature_distance: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """One loaded model version together with its preprocessing contract and input gate."""

    def __init__(
        self, model: keras.Model, metadata: dict, config: dict, gate: FeatureDistance | None,
        limits: ImageLimits = ImageLimits(), *, artifact: str | None = None,
    ) -> None:
        if list(metadata["class_names"]) != list(CLASS_NAMES):
            raise ModelUnavailableError(f"Model {metadata.get('version')} has class order {metadata['class_names']}, expected {list(CLASS_NAMES)}.")
        self.metadata = metadata
        self.version: str = metadata["version"]
        self.artifact = artifact
        self.spec = PreprocessingSpec.from_dict(metadata["preprocessing"])
        self.serving = config["serving"]
        self.rules = GateRules.from_config(config)
        self.gate = gate
        self.limits = limits
        if gate is None:
            logger.warning("Model %s has no %s; only rule-based input checks will run.", self.version, GATE_FILE)
        # One traced graph returns features and probabilities; it is safe to call from concurrent request threads.
        size = self.spec.image_size
        network = feature_model(model)
        self._infer = tf.function(
            lambda batch: network(batch, training=False),
            input_signature=[tf.TensorSpec([None, size, size, 1], tf.float32)],
        )

    @classmethod
    def from_registry(cls, registry: ModelRegistry, config: dict, version: str | None = None, limits: ImageLimits = ImageLimits()) -> Predictor:
        version = version or registry.active_version()
        if not version:
            raise ModelUnavailableError("No active model. Train one with: python scripts/train_model.py --activate")
        calibration = registry.read_attachment(version, GATE_FILE)
        gate = FeatureDistance.from_bytes(calibration) if calibration else None
        # Follow the active version's export, not the fixed top-level submission copy.
        return cls(registry.load_model(version, H5_FILE), registry.metadata(version), config, gate, limits, artifact=H5_FILE)

    def predict_bytes(self, data: bytes) -> Prediction:
        """Decode, standardize, gate and classify one image (raises ``ImageDecodeError`` for invalid files)."""
        batch, image = preprocess_bytes(data, self.spec, self.limits)
        standardized = np.round(batch[0, ..., 0] * 255.0).astype(np.uint8)
        reasons = rule_violations(image, standardized, self.rules)
        if reasons:
            return self._refused(reasons)

        features, probabilities = (output.numpy() for output in self._infer(tf.constant(batch)))
        distance = float(self.gate.distances(features)[0]) if self.gate else None
        if self.gate and distance > self.gate.threshold:
            return self._refused(
                [f"It does not resemble the MRI slices the model was trained on (feature distance {distance:.1f} > {self.gate.threshold:.1f})."],
                distance,
            )
        return self._accepted(probabilities[0], distance)

    def _refused(self, reasons: list[str], distance: float | None = None) -> Prediction:
        return Prediction(
            accepted=False, message=REFUSED_MESSAGE, model_version=self.version, rejection_reasons=reasons,
            feature_distance=None if distance is None else round(distance, 2),
        )

    def _accepted(self, probabilities: np.ndarray, distance: float | None) -> Prediction:
        index = int(np.argmax(probabilities))
        predicted = CLASS_NAMES[index]
        confidence = float(probabilities[index])
        return Prediction(
            accepted=True,
            message=tumor_status_message(predicted),
            model_version=self.version,
            predicted_class=predicted,
            tumor_detected=predicted != NO_TUMOR,
            confidence=round(confidence, 4),
            low_confidence=confidence < float(self.serving["low_confidence_threshold"]),
            probabilities={name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)},
            feature_distance=None if distance is None else round(distance, 2),
        )


class ActiveModelProvider:
    """Serves the registry's active model and swaps in a newly promoted one without downtime.

    The registry is re-checked at most every ``model_refresh_seconds``. A new model is
    loaded while the old one keeps answering; if loading fails, the old model
    stays in service and the error is logged.
    """

    def __init__(self, registry: ModelRegistry, config: dict, limits: ImageLimits = ImageLimits()) -> None:
        self.registry = registry
        self.config = config
        self.limits = limits
        self._predictor: Predictor | None = None
        self._lock = threading.Lock()
        self._next_check = 0.0
        self.last_error: str | None = None

    def get(self) -> Predictor:
        self._refresh_if_due()
        if self._predictor is None:
            raise ModelUnavailableError(self.last_error or "No active model is available.")
        return self._predictor

    @property
    def loaded_version(self) -> str | None:
        return self._predictor.version if self._predictor else None

    def _refresh_if_due(self) -> None:
        now = time.monotonic()
        if now < self._next_check:
            return
        with self._lock:
            if now < self._next_check:
                return
            self._next_check = now + float(self.config["serving"]["model_refresh_seconds"])
            try:
                active = self.registry.active_version()
                if active and active != self.loaded_version:
                    self._predictor = Predictor.from_registry(self.registry, self.config, active, self.limits)
                    self.last_error = None
                    logger.info("Serving model %s from %s", active, self._predictor.artifact)
                elif not active:
                    self.last_error = "No active model is registered."
            except Exception as exc:  # keep serving the previous model on any load failure
                self.last_error = f"Could not load active model: {exc}"
                logger.exception("Model refresh failed; keeping %s", self.loaded_version)
