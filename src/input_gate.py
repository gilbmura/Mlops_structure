"""Decide whether an image resembles the training MRIs closely enough to be classified at all.

Two layers, both without pretrained models:

1. **Rules** on the decoded and standardized image: colour, extreme aspect ratio,
   near-blank content, and a bright border (MRI backgrounds are black). Thresholds
   were chosen from the BRISC training data (see ``config/pipeline.yaml``).
2. **Feature distance**: the trained CNN's penultimate features of the image are
   compared with the training images' features using a class-conditional Gaussian
   with shared covariance (Mahalanobis distance to the nearest class mean). The
   refusal threshold is a high percentile of the distances of *real validation
   MRIs*, so about 1 % of genuine slices are refused by design.

Neither layer can prove an image is a brain MRI; they make confident answers on
obviously unrelated images much less likely.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import keras
import numpy as np
from PIL import Image
from sklearn.covariance import LedoitWolf

from src.preprocessing import to_model_input
from utils.images import channel_spread

FEATURE_LAYER = "probabilities"  # features are the input of the final softmax layer
GATE_FILE = "input_gate.npz"


@dataclass(frozen=True)
class GateRules:
    max_channel_spread: float
    max_aspect_ratio: float
    min_contrast_std: float
    border_width: int
    max_border_mean: float

    @classmethod
    def from_config(cls, config: dict) -> GateRules:
        section = config["input_gate"]
        return cls(**{name: section[name] for name in cls.__dataclass_fields__})


def border_mean(standardized: np.ndarray, width: int) -> float:
    """Mean intensity of the outer ``width``-pixel ring of a standardized ``uint8`` image."""
    mask = np.ones(standardized.shape, dtype=bool)
    mask[width:-width, width:-width] = False
    return float(standardized[mask].mean())


def rule_violations(image: Image.Image, standardized: np.ndarray, rules: GateRules) -> list[str]:
    """Human-readable reasons why an image cannot be a BRISC-like MRI slice (empty list = passes)."""
    reasons = []
    if channel_spread(image) > rules.max_channel_spread:
        reasons.append("It is a colour image; MRI slices are greyscale.")
    ratio = max(image.size) / min(image.size)
    if ratio > rules.max_aspect_ratio:
        reasons.append(f"Its shape ({ratio:.1f}:1) is far from the roughly square MRI slices.")
    if float(standardized.std()) < rules.min_contrast_std:
        reasons.append("It is almost uniform (no visible structure).")
    border = border_mean(standardized, rules.border_width)
    if border > rules.max_border_mean:
        reasons.append(f"Its edges are bright (mean {border:.0f}/255); MRI slices sit on a black background.")
    return reasons


def feature_model(model: keras.Model) -> keras.Model:
    """Same network, returning (penultimate features, class probabilities) in one pass."""
    return keras.Model(model.inputs, [model.get_layer(FEATURE_LAYER).input, model.outputs[0]])


@dataclass
class FeatureDistance:
    """Mahalanobis distance from an image's features to the nearest training class mean."""

    class_means: np.ndarray  # (classes, features)
    precision: np.ndarray    # (features, features), inverse of the shared covariance
    threshold: float
    summary: dict

    @classmethod
    def fit(cls, train_features: np.ndarray, train_labels: np.ndarray, validation_features: np.ndarray, percentile: float) -> FeatureDistance:
        """Fit on training features; set the threshold from *validation* distances."""
        classes = np.unique(train_labels)
        means = np.stack([train_features[train_labels == c].mean(axis=0) for c in classes])
        # Ledoit-Wolf shrinkage keeps the covariance invertible and stable with ~4k samples in 128 dimensions.
        covariance = LedoitWolf().fit(train_features - means[np.searchsorted(classes, train_labels)])
        gate = cls(means, covariance.precision_, 0.0, {})
        validation = gate.distances(validation_features)
        gate.threshold = float(np.percentile(validation, percentile))
        gate.summary = {
            "method": "min class-conditional Mahalanobis distance on penultimate CNN features (Ledoit-Wolf shared covariance)",
            "threshold": round(gate.threshold, 4),
            "threshold_percentile_of_validation": percentile,
            "train_images": int(len(train_features)),
            "validation_images": int(len(validation_features)),
            "validation_refused_fraction": round(float((validation > gate.threshold).mean()), 4),
            "train_refused_fraction": round(float((gate.distances(train_features) > gate.threshold).mean()), 4),
            "covariance_shrinkage": round(float(covariance.shrinkage_), 6),
        }
        return gate

    def distances(self, features: np.ndarray) -> np.ndarray:
        offsets = features[:, None, :] - self.class_means[None]
        return np.sqrt(np.einsum("nkd,de,nke->nk", offsets, self.precision, offsets).min(axis=1))

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(buffer, class_means=self.class_means, precision=self.precision,
                 threshold=np.float64(self.threshold), summary=np.array(json.dumps(self.summary)))
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> FeatureDistance:
        with np.load(io.BytesIO(data)) as arrays:
            return cls(arrays["class_means"], arrays["precision"], float(arrays["threshold"]), json.loads(str(arrays["summary"])))


def calibrate(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, percentile: float) -> FeatureDistance:
    """Fit the feature-distance gate for a trained model from standardized ``uint8`` arrays."""
    extractor = feature_model(model)
    train_features = extractor.predict(to_model_input(x_train), batch_size=128, verbose=0)[0]
    validation_features = extractor.predict(to_model_input(x_val), batch_size=128, verbose=0)[0]
    return FeatureDistance.fit(train_features, y_train, validation_features, percentile)
