"""Initial training and continued training with validation-based model selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import keras
import numpy as np

from src.augmentation import build_augmenter, make_dataset
from src.labels import CLASS_NAMES
from src.model import build_model, compile_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingOutcome:
    model: keras.Model
    history: dict[str, list[float]]
    best_epoch: int
    epochs_run: int
    class_weights: dict[str, float]
    duration_seconds: float


def compute_class_weights(labels: np.ndarray, num_classes: int = len(CLASS_NAMES)) -> dict[int, float]:
    """Inverse-frequency ("balanced") weights from *training* labels only."""
    counts = np.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        missing = [CLASS_NAMES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"Training data has no images for: {', '.join(missing)}.")
    return {index: float(len(labels) / (num_classes * count)) for index, count in enumerate(counts)}


def subsample_per_class(images: np.ndarray, labels: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random, class-balanced subset used by the fast smoke run."""
    rng = np.random.default_rng(seed)
    picks = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(len(CLASS_NAMES))]
    index = np.sort(np.concatenate(picks))
    return images[index], labels[index]


class _EpochLogger(keras.callbacks.Callback):
    """Write one concise line per epoch to the module logger (and so to job logs)."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        summary = " ".join(f"{key}={value:.4f}" for key, value in sorted(logs.items()) if np.isscalar(value))
        logger.info("epoch %d: %s", epoch + 1, summary)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    architecture: str | None = None,
    smoke: bool = False,
    initial_model: keras.Model | None = None,
) -> TrainingOutcome:
    """Train from random weights or continue a loaded model; keep the best validation epoch.

    Augmentation is applied to the training dataset only. Early stopping and
    learning-rate reduction watch validation metrics; the test split is never seen.
    """
    section = config["training"]
    seed = int(config["dataset"]["seed"])
    architecture = architecture or section["architecture"]
    epochs = int(section["smoke"]["max_epochs"] if smoke else section["max_epochs"])
    keras.utils.set_random_seed(seed)

    if initial_model is None:
        model = build_model(architecture, int(config["preprocessing"]["image_size"]), float(section["learning_rate"]))
    else:
        model = initial_model
        size = int(config["preprocessing"]["image_size"])
        if model.input_shape != (None, size, size, 1) or model.output_shape != (None, len(CLASS_NAMES)):
            raise ValueError("Saved model shape does not match the preprocessing and four-class contract.")
        compile_model(model, float(section["learning_rate"]))
    batch_size = int(section["batch_size"])
    train_ds = make_dataset(x_train, y_train, batch_size, build_augmenter(config, seed), shuffle=True, seed=seed)
    val_ds = make_dataset(x_val, y_val, batch_size)
    class_weights = compute_class_weights(y_train) if section["class_weighting"] == "balanced" else None
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=section["monitor"], mode="max", patience=int(section["early_stopping_patience"]), restore_best_weights=True
    )
    callbacks = [
        early_stopping,
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=int(section["reduce_lr_patience"]), min_lr=1e-5),
        _EpochLogger(),
    ]

    logger.info("Training %s (%s): %d train / %d validation images, up to %d epochs", architecture,
                "saved custom weights" if initial_model is not None else "random weights", len(y_train), len(y_val), epochs)
    started = time.perf_counter()
    # train_ds already reshuffles every epoch, so Keras-level shuffling is disabled.
    fit = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights,
        callbacks=callbacks, shuffle=False, verbose=0,
    )
    history = {key: [float(v) for v in values] for key, values in fit.history.items()}
    return TrainingOutcome(
        model=model,
        history=history,
        best_epoch=int(early_stopping.best_epoch) + 1,
        epochs_run=len(history["loss"]),
        class_weights={CLASS_NAMES[i]: w for i, w in (class_weights or {}).items()},
        duration_seconds=round(time.perf_counter() - started, 1),
    )
