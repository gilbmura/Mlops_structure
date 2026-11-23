"""CNN architectures trained from randomly initialised weights (no pretrained parts).

* ``baseline``: two convolution layers and a linear classifier. It shows how far a
  minimal model gets and gives the final model something concrete to beat.
* ``cnn`` (final): four convolution blocks with batch normalisation and dropout.
  Stacked 3x3 convolutions widen the receptive field to whole-lesion scale at
  128 px, batch norm stabilises training from scratch, and global average
  pooling keeps the parameter count (~0.3M) small for CPU training and serving.
"""

from __future__ import annotations

from collections.abc import Callable

import keras
from keras import layers

from src.labels import CLASS_NAMES


def build_baseline(image_size: int, num_classes: int = len(CLASS_NAMES)) -> keras.Model:
    """Minimal CNN: Conv(16) -> Pool -> Conv(32) -> Pool -> global average -> softmax."""
    inputs = keras.Input(shape=(image_size, image_size, 1), name="image")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def _conv_block(x: keras.KerasTensor, filters: int, name: str) -> keras.KerasTensor:
    """Two 3x3 conv + batch-norm + ReLU layers followed by 2x2 max pooling."""
    for index in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{index}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{index}")(x)
    return layers.MaxPooling2D(name=f"{name}_pool")(x)


def build_cnn(image_size: int, num_classes: int = len(CLASS_NAMES), dropout: float = 0.3) -> keras.Model:
    """Final model: 4 conv blocks (16-32-64-128 filters) -> global average -> dropout -> softmax."""
    inputs = keras.Input(shape=(image_size, image_size, 1), name="image")
    x = inputs
    for depth, filters in enumerate((16, 32, 64, 128), start=1):
        x = _conv_block(x, filters, name=f"block{depth}")
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="brain_mri_cnn")


ARCHITECTURES: dict[str, Callable[..., keras.Model]] = {"baseline": build_baseline, "cnn": build_cnn}


def build_model(architecture: str, image_size: int, learning_rate: float) -> keras.Model:
    """Build and compile an architecture with categorical cross-entropy and macro-F1 tracking."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}.")
    model = ARCHITECTURES[architecture](image_size)
    compile_model(model, learning_rate)
    return model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """Attach a fresh optimizer without changing any learned model weights."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="macro_f1")],
    )
