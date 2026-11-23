"""tf.data input pipelines and training-only augmentation.

Validation and test datasets are built with ``augmenter=None``, so they are
never augmented. Inputs arrive already standardized by :mod:`src.preprocessing`.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from src.labels import CLASS_NAMES


def build_augmenter(config: dict, seed: int) -> keras.Sequential:
    """Mild geometric and intensity changes plausible for re-positioned / re-scanned slices.

    Empty space introduced by rotation, zoom or shift is filled with black, which
    matches the MRI background and the letterbox padding.
    """
    section = config["augmentation"]
    steps: list[layers.Layer] = []
    if section.get("horizontal_flip"):
        steps.append(layers.RandomFlip("horizontal", seed=seed))
    steps += [
        layers.RandomRotation(section["rotation_degrees"] / 360.0, fill_mode="constant", seed=seed),
        layers.RandomZoom(section["zoom"], fill_mode="constant", seed=seed),
        layers.RandomTranslation(section["translation"], section["translation"], fill_mode="constant", seed=seed),
        layers.RandomContrast(section["contrast"], value_range=(0.0, 1.0), seed=seed),
        layers.RandomBrightness(section["brightness"], value_range=(0.0, 1.0), seed=seed),
    ]
    return keras.Sequential(steps, name="training_augmentation")


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    augmenter: keras.Sequential | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Batch ``uint8`` images ``(N, H, W)`` into float inputs in [0, 1] with one-hot labels."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)

    def to_inputs(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.expand_dims(tf.cast(x, tf.float32) / 255.0, -1)
        return x, tf.one_hot(y, len(CLASS_NAMES))

    dataset = dataset.batch(batch_size).map(to_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    if augmenter is not None:
        dataset = dataset.map(
            lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)
