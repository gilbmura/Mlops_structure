"""Interpretable image features for exploratory analysis (hand-crafted; nothing learned).

All features are measured on the standardized 128x128 greyscale images the
model sees. Brightness, contrast, texture and entropy use *foreground* pixels
only (intensity above a background threshold), so differences in how much black
background surrounds the head do not masquerade as tissue differences. The
foreground fraction and whole-image mean are reported separately precisely to
check whether framing / acquisition could explain class patterns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FOREGROUND_THRESHOLD = 20  # 0..255; MRI background is near-black after letterboxing

FEATURES: dict[str, str] = {
    "brightness": "Mean foreground intensity (0-255)",
    "contrast": "Std. dev. of foreground intensity",
    "texture": "Mean gradient magnitude in foreground",
    "entropy": "Foreground intensity entropy (bits)",
    "foreground_fraction": "Share of non-background pixels",
    "whole_image_mean": "Mean intensity incl. background",
}


def image_features(image: np.ndarray) -> dict[str, float]:
    """Compute :data:`FEATURES` for one ``uint8`` image."""
    pixels = image.astype(np.float64)
    mask = pixels > FOREGROUND_THRESHOLD
    foreground = pixels[mask] if mask.any() else pixels.ravel()
    gy, gx = np.gradient(pixels)
    gradient = np.hypot(gx, gy)[mask] if mask.any() else np.hypot(gx, gy).ravel()
    histogram = np.bincount(foreground.astype(np.uint8) // 4, minlength=64) / foreground.size
    nonzero = histogram[histogram > 0]
    return {
        "brightness": float(foreground.mean()),
        "contrast": float(foreground.std()),
        "texture": float(gradient.mean()),
        "entropy": float(-(nonzero * np.log2(nonzero)).sum()),
        "foreground_fraction": float(mask.mean()),
        "whole_image_mean": float(pixels.mean()),
    }


def acquisition_metadata(samples_csv: Path) -> pd.DataFrame:
    """Per-sample plane and source-image geometry from ``samples.csv``.

    The original resolution and aspect ratio are proxies for the source
    collection and acquisition pipeline; if they track the class, a model can
    learn them as shortcuts instead of anatomy.
    """
    rows = pd.read_csv(samples_csv, usecols=["sample_id", "plane", "width", "height"])
    rows["source_resolution"] = np.where((rows["width"] == 512) & (rows["height"] == 512), "512x512", "other")
    rows["source_shape"] = np.where(rows["width"] == rows["height"], "square", "non_square")
    return rows[["sample_id", "plane", "source_resolution", "source_shape"]]


def feature_table(images: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per image: metadata columns (label, plane, ...) plus every feature."""
    values = pd.DataFrame([image_features(image) for image in images])
    return pd.concat([metadata.reset_index(drop=True), values], axis=1)


def summarize_by(table: pd.DataFrame, group: str) -> list[dict]:
    """Median, interquartile range, mean and std of each feature per group value."""
    rows = []
    for feature in FEATURES:
        for key, values in table.groupby(group)[feature]:
            rows.append({
                "feature": feature, group: key, "median": round(float(values.median()), 3),
                "q1": round(float(values.quantile(0.25)), 3), "q3": round(float(values.quantile(0.75)), 3),
                "mean": round(float(values.mean()), 3), "std": round(float(values.std()), 3), "n": int(values.size),
            })
    return rows


def class_separation(table: pd.DataFrame, group: str = "class") -> dict[str, dict]:
    """Kruskal-Wallis test per feature with epsilon-squared effect size (0 = no separation, 1 = perfect).

    With thousands of images almost any difference is "significant", so the
    effect size, not the p-value, is the useful number here.
    """
    result = {}
    samples = len(table)
    for feature in FEATURES:
        groups = [values.to_numpy() for _, values in table.groupby(group)[feature]]
        h_statistic, p_value = stats.kruskal(*groups)
        result[feature] = {"kruskal_h": round(float(h_statistic), 2), "p_value": float(p_value), "epsilon_squared": round(float(h_statistic / (samples - 1)), 4)}
    return result
