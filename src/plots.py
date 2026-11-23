"""Matplotlib figures shared by the evaluation scripts and the notebook.

Figures are built with the object-oriented ``Figure`` API (no pyplot state), so
scripts run headless and the notebook displays the returned figures inline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.labels import CLASS_NAMES

CLASS_COLOURS = {"glioma": "#0072B2", "meningioma": "#E69F00", "pituitary": "#009E73", "no_tumor": "#CC79A7"}


def _finish(fig: Figure, path: Path | None) -> Figure:
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def confusion_matrix(matrix: Sequence[Sequence[int]], title: str, path: Path | None = None) -> Figure:
    matrix = np.asarray(matrix)
    fig = Figure(figsize=(5.2, 4.6))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, int(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    return _finish(fig, path)


def training_curves(history: dict[str, list[float]], path: Path | None = None, title: str = "") -> Figure:
    metrics = [m for m in ("loss", "accuracy", "macro_f1") if m in history]
    fig = Figure(figsize=(4.2 * len(metrics), 3.4))
    axes = fig.subplots(1, len(metrics))
    epochs = range(1, len(history["loss"]) + 1)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        ax.plot(epochs, history[metric], label="training (augmented)")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend()
    if title:
        fig.suptitle(title)
    return _finish(fig, path)


def feature_boxplots(table: pd.DataFrame, features: dict[str, str], path: Path | None = None) -> Figure:
    columns = 3
    rows = int(np.ceil(len(features) / columns))
    fig = Figure(figsize=(4.4 * columns, 3.3 * rows))
    axes = fig.subplots(rows, columns)
    for ax, (feature, description) in zip(axes.ravel(), features.items()):
        data = [table.loc[table["class"] == name, feature] for name in CLASS_NAMES]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, name in zip(box["boxes"], CLASS_NAMES):
            patch.set_facecolor(CLASS_COLOURS[name])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(CLASS_NAMES) + 1), CLASS_NAMES, rotation=20)
        ax.set_title(description, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    return _finish(fig, path)


def image_grid(images: Sequence[np.ndarray], titles: Sequence[str], columns: int = 4, path: Path | None = None) -> Figure:
    rows = int(np.ceil(len(images) / columns))
    fig = Figure(figsize=(3.0 * columns, 3.2 * rows))
    axes = fig.subplots(rows, columns)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, image, title in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)
