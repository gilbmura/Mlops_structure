"""Measure interpretable image features per class on the TRAINING split only.

Usage:
    python scripts/analyze_features.py

Writes reports/dataset/{features.csv, feature_summary.json, feature_boxplots.png}.
Validation and test images are deliberately excluded so exploratory analysis
cannot influence decisions with evaluation data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config.settings import get_settings, load_pipeline_config  # noqa: E402
from src import plots  # noqa: E402
from src.dataset.manifests import load_arrays, read_split  # noqa: E402
from src.dataset.splitting import TRAIN  # noqa: E402
from src.features import FEATURES, acquisition_metadata, class_separation, feature_table, summarize_by  # noqa: E402
from src.preprocessing import PreprocessingSpec  # noqa: E402
from utils.files import write_json  # noqa: E402
from utils.log import configure_logging  # noqa: E402


def _nested_medians(table: pd.DataFrame, features: list[str]) -> dict:
    """{feature: {class: {resolution: median}}} for combinations that exist (no NaN in JSON)."""
    result: dict = {}
    for feature in features:
        for (label, resolution), value in table.groupby(["class", "source_resolution"])[feature].median().items():
            result.setdefault(feature, {}).setdefault(label, {})[resolution] = round(float(value), 3)
    return result


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_pipeline_config()
    samples = read_split(settings.data_dir, TRAIN)
    images, _ = load_arrays(samples, PreprocessingSpec.from_config(config), settings.cache_dir)

    metadata = pd.DataFrame([{"sample_id": s.sample_id, "class": s.label} for s in samples]).merge(
        acquisition_metadata(settings.manifests_dir / "samples.csv"), on="sample_id", how="left"
    )
    table = feature_table(images, metadata)

    out = settings.reports_dir / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "features.csv", index=False)
    plots.feature_boxplots(table, FEATURES, out / "feature_boxplots.png")
    write_json(out / "feature_summary.json", {
        "split": TRAIN,
        "images": len(table),
        "features": FEATURES,
        "summary": summarize_by(table, "class"),
        "class_separation": class_separation(table),
        "by_plane": summarize_by(table, "plane"),
        "plane_by_class": pd.crosstab(table["class"], table["plane"]).to_dict(orient="index"),
        "source_resolution_by_class": pd.crosstab(table["class"], table["source_resolution"]).to_dict(orient="index"),
        "source_shape_by_class": pd.crosstab(table["class"], table["source_shape"]).to_dict(orient="index"),
        "median_by_class_and_resolution": _nested_medians(table, ["brightness", "contrast", "foreground_fraction"]),
    })
    print(table.groupby("class")[list(FEATURES)].median().round(3).to_string())
    print(pd.DataFrame(class_separation(table)).T.to_string())


if __name__ == "__main__":
    main()
