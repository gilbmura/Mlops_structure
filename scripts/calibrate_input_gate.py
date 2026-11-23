"""Calibrate the input gate for registered models and measure what it refuses.

Usage:
    python scripts/calibrate_input_gate.py                   # versions without a calibration
    python scripts/calibrate_input_gate.py --version v002 --force

New models are calibrated automatically when they are registered; this script
covers models registered before the gate existed. The threshold is fixed from
validation data; the test split and probe images are only *measured*:

* real MRIs: share of validation / test / retrain-demo slices refused;
* probes: everyday photos and maps shipped with matplotlib, synthetic noise and
  blank images (should be refused), plus brain MRIs that are not from BRISC
  (should ideally be accepted).

Writes reports/input_gate/<version>.json.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from config.settings import get_settings, load_pipeline_config  # noqa: E402
from src.dataset.manifests import load_arrays, read_split  # noqa: E402
from src.input_gate import GATE_FILE, calibrate  # noqa: E402
from src.prediction import Predictor  # noqa: E402
from src.preprocessing import PreprocessingSpec  # noqa: E402
from src.registry import ModelRegistry  # noqa: E402
from utils.files import write_json  # noqa: E402
from utils.log import configure_logging  # noqa: E402

SAMPLE_DATA = Path(matplotlib.get_data_path()) / "sample_data"


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def probe_images(settings) -> list[tuple[str, bool, bytes]]:
    """(name, should_be_accepted, PNG bytes) for images outside the BRISC training data."""
    portrait = Image.open(SAMPLE_DATA / "grace_hopper.jpg").convert("RGB")
    raw = np.frombuffer(gzip.open(SAMPLE_DATA / "s1045.ima.gz").read(), dtype=">u2").reshape(256, 256).astype(float)
    terrain = np.load(SAMPLE_DATA / "jacksboro_fault_dem.npz")["elevation"].astype(float)
    rng = np.random.default_rng(0)
    test_scan = Image.open(settings.data_dir / read_split(settings.data_dir, "test")[0].path).convert("L")
    return [
        ("portrait photo, colour", False, _png(portrait)),
        ("portrait photo, greyscale", False, _png(portrait.convert("L"))),
        ("product photo", False, (SAMPLE_DATA / "Minduka_Present_Blue_Pack.png").read_bytes()),
        ("matplotlib logo", False, (SAMPLE_DATA / "logo2.png").read_bytes()),
        ("terrain elevation map", False, _png(Image.fromarray(((terrain - terrain.min()) / np.ptp(terrain) * 255).astype(np.uint8)))),
        ("random noise", False, _png(Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8)))),
        ("dark random noise", False, _png(Image.fromarray(rng.integers(0, 60, (256, 256), dtype=np.uint8)))),
        ("blank grey image", False, _png(Image.new("L", (256, 256), 128))),
        ("brain MRI from another source (matplotlib s1045)", True, _png(Image.fromarray((raw / raw.max() * 255).astype(np.uint8)))),
        ("BRISC test slice rotated 90 degrees", True, _png(test_scan.rotate(90, expand=True))),
    ]


def _outcome(prediction) -> str:
    if prediction.accepted:
        return "accepted"
    return "refused: feature distance" if prediction.feature_distance is not None else "refused: rules"


def measure(predictor: Predictor, settings) -> dict:
    report = {"real_mri_refusals": {}, "probes": []}
    for split in ("validation", "test", "retrain_demo"):
        outcomes = Counter(_outcome(predictor.predict_bytes(sample.path.read_bytes())) for sample in read_split(settings.data_dir, split))
        total = sum(outcomes.values())
        report["real_mri_refusals"][split] = {"images": total, **dict(outcomes), "refused_fraction": round(1 - outcomes["accepted"] / total, 4)}
    for name, expected, data in probe_images(settings):
        result = predictor.predict_bytes(data)
        report["probes"].append({
            "probe": name, "expected": "accept" if expected else "refuse", "outcome": _outcome(result),
            "as_expected": result.accepted == expected, "feature_distance": result.feature_distance, "reasons": result.rejection_reasons,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="append", help="version(s) to calibrate (default: all without a calibration)")
    parser.add_argument("--force", action="store_true", help="recalibrate even if a calibration exists")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_pipeline_config()
    registry = ModelRegistry(settings.models_dir)
    versions = args.version or [v for v in registry.state()["versions"] if args.force or registry.read_attachment(v, GATE_FILE) is None]
    percentile = float(config["input_gate"]["feature_distance_percentile"])

    for version in versions:
        spec = PreprocessingSpec.from_dict(registry.metadata(version)["preprocessing"])
        x_train, y_train = load_arrays(read_split(settings.data_dir, "train"), spec, settings.cache_dir)
        x_val, _ = load_arrays(read_split(settings.data_dir, "validation"), spec, settings.cache_dir)
        gate = calibrate(registry.load_model(version), x_train, y_train, x_val, percentile)
        gate.summary["calibrated_by"] = "scripts/calibrate_input_gate.py (base training split)"
        registry.attach(version, GATE_FILE, gate.to_bytes())

        report = {"version": version, "rules": config["input_gate"], "calibration": gate.summary,
                  **measure(Predictor.from_registry(registry, config, version), settings)}
        write_json(settings.reports_dir / "input_gate" / f"{version}.json", report)
        print(json.dumps({"version": version, "threshold": gate.summary["threshold"], "real_mri_refusals": report["real_mri_refusals"],
                          "probes": {p["probe"]: p["outcome"] for p in report["probes"]}}, indent=2))


if __name__ == "__main__":
    main()
