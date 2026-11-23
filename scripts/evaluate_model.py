"""Evaluate a registered model on the validation or the final test split.

Usage:
    python scripts/evaluate_model.py --split validation              # development checks, any time
    python scripts/evaluate_model.py --split test                    # final, once per selected model
    python scripts/evaluate_model.py --split test --artifact model.h5  # same, loading the .h5 artifact

Writes reports/models/<version>/<split>_metrics.json, <split>_predictions.csv and figures.
A final-test report is never overwritten without --force, to discourage
repeatedly tuning against test results.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config.settings import get_settings, load_pipeline_config  # noqa: E402
from src import plots  # noqa: E402
from src.dataset.manifests import load_arrays, read_split  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402
from src.labels import CLASS_NAMES  # noqa: E402
from src.preprocessing import PreprocessingSpec  # noqa: E402
from src.registry import H5_FILE, KERAS_FILE, ModelRegistry  # noqa: E402
from utils.files import read_json, write_json  # noqa: E402
from utils.log import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="registered version (default: active)")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--artifact", choices=[KERAS_FILE, H5_FILE], default=KERAS_FILE)
    parser.add_argument("--force", action="store_true", help="overwrite an existing final-test report")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    registry = ModelRegistry(settings.models_dir)
    version = args.version or registry.active_version()
    if not version:
        sys.exit("No active model. Train one first: python scripts/train_model.py --activate")
    out = settings.reports_dir / "models" / version
    target = out / f"{args.split}_metrics.json"
    if args.split == "test" and target.is_file() and not args.force:
        sys.exit(f"{target} already exists. The test split is for a single final evaluation; pass --force to overwrite.")

    metadata = registry.metadata(version)
    spec = PreprocessingSpec.from_dict(metadata["preprocessing"])
    if spec != PreprocessingSpec.from_config(load_pipeline_config()):
        print("Note: evaluating with the preprocessing recorded in the model metadata.")
    samples = read_split(settings.data_dir, args.split)
    images, labels = load_arrays(samples, spec, settings.cache_dir)
    report, probabilities = evaluate_model(registry.load_model(version, args.artifact), images, labels)
    report.update({"version": version, "split": args.split, "artifact": args.artifact, "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds")})

    write_json(target, report)
    predictions = pd.DataFrame(probabilities.round(5), columns=[f"p_{c}" for c in CLASS_NAMES])
    predictions.insert(0, "sample_id", [s.sample_id for s in samples])
    predictions.insert(1, "label", [s.label for s in samples])
    predictions.insert(2, "predicted", [CLASS_NAMES[i] for i in probabilities.argmax(axis=1)])
    predictions.insert(3, "confidence", probabilities.max(axis=1).round(5))
    predictions.to_csv(out / f"{args.split}_predictions.csv", index=False)
    plots.confusion_matrix(report["confusion_matrix"], f"{version} — {args.split}", out / f"{args.split}_confusion_matrix.png")
    plots.training_curves(read_json(registry.version_dir(version) / "history.json"), out / "training_curves.png", f"{version} ({metadata['architecture']})")
    print(json.dumps({k: report[k] for k in ("version", "split", "artifact", "samples", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
