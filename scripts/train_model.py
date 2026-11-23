"""Train a model from scratch on the prepared splits and register it.

Usage:
    python scripts/train_model.py --smoke                 # fast end-to-end check (separate registry)
    python scripts/train_model.py --architecture baseline # simple baseline for comparison
    python scripts/train_model.py --activate              # final CNN; becomes the served model

Only train and validation data are used. Final-test evaluation is a separate,
deliberate step: scripts/evaluate_model.py --split test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings, load_pipeline_config  # noqa: E402
from src.pipeline import load_training_data, run_training  # noqa: E402
from src.registry import ModelRegistry  # noqa: E402
from utils.log import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=["cnn", "baseline"], default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny class-balanced subset and 2 epochs")
    parser.add_argument("--activate", action="store_true", help="serve this model after registration")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_pipeline_config()
    # Smoke runs never touch the real registry.
    registry = ModelRegistry(settings.cache_dir / "smoke_registry" if args.smoke else settings.models_dir)

    result = run_training(load_training_data(settings, config), config, registry, args.architecture, args.smoke)
    if args.activate:
        registry.activate(result.version, "initial training run via scripts/train_model.py")
    metrics = result.validation_metrics
    print(json.dumps({
        "version": result.version,
        "registry": str(registry.root),
        "activated": args.activate,
        "epochs_run": result.outcome.epochs_run,
        "best_epoch": result.outcome.best_epoch,
        "duration_seconds": result.outcome.duration_seconds,
        "validation": {k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
    }, indent=2))


if __name__ == "__main__":
    main()
