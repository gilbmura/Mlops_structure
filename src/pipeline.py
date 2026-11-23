"""Train -> evaluate on validation -> register. Shared by the CLI script and the retraining worker."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf

from config.settings import Settings
from src.dataset.manifests import Sample, load_arrays, read_split, samples_fingerprint
from src.dataset.splitting import TRAIN, VALIDATION
from src.evaluation import evaluate_model
from src.input_gate import GATE_FILE, calibrate
from src.labels import CLASS_NAMES
from src.preprocessing import PreprocessingSpec, to_model_input
from src.registry import ModelRegistry
from src.training import TrainingOutcome, subsample_per_class, train_model
from utils.files import read_json

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Standardized arrays plus the provenance recorded in model metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    dataset_info: dict


@dataclass
class PipelineResult:
    version: str
    validation_metrics: dict
    outcome: TrainingOutcome


def load_training_data(settings: Settings, config: dict, uploads: Sequence[Sample] = ()) -> TrainingData:
    """Prepared train/validation arrays, optionally extended with admitted uploads (training only)."""
    spec = PreprocessingSpec.from_config(config)
    train = read_split(settings.data_dir, TRAIN)
    validation = read_split(settings.data_dir, VALIDATION)
    x_train, y_train = load_arrays(train, spec, settings.cache_dir)
    x_val, y_val = load_arrays(validation, spec, settings.cache_dir)
    if uploads:
        x_extra, y_extra = load_arrays(uploads, spec, cache_dir=None)
        x_train, y_train = np.concatenate([x_train, x_extra]), np.concatenate([y_train, y_extra])

    base_version = read_json(settings.manifests_dir / "dataset_card.json")["dataset_version"]
    train_fingerprint = samples_fingerprint([*train, *uploads])
    return TrainingData(x_train, y_train, x_val, y_val, {
        "dataset_version": f"{base_version}+uploads-{train_fingerprint[:10]}" if uploads else base_version,
        "base_dataset_version": base_version,
        "train_fingerprint": train_fingerprint,
        "validation_fingerprint": samples_fingerprint(validation),
        "base_train_images": len(train),
        "uploaded_images": len(uploads),
        "upload_ids": [sample.sample_id for sample in uploads],
    })


def run_training(
    data: TrainingData,
    config: dict,
    registry: ModelRegistry,
    architecture: str | None = None,
    smoke: bool = False,
    extra_metadata: dict | None = None,
    initial_model: keras.Model | None = None,
) -> PipelineResult:
    """Train a model, score it on validation, and register a separate candidate."""
    seed = int(config["dataset"]["seed"])
    x_train, y_train, x_val, y_val = data.x_train, data.y_train, data.x_val, data.y_val
    if smoke:
        per_class = int(config["training"]["smoke"]["samples_per_class"])
        x_train, y_train = subsample_per_class(x_train, y_train, per_class, seed)
        x_val, y_val = subsample_per_class(x_val, y_val, per_class, seed)
        logger.info("Smoke run: %d train / %d validation images", len(y_train), len(y_val))

    outcome = train_model(x_train, y_train, x_val, y_val, config, architecture, smoke, initial_model)
    validation_metrics, _ = evaluate_model(outcome.model, data.x_val, data.y_val)
    logger.info("Validation macro-F1 %.4f, accuracy %.4f", validation_metrics["macro_f1"], validation_metrics["accuracy"])
    # Every model ships with its own input-gate calibration (features differ between models).
    gate = calibrate(outcome.model, x_train, y_train, data.x_val, float(config["input_gate"]["feature_distance_percentile"]))
    logger.info("Input gate threshold %.2f refuses %.1f%% of validation MRIs", gate.threshold, 100 * gate.summary["validation_refused_fraction"])

    metadata = {
        "architecture": architecture or config["training"]["architecture"],
        "class_names": list(CLASS_NAMES),
        "preprocessing": PreprocessingSpec.from_config(config).to_dict(),
        "parameters": int(outcome.model.count_params()),
        "framework": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "training": {
            "initialization": "saved_custom_weights" if initial_model is not None else "random_weights",
            "optimizer_state": "fresh",
            "smoke": smoke,
            "seed": seed,
            "epochs_run": outcome.epochs_run,
            "best_epoch": outcome.best_epoch,
            "duration_seconds": outcome.duration_seconds,
            "class_weights": outcome.class_weights,
            "train_images": int(len(y_train)),
            "validation_images": int(len(data.y_val)),
            "fit_validation_images": int(len(y_val)),
            "train_class_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_train.tolist()).items())},
            "hyperparameters": config["training"],
            "augmentation": config["augmentation"],
        },
        "dataset": data.dataset_info,
        "validation_metrics": validation_metrics,
        "input_gate": gate.summary,
        **(extra_metadata or {}),
    }
    version = registry.register(outcome.model, metadata, outcome.history, to_model_input(x_val[:16]), {GATE_FILE: gate.to_bytes()})
    return PipelineResult(version, validation_metrics, outcome)
