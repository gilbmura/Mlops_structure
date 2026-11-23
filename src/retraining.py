"""The retraining job executed by the background worker.

1. Collect every *admitted* upload (staged files a reviewer confirmed).
2. Continue the saved custom model on the original training split + those uploads.
3. Score the candidate *and* the currently active model on the same fixed
   validation split (the test split is never touched).
4. Promote the candidate only if it meets the promotion criteria; otherwise keep
   the active model. Either way the candidate stays registered for inspection.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

from config.settings import Settings
from src.dataset.manifests import Sample
from src.evaluation import evaluate_model
from src.labels import CLASS_NAMES
from src.pipeline import load_training_data, run_training
from src.preprocessing import PreprocessingSpec
from src.promotion import decide_promotion
from src.registry import ModelRegistry
from src.storage.uploads import UploadStore

logger = logging.getLogger(__name__)


class RetrainingError(RuntimeError):
    """Raised when retraining cannot start or produce a valid candidate."""


def _admitted_samples(uploads: UploadStore) -> list[Sample]:
    samples = []
    for row in uploads.admitted():
        path = Path(row["stored_path"])
        if not path.is_file():
            raise RetrainingError(f"Admitted upload {row['id']} is missing its file at {path}.")
        samples.append(Sample(f"upload-{row['id']}", row["label"], path))
    return samples


def run_retraining(job_id: str, params: dict, settings: Settings, config: dict, registry: ModelRegistry, uploads: UploadStore) -> dict:
    """Train, compare and (maybe) promote. Returns a JSON-serialisable job result."""
    samples = _admitted_samples(uploads)
    if not samples:
        raise RetrainingError("Admit at least one reviewed upload before retraining.")
    active = registry.active_version()
    if not active:
        raise RetrainingError("Train and activate an initial custom model before retraining.")
    parent = registry.metadata(active)
    if parent["preprocessing"] != PreprocessingSpec.from_config(config).to_dict():
        raise RetrainingError("Active model uses different preprocessing than the current config; compare manually.")
    if parent.get("class_names") != list(CLASS_NAMES):
        raise RetrainingError("Active model class order does not match the four-class contract.")
    if parent.get("training", {}).get("smoke"):
        raise RetrainingError("Activate a fully trained model before retraining; the active model is a smoke artifact.")
    config = deepcopy(config)
    config["training"]["learning_rate"] = config["retraining"]["learning_rate"]
    config["training"]["architecture"] = parent["architecture"]
    logger.info("Job %s: continuing custom model %s with %d admitted uploads", job_id, active, len(samples))
    data = load_training_data(settings, config, samples)
    initial_model = registry.load_model(active)
    active_metrics = evaluate_model(initial_model, data.x_val, data.y_val)[0]

    result = run_training(
        data, config, registry,
        smoke=bool(params.get("smoke", False)),
        extra_metadata={"job_id": job_id, "parent_version": active, "trigger": "manual retrain request",
                        "parent_artifact": parent["artifacts"]["model.keras"]},
        initial_model=initial_model,
    )
    decision = decide_promotion(result.validation_metrics, active_metrics, config)
    for reason in decision.reasons:
        logger.info("Promotion check: %s", reason)

    # Smoke runs exercise the pipeline but must never change the serving model.
    promote = decision.promote and not params.get("smoke", False) and registry.active_version() == active
    if promote:
        registry.activate(result.version, f"promoted by job {job_id}")
    else:
        registry.mark(result.version, "rejected")
        logger.info("Candidate %s not promoted; %s remains active", result.version, active)

    return {
        "candidate_version": result.version,
        "previous_active_version": active,
        "promoted": promote,
        "initialization": "saved_custom_weights",
        "parent_version": active,
        "promotion_blocked": "smoke run or active model changed" if decision.promote and not promote else None,
        "active_version": registry.active_version(),
        "decision": decision.as_dict(),
        "uploads_included": len(samples),
        "train_images": int(len(data.y_train)),
        "dataset_version": data.dataset_info["dataset_version"],
        "candidate_validation": {k: result.validation_metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")},
        "active_validation": {k: active_metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")} if active_metrics else None,
        "epochs_run": result.outcome.epochs_run,
        "training_seconds": result.outcome.duration_seconds,
    }
