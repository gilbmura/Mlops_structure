"""Background training from random weights, without replacing the served model."""

from config.settings import Settings
from src.pipeline import load_training_data, run_training
from src.registry import ModelRegistry


def run_initial_training(job_id: str, params: dict, settings: Settings, config: dict, registry: ModelRegistry) -> dict:
    data = load_training_data(settings, config)
    result = run_training(
        data, config, registry, architecture=params.get("architecture", "cnn"),
        smoke=bool(params.get("smoke", False)),
        extra_metadata={"job_id": job_id, "trigger": "manual initial training request"},
    )
    return {
        "candidate_version": result.version,
        "active_version": registry.active_version(),
        "initialization": "random_weights",
        "promoted": False,
        "smoke": bool(params.get("smoke", False)),
        "train_images": registry.metadata(result.version)["training"]["train_images"],
        "epochs_run": result.outcome.epochs_run,
        "training_seconds": result.outcome.duration_seconds,
        "validation": result.validation_metrics,
        "next_step": "Smoke candidates cannot be activated." if params.get("smoke") else
                     "Review validation results, then activate this candidate using Model activation & rollback.",
    }
