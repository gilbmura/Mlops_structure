"""Classify bounded ZIP entries with one model version and no dataset writes."""

from src.ingestion.archive import ArchiveEntry, SkippedEntry
from src.prediction import DISCLAIMER, REFUSED_MESSAGE, Predictor
from utils.images import ImageDecodeError


def classify_entries(entries: list[ArchiveEntry], skipped: list[SkippedEntry], predictor: Predictor) -> dict:
    """Reuse single-image preprocessing/inference; invalid or refused images get no class or score."""
    items = [
        {"filename": item.path, "status": "rejected", "prediction": None, "reason": item.reason}
        for item in skipped
    ]
    for entry in entries:
        try:
            prediction = predictor.predict_bytes(entry.data)
        except ImageDecodeError as exc:
            items.append({"filename": entry.path, "status": "rejected", "prediction": None, "reason": str(exc)})
            continue
        if not prediction.accepted:
            reason = f"{REFUSED_MESSAGE} {' '.join(prediction.rejection_reasons)}"
            items.append({"filename": entry.path, "status": "rejected", "prediction": None, "reason": reason})
        else:
            items.append({"filename": entry.path, "status": "predicted",
                          "prediction": {**prediction.as_dict(), "disclaimer": DISCLAIMER}, "reason": None})
    predicted = sum(item["status"] == "predicted" for item in items)
    return {"total_files": len(items), "predicted_files": predicted, "rejected_files": len(items) - predicted,
            "model_version": predictor.version, "items": items, "disclaimer": DISCLAIMER}
