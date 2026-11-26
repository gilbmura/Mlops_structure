"""Request and response models for the public API contract."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class PredictionResponse(BaseModel):
    accepted: bool = Field(description="False when the image failed the input checks; then no class or score is returned")
    message: str
    model_version: str
    predicted_class: str | None = Field(description="One of glioma, meningioma, pituitary, no_tumor; null when refused")
    tumor_detected: bool | None
    confidence: float | None = Field(description="Softmax score of the predicted class; a model score, not a clinical probability")
    low_confidence: bool | None
    probabilities: dict[str, float] | None
    rejection_reasons: list[str] = Field(description="Why the image was refused (empty when accepted)")
    feature_distance: float | None = Field(description="Distance from the training MRIs' features; refused above the model's calibrated threshold")
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    model_artifact: str | None = Field(default=None, description="Artifact loaded for prediction; null when no model is loaded")
    process_uptime_seconds: float
    process_started_at: str
    served_by: str
    detail: str | None = None


class BatchPredictionItem(BaseModel):
    filename: str
    status: Literal["predicted", "rejected"]
    prediction: PredictionResponse | None
    reason: str | None


class BatchPredictionResponse(BaseModel):
    total_files: int
    predicted_files: int
    rejected_files: int
    model_version: str
    items: list[BatchPredictionItem]
    disclaimer: str


class LabelUpdate(BaseModel):
    label: str


class AdmitRequest(BaseModel):
    upload_ids: list[str] = Field(min_length=1)
    confirm_labels_reviewed: bool = Field(description="Reviewer confirms each label was checked by a trustworthy person")


class RetrainRequest(BaseModel):
    smoke: bool = Field(default=False, description="Tiny 2-epoch run to exercise the pipeline quickly")
    batch_id: str | None = Field(default=None, min_length=1, description="Only admitted images from this batch; omit for the existing pool")


class TrainRequest(BaseModel):
    architecture: Literal["baseline", "cnn"] = "cnn"
    smoke: bool = False


class ActivationRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=300)
