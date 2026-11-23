"""The fixed four-class label contract shared by data, model, API and UI.

The order of ``CLASS_NAMES`` defines the model's output indices. It is stored in
every model's metadata and checked on load, so it must never be reordered.
"""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "no_tumor")
NO_TUMOR = "no_tumor"
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

# Codes used in BRISC 2025 filenames, e.g. brisc2025_train_00001_gl_ax_t1.jpg.
# The bundled README documents "nt" for no-tumor, but every archived file uses "no";
# both are accepted so either convention is validated against the manifest label.
FILENAME_CODES: dict[str, str] = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "no_tumor",
    "nt": "no_tumor",
}


class UnknownLabelError(ValueError):
    """Raised for any label outside the four supported classes."""


def normalize_label(label: str) -> str:
    """Return the canonical class name, accepting case and space/hyphen variants.

    Unknown labels raise instead of silently creating a new class.
    """
    canonical = label.strip().lower().replace("-", "_").replace(" ", "_")
    if canonical not in CLASS_TO_INDEX:
        raise UnknownLabelError(f"Unknown label {label!r}; expected one of {', '.join(CLASS_NAMES)}.")
    return canonical


def tumor_status_message(predicted_class: str) -> str:
    """User-facing sentence derived from the four-class prediction."""
    if predicted_class == NO_TUMOR:
        return "No tumor detected in this image."
    return f"Tumor detected — predicted type: {predicted_class}."
