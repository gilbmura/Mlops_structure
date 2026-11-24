"""Single-image prediction page."""

from __future__ import annotations

import hashlib

import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.charts import probability_bars
from ui.presentation import activity, card, page_header
from ui.views import batch_predict
from utils.images import ImageDecodeError, decode_image


def render(client: ApiClient) -> None:
    page_header("Brain MRI tumour classifier", "01 · Predict", "Explore one MRI slice or a batch with our four-class image model.")
    st.info(
        "**Educational prototype — not a diagnostic service.** The model sorts a single MRI slice into one of four "
        "classes it was trained on (glioma, meningioma, pituitary, no tumour). It cannot recognise other conditions "
        "and cannot tell whether an image is really a brain MRI."
    )
    mode = st.radio("Prediction input", ["Single image", "ZIP batch"], horizontal=True)
    if mode == "ZIP batch":
        batch_predict.render(client)
        return
    upload = st.file_uploader("Upload one brain MRI slice (JPEG or PNG)", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.session_state.pop("single_prediction_result", None)
        card("Ready for an MRI slice", "Choose a JPEG or PNG above, preview the image, then select Classify image.", label="Getting started")
        return

    data = upload.getvalue()
    fingerprint = hashlib.sha256(data).hexdigest()
    cached = st.session_state.get("single_prediction_result")
    if cached and cached[0] != fingerprint:
        st.session_state.pop("single_prediction_result", None)
        cached = None
    if len(data) > 10 * 1024 * 1024:
        st.error("Each image must be 10 MiB or smaller.")
        return
    try:
        preview = decode_image(data)
    except ImageDecodeError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1, 1.3])
    left.image(preview, caption=upload.name, width="stretch")
    if right.button("Classify image", type="primary"):
        st.session_state.pop("single_prediction_result", None)
        try:
            with right, activity("Checking the image and running the model…", "Image processing complete"):
                result = client.predict(upload.name, data)
            cached = (fingerprint, result)
            st.session_state["single_prediction_result"] = cached
        except ApiError as exc:
            right.error(f"Prediction failed — {exc}")
            return
    if not cached:
        return
    result = cached[1]
    with right:
        if not result["accepted"]:
            _render_refusal(result)
            return
        card(result["message"], "This is a classification of the uploaded image, not a clinical diagnosis.",
             label=f"Model {result['model_version']} · prediction")
        st.metric("Model confidence (score for the predicted class)", f"{result['confidence']:.1%}")
        if result["low_confidence"]:
            st.warning("Low confidence: the model's scores are spread across classes, so treat this output as uncertain.")
        st.altair_chart(probability_bars(result["probabilities"]), width="stretch")
        st.caption(result["disclaimer"])
        st.caption(f"Model {result['model_version']}")


def _render_refusal(result: dict) -> None:
    """No class, score or chart: the image did not pass the input checks."""
    st.warning(f"### {result['message']}")
    st.markdown("\n".join(f"- {reason}" for reason in result["rejection_reasons"]))
    st.caption(
        "The model only knows four classes of greyscale brain MRI slices, so it is not asked to label images that fail "
        "these checks. The checks catch most unrelated images but are not perfect, and about 1 % of genuine MRI slices "
        "are refused by design."
    )
    st.caption(f"Model {result['model_version']}")
