"""Single-image prediction page."""

from __future__ import annotations

import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.charts import probability_bars
from ui.views import batch_predict


def render(client: ApiClient) -> None:
    st.title("Brain MRI tumour classifier")
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
        return

    left, right = st.columns([1, 1.3])
    left.image(upload.getvalue(), caption=upload.name, width="stretch")
    if not right.button("Classify image", type="primary"):
        return
    try:
        with st.spinner("Running the model…"):
            result = client.predict(upload.name, upload.getvalue())
    except ApiError as exc:
        right.error(f"Prediction failed — {exc}")
        return

    with right:
        if not result["accepted"]:
            _render_refusal(result)
            return
        if result["tumor_detected"]:
            st.error(f"### {result['message']}")
        else:
            st.success(f"### {result['message']}")
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
