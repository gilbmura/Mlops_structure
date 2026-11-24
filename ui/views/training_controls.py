"""Initial training UI, separate from upload-triggered retraining."""

import streamlit as st

from ui.api_client import ApiClient, ApiError


def render(client: ApiClient) -> None:
    with st.expander("Train a new model from scratch"):
        st.caption("Uses random weights and the prepared training split, with validation for evaluation. "
                   "Uploads and test images are excluded. Your current model keeps serving. "
                   "A completed full-training candidate must be reviewed and activated manually below.")
        architecture = st.selectbox("Model architecture", ["cnn", "baseline"], key="initial-architecture")
        smoke = st.checkbox("Quick initial-training check (subset; cannot be activated)", key="initial-smoke")
        if st.button("Start training from scratch"):
            try:
                job = client.train(architecture, smoke)
                st.session_state["last_job"] = job["id"]
                st.success(f"Queued training job {job['id']}. Follow its status and logs below.")
            except ApiError as exc:
                st.error(f"Could not start training: {exc}")
