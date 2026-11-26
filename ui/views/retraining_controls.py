"""Explicit batch and existing-pool retraining actions."""

import streamlit as st

from ui.api_client import ApiClient, ApiError


def clear_current_batch() -> None:
    """A newly selected file must not inherit the previous batch's eligibility."""
    st.session_state.pop("current_upload_batch", None)


def _queue(client: ApiClient, smoke: bool, batch_id: str | None = None) -> None:
    try:
        job = client.retrain(smoke, batch_id=batch_id)
        st.session_state["last_job"] = job["id"]
        scope = f"batch {batch_id}" if batch_id else "the existing pool"
        st.success(f"Queued job {job['id']} using {scope}. Prediction keeps working while it runs.")
    except ApiError as exc:
        st.error(f"Could not start retraining: {exc}")


def render_controls(client: ApiClient) -> None:
    st.subheader("3 · Retrain")
    st.caption("Each job continues our saved custom CNN on the original training split plus the selected admitted images. "
               "The candidate must pass the validation promotion checks before replacing the active model.")
    smoke = st.checkbox("Quick demo run (2 epochs on a small subset — never promoted)")
    batch_id = st.session_state.get("current_upload_batch")
    admitted = client.uploads("admitted", batch_id=batch_id, limit=1)["total"] if batch_id else 0
    if batch_id:
        st.write(f"Current batch: {batch_id} — {admitted:,} admitted image(s).")
    if not admitted:
        st.info("Retraining with this batch is disabled. Stage valid images, review their labels and admit them first. "
                "Rejected images cannot enable this action.")
    else:
        st.caption("Uses only admitted images from this batch as additional training data; rejected images are excluded.")
    if st.button("Retrain with this batch", type="primary", disabled=not admitted):
        _queue(client, smoke, batch_id)

    with st.expander("Retrain with the existing pool (separate action)"):
        count = client.uploads(limit=1)["counts"].get("admitted", 0)
        st.write(f"Existing pool: {count} admitted image(s), across all batches.")
        st.caption("This action is independent of the latest upload, including when that upload was rejected.")
        if st.button("Retrain with existing pool", disabled=count == 0):
            _queue(client, smoke)
