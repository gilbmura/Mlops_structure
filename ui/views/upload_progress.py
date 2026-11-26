"""Read-only polling of durable ZIP jobs and explicit selection of completed batches."""

import streamlit as st

from ui.api_client import ApiClient, ApiError


def render_job(job: dict) -> None:
    state, result = job["status"], job.get("result") or {}
    processed = result.get("processed", 0)
    total = result.get("total", job.get("params", {}).get("total_files", 0))
    st.write(f"**{state.capitalize()}** · {processed:,} / {total:,} files screened")
    st.progress(min(processed / total, 1.0) if total else 0.0)
    if result.get("counts"):
        st.write(" · ".join(f"{name}: {count:,}" for name, count in result["counts"].items()))
    if state == "queued":
        st.info("The archive is saved. Waiting for the worker; an earlier upload or training job may be running.")
    elif state == "running":
        st.info(f"{result.get('phase', 'Processing').capitalize()}… You can leave this page and return later.")
    elif state == "failed":
        st.error(job.get("error") or "Upload processing failed.")
        st.caption("Partial results cannot be admitted. Resolve the error and submit the ZIP again.")
    elif state == "succeeded":
        st.success("Screening finished. Review the results and confirm valid labels before admission.")
    if st.button("Review this batch", key=f"review-upload-{job['id']}", disabled=state != "succeeded"):
        st.session_state["current_upload_batch"] = job["id"]
        st.rerun()


@st.fragment(run_every="3s")
def render(client: ApiClient) -> None:
    try:
        jobs = client.upload_jobs()
        if not jobs:
            return
        st.subheader("ZIP processing status")
        by_id = {job["id"]: job for job in jobs}
        latest = st.session_state.get("last_upload_job")
        if latest in by_id and latest != st.session_state.get("seen_upload_job"):
            st.session_state["inspect_upload_job"] = latest
            st.session_state["seen_upload_job"] = latest
        if st.session_state.get("inspect_upload_job") not in by_id:
            st.session_state["inspect_upload_job"] = jobs[0]["id"]
        selected = st.selectbox("Upload job", list(by_id), key="inspect_upload_job",
                                format_func=lambda key: f"{by_id[key]['params'].get('filename', 'ZIP')} · {key}")
        st.caption("Updates every 3 seconds. Processing continues if you close the browser.")
        render_job(client.upload_job(selected))
    except ApiError as exc:
        st.warning(f"Upload status unavailable — {exc}")
