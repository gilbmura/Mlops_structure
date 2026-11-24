"""Read-only live job panel. Epoch feedback comes from the worker's recorded log."""

import re

import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.presentation import card

_STATES = {
    "queued": ("Waiting for the worker", "Your request is queued. Training has not started yet."),
    "running": ("Training in progress", "The worker is preparing data, training and evaluating the candidate."),
    "succeeded": ("Training completed", "Review the candidate and its validation results below."),
    "failed": ("Training stopped", "The job did not complete. Review the error and log below."),
}


def render_job(job: dict, log: str) -> None:
    state = job["status"]
    title, detail = _STATES.get(state, ("Status unavailable", "Refresh to check the job again."))
    card(title, detail, label=f"{job['kind']} · {state}", active=state in ("queued", "running"), animate=False)
    epochs = re.findall(r"\bepoch (\d+):", log)
    if epochs:
        st.caption(f"Last completed epoch: {epochs[-1]}. Evaluation and model checks may continue after training.")
    result = job.get("result") or {}
    if state == "succeeded":
        if job.get("params", {}).get("smoke") or result.get("smoke"):
            st.info("Quick check completed. This candidate cannot be promoted or activated.")
        elif result.get("promoted"):
            st.success(f"Candidate {result['candidate_version']} passed the checks and was promoted.")
        else:
            st.info(f"Candidate {result.get('candidate_version', '—')} is registered. No model was activated by this job.")
        cols = st.columns(3)
        cols[0].metric("Candidate", result.get("candidate_version", "—"))
        cols[1].metric("Epochs completed", result.get("epochs_run", "—"))
        score = (result.get("validation") or result.get("candidate_validation") or {}).get("macro_f1")
        cols[2].metric("Validation macro F1", f"{score:.3f}" if score is not None else "—")
    if job.get("error"):
        st.error(job["error"])
    with st.expander("Job details"):
        st.json({key: job.get(key) for key in ("kind", "status", "params", "created_at", "started_at", "finished_at", "error", "result")})
    with st.expander("Job log"):
        st.code(log or "The worker has not written a log yet.", language="text")


@st.fragment(run_every="3s")
def render(client: ApiClient) -> None:
    """Refresh only this panel; polling cannot upload data or queue training."""
    try:
        jobs = client.jobs(5)
        if not jobs:
            return
        ids = [job["id"] for job in jobs]
        # Keep inspection stable as status changes; switch once when this user queues a new job.
        latest = st.session_state.get("last_job")
        if latest in ids and latest != st.session_state.get("progress_last_job"):
            st.session_state["progress_job"] = latest
            st.session_state["progress_last_job"] = latest
        if st.session_state.get("progress_job") not in ids:
            st.session_state["progress_job"] = ids[0]
        job_id = st.selectbox("Inspect job", ids, key="progress_job")
        st.caption("Live status · updates every 3 seconds while this page is open.")
        job = client.job(job_id)
        render_job(job, client.job_log(job_id))
    except ApiError as exc:
        st.warning(f"Live job status unavailable — {exc}")
