"""Service health, measured availability, active model and job status."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.presentation import page_header


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _render_health(client: ApiClient) -> None:
    st.subheader("API health")
    try:
        health = client.health()
    except Exception as exc:  # network failure is itself a status to show
        st.error(f"API unreachable: {exc}")
        return
    cols = st.columns(4)
    cols[0].metric("Status", health["status"].upper())
    cols[1].metric("Active model on replica", health["model_version"] or "none")
    cols[2].metric("Replica process uptime", _duration(health["process_uptime_seconds"]))
    cols[3].metric("Answered by replica", health["served_by"])
    if health.get("detail"):
        st.warning(health["detail"])

    st.subheader("Measured availability")
    availability = client.availability(24)["measured"]
    if not availability["probes"]:
        st.info("No probe results yet. Availability is measured by the probe service (python -m worker.probe).")
    else:
        cols = st.columns(3)
        cols[0].metric("Successful probes (24 h)", f"{availability['availability']:.2%}")
        cols[1].metric("Probes recorded", availability["probes"])
        cols[2].metric("Mean probe latency", f"{availability['mean_latency_ms']} ms" if availability["mean_latency_ms"] else "—")
        st.caption(
            f"Window {availability['first_probe']} → {availability['last_probe']} (UTC). Availability = share of periodic "
            "/health checks through the load balancer that succeeded; uptime above only says how long one process has run."
        )
        st.dataframe(pd.DataFrame(availability["recent"]), width="stretch", hide_index=True)


def _render_models(client: ApiClient) -> None:
    st.subheader("Models")
    try:
        active = client.model()
    except ApiError as exc:
        st.warning(f"No active model — {exc}")
        return
    meta = active["metadata"]
    cols = st.columns(4)
    cols[0].metric("Active version", active["active_version"])
    cols[1].metric("Architecture", meta["architecture"])
    cols[2].metric("Validation macro F1", f"{meta['validation_metrics']['macro_f1']:.3f}")
    cols[3].metric("Training images", meta["training"]["train_images"])
    st.caption(f"Dataset `{meta['dataset']['dataset_version']}` · registered {meta['registered_at']} · {meta['parameters']:,} parameters")

    registry = client.models()
    versions = pd.DataFrame.from_dict(registry["versions"], orient="index").rename_axis("version").reset_index()
    st.dataframe(versions, width="stretch", hide_index=True)
    if registry["activations"]:
        st.caption("Activation history")
        st.dataframe(pd.DataFrame(registry["activations"]).iloc[::-1], width="stretch", hide_index=True)


def _render_jobs(client: ApiClient) -> None:
    st.subheader("Training jobs")
    jobs = client.jobs(10)
    if not jobs:
        st.info("No jobs yet.")
        return
    frame = pd.DataFrame(jobs)[["id", "kind", "status", "created_at", "started_at", "finished_at", "error"]]
    st.dataframe(frame, width="stretch", hide_index=True)


def render(client: ApiClient) -> None:
    page_header("System status", "03 · Monitor", "Inspect service availability, model versions and training history.")
    if st.button("Refresh"):
        st.rerun()
    _render_health(client)
    st.divider()
    try:
        _render_models(client)
        st.divider()
        _render_jobs(client)
    except ApiError as exc:
        st.error(str(exc))
