"""Upload review and retraining controls (requires the admin key in the sidebar)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError

_NO_LABEL = "— no label (needs review) —"


def _show_batch(report: dict) -> None:
    st.success(f"Batch {report['batch_id']}: {report['counts']}")
    rows = [
        {"file": i["original_name"], "status": i["status"], "label": i.get("label"), "reason": i.get("reason"), "notes": " | ".join(i.get("notes") or [])}
        for i in report["items"]
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _upload_section(client: ApiClient, classes: list[str]) -> None:
    st.subheader("1 · Upload labelled images")
    mode = st.radio("Upload type", ["Single image", "Multiple images", "ZIP of class folders"], horizontal=True)
    try:
        if mode == "Single image":
            file = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="single")
            label = st.selectbox("Label", classes)
            if file and st.button("Stage image", type="primary"):
                _show_batch(client.upload_images([(file.name, file.getvalue())], [label]))
        elif mode == "Multiple images":
            files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi")
            shared = st.toggle("Use one label for all images", value=True)
            if shared:
                label = st.selectbox("Shared label", classes)
                labels = [label] * len(files or [])
            else:
                labels = [st.selectbox(f"Label for {f.name}", [_NO_LABEL, *classes], key=f"label-{i}-{f.name}") for i, f in enumerate(files or [])]
                labels = [None if value == _NO_LABEL else value for value in labels]
            if files and st.button("Stage images", type="primary"):
                _show_batch(client.upload_images([(f.name, f.getvalue()) for f in files], labels))
        else:
            st.markdown(
                "Folders must be named after a class: `glioma/`, `meningioma/`, `pituitary/`, `no_tumor/` — at the top of the "
                "ZIP or inside **one** wrapper folder (e.g. `new_data/glioma/…`). One class folder is enough. Files elsewhere "
                "are flagged *needs label*; nested archives are rejected."
            )
            archive = st.file_uploader("ZIP archive", type=["zip"], key="zip")
            if archive and st.button("Stage ZIP", type="primary"):
                _show_batch(client.upload_zip(archive.name, archive.getvalue()))
    except ApiError as exc:
        st.error(f"Upload failed — {exc}")


def _review_item(client: ApiClient, item: dict, classes: list[str]) -> bool:
    """Render one upload awaiting review; returns True if it is selected for admission."""
    preview, details = st.columns([1, 3])
    try:
        preview.image(client.upload_image(item["id"]), width=120)
    except ApiError:
        preview.caption("preview unavailable")
    details.markdown(f"**{item['original_name']}** · `{item['status']}` · {item['width']}×{item['height']}")
    for note in [item.get("reason"), *(item.get("notes") or [])]:
        if note:
            details.caption(note)
    current = item.get("label")
    options = [_NO_LABEL, *classes]
    choice = details.selectbox("Label", options, index=options.index(current) if current in classes else 0, key=f"rev-{item['id']}")
    action_cols = details.columns(3)
    if choice != _NO_LABEL and choice != current and action_cols[0].button("Save label", key=f"save-{item['id']}"):
        client.set_label(item["id"], choice)
        st.rerun()
    if action_cols[1].button("Discard", key=f"discard-{item['id']}"):
        client.discard(item["id"])
        st.rerun()
    return item["status"] == "staged" and action_cols[2].checkbox("Admit", key=f"admit-{item['id']}")


def _review_section(client: ApiClient, classes: list[str]) -> None:
    st.subheader("2 · Review staged uploads")
    st.caption(
        "Labels typed by uploaders are not verified ground truth. Only admit images whose labels a trustworthy, qualified "
        "person has checked. Model predictions are never used as labels."
    )
    try:
        pending = client.uploads("needs_label")["items"] + client.uploads("staged")["items"]
    except ApiError as exc:
        st.error(str(exc))
        return
    if not pending:
        st.info("Nothing awaiting review.")
        return
    selected = [item["id"] for item in pending if _review_item(client, item, classes)]
    confirmed = st.checkbox("I confirm the selected labels were reviewed by a trustworthy person.")
    if st.button(f"Admit {len(selected)} selected upload(s) to the training pool", disabled=not (selected and confirmed)):
        try:
            result = client.admit(selected)
            st.success(f"Admitted {len(result['admitted'])} upload(s).")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))


def _retrain_section(client: ApiClient) -> None:
    st.subheader("3 · Retrain")
    counts = client.uploads()["counts"]
    st.markdown(
        f"Training pool: **{counts.get('admitted', 0)} admitted upload(s)**. A retraining job continues training *our saved custom CNN* "
        "on the original training split plus all admitted uploads, then compares it with the active model on the fixed "
        "validation split. It is promoted only if macro-F1 is within 0.01 of the active model and no class recall drops by "
        "more than 0.05; otherwise the active model keeps serving."
    )
    smoke = st.checkbox("Quick demo run (2 epochs on a small subset — never promoted)")
    if st.button("Start retraining job", type="primary"):
        try:
            job = client.retrain(smoke)
            st.session_state["last_job"] = job["id"]
            st.success(f"Queued job {job['id']}. Prediction keeps working while it runs.")
        except ApiError as exc:
            st.error(f"Could not start retraining — {exc}")

    jobs = client.jobs(5)
    if jobs:
        job_ids = [j["id"] for j in jobs]
        default = job_ids.index(st.session_state["last_job"]) if st.session_state.get("last_job") in job_ids else 0
        job_id = st.selectbox("Inspect job", job_ids, index=default, format_func=lambda i: f"{i} ({next(j['status'] for j in jobs if j['id'] == i)})")
        job = client.job(job_id)
        st.json({k: job[k] for k in ("status", "created_at", "started_at", "finished_at", "error", "result")}, expanded=False)
        if st.button("Refresh job status"):
            st.rerun()
        with st.expander("Job log"):
            st.code(client.job_log(job_id) or "(empty)", language="text")


def _model_admin_section(client: ApiClient) -> None:
    st.subheader("4 · Model rollback")
    registry = client.models()
    st.caption(f"Active: {registry['active']}")
    reason = st.text_input("Reason (recorded in the activation history)", key="rollback-reason")
    cols = st.columns(2)
    if cols[0].button("Roll back to previous model", disabled=len(reason) < 3):
        try:
            st.success(f"Now serving {client.rollback(reason)['active_version']}")
        except ApiError as exc:
            st.error(str(exc))
    version = cols[1].selectbox("Or activate a specific version", sorted(registry["versions"]), key="activate-version")
    if cols[1].button("Activate version", disabled=len(reason) < 3):
        try:
            st.success(f"Now serving {client.activate(version, reason)['active_version']}")
        except ApiError as exc:
            st.error(str(exc))


def render(client: ApiClient) -> None:
    st.title("Uploads & retraining")
    if not client.admin_key:
        st.warning("Enter the admin key in the sidebar to upload data or start retraining.")
        return
    try:
        classes = client.labels()
    except ApiError as exc:
        st.error(str(exc))
        return
    _upload_section(client, classes)
    st.divider()
    _review_section(client, classes)
    st.divider()
    try:
        _retrain_section(client)
        st.divider()
        _model_admin_section(client)
    except ApiError as exc:
        st.error(str(exc))
