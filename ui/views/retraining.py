"""Upload review and retraining controls (requires the admin key in the sidebar)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.presentation import activity, page_header
from ui.views.retraining_controls import clear_current_batch, render_controls
from ui.views import job_progress, training_controls, upload_progress

_NO_LABEL = "— no label (needs review) —"


def _show_batch(report: dict) -> None:
    st.session_state["current_upload_batch"] = report["batch_id"]
    st.info(f"Batch {report['batch_id']}: {report['counts']}")
    rows = [
        {"file": i["original_name"], "status": i["status"], "label": i.get("label"), "reason": i.get("reason"), "notes": " | ".join(i.get("notes") or [])}
        for i in report["items"]
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _upload_section(client: ApiClient, classes: list[str]) -> None:
    st.subheader("1 · Upload labelled images")
    st.caption("Content screening runs before staging and admission. Rejected images cannot be relabelled into training. "
               "Trusted review of MRI content and class labels is still required. "
               "Limits: 320 MiB per batch, 640 MiB expanded ZIP, 16,000 files per batch, 10 MiB per image. "
               "ZIPs also have a 16,000-entry limit including folders.")
    mode = st.radio("Upload type", ["Single image", "Multiple images", "ZIP of class folders"], horizontal=True,
                    on_change=clear_current_batch)
    try:
        if mode == "Single image":
            file = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="single", on_change=clear_current_batch)
            label = st.selectbox("Label", classes)
            if file and st.button("Stage image", type="primary"):
                with activity("Uploading and screening the image…", "Screening complete — review the outcome below"):
                    report = client.upload_images([(file.name, file.getvalue())], [label])
                _show_batch(report)
        elif mode == "Multiple images":
            files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi",
                                     on_change=clear_current_batch)
            shared = st.toggle("Use one label for all images", value=True)
            if shared:
                label = st.selectbox("Shared label", classes)
                labels = [label] * len(files or [])
            else:
                labels = [st.selectbox(f"Label for {f.name}", [_NO_LABEL, *classes], key=f"label-{i}-{f.name}") for i, f in enumerate(files or [])]
                labels = [None if value == _NO_LABEL else value for value in labels]
            if files and st.button("Stage images", type="primary"):
                with activity("Uploading and screening the images…", "Screening complete — review the outcome below"):
                    report = client.upload_images([(f.name, f.getvalue()) for f in files], labels)
                _show_batch(report)
        else:
            st.markdown(
                "Folders must be named after a class: `glioma/`, `meningioma/`, `pituitary/`, `no_tumor/` — at the top of the "
                "ZIP or inside **one** wrapper folder (e.g. `new_data/glioma/…`). One class folder is enough. Files elsewhere "
                "are flagged *needs label*; nested archives are rejected."
            )
            archive = st.file_uploader("ZIP archive", type=["zip"], key="zip", on_change=clear_current_batch)
            st.caption("The ZIP is saved first, then screened in the background. Follow its progress below.")
            if archive and st.button("Stage ZIP", type="primary"):
                clear_current_batch()
                with activity("Receiving the ZIP…", "Archive received — screening queued"):
                    job = client.upload_zip(archive.name, archive.getvalue())
                st.session_state["last_upload_job"] = job["id"]
                st.success(f"Upload job {job['id']} is {job['status']}. Follow ZIP processing status below.")
    except ApiError as exc:
        clear_current_batch()
        st.error(f"Upload failed — {exc}")
    upload_progress.render(client)


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
    batch_id = st.session_state.get("current_upload_batch")
    if batch_id:
        st.caption(f"Reviewing batch {batch_id}.")
    filters = {"Images awaiting review": "review", "Rejected files": "rejected", "Duplicate files": "duplicate"}
    selected_filter = st.selectbox("Results to inspect", list(filters))
    status = filters[selected_filter]
    page = int(st.number_input("Results page", min_value=1, step=1, key=f"review-page-{batch_id}-{status}"))
    try:
        report = client.uploads(status, batch_id=batch_id, limit=25, offset=(page - 1) * 25)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.caption(f"{report['total']:,} matching files · 25 per page. Incomplete ZIP batches cannot be reviewed or admitted.")
    if not report["items"]:
        st.info("No files on this page.")
        return
    if status != "review":
        st.dataframe(pd.DataFrame([{"file": item["original_name"], "reason": item["reason"]}
                                   for item in report["items"]]), hide_index=True, width="stretch")
        return
    selected = [item["id"] for item in report["items"] if _review_item(client, item, classes)]
    confirmed = st.checkbox("I confirm the selected labels were reviewed by a trustworthy person.")
    if st.button(f"Admit {len(selected)} selected upload(s) to the training pool", disabled=not (selected and confirmed)):
        try:
            result = client.admit(selected)
            st.success(f"Admitted {len(result['admitted'])} upload(s).")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))


def _retrain_section(client: ApiClient) -> None:
    render_controls(client)
    job_progress.render(client)


def _model_admin_section(client: ApiClient) -> None:
    st.subheader("4 · Model activation & rollback")
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
    if cols[1].button("Activate version", disabled=not version or len(reason) < 3):
        try:
            st.success(f"Now serving {client.activate(version, reason)['active_version']}")
        except ApiError as exc:
            st.error(str(exc))


def render(client: ApiClient) -> None:
    page_header("Training, uploads & retraining", "04 · Improve", "Train a candidate, review new images and follow each job as it runs.")
    if not client.admin_key:
        st.warning("Enter the admin key in the sidebar to upload data or start retraining.")
        return
    try:
        classes = client.labels()
    except ApiError as exc:
        st.error(str(exc))
        return
    training_controls.render(client)
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
