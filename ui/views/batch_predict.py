"""ZIP prediction results and CSV export, separate from retraining uploads."""

import hashlib

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError
from utils.csv_export import csv_bytes


def result_rows(report: dict) -> list[dict]:
    rows = []
    for item in report["items"]:
        prediction = item["prediction"] or {}
        rows.append({
            "filename": item["filename"], "status": item["status"],
            "predicted_class": prediction.get("predicted_class", ""),
            "tumor_detected": prediction.get("tumor_detected"),
            "message": prediction.get("message", ""), "confidence": prediction.get("confidence"),
            "low_confidence": prediction.get("low_confidence"),
            "reason": item["reason"] or "", "model_version": report["model_version"],
            "disclaimer": report["disclaimer"],
        })
    return rows


def render(client: ApiClient) -> None:
    st.caption("Upload a ZIP of JPEG/PNG images, at its root or in folders. Class labels and an admin key are not required. "
               "These files are used only for prediction; they do not enter the retraining pool.")
    st.caption("Default limits: 200 files, 100 MiB ZIP, 300 MiB expanded, 10 MiB per image. Nested archives are rejected.")
    archive = st.file_uploader("ZIP for batch predictions", type=["zip"], key="prediction_zip")
    if archive is None:
        st.session_state.pop("zip_prediction_result", None)
        return
    data = archive.getvalue()
    fingerprint = hashlib.sha256(data).hexdigest()
    cached = st.session_state.get("zip_prediction_result")
    if cached and cached[0] != fingerprint:
        st.session_state.pop("zip_prediction_result", None)
        cached = None
    if st.button("Classify ZIP images", type="primary"):
        st.session_state.pop("zip_prediction_result", None)
        cached = None
        try:
            with st.spinner("Validating the archive and classifying images…"):
                report = client.predict_zip(archive.name, data)
            cached = (fingerprint, report)
            st.session_state["zip_prediction_result"] = cached
        except ApiError as exc:
            st.error(f"Batch prediction failed — {exc}")
    if not cached:
        return
    report = cached[1]
    st.write(f"**{report['predicted_files']} predicted · {report['rejected_files']} rejected** · Model {report['model_version']}")
    if not report["predicted_files"]:
        st.warning("No valid images were classified. Check the rejection reasons below.")
    st.caption(report["disclaimer"])
    st.caption("Confidence is a score from 0 to 1 (for example, 0.95 = 95%). Rejected files have no prediction.")
    rows = result_rows(report)
    st.dataframe(pd.DataFrame(rows).drop(columns=["disclaimer"]), width="stretch", hide_index=True)
    st.download_button("Download results as CSV", csv_bytes(rows, list(rows[0])),
                       file_name="brain_mri_batch_predictions.csv", mime="text/csv", on_click="ignore")
