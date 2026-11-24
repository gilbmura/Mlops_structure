"""Dataset visualisations and model evaluation results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.charts import confusion_matrix, feature_ranges, split_counts, training_curves
from ui.presentation import page_header


def _metric_table(report: dict) -> pd.DataFrame:
    frame = pd.DataFrame(report["per_class"]).T
    frame["support"] = frame["support"].astype(int)
    return frame.round(3)


def _render_dataset(client: ApiClient) -> None:
    data = client.dataset_insights()
    card, counts = data["card"], data["card"]["counts"]
    st.subheader("Dataset and cleaning")
    st.caption(f"{card['source']} · dataset version `{card['dataset_version']}`")
    cols = st.columns(4)
    cols[0].metric("Classification files", counts["classification_files"])
    cols[1].metric("Exact duplicates removed", counts["rejected_by_reason"].get("exact_duplicate", 0))
    cols[2].metric("Unique images kept", counts["retained_unique_images"])
    cols[3].metric("Linked related-image groups", counts["linked_groups"])
    st.markdown(
        f"- {counts['exact_duplicate_groups']} exact-duplicate groups, **{counts['exact_groups_crossing_original_split']} crossed the "
        f"original train/test split**, so the supplied split was replaced by a group-aware split.\n"
        f"- Near-duplicate / related pairs: {counts['similar_pairs_by_kind']}. Linked images always share a split.\n"
        "- No patient identifiers exist, so patient-level separation cannot be guaranteed."
    )
    st.altair_chart(split_counts(card["splits"]), width="stretch")

    features = data["features"]
    if not features:
        st.info("Image-feature analysis not generated yet (run scripts/analyze_features.py).")
        return
    st.subheader("Image features by class (training split)")
    st.caption("Median (dot) and interquartile range (bar). Measured on standardized 128×128 images.")
    for name, description in features["features"].items():
        st.altair_chart(feature_ranges(features["summary"], name, description), width="stretch")


def _render_evaluation(client: ApiClient) -> None:
    report = client.evaluation()
    st.subheader(f"Evaluation — model {report['version']}")
    sets = [("Validation", report["validation"])] + ([("Final test", report["test"])] if report["test"] else [])
    for name, metrics in sets:
        st.markdown(f"**{name} split** ({metrics['samples']} images)")
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        cols[2].metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
        cols[3].metric("Tumour sensitivity", f"{metrics['tumor_detection']['sensitivity']:.3f}")
        cols[4].metric("Tumour specificity", f"{metrics['tumor_detection']['specificity']:.3f}")
        left, right = st.columns([1.2, 1])
        left.dataframe(_metric_table(metrics), width="stretch")
        right.altair_chart(confusion_matrix(metrics["confusion_matrix"], metrics["class_order"], f"{name} confusion matrix"))
    if not report["test"]:
        st.caption("Final-test results appear after running scripts/evaluate_model.py --split test.")

    st.markdown("**Training curves**")
    history = report["history"]
    cols = st.columns(3)
    for col, metric in zip(cols, ("loss", "accuracy", "macro_f1")):
        col.altair_chart(training_curves(history, metric), width="stretch")


def render(client: ApiClient) -> None:
    page_header("Data & evaluation", "02 · Understand", "Explore the dataset, image features and measured model performance.")
    try:
        _render_dataset(client)
    except ApiError as exc:
        st.error(f"Dataset information unavailable — {exc}")
    st.divider()
    try:
        _render_evaluation(client)
    except ApiError as exc:
        st.error(f"Evaluation results unavailable — {exc}")
