"""Altair chart builders used by the Streamlit pages."""

from __future__ import annotations

import altair as alt
import pandas as pd

# Fixed class colours keep every chart consistent (colour-blind-safe Okabe-Ito hues).
CLASS_COLOURS = {"glioma": "#0072B2", "meningioma": "#E69F00", "pituitary": "#009E73", "no_tumor": "#CC79A7"}
_COLOUR_SCALE = alt.Scale(domain=list(CLASS_COLOURS), range=list(CLASS_COLOURS.values()))


def probability_bars(probabilities: dict[str, float]) -> alt.Chart:
    frame = pd.DataFrame({"class": list(probabilities), "probability": list(probabilities.values())})
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Model score (softmax)"),
            y=alt.Y("class:N", sort=list(CLASS_COLOURS), title=None),
            color=alt.Color("class:N", scale=_COLOUR_SCALE, legend=None),
            tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
        )
        .properties(height=160)
    )


def confusion_matrix(matrix: list[list[int]], classes: list[str], title: str) -> alt.Chart:
    rows = [{"actual": a, "predicted": p, "count": matrix[i][j]} for i, a in enumerate(classes) for j, p in enumerate(classes)]
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        x=alt.X("predicted:N", sort=classes, title="Predicted"), y=alt.Y("actual:N", sort=classes, title="Actual")
    )
    heat = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["actual", "predicted", "count"])
    text = base.mark_text(fontSize=13).encode(
        text="count:Q", color=alt.condition(alt.datum.count > frame["count"].max() / 2, alt.value("white"), alt.value("black"))
    )
    return (heat + text).properties(title=title, width=320, height=320)


def training_curves(history: dict[str, list[float]], metric: str) -> alt.Chart:
    frame = pd.DataFrame({"epoch": range(1, len(history[metric]) + 1), "training": history[metric], "validation": history[f"val_{metric}"]})
    long = frame.melt("epoch", var_name="split", value_name=metric)
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(x="epoch:Q", y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ")), color=alt.Color("split:N", title=None), tooltip=["epoch", "split", alt.Tooltip(f"{metric}:Q", format=".4f")])
        .properties(height=260)
    )


def split_counts(splits: dict[str, dict[str, int]]) -> alt.Chart:
    rows = [{"split": s, "class": c, "images": n} for s, counts in splits.items() for c, n in counts.items()]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("split:N", sort=["train", "validation", "test", "retrain_demo"], title=None),
            y=alt.Y("images:Q", title="Images"),
            color=alt.Color("class:N", scale=_COLOUR_SCALE, title="Class"),
            tooltip=["split", "class", "images"],
        )
        .properties(height=280)
    )


def feature_ranges(rows: list[dict], feature: str, title: str) -> alt.Chart:
    """Median with interquartile range per class for one image feature."""
    frame = pd.DataFrame([r for r in rows if r["feature"] == feature])
    base = alt.Chart(frame).encode(y=alt.Y("class:N", sort=list(CLASS_COLOURS), title=None, axis=alt.Axis(labelOverlap=False)))
    rule = base.mark_rule(strokeWidth=4).encode(
        x=alt.X("q1:Q", title=title, scale=alt.Scale(zero=False)), x2="q3:Q", color=alt.Color("class:N", scale=_COLOUR_SCALE, legend=None)
    )
    point = base.mark_point(filled=True, size=90, color="black").encode(x="median:Q", tooltip=["class", "median", "q1", "q3"])
    return (rule + point).properties(height=alt.Step(28), title=title)
