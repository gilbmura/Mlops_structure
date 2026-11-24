"""Live status reflects worker results and never turns failed/quick runs into promotions."""

from streamlit.testing.v1 import AppTest


def _app():
    import streamlit as st
    from ui.views.job_progress import render_job

    render_job(st.session_state["job"], "INFO epoch 2: loss=0.5")


def test_retraining_quick_result_displays_real_metrics_without_promotion():
    app = AppTest.from_function(_app)
    app.session_state["job"] = {
        "kind": "retrain", "status": "succeeded", "params": {"smoke": True},
        "result": {"candidate_version": "v005", "epochs_run": 2, "promoted": False,
                   "candidate_validation": {"macro_f1": .875}},
    }
    app.run()
    assert not app.exception
    assert "cannot be promoted" in app.info[0].value
    assert not app.success
    assert [metric.value for metric in app.metric] == ["v005", "2", "0.875"]


def test_failed_job_does_not_display_success_or_candidate_metrics():
    app = AppTest.from_function(_app)
    app.session_state["job"] = {"kind": "train", "status": "failed", "error": "Training data unavailable."}
    app.run()
    assert not app.exception
    assert app.error[0].value == "Training data unavailable."
    assert not app.success and not app.metric


def test_initial_training_candidate_stays_distinct_from_a_promoted_model():
    app = AppTest.from_function(_app)
    app.session_state["job"] = {
        "kind": "train", "status": "succeeded", "params": {"smoke": False},
        "result": {"candidate_version": "v005", "epochs_run": 10, "promoted": False,
                   "validation": {"macro_f1": .9}},
    }
    app.run()
    assert not app.exception
    assert "No model was activated" in app.info[0].value
    assert not app.success
    assert app.metric[2].value == "0.900"
