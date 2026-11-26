"""Review becomes available only after the upload worker reports success."""

import pytest
from streamlit.testing.v1 import AppTest


def _app():
    import streamlit as st
    from ui.views.upload_progress import render_job
    render_job(st.session_state["job"])


@pytest.mark.parametrize("state", ["queued", "running", "failed", "succeeded"])
def test_upload_progress_and_review_button(state):
    app = AppTest.from_function(_app)
    app.session_state["job"] = {"id": "job", "status": state, "params": {"total_files": 100},
                                 "result": {"processed": 40, "total": 100, "counts": {"staged": 40}}}
    app.run(timeout=15)
    assert not app.exception
    assert app.button[0].disabled == (state != "succeeded")
    if state == "failed":
        assert app.error and not app.success
    if state == "succeeded":
        app.button[0].click().run()
        assert app.session_state["current_upload_batch"] == "job"
