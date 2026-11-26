"""Exercise the Streamlit buttons without starting real training jobs."""

from streamlit.testing.v1 import AppTest


def _app():
    import streamlit as st
    from ui.views.retraining_controls import clear_current_batch, render_controls

    class Client:
        def uploads(self, status=None, batch_id=None, limit=500):
            return {"counts": {"admitted": 100}, "total": 1200 if batch_id == "accepted" else 0,
                    "items": [{"id": "good"}] if batch_id == "accepted" else []}

        def retrain(self, smoke, batch_id=None):
            st.session_state["requested_batch"] = batch_id
            return {"id": "test-job"}

    st.button("Select different upload", on_click=clear_current_batch)
    render_controls(Client())


def test_rejected_batch_disabled_despite_existing_pool():
    app = AppTest.from_function(_app)
    app.session_state["current_upload_batch"] = "rejected"
    app.run()
    assert not app.exception
    assert app.button[1].label == "Retrain with this batch" and app.button[1].disabled
    assert app.button[2].label == "Retrain with existing pool" and not app.button[2].disabled
    app.button[2].click().run()
    assert app.session_state["requested_batch"] is None


def test_admitted_batch_enabled_and_selection_change_clears_it():
    app = AppTest.from_function(_app)
    app.session_state["current_upload_batch"] = "accepted"
    app.run()
    assert not app.button[1].disabled
    assert any("1,200 admitted" in item.value for item in app.markdown)
    app.button[1].click().run()
    assert app.session_state["requested_batch"] == "accepted"
    app.button[0].click().run()
    assert app.button[1].disabled
