"""Initial-training UI forwards the user's architecture and mode without running training in the UI."""

from streamlit.testing.v1 import AppTest


def _app():
    import streamlit as st
    from ui.views.training_controls import render

    class Client:
        def train(self, architecture, smoke):
            st.session_state["request"] = (architecture, smoke)
            return {"id": "initial-job"}

    render(Client())


def test_initial_training_controls():
    app = AppTest.from_function(_app).run()
    app.selectbox[0].select("baseline")
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["request"] == ("baseline", True)
    assert app.session_state["last_job"] == "initial-job"
