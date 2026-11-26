"""Streamlit entry point.

Run locally:
    streamlit run ui/app.py
The UI only talks to the API (API_URL); it never loads the model itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `streamlit run` only adds ui/ to the path

from ui.api_client import ApiClient  # noqa: E402
from ui.presentation import apply_styles  # noqa: E402
from ui.views import insights, predict, retraining, status  # noqa: E402

st.set_page_config(page_title="Brain MRI classifier (educational)", page_icon="🧠", layout="wide")

with st.sidebar:
    st.caption("Educational prototype — not for diagnosis.")
    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
    admin_key = st.text_input("Admin key (uploads & retraining)", type="password")
    st.divider()
    reduce_motion = st.toggle("Reduce motion", key="reduce_motion", help="Disable UI animations. Your device's reduced-motion setting is also respected.")

apply_styles(reduce_motion)

client = ApiClient(api_url, admin_key)
pages = [
    st.Page(lambda: predict.render(client), title="Predict", url_path="predict", default=True),
    st.Page(lambda: insights.render(client), title="Data & evaluation", url_path="insights"),
    st.Page(lambda: status.render(client), title="System status", url_path="status"),
    st.Page(lambda: retraining.render(client), title="Uploads & retraining", url_path="retraining"),
]
st.navigation(pages).run()
