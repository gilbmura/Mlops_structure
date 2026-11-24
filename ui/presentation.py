"""Small, escaped presentation components; no model or API logic belongs here."""

from contextlib import contextmanager
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).with_name("assets")


@lru_cache(maxsize=1)
def _styles() -> str:
    return (_ASSETS / "motion.css").read_text(encoding="utf-8")


def apply_styles(reduce_motion: bool = False) -> None:
    css = _styles()
    if reduce_motion:
        css += "\n*, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }"
    st.html(f"<style>{css}</style>")


def page_header(title: str, section: str, description: str) -> None:
    st.html(f'<header class="mri-heading mri-enter"><p class="mri-eyebrow">MRI LAB / {escape(section)}</p>'
            f'<h1>{escape(title)}</h1><p class="mri-description">{escape(description)}</p></header>')


def card(title: str, detail: str, *, label: str = "", active: bool = False, animate: bool = True) -> None:
    """API/file text is always escaped; animation is decorative, never the only status signal."""
    classes = "mri-card" + (" mri-enter" if animate else "")
    indicator = '<span class="mri-activity" aria-hidden="true"></span>' if active else ""
    st.html(f'<section class="{classes}"><p class="mri-eyebrow">{indicator}{escape(label)}</p>'
            f'<h3>{escape(title)}</h3><p class="mri-description">{escape(detail)}</p></section>')


@contextmanager
def activity(label: str, complete: str):
    """Reflect the lifetime of a real operation without inventing a completion percentage."""
    status = st.status(label, expanded=False)
    try:
        yield
    except Exception:
        status.update(label="Operation did not complete — see the error below.", state="error")
        raise
    else:
        status.update(label=complete, state="complete")
