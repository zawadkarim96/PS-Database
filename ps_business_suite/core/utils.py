"""Utility helpers shared by PS Business Suite pages."""
from __future__ import annotations

import streamlit as st


def render_signature() -> None:
    """Show the required credit footer on every page."""
    st.markdown(
        "<div style='font-size:10px; text-align:right; opacity:0.6;'>by Zad</div>",
        unsafe_allow_html=True,
    )
